import os, json, time, random
from enum import Enum

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

import pandas as pd

from train import TrainOptions, evaluate, log_scalars, log_images, save_models

from adaptive_stereo.datasets.stereo_dataset import StereoDataset
from adaptive_stereo.utils.feature_contrast import feature_contrast_mean
from adaptive_stereo.utils.stereo_reservoir import StereoReservoir
from adaptive_stereo.utils.loss_functions import monodepth_leftright_loss, monodepth_loss, khamis_robust_loss
from adaptive_stereo.utils.ema import online_ema
from adaptive_stereo.models.stereo_net import StereoNet, FeatureExtractorNetwork
from adaptive_stereo.models.linear_warping import LinearWarping


# https://github.com/pytorch/pytorch/issues/15054
# https://discuss.pytorch.org/t/nondeterminism-even-when-setting-all-seeds-0-workers-and-cudnn-deterministic/26080
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)


class State(Enum):
  DONE = 0          # Adaptation is finished, no gradient descent updates.
  IN_PROGRESS = 1   # Adaptation is in progress.
  VALIDATION = 2    # Validating the model, so turn off


def predict_disparity_leftright(feature_net, stereo_net, left, right, adapt_state, opt):
  """
  Predicts left AND right disparity maps for a batch of images.
  NOTE: Set adapt_state to State.VALIDATION to avoid storing any gradients!
  """
  # If we're not doing updates for the feature network, disable grad.
  with torch.set_grad_enabled(adapt_state == State.IN_PROGRESS):
    left_batch = torch.cat([left, torch.flip(right, (-1,))], dim=0)
    right_batch = torch.cat([right, torch.flip(left, (-1,))], dim=0)

    fl, fr = feature_net(left_batch), feature_net(right_batch)
    outputs = stereo_net(left_batch, fl, fr, "x", output_cost_volume=True)

    outputs_lr = {}
    for key in outputs:
      lkey = key.replace("_x", "_l")
      rkey = key.replace("_x", "_r")
      outputs_lr[lkey] = outputs[key][0].unsqueeze(0)
      outputs_lr[rkey] = torch.flip(outputs[key][1], (-1,)).unsqueeze(0)

    del outputs

    return outputs_lr


def predict_disparity_left(feature_net, stereo_net, left, right, adapt_state, opt):
  """
  Predicts only the left disparity map for a batch of images.
  NOTE: Set adapt_state to State.VALIDATION to avoid storing any gradients!
  """
  # If we're not doing updates for the feature network, disable grad.
  with torch.set_grad_enabled(adapt_state == State.IN_PROGRESS):
    fl, fr = feature_net(left), feature_net(right)
    outputs = stereo_net(left, fl, fr, "l", output_cost_volume=True)

  return outputs


def monodepth_single_loss(left_img, right_img, outputs, warper, scale, opt):
  losses = {}
  left_warped, mask = warper(right_img, outputs["pred_disp_l/{}".format(scale)], right_to_left=True)
  losses["Monodepth/total_loss"] = monodepth_loss(
      outputs["pred_disp_l/{}".format(scale)],
      left_img, left_warped, smoothness_weight=1e-3)[0][mask].mean()

  outputs["left_warped/{}".format(scale)] = left_warped
  return losses


class StateMachine(object):
  def __init__(self, initial_state, ovs_buffer_size=8):
    self.initial_state = initial_state
    self.current_state = initial_state

    # Store an online validation set (OVS) of images from the novel domain.
    self.ovs = StereoReservoir(ovs_buffer_size)

    self.prev_ovs_loss = float('inf')
    self.ovs_did_change = True
    self.ovs_iters_without_improvement = 0

  def add_to_ovs(self, left_img, right_img, loss, batch_idx):
    """
    Add an image pair (maybe) to the online validation set (OVS).
    """
    did_add = self.ovs.add(left_img.detach(), right_img.detach(), loss.detach(), batch_idx)

    if did_add:
      print("[ OVS ] ADDED a new pair to the OVS (INDEX={} LOSS={})".format(batch_idx, loss))
      self.ovs_did_change = True

    # If an image was added to the OVS, it must have been novel. In we thought we were DONE with
    # adaptation, then we should restart.
    if self.current_state == State.DONE:
      self.restart()

    return did_add

  def restart(self):
    self.current_state = self.initial_state
    print("[ OVS ] RESTARTING adaptation!")

  def validate(self, feature_net, stereo_net, warper, opt):
    """
    Re-compute OVS loss using the current weights of feature_net and stereo_net.

    NOTE: Because we're just using loss as a relative performance metric and not for training, only
    use the single-image Monodepth loss here, since it requires less computation.
    """
    feature_net.eval()
    stereo_net.eval()

    with torch.no_grad():
      for i in range(self.ovs.size()):
        _, _, left, right = self.ovs.buf[i]
        # NOTE: Validation state ensures that no gradients are computed.
        outputs = predict_disparity_left(feature_net, stereo_net, left, right, State.VALIDATION, opt)
        losses = monodepth_single_loss(left, right, outputs, warper, opt.stereonet_input_scale, opt)
        self.ovs.update_value(i, losses["Monodepth/total_loss"].item())

    # Make sure networks are back in train mode.
    feature_net.train()
    stereo_net.train()

  def transition(self, opt):
    ovs_loss = self.ovs.average_value()
    print("\n[ OVS ] -----------------------------------------")
    print("[ OVS ] VALIDATION LOSS | PREVIOUS={} | UPDATED={}".format(self.prev_ovs_loss, ovs_loss))
    print("[ OVS ] -----------------------------------------\n")

    if ovs_loss >= self.prev_ovs_loss and self.ovs_did_change == False:
      self.ovs_iters_without_improvement += 1

      if self.ovs_iters_without_improvement >= opt.val_improve_retries:
        print("[ OVS ] Transitioned to DONE! Loss didn't improve in the last {} evaluations".format(
            self.ovs_iters_without_improvement))
        self.current_state = State.DONE
        self.prev_ovs_loss = float('inf')

    # Otherwise, if loss improved, keep adapting.
    else:
      self.ovs_did_change = False # Indicate that we've validated the current buffer.
      self.ovs_iters_without_improvement = 0
      self.prev_ovs_loss = ovs_loss
      print("[ OVS ] Transitioned to IN_PROGRESS. Loss improved or buffer changed.")

    return self.current_state

  def state(self):
    return self.current_state

  def ovs_buffer_size(self):
    return self.ovs.size()


def append_to_df(df, metrics_adapt, metrics_train, gradient_updates_dict, trial, step):
  d = {"trial": trial, "step": step}
  for key in metrics_adapt:
    d[key + "_ADAPT"] = [metrics_adapt[key]]
  for key in metrics_train:
    d[key + "_TRAIN"] = [metrics_train[key]]
  for key in gradient_updates_dict:
    d[key] = [gradient_updates_dict[key]]
  df = df.append(pd.DataFrame(d))
  return df


def adapt(opt):
  log_path = os.path.join(opt.log_dir, opt.model_name)
  os.makedirs(log_path, exist_ok=True)

  with open(os.path.join(log_path, "opt.json"), "w") as f:
    opt_readable = json.dumps(opt.__dict__, sort_keys=True, indent=4)
    print("--------------------------------------------------------------------")
    print("TRAINING OPTIONS:")
    print(opt_readable)
    f.write(opt_readable + "\n")
    print("--------------------------------------------------------------------")

  feature_net = FeatureExtractorNetwork(opt.stereonet_k).cuda()

  # NOTE: maxdisp is defined for the full-resolution image (not the coarse cost volume scale).
  stereo_net = StereoNet(opt.stereonet_k, 1, opt.stereonet_input_scale, maxdisp=192).cuda()
  feature_net.load_state_dict(
      torch.load(os.path.join(opt.load_weights_folder, "feature_net.pth")), strict=True)
  stereo_net.load_state_dict(
      torch.load(os.path.join(opt.load_weights_folder, "stereo_net.pth")), strict=True)

  optimizer = optim.Adam([{"params": stereo_net.parameters()},
                          {"params": feature_net.parameters()}],
                          lr=opt.learning_rate)

  image_scales = [opt.stereonet_input_scale, opt.stereonet_input_scale + opt.stereonet_k]

  adapt_dataset = StereoDataset(opt.dataset_path, opt.dataset_name, opt.split, opt.height, opt.width, "train",
                                scales=image_scales, do_hflip=False, random_crop=False, load_disp_left=True,
                                load_disp_right=True)
  adapt_val_dataset = StereoDataset(opt.dataset_path, opt.dataset_name, opt.split, opt.height, opt.width, "val",
                                    scales=image_scales, do_hflip=False, random_crop=False, load_disp_left=True,
                                    load_disp_right=False)
  train_val_dataset = StereoDataset(opt.train_dataset_path, opt.train_dataset_name, opt.train_split, opt.height, opt.width, "val",
                                    scales=image_scales, do_hflip=False, random_crop=False, load_disp_left=True,
                                    load_disp_right=False)

  adapt_loader = DataLoader(
      adapt_dataset, opt.batch_size, False, num_workers=opt.num_workers, pin_memory=True,
      drop_last=False, collate_fn=None)
  adapt_val_loader = DataLoader(
      adapt_val_dataset, 6, False, num_workers=4, pin_memory=True, drop_last=False, collate_fn=None)
  train_val_loader = DataLoader(
      train_val_dataset, 6, False, num_workers=4, pin_memory=True, drop_last=False, collate_fn=None)

  print("----------------------------------------------------------------------")
  print("DATASET SIZES:\n  TRAIN={} VAL={}".format(len(adapt_dataset), len(adapt_val_dataset)))
  print("----------------------------------------------------------------------")

  # Create tensorboard writers for visualizing in the browser.
  adapt_writer = SummaryWriter(os.path.join(log_path, "adapt"))
  train_writer = SummaryWriter(os.path.join(log_path, "train"))

  warpers = {}
  for s in image_scales:
    warpers[s] = LinearWarping(opt.height // 2**s, opt.width // 2**s, torch.device("cuda"))

  initial_state = State.DONE if (opt.adapt_mode == "NONE") else State.IN_PROGRESS
  state_machine = StateMachine(initial_state, ovs_buffer_size=opt.ovs_buffer_size)

  # Make a pd.DataFrame to store end-of-epoch results.
  path_to_trials_log = os.path.join(log_path, "trials.csv")
  if os.path.exists(path_to_trials_log):
    df = pd.read_csv(path_to_trials_log)
    trial_index = df["trial"].max() + 1
    print("\nNOTE: Found existing trials.csv, running trial #{}".format(trial_index))
  else:
    df = pd.DataFrame()
    trial_index = 0
    print("\nNOTE: No existing trials.csv, starting from trial #0")

  if not opt.skip_initial_eval:
    print("========================= PRE-ADAPTATION EVALUATION ============================")
    print("Doing evaluation for ADAPTATION set")
    metrics_adapt = evaluate(feature_net, stereo_net, adapt_val_loader, opt)
    log_scalars(adapt_writer, metrics_adapt, {}, 0, 0, 0)
    print("Done")

    print("Doing evaluation for TRAINING set")
    metrics_train = evaluate(feature_net, stereo_net, train_val_loader, opt)
    log_scalars(train_writer, metrics_train, {}, 0, 0, 0)
    print("Done")

    df = append_to_df(df, metrics_adapt, metrics_train, {}, trial_index, -1)
    df.to_csv(path_to_trials_log, index=False)
  else:
    print("----------------- WARNING: Skipped pre-adaptation evaluation -------------------")

  #================================== TRAINING LOOP ======================================
  epoch, step, gradient_updates = 0, 0, 0
  fcs_raw_list = []
  fcs_smoothed_list = []

  for epoch in range(opt.num_epochs):
    feature_net.train()
    stereo_net.train()

    # Exit adaptation if finished.
    if opt.num_steps > 0 and step >= opt.num_steps:
      break

    t0_epoch = time.time()

    for batch_idx, inputs in enumerate(adapt_loader):
      # Periodically validate the buffer of images.
      do_validation = (step % opt.ovs_validate_hz == 0)
      ovs_not_empty = state_machine.ovs_buffer_size() > 0
      adapt_in_progress = state_machine.state() == State.IN_PROGRESS

      if do_validation and ovs_not_empty and adapt_in_progress:
        state_machine.validate(feature_net, stereo_net, warpers[opt.stereonet_input_scale], opt)

        # If using the NONSTOP or ER methods (no validation), adaptation never stops.
        # Also, if not using adaptation, then don't worry about transitions.
        if opt.adapt_mode not in ("NONSTOP", "ER", "NONE"):
          state_machine.transition(opt)

      t0 = time.time()

      for key in inputs:
        inputs[key] = inputs[key].cuda().detach()

      if state_machine.state() == State.DONE:
        feature_net.eval()
        stereo_net.eval()
      else:
        feature_net.train()
        stereo_net.train()

      if opt.leftright_consistency:
        outputs = predict_disparity_leftright(
            feature_net, stereo_net,
            inputs["color_l/{}".format(opt.stereonet_intput_scale)],
            inputs["color_r/{}".format(opt.stereonet_input_scale)],
            state_machine.state(), opt)
        # with torch.set_grad_enabled(state_machine.state == State.IN_PROGRESS):
        losses = monodepth_leftright_loss(
            inputs["color_l/{}".format(opt.stereonet_input_scale)],
            inputs["color_r/{}".format(opt.stereonet_input_scale)],
            outputs, warpers[opt.stereonet_input_scale], opt.stereonet_input_scale)
      else:
        outputs = predict_disparity_left(
            feature_net, stereo_net,
            inputs["color_l/{}".format(opt.stereonet_input_scale)],
            inputs["color_r/{}".format(opt.stereonet_input_scale)],
            state_machine.state(), opt)
        # with torch.set_grad_enabled(state_machine.state == State.IN_PROGRESS):
        losses = monodepth_single_loss(
          inputs["color_l/{}".format(opt.stereonet_input_scale)],
            inputs["color_r/{}".format(opt.stereonet_input_scale)],
            outputs, warpers[opt.stereonet_input_scale], opt.stereonet_input_scale, opt)

      if opt.adapt_mode in ("ER", "VS+ER"):
        # Choose a "random" training image.
        inputs_er = train_val_dataset[step % len(train_val_dataset)]
        outputs_er = predict_disparity_left(
            feature_net, stereo_net,
            inputs_er["color_l/{}".format(opt.stereonet_input_scale)].cuda().unsqueeze(0),
            inputs_er["color_r/{}".format(opt.stereonet_input_scale)].cuda().unsqueeze(0),
            state_machine.state(), opt)
        losses["Replay/total_loss"] = khamis_robust_loss(
            outputs_er["pred_disp_l/{}".format(opt.stereonet_input_scale)],
            inputs_er["gt_disp_l/{}".format(opt.stereonet_input_scale)].cuda().unsqueeze(0))

      # Compute feature contrast score (FCS).
      fcs_raw = feature_contrast_mean(
          outputs["cost_volume_l/{}".format(opt.stereonet_input_scale + opt.stereonet_k)]).mean()

      # Smooth the FCS with an exponential moving average.
      if len(fcs_smoothed_list) > 0:
        fcs_smoothed = online_ema(fcs_smoothed_list[-1], fcs_raw, weight=opt.fcs_ema_weight)
      else:
        fcs_smoothed = fcs_raw

      fcs_raw_list.append(fcs_raw)
      fcs_smoothed_list.append(fcs_smoothed)
      adapt_writer.add_scalar("fcs/raw", fcs_raw.item(), step)
      adapt_writer.add_scalar("fcs/smoothed", fcs_smoothed.item(), step)

      # OOD Detection!
      image_is_novel = (fcs_smoothed.item() < opt.ood_threshold)

      # Add the current stereo pair to the OVS.
      did_add_to_ovs = False
      if opt.adapt_mode not in ("NONSTOP", "ER", "NONE"):
        if image_is_novel:
          print("[ OOD ] Novel image detected! fcs_raw={:.03f} fcs_smoothed={:.03f} threshold={:.03f}".format(
              fcs_raw, fcs_smoothed, opt.ood_threshold))
          did_add_to_ovs = state_machine.add_to_ovs(
              inputs["color_l/{}".format(opt.stereonet_input_scale)],
              inputs["color_r/{}".format(opt.stereonet_input_scale)],
              losses["Monodepth/total_loss"], batch_idx)

      # If still adapting, do backprop.
      if state_machine.state() == State.IN_PROGRESS:
        optimizer.zero_grad()

        # Only adapt to this image if if wasn't added to the validation set.
        if not did_add_to_ovs:
          backprop_loss = losses["Monodepth/total_loss"]
          if "Replay/total_loss" in losses:
            backprop_loss += opt.er_loss_weight*losses["Replay/total_loss"]

          backprop_loss.backward()
          if opt.clip_grad_norm:
            nn.utils.clip_grad_norm_(stereo_net.parameters(), 1.0)
          optimizer.step()
          gradient_updates += 1
        else:
          print("[ ADAPT ] Skipping gradient update because image was added to OVS.")

      elapsed_this_batch = time.time() - t0
      do_logging = (step % opt.log_frequency) == 0 and step > 0

      if do_logging:
        # If groundtruth disparity available, compute the EPE for each image.
        if "gt_disp_l/{}".format(opt.stereonet_input_scale) in inputs:
          metrics = {}
          gt_disp = inputs["gt_disp_l/{}".format(opt.stereonet_input_scale)]
          pred_disp = outputs["pred_disp_l/{}".format(opt.stereonet_input_scale)]
          metrics["EPE"] = torch.abs(gt_disp - pred_disp)[gt_disp > 0].mean()

        log_scalars(adapt_writer, metrics, losses, opt.batch_size / elapsed_this_batch, epoch, step)
        log_images(adapt_writer, inputs, outputs, step)

      step += 1

      mid_epoch_eval = opt.eval_hz > 0 and step % opt.eval_hz == 0
      end_epoch_eval = (opt.eval_hz <= 0) and batch_idx == (len(adapt_loader) - 1)

      if mid_epoch_eval or end_epoch_eval:
        print("=============== MID-ADAPTATION EVALUATION (step {}) ==================".format(step))
        adapt_writer.add_scalar("GRADIENT_UPDATES", gradient_updates, step)

        print("Evaluating on ADAPTATION set")
        metrics_adapt = evaluate(feature_net, stereo_net, adapt_val_loader, opt)
        log_scalars(adapt_writer, metrics_adapt, {}, 0, epoch, step)
        print("DONE")
        print("Evaluating on TRAINING set")
        metrics_train = evaluate(feature_net, stereo_net, train_val_loader, opt)
        log_scalars(train_writer, metrics_train, {}, 0, epoch, step)
        print("DONE")

        save_models(feature_net, stereo_net, optimizer, log_path, step)

        df = append_to_df(df, metrics_adapt, metrics_train, {"GRADIENT_UPDATES": gradient_updates},
                          trial_index, step)
        df.to_csv(path_to_trials_log, index=False)
        print("Wrote data to {} (step={})".format(path_to_trials_log, step))

      # Exit adaptation if finished.
      if opt.num_steps > 0 and step >= opt.num_steps:
        break

    elapsed_epoch = time.time() - t0_epoch
    print("Finished {} adaptation steps in {:.02f}s ({:.02f} examples/s)".format(
        len(adapt_loader), elapsed_epoch, len(adapt_loader) / elapsed_epoch))


if __name__ == "__main__":
  options = TrainOptions()
  print("\nStarting adaptation ...")
  adapt(options.parse())
  print("Done with adaptation!")

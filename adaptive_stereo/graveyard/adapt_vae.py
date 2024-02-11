import os, json, time, random
from enum import Enum

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

import pandas as pd

from train_options import TrainOptions
from train import evaluate, compute_loss
from train_autoencoder import evaluate_vae, AETrainOptions

from datasets.stereo_dataset import StereoDataset
from utils.stereo_reservoir import StereoReservoir
from utils.loss_functions import monodepth_leftright_loss, monodepth_loss, khamis_robust_loss
from utils.visualization import *
from models.stereo_net import StereoNet, FeatureExtractorNetwork
from models.vae import VAE, vae_loss_function
from models.linear_warping import LinearWarping


# https://github.com/pytorch/pytorch/issues/15054
# https://discuss.pytorch.org/t/nondeterminism-even-when-setting-all-seeds-0-workers-and-cudnn-deterministic/26080
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class State(Enum):
  DONE = 0          # Adaptation is finished, no gradient descent updates.
  IN_PROGRESS = 1   # Adaptation is in progress.
  VALIDATION = 2    # Validating the model, so turn off


def log_scalars(writer, metrics, losses, examples_per_sec, epoch, step):
  with torch.no_grad():
    for name in losses:
      writer.add_scalar(name, losses[name].detach().item(), step)

    for name in metrics:
      writer.add_scalar(name, metrics[name], step)

    writer.add_scalar("examples_per_sec", examples_per_sec, step)

    print("\n{}|{}========================================================================".format(epoch, step))
    print("TIMING  // examples/sec={:.3f}".format(examples_per_sec))
    if len(metrics) > 0:
      print("METRICS // EPE={:.3f} | >2px={:.3f} | >3px={:.3f} | >4px={:.3f} | >5px={:.3f}".format(
        metrics["EPE"], metrics["D1_all_2px"], metrics["D1_all_3px"], metrics["D1_all_4px"], metrics["D1_all_5px"]))

    if len(losses) > 0:
      loss_str = "LOSS    // "
      for name in losses:
        loss_str += " | {}={:.3f}".format(name, losses[name])
      print(loss_str)
    print("===========================================================================\n")

  # writer.close()


def contains_prefix(name, prefixes):
  for p in prefixes:
    if p in name:
      return True
  return False


def log_images(writer, inputs, outputs, step, skip_prefixes=["cost_volume", "mu", "logvar"]):
  with torch.no_grad():
    for io in (inputs, outputs):
      for name in io:
        if contains_prefix(name, skip_prefixes):
          continue
        if "disp" in name:
          viz = visualize_disp_tensorboard(io[name][0].detach().cpu())
        else:
          viz = io[name][0].detach().cpu()
        writer.add_image(name, viz, step)

  # NOTE: It seems like the last few images don't get logged properly without this...
  # writer.close()


def save_models(feature_net, stereo_net, vae_net, optimizer, log_path, epoch):
  save_folder = os.path.join(log_path, "models", "weights_{}".format(epoch))
  os.makedirs(save_folder, exist_ok=True)

  torch.save(feature_net.state_dict(), os.path.join(save_folder, "feature_net.pth"))
  torch.save(stereo_net.state_dict(), os.path.join(save_folder, "stereo_net.pth"))
  torch.save(vae_net.state_dict(), os.path.join(save_folder, "vae_net.pth"))


def process_batch_stereo(feature_net, stereo_net, left, right, adapt_state, opt):
  """
  Predicts only the left disparity map for a batch of images.

  NOTE: Set adapt_state to State.VALIDATION to avoid storing any gradients!
  """
  store_feature_gradients = adapt_state == State.IN_PROGRESS
  store_refinement_gradients = adapt_state == State.IN_PROGRESS

  # If we're not doing updates for the feature network, disable grad.
  with torch.set_grad_enabled(store_feature_gradients):
    left_feat, right_feat = feature_net(left), feature_net(right)

  outputs = stereo_net(left, left_feat, right_feat, "l",
                      store_feature_gradients=store_feature_gradients,
                      store_refinement_gradients=store_refinement_gradients,
                      do_refinement=True,
                      output_cost_volume=True)

  return outputs


def monodepth_left_loss(left_img, right_img, outputs, warper, scale, opt):
  losses = {}
  left_warped, mask = warper(right_img, outputs["pred_disp_l/{}".format(scale)], right_to_left=True)
  losses["stereo/total_loss"] = monodepth_loss(outputs["pred_disp_l/{}".format(scale)],
                                        left_img, left_warped,
                                        smoothness_weight=1e-3)[0][mask].mean()
  outputs["left_warped/0"] = left_warped

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
        outputs = process_batch_stereo(feature_net, stereo_net, left, right, State.VALIDATION, opt)
        losses = monodepth_left_loss(left, right, outputs, warper, 0, opt)
        self.ovs.update_value(i, losses["stereo/total_loss"].item())

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
  random.seed(123)
  torch.manual_seed(123)
  torch.cuda.manual_seed(123)
  torch.cuda.manual_seed_all(123)

  log_path = os.path.join(opt.log_dir, opt.name)
  os.makedirs(log_path, exist_ok=True)

  with open(os.path.join(log_path, "opt.json"), "w") as f:
    opt_readable = json.dumps(opt.__dict__, sort_keys=True, indent=4)
    print("--------------------------------------------------------------------")
    print("TRAINING OPTIONS:")
    print(opt_readable)
    f.write(opt_readable + "\n")
    print("--------------------------------------------------------------------")

  feature_net = FeatureExtractorNetwork(opt.stereonet_k).cuda()
  stereo_net = StereoNet(k=opt.stereonet_k, r=1).cuda()
  vae_net = VAE(image_channels=3, z_dim=opt.vae_bottleneck,
                input_height=opt.height // (2**opt.decoder_loss_scale),
                input_width=opt.width // (2**opt.decoder_loss_scale)).cuda()

  feature_net.load_state_dict(torch.load(os.path.join(opt.load_weights_folder, "feature_net.pth")), strict=True)
  stereo_net.load_state_dict(torch.load(os.path.join(opt.load_weights_folder, "stereo_net.pth")), strict=True)
  vae_net.load_state_dict(torch.load(os.path.join(opt.vae_weights_folder, "vae_net.pth")), strict=True)

  stereo_optimizer = optim.Adam([{"params": stereo_net.parameters()}, {"params": feature_net.parameters()}], lr=opt.learning_rate)
  vae_optimizer = optim.Adam(vae_net.parameters(), lr=opt.vae_learning_rate)

  image_scales = [0]
  if opt.decoder_loss_scale != 0:
    image_scales.append(opt.decoder_loss_scale)

  adapt_dataset = StereoDataset(opt.dataset_path, opt.dataset_name, opt.split, opt.height, opt.width, "train",
                                scales=image_scales, do_hflip=False, random_crop=False, load_disp_left=True,
                                load_disp_right=True)
  adapt_val_dataset = StereoDataset(opt.dataset_path, opt.dataset_name, opt.split, opt.height, opt.width, "val",
                                    scales=image_scales, do_hflip=False, random_crop=False, load_disp_left=True,
                                    load_disp_right=False)
  train_val_dataset = StereoDataset(opt.train_dataset_path, opt.train_dataset_name, opt.train_split, opt.height, opt.width, "val",
                                    scales=image_scales, do_hflip=False, random_crop=False, load_disp_left=True,
                                    load_disp_right=False)

  adapt_loader = DataLoader(adapt_dataset, opt.batch_size, False, num_workers=opt.num_workers, pin_memory=True, drop_last=False, collate_fn=None)
  adapt_val_loader = DataLoader(adapt_val_dataset, 6, False, num_workers=4, pin_memory=True, drop_last=False, collate_fn=None)
  train_val_loader = DataLoader(train_val_dataset, 6, False, num_workers=4, pin_memory=True, drop_last=False, collate_fn=None)

  print("----------------------------------------------------------------------")
  print("DATASET SIZES:\n  TRAIN={} VAL={}".format(len(adapt_dataset), len(adapt_val_dataset)))
  print("----------------------------------------------------------------------")

  # Create tensorboard writers for visualizing in the browser.
  adapt_writer = SummaryWriter(os.path.join(log_path, "adapt"))
  train_writer = SummaryWriter(os.path.join(log_path, "train"))

  warpers = {}
  warpers[0] = LinearWarping(opt.height, opt.width, torch.device("cuda"))

  state_machine = StateMachine(State.IN_PROGRESS, ovs_buffer_size=opt.ovs_buffer_size)

  # Make a pd.DataFrame to store end-of-epoch results.
  path_to_runs_log = os.path.join(log_path, "trials.csv")
  if os.path.exists(path_to_runs_log):
    df = pd.read_csv(path_to_runs_log)
    trial_index = df["trial"].max() + 1
    print("\nNOTE: Found existing trials.csv, running trial #{}".format(trial_index))
  else:
    df = pd.DataFrame()
    trial_index = 0
    print("\nNOTE: No existing trials.csv, starting from 0")

  if not opt.skip_initial_eval:
    print("========================= PRE-ADAPTATION EVALUATION ============================")
    print("Doing evaluation for ADAPTATION set")
    metrics_adapt = evaluate(feature_net, stereo_net, adapt_val_loader, opt)
    metrics_adapt_vae = evaluate_vae(vae_net, adapt_val_loader, opt)
    log_scalars(adapt_writer, metrics_adapt, metrics_adapt_vae, 0, 0, 0)
    print("Done")

    print("Doing evaluation for TRAINING set")
    metrics_train = evaluate(feature_net, stereo_net, train_val_loader, opt)
    metrics_train_vae = evaluate_vae(vae_net, train_val_loader, opt)
    log_scalars(train_writer, metrics_train, metrics_train_vae, 0, 0, 0)
    print("Done")

    df = append_to_df(df, metrics_adapt, metrics_train, {}, trial_index, -1)
    df.to_csv(path_to_runs_log, index=False)
  else:
    print("----------------- WARNING: Skipped pre-adaptation evaluation -------------------")

  #================================== TRAINING LOOP ======================================
  epoch, step, gradient_updates = 0, 0, 0

  for epoch in range(opt.num_epochs):
    feature_net.train()
    stereo_net.train()
    vae_net.train()

    # Exit adaptation if finished.
    if opt.num_steps > 0 and step >= opt.num_steps:
      break

    t0_epoch = time.time()

    for batch_idx, inputs in enumerate(adapt_loader):
      # Periodically validate the buffer of images.
      time_to_validate = (step % opt.ovs_validate_hz == 0)
      ovs_not_empty = state_machine.ovs_buffer_size() > 0
      adapt_in_progress = state_machine.state() == State.IN_PROGRESS

      if time_to_validate and ovs_not_empty and adapt_in_progress:
        state_machine.validate(feature_net, stereo_net, warpers[0], opt)

        # If using the NONSTOP or ER methods (no validation), adaptation never stops.
        if opt.adapt_mode not in ("NONSTOP", "ER"):
          state_machine.transition(opt)

      # Train on a single image.
      t0 = time.time()

      for key in inputs:
        inputs[key] = inputs[key].cuda().detach()

      loss_scale = 0

      if state_machine.state() == State.DONE:
        feature_net.eval()
        stereo_net.eval()
      else:
        feature_net.train()
        stereo_net.train()

      outputs = process_batch_stereo(
          feature_net, stereo_net,
          inputs["color_l/0"], inputs["color_r/0"],
          State.DONE, opt)
      losses = monodepth_left_loss(
        inputs["color_l/0"], inputs["color_r/0"],
        outputs, warpers[0], loss_scale, opt)

      # outputs = process_batch_stereo(
      #     feature_net, stereo_net,
      #     inputs["color_l/0"], inputs["color_r/0"],
      #     state_machine.state(), opt)
      # losses = monodepth_left_loss(
      #   inputs["color_l/0"], inputs["color_r/0"],
      #   outputs, warpers[0], loss_scale, opt)

      # if opt.adapt_mode in ("ER", "VS+ER"):
      #   # Choose a "random" training image.
      #   inputs_erb = train_val_dataset[step % len(train_val_dataset)]
      #   outputs_erb = process_batch_stereo(
      #       feature_net, stereo_net,
      #       inputs_erb["color_l/0"].cuda().unsqueeze(0),
      #       inputs_erb["color_r/0"].cuda().unsqueeze(0),
      #       state_machine.state(), opt)
      #   losses["stereo/ER_loss"] = khamis_robust_loss(
      #       outputs_erb["pred_disp_l/0"],
      #       inputs_erb["gt_disp_l/0"].cuda().unsqueeze(0))

      # If still adapting, do backprop.
      if state_machine.state() == State.IN_PROGRESS:
        stereo_optimizer.zero_grad()

        # backprop_loss = losses["stereo/total_loss"]

        # if "stereo/ER_loss" in losses:
        #   backprop_loss += opt.er_loss_weight*losses["stereo/ER_loss"]

        # backprop_loss.backward()

        # if opt.clip_grad_norm:
        #   nn.utils.clip_grad_norm_(stereo_net.parameters(), 1.0)

        # stereo_optimizer.step()
        gradient_updates += 1

      # Add the current stereo pair to the OVS.
      if opt.adapt_mode not in ("NONSTOP", "ER"):
        # Compute VAE reconstruction loss.
        outputs["decoded_l/{}".format(opt.decoder_loss_scale)], mu, logvar = \
            vae_net(inputs["color_l/{}".format(opt.decoder_loss_scale)])
        losses["vae/reconstruction_loss"] = torch.abs(
            outputs["decoded_l/{}".format(opt.decoder_loss_scale)] - \
            inputs["color_l/{}".format(opt.decoder_loss_scale)]).mean()

        image_is_novel = (losses["vae/reconstruction_loss"].item() > opt.ood_threshold)
        if image_is_novel:
          print("[ OOD ] Novel image detected! (image={} threshold={})".format(losses["vae/reconstruction_loss"].item(), opt.ood_threshold))
          state_machine.add_to_ovs(inputs["color_l/0"], inputs["color_r/0"], losses["stereo/total_loss"], batch_idx)

          vae_optimizer.zero_grad()

          # Adapt the VAE using reconstruction loss + KL regularizatin.
          color_l_er = train_val_dataset[step % len(train_val_dataset)]["color_l/{}".format(opt.decoder_loss_scale)].cuda().unsqueeze(0)
          losses["vae/total_loss"], losses["vae/reconstruction_loss"], losses["vae/KL_divergence_loss"] = \
              vae_loss_function(inputs["color_l/{}".format(opt.decoder_loss_scale)],
                                outputs["decoded_l/{}".format(opt.decoder_loss_scale)], mu, logvar)

          outputs["decoded_er/{}".format(opt.decoder_loss_scale)], mu_er, logvar_er = vae_net(color_l_er)
          losses["vae/er_loss"] = vae_loss_function(color_l_er, outputs["decoded_er/{}".format(opt.decoder_loss_scale)], mu_er, logvar_er)[0]
          (losses["vae/total_loss"] + losses["vae/er_loss"]).backward()
          vae_optimizer.step()

      elapsed_this_batch = time.time() - t0
      do_logging = (step % opt.log_frequency) == 0 and step > 0

      if do_logging:
        print("Logging")
        log_scalars(adapt_writer, {}, losses, opt.batch_size / elapsed_this_batch, epoch, step)
        log_images(adapt_writer, inputs, outputs, step)

      step += 1

      mid_epoch_eval = opt.eval_hz > 0 and step % opt.eval_hz == 0
      end_epoch_eval = (opt.eval_hz <= 0) and batch_idx == (len(adapt_loader) - 1)

      if mid_epoch_eval or end_epoch_eval:
        print("=============== MID-ADAPTATION EVALUATION (step {}) ==================".format(step))
        adapt_writer.add_scalar("GRADIENT_UPDATES", gradient_updates, step)

        print("Evaluating on ADAPTATION set")
        metrics_adapt = evaluate(feature_net, stereo_net, adapt_val_loader, opt)
        metrics_adapt_vae = evaluate_vae(vae_net, adapt_val_loader, opt)
        log_scalars(adapt_writer, metrics_adapt, metrics_adapt_vae, 0, epoch, step)
        print("DONE")
        print("Evaluating on TRAINING set")
        metrics_train = evaluate(feature_net, stereo_net, train_val_loader, opt)
        metrics_train_vae = evaluate_vae(vae_net, train_val_loader, opt)
        log_scalars(train_writer, metrics_train, metrics_train_vae, 0, epoch, step)
        print("DONE")

        save_models(feature_net, stereo_net, vae_net, stereo_optimizer, log_path, step)

        df = append_to_df(df, metrics_adapt, metrics_train, {"GRADIENT_UPDATES": gradient_updates},
                          trial_index, step)
        df.to_csv(path_to_runs_log, index=False)
        print("Wrote data to {} (step={})".format(path_to_runs_log, step))

      # Exit adaptation if finished.
      if opt.num_steps > 0 and step >= opt.num_steps:
        break

    elapsed_epoch = time.time() - t0_epoch
    print("Finished {} adaptation steps in {:.02f}s ({:.02f} examples/s)".format(
        len(adapt_loader), elapsed_epoch, len(adapt_loader) / elapsed_epoch))


if __name__ == "__main__":
  options = AETrainOptions()
  print("\nStarting adaptation ...")
  adapt(options.parse())
  print("Done with adaptation!")

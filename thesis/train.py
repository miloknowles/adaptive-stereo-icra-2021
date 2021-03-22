# Copyright 2020 Massachusetts Institute of Technology
#
# @file train.py
# @author Milo Knowles
# @date 2020-07-08 19:08:01 (Wed)

import os, json, time

import git
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from train_options import TrainOptions
from models.linear_warping import LinearWarping
from models.madnet import MadNet
from models.mgc_net import MGCNet
from datasets.stereo_dataset import StereoDataset
from utils.utils import *
from utils.dataset_utils import *
from utils.loss_functions import *
from utils.visualization import *


def get_R2_image_or_downsample_R0(inputs, side):
  """
  Convenience function to either return the RGB image at resolution 2, or downsample it
  from resolution 0 if needed.

  Args:
    inputs (dict) : Should contain color_{}/0 or color_{}/2, where {} is a placeholder for "side".
    side (str) : Should be "l" or "r", specifies which image of the stereo pair to return.

  Returns:
    left or right image at 1/4 full resolution (R2).
  """
  if "color_{}/2".format(side) in inputs:
    return inputs["color_{}/2".format(side)]
  else:
    r2_width = inputs["color_{}/0".format(side)].shape[3] // 4
    r2_height = inputs["color_{}/0".format(side)].shape[2] // 4
    return F.interpolate(inputs["color_l/0"], size=(r2_height, r2_width), mode="bilinear", align_corners=False)


def process_batch(feature_network, depth_network, inputs, output_scales=[0], predict_variance=False,
                  variance_mode="disparity", leftright_consistency=False, device=torch.device("cuda")):
  """
  Pass a batch of inputs through the network and return outputs.
  """
  assert(variance_mode == "disparity" or variance_mode == "photometric")

  for key in inputs:
    inputs[key] = inputs[key].to(device).detach()

  if isinstance(depth_network, MadNet):
    outputs = depth_network(inputs)

  elif isinstance(depth_network, MGCNet):
    left_feat = feature_network(inputs["color_l/0"])
    right_feat = feature_network(inputs["color_r/0"])

    # Get outputs for the left ==> right stereo pair.
    variance_input = None
    if predict_variance and variance_mode == "photometric":
      variance_input = get_R2_image_or_downsample_R0(inputs, "l")

    outputs = depth_network(left_feat, right_feat, side="l", output_scales=output_scales, variance_input=variance_input)

    # Optionally get outputs for the right ==> left stereo pair.
    # NOTE: The flipping is tricky here. We switch the order of the image pair, but also apply a horizontal flip.
    # After getting outputs from the network, horizontally flip everything again so that it's aligned with the
    # original left and right images.
    if leftright_consistency:
      left_feat = feature_network(torch.flip(inputs["color_r/0"], (3,)))
      right_feat = feature_network(torch.flip(inputs["color_l/0"], (3,)))

      variance_input = None
      if predict_variance and variance_mode == "photometric":
        variance_input = get_R2_image_or_downsample_R0(inputs, "r")

      outputs_flipped = depth_network(left_feat, right_feat, side="r", output_scales=output_scales, variance_input=variance_input)

      for key in outputs_flipped:
        outputs_flipped[key] = torch.flip(outputs_flipped[key], (3,))

      outputs.update(outputs_flipped)

  else:
    raise NotImplementedError()

  return outputs


def compute_loss(inputs, outputs, opt, warpers):
  """
  Computes the specified loss function and returns it. Backprop is computed w.r.t
  losses["total_loss"] - all of the other entries are for debugging.
  """
  if opt.loss_type == "madnet":
    return madnet_loss(inputs, outputs)

  elif opt.loss_type == "supervised_likelihood":
    return supervised_likelihood_loss(inputs, outputs, scale=2, distribution=opt.distribution)

  elif opt.loss_type == "monodepth_l1_likelihood":
    return monodepth_l1_likelihood_loss_leftright(inputs, outputs, warpers, opt)

  elif opt.loss_type == "monodepth":
    return monodepth_leftright_loss(inputs["color_l/0"], inputs["color_r/0"], outputs, warpers[0], 0)[0]

  else:
    raise NotImplementedError()


def log_scalars(writer, metrics, losses, examples_per_sec, epoch, step):
  with torch.no_grad():
    for name in losses:
      writer.add_scalar(name, losses[name], step)

    for name in metrics:
      writer.add_scalar(name, metrics[name], step)

    writer.add_scalar("examples_per_sec", examples_per_sec, step)
    writer.close()

    print("\n{}|{}========================================================================".format(epoch, step))
    print("TIMING // examples/sec={:.3f}".format(examples_per_sec))
    if len(metrics) > 0:
      print("METRICS // EPE={:.3f} | >2px={:.3f} | >3px={:.3f} | >4px={:.3f} | >5px={:.3f}".format(
        metrics["EPE"], metrics["D1_all_2px"], metrics["D1_all_3px"], metrics["D1_all_4px"], metrics["D1_all_5px"]))

    if len(losses) > 0:
      loss_str = "LOSS // "
      for name in losses:
        loss_str += " | {}={:.3f}".format(name, losses[name])
      print(loss_str)
  # writer.close()


def contains_prefix(name, prefixes):
  for p in prefixes:
    if p in name:
      return True
  return False


def log_images(writer, inputs, outputs, step, skip_prefixes=["cost_volume"]):
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
  writer.close()


def evaluate(feature_net, stereo_net, val_loader, opt):
  feature_net.eval()
  stereo_net.eval()

  num_batches_eval = len(val_loader) // 10 if opt.fast_eval else len(val_loader)
  variance_mode = "disparity" if opt.loss_type in ["supervised_likelihood"] else "photometric"

  with torch.no_grad():
    EPEs = torch.zeros(num_batches_eval)
    D1_alls = torch.zeros(num_batches_eval, 4)
    FCSs = torch.zeros(num_batches_eval)

    for i, inputs in enumerate(val_loader):
      if i >= num_batches_eval:
        break

      for key in inputs:
        inputs[key] = inputs[key].cuda()

      outputs = process_batch(feature_net, stereo_net, inputs, output_scales=opt.scales,
                              predict_variance=True, variance_mode=variance_mode,
                              leftright_consistency=opt.leftright_consistency)

      pred_disp = outputs["pred_disp_l/0"]
      gt_disp = inputs["gt_disp_l/0"]
      valid_mask = (gt_disp > 0)

      # Compute EPE.
      EPEs[i] = torch.abs(pred_disp - gt_disp)[valid_mask].mean()

      # Compute D1-all for several outlier thresholds.
      for oi, ot in enumerate([2, 3, 4, 5]):
        D1_alls[i, oi] = (valid_mask * (torch.abs(pred_disp - gt_disp) > ot)).sum() / float(valid_mask.sum())

    EPEs = EPEs.mean()
    D1_alls = D1_alls.mean(dim=0)
    FCSs = FCSs.mean()

    metrics = {"EPE": EPEs.item(),
              "FCS": FCSs.item(),
              "D1_all_2px": D1_alls[0].item(),
              "D1_all_3px": D1_alls[1].item(),
              "D1_all_4px": D1_alls[2].item(),
              "D1_all_5px": D1_alls[3].item()}

  feature_net.train()
  stereo_net.train()

  return metrics


def save_models(feature_net, stereo_net, optimizer, log_path, epoch):
  save_folder = os.path.join(log_path, "models", "weights_{}".format(epoch))
  os.makedirs(save_folder, exist_ok=True)

  torch.save(feature_net.state_dict(), os.path.join(save_folder, "feature_net.pth"))
  torch.save(stereo_net.state_dict(), os.path.join(save_folder, "stereo_net.pth"))

  if optimizer is not None:
    torch.save(optimizer.state_dict(), os.path.join(save_folder, "adam.pth"))


def train(opt):
  torch.manual_seed(123)

  # https://github.com/pytorch/pytorch/issues/15054
  torch.backends.cudnn.benchmark = True

  log_path = os.path.join(opt.log_dir, opt.model_name)
  os.makedirs(log_path, exist_ok=True)

  # https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
  repo = git.Repo(search_parent_directories=True)
  sha = repo.head.object.hexsha
  opt.commit_hash = sha

  with open(os.path.join(log_path, "opt.json"), "w") as f:
    opt_readable = json.dumps(opt.__dict__, sort_keys=True, indent=4)
    print("==========" * 10)
    print("TRAINING OPTIONS:")
    print(opt_readable)
    f.write(opt_readable + "\n")

  variance_mode = "disparity" if opt.loss_type in ["supervised_likelihood"] else "photometric"
  print("VARIANCE MODE:  ", variance_mode)

  if opt.network == "MGC-Net":
    stereo_net = MGCNet(opt.radius_disp, img_height=opt.height, img_width=opt.width,
                        device=torch.device("cuda"), predict_variance=opt.predict_variance,
                        image_channels=1 if opt.use_grayscale else 3,
                        gradient_bulkhead="VarianceModule" if opt.freeze_disp else None,
                        variance_mode=variance_mode).cuda()
    feature_net = stereo_net.feature_extractor.cuda()
  elif opt.network == "MAD-Net":
    stereo_net = MadNet(opt.radius_disp, img_height=opt.height, img_width=opt.width, device=torch.device("cuda"))
    feature_net = stereo_net.feature_extractor
  else:
    raise NotImplementedError()

  # NOTE: This only works with MGC-Net.
  if opt.predict_variance:
    if opt.freeze_disp:
      print("NOTE: Freezing DEPTH network weights")
      parameters_to_train = [{"params": stereo_net.variance_module.parameters(), "lr": opt.variance_module_lr}]
    elif opt.freeze_variance:
      print("NOTE: Freezing VARIANCE network weights")
      parameters_to_train = [{"params": stereo_net.feature_extractor.parameters()},
                            {"params": stereo_net.refinement_modules.parameters()},
                            {"params": stereo_net.cost_volume_filters.parameters()}]
    else:
      parameters_to_train = [{"params": stereo_net.feature_extractor.parameters()},
                            {"params": stereo_net.refinement_modules.parameters()},
                            {"params": stereo_net.cost_volume_filters.parameters()},
                            {"params": stereo_net.variance_module.parameters(), "lr": opt.variance_module_lr}]
  else:
    parameters_to_train = [{"params": stereo_net.parameters()}]

  optimizer = optim.Adam(parameters_to_train, lr=opt.learning_rate)
  scheduler = optim.lr_scheduler.StepLR(optimizer, opt.scheduler_step_size, 0.5)

  # If a folder for pretrained weights is given, load from there.
  if opt.load_weights_folder is not None:
    print("Loading models from: ", opt.load_weights_folder)
    # feature_net.load_state_dict(torch.load(os.path.join(opt.load_weights_folder, "feature_net.pth")), strict=True)
    stereo_net.load_state_dict(torch.load(os.path.join(opt.load_weights_folder, "model_strict.pth")), strict=False)

  # Make sure that loss type is compatible with other options.
  if opt.loss_type == "madnet" and opt.scales != [0, 1, 2, 3, 4, 5, 6]:
    raise ValueError("The 'madnet' loss function requires full image pyramid (i.e set opt.scales to [0, 1, 2, 3, 4, 5, 6])")

  if "monodepth" in opt.loss_type and opt.leftright_consistency == False:
    raise ValueError("The 'monodepth' loss function requires opt.leftright_consistency to be ON")

  if "likelihood" in opt.loss_type and opt.predict_variance == False:
    raise ValueError("The '{}' loss type requires variance predictions".format(opt.loss_type))

  train_dataset = StereoDataset(opt.dataset_path, opt.dataset_name, opt.split, opt.height, opt.width, "train",
                                scales=opt.scales, do_hflip=opt.do_hflip, random_crop=True, load_disp_left=True,
                                load_disp_right=True)

  val_dataset = StereoDataset(opt.dataset_path, opt.dataset_name, opt.split, opt.height, opt.width, "val",
                              scales=opt.scales, do_hflip=False, random_crop=False, load_disp_left=True,
                              load_disp_right=False)

  train_loader = DataLoader(train_dataset, opt.batch_size, not opt.no_shuffle,
      num_workers=opt.num_workers, pin_memory=True, drop_last=False, collate_fn=None)

  val_loader = DataLoader(val_dataset, opt.batch_size, False,
      num_workers=opt.num_workers, pin_memory=True, drop_last=False, collate_fn=None)

  print("DATASET SIZES:  TRAIN={} VAL={}".format(len(train_dataset), len(val_dataset)))

  # Create tensorboard writers for visualizing in the browser.
  writer = SummaryWriter(os.path.join(log_path, "train"))

  warpers = {}
  for s in opt.scales:
    warpers[s] = LinearWarping(opt.height // 2**s, opt.width // 2**s, torch.device("cuda"))

  #================================== TRAINING LOOP ======================================
  epoch, step = 0, 0

  for epoch in range(opt.num_epochs):
    feature_net.train()
    stereo_net.train()

    for bi, inputs in enumerate(train_loader):
      t0 = time.time()

      for key in inputs:
        inputs[key] = inputs[key].cuda()

      outputs = process_batch(feature_net, stereo_net, inputs, output_scales=opt.scales,
                        predict_variance=opt.predict_variance, variance_mode=variance_mode,
                        leftright_consistency=opt.leftright_consistency)
      losses = compute_loss(inputs, outputs, opt, warpers)

      optimizer.zero_grad()
      losses["total_loss"].backward()

      if opt.clip_grad_norm:
        nn.utils.clip_grad_norm_(stereo_net.parameters(), 1.0)

      optimizer.step()

      elapsed_this_batch = time.time() - t0

      early_phase = (step % opt.log_frequency) == 0 and step < 2000
      late_phase = (step % 2000) == 0 or (bi == 0) # Log at start of each epoch.

      if early_phase or late_phase:
        metrics = evaluate(feature_net, stereo_net, val_loader, opt)
        log_scalars(writer, metrics, losses, opt.batch_size / elapsed_this_batch, epoch, step)
        log_images(writer, inputs, outputs, step)

      step += 1

    if epoch >= 1 and (epoch % opt.save_freq) == 0:
      save_models(feature_net, stereo_net, optimizer, log_path, epoch)

    scheduler.step()

  # Do a final stereo_net save after training.
  save_models(feature_net, stereo_net, optimizer, log_path, epoch)


if __name__ == "__main__":
  options = TrainOptions()
  train(options.parse())
  print("Done with training!")

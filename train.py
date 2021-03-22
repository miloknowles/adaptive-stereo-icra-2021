# Copyright 2020 Massachusetts Institute of Technology
#
# @file train.py
# @author Milo Knowles
# @date 2020-07-08 19:08:01 (Wed)

import os, json, time, argparse

import git
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from models.stereo_net import StereoNet, FeatureExtractorNetwork
from models.linear_warping import LinearWarping
from datasets.stereo_dataset import StereoDataset
from utils.loss_functions import khamis_robust_loss_multiscale
from utils.visualization import visualize_disp_tensorboard
from utils.feature_contrast import feature_contrast_mean


def process_batch(feature_net, stereo_net, left, right, opt, output_cost_volume=False):
  left_feat, right_feat = feature_net(left), feature_net(right)
  outputs = stereo_net(left, left_feat, right_feat, "l", output_cost_volume=output_cost_volume)
  return outputs


def log_scalars(writer, metrics, losses, examples_per_sec, epoch, step):
  with torch.no_grad():
    for name in losses:
      writer.add_scalar(name, losses[name], step)
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
  writer.close()


def evaluate(feature_net, stereo_net, val_loader, opt):
  feature_net.eval()
  stereo_net.eval()

  num_batches_eval = len(val_loader) // 10 if opt.fast_eval else len(val_loader)
  if opt.num_steps > 0:
    num_batches_eval = min(opt.num_steps // val_loader.batch_size, len(val_loader))

  with torch.no_grad():
    EPEs = torch.zeros(num_batches_eval)
    D1_alls = torch.zeros(num_batches_eval, 4)
    FCSs = torch.zeros(num_batches_eval)

    for i, inputs in enumerate(val_loader):
      if i >= num_batches_eval:
        break

      for key in inputs:
        inputs[key] = inputs[key].cuda()

      left = inputs["color_l/{}".format(opt.stereonet_input_scale)]
      right = inputs["color_r/{}".format(opt.stereonet_input_scale)]
      outputs = process_batch(feature_net, stereo_net, left, right, opt, output_cost_volume=True)

      pred_disp = outputs["pred_disp_l/{}".format(opt.stereonet_input_scale)]
      gt_disp = inputs["gt_disp_l/{}".format(opt.stereonet_input_scale)]
      valid_mask = (gt_disp > 0)

      # Compute EPE.
      EPEs[i] = torch.abs(pred_disp - gt_disp)[valid_mask].mean()

      # Compute D1-all for several outlier thresholds.
      for oi, ot in enumerate([2, 3, 4, 5]):
        D1_alls[i, oi] = (valid_mask * (torch.abs(pred_disp - gt_disp) > ot)).sum() / float(valid_mask.sum())

      # Compute feature contrast score (FCS).
      FCSs[i] = feature_contrast_mean(outputs["cost_volume_l/{}".format(opt.stereonet_input_scale + opt.stereonet_k)]).mean()

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
    opt_readable = json.dumps(opt.__dict__, sort_keys=True, indent=2)
    print("--------------------------------------------------------------------")
    print("TRAINING OPTIONS:")
    print(opt_readable)
    f.write(opt_readable + "\n")
    print("--------------------------------------------------------------------")

  feature_net = FeatureExtractorNetwork(opt.stereonet_k).cuda()
  stereo_net = StereoNet(opt.stereonet_k, 1, opt.stereonet_input_scale).cuda()

  parameters_to_train = [{"params": stereo_net.parameters()}, {"params": feature_net.parameters()}]
  optimizer = optim.Adam(parameters_to_train, lr=opt.learning_rate)
  scheduler = optim.lr_scheduler.StepLR(optimizer, opt.scheduler_step_size, 0.5)

  # If a folder for pretrained weights is given, load from there.
  if opt.load_weights_folder is not None:
    print("Loading models from: ", opt.load_weights_folder)
    feature_net.load_state_dict(torch.load(os.path.join(opt.load_weights_folder, "feature_net.pth")), strict=True)
    stereo_net.load_state_dict(torch.load(os.path.join(opt.load_weights_folder, "stereo_net.pth")), strict=True)

  loss_scales = [opt.stereonet_input_scale, opt.stereonet_input_scale + opt.stereonet_k]
  train_dataset = StereoDataset(opt.dataset_path, opt.dataset_name, opt.split, opt.height, opt.width, "train",
                                scales=loss_scales, do_hflip=opt.do_hflip, random_crop=True, load_disp_left=True,
                                load_disp_right=True)

  val_dataset = StereoDataset(opt.dataset_path, opt.dataset_name, opt.split, opt.height, opt.width, "val",
                              scales=loss_scales, do_hflip=False, random_crop=False, load_disp_left=True,
                              load_disp_right=False)

  train_loader = DataLoader(train_dataset, opt.batch_size, not opt.no_shuffle,
      num_workers=opt.num_workers, pin_memory=True, drop_last=False, collate_fn=None)

  val_loader = DataLoader(val_dataset, opt.batch_size, False,
      num_workers=opt.num_workers, pin_memory=True, drop_last=False, collate_fn=None)

  print("----------------------------------------------------------------------")
  print("DATASET SIZES:\n  TRAIN={} VAL={}".format(len(train_dataset), len(val_dataset)))
  print("----------------------------------------------------------------------")

  # Create tensorboard writers for visualizing in the browser.
  writer = SummaryWriter(os.path.join(log_path, "val"))

  #================================== TRAINING LOOP ======================================
  epoch, step = 0, 0

  for epoch in range(opt.num_epochs):
    feature_net.train()
    stereo_net.train()

    for bi, inputs in enumerate(train_loader):
      t0 = time.time()

      for key in inputs:
        inputs[key] = inputs[key].cuda()

      outputs = process_batch(
          feature_net, stereo_net,
          inputs["color_l/{}".format(opt.stereonet_input_scale)],
          inputs["color_r/{}".format(opt.stereonet_input_scale)], opt)

      losses = khamis_robust_loss_multiscale(inputs, outputs, scales=loss_scales, gt_disp_scale=opt.stereonet_input_scale)

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


class TrainOptions(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser(description="Options for training StereoNet")

    self.parser.add_argument("--height", type=int, default=320, help="Image height (must be divisble by 2**(opt.stereonet_k + opt.stereonet_input_scale)")
    self.parser.add_argument("--width", type=int, default=960, help="Image width (must be divisble by 2**(opt.stereonet_k + opt.stereonet_input_scale)")
    self.parser.add_argument("--model_name", type=str, help="The name for this training experiment")

    self.parser.add_argument("--stereonet_input_scale", default=0, type=int, help="Scale for input images to StereoNet")
    self.parser.add_argument("--stereonet_k", type=int, default=3, choices=[3, 4], help="The cost volume downsampling factor")

    # Dataset options.
    self.parser.add_argument("--dataset_path", type=str, help="Top level folder for the dataset being used")
    self.parser.add_argument("--dataset_name", type=str, default="SceneFlowDriving", help="Which dataset to train on")
    self.parser.add_argument("--split", type=str, help="The dataset split to train on")
    self.parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    self.parser.add_argument("--do_hflip", action="store_true", default=False, help="Do horizontal flip augmentation during training")
    self.parser.add_argument("--no_shuffle", action="store_true", default=False, help="Turn off dataset shuffling (i.e for sequential adaptation)")
    self.parser.add_argument("--use_grayscale", action="store_true", help="Convert training images to grayscale")

    # Output and saving options.
    self.parser.add_argument("--log_dir", type=str, default="/home/milo/training_logs", help="Directory for saving tensorboard events and models")
    self.parser.add_argument("--load_weights_folder", default=None, type=str, help="Path to load pretrained weights from")
    self.parser.add_argument("--load_adam", action="store_true", default=False, help="Load the Adam optimizer state to resume training")
    self.parser.add_argument("--scheduler_step_size", default=5, type=int, help="Reduce LR by 1/2 after this many epochs")
    self.parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    self.parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    self.parser.add_argument("--log_frequency", type=int, default=250, help="Number of batches between tensorboard logging")
    self.parser.add_argument("--save_freq", type=int, default=1, help="Save model weights after every x epochs")
    self.parser.add_argument("--fast_eval", action="store_true", default=False, help="Speeds up evaluation by only doing the first few batches")

    # Optimizer options.
    self.parser.add_argument("--learning_rate", default=1e-5, type=float, help="Learning rate for Adam optimizer")
    self.parser.add_argument("--clip_grad_norm", action="store_true", default=False, help="Clip gradient norms to 1.0")

    # Adaptation options.
    self.parser.add_argument("--leftright_consistency", action="store_true", default=False, help="Predict disparity for the left and right images")
    self.parser.add_argument("--smoothness_weight", type=float, default=1e-3, help="Smoothness loss coefficient, as in MonoDepth2")
    self.parser.add_argument("--consistency_weight", type=float, default=1e-3, help="Left-right consistency loss coefficient, as in MonoDepth1")
    self.parser.add_argument("--num_steps", type=int, default=-1, help="Limit to this many adaptation gradient descent updates")
    self.parser.add_argument("--ovs_buffer_size", type=int, default=10, help="Size of the online validation set (OVS)")
    self.parser.add_argument("--skip_initial_eval", action="store_true", help="Skip evaluation the pre-adaptation model")
    self.parser.add_argument("--ovs_validate_hz", type=int, default=100, help="How often to test the validation buffer")
    self.parser.add_argument("--adapt_mode", choices=["NONSTOP", "VS", "ER", "VS+ER"], help="Adaptation method")
    self.parser.add_argument("--val_improve_retries", type=int, default=1, help="Stop adaptation if loss hasn't improved after this many re-validations")
    self.parser.add_argument("--eval_hz", type=int, default=1000, help="Evaluate after this many steps. 0 means at the end of each epoch.")
    self.parser.add_argument("--er_loss_weight", type=float, default=0.05, help="Weight for the experience replay loss during adaptation.")
    self.parser.add_argument("--train_dataset_path", type=str, help="Path to the training dataset folder. Used to evaluate the effect of adaptation on training domain performance.")
    self.parser.add_argument("--train_dataset_name", type=str, help="Name of the training set class (e.g KittiRaw)")
    self.parser.add_argument("--train_split", type=str, help="Training split used for validation")
    self.parser.add_argument("--ood_threshold", type=float, default=15.0, help="Reconstruction loss cutoff threshold for 'OOD'")
    self.parser.add_argument("--fcs_ema_weight", type=float, default=0.999, help="Weight parameter for exponential moving average")

  def parse(self):
    self.options = self.parser.parse_args()
    return self.options


if __name__ == "__main__":
  options = TrainOptions()
  train(options.parse())
  print("Done with training!")

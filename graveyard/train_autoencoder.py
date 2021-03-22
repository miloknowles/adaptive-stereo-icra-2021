# Copyright 2020 Massachusetts Institute of Technology
#
# @file train_autoencoder.py
# @author Milo Knowles
# @date 2020-10-09 15:54:16 (Fri)

import os, json, time, argparse

import git
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from train_options import TrainOptions
from models.stereo_net import FeatureExtractorNetwork
from models.autoencoder import ConvolutionalEncoder, ConvolutionalDecoder
from models.vae import VAE, vae_loss_function
from datasets.stereo_dataset import StereoDataset
from utils.loss_functions import SSIM


def process_batch(feature_net, decoder, left, right, opt):
  """
  Predicts the reconstructed left image using the encoder (feature_net) and decoder_net.
  """
  with torch.set_grad_enabled(opt.train_encoder):
    fl = feature_net(left)

  reconstructed_left = decoder(fl)
  return reconstructed_left


def image_reconstruction_loss(decoder_l, image_l, ssim=False):
  # print("Reconstruction loss:", decoder_l.shape, image_l.shape)
  assert(decoder_l.shape == image_l.shape)

  if ssim:
    return 0.85*SSIM(decoder_l, image_l).mean() + 0.15*F.l1_loss(decoder_l, image_l)
  else:
    # NOTE: Fractional norm here seems to cause numerical issues...
    return torch.abs(decoder_l - image_l).mean()


def log_scalars(writer, losses, examples_per_sec, epoch, step):
  with torch.no_grad():
    for name in losses:
      writer.add_scalar(name, losses[name], step)

    writer.add_scalar("examples_per_sec", examples_per_sec, step)
    writer.close()

    print("\n{}|{}========================================================================".format(epoch, step))
    print("TIMING  // examples/sec={:.3f}".format(examples_per_sec))

    if len(losses) > 0:
      loss_str = "LOSS    // "
      for name in losses:
        loss_str += " | {}={:.5f}".format(name, losses[name])
      print(loss_str)
    print("===========================================================================\n")

  writer.close()


def contains_prefix(name, prefixes):
  for p in prefixes:
    if p in name:
      return True
  return False


def log_images(writer, inputs, outputs, step, skip_prefixes=["cost_volume", "gt_disp", "color_r", "mu", "logvar"]):
  with torch.no_grad():
    for io in (inputs, outputs):
      for name in io:
        if contains_prefix(name, skip_prefixes):
          continue

        viz = io[name][0].detach().cpu()
        writer.add_image(name, viz, step)

  # NOTE: It seems like the last few images don't get logged properly without this...
  writer.close()


def evaluate(feature_net, decoder_net, val_loader, opt):
  feature_net.eval()
  decoder_net.eval()

  # Fast eval mode only validates on 1/10th of validation set.
  num_batches_eval = len(val_loader) // 10 if opt.fast_eval else len(val_loader)

  losses = {}
  val_loss = torch.zeros(num_batches_eval)

  with torch.no_grad():
    for i, inputs in enumerate(val_loader):
      if i >= num_batches_eval:
        break

      for key in inputs:
        inputs[key] = inputs[key].cuda()

      outputs = {}
      color_l_input = inputs["color_l/{}".format(opt.decoder_loss_scale if opt.encoder_type == "ConvolutionalEncoder" else 0)]
      outputs["decoded_l/{}".format(opt.decoder_loss_scale)] = process_batch(
          feature_net, decoder_net, color_l_input, inputs["color_r/0"], opt)

      val_loss[i] = image_reconstruction_loss(
          outputs["decoded_l/{}".format(opt.decoder_loss_scale)],
          inputs["color_l/{}".format(opt.decoder_loss_scale)])

  losses["total_loss"] = val_loss.mean()

  decoder_net.train()

  return losses


def evaluate_vae(vae_net, val_loader, opt):
  vae_net.eval()

  # Fast eval mode only validates on 1/10th of validation set.
  num_batches_eval = len(val_loader) // 10 if opt.fast_eval else len(val_loader)

  losses = {
    "vae/total_loss": torch.zeros(num_batches_eval),
    "vae/reconstruction_loss": torch.zeros(num_batches_eval),
    "vae/KL_divergence_loss": torch.zeros(num_batches_eval)
  }

  with torch.no_grad():
    for i, inputs in enumerate(val_loader):
      if i >= num_batches_eval:
        break

      for key in inputs:
        inputs[key] = inputs[key].cuda()

      decoded, mu, logvar = vae_net(inputs["color_l/{}".format(opt.decoder_loss_scale)])
      losses["vae/total_loss"][i], losses["vae/reconstruction_loss"][i], losses["vae/KL_divergence_loss"][i] = \
          vae_loss_function(decoded, inputs["color_l/{}".format(opt.decoder_loss_scale)], mu, logvar)

  for key in losses:
    losses[key] = losses[key].mean()

  vae_net.train()

  return losses


def save_models(feature_net, decoder_net, vae_net, optimizer, log_path, epoch):
  save_folder = os.path.join(log_path, "models", "weights_{}".format(epoch))
  os.makedirs(save_folder, exist_ok=True)

  if feature_net is not None:
    torch.save(feature_net.state_dict(), os.path.join(save_folder, "feature_net.pth"))

  if decoder_net is not None:
    torch.save(decoder_net.state_dict(), os.path.join(save_folder, "decoder_net.pth"))

  if vae_net is not None:
    torch.save(vae_net.state_dict(), os.path.join(save_folder, "vae_net.pth"))

  if optimizer is not None:
    torch.save(optimizer.state_dict(), os.path.join(save_folder, "adam.pth"))


def train(opt):
  torch.manual_seed(123)

  # https://github.com/pytorch/pytorch/issues/15054
  torch.backends.cudnn.benchmark = True

  log_path = os.path.join(opt.log_dir, opt.name)
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

  if opt.vae:
    vae_net = VAE(image_channels=3, z_dim=opt.vae_bottleneck, input_height=opt.height // 2**opt.decoder_loss_scale,
                  input_width=opt.width // 2**opt.decoder_loss_scale).cuda()
    parameters_to_train = vae_net.parameters()
  else:
    if opt.encoder_type == "ConvolutionalEncoder":
      feature_net = ConvolutionalEncoder(3, 16, opt.stereonet_k - opt.decoder_loss_scale).cuda()
      decoder_net = ConvolutionalDecoder(16, 3, opt.stereonet_k - opt.decoder_loss_scale).cuda()
    elif opt.encoder_type == "FeatureExtractorNetwork":
      feature_net = FeatureExtractorNetwork(opt.stereonet_k).cuda()
      decoder_net = ConvolutionalDecoder(32, 3, opt.stereonet_k - opt.decoder_loss_scale).cuda()
    parameters_to_train = [{"params": decoder_net.parameters()}]
    if opt.train_encoder:
      parameters_to_train.append({"params": feature_net.parameters()})

  optimizer = optim.Adam(parameters_to_train, lr=opt.learning_rate)
  scheduler = optim.lr_scheduler.StepLR(optimizer, opt.scheduler_step_size, 0.5)

  # If a folder for pretrained weights is given, load from there.
  if opt.feature_weights_folder is not None:
    print("Loading feature network from:", opt.feature_weights_folder)
    feature_net.load_state_dict(torch.load(os.path.join(opt.feature_weights_folder, "feature_net.pth")), strict=True)

  if opt.decoder_weights_folder is not None:
    print("Loading decoder network from:", opt.decoder_weights_folder)
    decoder_net.load_state_dict(torch.load(os.path.join(opt.decoder_weights_folder, "decoder_net.pth")), strict=True)

  if opt.vae_weights_folder is not None:
    print("Loading VAE from:", opt.vae_weights_folder)
    vae_net.load_state_dict(torch.load(os.path.join(opt.vae_weights_folder, "vae_net.pth")), strict=True)

  image_scales = [0]
  if opt.decoder_loss_scale != 0:
    image_scales.append(opt.decoder_loss_scale)
  train_dataset = StereoDataset(opt.dataset_path, opt.dataset_name, opt.split, opt.height, opt.width, "train",
                                scales=image_scales, do_hflip=opt.do_hflip, random_crop=True, load_disp_left=True,
                                load_disp_right=True)

  val_dataset = StereoDataset(opt.dataset_path, opt.dataset_name, opt.split, opt.height, opt.width, "val",
                              scales=image_scales, do_hflip=False, random_crop=False, load_disp_left=True,
                              load_disp_right=False)

  train_loader = DataLoader(train_dataset, opt.batch_size, True,
      num_workers=opt.num_workers, pin_memory=True, drop_last=False, collate_fn=None)

  val_loader = DataLoader(val_dataset, opt.batch_size, False,
      num_workers=opt.num_workers, pin_memory=True, drop_last=False, collate_fn=None)

  print("----------------------------------------------------------------------")
  print("DATASET SIZES:\n  TRAIN={} VAL={}".format(len(train_dataset), len(val_dataset)))
  print("----------------------------------------------------------------------")

  # Create tensorboard writers for visualizing in the browser.
  train_writer = SummaryWriter(os.path.join(log_path, "train"))
  val_writer = SummaryWriter(os.path.join(log_path, "val"))

  #================================== TRAINING LOOP ======================================
  epoch, step = 0, 0

  for epoch in range(opt.num_epochs):
    if opt.vae:
      vae_net.train()
    else:
      if opt.train_encoder:
        feature_net.train()
      else:
        feature_net.eval()
      decoder_net.train()

    for bi, inputs in enumerate(train_loader):
      t0 = time.time()

      for key in inputs:
        inputs[key] = inputs[key].cuda()

      outputs = {}
      # encoder_input_scale = opt.decoder_loss_scale if opt.encoder_type == "ConvolutionalEncoder" else 0
      # color_l = inputs["color_l/{}".format(encoder_input_scale)]
      # print("Decoder scale:", opt.decoder_loss_scale)
      if opt.vae:
        decoded, mu, logvar = vae_net(inputs["color_l/{}".format(opt.decoder_loss_scale)])
        outputs["decoded_l/{}".format(opt.decoder_loss_scale)] = decoded
        outputs["mu_l/{}".format(opt.decoder_loss_scale)] = mu
        outputs["logvar_l/{}".format(opt.decoder_loss_scale)] = logvar

        losses = {}
        losses["total_loss"], losses["reconstruction_loss"], losses["KL_divergence_loss"] = \
            vae_loss_function(decoded, inputs["color_l/{}".format(opt.decoder_loss_scale)], mu, logvar)
      else:
        color_l_input = inputs["color_l/{}".format(opt.decoder_loss_scale if opt.encoder_type == "ConvolutionalEncoder" else 0)]
        outputs["decoded_l/{}".format(opt.decoder_loss_scale)] = process_batch(
            feature_net, decoder_net,
            color_l_input,
            inputs["color_r/0"], opt)

        losses = {}
        losses["total_loss"] = image_reconstruction_loss(
            outputs["decoded_l/{}".format(opt.decoder_loss_scale)],
            inputs["color_l/{}".format(opt.decoder_loss_scale)], ssim=opt.ssim)

      optimizer.zero_grad()
      losses["total_loss"].backward()
      optimizer.step()

      elapsed_this_batch = time.time() - t0

      early_phase = (step % opt.log_frequency) == 0 and epoch < 1
      late_phase = (step % 2000) == 0 or (bi == 0) # Log at start of each epoch.

      if early_phase or late_phase:
        if opt.vae:
          val_losses = evaluate_vae(vae_net, val_loader, opt)
        else:
          val_losses = evaluate(feature_net, decoder_net, val_loader, opt)

        # Log the validation losses.
        log_scalars(val_writer, val_losses, opt.batch_size / elapsed_this_batch, epoch, step)

        # Periodically show examples of images and reconstructions.
        log_images(train_writer, inputs, outputs, step)

      step += 1

    if epoch >= 1 and (epoch % opt.save_freq) == 0:
      save_models(None if opt.vae else feature_net, None if opt.vae else decoder_net,
                  vae_net if opt.vae else None, optimizer, log_path, epoch)

    scheduler.step()

  # Do a final stereo_net save after training.
  save_models(None if opt.vae else feature_net, None if opt.vae else decoder_net,
                  vae_net if opt.vae else None, optimizer, log_path, epoch)


class AETrainOptions(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser(description="Options for training the autoencoder")

    self.parser.add_argument("--height", type=int, default=320, help="Image height (must be // by 16)")
    self.parser.add_argument("--width", type=int, default=960, help="Image width (must be // by 6)")
    self.parser.add_argument("--name", type=str, help="The name for this training experiment")
    self.parser.add_argument("--train_encoder", action="store_true", default=False)
    self.parser.add_argument("--encoder_type", default="FeatureExtractorNetwork", choices=["ConvolutionalEncoder", "FeatureExtractorNetwork"])
    self.parser.add_argument("--vae", action="store_true", default=False, help="Use a Variational Autoencoder")
    self.parser.add_argument("--vae_bottleneck", type=int, default=32, help="Dimension for VAE bottleneck features")

    # Training dataset options.
    self.parser.add_argument("--dataset_path", type=str, help="Top level folder for the dataset being used")
    self.parser.add_argument("--dataset_name", type=str, default="SceneFlowFlying", help="Which dataset to train on")
    self.parser.add_argument("--split", type=str, help="The dataset split to train on")
    self.parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    self.parser.add_argument("--do_hflip", action="store_true", default=False, help="Do horizontal flip augmentation during training")

    # Output and saving options.
    self.parser.add_argument("--log_dir", type=str, default="/home/milo/training_logs", help="Directory for saving tensorboard events and models")
    self.parser.add_argument("--feature_weights_folder", default=None, type=str, help="Path to load pretrained feature network from")
    self.parser.add_argument("--decoder_weights_folder", default=None, type=str, help="Path to load pretrained decoder network from")
    self.parser.add_argument("--vae_weights_folder", default=None, type=str, help="Path to load pretrained VAE network from")
    self.parser.add_argument("--load_adam", action="store_true", default=False, help="Load the Adam optimizer state to resume training")
    self.parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate for Adam optimizer")
    self.parser.add_argument("--scheduler_step_size", default=10, type=int, help="Reduce LR by 1/2 after this many epochs")
    self.parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    self.parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    self.parser.add_argument("--log_frequency", type=int, default=250, help="Number of batches between tensorboard logging")
    self.parser.add_argument("--save_freq", type=int, default=1, help="Save model weights after every x epochs")
    self.parser.add_argument("--fast_eval", action="store_true", default=False, help="Speeds up evaluation by only doing the first few batches")

    # Loss function options.
    self.parser.add_argument("--decoder_loss_scale", type=int, default=2, help="The scale at which image reconstruction loss is computed (e.g 0 means full resolution, 1 means 1/2 resolution)")
    self.parser.add_argument("--stereonet_k", type=int, default=4, choices=[4, 5], help="The cost volume downsampling factor (e.g 4 means 1/16 resolution cost volume)")
    self.parser.add_argument("--ssim", action="store_true", default=False, help="Use perceptual loss (0.85*SSIM + 0.15*L1)")

    # Adaptation options.
    self.parser.add_argument("--clip_grad_norm", action="store_true", default=False, help="Clip gradient norms to 1.0")
    self.parser.add_argument("--num_steps", type=int, default=4000, help="Limit to this many adaptation gradient descent updates")
    self.parser.add_argument("--ovs_buffer_size", type=int, default=8, help="Size of the online validation set (OVS)")
    self.parser.add_argument("--skip_initial_eval", action="store_true", help="Skip evaluation the pre-adaptation model")
    self.parser.add_argument("--ovs_validate_hz", type=int, default=100, help="How often to test the validation buffer")
    self.parser.add_argument("--adapt_mode", choices=["NONSTOP", "VS", "ER", "VS+ER"], help="Adaptation method")
    self.parser.add_argument("--val_improve_retries", type=int, default=1, help="Stop adaptation if loss hasn't improved after this many re-validations")
    self.parser.add_argument("--eval_hz", type=int, default=1000, help="Evaluate after this many steps. 0 means at the end of each epoch.")
    self.parser.add_argument("--er_loss_weight", type=float, default=0.05, help="Weight for the experience replay loss during adaptation.")
    self.parser.add_argument("--train_dataset_path", type=str, help="Path to the training dataset folder. Used to evaluate the effect of adaptation on training domain performance.")
    self.parser.add_argument("--train_dataset_name", type=str, help="Name of the training set class (e.g KittiRaw)")
    self.parser.add_argument("--train_split", type=str, help="Training split used for validation")
    self.parser.add_argument("--ood_threshold", type=float, default=0.0, help="The cutoff threshold for us to classify a feature contrast score as 'OOD'")
    self.parser.add_argument("--load_weights_folder", default=None, type=str, help="Path to load pretrained StereoNet weights from")
    self.parser.add_argument("--vae_learning_rate", default=1e-4, type=float, help="Learning rate for Adam optimizer")

  def parse(self):
    self.options = self.parser.parse_args()
    return self.options


if __name__ == "__main__":
  options = AETrainOptions()
  train(options.parse())
  print("Done with training!")

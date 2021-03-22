# Copyright 2020 Massachusetts Institute of Technology
#
# @file vae_ood_analysis.py
# @author Milo Knowles
# @date 2020-10-12 13:39:05 (Mon)

import os, argparse, math, random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
import pandas as pd
import scipy.stats as stats

from models.vae import *
from datasets.stereo_dataset import StereoDataset
from utils.path_utils import *
from utils.visualization import *

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 20

torch.manual_seed(123)
random.seed(123)


def image_reconstruction_loss(decoder_l, image_l, p=1.0):
  """
  Return image reconstruction loss, preserving batch dimension.
  """
  assert(decoder_l.shape == image_l.shape)
  assert(len(decoder_l.shape) == 4)

  loss = torch.abs(decoder_l - image_l)**p
  return loss.mean(dim=(-3, -2, -1))


def save_loss(train_loaders, novel_loaders, output_folder, opt, num_train=1000, num_novel=1000, save_images=True):
  os.makedirs(output_folder, exist_ok=True)

  vae_net = VAE(image_channels=3, z_dim=opt.vae_bottleneck,
                input_height=opt.height // 2**opt.decoder_loss_scale,
                input_width=opt.width // 2**opt.decoder_loss_scale).cuda()
  vae_net.load_state_dict(torch.load(os.path.join(
      opt.load_weights_folder, "vae_net.pth")), strict=True)

  vae_net.eval()

  train_mu = torch.zeros(num_train, opt.vae_bottleneck)
  novel_mu = torch.zeros(num_novel, opt.vae_bottleneck)
  train_loss = torch.zeros(num_train)
  novel_loss = torch.zeros(num_novel)

  with torch.no_grad():
    num_each_train = num_train // len(train_loaders)
    num_each_novel = num_novel // len(novel_loaders)
    print("Will use {} images from each training set".format(num_each_train))
    print("Will use {} images from each novel set".format(num_each_novel))

    for loader_type, loader_list in [("train", train_loaders), ("novel", novel_loaders)]:
      print("Processing {} datasets".format(loader_type))
      ctr = 0
      save_ctr = 0
      for loader in loader_list:
        for i, inputs in enumerate(loader):
          if (loader_type == "train" and i*opt.batch_size >= num_each_train) \
            or (loader_type == "novel" and i*opt.batch_size >= num_each_novel):
            break

          for key in inputs:
            inputs[key] = inputs[key].cuda()

          color_l = inputs["color_l/{}".format(opt.decoder_loss_scale)]
          decoded_l, mu_batch, _ = vae_net(color_l)
          loss_batch = image_reconstruction_loss(decoded_l, color_l, p=1.0)

          assert(loss_batch.shape[0] == color_l.shape[0])

          if save_images and save_ctr < 10:
            viz_decoded = tensor_to_cv_rgb(decoded_l[0].cpu())
            viz_original = tensor_to_cv_rgb(color_l[0].cpu())
            cv.imwrite(os.path.join(output_folder, "{}_decoded_{}.png".format(loader_type, save_ctr)), viz_decoded)
            cv.imwrite(os.path.join(output_folder, "{}_original_{}.png".format(loader_type, save_ctr)), viz_original)
            save_ctr += 1

          if loader_type == "train":
            num_remaining = min(len(loss_batch), len(train_loss) - ctr)
            train_loss[ctr:ctr+num_remaining] = loss_batch[:num_remaining]
            train_mu[ctr:ctr+num_remaining] = mu_batch[:num_remaining]
          else:
            num_remaining = min(len(loss_batch), len(novel_loss) - ctr)
            novel_loss[ctr:ctr+num_remaining] = loss_batch[:num_remaining]
            novel_mu[ctr:ctr+num_remaining] = mu_batch[:num_remaining]

          ctr += num_remaining
          print("Finished {}/{} images".format(ctr, len(train_loss)))

  # Approximating the entire training set as a multivariate Gaussian, compete the mean and variance.
  mean = train_mu.mean(dim=0)
  residuals = train_mu - mean.unsqueeze(0) # N x d
  N = train_mu.shape[0]
  covariance = torch.matmul(residuals.transpose(0, 1), residuals) / (N - 1) # d x d
  assert(mean.shape == (opt.vae_bottleneck,))
  assert(residuals.shape == (N, opt.vae_bottleneck))

  torch.save(mean, os.path.join(output_folder, "train_dist_mu.pt"))
  torch.save(covariance, os.path.join(output_folder, "train_dist_sigma.pt"))
  print("Saved mean and covariance to:", output_folder)

  assert((train_loss > 0).all())
  assert((novel_loss > 0).all())

  torch.save(train_loss, os.path.join(output_folder, "train_loss.pt"))
  torch.save(novel_loss, os.path.join(output_folder, "novel_loss.pt"))
  torch.save(train_mu, os.path.join(output_folder, "train_mu.pt"))
  torch.save(novel_mu, os.path.join(output_folder, "novel_mu.pt"))
  print("Saved to", output_folder)


def compute_precision_recall(train_values, novel_values, cutoff):
  """
  We classify an example as "novel" if its value is <= cutoff, and "train" otherwise.
  """
  tp = (novel_values <= cutoff).sum().item()   # Labelled positive and should be positive.
  fn = (novel_values > cutoff).sum().item()    # Labelled negative but actually positive.
  tn = (train_values > cutoff).sum().item()    # Labelled negative and should be negative.
  fp = (train_values <= cutoff).sum().item()   # Labelled positive but actually negative.
  pr = float(tp) / (tp + fp) if (tp + fp) > 0 else 1.0
  re = float(tp) / (tp + fn)
  return pr, re


def plot_precision_recall(loss_train, loss_novel, output_folder, opt, show=False):
  """
  Make a precision recall plot using image reconstruction losses for train and novel images.
  """
  plt.clf()

  cutoff_values = np.linspace(loss_novel.min(), loss_novel.max(), num=100)
  print("Cutoff value range: min={} max={}".format(cutoff_values.min(), cutoff_values.max()))
  precision = np.zeros(len(cutoff_values))
  recall = np.zeros(len(cutoff_values))
  for i, cutoff in enumerate(cutoff_values):
    pr, re = compute_precision_recall(loss_train, loss_novel, cutoff)
    precision[i] = pr
    recall[i] = re

  torch.save(torch.from_numpy(precision), os.path.join(output_folder, "precision.pt"))
  torch.save(torch.from_numpy(recall), os.path.join(output_folder, "recall.pt"))

  df = pd.DataFrame({"precision": precision, "recall": recall})
  plt.plot(recall, precision, color="tab:red")
  plt.xlabel("recall")
  plt.ylabel("precision")
  if show:
    plt.show()

  plt.savefig(os.path.join(output_folder, "pr_{}.pdf".format(opt.environment)), bbox_inches="tight")
  print("Saved pr_{}.pdf".format(opt.environment))


def strictly_decreasing_precision(pr, re):
  """
  Post-processes a precision recall curve so that it is strictly decreasing and looks nicer.
  """
  # Sort recall from highest to lowest.
  index_order = np.argsort(re)
  re_sorted = np.flip(re[index_order])
  pr_sorted = np.flip(pr[index_order])

  highest_pr_so_far = 0
  pr_fixed = np.zeros(len(pr))
  for i, r in enumerate(re_sorted):
    highest_pr_so_far = max(highest_pr_so_far, pr_sorted[i])
    pr_fixed[i] = highest_pr_so_far

  return np.flip(re_sorted), np.flip(pr_fixed)


def plot_precision_recall_multiple(pr, output_folder, opt):
  """
  pr (dict) :
    Each key should be the desired name to be displayed in the legend (e.g SF → VK).
    Each value should be a tuple of (precision values, recall values, color string).
  """
  plt.clf()
  for label_name, (precision, recall, color) in pr.items():
    print("Plotting", label_name)
    recall_fixed, precision_fixed = strictly_decreasing_precision(precision, recall)
    plt.plot(recall_fixed, precision_fixed, color=color, label=label_name)

  plt.xlabel("recall")
  plt.ylabel("precision")
  plt.legend(fontsize="small")
  plt.savefig(os.path.join(output_folder, "pr_multiple.pdf"), bbox_inches="tight")
  print("Saved pr_multiple.pdf to", output_folder)


def plot_histogram(loss_train, loss_novel, output_folder, opt, percentile=0.01,
                   show=False, normal=True, legend=False):
  plt.clf()

  bins = np.histogram(np.hstack((loss_train, loss_novel)), bins=40)[1]

  y1, x, _ = plt.hist(loss_train, bins, facecolor="blue", density=True, alpha=0.5, label="train")
  y2, x, _ = plt.hist(loss_novel, bins, facecolor="red", density=True, alpha=0.5, label="novel")
  plt.xlabel("image reconstruction loss")
  plt.ylabel("frequency")

  if normal:
    mu, sigma = loss_train.mean(), math.sqrt(loss_train.var())
    pct_x = stats.norm.ppf(percentile, loc=mu, scale=sigma)
    print("Plotting {}th percentile".format(percentile))
    print("{}th percentile occurs at x={}".format(percentile, pct_x))
    plt.vlines(pct_x, 0, max(y1.max(), y2.max()), colors="black", linestyles=(0, (5, 5)))
    plt.plot(bins, stats.norm.pdf(bins, mu, sigma), color="black", linestyle="solid")

  if legend:
    plt.legend(loc="upper right", fontsize="large")

  if show:
    plt.show()

  plt.savefig(os.path.join(output_folder, "histogram_{}.pdf".format(opt.environment)), bbox_inches="tight")
  plt.savefig(os.path.join(output_folder, "histogram_{}.png".format(opt.environment)), bbox_inches="tight")
  print("Saved histogram_{}.pdf/png".format(opt.environment))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--load_weights_folder", default=None, type=str,
                      help="Path to load pretrained weights from")
  parser.add_argument("--stereonet_k", type=int, default=4, choices=[3, 4],
                      help="The cost volume downsampling factor")
  parser.add_argument("--decoder_loss_scale", type=int, default=2, help="Scale that the decoder was trained to predict images at")
  parser.add_argument("--batch_size", type=int, default=8, help="Batch size for loading images")
  parser.add_argument("--mode", type=str, choices=["save_loss", "histogram", "pr", "pr_multiple"])
  parser.add_argument("--percentile", type=float, default=0.99,
                      help="Visualize percentile as a vertical line on the histogram")
  parser.add_argument("--environment", type=str, default="sf_to_vk",
                      choices=["vk_to_sf", "sf_to_kitti", "vk_to_kitti", "vk_weather", "sf_to_vk"])
  parser.add_argument("--ssim", action="store_true", default=False, help="Use SSIM in autoencoder loss")
  parser.add_argument("--normalize", default=None, choices=["color", "intensity", "stdev"],
                      help="Method for normalizing loss values based on image textures")
  parser.add_argument("--vae_bottleneck", type=int, default=64, help="Dimension of the VAE latent embedding")
  parser.add_argument("--height", type=int, default=320, help="Height of input images at full resolution")
  parser.add_argument("--width", type=int, default=960, help="Width of input images at full resolution")
  parser.add_argument("--alpha", type=float, default=1.0, help="Weight for the reconstruction loss")
  parser.add_argument("--beta", type=float, default=0, help="Weight for the Mahalanobis distance loss")
  opt = parser.parse_args()

  assert(opt.percentile >= 0.001 and opt.percentile <= 0.999)
  print("------------------- ENVIRONMENT: {} --------------------".format(opt.environment))

  scale_list = [opt.decoder_loss_scale]

  # Virtual KITTI => SceneFlow Flying
  if opt.environment == "vk_to_sf":
    train_datasets = [
      StereoDataset("/home/milo/datasets/virtual_kitti", "VirtualKitti", "virtual_kitti_clone",
                    opt.height, opt.width, "train", scales=scale_list, load_disp_left=False, load_disp_right=False)
    ]
    novel_datasets = [
      StereoDataset("/home/milo/datasets/sceneflow_flying_things_3d", "SceneFlowFlying", "sceneflow_flying",
                  opt.height, opt.width, "train", scales=scale_list, load_disp_left=False, load_disp_right=False)
    ]

  # Virtual KITTI Clone => Virtual KITTI Fog/Rain
  elif opt.environment == "vk_weather":
    train_datasets = [
      StereoDataset("/home/milo/datasets/virtual_kitti", "VirtualKitti", "virtual_kitti_clone",
                    opt.height, opt.width, "train", scales=scale_list, load_disp_left=False, load_disp_right=False)
    ]
    novel_datasets = [
      StereoDataset("/home/milo/datasets/virtual_kitti", "VirtualKitti", "virtual_kitti_fog",
                    opt.height, opt.width, "train", scales=scale_list, load_disp_left=False, load_disp_right=False),
      StereoDataset("/home/milo/datasets/virtual_kitti", "VirtualKitti", "virtual_kitti_rain",
                    opt.height, opt.width, "train", scales=scale_list, load_disp_left=False, load_disp_right=False),
    ]

  # Virtual KITTI => KITTI Raw
  elif opt.environment == "vk_to_kitti":
    train_datasets = [
      StereoDataset("/home/milo/datasets/virtual_kitti", "VirtualKitti", "virtual_kitti_clone",
                    opt.height, opt.width, "train", scales=scale_list, load_disp_left=False, load_disp_right=False)
    ]
    novel_datasets = [
      StereoDataset("/home/milo/datasets/kitti_data_raw", "KittiRaw", "kitti_raw_campus_adapt",
                    opt.height, opt.width, "train", scales=scale_list, load_disp_left=False, load_disp_right=False),
      StereoDataset("/home/milo/datasets/kitti_data_raw", "KittiRaw", "kitti_raw_city_adapt",
                    opt.height, opt.width, "train", scales=scale_list, load_disp_left=False, load_disp_right=False),
      StereoDataset("/home/milo/datasets/kitti_data_raw", "KittiRaw", "kitti_raw_road_adapt",
                    opt.height, opt.width, "train", scales=scale_list, load_disp_left=False, load_disp_right=False),
      StereoDataset("/home/milo/datasets/kitti_data_raw", "KittiRaw", "kitti_raw_residential_adapt",
              opt.height, opt.width, "train", scales=scale_list, load_disp_left=False, load_disp_right=False)
    ]

  # SceneFlow Flying ==> KITTI Raw
  elif opt.environment == "sf_to_kitti":
    train_datasets = [
      StereoDataset("/home/milo/datasets/sceneflow_flying_things_3d", "SceneFlowFlying", "sceneflow_flying",
                    opt.height, opt.width, "train", scales=scale_list, load_disp_left=False, load_disp_right=False)
    ]
    novel_datasets = [
      StereoDataset("/home/milo/datasets/kitti_data_raw", "KittiRaw", "kitti_raw_campus_adapt",
                    opt.height, opt.width, "train", scales=scale_list, load_disp_left=False, load_disp_right=False),
      StereoDataset("/home/milo/datasets/kitti_data_raw", "KittiRaw", "kitti_raw_city_adapt",
                    opt.height, opt.width, "train", scales=scale_list, load_disp_left=False, load_disp_right=False),
      StereoDataset("/home/milo/datasets/kitti_data_raw", "KittiRaw", "kitti_raw_road_adapt",
                    opt.height, opt.width, "train", scales=scale_list, load_disp_left=False, load_disp_right=False),
      StereoDataset("/home/milo/datasets/kitti_data_raw", "KittiRaw", "kitti_raw_residential_adapt",
              opt.height, opt.width, "train", scales=scale_list, load_disp_left=False, load_disp_right=False)
    ]

  # SceneFlow Flying ==> Virtual KITTI Clone
  elif opt.environment == "sf_to_vk":
    train_datasets = [
      StereoDataset("/home/milo/datasets/sceneflow_flying_things_3d", "SceneFlowFlying", "sceneflow_flying",
                    opt.height, opt.width, "train", scales=scale_list, load_disp_left=False, load_disp_right=False)
    ]
    novel_datasets = [
      StereoDataset("/home/milo/datasets/virtual_kitti", "VirtualKitti", "virtual_kitti_clone",
                    opt.height, opt.width, "train", scales=scale_list, load_disp_left=False, load_disp_right=False)
    ]
  else:
    raise NotImplementedError()

  if opt.mode == "save_loss":
    train_loaders = [DataLoader(d, opt.batch_size, shuffle=True, pin_memory=True,
                                drop_last=False, num_workers=6) for d in train_datasets]
    novel_loaders = [DataLoader(d, opt.batch_size, shuffle=True, pin_memory=True,
                                drop_last=False, num_workers=6) for d in novel_datasets]
    save_loss(train_loaders, novel_loaders, path_to_output(reldir="vae_ood_{}".format(opt.environment)),
              opt, num_train=1000, num_novel=1000)

  elif opt.mode == "histogram":
    loss_train = torch.load(path_to_output(reldir="vae_ood_{}/train_loss.pt".format(opt.environment)))
    loss_novel = torch.load(path_to_output(reldir="vae_ood_{}/novel_loss.pt".format(opt.environment)))

    mu_train = torch.load(path_to_output(reldir="vae_ood_{}/train_mu.pt".format(opt.environment)))
    mu_novel = torch.load(path_to_output(reldir="vae_ood_{}/novel_mu.pt".format(opt.environment)))

    train_dist_mu = torch.load(path_to_output(reldir="vae_ood_{}/train_dist_mu.pt".format(opt.environment)))
    train_dist_sigma = torch.load(path_to_output(reldir="vae_ood_{}/train_dist_sigma.pt".format(opt.environment)))

    train_residual = (mu_train - train_dist_mu).unsqueeze(-1)
    novel_residual = (mu_novel - train_dist_mu).unsqueeze(-1)

    mdist_train = torch.sqrt(torch.matmul(train_residual.transpose(-2, -1), torch.matmul(train_dist_sigma.inverse(), train_residual)))
    mdist_novel = torch.sqrt(torch.matmul(novel_residual.transpose(-2, -1), torch.matmul(train_dist_sigma.inverse(), novel_residual)))

    loss_train = opt.alpha*loss_train + opt.beta*mdist_train.squeeze()
    loss_novel = opt.alpha*loss_novel + opt.beta*mdist_novel.squeeze()

    plot_histogram(
        loss_train, loss_novel, path_to_output(reldir="vae_ood_{}".format(opt.environment)),
        opt, percentile=opt.percentile, show=False, legend=(opt.environment == "vk_to_sf"))

  elif opt.mode == "pr":
    loss_train_path = path_to_output(reldir="vae_ood_{}/train_loss.pt".format(opt.environment))
    loss_novel_path = path_to_output(reldir="vae_ood_{}/novel_loss.pt".format(opt.environment))
    loss_train = torch.load(loss_train_path)
    loss_novel = torch.load(loss_novel_path)
    plot_precision_recall(
        loss_train, loss_novel, path_to_output(reldir="ood_{}".format(opt.environment)),
        opt, show=False)

  # NOTE: Need to run this script for each of the environments below in --pr mode first.
  # This will save the "precision.pt" and "recall.pt" data that is needed.
  elif opt.mode == "pr_multiple":
    output_folder = path_to_output(reldir="vae_ood_multiple")
    os.makedirs(output_folder, exist_ok=True)

    short_names = ["vk_to_sf", "sf_to_kitti", "vk_to_kitti", "vk_weather"]
    legend_names = ["VK → SF", "SF → KITTI", "VK → KITTI", "Sun → Fog + Rain"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    pr_dict = {}
    for i, (short_name, legend_name, color) in enumerate(zip(short_names, legend_names, colors)):
      input_folder = path_to_output(reldir="vae_ood_{}".format(short_name))
      precision = torch.load(os.path.join(input_folder, "precision.pt")).numpy()
      recall = torch.load(os.path.join(input_folder, "recall.pt")).numpy()
      pr_dict[legend_name] = (precision, recall, color)

    plot_precision_recall_multiple(pr_dict, output_folder, opt)

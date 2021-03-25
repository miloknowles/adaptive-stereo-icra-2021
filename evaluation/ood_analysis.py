# Copyright 2020 Massachusetts Institute of Technology
#
# @file ood.py
# @author Milo Knowles
# @date 2020-07-29 13:37:06 (Wed)

import os, argparse, math, random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
import pandas as pd
import scipy.stats as stats

from models.stereo_net import StereoNet, FeatureExtractorNetwork, DisparityRegression
from datasets.stereo_dataset import StereoDataset
from utils.feature_contrast import *
from utils.path_utils import *

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 20
plt.rcParams['pdf.fonttype'] = 42 # Solve Type 3 font problem.


torch.manual_seed(123)
random.seed(123)


def process_batch(feature_net, stereo_net, left, right, opt):
  with torch.no_grad():
    left_feat, right_feat = feature_net(left), feature_net(right)

    # Don't need to do refinement - all we care about is the cost volume.
    outputs = stereo_net(left, left_feat, right_feat, "l", output_cost_volume=True)
    return outputs


def save_data(train_loaders, novel_loaders, opt, num_train=1000, num_novel=1000,
              output_folder=path_to_output(reldir="ood")):
  """
  Save feature contrast scores (FCS) for training and novel images.
  """
  os.makedirs(output_folder, exist_ok=True)

  feature_net = FeatureExtractorNetwork(opt.stereonet_k).cuda()
  stereo_net = StereoNet(opt.stereonet_k, 1, opt.stereonet_input_scale).cuda()
  feature_net.load_state_dict(torch.load(os.path.join(
      opt.load_weights_folder, "feature_net.pth")), strict=True)
  stereo_net.load_state_dict(torch.load(os.path.join(
      opt.load_weights_folder, "stereo_net.pth")), strict=True)

  feature_net.eval()
  stereo_net.eval()

  train_fcs = torch.zeros(num_train)
  novel_fcs = torch.zeros(num_novel)

  with torch.no_grad():
    num_each_train = num_train // len(train_loaders)
    num_each_novel = num_novel // len(novel_loaders)
    print("Will use {} images from each training set".format(num_each_train))
    print("Will use {} images from each novel set".format(num_each_novel))

    for loader_type, loader_list in [("train", train_loaders), ("novel", novel_loaders)]:
      print("Processing {} datasets".format(loader_type))
      ctr = 0

      for loader in loader_list:
        for i, inputs in enumerate(loader):
          if loader_type == "train" and (i*opt.batch_size) >= num_each_train \
            or loader_type == "novel" and (i*opt.batch_size) >= num_each_novel:
            break

          for key in inputs:
            inputs[key] = inputs[key].cuda()

          outputs = process_batch(feature_net, stereo_net,
                                  inputs["color_l/{}".format(opt.stereonet_input_scale)],
                                  inputs["color_r/{}".format(opt.stereonet_input_scale)], opt)
          cost_volume = outputs["cost_volume_l/{}".format(opt.stereonet_input_scale + opt.stereonet_k)]
          fcs_avg = feature_contrast_mean(cost_volume).mean(dim=(-2, -1)).cpu()

          if loader_type == "train":
            num_remaining = min(len(fcs_avg), len(train_fcs) - ctr)
            train_fcs[ctr:ctr+num_remaining] = fcs_avg[:num_remaining]
          else:
            num_remaining = min(len(fcs_avg), len(novel_fcs) - ctr)
            novel_fcs[ctr:ctr+num_remaining] = fcs_avg[:num_remaining]

          ctr += len(fcs_avg)
          print("Finished {}/{} images".format(ctr, len(train_fcs)))

  # All FCSs should be > 0, otherwise we probably didn't compute/index correctly.
  assert((train_fcs > 0).all())
  assert((novel_fcs > 0).all())

  # print("Training FCS range  :", torch.argmin(train_fcs), train_fcs.min())
  # print("Novel FCS range     :", torch.argmax(novel_fcs), novel_fcs.max())

  torch.save(train_fcs, os.path.join(output_folder, "train_fcs.pt"))
  torch.save(novel_fcs, os.path.join(output_folder, "novel_fcs.pt"))
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


def plot_precision_recall(fcs_train, fcs_novel, output_folder, opt, show=False):
  """
  Make a precision recall plot using FCS scores for train and novel images.
  """
  plt.clf()

  cutoff_values = np.linspace(fcs_novel.min(), fcs_novel.max(), num=100)
  print("Cutoff value range: min={} max={}".format(cutoff_values.min(), cutoff_values.max()))
  precision = np.zeros(len(cutoff_values))
  recall = np.zeros(len(cutoff_values))
  for i, cutoff in enumerate(cutoff_values):
    pr, re = compute_precision_recall(fcs_train, fcs_novel, cutoff)
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
  matplotlib.rcParams['font.size'] = 18
  plt.figure(figsize=(8, 4))

  for label_name, (precision, recall, color) in pr.items():
    print("Plotting", label_name)
    recall_fixed, precision_fixed = strictly_decreasing_precision(precision, recall)
    plt.plot(recall_fixed, precision_fixed, color=color, label=label_name)

  plt.xlabel("recall")
  plt.ylabel("precision")
  plt.legend(fontsize="small")
  plt.savefig(os.path.join(output_folder, "pr_multiple.pdf"), bbox_inches="tight")
  print("Saved pr_multiple.pdf to", output_folder)


def plot_histogram(fcs_train, fcs_novel, output_folder, opt, percentile=0.01,
                   show=False, normal=True, legend=False):
  """
  Compare the distribution of FCSs for training and novel images.
  """
  plt.clf()

  bins = np.histogram(np.hstack((fcs_train, fcs_novel)), bins=40)[1]

  y1, x, _ = plt.hist(fcs_train, bins, facecolor="blue", density=True, alpha=0.5, label="train")
  y2, x, _ = plt.hist(fcs_novel, bins, facecolor="red", density=True, alpha=0.5, label="novel")
  plt.xlabel("feature contrast score")
  plt.ylabel("frequency")

  if normal:
    mu, sigma = fcs_train.mean(), math.sqrt(fcs_train.var())
    pct_x = stats.norm.ppf(percentile, loc=mu, scale=sigma)
    print("Plotting {}th percentile".format(percentile))
    print("{}th percentile occurs at FCS={}".format(percentile, pct_x))
    plt.vlines(pct_x, 0, max(y1.max(), y2.max()), colors="black", linestyles=(0, (5, 5)))
    plt.plot(bins, stats.norm.pdf(bins, mu, sigma), color="black", linestyle="solid")

  if legend:
    plt.legend(loc="upper left", fontsize="large")

  if show:
    plt.show()

  plt.savefig(os.path.join(output_folder, "histogram_{}.pdf".format(opt.environment)), bbox_inches="tight")
  print("Saved histogram_{}.pdf".format(opt.environment))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--height", type=int, default=320)
  parser.add_argument("--width", type=int, default=960)
  parser.add_argument("--load_weights_folder", default=None, type=str,
                      help="Path to load pretrained weights from")
  parser.add_argument("--stereonet_k", type=int, default=4, choices=[3, 4],
                      help="The cost volume downsampling factor")
  parser.add_argument("--stereonet_input_scale", type=int, default=0, help="Loss scale for StereoNet input images")
  parser.add_argument("--batch_size", type=int, default=32, help="Batch size for loading images")
  parser.add_argument("--save", action="store_true",
                      help="If set, saves FCS for training and novel data")
  parser.add_argument("--histogram", action="store_true", help="If set, display the FCS histograms")
  parser.add_argument("--percentile", type=float, default=0.05,
                      help="Visualize percentile as a vertical line on the histogram")
  parser.add_argument("--environment", type=str, default="sf_to_vk",
                      choices=["vk_to_sf", "sf_to_kitti", "vk_to_kitti", "vk_weather", "sf_to_vk", "sd_to_vk", "sd_to_kitti"])
  parser.add_argument("--pr", action="store_true", help="If set, compute precision recall curves")
  parser.add_argument("--pr_multiple", action="store_true", help="If set, compute overlaid precision recall curves")
  opt = parser.parse_args()

  assert(opt.percentile >= 0.01 and opt.percentile <= 0.99)
  print("------------------- ENVIRONMENT: {} --------------------".format(opt.environment))

  image_scales = [opt.stereonet_input_scale]

  if opt.save:
    # Virtual KITTI => SceneFlow Flying
    if opt.environment == "vk_to_sf":
      train_datasets = [
        StereoDataset("/home/milo/datasets/virtual_kitti", "VirtualKitti", "virtual_kitti_clone",
                      opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False)
      ]
      novel_datasets = [
        StereoDataset("/home/milo/datasets/sceneflow_flying_things_3d", "SceneFlowFlying", "sceneflow_flying",
                    opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False)
      ]

    # Virtual KITTI Clone => Virtual KITTI Fog/Rain
    elif opt.environment == "vk_weather":
      train_datasets = [
        StereoDataset("/home/milo/datasets/virtual_kitti", "VirtualKitti", "virtual_kitti_clone",
                      opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False)
      ]
      novel_datasets = [
        StereoDataset("/home/milo/datasets/virtual_kitti", "VirtualKitti", "virtual_kitti_fog",
                      opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False),
        StereoDataset("/home/milo/datasets/virtual_kitti", "VirtualKitti", "virtual_kitti_rain",
                      opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False),
      ]

    # Virtual KITTI => KITTI Raw
    elif opt.environment == "vk_to_kitti":
      train_datasets = [
        StereoDataset("/home/milo/datasets/virtual_kitti", "VirtualKitti", "virtual_kitti_clone",
                      opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False)
      ]
      novel_datasets = [
        StereoDataset("/home/milo/datasets/kitti_data_raw", "KittiRaw", "kitti_raw_campus_adapt",
                      opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False),
        StereoDataset("/home/milo/datasets/kitti_data_raw", "KittiRaw", "kitti_raw_city_adapt",
                      opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False),
        StereoDataset("/home/milo/datasets/kitti_data_raw", "KittiRaw", "kitti_raw_road_adapt",
                      opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False),
        StereoDataset("/home/milo/datasets/kitti_data_raw", "KittiRaw", "kitti_raw_residential_adapt",
                opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False)
      ]

    # SceneFlow Flying ==> KITTI Raw
    elif opt.environment == "sf_to_kitti":
      train_datasets = [
        StereoDataset("/home/milo/datasets/sceneflow_flying_things_3d", "SceneFlowFlying", "sceneflow_flying",
                      opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False)
      ]
      novel_datasets = [
        StereoDataset("/home/milo/datasets/kitti_data_raw", "KittiRaw", "kitti_raw_campus_adapt",
                      opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False),
        StereoDataset("/home/milo/datasets/kitti_data_raw", "KittiRaw", "kitti_raw_city_adapt",
                      opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False),
        StereoDataset("/home/milo/datasets/kitti_data_raw", "KittiRaw", "kitti_raw_road_adapt",
                      opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False),
        StereoDataset("/home/milo/datasets/kitti_data_raw", "KittiRaw", "kitti_raw_residential_adapt",
                opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False)
      ]

    # SceneFlow Flying ==> Virtual KITTI Clone
    elif opt.environment == "sf_to_vk":
      train_datasets = [
        StereoDataset("/home/milo/datasets/sceneflow_flying_things_3d", "SceneFlowFlying", "sceneflow_flying",
                      opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False)
      ]
      novel_datasets = [
        StereoDataset("/home/milo/datasets/virtual_kitti", "VirtualKitti", "virtual_kitti_clone",
                      opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False)
      ]
    elif opt.environment == "sd_to_vk":
      train_datasets = [
        StereoDataset("/home/milo/datasets/sceneflow_driving", "SceneFlowDriving", "sceneflow_driving",
                      opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False)
      ]
      novel_datasets = [
        StereoDataset("/home/milo/datasets/virtual_kitti", "VirtualKitti", "virtual_kitti_clone",
                      opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False)
      ]
    elif opt.environment == "sd_to_kitti":
      train_datasets = [
        StereoDataset("/home/milo/datasets/sceneflow_driving", "SceneFlowDriving", "sceneflow_driving",
                      opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False)
      ]
      novel_datasets = [
        StereoDataset("/home/milo/datasets/kitti_data_raw", "KittiRaw", "kitti_raw_campus_adapt",
                      opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False),
        StereoDataset("/home/milo/datasets/kitti_data_raw", "KittiRaw", "kitti_raw_city_adapt",
                      opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False),
        StereoDataset("/home/milo/datasets/kitti_data_raw", "KittiRaw", "kitti_raw_road_adapt",
                      opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False),
        StereoDataset("/home/milo/datasets/kitti_data_raw", "KittiRaw", "kitti_raw_residential_adapt",
                opt.height, opt.width, "train", scales=image_scales, load_disp_left=False, load_disp_right=False)
      ]
    else:
      raise NotImplementedError()

    train_loaders = [DataLoader(d, opt.batch_size, shuffle=True, pin_memory=True,
                                drop_last=False, num_workers=4) for d in train_datasets]
    novel_loaders = [DataLoader(d, opt.batch_size, shuffle=True, pin_memory=True,
                                drop_last=False, num_workers=4) for d in novel_datasets]
    save_data(train_loaders, novel_loaders, opt, num_train=1000, num_novel=1000,
              output_folder=path_to_output(reldir="ood_{}".format(opt.environment)))

  if opt.histogram:
    fcs_train_path = path_to_output(reldir="ood_{}/train_fcs.pt".format(opt.environment))
    fcs_novel_path = path_to_output(reldir="ood_{}/novel_fcs.pt".format(opt.environment))
    fcs_train = torch.load(fcs_train_path)
    fcs_novel = torch.load(fcs_novel_path)

    plot_histogram(
        fcs_train, fcs_novel, path_to_output(reldir="ood_{}".format(opt.environment)),
        opt, percentile=opt.percentile, show=False, legend=(opt.environment == "vk_to_kitti"))

  if opt.pr:
    fcs_train_path = path_to_output(reldir="ood_{}/train_fcs.pt".format(opt.environment))
    fcs_novel_path = path_to_output(reldir="ood_{}/novel_fcs.pt".format(opt.environment))
    fcs_train = torch.load(fcs_train_path)
    fcs_novel = torch.load(fcs_novel_path)
    plot_precision_recall(
        fcs_train, fcs_novel, path_to_output(reldir="ood_{}".format(opt.environment)),\
        opt, show=False)

  # NOTE: Need to run this script for each of the environments below in --pr mode first.
  # This will save the "precision.pt" and "recall.pt" data that is needed.
  if opt.pr_multiple:
    output_folder = path_to_output(reldir="ood_multiple")
    os.makedirs(output_folder, exist_ok=True)

    short_names = ["vk_to_sf", "sf_to_kitti", "vk_to_kitti", "vk_weather"]
    legend_names = ["VK → SF", "SF → KITTI", "VK → KITTI", "Sun → Fog + Rain"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    pr_dict = {}
    for i, (short_name, legend_name, color) in enumerate(zip(short_names, legend_names, colors)):
      input_folder = path_to_output(reldir="ood_{}".format(short_name))
      precision = torch.load(os.path.join(input_folder, "precision.pt")).numpy()
      recall = torch.load(os.path.join(input_folder, "recall.pt")).numpy()
      pr_dict[legend_name] = (precision, recall, color)

    plot_precision_recall_multiple(pr_dict, output_folder, opt)

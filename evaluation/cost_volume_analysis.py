# Copyright 2020 Massachusetts Institute of Technology
#
# @file cost_volume_analysis.py
# @author Milo Knowles
# @date 2020-10-05 12:40:12 (Mon)

import os, argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
import pandas as pd

from models.stereo_net import StereoNet, FeatureExtractorNetwork
from datasets.stereo_dataset import StereoDataset
from utils.feature_contrast import *
from utils.path_utils import *

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 20
plt.rcParams['pdf.fonttype'] = 42 # Solve Type 3 font problem.


def process_batch(feature_net, stereo_net, left, right, opt):
  with torch.no_grad():
    left_feat, right_feat = feature_net(left), feature_net(right)

    # Don't need to do refinement - all we care about is the cost volume.
    outputs = stereo_net(left, left_feat, right_feat, "l", output_cost_volume=True)
    return outputs


def save_cost_volumes(loader, output_folder, opt):
  """
  Save the cost volumes for the first num_images pairs from a dataset.
  """
  os.makedirs(output_folder, exist_ok=True)

  feature_net = FeatureExtractorNetwork(opt.stereonet_k).cuda()
  stereo_net = StereoNet(opt.stereonet_k, 1, 0).cuda()
  feature_net.load_state_dict(torch.load(os.path.join(
      opt.load_weights_folder, "feature_net.pth")), strict=True)
  stereo_net.load_state_dict(torch.load(os.path.join(
      opt.load_weights_folder, "stereo_net.pth")), strict=True)

  feature_net.eval()
  stereo_net.eval()

  with torch.no_grad():
    for i, inputs in enumerate(loader):
      if i >= opt.num_images:
        break

      for key in inputs:
        inputs[key] = inputs[key].cuda().unsqueeze(0)

      outputs = process_batch(feature_net, stereo_net, inputs["color_l/0"], inputs["color_r/0"], opt)
      cost_volume = outputs["cost_volume_l/{}".format(opt.stereonet_k)]

      cv_output_file = os.path.join(output_folder, "{}_cost_volume.pt".format(i))
      torch.save(cost_volume.cpu(), cv_output_file)

      gt_output_file = os.path.join(output_folder, "{}_gt.pt".format(i))
      torch.save(inputs["gt_disp_l/{}".format(opt.stereonet_k)].cpu(), gt_output_file)

      print("Finished {}/{} images".format(i+1, opt.num_images))


def visualize_cost_volumes(output_folder, argmin_or_argmax, line_color, legend, opt, ylim=None, ylabel=True):
  plt.clf()
  plt.figure(figsize=(4, 3))
  matplotlib.rcParams['font.size'] = 16

  for i in range(opt.num_images):
    print("Loading cost volume {}/{}".format(i+1, opt.num_images))
    cv_output_file = os.path.join(output_folder, "{}_cost_volume.pt".format(i))
    gt_output_file = os.path.join(output_folder, "{}_gt.pt".format(i))

    cost_volume = torch.load(cv_output_file)
    gt_disp = torch.load(gt_output_file)[0,0]
    fcs = feature_contrast_mean(cost_volume)

    fcs_idx = argmin_or_argmax(fcs.flatten(), None, keepdim=False)
    row, col = fcs_idx // fcs.shape[-1], fcs_idx % fcs.shape[-1]

    cost_volume_slice = cost_volume[0,:,row,col]
    cost_volume_slice_sorted = torch.sort(cost_volume_slice, descending=True)[0]
    max_value = cost_volume_slice_sorted[0]
    mean_value = cost_volume_slice_sorted[2:].mean()

    gt_disp_value = gt_disp[row, col].item()

    plt.clf()
    plt.plot(np.arange(len(cost_volume_slice)), cost_volume_slice, color=line_color, linestyle="-")
    plt.xticks(np.arange(0, len(cost_volume_slice), step=2))
    plt.xlabel("disparity")

    if ylabel:
      plt.ylabel("feature similarity score $\mathcal{C}(u, v)$")

    if ylim is not None:
      plt.ylim(ylim)

    # Plot the max similarity score.
    plt.hlines(max_value, xmin=0, xmax=len(cost_volume_slice)-2.5, linestyles="dashed", colors="gray", label="max")

    # Plot the mean similarity score.
    plt.hlines(mean_value, xmin=0, xmax=len(cost_volume_slice)-2.5, linestyles="dashed", colors="gray", label="mean")

    # Plot the groundtruth disparity value.
    if ylim is not None:
      plt.vlines(gt_disp_value, ymin=ylim[0], ymax=ylim[1], linestyles="dashed", colors="black", label="true disparity")
      plt.text(gt_disp_value, (ylim[0] + ylim[1]) / 2, "true disparity", rotation=90, va="center", ha="right")
    else:
      plt.vlines(gt_disp_value, ymin=cost_volume_slice.min(), ymax=cost_volume_slice.max(),
                 linestyles="dashed", colors="black", label="true disparity")

    plt.text(len(cost_volume_slice) - 2.5, max_value, "max", rotation=0, va="center")
    plt.text(len(cost_volume_slice) - 2.5, mean_value, "mean", rotation=0, va="center")

    # plt.grid(True, which='major')
    plt.savefig(os.path.join(output_folder, "{}_cost_volume_slice.pdf".format(i)), bbox_inches="tight")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--load_weights_folder", default=None, type=str, help="Path to load pretrained weights from")
  parser.add_argument("--stereonet_k", type=int, default=4, choices=[3, 4], help="The cost volume downsampling factor")
  parser.add_argument("--save", action="store_true", help="If set, saves cost volumes for training and novel data")
  parser.add_argument("--debug", action="store_true", help="Used for debugging the cost volume for specific images")
  parser.add_argument("--visualize", action="store_true", help="If set, compute precision recall curves")
  parser.add_argument("--num_images", type=int, default=10, help="The number of images to analyze")
  opt = parser.parse_args()

  if opt.save:
    train_dataset = StereoDataset("/home/milo/datasets/sceneflow_flying_things_3d", "SceneFlowFlying",
                                  "sceneflow_flying", 320, 960, "train", scales=[0, opt.stereonet_k],
                                  load_disp_left=True, load_disp_right=False)
    novel_dataset = StereoDataset("/home/milo/datasets/virtual_kitti", "VirtualKitti", "virtual_kitti_01_adapt",
                      320, 960, "train", scales=[0, opt.stereonet_k], load_disp_left=True, load_disp_right=False)
    save_cost_volumes(train_dataset, path_to_output(reldir="cost_volume_analysis/train"), opt)
    save_cost_volumes(novel_dataset, path_to_output(reldir="cost_volume_analysis/novel"), opt)

  elif opt.visualize:
    print("Visualizing training cost volumes")
    visualize_cost_volumes(path_to_output(reldir="cost_volume_analysis/train"), torch.argmax, "tab:blue", False, opt, ylim=[-22, 12])

    print("Visualizing novel cost volumes")
    visualize_cost_volumes(path_to_output(reldir="cost_volume_analysis/novel"), torch.argmin, "tab:red", True, opt, ylim=[-22, 12], ylabel=False)

  else:
    raise NotImplementedError()

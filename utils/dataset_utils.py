# Copyright 2020 Massachusetts Institute of Technology
#
# @file dataset_utils.py
# @author Milo Knowles
# @date 2020-05-19 17:55:52 (Tue)

import os
import imageio
import numpy as np
import cv2 as cv
import torch

from .io import read_pfm_tensor


# Source: https://github.com/nianticlabs/monodepth2
def read_lines(filename):
  """
  Read all the lines in a text file and return as a list.
  """
  with open(filename, 'r') as f:
    lines = f.read().splitlines()
  return lines


def flip_stereo_pair(l, r):
  tmp = l
  l = torch.flip(r, dims=(-1,))
  r = torch.flip(tmp, dims=(-1,))
  return l, r


def load_disp_sceneflow(path):
  return read_pfm_tensor(path).unsqueeze(0)


def load_disp_kitti_stereo(path):
  return torch.Tensor(imageio.imread(path).astype(np.float32) / 256.0).unsqueeze(0)


def load_disp_kitti_raw(path):
  return (torch.from_numpy(np.load(path).astype(np.float32)) / 128.0).unsqueeze(0)


def load_disp_vkitti(path):
  """
  Virtual KITTI contains depth instead of disparity, so need to convert.

  https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
  """
  baseline = 0.532725 # meters.
  focal_length = 725.0087 # pixels.
  depth = 0.01 * cv.imread(path, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
  return torch.from_numpy(baseline * focal_length / depth).unsqueeze(0)


def get_disp_loader(dataset_name):
  return {"SceneFlowDriving": load_disp_sceneflow,
          "SceneFlowFlying": load_disp_sceneflow,
          "SceneFlowMonkaa": load_disp_sceneflow,
          "KittiStereo2015": load_disp_kitti_stereo,
          "KittiStereo2012": load_disp_kitti_stereo,
          "KittiRaw": load_disp_kitti_raw,
          "VirtualKitti": load_disp_vkitti}[dataset_name]

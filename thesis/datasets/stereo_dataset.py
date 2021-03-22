# Copyright 2020 Massachusetts Institute of Technology
#
# @file stereo_dataset.py
# @author Milo Knowles
# @date 2020-07-07 11:52:15 (Tue)

import random, os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as tf

from PIL import Image

from utils.dataset_utils import read_lines, get_disp_loader, flip_stereo_pair
from utils.io import read_pfm_tensor


class StereoDataset(Dataset):
  def __init__(self, dataset_path, dataset_name, split, height, width, subsplit,
               scales=[0], do_hflip=False, random_crop=False, load_disp_left=True,
               load_disp_right=True, do_vflip=False):
    super(StereoDataset, self).__init__()

    self.dataset_path = dataset_path
    self.height = height
    self.width = width
    self.do_hflip = do_hflip
    self.scales = scales
    self.random_crop = random_crop
    self.load_disp_left = load_disp_left
    self.load_disp_right = load_disp_right
    self.do_vflip = do_vflip

    splits_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../splits/"))
    self.dataset = dataset_name
    self.lines = read_lines(os.path.join(splits_path, split, "{}_lines.txt".format(subsplit)))
    self.to_tensor = transforms.ToTensor()

    self.load_disp_fn = get_disp_loader(dataset_name)

  def transform(self, rgb_l, rgb_r, disp_l, disp_r):
    """
    Optionally applies a horizontal flip with 50% probability, and then performs a random or
    center crop to achieve the desired resolution.
    """
    # Don't do height resizing for now.
    assert(self.height <= rgb_l.shape[-2])

    # If the desired width is larger than the input image, resize first.
    if self.width > rgb_l.shape[-1]:
      width_scale_factor = self.width / rgb_l.shape[-1]
      resize_height = int(width_scale_factor * rgb_l.shape[-2])
      rgb_l = F.interpolate(rgb_l.unsqueeze(0), size=(resize_height, self.width), mode="bilinear", align_corners=False).squeeze(0)
      rgb_r = F.interpolate(rgb_r.unsqueeze(0), size=(resize_height, self.width), mode="bilinear", align_corners=False).squeeze(0)

      if disp_l is not None:
        disp_l = width_scale_factor * F.interpolate(disp_l.unsqueeze(0), size=(resize_height, self.width), mode="bilinear", align_corners=False).squeeze(0)
      if disp_r is not None:
        disp_r = width_scale_factor * F.interpolate(disp_r.unsqueeze(0), size=(resize_height, self.width), mode="bilinear", align_corners=False).squeeze(0)

    if self.random_crop:
      i, j, h, w = transforms.RandomCrop.get_params(rgb_l, output_size=(self.height, self.width))
    else:
      i, j = (rgb_l.shape[-2] - self.height) // 2, (rgb_l.shape[-1] - self.width) // 2
      h, w = self.height, self.width

    assert(self.height <= rgb_l.shape[-2])

    if self.do_hflip and random.random() < 0.5:
      rgb_l, rgb_r = flip_stereo_pair(rgb_l, rgb_r)

      if disp_l is not None and disp_r is not None:
        disp_l, disp_r = flip_stereo_pair(disp_l, disp_r)

    if self.do_vflip:
      rgb_l = torch.flip(rgb_l, (-2,))
      rgb_r = torch.flip(rgb_r, (-2,))

      if disp_l is not None:
        disp_l = torch.flip(disp_l, (-2,))
      if disp_r is not None:
        disp_r = torch.flip(disp_r, (-2,))

    return rgb_l[:,i:i+h,j:j+w], rgb_r[:,i:i+h,j:j+w], \
           disp_l[:,i:i+h,j:j+w] if disp_l is not None else None, \
           disp_r[:,i:i+h,j:j+w] if disp_r is not None else None

  def __getitem__(self, index):
    """
    Crop and flip all of the inputs in the same way.
    """
    outputs = {}
    rgb_l_path, rgb_r_path, disp_l_path, disp_r_path = [os.path.join(self.dataset_path, p) for p in self.lines[index].split(" ")]

    rgb_l = self.to_tensor(Image.open(rgb_l_path))
    rgb_r = self.to_tensor(Image.open(rgb_r_path))

    disp_l = self.load_disp_fn(disp_l_path) if self.load_disp_left else None
    disp_r = self.load_disp_fn(disp_r_path) if self.load_disp_right else None

    rgb_l, rgb_r, disp_l, disp_r = self.transform(rgb_l, rgb_r, disp_l, disp_r)

    for s in self.scales:
      if s != 0:
        size_for_scale = (self.height // 2**s, self.width // 2**s)
        outputs["color_l/{}".format(s)] = F.interpolate(rgb_l.unsqueeze(0), size=size_for_scale, mode="bilinear", align_corners=False).squeeze(0)
        outputs["color_r/{}".format(s)] = F.interpolate(rgb_r.unsqueeze(0), size=size_for_scale, mode="bilinear", align_corners=False).squeeze(0)

        if self.load_disp_left:
          outputs["gt_disp_l/{}".format(s)] = F.interpolate(disp_l.unsqueeze(0), size=size_for_scale,
                                              mode="bilinear", align_corners=False).squeeze(0) / 2**s
        if self.load_disp_right:
          outputs["gt_disp_r/{}".format(s)] = F.interpolate(disp_r.unsqueeze(0), size=size_for_scale,
                                              mode="bilinear", align_corners=False).squeeze(0) / 2**s

    outputs["color_l/0"] = rgb_l
    outputs["color_r/0"] = rgb_r

    if self.load_disp_left:
      outputs["gt_disp_l/0"] = disp_l
    if self.load_disp_right:
      outputs["gt_disp_r/0"] = disp_r

    return outputs

  def __len__(self):
    return len(self.lines)

  def get_baseline_meters(self):
    """
    Returns the stereo baseline in meters.
    """
    return {"KittiStereo2012": 0.54,
            "KittiStereo2015": 0.54,
            "SceneFlowFlying": 1.0,
            "SceneFlowMonkaa": 1.0,
            "SceneFlowDriving": 1.0,
            "VirtualKitti": 0.532725}[self.dataset]

  def get_intrinsics_normalized(self):
    """
    Returns the normalized intrinsic matrix.
    """
    if self.dataset in ["KittiStereo2012", "KittiStereo2015", "KittiRaw"]:
      return torch.Tensor([
        [0.5885,  0.0,      0.4972],
        [0.0,     1.9501,   0.4972],
        [0.0,     0.0,      1.0]
      ])
    elif "SceneFlow" in self.dataset:
      return torch.Tensor([
        [1.09375, 0.0,      0.5],
        [0,       1.94444,  0.5],
        [0.0,     0.0,      1.0]
      ])
    elif self.dataset == "VirtualKitti":
      raise NotImplementedError()

    elif self.dataset == "Synthia-SF":
      raise NotImplementedError()

    else:
      raise NotImplementedError()

  def get_intrinsics(self, height, width):
    K = self.get_intrinsics_normalized().clone()
    K[0] *= width
    K[1] *= height
    return K

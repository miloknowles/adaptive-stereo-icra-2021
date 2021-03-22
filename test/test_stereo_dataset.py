# Copyright 2020 Massachusetts Institute of Technology
#
# @file test_stereo_dataset.py
# @author Milo Knowles
# @date 2020-07-07 22:56:02 (Tue)

import random
import unittest
import torch

from datasets.stereo_dataset import StereoDataset


def test_sparse_indices(d, n):
  indices_to_test = torch.linspace(0, len(d) - 1, steps=n).int()
  for i in indices_to_test:
    inputs = d[i.item()]


class StereoDatasetTest(unittest.TestCase):
  def test_sceneflow_driving(self):
    d = StereoDataset("/home/milo/datasets/sceneflow_driving", "SceneFlowDriving", "sceneflow_driving",
                      540, 960, "train", scales=[0], do_hflip=False, random_crop=False)
    self.assertEqual(len(d), 1540)
    test_sparse_indices(d, 10)

    d = StereoDataset("/home/milo/datasets/sceneflow_driving", "SceneFlowDriving", "sceneflow_driving",
                      540, 960, "val", scales=[0], do_hflip=False, random_crop=False)
    self.assertEqual(len(d), 330)
    test_sparse_indices(d, 10)

    d = StereoDataset("/home/milo/datasets/sceneflow_driving", "SceneFlowDriving", "sceneflow_driving",
                      540, 960, "test", scales=[0], do_hflip=False, random_crop=False)
    self.assertEqual(len(d), 330)
    test_sparse_indices(d, 10)

  def test_sceneflow_flying(self):
    d = StereoDataset("/home/milo/datasets/sceneflow_flying_things_3d", "SceneFlowFlying",
                      "sceneflow_flying", 540, 960, "train", scales=[0], do_hflip=False, random_crop=False)
    self.assertEqual(len(d), 19031)
    test_sparse_indices(d, 10)

    d = StereoDataset("/home/milo/datasets/sceneflow_flying_things_3d", "SceneFlowFlying",
                      "sceneflow_flying", 540, 960, "val", scales=[0], do_hflip=False, random_crop=False)
    self.assertEqual(len(d), 3359)
    test_sparse_indices(d, 10)

    d = StereoDataset("/home/milo/datasets/sceneflow_flying_things_3d", "SceneFlowFlying",
                      "sceneflow_flying", 540, 960, "test", scales=[0], do_hflip=False, random_crop=False)
    self.assertEqual(len(d), 4370)
    test_sparse_indices(d, 10)

  def test_sceneflow_monkaa(self):
    pass

  def test_kitti_2012(self):
    d = StereoDataset("/home/milo/datasets/kitti_stereo_2012", "KittiStereo2012", "kitti_stereo_2012",
                      320, 960, "train", scales=[0], do_hflip=False, random_crop=False, load_disp_right=False)
    self.assertEqual(len(d), 194)
    test_sparse_indices(d, 10)

    d = StereoDataset("/home/milo/datasets/kitti_stereo_2012", "KittiStereo2012", "kitti_stereo_2012",
                      320, 960, "val", scales=[0], do_hflip=False, random_crop=False, load_disp_right=False)
    self.assertEqual(len(d), 194)
    test_sparse_indices(d, 10)

    d = StereoDataset("/home/milo/datasets/kitti_stereo_2012", "KittiStereo2012", "kitti_stereo_2012",
                      320, 960, "test", scales=[0], do_hflip=False, random_crop=False,
                      load_disp_left=False, load_disp_right=False)
    self.assertEqual(len(d), 194)
    test_sparse_indices(d, 10)


  def test_kitti_2015(self):
    d = StereoDataset("/home/milo/datasets/kitti_stereo_2015", "KittiStereo2015", "kitti_stereo_2015",
                      320, 960, "train", scales=[0], do_hflip=False, random_crop=False)
    self.assertEqual(len(d), 200)
    test_sparse_indices(d, 10)

    d = StereoDataset("/home/milo/datasets/kitti_stereo_2015", "KittiStereo2015", "kitti_stereo_2015",
                      320, 960, "val", scales=[0], do_hflip=False, random_crop=False)
    self.assertEqual(len(d), 200)
    test_sparse_indices(d, 10)

    d = StereoDataset("/home/milo/datasets/kitti_stereo_2015", "KittiStereo2015", "kitti_stereo_2015",
                      320, 960, "test", scales=[0], do_hflip=False, random_crop=False,
                      load_disp_left=False, load_disp_right=False)
    self.assertEqual(len(d), 200)
    test_sparse_indices(d, 10)

  def test_kitti_raw(self):
    pass

  def test_virtual_kitti(self):
    d = StereoDataset("/home/milo/datasets/virtual_kitti", "VirtualKitti", "virtual_kitti_sim2real",
                      320, 960, "train", scales=[0], do_hflip=False, random_crop=False)
    self.assertEqual(len(d), 21260)
    test_sparse_indices(d, 10)

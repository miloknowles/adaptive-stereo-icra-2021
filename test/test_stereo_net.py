# Copyright 2020 Massachusetts Institute of Technology
#
# @file test_stereo_net.py
# @author Milo Knowles
# @date 2020-10-13 20:35:48 (Tue)

import unittest, time
import torch

from models.stereo_net import StereoNet, FeatureExtractorNetwork


class StereoNetTest(unittest.TestCase):
  def setUp(self):
    torch.backends.cudnn.benchmark = True
    self.height = 320
    self.width = 960
    self.images = {
      0: torch.zeros(1, 3, self.height, self.width).cuda(),
      1: torch.zeros(1, 3, self.height // 2, self.width // 2).cuda()
    }
    self.num_forward = 100

  def inference_timing(self, stereonet_k, input_scale):
    """
    Test inference time for StereoNet in different configurations.
    """
    feature_net = FeatureExtractorNetwork(stereonet_k).cuda()
    stereo_net = StereoNet(stereonet_k, 1, input_scale, maxdisp=192).cuda()

    t0 = time.time()
    with torch.no_grad():
      for _ in range(self.num_forward):
        fl, fr = feature_net(self.images[input_scale]), feature_net(self.images[input_scale])
        outputs = stereo_net(self.images[input_scale], fl, fr, "l", output_cost_volume=False)

    elap = time.time() - t0
    print("{}X_L{}: {} sec/im".format(2**stereonet_k, input_scale, elap / self.num_forward))

  def test_timing_16X_L0(self):
    """
    16X downsampling for cost volume, input image at R0 (full resolution).
    """
    self.inference_timing(4, 0)

  def test_timing_8X_L1(self):
    """
    8X downsampling for cost volume, input image at R1 (1/2 resolution).
    """
    self.inference_timing(3, 1)


if __name__ == "__main__":
  unittest.main()

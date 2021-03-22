# Copyright 2020 Massachusetts Institute of Technology
#
# @file test_stereo_net.py
# @author Milo Knowles
# @date 2020-10-13 20:35:48 (Tue)

import unittest, time
import torch

from models.stereo_net import StereoNet, FeatureExtractorNetwork
from utils.feature_contrast import feature_contrast_mean


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

  def inference_timing(self, stereonet_k, input_scale, compute_fcs=False):
    """
    Test inference time for StereoNet in different configurations.
    """
    torch.cuda.synchronize()

    feature_net = FeatureExtractorNetwork(stereonet_k).cuda()
    stereo_net = StereoNet(stereonet_k, 1, input_scale, maxdisp=192).cuda()

    t0 = time.time()
    with torch.no_grad():
      for _ in range(self.num_forward):
        fl, fr = feature_net(self.images[input_scale]), feature_net(self.images[input_scale])
        outputs = stereo_net(self.images[input_scale], fl, fr, "l", output_cost_volume=compute_fcs)

        if compute_fcs:
          feature_contrast_mean(outputs["cost_volume_l/{}".format(input_scale + stereonet_k)])

    elap = time.time() - t0
    print("{}X_L{}: {:.05f} sec/im".format(2**stereonet_k, input_scale, elap / self.num_forward))

  def test_timing_16X_L0(self):
    """
    16X downsampling for cost volume, input image at R0 (full resolution).
    NOTE: Need to perform several "burn in" forward passes before inference time converges.
    """
    self.inference_timing(4, 0)
    self.inference_timing(4, 0)
    self.inference_timing(4, 0)

  def test_timing_8X_L1(self):
    """
    8X downsampling for cost volume, input image at R1 (1/2 resolution).
    NOTE: Need to perform several "burn in" forward passes before inference time converges.
    """
    self.inference_timing(3, 1)
    self.inference_timing(3, 1)
    self.inference_timing(3, 1)

  def test_timing_16X_L0_FCS(self):
    """
    Compare the time to compute FCS with network inference.
    NOTE: Need to perform several "burn in" forward passes before inference time converges.
    """
    print("\n======================== INFERENCE W/ FEATURE CONTRAST SCORE ========================")
    self.inference_timing(4, 0, True)
    self.inference_timing(4, 0, True)
    self.inference_timing(4, 0, True)


if __name__ == "__main__":
  unittest.main()

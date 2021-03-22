# Copyright 2020 Massachusetts Institute of Technology
#
# @file test_autoencoder.py
# @author Milo Knowles
# @date 2020-10-09 14:55:40 (Fri)

import unittest
import torch

from models.autoencoder import ConvolutionalEncoder, ConvolutionalDecoder


class DecoderTest(unittest.TestCase):
  def test_forward_8X(self):
    b, c, h, w = 8, 32, 23, 60
    feature_map = torch.zeros(b, c, h, w).cuda()

    # 3 upsample stages should give an output resolution of 184x480
    decoder = ConvolutionalDecoder(c, 3, 3).cuda()

    im_predicted = decoder(feature_map)
    self.assertEqual(im_predicted.shape, (b, 3, 184, 480))

  def test_forward_4X(self):
    b, c, h, w = 8, 32, 23, 60
    feature_map = torch.zeros(b, c, h, w).cuda()

    # 3 upsample stages should give an output resolution of 184x480
    decoder = ConvolutionalDecoder(c, 3, 2).cuda()

    im_predicted = decoder(feature_map)
    self.assertEqual(im_predicted.shape, (b, 3, 92, 240))


class EncoderTest(unittest.TestCase):
  def test_forward_8X(self):
    im = torch.zeros(8, 3, 368, 960).cuda()
    encoder = ConvolutionalEncoder(3, 32, 3).cuda()
    f = encoder(im)
    self.assertEqual(f.shape, (8, 32, 46, 120))

  def test_forward_16X(self):
    im = torch.zeros(8, 3, 368, 960).cuda()
    encoder = ConvolutionalEncoder(3, 32, 4).cuda()
    f = encoder(im)
    self.assertEqual(f.shape, (8, 32, 23, 60))


if __name__ == "__main__":
  unittest.main()

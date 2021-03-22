# Copyright 2020 Massachusetts Institute of Technology
#
# @file test_entropy.py
# @author Milo Knowles
# @date 2020-10-07 10:49:39 (Wed)

import unittest
import torch

from utils.entropy import *


class EntropyTest(unittest.TestCase):
  def test_zero_bits_01(self):
    img_zeros = torch.zeros(1, 30, 40).cuda()
    h = grayscale_shannon_entropy(img_zeros)
    self.assertEqual(h, 0)

  def test_zero_bits_02(self):
    img_ones = torch.ones(1, 30, 40).cuda()
    h = grayscale_shannon_entropy(img_ones)
    self.assertEqual(h, 0)

  def test_one_bit(self):
    img = torch.zeros(1, 30, 40).cuda()
    img[:,:,:20] = 0.123
    h = grayscale_shannon_entropy(img)
    self.assertAlmostEqual(h.item(), 1.0)

  def test_two_bits(self):
    img = torch.zeros(1, 30, 40).cuda()
    img[:,:,:10] = 0.123
    img[:,:,10:20] = 0.456
    img[:,:,20:30] = 0.789
    h = grayscale_shannon_entropy(img)
    self.assertAlmostEqual(h.item(), 2.0)

if __name__ == "__main__":
  unittest.main()

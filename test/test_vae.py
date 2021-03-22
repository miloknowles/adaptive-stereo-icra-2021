# Copyright 2020 Massachusetts Institute of Technology
#
# @file test_vae.py
# @author Milo Knowles
# @date 2020-10-12 09:38:58 (Mon)

import unittest
import torch

from models.vae import VAE


class VAETest(unittest.TestCase):
  def test_forward(self):
    """
    - Encoder downsamples by a factor of 16 with 4x4 kernels, but doesn't use padding.
    - 3x128x128 encoded to 256x6x6 = 9,216 when flattened.
    """
    vae = VAE(image_channels=3, z_dim=32, input_height=80, input_width=240).cuda()

    input_im = torch.zeros(4, 3, 80, 240).cuda()
    output_im, mu, logvar = vae(input_im)

    self.assertEqual(output_im.shape, input_im.shape)
    print(mu.shape, logvar.shape)


if __name__ == "__main__":
  unittest.main()

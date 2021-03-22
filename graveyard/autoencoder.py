# Copyright 2020 Massachusetts Institute of Technology
#
# @file autoencoder.py
# @author Milo Knowles
# @date 2020-10-09 14:29:16 (Fri)

import torch
import torch.nn as nn


class ConvolutionalEncoder(nn.Module):
  def __init__(self, input_channels, output_channels, downsample_stages):
    """
    input_channels (int) : The inpt image channels (probably 3 for RGB).
    output_channels (int) : The latent feature dimension (32 for StereoNet).
    upsample_stages (int) : The number of times to downsample and halve the spatial resolution.
    """
    super(ConvolutionalEncoder, self).__init__()

    self.encoder = nn.ModuleList()

    # 8, 12, 16, 16, ... 16 conv channels
    # self.conv_channels = [input_channels] + [min(32, 4*2**(k+1)) for k in range(downsample_stages)]
    self.conv_channels = [input_channels, 8, 12, 16, 16, 16]

    for k in range(downsample_stages):
      self.encoder.append(
        nn.Conv2d(self.conv_channels[k], self.conv_channels[k+1], 5, stride=2, padding=2)
      )
      self.encoder.append(nn.ReLU())

      self.encoder.append(
        nn.Conv2d(self.conv_channels[k+1], self.conv_channels[k+1], 5, stride=1, padding=2)
      )
      self.encoder.append(nn.ReLU())

    self.encoder.append(
      nn.Conv2d(self.conv_channels[k+1], output_channels, 3, stride=1, padding=1)
    )
    self.encoder = nn.Sequential(*self.encoder)

    # Set bias weights to zero.
    def weights_init(m):
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.zeros_(m.bias)
    self.decoder.apply(weights_init)

  def forward(self, x):
    """
    x (torch.Tensor) : An image with shape (b, in_channels, h, w).
    """
    return self.encoder(x)


class ConvolutionalDecoder(nn.Module):
  def __init__(self, input_channels, output_channels, upsample_stages, additional_downsample=False):
    """
    input_channels (int) : The latent feature dimension (32 for StereoNet).
    output_channels (int) : The number of output image channels (probably 3 for RGB).
    upsample_stages (int) : The number of times to upsample and double the spatial resolution.
    """
    super(ConvolutionalDecoder, self).__init__()

    self.decoder = nn.ModuleList()

    self.conv_channels = [input_channels] + [16 for _ in range(upsample_stages)]

    # if additional_downsample:
    #   self.encoder= nn.ModuleList()
    #   self.encoder.append(
    #     nn.Conv2d(input_channels, input_channels, 3, stride=2, padding=1)
    #   )
    #   self.encoder.append(nn.ReLU())
    #   self.encoder.append(
    #     nn.Conv2d(input_channels, input_channels, 3, stride=1, padding=1)
    #   )

    for k in range(upsample_stages):
      # Upsample by a factor of 2.
      self.decoder.append(
          nn.ConvTranspose2d(self.conv_channels[k], self.conv_channels[k + 1], 6, stride=2, padding=2)
      )
      self.decoder.append(nn.ReLU())

      # Give the network a chance to do some other learning...
      self.decoder.append(
          nn.ConvTranspose2d(self.conv_channels[k+1], self.conv_channels[k+1], 5, stride=1, padding=2)
      )
      self.decoder.append(nn.ReLU())

    # Map to correct # of output channels (probably 3 for RGB).
    self.decoder.append(
      nn.Conv2d(self.conv_channels[upsample_stages], output_channels, 5, stride=1, padding=2)
    )

    # Limit output to be between 0 and 1 (valid intensity values).
    self.decoder.append(nn.Sigmoid())

    self.decoder = nn.Sequential(*self.decoder)

    # Set bias weights to zero.
    def weights_init(m):
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.zeros_(m.bias)
    self.decoder.apply(weights_init)

  def forward(self, x):
    """
    x (torch.Tensor) : A feature map with shape (b, in_channels, h, w).
    """
    return self.decoder(x)

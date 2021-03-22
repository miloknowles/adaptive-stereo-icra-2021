# Adapted from: https://github.com/meteorshowers/StereoNet-ActiveStereoNet/blob/e2abce4e855767afd23f066626920cfd4d67799b/models/StereoNet_single.py

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
  return nn.Sequential(
    nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=dilation if dilation > 1 else pad,
        dilation=dilation),
    nn.BatchNorm2d(out_channels)
  )


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
  return nn.Sequential(
    nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        padding=pad,
        stride=stride),
    nn.BatchNorm3d(out_channels)
  )


class BasicBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride, downsample, pad, dilation):
    super(BasicBlock, self).__init__()

    self.conv1 = nn.Sequential(
      convbn(in_channels, out_channels, 3, stride, pad, dilation),
      nn.LeakyReLU(negative_slope=0.2, inplace=True))
    self.conv2 = convbn(out_channels, out_channels, 3, 1, pad, dilation)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    out = self.conv1(x)

    if self.downsample is not None:
      x = self.downsample(x)

    out = x + out
    return out


class FeatureExtractorNetwork(nn.Module):
  def __init__(self, k):
    super(FeatureExtractorNetwork, self).__init__()

    self.k = k
    self.downsample = nn.ModuleList()
    in_channels = 3
    out_channels = 32
    for _ in range(k):
      self.downsample.append(
          nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=5,
            stride=2,
            padding=2)
          )
      in_channels = out_channels
      out_channels = 32
    self.residual_blocks = nn.ModuleList()
    for _ in range(6):
      self.residual_blocks.append(
        BasicBlock(32, 32, stride=1, downsample=None, pad=1, dilation=1))
    self.conv_alone = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

  def forward(self, rgb_img):
    output = rgb_img
    for i in range(self.k):
      output = self.downsample[i](output)
    for block in self.residual_blocks:
      output = block(output)
    return self.conv_alone(output)


class EdgeAwareRefinement(nn.Module):
  def __init__(self, in_channels):
    super(EdgeAwareRefinement, self).__init__()

    self.conv2d_feature = nn.Sequential(
        convbn(in_channels, 32, kernel_size=3, stride=1, pad=1, dilation=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True))

    self.residual_astrous_blocks = nn.ModuleList()
    dilation_list = [1, 2, 4, 8, 1, 1]
    for di in dilation_list:
      self.residual_astrous_blocks.append(
          BasicBlock(32, 32, stride=1, downsample=None, pad=1, dilation=di))

    self.conv2d_out = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

  def forward(self, coarse_disparity, guidance_rgb):
    output = torch.unsqueeze(coarse_disparity, dim=1)
    upsampled_disparity = F.interpolate(
      output,
      size = guidance_rgb.size()[-2:],
      mode='bilinear',
      align_corners=False)

    # NOTE: Scale up the disparity values according to the upsampling factor.
    disparity_scale_factor = guidance_rgb.size()[-1] / coarse_disparity.size()[-1]
    upsampled_disparity *= disparity_scale_factor

    output = self.conv2d_feature(
      torch.cat([upsampled_disparity, guidance_rgb], dim=1))
    for astrous_block in self.residual_astrous_blocks:
      output = astrous_block(output)

    return nn.ReLU(inplace=True)(upsampled_disparity + self.conv2d_out(output))


class DisparityRegression(nn.Module):
  def __init__(self, maxdisp):
    super(DisparityRegression, self).__init__()

    self.disp = torch.FloatTensor(
      np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])).cuda()

  def forward(self, x):
    disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
    out = torch.sum(x * disp, 1)
    return out


class StereoNet(nn.Module):
  def __init__(self, k, r, input_scale, maxdisp=192):
    """
    k (int) : The downsampling factor for the cost volume. The spatial dimensions of the cost volume
              will be w / 2^k and h / 2^k.
    r (int) : The number of residual refinement stages.
    input_scale (int) : The scale that input and output should be at (e.g 1 = 1/2 resolution).
    """
    super(StereoNet, self).__init__()
    self.maxdisp = maxdisp
    self.k = k
    self.r = r
    self.input_scale = input_scale
    self.filter = nn.ModuleList()

    for ci in range(4):
      self.filter.append(
          nn.Sequential(
            convbn_3d(32, 32, kernel_size=3, stride=1, pad=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
          )
      )
      self.conv3d_alone = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)

    self.edge_aware_refinements = nn.ModuleList()
    for _ in range(1):
      self.edge_aware_refinements.append(EdgeAwareRefinement(4))

  def forward(self, left_img, left_features, right_features, side, output_cost_volume=False):
    coarse_max_disp = (self.maxdisp + 1) // pow(2, self.input_scale + self.k)
    outputs = {}

    # A learned matching cost is implemented through 3D convolutions.
    cost = torch.FloatTensor(left_features.size()[0],
                left_features.size()[1],
                coarse_max_disp,
                left_features.size()[2],
                left_features.size()[3]).zero_().cuda()
    for i in range(coarse_max_disp):
      if i > 0:
        cost[:, :, i, :, i:] = left_features[ :, :, :, i:] - right_features[:, :, :, :-i]
      else:
        cost[:, :, i, :, :] = left_features - right_features

    cost = cost.contiguous()
    for f in self.filter:
      cost = f(cost)
    cost = self.conv3d_alone(cost)

    # Take the soft-argmax of the cost volume.
    cost = torch.squeeze(cost, 1)
    pred = F.softmax(cost, dim=1)
    pred = DisparityRegression(coarse_max_disp)(pred)

    coarse_scale = self.input_scale + self.k

    # Optionally store the cost volume.
    if output_cost_volume:
      outputs["cost_volume_{}/{}".format(side, coarse_scale)] = cost

    # Resize the low-resolution disparity to the full resolution (and multiply by scaling factor).
    outputs["pred_disp_{}/{}".format(side, coarse_scale)] = (2**self.k) * F.interpolate(
        pred.unsqueeze(1), size=left_img.size()[-2:], mode="bilinear", align_corners=False)

    outputs["pred_disp_{}/{}".format(side, self.input_scale)] = \
        self.edge_aware_refinements[0](pred, left_img)

    return outputs

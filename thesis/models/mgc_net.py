# Copyright 2020 Massachusetts Institute of Technology
#
# @file mgc_net.py
# @author Milo Knowles
# @date 2020-03-11 17:23:46 (Wed)

from collections import OrderedDict
from math import log

import torch
import torch.nn as tnn
import torch.nn.functional as F
import torchvision as tv

from .linear_warping import LinearWarping

MATH_LN_4 = log(4)


def conv2d_padded(in_channels, out_channels, kernel_size, stride=1, bias=True, dilation=1):
  conv = tnn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                    padding=kernel_size // 2 if dilation == 1 else dilation,
                    bias=bias, dilation=dilation, groups=1)
  tnn.init.xavier_uniform_(conv.weight)
  if bias:
    tnn.init.zeros_(conv.bias)
  return conv


class CostVolumeFilter(tnn.Module):
  """
  Filters stereo cost volume using 3d convolutions.

  Based on StereoNet (Khamis, 2018).
  """
  def conv3d(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
    conv = tnn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=kernel_size//2, dilation=1, groups=1, bias=bias)
    tnn.init.constant_(conv.weight, 1.0 / (conv.weight.numel()))
    if bias:
      tnn.init.zeros_(conv.bias)
    return conv

  def __init__(self, in_channels, hidden_channels=8, out_channels=1):
    super(CostVolumeFilter, self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels

    self.relu = tnn.LeakyReLU(0.2, inplace=False)

    self.conv0 = self.conv3d(self.in_channels, self.hidden_channels, bias=False)
    # self.bn0 = tnn.modules.batchnorm.BatchNorm3d(self.out_channels)

    self.conv1 = self.conv3d(self.hidden_channels, self.hidden_channels, bias=False)
    # self.bn1 = tnn.modules.batchnorm.BatchNorm3d(self.out_channels)

    self.conv2 = self.conv3d(self.hidden_channels, self.hidden_channels, bias=False)
    # self.bn2 = tnn.modules.batchnorm.BatchNorm3d(self.out_channels)

    self.conv3 = self.conv3d(self.hidden_channels, self.hidden_channels, bias=False)
    # self.bn3 = tnn.modules.batchnorm.BatchNorm3d(self.out_channels)

    self.conv4 = self.conv3d(self.hidden_channels, self.out_channels, bias=False)

  def forward(self, volume):
    """
    volume (torch.Tensor) : Shape (b, channels, depth, height, width).
    """
    assert(len(volume.shape) == 5)

    # volume = self.relu(self.bn0(self.conv0(volume)))
    # volume = self.relu(self.bn1(self.conv1(volume)))
    # volume = self.relu(self.bn2(self.conv2(volume)))
    # volume = self.relu(self.bn3(self.conv3(volume)))
    volume = self.relu(self.conv0(volume))
    volume = self.relu(self.conv1(volume))
    volume = self.relu(self.conv2(volume))
    volume = self.relu(self.conv3(volume))

    volume = self.conv4(volume)

    return volume


class CostVolumeAbsDiff(tnn.Module):
  """
  Makes a cost volume using the absolute difference between feature vectors.
  """
  def __init__(self, max_disp, plus_and_minus=True):
    """
    max_disp (int) : The pixel radius for forming the cost volume (i.e MADNet uses a radius of 2px).
    plus_and_minus (bool) : If false, then disparity ranges from [0, max_disp] instead of [-max_disp, max_disp].
    """
    super(CostVolumeAbsDiff, self).__init__()
    self._max_disp = max_disp
    self._offset_range = range(-max_disp, max_disp+1) if plus_and_minus else range(0, max_disp+1)
    self._pad = tnn.ZeroPad2d((max_disp, max_disp, 0, 0))

  def forward(self, x, y):
    """
    Forms a cost volume by computing correlation between x and y.

    x (torch.Tensor) : Shape (b, c, h, w)
    y (torch.Tensor) : Shape (b, c, h, w)

    Returns: (torch.Tensor) of shape (b, c, 2*max_disp+1, h, w)
    """
    # Padding of max_disp on left and right.
    y_feature = self._pad(y)
    w = x.shape[3]

    absdiff_each_shift = []

    for i in self._offset_range:
      offset = self._max_disp - i
      shifted = y_feature[:,:,:,offset:offset+w]
      absdiff = torch.abs(shifted - x)
      absdiff_each_shift.append(absdiff.unsqueeze(2))

    # List of (b, c, 1, h, w) tensors, concat along 2th dim so that we get (b, c, d, h, w).
    result = torch.cat(absdiff_each_shift, axis=2)
    return result


class SoftArgMin(tnn.Module):
  """
  Compute a differentiable argmin (see GC-Net, Kendall et. al. 2017).
  """
  def __init__(self, radius_disp, device=torch.device("cuda"), plus_and_minus=True):
    super(SoftArgMin, self).__init__()
    # Gets multiplied by the probability distribution over disparity values to
    # take expected value of disparity.
    if plus_and_minus:
      self._disp_arg = torch.arange(-radius_disp, radius_disp+1).float()
      self._disp_arg = self._disp_arg.view(1, 2*radius_disp+1, 1, 1).to(device)
    else:
      self._disp_arg = torch.arange(0, radius_disp+1).float()
      self._disp_arg = self._disp_arg.view(1, radius_disp+1, 1, 1).to(device)

    self._disp_arg.requires_grad_(False)
    self.beta = 1.0

  def forward(self, cost_volume):
    """
    cost_volume (torch.Tensor) : Shape (b, num_disp, h, w).
    """
    assert(len(cost_volume.shape) == 4)
    probs = F.softmin(self.beta * cost_volume, dim=1) # Softmin along disparity dim (we squeezed out channel dim).
    return torch.sum(probs * self._disp_arg, dim=1, keepdim=True)


class PyramidFeatureExtractor(tnn.Module):
  def conv3x3(self, stride, in_channels, out_channels):
    conv = tnn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, padding_mode='zeros')
    tnn.init.xavier_uniform_(conv.weight.data)
    tnn.init.zeros_(conv.bias.data)
    return conv

  def conv5x5(self, stride, in_channels, out_channels):
    conv = tnn.Conv2d(in_channels, out_channels, 5, stride=stride, padding=2, padding_mode='zeros')
    tnn.init.xavier_uniform_(conv.weight.data)
    tnn.init.zeros_(conv.bias.data)
    return conv

  def __init__(self, layer_prefix, in_channels=3, out_channels=[64, 64, 64, 64, 64, 64]):
    super(PyramidFeatureExtractor, self).__init__()
    self._layer_prefix = layer_prefix

    self.conv1 = self.conv5x5(2, in_channels, 32)
    self.conv2 = self.conv5x5(1, 32, out_channels[0])

    self.conv3 = self.conv5x5(2, out_channels[0], out_channels[1])
    self.conv4 = self.conv3x3(1, out_channels[1], out_channels[1])

    self.conv5 = self.conv5x5(2, out_channels[1], out_channels[2])
    self.conv6 = self.conv3x3(1, out_channels[2], out_channels[2])

    self.conv7 = self.conv5x5(2, out_channels[2], out_channels[3])
    self.conv8 = self.conv3x3(1, out_channels[3], out_channels[3])

    self.conv9 = self.conv5x5(2, out_channels[3], out_channels[4])
    self.conv10 = self.conv3x3(1, out_channels[4], out_channels[4])

    self.conv11 = self.conv5x5(2, out_channels[4], out_channels[5])
    self.conv12 = self.conv3x3(1, out_channels[5], out_channels[5])

    self.relu = tnn.LeakyReLU(negative_slope=0.2)

  def forward(self, x):
    # Use normalization from Monodepth2.
    x = (x - 0.45) / 0.225

    feat = {}

    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    feat[2] = x

    x = self.relu(self.conv3(x))
    x = self.relu(self.conv4(x))
    feat[4] = x

    x = self.relu(self.conv5(x))
    x = self.relu(self.conv6(x))
    feat[6] = x

    x = self.relu(self.conv7(x))
    x = self.relu(self.conv8(x))
    feat[8] = x

    x = self.relu(self.conv9(x))
    x = self.relu(self.conv10(x))
    feat[10] = x

    x = self.relu(self.conv11(x))
    x = self.relu(self.conv12(x))
    feat[12] = x

    return feat


class RefinementNetwork(tnn.Module):
  def __init__(self, in_channels, output_residual=True, output_relu=True):
    """
    in_channels (int) : The dimension of input features + 1 (to include disparity input).
    """
    super(RefinementNetwork, self).__init__()

    self.output_residual = output_residual
    self.output_relu = output_relu

    self.conv1 = conv2d_padded(in_channels, 128, 3, stride=1, bias=True, dilation=1)
    self.conv2 = conv2d_padded(128, 128, 3, stride=1, bias=True, dilation=2)
    self.conv3 = conv2d_padded(128, 128, 3, stride=1, bias=True, dilation=4)
    self.conv4 = conv2d_padded(128, 96, 3, stride=1, bias=True, dilation=8)
    self.conv5 = conv2d_padded(96, 64, 3, stride=1, bias=True, dilation=16)
    self.conv6 = conv2d_padded(64, 32, 3, stride=1, bias=True, dilation=1)
    self.conv7 = conv2d_padded(32, 1, 3, stride=1, bias=True, dilation=1)

    self.leaky_relu = tnn.LeakyReLU(negative_slope=0.2)

  def forward(self, left_feat, disp):
    """
    Instead of computing a cost volume between left and right features, this layer simply concats
    the features from the left image with the predicted disparity and does some convolutions to add
    "context" from the image. The conv layers output a disparity correction that is added to the
    disp input.

    left_feat (torch.Tensor) : Shape (b, c, h, w)
    disp (torch.Tensor) : Shape (b, 1, h, w)
    """
    out = torch.cat([left_feat, disp], axis=1)

    out = self.leaky_relu(self.conv1(out))
    out = self.leaky_relu(self.conv2(out))
    out = self.leaky_relu(self.conv3(out))
    out = self.leaky_relu(self.conv4(out))
    out = self.leaky_relu(self.conv5(out))
    out = self.leaky_relu(self.conv6(out))
    out = self.conv7(out)

    if self.output_residual:
      return F.relu(out + disp) if self.output_relu else out + disp

    return F.relu(out) if self.output_relu else out


class MGCNet(tnn.Module):
  def __init__(self, radius_disp, img_height=192, img_width=640, device=torch.device("cuda"),
               predict_variance=False, image_channels=3, gradient_bulkhead=None,
               output_cost_volume=False, variance_mode="disparity"):
    """
    A fast MAD-Net pyramid architecture with some network ideas taken from GC-Net, such as
    cost volume filtering and the differentiable argmin operation.

    radius_disp (int)
      The max +/- disparity correction for each refinement module (2 in paper).
    img_height (int)
      Height of the full resolution images (must be multiple of 64).
    img_width (int)
      Width of the full resolution images (must be multiple of 64).
    gradient_bulkhead (None or str)
      An optional location in the network to stop storing gradients. The "bulkhead" is used in
      the MAD-Net paper for faster adaptation when only one part of the network is being updated.
      "None" means the entire network stores gradients. A str such as "D3" means put a bulkhead
      BEFORE the D3 disparity module.

    This means gradients are backpropagated from the loss function, through D3, but not past there:
    FeatureExtractor ==> D6 => D5 => D4 | D3 => D2 => RefinementNetwork => VarianceModule
    """
    super(MGCNet, self).__init__()

    valid_bulkhead_locations = ["FeatureExtractor", "D6", "D5", "D4", "D3", "D2", "RefinementNetwork", "VarianceModule"]
    if gradient_bulkhead is not None and gradient_bulkhead not in valid_bulkhead_locations:
      raise ValueError("MGC-Net initialized with invalid gradient_bulkhead {}".format(gradient_bulkhead))

    self._predict_variance = predict_variance
    self._variance_mode = variance_mode
    self._output_cost_volume = output_cost_volume

    self._cost_volume_filters = OrderedDict()
    self.resize = OrderedDict()
    self.warp = OrderedDict()

    self._radius_disp = radius_disp
    self._img_height = img_height
    self._img_width = img_width
    self._scales = [1, 2, 4, 8, 16, 32, 64]

    self._make_cost_volume_plus = CostVolumeAbsDiff(radius_disp, plus_and_minus=False)
    self._make_cost_volume_plus_and_minus = CostVolumeAbsDiff(radius_disp, plus_and_minus=True)

    self._soft_argmin_plus = SoftArgMin(radius_disp, device=device, plus_and_minus=False)
    self._soft_argmin_plus_and_minus = SoftArgMin(radius_disp, device=device, plus_and_minus=True)

    self.feature_dim = [image_channels, 16, 32, 64, 96, 128, 192]
    self.feature_extractor = PyramidFeatureExtractor("pyramid", in_channels=image_channels,
                                                     out_channels=self.feature_dim[1:]).to(device)

    if self._predict_variance:
      # Variance module shouldn't be constrained to produce a positive output.
      self.variance_module = RefinementNetwork(
          self.feature_dim[2] + 1 if variance_mode == "disparity" else 3 + 1,
          output_relu=False, output_residual=False).to(device)

    for i in range(6, 0, -1):
      # For G5, G4, ..., and below, need to warp using the disparity from module above.
      if i <= 5:
        self.warp[i] = LinearWarping(self._img_height // self._scales[i], self._img_width // self._scales[i], device)

      if i >= 2:
        self._cost_volume_filters[i] = CostVolumeFilter(in_channels=self.feature_dim[i], hidden_channels=16).to(device)

      # Each module resizes its predicted disparity for the module below.
      upsample_size = (self._img_height // self._scales[i-1], self._img_width // self._scales[i-1])
      self.resize[i-1] = tnn.Upsample(size=upsample_size, mode="bilinear", align_corners=False)

    self._refinement_modules = OrderedDict()
    self._refinement_modules[2] = RefinementNetwork(self.feature_dim[2] + 1).to(device)
    self.refinement_modules = tnn.ModuleList(list(self._refinement_modules.values()))

    self.cost_volume_filters = tnn.ModuleList(list(self._cost_volume_filters.values()))

    self.resize[0] = tnn.Upsample(size=(self._img_height, self._img_width), mode="bilinear", align_corners=False)
    self.resize[6] = tnn.Upsample(size=(self._img_height // self._scales[6], self._img_width // self._scales[6]), mode="bilinear", align_corners=False)
    self._activation = tnn.LeakyReLU(negative_slope=0.2)

    # Disable autograd on network layers behind the gradient bulkhead.
    if gradient_bulkhead is not None:
      print("NOTE: MGC-Net has gradient bulkhead at {}".format(gradient_bulkhead))
      index_of_bulkhead = valid_bulkhead_locations.index(gradient_bulkhead)
      for i, module_name in enumerate(valid_bulkhead_locations):
        if i < index_of_bulkhead:
          if module_name in ["D6", "D5", "D4", "D3", "D2"]:
            for param in self._cost_volume_filters[int(module_name[1])].parameters():
              param.requires_grad = False
            if module_name != "D6":
              for param in self.warp[int(module_name[1])].parameters():
                param.requires_grad = False
          elif module_name == "RefinementNetwork":
            for param in self._refinement_modules[2].parameters():
              param.requires_grad = False
          elif module_name == "FeatureExtractor":
            for param in self.feature_extractor.parameters():
              param.requires_grad = False
          else:
            raise Exception()

  def forward(self, left_feat, right_feat, side="l", output_scales=set([0, 1, 2, 3, 4, 5, 6]), variance_input=None):
    """
    variance_input (None or torch.Tensor) : Should have shape (b, c, h // 4, w // 4).
    """
    outputs = {}

    cost_volume = self._make_cost_volume_plus(left_feat[12], right_feat[12])
    cost_volume = self._cost_volume_filters[6](cost_volume)

    pred_disp = F.relu(self._soft_argmin_plus(cost_volume[:,0,:,:,:]))
    outputs["pred_disp_{}/6".format(side)] = pred_disp
    pred_disp = 2.0 * self.resize[5](outputs["pred_disp_{}/6".format(side)])

    for i in range(5, 1, -1):
      left_input_feat = left_feat[2*i]

      # Align the right features with the left image using disparity from the resolution below.
      right_input_feat, _ = self.warp[i](right_feat[2*i], pred_disp, mode="bilinear")
      cost_volume = self._make_cost_volume_plus_and_minus(left_input_feat, right_input_feat)
      cost_volume = self._cost_volume_filters[i](cost_volume)

      # NOTE: Argmin is over +/- disparity corrections.
      disp_delta = self._soft_argmin_plus_and_minus(cost_volume[:,0,:,:,:])
      pred_disp = F.relu(pred_disp + disp_delta)

      if i == 2:
        pred_disp = F.relu(self._refinement_modules[i](left_feat[4], pred_disp))

      # Do a ReLU to make all disparity positive.
      outputs["pred_disp_{}/{}".format(side, i)] = pred_disp

      # Since we're upsampling the disp to the next pyramid level, need to multiply it's values by 2.
      pred_disp = 2.0 * self.resize[i-1](outputs["pred_disp_{}/{}".format(side, i)])

    # Don't do cost volume step for R1, just refine.
    outputs["pred_disp_{}/1".format(side)] = pred_disp

    # Just resize to R0.
    pred_disp = 2.0 * self.resize[0](outputs["pred_disp_{}/1".format(side)])
    outputs["pred_disp_{}/0".format(side)] = pred_disp

    if self._predict_variance:
      outputs["pred_log_variance_{}/2".format(side)] = self.variance_module(
          left_feat[4] if variance_input is None else variance_input, 0.01 * outputs["pred_disp_{}/2".format(side)])

      # If predicting a variance over disparity, need to multiply upsampled variance predictions by a factor of 4.
      if self._variance_mode == "disparity":
        outputs["pred_log_variance_{}/1".format(side)] = self.resize[1](outputs["pred_log_variance_{}/2".format(side)]) + MATH_LN_4
        outputs["pred_log_variance_{}/0".format(side)] = self.resize[0](outputs["pred_log_variance_{}/1".format(side)]) + MATH_LN_4

      # If predicting a photometric variance, it's less clear how that should change with scale...
      # For now I'm just upsampling the values but leaving them unscaled.
      else:
        outputs["pred_log_variance_{}/1".format(side)] = self.resize[1](outputs["pred_log_variance_{}/2".format(side)])
        outputs["pred_log_variance_{}/0".format(side)] = self.resize[0](outputs["pred_log_variance_{}/1".format(side)])

        outputs["pred_log_variance_{}/3".format(side)] = self.resize[3](outputs["pred_log_variance_{}/2".format(side)])
        outputs["pred_log_variance_{}/4".format(side)] = self.resize[4](outputs["pred_log_variance_{}/3".format(side)])
        outputs["pred_log_variance_{}/5".format(side)] = self.resize[5](outputs["pred_log_variance_{}/4".format(side)])
        outputs["pred_log_variance_{}/6".format(side)] = self.resize[6](outputs["pred_log_variance_{}/5".format(side)])

    # Delete outputs that won't be used to free up memory.
    for scale in range(7):
      if scale not in output_scales:
        del outputs["pred_disp_{}/{}".format(side, scale)]

    return outputs

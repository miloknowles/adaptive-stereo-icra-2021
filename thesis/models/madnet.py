# Copyright 2020 Massachusetts Institute of Technology
#
# @file madnet.py
# @author Milo Knowles
# @date 2020-03-16 16:46:19 (Mon)

from collections import OrderedDict

import torch
import torch.nn as tnn
import torch.nn.functional as F

from .linear_warping import LinearWarping


def conv2d_padded(in_channels, out_channels, kernel_size, stride=1, bias=True, dilation=1):
  conv = tnn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                    padding=kernel_size // 2 if dilation == 1 else dilation,
                    bias=bias, dilation=dilation, groups=1)
  tnn.init.xavier_uniform_(conv.weight)
  if bias:
    tnn.init.zeros_(conv.bias)
  return conv


class CorrelationLayer(tnn.Module):
  def __init__(self, max_disp, plus_and_minus=True):
    """
    max_disp (int) :
      The maximum disparity offset to compute correlation at.
    plus_and_minus (bool) :
      Whether or not to consider negative correlation shifts. For the first disparity module, this
      should be False, since we are estimation aboslute disparity. At subsequent pyramid levels,
      we consider negative corrections, so set to True.
    """
    super(CorrelationLayer, self).__init__()
    self._max_disp = max_disp
    self._offset_range = range(-max_disp, max_disp+1) if plus_and_minus else range(0, max_disp+1)

    # If plus_and_minus=False, we only need to pad to the LEFT.
    self._pad = tnn.ZeroPad2d((max_disp, max_disp if plus_and_minus else 0, 0, 0))

  def forward(self, x, y):
    """
    Forms a cost volume by computing the correlation (dot product) between x and y.

    x (torch.Tensor) : Shape (b, c, h, w)
    y (torch.Tensor) : Shape (b, c, h, w)

    Returns: (torch.Tensor) of shape (b, 2*max_disp+1, h, w)
    """
    y_padded = self._pad(y)
    b, c, h, w = x.shape

    correlation_each_shift = []

    # A positive disparity means go to the LEFT in the y image.
    for d in self._offset_range:
      offset = self._max_disp - d
      y_shifted_by_d = y_padded[:,:,:,offset:offset+w]
      dot_product = y_shifted_by_d * x

      # Take the mean along the channel dimension to get normalized dot product.
      correlation_each_shift.append(dot_product.mean(dim=1, keepdim=True))

    # List of (b, 1, h, w) tensors, concat along 1th dim so that we get (b, d, h, w).
    return torch.cat(correlation_each_shift, axis=1)


class DisparityModule(tnn.Module):
  def __init__(self, radius_disp, has_disp_input=True):
    """
    radius_disp (int) : The radius for disparity calculations.
    layer_prefix (str) : String to prepend to all layer names.
    has_disp_input (bool) : Does this layer expect an extra disparity channel in volume?
    """
    super(DisparityModule, self).__init__()
    self._in_channels = 2*radius_disp + 2 if has_disp_input else 2*radius_disp + 1

    self.conv1 = conv2d_padded(self._in_channels, 128, 3, stride=1, bias=True)
    self.conv2 = conv2d_padded(128, 128, 3, stride=1, bias=True)
    self.conv3 = conv2d_padded(128, 96, 3, stride=1, bias=True)
    self.conv4 = conv2d_padded(96, 64, 3, stride=1, bias=True)
    self.conv5 = conv2d_padded(64, 32, 3, stride=1, bias=True)
    self.conv6 = conv2d_padded(32, 1, 3, stride=1, bias=True)

    self.leaky_relu = tnn.LeakyReLU(negative_slope=0.2)

  def forward(self, cost_volume, upsampled_disp=None):
    """
    cost_volume (torch.Tensor) : Shape (b, 2*radius_disp+1, h, w)
    upsampled_disp (torch.Tensor) : Shape (b, 1, h, w) if not None.

    Returns: (torch.Tensor) : Shape (b, 1, h, w).
    """
    if upsampled_disp is not None:
      out = torch.cat([cost_volume, upsampled_disp], axis=1)
    else:
      out = cost_volume

    out = self.leaky_relu(self.conv1(out))
    out = self.leaky_relu(self.conv2(out))
    out = self.leaky_relu(self.conv3(out))
    out = self.leaky_relu(self.conv4(out))
    out = self.leaky_relu(self.conv5(out))
    out = self.conv6(out)

    return out


def make_disparity(x, out_height, out_width):
  """
  Converts a raw network output 'x' into a 'disparity' output.

  NOTE(milo): I think the MADNet authors use a ReLU here to set any negative disparities to zero,
  putting those pixels at infinite depth. They scale by 20 because they want the network to learn
  disparity at 1/20th full resolution pixel scale, which improves convergence.

  https://github.com/CVLAB-Unibo/Real-time-self-adaptive-deep-stereo/issues/41

  Instead of put a negative in the ReLU here, do it in the LinearWarping module.
  """
  out = F.relu(20.0 * x)
  out = F.interpolate(out, size=(out_height, out_width), mode='bilinear', align_corners=False)
  return out


def make_disparity_no_resize(x):
  return F.relu(20.0 * x)


class RefinementNetwork(tnn.Module):
  def __init__(self, in_channels):
    """
    in_channels (int) : The dimension of input features + 1 (to include disparity input).
    """
    super(RefinementNetwork, self).__init__()
    self._convs = OrderedDict()

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

    return F.relu(out + disp)


class PyramidFeatureExtractor(tnn.Module):
  def __init__(self, in_channels=3, out_channels=[16, 32, 64, 96, 128, 192]):
    super(PyramidFeatureExtractor, self).__init__()

    self.conv1 = conv2d_padded(in_channels, out_channels[0], 3, stride=2)
    self.conv2 = conv2d_padded(out_channels[0], out_channels[0], 3, stride=1)

    self.conv3 = conv2d_padded(out_channels[0], out_channels[1], 3, stride=2)
    self.conv4 = conv2d_padded(out_channels[1], out_channels[1], 3, stride=1)

    self.conv5 = conv2d_padded(out_channels[1], out_channels[2], 3, stride=2)
    self.conv6 = conv2d_padded(out_channels[2], out_channels[2], 3, stride=1)

    self.conv7 = conv2d_padded(out_channels[2], out_channels[3], 3, stride=2)
    self.conv8 = conv2d_padded(out_channels[3], out_channels[3], 3, stride=1)

    self.conv9 = conv2d_padded(out_channels[3], out_channels[4], 3, stride=2)
    self.conv10 = conv2d_padded(out_channels[4], out_channels[4], 3, stride=1)

    self.conv11 = conv2d_padded(out_channels[4], out_channels[5], 3, stride=2)
    self.conv12 = conv2d_padded(out_channels[5], out_channels[5], 3, stride=1)

    self.leaky_relu = tnn.LeakyReLU(negative_slope=0.2)

  def forward(self, x):
    feat = {}

    x = self.leaky_relu(self.conv1(x))
    x = self.leaky_relu(self.conv2(x))
    feat[2] = x

    x = self.leaky_relu(self.conv3(x))
    x = self.leaky_relu(self.conv4(x))
    feat[4] = x

    x = self.leaky_relu(self.conv5(x))
    x = self.leaky_relu(self.conv6(x))
    feat[6] = x

    x = self.leaky_relu(self.conv7(x))
    x = self.leaky_relu(self.conv8(x))
    feat[8] = x

    x = self.leaky_relu(self.conv9(x))
    x = self.leaky_relu(self.conv10(x))
    feat[10] = x

    x = self.leaky_relu(self.conv11(x))
    x = self.leaky_relu(self.conv12(x))
    feat[12] = x

    return feat


class MadNet(tnn.Module):
  def __init__(self, radius_disp, img_height=192, img_width=640, device=torch.device("cuda")):
    """
    radius_disp (int) : The max +/- disparity correction for each refinement module (2 in paper).
    """
    super(MadNet, self).__init__()

    self._radius_disp = radius_disp
    self._img_height = img_height
    self._img_width = img_width
    self._scales = [1, 2, 4, 8, 16, 32, 64]
    self._feature_dims = [16, 32, 64, 96, 128, 192]

    self.feature_extractor = PyramidFeatureExtractor(in_channels=3, out_channels=self._feature_dims).to(device)
    self._disparity_modules = OrderedDict()
    self.resizers = OrderedDict()
    self.warpers = OrderedDict()
    self.correlation = CorrelationLayer(self._radius_disp)

    for i in range(6, 1, -1):
      self.warpers[i] = LinearWarping(self._img_height // self._scales[i], self._img_width // self._scales[i], device)
      self._disparity_modules[i] = DisparityModule(self._radius_disp, has_disp_input=(i < 6)).to(device)

      # Each module resizes its predicted disparity for the module below.
      upsample_size = (self._img_height // self._scales[i-1], self._img_width // self._scales[i-1])
      self.resizers[i-1] = tnn.Upsample(size=upsample_size, mode="bilinear", align_corners=False)

    self.disparity_modules = tnn.ModuleList(list(self._disparity_modules.values()))
    self.refinement_module = RefinementNetwork(32 + 1).to(device)

    # Resize things to the full resolution (R0).
    self.resizers[1] = tnn.Upsample(size=(self._img_height // 2, self._img_width // 2), mode="bilinear", align_corners=False)
    self.resizers[0] = tnn.Upsample(size=(self._img_height, self._img_width), mode="bilinear", align_corners=False)

  def forward(self, inputs):
    im_left = inputs["color_l/0"]
    im_right = inputs["color_r/0"]

    outputs = {}
    b, c, h, w = im_left.shape
    assert(h == self._img_height)
    assert(w == self._img_width)
    assert(im_left.min() >= 0 and im_left.max() <= 1.0)
    assert(im_right.min() >= 0 and im_right.max() <= 1.0)

    # Apply image normalization from Monodepth2 ResNet encoder.
    im_left_normalized = (im_left - 0.45) / 0.225
    im_right_normalized = (im_right - 0.45) / 0.225

    left_feat = self.feature_extractor(im_left_normalized)
    right_feat = self.feature_extractor(im_right_normalized)

    pred_disp = None

    cost_volume = self.correlation(left_feat[12], right_feat[12])

    pred_disp = F.relu(self._disparity_modules[6](cost_volume, upsampled_disp=None))
    outputs["pred_disp_l/6"] = make_disparity_no_resize(pred_disp)
    pred_disp = self.resizers[5](pred_disp) * 20.0 / self._scales[5]

    for i in range(5, 1, -1):
      left_input_feat = left_feat[2*i]
      right_input_feat, _ = self.warpers[i](right_feat[2*i], pred_disp)
      cost_volume = self.correlation(left_input_feat, right_input_feat)
      pred_disp = self._disparity_modules[i](cost_volume, upsampled_disp=pred_disp)

      if i > 2:
        outputs["pred_disp_l/{}".format(i)] = make_disparity_no_resize(pred_disp)
        pred_disp = self.resizers[i-1](pred_disp) * 20.0 / self._scales[i-1]

    pred_disp = self.refinement_module(left_feat[4], pred_disp)
    outputs["pred_disp_l/2"] = make_disparity_no_resize(pred_disp) / self._scales[2]
    outputs["pred_disp_l/1"] = 2.0 * self.resizers[1](outputs["pred_disp_l/2"])
    outputs["pred_disp_l/0"] = 2.0 * self.resizers[0](outputs["pred_disp_l/1"])

    return outputs

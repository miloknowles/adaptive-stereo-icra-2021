# Copyright 2020 Massachusetts Institute of Technology
#
# @file visualization.py
# @author Milo Knowles
# @date 2020-03-11 20:28:13 (Wed)

import math

import torch
from torchvision import transforms

from PIL import Image
import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm


def maybe_put_channel_dim_first(x):
  if not (x.shape[0] == 1 or x.shape[0] == 3):
    x = np.moveaxis(x, -1, 0)
  return x


def maybe_put_channel_dim_last(x):
  if not (x.shape[-1] == 1 or x.shape[-1] == 3):
    x = np.moveaxis(x, 0, -1)
  return x


def tensor_to_cv_disp(disp_t, cast_uint8=True):
  """
  disp_t (torch.Tensor) : Shape must be one of:
    (1, height, width)
    (height, width, 1)
    (height, width)
  """
  # Make sure a singleton channel dim exists.
  if len(disp_t.shape) == 2:
    np_3d = disp_t.unsqueeze(-1).numpy()
  else:
    np_3d = disp_t.numpy()

  np_3d = maybe_put_channel_dim_last(np_3d)

  # Normalize by width.
  if cast_uint8:
    return (255.0 * np_3d / np_3d.shape[1]).astype(np.uint8)
  else:
    return (255.0 * np_3d / np_3d.shape[1])


def tensor_to_cv_rgb(rgb_t):
  """
  rgb_t (torch.Tensor) : Shape must be one of:
    (3, height, width)
    (height, width, 3)

  Assumes that the input image has pixel values in range [0, 1].
  """
  assert(len(rgb_t.shape) == 3)
  im = maybe_put_channel_dim_last(rgb_t.cpu().numpy())
  im = (255.0 * im).astype(np.uint8)
  return cv.cvtColor(im, cv.COLOR_RGB2BGR)


def tensor_to_cv_gray(gray_t):
  """
  gray_t (torch.Tensor) : Shape must be one of:
    (1, height, width)
    (height, width, 1)

  Assumes that the input image has pixel values in range [0, 1].
  """
  assert(len(gray_t.shape) == 3)
  im = maybe_put_channel_dim_last(gray_t.cpu().numpy())
  im = (255.0 * im).astype(np.uint8)
  return im


def visualize_disp_tensorboard(disp_t, cmap=plt.get_cmap("magma"), vmin=None, vmax=None):
  """
  Prepare a disparity image for logging with tensorboard.

  disp_t (torch.Tensor) : Disparity image in pixel units, can be 4D (batched) or 3D.
  cmap (maplotlib.Colormap) : Any option from plt.get_cmap.

  """
  if len(disp_t) == 3:
    disp_t = disp_t.unsqueeze(0)

  # Colormap (internal normalization), remove batch dim and alpha channel.
  out = apply_cmap(disp_t.unsqueeze(0), cmap=cmap, vmin=vmin, vmax=vmax)[0,:,:,:3]

  # Make sure the channel
  return maybe_put_channel_dim_first(out)


def visualize_disp_cv(disp_t, cmap=plt.get_cmap("magma"), vmin=None, vmax=None):
  """
  disp_t (torch.Tensor) : Disparity image in pixel units.

  Shape must be one of:
    (channels, height, width)
    (height, width, channels)
    (height, width)
  """
  if len(disp_t.shape) == 3:
    disp_t = disp_t.unsqueeze(0)

  # Colormap (internal normalization).
  out = apply_cmap(disp_t, cmap=cmap, vmin=vmin, vmax=vmax)[0,:,:,:3]

  # Convert to an OpenCV showable image.
  out = float_image_to_cv_uint8(out, encoding="rgb")

  return out


# From: https://github.com/robustrobotics/rrg/blob/master/src/perception/depthest/python/depthest/utils/visualization.py
def apply_cmap(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function that maps a grayscale image with a matplotlib colormap for
    use with TensorBoard image summaries.

    By default it will normalize the input value to the range 0..1 per batch
    item.

    Arguments:
      - value: 4D Tensor of shape [batch, 1, height, width].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`

    Returns a 4D RGBA numpy array of shape [batch, height, width, 4].
    """
    assert(len(value.shape) == 4)

    value_cpu = value.detach().squeeze(1).cpu()

    batch_size = value_cpu.shape[0]
    rows = value_cpu.shape[1]
    cols = value_cpu.shape[2]

    if vmin is None:
      vmin, _ = torch.min(value_cpu.view(batch_size, -1), 1, keepdim=True)
      vmin = vmin.unsqueeze(2).expand(-1, rows, cols)

    if vmax is None:
      vmax, _ = torch.max(value_cpu.view(batch_size, -1), 1, keepdim=True)
      vmax = vmax.unsqueeze(2).expand(-1, rows, cols)

    normalized_value = (value_cpu - vmin) / (vmax - vmin)
    normalized_value = normalized_value.numpy()

    if cmap is None:
      cmap = matplotlib.cm.get_cmap("gray")

    # https://matplotlib.org/_modules/matplotlib/colors.html#Colormap.set_bad
    mapped = cmap(normalized_value)

    return mapped


def float_image_to_cv_uint8(float_im, encoding="rgb"):
  """
  Converts a np.ndarray image to an OpenCV showable BGR image.

  float_im (np.ndarray) : Should be normalized to the range [0, 1].
  encoding (str) : Either "rgb" or "bgr". If "rgb", then the red and blue
                   channels are flipped to get BGR output.
  """
  assert(len(float_im.shape) == 3)
  if float_im.min() < 0 or float_im.max() > 1:
    print("WARNING: uint8_image overflow (min={} max={}".format(float_im.min(), float_im.max()))

  out = (255.0 * float_im).astype(np.uint8)

  if encoding == "rgb":
    return cv.cvtColor(out, cv.COLOR_RGB2BGR)

  return out

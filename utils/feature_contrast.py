# Copyright 2020 Massachusetts Institute of Technology
#
# @file feature_contrast.py
# @author Milo Knowles
# @date 2020-07-09 16:22:26 (Thu)

import torch


def feature_contrast_median(cost_volume):
  """
  "Max-minus-median" version of the feature contrast score (FCS).
  """
  with torch.no_grad():
    max_each_pixel = torch.max(cost_volume, dim=1)[0]
    med_each_pixel = torch.median(cost_volume, dim=1)[0]
    return (max_each_pixel - med_each_pixel)


def feature_contrast_mean(cost_volume):
  """
  "Max-minus-mean" version of the feature contrast score (FCS).
  """
  with torch.no_grad():
    sorted_each_pixel = torch.sort(cost_volume, dim=1, descending=True)[0]

    max_each_pixel = sorted_each_pixel[:,0,:,:]

    # NOTE: We ignore the top 2 pixels, since the network will sometimes produce high similarity
    # scores for adjacent pixels. This is needed for subpixel interpolation through the soft-argmax.
    mean_nonmax = sorted_each_pixel[:,2:,:,:].mean(dim=1)

    return (max_each_pixel - mean_nonmax)

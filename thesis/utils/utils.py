# Copyright 2020 Massachusetts Institute of Technology
#
#	@file utils.py
#	@author Milo Knowles
#	@date 2020-02-13 09:46:39 (Thu)

import os
from PIL import Image


# Source: https://github.com/nianticlabs/monodepth2
def sec_to_hm(t):
  """
  Convert time in seconds to time in hours, minutes and seconds
  e.g. 10239 -> (2, 50, 39)
  """
  t = int(t)
  s = t % 60
  t //= 60
  m = t % 60
  t //= 60
  return t, m, s

# Source: https://github.com/nianticlabs/monodepth2
def sec_to_hm_str(t):
  """
  Convert time in seconds to a nice string
  e.g. 10239 -> '02h50m39s'
  """
  h, m, s = sec_to_hm(t)
  return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


# NOTE: Adapted from https://github.com/nianticlabs/monodepth2
def pil_loader(path):
  # open path as file to avoid ResourceWarning
  # (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
    with Image.open(f) as img:
      return img.convert('RGB')

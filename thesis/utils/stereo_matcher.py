# Copyright 2020 Massachusetts Institute of Technology
#
# @file stereo_matcher.py
# @author Milo Knowles
# @date 2020-06-18 10:27:32 (Thu)

import numpy as np
import cv2 as cv

# NOTE(milo): Couldn't find this enum in Python, but it looks like it has value 1 here:
# https://docs.opencv.org/3.4/d7/d8e/classcv_1_1stereo_1_1StereoBinaryBM.html#afc1f818cefb6256642f29575d0f90925
PREFILTER_XSOBEL = 1


class StereoMatcher(object):
  def __init__(self, height, width, min_disp, max_disp):
    """
    Details on the meaning of the parameters can be found here:
    https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html#aa0f168808513cabc221b763664791185
    """
    self.min_disp = min_disp
    self.max_disp = max_disp
    self.num_disp = max_disp - min_disp + 1

    self.height = height
    self.width = width

    # https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
    self.bm = cv.StereoBM_create(numDisparities = self.num_disp, blockSize = 7)
    self.bm.setMinDisparity(self.min_disp)
    # self.bm.setP1(8*11*11)
    # self.bm.setP2(3*11*11)
    self.bm.setTextureThreshold(100)
    self.bm.setDisp12MaxDiff(1)
    self.bm.setUniquenessRatio(0)
    self.bm.setPreFilterSize(5)
    self.bm.setSpeckleWindowSize(500)
    self.bm.setSpeckleRange(3)
    self.bm.setPreFilterType(PREFILTER_XSOBEL)
    self.bm.setPreFilterCap(31)

  def compute_disparity(self, im_left, im_right):
    """
    Uses block matching to compute a disparity map for two images.
    """
    assert(im_left.shape == (self.height, self.width))
    assert(im_right.shape == (self.height, self.width))
    disp = self.bm.compute(im_left, im_right)
    return disp.astype(np.float32) / 16.0

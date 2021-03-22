# Copyright 2020 Massachusetts Institute of Technology
#
#	@file config.py
#	@author Milo Knowles
#	@date 2020-03-09 21:25:54 (Mon)

import numpy as np

from utils.path_utils import *


class Config(object):
  """
  Stores paramaters for the StereoDepthNode.
  """
  # ROS parameters.
  LEFT_IMAGE_CHANNEL = "/stereo_camera/left/compressed_0"
  RIGHT_IMAGE_CHANNEL = "/stereo_camera/right/compressed_0"
  LEFT_CAMERA_POSE_CHANNEL = "/stereo_camera/left/pose"
  RIGHT_CAMERA_POSE_CHANNEL = "/stereo_camera/right/pose"
  OUTPUT_DISP_CHANNEL_FMT = "/stereo_depth_node/disp_{}"
  OUTPUT_VOXEL_CHANNEL = "/stereo_depth_node/local_voxel_map"
  PUBLISH_DISP_HZ = 20
  PUBLISH_COLOR_POINT_CLOUD = True

  # Stereo model parameters.
  CAMERA_INPUT_HEIGHT = 320
  CAMERA_INPUT_WIDTH = 1216
  MODEL_INPUT_HEIGHT = 320
  MODEL_INPUT_WIDTH = 1216
  STEREONET_K = 4
  LOAD_WEIGHTS_FOLDER = path_to_resources(reldir="pretrained_models/stereo_net/vk_clone_368x960_16X")
  MAX_DEPTH = 100

  # Voxel-related parameters.
  # FOCAL_LENGTH_FX = 7.215377e+02
  # STEREO_BASELINE_METERS = 0.54
  STEREO_BASELINE_METERS = 1.0
  VOXEL_DISP_SCALE = 2 # The disparity pyramid scale used to build the voxel map.

  # This should be the intrinsics for the camera at the input that images arrive at.
  # TODO: multiply fx * 1216 / 1242, fy * 320 / 375
  # CAMERA_INTRINSICS = np.array([
  #   [7.215377e+02, 0.000000e+00, 6.095593e+02],
  #   [0.000000e+00, 7.215377e+02, 1.728540e+02],
  #   [0.000000e+00, 0.000000e+00, 1.000000e+00]
  # ])

  CAMERA_INTRINSICS_INPUT_IMAGE = np.array([
    [1329, 0,     607.5],
    [0,    1329,  159.5],
    [0,    0,     1]
  ])

  # CAMERA_INTRINSICS = np.array([
  #   [1.09375, 0.0,      0.5],
  #   [0,       1.94444,  0.5],
  #   [0.0,     0.0,      1.0]
  # ])

  DEPTH_SCALE_FACTOR = 100.0
  VOXEL_SCALE_METERS = 0.15

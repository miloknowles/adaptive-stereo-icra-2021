# Copyright 2020 Massachusetts Institute of Technology
#
#	@file stereo_depth_node.py
#	@author Milo Knowles
#	@date 2020-03-09 20:50:24 (Mon)

import time, os

import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np

import rospy
import message_filters as mf
from cv_bridge import CvBridge
import open3d as o3d
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

from .config import Config
from utils.visualization import visualize_disp_cv, tensor_to_cv_rgb
from utils.training_utils import load_model
from models.stereo_net import StereoNet, FeatureExtractorNetwork


from .open3d_to_ros import convert_pointcloud_open3d_to_ros


def image_msg_to_tensor(msg):
  """
  The network expects RGB images with shape: (c, h, w).
  """
  return torch.from_numpy(CvBridge().imgmsg_to_cv2(msg, desired_encoding="rgb8")).permute(2, 0, 1).float() / 255.0


def disp_tensor_to_image_msg(t, stamp, bridge):
  """
  Jet-map a disparity tensor into an OpenCV BGR image, then convert to a ros sensor_msgs::Image.
  """
  # colorized = visualize_disp_cv(t.cpu())
  # t = (255 * t / 100).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
  t = visualize_disp_cv(t, vmin=0, vmax=80)
  # t = (255 * (t - t.min()) / (t.max() - t.min())).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
  msg = bridge.cv2_to_imgmsg(t, encoding="bgr8")
  # msg = CompressedImage()
  # msg.format = "jpeg"
  # msg.data = np.array(cv.imencode(".jpg", t)[1]).tostring()
  msg.header.stamp = stamp
  return msg


class StereoDepthNode(object):
  """
  Run from the directory above with Python3:
    python -m ros.stereo_depth_node
  """
  def __init__(self):
    # https://github.com/pytorch/pytorch/issues/15054
    torch.backends.cudnn.benchmark=True

    self.left_img_sub = mf.Subscriber(Config.LEFT_IMAGE_CHANNEL, Image)
    self.right_img_sub = mf.Subscriber(Config.RIGHT_IMAGE_CHANNEL, Image)

    # Publish several pyramid levels of disparity predictions.
    self.disp_pubs = {
      0: rospy.Publisher(Config.OUTPUT_DISP_CHANNEL_FMT.format(0), Image, queue_size=3),
      1: rospy.Publisher(Config.OUTPUT_DISP_CHANNEL_FMT.format(1), Image, queue_size=3),
      2: rospy.Publisher(Config.OUTPUT_DISP_CHANNEL_FMT.format(2), Image, queue_size=3)
    }

    self.voxel_pub = rospy.Publisher(Config.OUTPUT_VOXEL_CHANNEL, PointCloud2, queue_size=3)
    self.bridge = CvBridge()

    # TODO(milo): What is a reasonable queue size?
    self.ts = mf.TimeSynchronizer([self.left_img_sub, self.right_img_sub], 5)
    self.ts.registerCallback(self.image_pair_callback)

    self.device = torch.device("cuda")
    self.feature_net = FeatureExtractorNetwork(Config.STEREONET_K).cuda()
    self.stereo_net = StereoNet(k=Config.STEREONET_K, r=1).cuda()

    print("Loading models from: ", Config.LOAD_WEIGHTS_FOLDER)
    self.feature_net.load_state_dict(torch.load(
        os.path.join(Config.LOAD_WEIGHTS_FOLDER, "feature_net.pth")), strict=True)
    self.stereo_net.load_state_dict(torch.load(
        os.path.join(Config.LOAD_WEIGHTS_FOLDER, "stereo_net.pth")), strict=True)
    print("[StereoDepthNode]", "Loaded in model from {}".format(Config.LOAD_WEIGHTS_FOLDER))

    # Intrinsics after cropping/resizing to the model's input resolution.
    self.K_crop = Config.CAMERA_INTRINSICS_INPUT_IMAGE.copy()
    # self.K_crop[0] *= Config.MODEL_INPUT_WIDTH / Config.CAMERA_INPUT_WIDTH
    # self.K_crop[1] *= Config.MODEL_INPUT_HEIGHT / Config.CAMERA_INPUT_HEIGHT
    # self.K_crop[0] *= Config.MODEL_INPUT_WIDTH
    # self.K_crop[1] *= Config.MODEL_INPUT_HEIGHT

    # Intrinsics at the pyramid level that we make the voxel map.
    self.K_voxel = self.K_crop.copy()
    self.K_voxel[0] /= 2**Config.VOXEL_DISP_SCALE
    self.K_voxel[1] /= 2**Config.VOXEL_DISP_SCALE

    # Set up the camera intrinsics for Open3D.
    self.K_o3d = o3d.camera.PinholeCameraIntrinsic()
    self.width_voxel_scale = Config.MODEL_INPUT_WIDTH / 2**Config.VOXEL_DISP_SCALE
    self.height_voxel_scale = Config.MODEL_INPUT_HEIGHT / 2**Config.VOXEL_DISP_SCALE
    fx, fy = self.K_voxel[0,0], self.K_voxel[1,1]
    cx, cy = self.K_voxel[0,2], self.K_voxel[1,2]
    self.K_o3d.set_intrinsics(self.width_voxel_scale, self.height_voxel_scale, fx, fy, cx, cy)

    self.last_publish_disp_time = 0

  def image_pair_callback(self, left_img_msg, right_img_msg):
    """
    Predict disparity for a time-synchronized pair of images.
    """
    t0 = time.time()
    left_img_t = image_msg_to_tensor(left_img_msg).to(self.device)
    right_img_t = image_msg_to_tensor(right_img_msg).to(self.device)

    # Make sure that images are the correct numerical range! Otherwise network outputs garbage.
    assert(left_img_t.min() >= 0 and left_img_t.max() <= 1.0)
    assert(right_img_t.min() >= 0 and right_img_t.max() <= 1.0)

    inputs = {"color_l/0": left_img_t.unsqueeze(0),
              "color_r/0": right_img_t.unsqueeze(0)}

    with torch.no_grad():
      left_feat = self.feature_net(inputs["color_l/0"])
      right_feat = self.feature_net(inputs["color_r/0"])
      outputs = self.stereo_net(inputs["color_l/0"], left_feat, right_feat, "l",
                                store_feature_gradients=False, store_refinement_gradients=False,
                                do_refinement=True, output_cost_volume=False)

      # NOTE(milo): Without this synchronize, timing can be misleading. The GPU stuff will happen
      # asynchronously and the copy back to CPU will appear to take a really long time later.
      torch.cuda.synchronize()

      elapsed = time.time() - t0
      print("Total inference time = %f sec" % elapsed)

      # Publish with a timestamp that matches the input images.
      t0 = time.time()

      outputs["pred_disp_l/{}".format(min(2, Config.VOXEL_DISP_SCALE))] = F.interpolate(
          outputs["pred_disp_l/0"], scale_factor=0.25, align_corners=False, mode="bilinear")

      # NOTE: Even for lower resolutions, still want to use at least R2 disparity, since the
      # network output will be much better at R2.
      disp = outputs["pred_disp_l/{}".format(min(2, Config.VOXEL_DISP_SCALE))].squeeze(0)

      # TODO(milo): Figure out why publishing an image is so slow.
      # For now, publish infrequently to speed things up.
      if (time.time() - self.last_publish_disp_time) > (1.0 / Config.PUBLISH_DISP_HZ):
        disp_msg = disp_tensor_to_image_msg(outputs["pred_disp_l/2"].squeeze(0), left_img_msg.header.stamp, self.bridge)
        self.disp_pubs[2].publish(disp_msg)
        self.last_publish_disp_time = time.time()

      depth = (self.K_voxel[0,0] * Config.STEREO_BASELINE_METERS) / disp
      depth = torch.clamp(depth, 0.0, Config.MAX_DEPTH)

      # NOTE: Open3D uses uint16 to store depth values, so we scale by a large number (i.e 100) to
      # keep precision when casting from float to uint16.
      scaled_depth_o3d = (Config.DEPTH_SCALE_FACTOR * depth.squeeze(0)).cpu().numpy().astype(np.uint16)
      im_depth_o3d = o3d.geometry.Image(scaled_depth_o3d) # Expects a 16-bit, 1-channel depth image.

      if Config.PUBLISH_COLOR_POINT_CLOUD:
        # Make Open3D color and depth images.
        # NOTE: Using np.moveaxis here caused problems! I think it's something to do with contiguous arrays...
        scaled_color_o3d = F.interpolate(left_img_t.unsqueeze(0), size=(self.height_voxel_scale, self.width_voxel_scale), mode="bilinear")
        scaled_color_o3d = (255.0 * scaled_color_o3d.squeeze(0).permute(1, 2, 0))
        scaled_color_o3d = scaled_color_o3d.byte().cpu().numpy()
        im_color_o3d = o3d.geometry.Image(scaled_color_o3d) # Expects an 8-bit, 3-channel RGB image, shape (3, h, w).

        rgbd_image = o3d.geometry.create_rgbd_image_from_color_and_depth(
            im_color_o3d, im_depth_o3d, depth_scale=Config.DEPTH_SCALE_FACTOR,
            depth_trunc=80.0, convert_rgb_to_intensity=False)

        pointcloud = o3d.geometry.create_point_cloud_from_rgbd_image(rgbd_image, self.K_o3d)

      else:
        pointcloud = o3d.geometry.create_point_cloud_from_depth_image(im_depth_o3d, self.K_o3d, depth_scale=Config.DEPTH_SCALE_FACTOR)

      pointcloud = o3d.geometry.voxel_down_sample(pointcloud, Config.VOXEL_SCALE_METERS)

      # Crop the point cloud to reduce the amount of data that needs to be published.
      # min_pt_cam = np.array([-5, -5, 0]).reshape((3, 1))
      # max_pt_cam = np.array([5, 1, 20]).reshape((3, 1))
      # pointcloud = o3d.geometry.crop_point_cloud(pointcloud, min_pt_cam, max_pt_cam)

      # o3d.visualization.draw_geometries([pointcloud])
      # NOTE: The point cloud will have the same timestamp as the images that were used to build it.
      pointcloud_msg = convert_pointcloud_open3d_to_ros(pointcloud, "stereo_camera_left", left_img_msg.header.stamp)

      self.voxel_pub.publish(pointcloud_msg)
      elapsed = time.time() - t0
      print("Total voxel process time = %f sec" % elapsed)


if __name__ == "__main__":
  node = StereoDepthNode()
  rospy.init_node("stereo_depth_node", anonymous=True)
  print("[StereoDepthNode]", "Created stereo_depth_node")
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down stereo_depth_node")

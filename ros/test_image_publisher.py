import rospy
import cv2 as cv
import numpy as np

from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped, Quaternion, Point
from cv_bridge import CvBridge
import tf2_ros
import tf2_msgs.msg

from .config import Config
from datasets.stereo_dataset import StereoDataset
from utils.dataset_utils import read_lines

from torchvision import transforms
from scipy.spatial.transform import Rotation as R
import PIL


def get_crop_dims(h, w, height, width):
  aspect = w / h
  target_aspect = width / height

  # Crop the sides.
  if aspect > target_aspect:
    new_width = int(target_aspect * h)
    offset = (w - new_width) // 2
    resize = (offset, 0, w - offset, h)
  # Crop the top and bottom.
  else:
    new_height = int(w / target_aspect)
    offset = (h - new_height) // 2
    resize = (0, offset, w, h - offset)

  return resize


def resize_image(img, height, width):
  """
  Resize images to the target height and width. Does a center crop to achieve the desired aspect
  ratio and then resizes to the target size.

  https://stackoverflow.com/questions/4744372/reducing-the-width-height-of-an-image-to-fit-a-given-aspect-ratio-how-python
  """
  resize = get_crop_dims(img.height, img.width, height, width)
  cropped = img.crop(resize).resize((width, height), PIL.Image.ANTIALIAS)
  return cropped


def tensor_to_image_msg(t):
  """
  The image tensor will have shape (c, h, w) and float range [0, 1]. Need to convert to shape
  (h, w, c) and uint8.

  NOTE: Expects the channels to be in RGB order.
  """
  scaled_im = (255 * t.permute(1, 2, 0)).cpu().numpy().astype(np.uint8)
  return CvBridge().cv2_to_imgmsg(scaled_im, encoding="rgb8")


class TestImagePublisher(object):
  """
  Run from the directory above with Python3:
    python -m ros.test_image_publisher
  """
  def __init__(self, hz=1):
    self.left_pub = rospy.Publisher(Config.LEFT_IMAGE_CHANNEL, Image, queue_size=1)
    self.right_pub = rospy.Publisher(Config.RIGHT_IMAGE_CHANNEL, Image, queue_size=1)
    # self.left_pose_pub = rospy.Publisher(Config.LEFT_CAMERA_POSE_CHANNEL, TransformStamped, queue_size=1)
    # self.right_pose_pub = rospy.Publisher(Config.RIGHT_CAMERA_POSE_CHANNEL, TransformStamped, queue_size=1)
    self.pose_pub = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=1)
    self.hz = hz
    self.bridge = CvBridge()

  def run(self):
    rate = rospy.Rate(self.hz)
    while not rospy.is_shutdown():
      self.left_pub.publish(self.left_msg)
      self.right_pub.publish(self.right_msg)
      rate.sleep()

  def run_sequence(self, dataset_name, dataset_path, split_name):
    lines = read_lines("splits/{}/train_lines.txt".format(split_name))
    dataset = StereoDataset(dataset_path, dataset_name, split_name, Config.MODEL_INPUT_HEIGHT, Config.MODEL_INPUT_WIDTH, "train")
    rate = rospy.Rate(self.hz)

    while not rospy.is_shutdown():
      for i, inputs in enumerate(dataset):
        if rospy.is_shutdown():
          return

        timestamp_now = rospy.Time.now()

        left_msg = tensor_to_image_msg(inputs["color_l/0"])
        right_msg = tensor_to_image_msg(inputs["color_r/0"])
        left_msg.header.stamp = timestamp_now
        left_msg.header.frame_id = "stereo_camera_left"
        left_msg.header.seq = i

        right_msg.header.stamp = timestamp_now
        right_msg.header.frame_id = "stereo_camera_right"
        right_msg.header.seq = i

        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)

        # TODO: Add camera pose to StereoDataset inputs.
        # Make groundtruth pose messages (camera in world).
        # left_pose = TransformStamped()
        # left_pose.header.stamp = timestamp_now
        # left_pose.header.frame_id = "world"
        # left_pose.child_frame_id = "stereo_camera_left"
        # left_pose.header.seq = i
        # xyz = inputs["camera_pose_l"][:3,3].numpy()
        # left_pose.transform.translation = Point(xyz[0], xyz[1], xyz[2])
        # xyzw = R.from_dcm(inputs["camera_pose_l"][:3,:3].numpy()).as_quat()
        # xyzw /= np.linalg.norm(xyzw)
        # left_pose.transform.rotation = Quaternion(xyzw[0], xyzw[1], xyzw[2], xyzw[3]) # NOTE: Newer scipy version uses "from_matrix"

        # right_pose = TransformStamped()
        # right_pose.header.stamp = timestamp_now
        # right_pose.header.frame_id = "world"
        # right_pose.child_frame_id = "stereo_camera_right"
        # right_pose.header.seq = i
        # xyz = inputs["camera_pose_r"][:3,3].numpy()
        # right_pose.transform.translation = Point(xyz[0], xyz[1], xyz[2])
        # xyzw = R.from_dcm(inputs["camera_pose_r"][:3,:3].numpy()).as_quat()
        # xyzw /= np.linalg.norm(xyzw)
        # right_pose.transform.rotation = Quaternion(xyzw[0], xyzw[1], xyzw[2], xyzw[3]) # NOTE: Newer scipy version uses "from_matrix"

        # assert(right_pose.transform.rotation.w == xyzw[3])

        # wiki.ros.org/tf2/Tutorials/Adding a frame (Python)
        # tfm = tf2_msgs.msg.TFMessage([left_pose, right_pose])
        # self.pose_pub.publish(tfm)

        rate.sleep()


if __name__ == "__main__":
  dataset_name = "VirtualKitti"
  split_name = "virtual_kitti_clone"
  dataset_path = "/home/milo/datasets/virtual_kitti/"

  rospy.init_node("test_image_publisher", anonymous=True)
  node = TestImagePublisher(hz=5)
  print("[TestImagePublisher]", "Created test_image_publisher")
  try:
    node.run_sequence(dataset_name, dataset_path, split_name)
  except KeyboardInterrupt:
    print("Shutting down test_image_publisher")

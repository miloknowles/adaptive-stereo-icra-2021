# Source: https://github.com/felixchenfy/open3d_ros_pointcloud_conversion/blob/master/lib_cloud_conversion_between_Open3D_and_ROS.py

import open3d
import numpy as np
from ctypes import * # convert float to uint32

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

# NOTE(milo): 8 hex digits = 32 bit digits

# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
  PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
  PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
  PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]

FIELDS_XYZRGB = FIELDS_XYZ + \
  [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

# Bit operations
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8

# NOTE(milo): Looks like only 24 bits of the 32 are actually used for RGB? 8 bits for each color?
convert_rgbUint32_to_tuple = lambda rgb_uint32: (
  (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
)

convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
  int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)


def convert_pointcloud_open3d_to_ros(open3d_cloud, frame_id, stamp=None):
  """
  Make a ROS sensor_msgs::PointCloud2 from an Open3D point cloud.

  open3d_cloud (o3d.PointCloud) : http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html
  frame_id (str) : The frame that this point cloud is in.
  stamp (rospy.Time) : Could be the current time OR the timestamp on images used to build the pointcloud.
  """
  header = Header()
  header.stamp = stamp if stamp is not None else rospy.Time.now()
  header.frame_id = frame_id

  points_xyz = np.asarray(open3d_cloud.points)

  if open3d_cloud.colors:
    # Make an Nx3 matrix of colors.
    colors = np.floor(255*np.asarray(open3d_cloud.colors))
    # Convert colors from 3 floats to one 24-byte int.
    # 0x00FFFFFF is white, 0x00000000 is black.
    colors = colors[:,0]*BIT_MOVE_16 + colors[:,1]*BIT_MOVE_8 + colors[:,2]  
    cloud_data=np.c_[points_xyz, colors]
    return pc2.create_cloud(header, FIELDS_XYZRGB, cloud_data)
  else:
    fields=FIELDS_XYZ
    return pc2.create_cloud_xyz32(header, points_xyz)


def convert_pointcloud_ros_to_open3d(ros_cloud):
  # Get cloud data from ros_cloud
  field_names=[field.name for field in ros_cloud.fields]
  cloud_data = list(pc2.read_points(ros_cloud, skip_nans=True, field_names = field_names))

  # Check empty
  open3d_cloud = open3d.PointCloud()
  if len(cloud_data)==0:
    print("Converting an empty cloud")
    return None

  # Set open3d_cloud
  if "rgb" in field_names:
    IDX_RGB_IN_FIELD=3 # x, y, z, rgb
    
    # Get xyz
    xyz = [(x,y,z) for x,y,z,rgb in cloud_data ] # (why cannot put this line below rgb?)

    # Get rgb
    # Check whether int or float
    if type(cloud_data[0][IDX_RGB_IN_FIELD])==float: # if float (from pcl::toROSMsg)
      rgb = [convert_rgbFloat_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
    else:
      rgb = [convert_rgbUint32_to_tuple(rgb) for x,y,z,rgb in cloud_data ]

    # combine
    open3d_cloud.points = open3d.Vector3dVector(np.array(xyz))
    open3d_cloud.colors = open3d.Vector3dVector(np.array(rgb)/255.0)
  else:
    xyz = [(x,y,z) for x,y,z in cloud_data ] # get xyz
    open3d_cloud.points = open3d.Vector3dVector(np.array(xyz))

  # return
  return open3d_cloud
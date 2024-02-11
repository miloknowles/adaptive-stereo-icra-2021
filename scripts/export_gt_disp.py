# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os, glob

import numpy as np

from collections import Counter


def readlines(filename):
  """
  Read all the lines in a text file and return as a list.
  """
  with open(filename, 'r') as f:
    lines = f.read().splitlines()
  return lines


def load_velodyne_points(filename):
  """
  Load 3D point cloud from KITTI file format
  (adapted from https://github.com/hunse/kitti)
  """
  points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
  points[:, 3] = 1.0  # homogeneous
  return points


def read_calib_file(path):
  """
  Read KITTI calibration file
  (from https://github.com/hunse/kitti)
  """
  float_chars = set("0123456789.e+- ")
  data = {}
  with open(path, 'r') as f:
    for line in f.readlines():
      key, value = line.split(':', 1)
      value = value.strip()
      data[key] = value
      if float_chars.issuperset(value):
        # try to cast to float array
        try:
          data[key] = np.array(list(map(float, value.split(' '))))
        except ValueError:
          # casting error: data[key] already eq. value, so pass
          pass

  return data


def sub2ind(matrixSize, rowSub, colSub):
  """
  Convert row, col matrix subscripts to linear indices
  """
  m, n = matrixSize
  return rowSub * (n-1) + colSub - 1


def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
  """
  Generate a depth map from velodyne data
  """
  # load calibration files
  cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
  velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
  velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
  velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

  # get image shape
  im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

  # compute projection matrix velodyne->image plane
  R_cam2rect = np.eye(4)
  R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
  P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
  P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

  # load velodyne points and remove all behind image plane (approximation)
  # each row of the velodyne data is forward, left, up, reflectance
  velo = load_velodyne_points(velo_filename)
  velo = velo[velo[:, 0] >= 0, :]

  # project the points to the camera
  velo_pts_im = np.dot(P_velo2im, velo.T).T
  velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

  if vel_depth:
    velo_pts_im[:, 2] = velo[:, 0]

  # check if in bounds
  # use minus 1 to get the exact same value as KITTI matlab code
  velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
  velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
  val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
  val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
  velo_pts_im = velo_pts_im[val_inds, :]

  # project to image
  depth = np.zeros((im_shape[:2]))
  depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

  # find the duplicate points and choose the closest depth
  inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
  dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
  for dd in dupe_inds:
    pts = np.where(inds == dd)[0]
    x_loc = int(velo_pts_im[pts[0], 0])
    y_loc = int(velo_pts_im[pts[0], 1])
    depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
  depth[depth < 0] = 0

  return depth


def export_gt_disp(dataset_path, cleanup_old=True):
  """
  Compute groundtruth disparity maps for the KITTI Raw Dataset:

  Creates a disp_02/ folder alongside image_02, image_03 etc. which contains disparity images
  projected into the left RGB camera.
  """
  imgs_left = glob.glob(os.path.join(dataset_path, "*/*/image_02/*/*.jpg"), recursive=True)
  imgs_right = glob.glob(os.path.join(dataset_path, "*/*/image_03/*/*.jpg"), recursive=True)

  print("Found {} left images and {} right images".format(len(imgs_left), len(imgs_right)))
  assert(len(imgs_left) == len(imgs_right))

  for ii, im in enumerate(imgs_left):
    if ii % 100 == 0:
      print("Finished {}/{} images".format(ii, len(imgs_left)))
    velo = im.replace("image_02", "velodyne_points").replace(".jpg", ".bin")

    if not os.path.exists(velo):
      with open("./no_groundtruth.txt", "a") as f:
        print("WARNING: Had to skip {} because no velodyne file was found".format(im))
        f.write(im + "\n")
      continue

    index_of_folder = im.find("kitti_data_raw") + len("kitti_data_raw/") + 10
    calib_dir = im[:index_of_folder]

    for camera_index in [2, 3]:
      gt_depth = generate_depth_map(calib_dir, velo, camera_index, True).astype(np.float32)

      # Convert depth back to disparity using d = bf / z.
      cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
      P_rect = cam2cam['P_rect_02'].reshape(3, 4)
      fx = P_rect[0, 0]
      baseline_meters = 0.54 # Baseline is 54cm according to devkit.

      gt_disp = baseline_meters * fx / gt_depth
      gt_disp[gt_depth == 0] = 0
      gt_disp[gt_depth > 80] = 0

      # Make sure we can cast to uint16 without overflow.
      assert((128.0 * gt_disp.max()) <= 65535)
      gt_disp = (128.0 * gt_disp).astype(np.uint16)

      # Save a groundtruth disparity image with the same frame_id as image.
      disp_path = im.replace("image_02", "disp_0{}".format(camera_index)).replace(".jpg", ".npy")
      disp_folder = os.path.abspath(os.path.join(disp_path, ".."))

      # Optionally clean out old disparity files that were saved.
      if cleanup_old:
        files_not_in_data = glob.glob(os.path.join(disp_folder, "./../*.npy"), recursive=False)
        if len(files_not_in_data) > 0:
          print("Found {} existing files not in data/, deleting".format(len(files_not_in_data)))
          for f in files_not_in_data: os.remove(f)

      os.makedirs(disp_folder, exist_ok=True)
      np.save(disp_path, gt_disp)

      # print("Saved {}".format(disp_path))


if __name__ == "__main__":
  export_gt_disp("/home/milo/datasets/kitti_data_raw/")

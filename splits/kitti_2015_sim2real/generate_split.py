import os, glob


def generate_split():
  path_to_kitti = "/home/milo/datasets/kitti_stereo_2015/training/"

  lines = []
  for i in range(200):
    rgb_l = os.path.join(path_to_kitti, "image_2/{:06d}_10.png".format(i))
    rgb_r = os.path.join(path_to_kitti, "image_3/{:06d}_10.png".format(i))
    disp_l = os.path.join(path_to_kitti, "disp_occ_0/{:06d}_10.png".format(i))
    disp_r = os.path.join(path_to_kitti, "disp_occ_1/{:06d}_10.png".format(i))
    lines.append("{} {} {} {}".format(rgb_l, rgb_r, disp_l, disp_r))

  with open("train_lines.txt", "w") as f:
    for l in lines:
      f.write(l.replace("/home/milo/datasets/kitti_stereo_2015/", "") + "\n")


if __name__ == "__main__":
  generate_split()

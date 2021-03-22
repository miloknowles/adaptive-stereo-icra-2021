import os, glob


def generate_split():
  lines = []

  scene = "Scene01"
  seq = "clone"

  folder = os.path.join("/home/milo/datasets/virtual_kitti/{}/{}/frames/".format(scene, seq))

  rgb_l = os.path.join(folder, "rgb", "Camera_0")
  rgb_r = os.path.join(folder, "rgb", "Camera_1")
  disp_l = os.path.join(folder, "depth", "Camera_0")
  disp_r = os.path.join(folder, "depth", "Camera_1")

  rgb_l_files = sorted(glob.glob(rgb_l + "/*.jpg"))
  rgb_r_files = sorted(glob.glob(rgb_r + "/*.jpg"))
  disp_l_files = sorted(glob.glob(disp_l + "/*.png"))
  disp_r_files = sorted(glob.glob(disp_r + "/*.png"))

  for i in range(len(rgb_l_files)):
    lines.append("{} {} {} {}".format(rgb_l_files[i], rgb_r_files[i], disp_l_files[i], disp_r_files[i]))

  with open("train_lines.txt", "w") as f:
    for l in lines:
      f.write(l.replace("/home/milo/datasets/virtual_kitti/", "") + "\n")


if __name__ == "__main__":
  generate_split()

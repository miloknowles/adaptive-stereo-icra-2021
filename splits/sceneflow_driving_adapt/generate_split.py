import os, random, glob
import numpy as np


def write_lines(dataset_path, focal_folder, val_split=0.15, test_split=0.15):
  random.seed(123)
  np.random.seed(123)

  # NOTE(milo): Sceneflow driving has 800 images in the forward/backward sequences.
  lines = []
  direction = "scene_forwards"
  speed = "slow"

  rgb_l = os.path.join(dataset_path, "frames_cleanpass/{}/{}/{}/left/".format(focal_folder, direction, speed))
  rgb_r = os.path.join(dataset_path, "frames_cleanpass/{}/{}/{}/right/".format(focal_folder, direction, speed))
  disp_l = os.path.join(dataset_path, "disparity/{}/{}/{}/left/".format(focal_folder, direction, speed))
  disp_r = os.path.join(dataset_path, "disparity/{}/{}/{}/right/".format(focal_folder, direction, speed))

  rgb_l = sorted(glob.glob(os.path.join(rgb_l, "*.png")))
  rgb_r = sorted(glob.glob(os.path.join(rgb_r, "*.png")))
  disp_l = sorted(glob.glob(os.path.join(disp_l, "*.pfm")))
  disp_r = sorted(glob.glob(os.path.join(disp_r, "*.pfm")))

  assert(len(rgb_l) == len(rgb_r) and len(rgb_r) == len(disp_l) and len(disp_l) == len(disp_r))

  for i in range(len(rgb_l)):
    lines.append(" ".join([rgb_l[i], rgb_r[i], disp_l[i], disp_r[i]]))

  for subsplit in ["train", "val"]:
    with open("./{}_lines.txt".format(subsplit), "w") as f:
      for l in lines:
        l = l.replace(dataset_path, "")
        f.write(l + "\n")
      print("Wrote {}_lines.txt with {} lines".format(subsplit, len(lines)))


if __name__ == "__main__":
  dataset_path = "/home/milo/datasets/sceneflow_driving/"
  focal_folder = "35mm_focallength"
  write_lines(dataset_path, focal_folder)

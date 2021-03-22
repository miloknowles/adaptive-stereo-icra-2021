import os, random, glob
import numpy as np


def write_lines(dataset_path, focal_folder, val_split=0.15, test_split=0.15):
  random.seed(123)
  np.random.seed(123)

  # NOTE(milo): Sceneflow driving has 800 images in the forward/backward sequences.
  lines = []
  for direction in ["scene_forwards", "scene_backwards"]:
    for speed in ["fast", "slow"]:
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

  # Shuffle the lines around to get random sets.
  indices_shuffled = np.arange(len(lines))
  np.random.shuffle(indices_shuffled)
  lines_shuffled = list(map(lambda i: lines[i], indices_shuffled))

  N_train = int((1.0 - val_split - test_split) * len(lines_shuffled))
  N_val = int(val_split * len(lines_shuffled))

  lines_dict = {
    "train": lines_shuffled[:N_train],
    "val": lines_shuffled[N_train:N_train+N_val],
    "test": lines_shuffled[N_train+N_val:]
  }

  for split_name, split_lines in lines_dict.items():
    with open("./{}_lines.txt".format(split_name), "w") as f:
      for l in split_lines:
        l = l.replace(dataset_path, "")
        f.write(l + "\n")
      print("Wrote {}_lines.txt with {} lines".format(split_name, len(split_lines)))


if __name__ == "__main__":
  dataset_path = "/home/milo/datasets/sceneflow_driving/"
  focal_folder = "35mm_focallength"
  write_lines(dataset_path, focal_folder)

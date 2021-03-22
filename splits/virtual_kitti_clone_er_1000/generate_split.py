import os, random
import numpy as np


if __name__ == "__main__":
  random.seed(123)
  np.random.seed(123)
  path_to_train_lines = "../virtual_kitti_clone_aug/train_lines.txt"

  with open(path_to_train_lines, "r") as f:
    train_lines = [l for l in f]

  # Shuffle the lines around to get random sets.
  indices_shuffled = np.arange(len(train_lines))
  np.random.shuffle(indices_shuffled)
  lines_shuffled = list(map(lambda i: train_lines[i], indices_shuffled))

  er_lines = lines_shuffled[:1000]

  with open("val_lines.txt", "w") as f:
    for l in er_lines:
      f.write(l)

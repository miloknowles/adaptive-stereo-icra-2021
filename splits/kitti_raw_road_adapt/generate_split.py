import os, glob

# http://www.cvlibs.net/datasets/kitti/raw_data.php
SEQUENCE_NAMES = [
  "2011_09_26_drive_0015_sync",
  "2011_09_26_drive_0027_sync",
  "2011_09_26_drive_0028_sync",
  "2011_09_26_drive_0029_sync",
  "2011_09_26_drive_0032_sync",
  "2011_09_26_drive_0052_sync",
  "2011_09_26_drive_0070_sync",
  "2011_09_26_drive_0101_sync",
  "2011_09_29_drive_0004_sync",
  "2011_09_30_drive_0016_sync",
  "2011_10_03_drive_0042_sync",
  "2011_10_03_drive_0047_sync",
]


def generate_split(dataset_path):
  lines = []
  for sequence_name in SEQUENCE_NAMES:
    # Get the part of the name before _drive.
    print("Processing sequence:", sequence_name)
    sequence_folder = os.path.join(sequence_name[:10], sequence_name)
    image_02_folder = os.path.join(dataset_path, sequence_folder, "image_02/data/")
    image_03_folder = os.path.join(dataset_path, sequence_folder, "image_03/data/")
    disp_02_folder = os.path.join(dataset_path, sequence_folder, "disp_02/data/")
    disp_03_folder = os.path.join(dataset_path, sequence_folder, "disp_03/data/")

    rgb_l = sorted(glob.glob(os.path.join(image_02_folder, "*.jpg")))
    rgb_r = sorted(glob.glob(os.path.join(image_03_folder, "*.jpg")))
    disp_l = sorted(glob.glob(os.path.join(disp_02_folder, "*.npy")))
    disp_r = sorted(glob.glob(os.path.join(disp_03_folder, "*.npy")))
    assert(len(rgb_l) == len(rgb_r) and len(rgb_r) == len(disp_l) and len(disp_l) == len(disp_r))

    for i in range(len(rgb_l)):
      lines.append(" ".join([rgb_l[i], rgb_r[i], disp_l[i], disp_r[i]]))

  for split_name in ["train", "val"]:
    with open("{}_lines.txt".format(split_name), "w") as f:
      for line in lines:
        f.write(line + "\n")


if __name__ == "__main__":
  generate_split("/home/milo/datasets/kitti_data_raw/")

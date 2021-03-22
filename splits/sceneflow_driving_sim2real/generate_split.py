import os, glob


def generate_split():
  lines = []
  template_path = "/home/milo/datasets/sceneflow_driving/{}/35mm_focallength/{}/{}/{}/"

  for direction in ["scene_forwards", "scene_backwards"]:
    for speed in ["fast", "slow"]:
      rgb_l = template_path.format("frames_cleanpass", direction, speed, "left")
      rgb_r = template_path.format("frames_cleanpass", direction, speed, "right")
      disp_l = template_path.format("disparity", direction, speed, "left")
      disp_r = template_path.format("disparity", direction, speed, "right")

      rgb_l_files = sorted(glob.glob(rgb_l + "/*.png"))
      rgb_r_files = sorted(glob.glob(rgb_r + "/*.png"))
      disp_l_files = sorted(glob.glob(disp_l + "/*.pfm"))
      disp_r_files = sorted(glob.glob(disp_r + "/*.pfm"))

      for i in range(len(rgb_l_files)):
        lines.append("{} {} {} {}".format(rgb_l_files[i], rgb_r_files[i], disp_l_files[i], disp_r_files[i]))

  with open("train_lines.txt", "w") as f:
    for l in lines:
      f.write(l + "\n")


if __name__ == "__main__":
  generate_split()

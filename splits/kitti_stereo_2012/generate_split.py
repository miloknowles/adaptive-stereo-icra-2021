import os
import numpy as np


def write_lines():
  """
  200 training images: (000000_10.png - 000199_10.png)
  200 testing images: (000000_10.png - 000199_10.png)

  NOTE: Only the images with the _10 suffix have corresponding groundtruth disparity. I think the
  _11 images have flow since they are temporal pairs to the _10 images?
  """
  for subsplit in ["training", "testing"]:
    with open("train_lines.txt" if subsplit == "training" else "test_lines.txt", "w") as f:
      for i in range(194):
        rgb_l = "{}/colored_0/{:06d}_10.png".format(subsplit, i)
        rgb_r = "{}/colored_1/{:06d}_10.png".format(subsplit, i)
        disp_l = "{}/disp_occ/{:06d}_10.png".format(subsplit, i)
        disp_r = "None"
        f.write(" ".join([rgb_l, rgb_r, disp_l, disp_r]) + "\n")
    print("Finished writing lines for split {}".format(subsplit))


if __name__ == "__main__":
  write_lines()

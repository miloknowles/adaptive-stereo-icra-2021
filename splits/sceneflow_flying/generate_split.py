import os

def generate_split():
  for subsplit in ["train", "val", "test"]:
    path_to_old = "../../graveyard/splits/sceneflow_flying/{}_lines.txt".format(subsplit)

    with open("{}_lines.txt".format(subsplit), "w") as fout:
      with open(path_to_old, "r") as fin:
        for line in fin:
          tt, letter, seq, i = line.split(" ")
          seq = int(seq)
          i = int(i)
          rgb_l = "frames_cleanpass/{}/{}/{:04d}/left/{:04d}.png".format(tt, letter, seq, i)
          rgb_r = "frames_cleanpass/{}/{}/{:04d}/right/{:04d}.png".format(tt, letter, seq, i)
          disp_l = "disparity/{}/{}/{:04d}/left/{:04d}.pfm".format(tt, letter, seq, i)
          disp_r = "disparity/{}/{}/{:04d}/right/{:04d}.pfm".format(tt, letter, seq, i)
          fout.write(" ".join([rgb_l, rgb_r, disp_l, disp_r]) + "\n")


if __name__ == "__main__":
  generate_split()

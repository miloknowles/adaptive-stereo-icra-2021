# Copyright 2020 Massachusetts Institute of Technology
#
# @file split_pretrained_weights.py
# @author Milo Knowles
# @date 2020-08-09 10:53:15 (Sun)

import argparse, os
import torch


def split_and_save(opt):
  d = torch.load(opt.pth)
  f, s = {}, {}
  for key in d:
    # Ignore some legacy keys.
    if key in ["height", "width", "radius_disp"]:
      continue
    skey = key.replace("feature_extractor.", "")
    if "feature_extractor" in key:
      f[skey] = d[key]
    else:
      s[key] = d[key]

  for key in ["height", "width", "radius_disp"]:
    if key in d:
      del d[key]

  output_folder = os.path.dirname(opt.pth)
  # torch.save(f, os.path.join(output_folder, "feature_net.pth"))
  # torch.save(s, os.path.join(output_folder, "stereo_net.pth"))
  torch.save(d, os.path.join(output_folder, "model_strict.pth"))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Splits old madnet.pth files into a feature_net.pth and stereo_net.pth")
  parser.add_argument("--pth", type=str, help="Path to the .pth file (i.e /path/to/madnet.pth)")
  opt = parser.parse_args()
  split_and_save(opt)

# Copyright 2020 Massachusetts Institute of Technology
#
# @file print_adaptation_trials.py
# @author Milo Knowles
# @date 2020-07-30 19:10:50 (Thu)

import argparse, math
import pandas as pd


_METRICS = [
  "D1_all_3px_TRAIN",
  "D1_all_3px_ADAPT",
  "EPE_TRAIN",
  "EPE_ADAPT",
  "FCS_TRAIN",
  "FCS_ADAPT",
  "all"
]


def print_average(opt):
  df = pd.read_csv(opt.csv)
  print("Found {} trials".format(df["trial"].max() + 1))

  for step in opt.steps:
    if opt.trial is not None:
      metrics_this_step = df[(df["step"] == step) & (df["trial"] == opt.trial)]
    else:
      metrics_this_step = df[df["step"] == step]
    metrics_avg = metrics_this_step.mean(axis=0).to_frame().T
    metrics_std = metrics_this_step.std(axis=0).to_frame().T
    metrics_min = metrics_this_step.min(axis=0).to_frame().T
    metrics_max = metrics_this_step.max(axis=0).to_frame().T
    print("STEP {}:".format(step))
    for colname in _METRICS:
      if colname in metrics_avg.columns:
        ci = 1.96 * metrics_std[colname][0] / math.sqrt(metrics_this_step["trial"].max() + 1)
        print("  {} avg={:.03f} 95CI=(+/-) {:.03f} min={:.03f} max={:.03f}".format(
            colname, metrics_avg[colname][0], ci, metrics_min[colname][0], metrics_max[colname][0]))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--csv", type=str, help="Path to trials.csv file")
  parser.add_argument("--steps", type=int, nargs="+", default=[-1, 1000, 2000, 3000, 4000])
  parser.add_argument("--trial", type=int, default=None, help="Optionally specify a specific trial")
  opt = parser.parse_args()
  print_average(opt)

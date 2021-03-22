# Copyright 2020 Massachusetts Institute of Technology
#
# @file preprocess_adapt_results.py
# @author Milo Knowles
# @date 2020-10-08 16:43:08 (Thu)

import os
import pandas as pd

from utils.path_utils import *


if __name__ == "__main__":
  method_names = ["nonstop", "vs", "er", "vs+er"]
  methods_names_map = {"nonstop": "MAD-FULL", "vs": "VS", "er": "ER", "vs+er": "VS + ER"}

  experiment_names = [
    "clone_to_campus",
    "clone_to_city",
    "clone_to_fog",
    "clone_to_rain",
    "flying_to_campus",
    "flying_to_city",
    "flying_to_road",
    "flying_to_vk01",
    "flying_to_vk20",
  ]

  for en in experiment_names:
    print("\n-------------------------------------------------------------------")
    print("Processing results for:", en)
    df_plot_exp = {"Method": [], "Step": [], "EPE": [], "FCS": [], "Domain": [], "GradientUpdates": []}

    for mn in method_names:
      print("  ", methods_names_map[mn])
      path_to_csv = os.path.join("/home/milo/training_logs", "adapt_{}_{}/trials.csv".format(en, mn))

      if not os.path.exists(path_to_csv):
        raise FileNotFoundError("Couldn't find {}, make sure that experiment was run".format(path_to_csv))

      df = pd.read_csv(path_to_csv, header=0)

      # Make sure only a single trial was run.
      assert((df["trial"].nunique() == 1))

      # NOTE: To save time running the experiments, we only do a pre-adaptation evaluation for the
      # Nonstop (MAD-FULL) method, and skip it for subsequent methods.
      steps = [1000, 2000, 3000, 4000]
      if mn == "nonstop":
        steps.insert(0, -1)

      for step in steps:
        for train_or_adapt in ["TRAIN", "ADAPT"]:
          mn_legend = methods_names_map[mn]

          df_plot_exp["Method"].append(mn_legend)
          df_plot_exp["Step"].append(step)

          EPE = df[df["step"] == step]["EPE_{}".format(train_or_adapt)].iloc[0]
          FCS = df[df["step"] == step]["FCS_{}".format(train_or_adapt)].iloc[0]
          gradient_updates = df[df["step"] == step]["GRADIENT_UPDATES"].iloc[0]

          df_plot_exp["EPE"].append(EPE)
          df_plot_exp["FCS"].append(FCS)
          df_plot_exp["Domain"].append(train_or_adapt)
          df_plot_exp["GradientUpdates"].append(gradient_updates)

    df_plot_exp = pd.DataFrame(df_plot_exp)

    output_folder = path_to_output(reldir="adapt_results/{}".format(en))
    os.makedirs(output_folder, exist_ok=True)
    df_plot_exp.to_csv(os.path.join(output_folder, "results.csv"), index=False)
    print("Saved results.csv to", output_folder)

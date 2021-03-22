# Copyright 2020 Massachusetts Institute of Technology
#
# @file make_adaptation_plots.py
# @author Milo Knowles
# @date 2020-08-21 10:15:43 (Fri)

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from utils.path_utils import *

matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def parse_csv(path_to_csv):
  df = pd.read_csv(
      path_to_csv,
      skiprows=2,
      names=["Strategy", "EPE_T_0", "EPE_A_0", "EPE_T_1000", "EPE_A_1000",
             "EPE_T_2000", "EPE_A_2000", "EPE_T_3000", "EPE_A_3000",
             "EPE_T_4000", "EPE_A_4000", "GradientUpdates"])
  return df


def reorganize_df(df, sequence_name, strategies=["Nonstop", "VS", "ER", "VS + ER"]):
  """
  Helper function to reorganize the adaptation df so that it's compatible with Seaborn.
  """
  # First make a df that's compatible with seaborn.
  df_plot = {"Strategy": [], "Step": [], "EPE": [], "Domain": [], "GradientUpdates": []}

  start_row = None
  for i, entry in enumerate(df["Strategy"]):
    if sequence_name in entry:
      start_row = i
      break

  if start_row is None:
    raise ValueError("Could not find sequence name {}".format(sequence_name))

  df = df.iloc[i:i+5,:]

  row_index_for_strategy = {"Nonstop": 0, "VS": 1, "ER": 2, "VS + ER": 3}

  for strategy in strategies:
    for step in [0, 1000, 2000, 3000, 4000]:
      for domain in ["EPE_T", "EPE_A"]:
        i = row_index_for_strategy[strategy]
        row_this_strategy = df.iloc[i+1,:]
        assert(row_this_strategy["Strategy"] == strategy)
        df_plot["Strategy"].append("MAD-FULL" if strategy == "Nonstop" else strategy)
        df_plot["Step"].append(step)
        df_plot["EPE"].append(row_this_strategy["{}_{}".format(domain, str(step))])
        df_plot["Domain"].append("Train" if domain == "EPE_T" else "Novel")
        df_plot["GradientUpdates"].append(row_this_strategy["GradientUpdates"])

  df_plot = pd.DataFrame(df_plot)
  return df_plot


def lineplots_adaptation(df_plot, sequence_name, show_legend=False):
  matplotlib.rcParams['font.size'] = 26
  matplotlib.rcParams['lines.markersize'] = 12
  matplotlib.rcParams['figure.figsize'] = 8, 6

  plt.clf()
  df_plot = df_plot[df_plot["Strategy"].isin(["MAD-FULL", "VS + ER"])]

  palette = {"MAD-FULL": "tab:blue", "VS + ER": "tab:red"}
  ax = sns.lineplot(x="Step", y="EPE", hue="Strategy", style="Domain", data=df_plot,
                    legend="full" if show_legend else False, palette=palette, markers=True)
  plt.xlabel("adaptation step")
  plt.ylabel("end-point-error (EPE)")
  plt.xticks([0, 1000, 2000, 3000, 4000])

  if show_legend:
    plt.legend(loc="upper right", fontsize="small")
  ax.grid(True, which='both')

  fig_name = sequence_name.replace(" ", "_") + "_lineplot.pdf"
  fig_path = path_to_output(reldir="lineplots")
  os.makedirs(fig_path, exist_ok=True)
  plt.savefig(os.path.join(fig_path, fig_name), bbox_inches="tight")
  print("Saved", fig_name)


def barcharts_adaptation(df_plot, sequence_name, show_left_y_axis=False,
                         show_right_y_axis=False, show_legend=False):
  matplotlib.rcParams['font.size'] = 18
  # matplotlib.rcParams['figure.figsize'] = 8, 8

  plt.clf()
  # plt.figure(figsize=(8, 8))

  # Need to add the 'None' strategy (no adaptation).
  df_concat = pd.DataFrame([["None", 4000, "Train", df_plot.iloc[0]["EPE"], 0],
                            ["None", 4000, "Novel", df_plot.iloc[1]["EPE"], 0]],
                            columns=["Strategy", "Step", "Domain", "EPE", "GradientUpdates"])
  df_plot = df_plot.append(df_concat)
  df_plot = df_plot[df_plot["Step"] == 4000]
  df_plot = df_plot.replace({"MAD-FULL": "MAD"})

  palette = {"Train": "tab:blue", "Novel": "tab:red"}

  g = sns.catplot(
      data=df_plot, kind="bar", order=["None", "MAD", "VS", "ER", "VS + ER"],
      x="Strategy", y="EPE", hue="Domain", legend=False,
      ci=None, palette=palette, alpha=0.8, height=5, aspect=1
  )

  if show_legend:
    plt.legend()

  ax = plt.gca()
  ax2 = ax.twinx()

  # Plot the number of gradient update steps on a RHS axis.
  # print(df_plot[(df_plot["Strategy"] == "None") & (df_plot["Domain"] == "Train")])
  x_no_vs = [0, 1, 3]
  x_with_vs = [2, 4]
  updates_no_vs = [
      df_plot[(df_plot["Domain"] == "Train") & (df_plot["Strategy"] == s)].iloc[0]["GradientUpdates"] \
          for s in ["None", "MAD", "ER"]]
  updates_with_vs = [
      df_plot[(df_plot["Domain"] == "Train") & (df_plot["Strategy"] == s)].iloc[0]["GradientUpdates"] \
          for s in ["VS", "VS + ER"]]
  ax2.plot(x_no_vs, updates_no_vs, linestyle=None, linewidth=0, marker="^", markersize=12, color="tab:green", label="Gradient Updates")
  ax2.plot(x_with_vs, updates_with_vs, linestyle=None, linewidth=0, marker="o", markersize=12, color="tab:green", label="Gradient Updates")
  ax2.set_ylabel("gradient updates", color="tab:green")

  if not show_right_y_axis:
    ax2.get_yaxis().set_visible(False)
    # ax2.get_yaxis().set_label("Gradient Updates")
    # ax2.set_axis_labels("", "gradient updates")
    # print(dir(ax2))
    # print(dir(ax2.get_yaxis()))


  g.set_axis_labels("", "end-point-error (EPE)" if show_left_y_axis else "")
  ax.grid(b=True, axis="y")

  fig_name = sequence_name.replace(" ", "_") + "_barchart.pdf"
  fig_path = path_to_output(reldir="barcharts")
  os.makedirs(fig_path, exist_ok=True)
  plt.savefig(os.path.join(fig_path, fig_name), bbox_inches="tight")
  print("Saved", fig_name)


if __name__ == "__main__":
  # Write all of the data to a format that will work with Seaborn.
  # sequences = [
  #   "Virtual KITTI Scene 01",
  #   "Virtual KITTI Scene 20",
  #   "KITTI Campus",
  #   "KITTI City",
  #   "KITTI Road",
  #   "Virtual KITTI Fog",
  #   "Virtual KITTI Rain"
  # ]

  # output_folder = path_to_output(reldir="adaptation_summaries")
  # os.makedirs(output_folder, exist_ok=True)

  #===== NOTE: If running evaluation for the first time, run this code first ===
  # df = parse_csv("/home/milo/rrg/src/perception/adaptive_stereo/resources/adaptation.csv")
  # for s in sequences:
  #   df_plot = reorganize_df(df, s)
  #   output_file = os.path.join(output_folder, s.replace(" ", "_") + ".csv")
  #   df_plot.to_csv(output_file)
  #   print("Saved", output_file)
  #=============================================================================

  # Make barchart to summarize pre and post-adaptation EPE.
  # for s in sequences:
  #   df_plot = pd.read_csv(os.path.join(output_folder, s.replace(" ", "_") + ".csv"))
  #   barcharts_adaptation(df_plot, s, show_legend=(s == "KITTI Road"),
  #                        show_left_y_axis=(s == "Virtual KITTI Scene 01"),
  #                        show_right_y_axis=(s == "KITTI Road"))

  # Make line plots to show adaptation progress.
  # for s in sequences:
    # df_plot = pd.read_csv(os.path.join(output_folder, s.replace(" ", "_") + ".csv"))
    # lineplots_adaptation(df_plot, s, show_legend=(s == "Virtual KITTI Scene 01"))

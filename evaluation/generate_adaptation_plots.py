# Copyright 2020 Massachusetts Institute of Technology
#
# @file generate_adaptation_plots.py
# @author Milo Knowles
# @date 2020-10-08 16:54:36 (Thu)

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from utils.path_utils import *

matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['pdf.fonttype'] = 42 # Solve Type 3 font problem.


def lineplots_adaptation(df, experiment_name, show_legend=False):
  matplotlib.rcParams['font.size'] = 18
  matplotlib.rcParams['lines.markersize'] = 12

  plt.clf()
  plt.figure(figsize=(4, 3))

  df_concat = pd.DataFrame([["VS + ER", 0, "TRAIN", df.iloc[0]["EPE"], df.iloc[0]["FCS"], 0],
                          ["VS + ER", 0, "ADAPT", df.iloc[1]["EPE"], df.iloc[0]["FCS"], 0]],
                          columns=["Method", "Step", "Domain", "EPE", "FCS", "GradientUpdates"])
  df = df.append(df_concat)
  df = df[df["Method"].isin(["MAD-FULL", "VS + ER"])]
  df = df.replace({"TRAIN": "Train", "ADAPT": "Novel"})

  palette = {"MAD-FULL": "tab:blue", "VS + ER": "tab:red"}
  ax = sns.lineplot(x="Step", y="EPE", hue="Method", style="Domain", data=df,
                    legend="full" if show_legend else False,
                    palette=palette, markers=True)
  plt.xlabel("adaptation step")
  plt.ylabel("end-point-error (EPE)")
  plt.xticks([0, 1000, 2000, 3000, 4000])

  if show_legend:
    handles, labels = ax.get_legend_handles_labels()

    # Remove the legend category labels ("Method" and "Domain").
    handles = [handles[i] for i in (1, 2, 4, 5)]
    labels = [labels[i] for i in (1, 2, 4, 5)]
    plt.legend(handles, labels, loc="upper right", fontsize="small")

  ax.grid(True, which='both')

  fig_name = experiment_name + "_lineplot.pdf"
  fig_path = path_to_output(reldir="lineplots")
  os.makedirs(fig_path, exist_ok=True)

  plt.savefig(os.path.join(fig_path, fig_name), bbox_inches="tight")
  print("Saved", fig_name)


def barcharts_adaptation(df, experiment_name, show_left_y_axis=False,
                         show_right_y_axis=False, show_legend=False):
  matplotlib.rcParams['font.size'] = 18

  plt.clf()
  plt.figure(figsize=(4, 4))

  # Need to add the 'None' Method (no adaptation).
  df_concat = pd.DataFrame([["None", 4000, "TRAIN", df.iloc[0]["EPE"], df.iloc[0]["FCS"], 0],
                            ["None", 4000, "ADAPT", df.iloc[1]["EPE"], df.iloc[0]["FCS"], 0]],
                            columns=["Method", "Step", "Domain", "EPE", "FCS", "GradientUpdates"])
  df = df.append(df_concat)
  df = df[df["Step"] == 4000]
  df = df.replace({"MAD-FULL": "MAD"})
  df = df.replace({"TRAIN": "Train", "ADAPT": "Novel"})

  palette = {"Train": "tab:blue", "Novel": "tab:red"}

  g = sns.catplot(
      data=df, kind="bar", order=["None", "MAD", "VS", "ER", "VS + ER"],
      x="Method", y="EPE", hue="Domain", legend=False,
      ci=None, palette=palette, alpha=0.8, height=5, aspect=1
  )

  if show_legend:
    plt.legend()

  ax = plt.gca()
  ax2 = ax.twinx()
  ax2.set_ylim([0, 4100])

  # Plot the number of gradient update steps on a RHS axis.
  x_no_vs = [1, 3]
  x_with_vs = [2, 4]
  updates_no_vs = [
      df[(df["Domain"] == "Train") & (df["Method"] == s)].iloc[0]["GradientUpdates"] \
          for s in ["MAD", "ER"]]
  updates_with_vs = [
      df[(df["Domain"] == "Train") & (df["Method"] == s)].iloc[0]["GradientUpdates"] \
          for s in ["VS", "VS + ER"]]
  ax2.plot(x_no_vs, updates_no_vs, linestyle=None, linewidth=0, marker="x", markersize=12,
          color="tab:green", label="Gradient Updates")
  ax2.plot(x_with_vs, updates_with_vs, linestyle=None, linewidth=0, marker="x", markersize=12,
           color="tab:green", label="Gradient Updates")
  ax2.set_ylabel("gradient updates", color="tab:green")

  if not show_right_y_axis:
    ax2.get_yaxis().set_visible(False)

  g.set_axis_labels("", "end-point-error (EPE)" if show_left_y_axis else "")
  ax.grid(b=True, axis="y")

  fig_name = experiment_name + "_barchart.pdf"
  fig_path = path_to_output(reldir="barcharts")
  os.makedirs(fig_path, exist_ok=True)
  plt.savefig(os.path.join(fig_path, fig_name), bbox_inches="tight")
  print("Saved", fig_name)


def barcharts_separate_novel_train(df, en, show_left_y_axis=False):
  """
  Show the EPE after adaptation is finished in the novel domain.
  """
  matplotlib.rcParams['font.size'] = 18

  # Need to add the 'None' Method (no adaptation).
  df_concat = pd.DataFrame([["None", 4000, "TRAIN", df.iloc[0]["EPE"], df.iloc[0]["FCS"], 0],
                            ["None", 4000, "ADAPT", df.iloc[1]["EPE"], df.iloc[0]["FCS"], 0]],
                            columns=["Method", "Step", "Domain", "EPE", "FCS", "GradientUpdates"])
  df = df.append(df_concat)
  df = df[df["Step"] == 4000]
  df = df.replace({"MAD-FULL": "MAD"})
  df = df.replace({"TRAIN": "Train", "ADAPT": "Novel"})

  palette = {"Train": "tab:blue", "Novel": "tab:red"}

  for domain_name in ("Train", "Novel"):
    plt.clf()
    plt.figure(figsize=(4, 4))

    g = sns.catplot(
        data=df[df["Domain"] == domain_name],
        kind="bar", order=["None", "MAD", "VS", "ER", "VS + ER"],
        x="Method", y="EPE", hue="Domain", # color="tab:red" if domain_name == "Novel" else "tab:blue",
        legend=False, ci=None, palette=palette, alpha=0.8, height=5, aspect=1
    )

    ax = plt.gca()

    g.set_axis_labels("", "end-point-error (EPE)" if show_left_y_axis else "")
    ax.grid(b=True, axis="y")

    if domain_name == "Novel":
      ax.set_ylim([0, 17])
    else:
      ax.set_ylim([0, 10])

    # NOTE(milo): Saving for presentation with .png
    fig_name = "{}_{}.png".format(en, domain_name)
    fig_path = path_to_output(reldir="barcharts_separate")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(os.path.join(fig_path, fig_name), bbox_inches="tight")
    print("Saved", fig_name)


if __name__ == "__main__":
  experiment_names = [
    "clone_to_campus",
    "clone_to_city",
    "clone_to_fog",
    "clone_to_rain",
    "flying_to_campus",
    "flying_to_city",
    "flying_to_road",
    "flying_to_vk01",
    "flying_to_vk20"
  ]

  # # Make barchart to summarize pre and post-adaptation EPE.
  # for en in experiment_names:
  #   output_folder = path_to_output(reldir="adapt_results/{}".format(en))
  #   df = pd.read_csv(os.path.join(output_folder, "results.csv"))
  #   barcharts_adaptation(df, en, show_legend=(en == "flying_to_road"),
  #                        show_left_y_axis=(en == "flying_to_campus"),
  #                        show_right_y_axis=(en == "flying_to_road"))

  # # Make line plots to show adaptation progress.
  # for en in experiment_names:
  #   output_folder = path_to_output(reldir="adapt_results/{}".format(en))
  #   df = pd.read_csv(os.path.join(output_folder, "results.csv"))
  #   lineplots_adaptation(df, en, show_legend=(en == "flying_to_vk01"))

  # Make barchart to summarize pre and post-adaptation EPE.
  for en in experiment_names:
    output_folder = path_to_output(reldir="adapt_results/{}".format(en))
    df = pd.read_csv(os.path.join(output_folder, "results.csv"))
    barcharts_separate_novel_train(df, en, show_left_y_axis=(en == "flying_to_vk01"))

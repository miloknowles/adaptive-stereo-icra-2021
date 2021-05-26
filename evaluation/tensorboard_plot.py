import os
from numpy.core.fromnumeric import sort
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from utils.path_utils import *
from utils.ema import online_ema

# matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['pdf.fonttype'] = 42 # Solve Type 3 font problem.
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['lines.markersize'] = 12

from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record


# https://stackoverflow.com/questions/37304461/tensorflow-importing-data-from-a-tensorboard-tfevent-file
def get_tag_from_logfiles(summary_dir, tag):
  steps = []
  output = []

  def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
      yield event_pb2.Event.FromString(r)

  for filename in os.listdir(summary_dir):
    path = os.path.join(summary_dir, filename)
    # print(path)
    for event in my_summary_iterator(path):
      for value in event.summary.value:
        if value.tag == tag:
          output.append(value.simple_value)
          steps.append(event.step)

  return steps, output


def joint_list_sort(sort_by_list, other_list):
  """
  - Sorts two lists based on the values in the sort_by_list.
  - The sorted lists are returned in the same order as the args.
  - https://www.kite.com/python/answers/how-to-sort-a-list-based-on-another-list-in-python
  """
  zipped_lists = sorted(zip(sort_by_list, other_list))
  return [e for e, _ in zipped_lists], [e for _, e in zipped_lists]


def plot_series_epe(path_to_adapt, path_to_baseline, steps=1000):
  """
  Plots the progression of EPE (end-point-error) during adaptation experiments.
  """
  for label in ("adaptation", "baseline"):
    print("Processing label {}".format(label))
    print("Getting all data with tag {} from logfiles".format("EPE"))
    step_values, epe_values = get_tag_from_logfiles(
        path_to_adapt if label == "adaptation" else path_to_baseline, "EPE")
    step_values, epe_values = joint_list_sort(step_values, epe_values)
    print("Found {} values".format(len(epe_values)))
    print("Done")

    # Truncate to a limited number of steps.
    step_values = step_values[:min(steps, len(step_values))]
    epe_values = epe_values[:min(steps, len(epe_values))]

    # Apply EMA smoothing.
    s_last = epe_values[0]
    for i in range(len(epe_values)):
      s_last = online_ema(s_last, epe_values[i], weight=0.8)
      epe_values[i] = s_last

    assert(len(step_values) == len(epe_values))

    print("Plotting {} steps".format(len(step_values)))
    plt.plot(step_values, epe_values, label=label, color="tab:red" if label == "adaptation" else "tab:blue")

  plt.xlabel("step")
  plt.ylabel("end-point-error (EPE)")
  plt.xlim(0, steps)
  plt.legend(loc="upper right", fontsize="small")
  plt.grid(True, which='both')
  plt.show()


def plot_series_fcs(path_to_adapt, nsteps=1000):
  steps, values = get_tag_from_logfiles(path_to_adapt, "fcs/raw")
  steps, values = joint_list_sort(steps, values)
  print("Found {} values".format(len(steps)))
  print("Done")

  # Truncate to a limited number of steps.
  steps = steps[:min(nsteps, len(steps))]
  values = values[:min(nsteps, len(values))]

  # Apply EMA smoothing.
  s_last = values[0]
  for i in range(len(values)):
    s_last = online_ema(s_last, values[i], weight=0.95)
    values[i] = s_last

  plt.plot(steps, values, label="adaptation", color="tab:red")
  plt.xlabel("step")
  plt.ylabel("feature contrast score ($\mathcal{S}_{I}$)")
  plt.xlim(0, nsteps)
  plt.legend(loc="upper right", fontsize="small")
  plt.grid(True, which='both')
  plt.show()


if __name__ == "__main__":
  # plot_series_epe("/home/milo/training_logs/plateau_example_adapt/adapt/",
  #                 "/home/milo/training_logs/plateau_example_baseline/adapt/",
  #                 steps=1000)

  plot_series_fcs("/home/milo/training_logs/plateau_example_adapt/adapt/", nsteps=1000)

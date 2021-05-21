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

from tensorflow.python.summary.summary_iterator import summary_iterator


def get_tag_from_logfile(path_to_logfile, tag):
  """
  Retrieves a time-ordered sequences of scalars with the specified tag (e.g "loss").
  """
  output = []
  for e in summary_iterator(path_to_logfile):
    for v in e.summary.value:
      if v.tag == tag:
        output.append(v.simple_value)
  return output


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
    for event in my_summary_iterator(path):
      for value in event.summary.value:
        if value.tag == tag:
          output.append(value.simple_value)
          steps.append(event.step)

  return steps, output


def coupled_list_sort(list1, list2, sort_by=0):
  # Sort by the first list.
  if sort_by == 0:
    zipped_lists = sorted(zip(list1, list2))
    return [e for e, _ in zipped_lists], [e for _, e in zipped_lists]
  # Sort by the second list.
  else:
    zipped_lists = sorted(zip(list2, list1))
    return [e for _, e in zipped_lists], [e for e, _ in zipped_lists]


def plot_series_epe(path_to_adapt, path_to_baseline, steps=1000):
  for label in ("adaptation", "baseline"):
    print("Processing label {}".format(label))
    print("Getting all data with tag {} from logfiles".format("EPE"))
    step_values, epe_values = get_tag_from_logfiles(
        path_to_adapt if label == "adaptation" else path_to_baseline, "EPE")
    step_values, epe_values = coupled_list_sort(step_values, epe_values, sort_by=0)
    print("Found {} values".format(len(epe_values)))
    print("Done")

    # Truncate to a limited number of steps.
    step_values = step_values[:min(steps, len(step_values))]
    epe_values = epe_values[:min(steps, len(epe_values))]

    # Apply EMA smoothing.
    s_last = epe_values[0]
    for i in range(len(epe_values)):
      s_last = online_ema(s_last, epe_values[i], weight=0.7)
      epe_values[i] = s_last

    assert(len(step_values) == len(epe_values))

    print("Plotting {} steps".format(len(step_values)))
    plt.plot(step_values, epe_values, label=label, color="tab:red" if label == "adaptation" else "tab:blue")

  plt.xlabel("step")
  plt.ylabel("end-point-error (EPE)")
  plt.legend(loc="upper right", fontsize="small")
  plt.grid(True, which='both')
  plt.show()


if __name__ == "__main__":
  plot_series_epe("/home/milo/training_logs/adapt_flying_to_vk01_nonstop/adapt/",
                  "/home/milo/training_logs/adapt_flying_to_vk01_nonstop/adapt/",
                  steps=1000)

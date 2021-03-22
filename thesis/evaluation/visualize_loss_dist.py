# Copyright 2020 Massachusetts Institute of Technology
#
# @file visualize_loss_dist.py
# @author Milo Knowles
# @date 2020-04-02 11:06:54 (Thu)

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sbn
import pandas as pd

import sys
sys.path.append("../")
from utils.dataset_utils import read_lines


def visualize_loss_histograms_supervised():
  loss_nll_fly_train = np.load("../output/save_loss_hist_supervised/sup_nll_loss_sceneflow_flying_train_240.npy").clip(min=0, max=10)
  loss_nll_fly_val = np.load("../output/save_loss_hist_supervised/sup_nll_loss_sceneflow_flying_240.npy").clip(min=0, max=10)
  loss_nll_drv = np.load("../output/save_loss_hist_supervised/sup_nll_loss_sceneflow_driving.npy").clip(min=0, max=10)
  loss_nll_kit = np.load("../output/save_loss_hist_supervised/sup_nll_loss_kitti_stereo_full.npy").clip(min=0, max=10)

  loss_l1_fly_train = np.load("../output/save_loss_hist_supervised/sup_sad_loss_sceneflow_flying_train_240.npy").clip(min=0, max=40)
  loss_l1_fly_val = np.load("../output/save_loss_hist_supervised/sup_sad_loss_sceneflow_flying_train_240.npy").clip(min=0, max=40)
  loss_l1_drv = np.load("../output/save_loss_hist_supervised/sup_sad_loss_sceneflow_driving.npy").clip(min=0, max=40)
  loss_l1_kit = np.load("../output/save_loss_hist_supervised/sup_sad_loss_kitti_stereo_full.npy").clip(min=0, max=40)

  plt.hist(loss_l1_fly_train, bins=50, facecolor="blue", density=True, alpha=0.5, label="Flying (Train)")
  plt.hist(loss_l1_fly_val, bins=50, facecolor="green", density=True, alpha=0.5, label="Flying (Val)")
  plt.hist(loss_l1_drv, bins=50, facecolor="orange", density=True, alpha=0.5, label="Driving (Novel)")
  plt.hist(loss_l1_kit, bins=50, facecolor="red", density=True, alpha=0.5, label="KITTI (Novel)")
  plt.title("Disparity-MAD Loss")
  plt.xlabel("Loss")
  plt.ylabel("Frequency")
  plt.legend()
  plt.show()

  plt.hist(loss_nll_fly_train, bins=50, facecolor="blue", density=True, alpha=0.5, label="Flying (Train)")
  plt.hist(loss_nll_fly_val, bins=50, facecolor="green", density=True, alpha=0.5, label="Flying (Val)")
  plt.hist(loss_nll_drv, bins=50, facecolor="orange", density=True, alpha=0.5, label="Driving (Novel)")
  plt.hist(loss_nll_kit, bins=50, facecolor="red", density=True, alpha=0.5, label="KITTI (Novel)")
  plt.title("Disparity-NLL Loss")
  plt.xlabel("Loss")
  plt.ylabel("Frequency")
  plt.legend()
  plt.show()


def visualize_loss_histograms_monodepth():
  loss_raw_fly_train = np.load("../output/save_loss_hist_monodepth/md_loss_sceneflow_flying_train_240.npy", allow_pickle=True)
  loss_raw_fly_val = np.load("../output/save_loss_hist_monodepth/md_loss_sceneflow_flying_240.npy", allow_pickle=True)
  loss_raw_drv = np.load("../output/save_loss_hist_monodepth/md_loss_sceneflow_driving.npy", allow_pickle=True)
  loss_raw_kit = np.load("../output/save_loss_hist_monodepth/md_loss_kitti_stereo_full.npy", allow_pickle=True)

  loss_nll_fly_train = np.load("../output/save_loss_hist_monodepth/md_nll_loss_sceneflow_flying_train_240.npy")
  loss_nll_fly_val = np.load("../output/save_loss_hist_monodepth/md_nll_loss_sceneflow_flying_240.npy")
  loss_nll_drv = np.load("../output/save_loss_hist_monodepth/md_nll_loss_sceneflow_driving.npy")
  loss_nll_kit = np.load("../output/save_loss_hist_monodepth/md_nll_loss_kitti_stereo_full.npy")

  plt.hist(loss_raw_fly_train, bins=30, facecolor="blue", density=True, alpha=0.5, label="Flying (Train)")
  plt.hist(loss_raw_fly_val, bins=30, facecolor="green", density=True, alpha=0.5, label="Flying (Val)")
  plt.hist(loss_raw_drv, bins=30, facecolor="orange", density=True, alpha=0.5, label="Driving (Novel)")
  plt.hist(loss_raw_kit, bins=30, facecolor="red", density=True, alpha=0.5, label="KITTI (Novel)")
  plt.title("Reconstruction-MAD")
  plt.xlabel("Loss")
  plt.ylabel("Frequency")
  plt.legend()
  plt.show()

  clamp_max = 20.0
  plt.hist(loss_nll_fly_train.clip(max=clamp_max), bins=30, facecolor="blue", density=True, alpha=0.5, label="Flying (Train)")
  plt.hist(loss_nll_fly_val.clip(max=clamp_max), bins=30, facecolor="green", density=True, alpha=0.5, label="Flying (Val)")
  plt.hist(loss_nll_drv.clip(max=clamp_max), bins=30, facecolor="orange", density=True, alpha=0.5, label="Driving (Novel)")
  plt.hist(loss_nll_kit.clip(max=clamp_max), bins=30, facecolor="red", density=True, alpha=0.5, label="KITTI (Novel)")
  plt.title("Reconstruction-NLL")
  plt.xlabel("Loss")
  plt.ylabel("Frequency")
  plt.legend()
  plt.show()


def generate_precision_recall(train_losses, novel_losses, thresholds):
  precision = torch.zeros(len(thresholds))
  recall = torch.zeros(len(thresholds))

  for i, cutoff in enumerate(thresholds):
    tp = (novel_losses >= cutoff).sum()   # Labelled positive and should be positive.
    fn = (novel_losses < cutoff).sum()    # Labelled negative but actually positive.
    tn = (train_losses < cutoff).sum()    # Labelled negative and should be negative.
    fp = (train_losses >= cutoff).sum()   # Labelled positive but actually negative.

    pr = tp / (tp + fp)
    re = tp / (tp + fn)
    precision[i] = pr
    recall[i] = re

  return precision, recall


def precision_recall_supervised():
  """
  Precision = % of predicted (+) that are actually (+)
  Recall = % of true (+) that are predicted to be (+)
  """
  loss_nll_fly_train = np.load("../output/save_loss_hist_supervised/sup_nll_loss_sceneflow_flying_train_240.npy").clip(min=0, max=10)
  loss_nll_fly_val = np.load("../output/save_loss_hist_supervised/sup_nll_loss_sceneflow_flying_240.npy").clip(min=0, max=10)
  loss_nll_drv = np.load("../output/save_loss_hist_supervised/sup_nll_loss_sceneflow_driving.npy").clip(min=0, max=10)
  loss_nll_kit = np.load("../output/save_loss_hist_supervised/sup_nll_loss_kitti_stereo_full.npy").clip(min=0, max=10)

  loss_l1_fly_train = np.load("../output/save_loss_hist_supervised/sup_sad_loss_sceneflow_flying_train_240.npy").clip(min=0, max=40)
  loss_l1_fly_val = np.load("../output/save_loss_hist_supervised/sup_sad_loss_sceneflow_flying_train_240.npy").clip(min=0, max=40)
  loss_l1_drv = np.load("../output/save_loss_hist_supervised/sup_sad_loss_sceneflow_driving.npy").clip(min=0, max=40)
  loss_l1_kit = np.load("../output/save_loss_hist_supervised/sup_sad_loss_kitti_stereo_full.npy").clip(min=0, max=40)

  thresholds_nll = np.linspace(1.0, 10.0, 500)
  train_nll_loss_all = np.concatenate([loss_nll_fly_train, loss_nll_fly_val])
  novel_nll_loss_all = np.concatenate([loss_nll_drv, loss_nll_kit])
  precision_nll, recall_nll = generate_precision_recall(train_nll_loss_all, novel_nll_loss_all, thresholds_nll)

  thresholds_sad = np.linspace(0.1, 40, 500)
  train_sad_loss_all = np.concatenate([loss_l1_fly_train, loss_l1_fly_val])
  novel_sad_loss_all = np.concatenate([loss_l1_drv, loss_l1_kit])
  precision_sad, recall_sad = generate_precision_recall(train_sad_loss_all, novel_sad_loss_all, thresholds_sad)

  # plt.plot(recall_sad, precision_sad, color="red", label="Supervised-SAD")
  # plt.plot(recall_nll, precision_nll, color="blue", label="Supervised-NLL")
  # plt.title("Supervised Novelty Detection")

  df_mad = pd.DataFrame({
    "Recall": recall_sad,
    "Precision": precision_sad,
    "Loss": "Disparity-MAD"
  })

  df_nll = pd.DataFrame({
    "Recall": recall_nll,
    "Precision": precision_nll,
    "Loss": "Disparity-NLL"
  })

  df_all = pd.concat([df_mad, df_nll])

  return df_all

  # sbn.lineplot(x="Recall", y="Precision", data=df_all, hue="Loss")
  # plt.xlabel("Recall")
  # plt.ylabel("Precision")
  # plt.legend()
  # plt.show()


def precision_recall_monodepth():
  loss_raw_fly_train = np.load("../output/save_loss_hist_monodepth/md_loss_sceneflow_flying_train_240.npy", allow_pickle=True)
  loss_raw_fly_val = np.load("../output/save_loss_hist_monodepth/md_loss_sceneflow_flying_240.npy", allow_pickle=True)
  loss_raw_drv = np.load("../output/save_loss_hist_monodepth/md_loss_sceneflow_driving.npy", allow_pickle=True)
  loss_raw_kit = np.load("../output/save_loss_hist_monodepth/md_loss_kitti_stereo_full.npy", allow_pickle=True)

  loss_nll_fly_train = np.load("../output/save_loss_hist_monodepth/md_nll_loss_sceneflow_flying_train_240.npy")
  loss_nll_fly_val = np.load("../output/save_loss_hist_monodepth/md_nll_loss_sceneflow_flying_240.npy")
  loss_nll_drv = np.load("../output/save_loss_hist_monodepth/md_nll_loss_sceneflow_driving.npy")
  loss_nll_kit = np.load("../output/save_loss_hist_monodepth/md_nll_loss_kitti_stereo_full.npy")

  thresholds_nll = np.linspace(-10, 30, 500)
  train_nll_loss_all = np.concatenate([loss_nll_fly_train, loss_nll_fly_val])
  novel_nll_loss_all = np.concatenate([loss_nll_drv, loss_nll_kit])
  precision_nll, recall_nll = generate_precision_recall(train_nll_loss_all, novel_nll_loss_all, thresholds_nll)

  thresholds_md = np.linspace(0.05, 0.6, 500)
  train_md_loss_all = np.concatenate([loss_raw_fly_train, loss_raw_fly_val])
  novel_md_loss_all = np.concatenate([loss_raw_drv, loss_raw_kit])
  precision_md, recall_md = generate_precision_recall(train_md_loss_all, novel_md_loss_all, thresholds_md)

  df_md = pd.DataFrame({
    "Recall": recall_md,
    "Precision": precision_md,
    "Loss": "Reconstruction-MAD"
  })

  df_nll = pd.DataFrame({
    "Recall": recall_nll,
    "Precision": precision_nll,
    "Loss": "Reconstruction-NLL"
  })

  df_all = pd.concat([df_md, df_nll])

  return df_all

  # sbn.lineplot(x="Recall", y="Precision", data=df_all, hue="Loss")

  # plt.plot(recall_md, precision_md, color="red", label="Monodepth")
  # plt.plot(recall_nll, precision_nll, color="blue", label="Monodepth-NLL")
  # plt.title("Self-Supervised Novelty Detection")
  # plt.xlabel("Recall")
  # plt.ylabel("Precision")
  # plt.legend()
  # plt.show()


def find_overlap_images(k=5):
  """
  Prints out the filenames of novel images with the lowest loss.
  """
  loss_nll_drv = np.load("../output/save_loss_hist_supervised/sup_nll_loss_sceneflow_driving.npy").clip(min=0, max=10)
  loss_nll_kit = np.load("../output/save_loss_hist_supervised/sup_nll_loss_kitti_stereo_full.npy").clip(min=0, max=10)
  loss_sad_drv = np.load("../output/save_loss_hist_supervised/sup_sad_loss_sceneflow_driving.npy").clip(min=0, max=40)
  loss_sad_kit = np.load("../output/save_loss_hist_supervised/sup_sad_loss_kitti_stereo_full.npy").clip(min=0, max=40)

  driving_lines = read_lines("../splits/sceneflow_driving/test_lines.txt")
  kitti_lines = read_lines("../splits/kitti_stereo_full/test_lines.txt")

  loss_sad_drv_sort, sad_drv_indices = torch.from_numpy(loss_sad_drv).sort()
  loss_sad_kit_sort, sad_kit_indices = torch.from_numpy(loss_sad_kit).sort()

  loss_nll_drv_sort, nll_drv_indices = torch.from_numpy(loss_nll_drv).sort()
  loss_nll_kit_sort, nll_kit_indices = torch.from_numpy(loss_nll_kit).sort()

  print("\n==== Supervised-SAD Loss ====")
  print("SceneFlow Driving:")
  for i in range(k):
    idx = sad_drv_indices[i]
    print(driving_lines[idx])
  print("KITTIStereoFull:")
  for i in range(k):
    idx = sad_kit_indices[i]
    print(kitti_lines[idx])

  print("\n==== Supervised-NLL Loss ====")
  print("SceneFlow Driving:")
  for i in range(k):
    idx = nll_drv_indices[i]
    print(driving_lines[idx])
  print("KITTIStereoFull:")
  for i in range(k):
    idx = nll_kit_indices[i]
    print(kitti_lines[idx])

  loss_raw_drv = np.load("../output/save_loss_hist_monodepth/md_loss_sceneflow_driving.npy", allow_pickle=True)
  loss_raw_kit = np.load("../output/save_loss_hist_monodepth/md_loss_kitti_stereo_full.npy", allow_pickle=True)
  loss_nll_drv = np.load("../output/save_loss_hist_monodepth/md_nll_loss_sceneflow_driving.npy")
  loss_nll_kit = np.load("../output/save_loss_hist_monodepth/md_nll_loss_kitti_stereo_full.npy")

  loss_md_drv_sort, md_drv_indices = torch.from_numpy(loss_raw_drv).sort()
  loss_md_kit_sort, md_kit_indices = torch.from_numpy(loss_raw_kit).sort()

  loss_nll_drv_sort, nll_drv_indices = torch.from_numpy(loss_nll_drv).sort()
  loss_nll_kit_sort, nll_kit_indices = torch.from_numpy(loss_nll_kit).sort()

  print("\n==== Monodepth Loss ====")
  print("SceneFlow Driving:")
  for i in range(k):
    idx = md_drv_indices[i]
    print(driving_lines[idx])
  print("KITTIStereoFull:")
  for i in range(k):
    idx = md_kit_indices[i]
    print(kitti_lines[idx])

  print("\n==== Monodepth-NLL Loss ====")
  print("SceneFlow Driving:")
  for i in range(k):
    idx = nll_drv_indices[i]
    print(loss_nll_drv[idx], driving_lines[idx])
  print("KITTIStereoFull:")
  for i in range(k):
    idx = nll_kit_indices[i]
    print(loss_nll_kit[idx], kitti_lines[idx])



if __name__ == "__main__":
  visualize_loss_histograms_supervised()
  visualize_loss_histograms_monodepth()
  # df1 = precision_recall_supervised()
  # df2 = precision_recall_monodepth()
  # df_all = pd.concat([df1, df2])
  # sbn.lineplot(x="Recall", y="Precision", data=df_all, hue="Loss")
  # plt.show()

  # find_overlap_images()

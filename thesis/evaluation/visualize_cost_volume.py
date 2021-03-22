# Copyright 2020 Massachusetts Institute of Technology
#
# @file visualize_cost_volume.py
# @author Milo Knowles
# @date 2020-03-23 15:31:28 (Mon)

from models import MGCNet
from utils.dataset_utils import read_lines
from utils.visualization import *
from utils.training import load_model
from datasets.sceneflow_flying_dataset import SceneFlowFlyingDataset

import matplotlib.pyplot as plt
import cv2 as cv


def visualize_disp_pyramid(outputs, cmap):
  for scale in range(6, -1, -1):
    disp = outputs["pred_disp/{}".format(scale)]
    disp_viz = apply_cmap(disp, cmap=cmap)[0,:,:,:3]
    cv.imshow("disp/{}".format(scale), float_image_to_cv_uint8(disp_viz))


def visualize_cost_volume(outputs):
  row, col = 2, 10
  # This has shape (num_disp, 5, 19).
  slice_filt_6 = outputs["cost_volume_filt/6"].squeeze()[:, row, col]
  print(slice_filt_6)


if __name__ == "__main__":
  mgc = MGCNet(2, 320, 1216, predict_variance=False, image_channels=3, gradient_bulkhead=None, output_cost_volume=True)

  load_weights_folder = "/home/milo/training_logs/sm_flying_03/models/weights_80/"
  load_model(mgc, load_weights_folder, "madnet.pth")

  lines = read_lines("./splits/sceneflow_flying_240/test_lines.txt")
  dataset = SceneFlowFlyingDataset("/home/milo/datasets/sceneflow_flying_things_3d", lines, 320, 1216, False)

  cmap = plt.get_cmap("magma")

  for i, inputs in enumerate(dataset):
    for key in inputs:
      inputs[key] = inputs[key].unsqueeze(0).cuda()

    outputs = mgc(inputs)

    # Show the input RGB, predicted disp, and gt disp.
    visualize_disp_pyramid(outputs, cmap)
    # visualize_cost_volume(outputs)

    gt_disp_viz = apply_cmap(inputs["gt_disp_l/0"], cmap=cmap)[0,:,:,:3]
    cv.imshow("gt_disp/0", float_image_to_cv_uint8(gt_disp_viz))
    cv.imshow("color_l/0", tensor_to_cv_rgb(inputs["color_l/0"].squeeze(0)))
    cv.waitKey(0)



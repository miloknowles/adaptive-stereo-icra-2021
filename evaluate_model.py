# Copyright 2020 Massachusetts Institute of Technology
#
# @file evaluate_model.py
# @author Milo Knowles
# @date 2020-07-30 19:10:04 (Thu)

import argparse
import os, sys
from utils.path_utils import path_to_output

import torch
from torch.utils.data import DataLoader

import cv2 as cv

from train import evaluate
from datasets.stereo_dataset import StereoDataset
from models.stereo_net import StereoNet, FeatureExtractorNetwork
from utils.visualization import *


def get_save_filename(save_folder, outputs_key, index):
  key_save_folder = os.path.join(save_folder, outputs_key.replace("/", "_"))
  os.makedirs(key_save_folder, exist_ok=True)
  save_filename = os.path.join(key_save_folder, "{:04d}.pt".format(index))
  return save_filename


def main(opt):
  # https://github.com/pytorch/pytorch/issues/15054
  torch.backends.cudnn.benchmark = True

  save_folder = os.path.join(opt.load_weights_folder, "outputs", opt.split)
  os.makedirs(save_folder, exist_ok=True)

  dataset = StereoDataset(opt.dataset_path, opt.dataset_name, opt.split, opt.height, opt.width, opt.subsplit,
                        scales=opt.scales, do_hflip=False, random_crop=False, load_disp_left=True,
                        load_disp_right=False)

  if opt.mode == "save":
    feature_net = FeatureExtractorNetwork(opt.stereonet_k).cuda()
    stereo_net = StereoNet(opt.stereonet_k, 1, 0).cuda()
    feature_net.load_state_dict(torch.load(os.path.join(opt.load_weights_folder, "feature_net.pth")), strict=True)
    stereo_net.load_state_dict(torch.load(os.path.join(opt.load_weights_folder, "stereo_net.pth")), strict=True)

    loader = DataLoader(dataset, opt.batch_size, False, num_workers=opt.batch_size, pin_memory=True,
                        drop_last=False, collate_fn=None)

    feature_net.eval()
    stereo_net.eval()

    with torch.no_grad():
      for i, inputs in enumerate(loader):
        for key in inputs:
          inputs[key] = inputs[key].cuda()

        lines = dataset.lines[opt.batch_size*i:opt.batch_size*(i+1)]
        left_feat, right_feat = feature_net(inputs["color_l/0"]), feature_net(inputs["color_r/0"])
        # outputs = stereo_net(inputs["color_l/0"], left_feat, right_feat, "l",
        #                     store_feature_gradients=False,
        #                     store_refinement_gradients=False,
        #                     do_refinement=True,
        #                     output_cost_volume=True)

        outputs = stereo_net(inputs["color_l/0"], left_feat, right_feat, "l",
                    output_cost_volume=True)

        # Save all network outputs.
        for key in outputs:
          if "pred_disp" in key:
            for j in range(len(outputs[key])):
              save_filename = get_save_filename(save_folder, key, opt.batch_size*i + j)
              torch.save(outputs[key][j], save_filename)

        print("Finished {}/{} batches".format(i, len(loader)), end='\r')
        sys.stdout.flush()

  elif opt.mode == "playback":
    cv.namedWindow("pred_disp_l/0", cv.WINDOW_NORMAL)
    cv.namedWindow("pred_disp_l/{}".format(opt.stereonet_k), cv.WINDOW_NORMAL)
    cv.namedWindow("color_l/0", cv.WINDOW_NORMAL)
    cv.namedWindow("gt_disp_l/0", cv.WINDOW_NORMAL)

    for i, inputs in enumerate(dataset):
      color_l_0 = tensor_to_cv_rgb(inputs["color_l/0"].contiguous())
      gt_disp_l_0 = visualize_disp_cv(inputs["gt_disp_l/0"].contiguous(), cmap=plt.get_cmap("inferno"), vmin=0, vmax=0.6*192)

      pred_disp_l_0 = torch.load(get_save_filename(save_folder, "pred_disp_l/0", i)).contiguous()
      pred_disp_l_k = torch.load(get_save_filename(save_folder, "pred_disp_l/{}".format(opt.stereonet_k), i)).contiguous()

      EPE = torch.abs(pred_disp_l_0.cpu() - inputs["gt_disp_l/0"].cpu())[inputs["gt_disp_l/0"].cpu() > 0].mean()
      print("EPE:", EPE)
      pred_disp_l_0 = visualize_disp_cv(pred_disp_l_0, cmap=plt.get_cmap("inferno"), vmin=0, vmax=0.6*192)
      pred_disp_l_k = visualize_disp_cv(pred_disp_l_k, cmap=plt.get_cmap("inferno"), vmin=0, vmax=0.6*192)

      err = torch.abs(gt_disp_l_0 - pred_disp_l_0)

      cv.imshow("pred_disp_l/0", pred_disp_l_0)
      cv.imshow("pred_disp_l/{}".format(opt.stereonet_k), pred_disp_l_k)
      cv.imshow("color_l/0", color_l_0)
      cv.imshow("gt_disp_l/0", gt_disp_l_0)
      cv.imshow("l1_error", visualize_disp_cv(err, cmap=plt.get_cmap("hot")))
      cv.waitKey(0)

  elif opt.mode == "video":
    output_folder = path_to_output("video")

    for i, inputs in enumerate(dataset):
      if opt.frames > 0 and i >= opt.frames:
        break

      color_l_0 = tensor_to_cv_rgb(inputs["color_l/0"].contiguous())
      gt_disp_l_0 = visualize_disp_cv(inputs["gt_disp_l/0"].contiguous(), cmap=plt.get_cmap("inferno"), vmin=0, vmax=0.6*192)

      pred_disp_l_0 = torch.load(get_save_filename(save_folder, "pred_disp_l/0", i)).contiguous()

      EPE = torch.abs(pred_disp_l_0.cpu() - inputs["gt_disp_l/0"].cpu())[inputs["gt_disp_l/0"].cpu() > 0].mean()
      print("EPE:", EPE)
      pred_disp_l_0 = visualize_disp_cv(pred_disp_l_0, cmap=plt.get_cmap("inferno"), vmin=0, vmax=0.6*192)

      cv.putText(color_l_0, "Frame {}".format(i), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
      cv.putText(pred_disp_l_0, "EPE: {:.02f}".format(EPE), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

      cv.imwrite(path_to_output("video/left_{:05d}.png".format(i)), color_l_0)
      cv.imwrite(path_to_output("video/gt_{:05d}.png".format(i)), gt_disp_l_0)
      cv.imwrite(path_to_output("video/pred_{:05d}.png".format(i)), pred_disp_l_0)

  elif opt.mode == "eval":
    pass

  else:
    raise NotImplementedError()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Options for evaluating depth prediction")
  parser.add_argument("--mode", type=str,
                      choices=["save", "playback", "eval", "video"])

  parser.add_argument("--dataset_path", type=str, help="Top level folder for the dataset being used")
  parser.add_argument("--dataset_name", type=str, help="Which dataset to evaluate on")
  parser.add_argument("--split", type=str, help="The dataset split to evaluate on")
  parser.add_argument("--subsplit", choices=["train", "val", "test"], help="Which split of the training data to use?")
  parser.add_argument("--load_weights_folder", default=None, type=str, help="Path to load madnet.pth weights from")
  parser.add_argument("--stereonet_k", type=int, default=3, choices=[3, 4],
                      help="The cost volume downsampling factor (i.e 4 ==> 1/16th resolution cost volume)")
  parser.add_argument("--scales", type=int, nargs="+", default=[0], help="Image scales used for the loss function")
  parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluating")

  parser.add_argument("--height", type=int, default=512, help="Image height (must be multiple of 64)")
  parser.add_argument("--width", type=int, default=960, help="Image width (must be multiple of 64)")

  parser.add_argument("--max_disp_viz", type=float, default=300, help="Divide the disparity by this to normalize it for viewing")
  parser.add_argument("--window_sf", type=float, default=1.5, help="Scale factor for OpenCV windows")
  parser.add_argument("--wait", default=False, action="store_true", help="If set, user must keypress through all images")
  parser.add_argument("--frames", default=-1, type=int, help="Number of playback frames")

  # parser.add_argument("--do_vertical_flip", default=False, action="store_true", help="Vertically flip input images")
  # parser.add_argument("--predict_variance", default=False, action="store_true", help="Save variance predictions from the network (MGC-Net only)")

  # parser.add_argument("--leftright_consistency", action="store_true", default=False, help="Predict disparity for the left and right images")
  # parser.add_argument("--variance_mode", default="disparity", choices=["disparity", "photometric"])
  parser.add_argument("--save_disp_as_image", action="store_true", default=False, help="Write color-mapped disparity to disk")

  opt = parser.parse_args()
  main(opt)

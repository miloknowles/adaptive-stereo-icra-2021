# Copyright 2020 Massachusetts Institute of Technology
#
# @file evaluate_model.py
# @author Milo Knowles
# @date 2020-08-09 12:14:11 (Sun)

import argparse
import os, sys

import torch
from torch.utils.data import DataLoader

import cv2 as cv

from train import evaluate, process_batch
from models.madnet import MadNet
from models.mgc_net import MGCNet
from datasets.stereo_dataset import StereoDataset
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
                        load_disp_right=False, do_vflip=opt.do_vertical_flip)
  loader = DataLoader(dataset, opt.batch_size, False, num_workers=opt.batch_size, pin_memory=True,
                    drop_last=False, collate_fn=None)

  if opt.mode == "save":
    # Load the network from pretrained weights.
    if opt.network == "MGC-Net":
      stereo_net = MGCNet(opt.radius_disp, img_height=opt.height, img_width=opt.width,
                          device=torch.device("cuda"), predict_variance=opt.predict_variance,
                          image_channels=3,
                          gradient_bulkhead=None,
                          variance_mode=opt.variance_mode).cuda()
      feature_net = stereo_net.feature_extractor.cuda()
    elif opt.network == "MAD-Net":
      stereo_net = MadNet(opt.radius_disp, img_height=opt.height, img_width=opt.width, device=torch.device("cuda"))
      feature_net = stereo_net.feature_extractor
    else:
      raise NotImplementedError()

    stereo_net.load_state_dict(torch.load(os.path.join(opt.load_weights_folder, "model_strict.pth")), strict=False)

    feature_net.eval()
    stereo_net.eval()

    with torch.no_grad():
      for i, inputs in enumerate(loader):
        for key in inputs:
          inputs[key] = inputs[key].cuda()

        outputs = process_batch(feature_net, stereo_net, inputs, output_scales=opt.scales, predict_variance=True,
                                variance_mode=opt.variance_mode, leftright_consistency=opt.leftright_consistency)

        # Save all network outputs.
        for key in outputs:
          if "pred_disp" in key or "pred_logvar" in key:
            for j in range(len(outputs[key])):
              save_filename = get_save_filename(save_folder, key, opt.batch_size*i + j)
              torch.save(outputs[key][j], save_filename)

        print("Finished {}/{} batches".format(i, len(loader)), end='\r')
        sys.stdout.flush()

  elif opt.mode == "playback":
    for s in opt.scales:
      cv.namedWindow("pred_disp_l/{}".format(s), cv.WINDOW_NORMAL)

    cv.namedWindow("color_l/0", cv.WINDOW_NORMAL)
    cv.namedWindow("gt_disp_l/0", cv.WINDOW_NORMAL)
    cv.namedWindow("pred_logvar_l/0", cv.WINDOW_NORMAL)

    for i, inputs in enumerate(dataset):
      color_l_0 = tensor_to_cv_rgb(inputs["color_l/0"].contiguous())
      gt_disp_l_0 = visualize_disp_cv(inputs["gt_disp_l/0"].contiguous(), cmap=plt.get_cmap("magma"), vmin=0, vmax=0.6*192)

      for s in opt.scales:
        pred_disp_l_s = torch.load(get_save_filename(save_folder, "pred_disp_l/{}".format(s), i)).contiguous()
        pred_disp_l_s = visualize_disp_cv(pred_disp_l_s, cmap=plt.get_cmap("magma"), vmin=0, vmax=0.6*192 / 2**s)
        cv.imshow("pred_disp_l/{}".format(s), pred_disp_l_s)

      path_to_logvar_l = get_save_filename(save_folder, "pred_logvar_l/0", i)
      if os.path.exists(path_to_logvar_l):
        pred_logvar_l_0 = torch.load(path_to_logvar_l).contiguous()
        pred_logvar_l_0 = visualize_disp_cv(pred_logvar_l_0, cmap=plt.get_cmap("magma"))
        cv.imshow("pred_logvar_l/0", pred_logvar_l_0)

      cv.imshow("color_l/0", color_l_0)
      cv.imshow("gt_disp_l/0", gt_disp_l_0)
      cv.waitKey(0)

  elif opt.mode == "eval":
    with torch.no_grad():
      EPEs = torch.zeros(len(loader))
      D1_alls = torch.zeros(len(loader), 5)
      FCSs = torch.zeros(len(loader))

      for i, inputs in enumerate(loader):
        for key in inputs:
          inputs[key] = inputs[key].cuda()

        # Load disparity from disk.
        for j in range(len(inputs["color_l/0"])):
          pred_disp_0 = torch.load(get_save_filename(save_folder, "pred_disp_l/0", opt.batch_size*i + j))

          gt_disp = inputs["gt_disp_l/0"][j]
          valid_mask = (gt_disp > 0)

          # Compute EPE.
          EPEs[i] = torch.abs(pred_disp_0 - gt_disp)[valid_mask].mean()

          # Compute D1-all for several outlier thresholds.
          for oi, ot in enumerate([2, 3, 4, 5, 10]):
            D1_alls[i, oi] = (valid_mask * (torch.abs(pred_disp_0 - gt_disp) > ot)).sum() / float(valid_mask.sum())

        print("Finished {}/{} batches".format(i, len(loader)), end='\r')
        sys.stdout.flush()

      EPEs = EPEs.mean()
      D1_alls = D1_alls.mean(dim=0)
      FCSs = FCSs.mean()

      metrics = {"EPE": EPEs.item(),
                "FCS": FCSs.item(),
                "D1_all_2px": D1_alls[0].item(),
                "D1_all_3px": D1_alls[1].item(),
                "D1_all_4px": D1_alls[2].item(),
                "D1_all_5px": D1_alls[3].item(),
                "D1_all_10px": D1_alls[4].item()}

      print("METRICS // EPE={:.3f} | >2px={:.3f} | >3px={:.3f} | >4px={:.3f} | >5px={:.3f} | >10px={:.3f}".format(
        metrics["EPE"], metrics["D1_all_2px"], metrics["D1_all_3px"], metrics["D1_all_4px"], metrics["D1_all_5px"], metrics["D1_all_10px"]))

  else:
    raise NotImplementedError()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Options for evaluating depth prediction")
  parser.add_argument("--network", type=str, default="MGC-Net", choices=["MGC-Net", "MAD-Net"])
  parser.add_argument("--mode", type=str,
                      choices=["save", "playback", "eval"])

  parser.add_argument("--dataset_path", type=str, help="Top level folder for the dataset being used")
  parser.add_argument("--dataset_name", type=str, help="Which dataset to evaluate on")
  parser.add_argument("--split", type=str, help="The dataset split to evaluate on")
  parser.add_argument("--subsplit", choices=["train", "val", "test"], help="Which split of the training data to use?")
  parser.add_argument("--load_weights_folder", default=None, type=str, help="Path to load madnet.pth weights from")

  parser.add_argument("--scales", type=int, nargs="+", default=[0], help="Visualize predicted disparity at these scales")
  parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluating")
  parser.add_argument("--radius_disp", type=int, default=2, help="Radius for disparity correction")

  parser.add_argument("--height", type=int, default=512, help="Image height (must be multiple of 64)")
  parser.add_argument("--width", type=int, default=960, help="Image width (must be multiple of 64)")

  parser.add_argument("--max_disp_viz", type=float, default=300, help="Divide the disparity by this to normalize it for viewing")
  parser.add_argument("--window_sf", type=float, default=1.5, help="Scale factor for OpenCV windows")
  parser.add_argument("--wait", default=False, action="store_true", help="If set, user must keypress through all images")

  parser.add_argument("--do_vertical_flip", default=False, action="store_true", help="Vertically flip input images")
  parser.add_argument("--predict_variance", default=False, action="store_true", help="Save variance predictions from the network (MGC-Net only)")
  parser.add_argument("--leftright_consistency", action="store_true", default=False, help="Predict disparity for the left and right images")
  parser.add_argument("--variance_mode", default="disparity", choices=["disparity", "photometric"])
  parser.add_argument("--save_disp_as_image", action="store_true", default=False, help="Write color-mapped disparity to disk")

  opt = parser.parse_args()
  main(opt)

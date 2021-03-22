import os, time

import cv2 as cv

import torch
import torch.nn.functional as F

from models.mgc_net import MGCNet
from models.mgc_net_guided import MGCNetGuided
from models.madnet import MadNet
from utils.dataset_utils import read_lines
from utils.training_utils import load_model, process_batch
from utils.visualization import tensor_to_cv_rgb, tensor_to_cv_disp, visualize_disp_cv
from utils.dataset_utils import dataset_cls_from_str, get_dataset
from datasets.sceneflow_driving_dataset import SceneFlowDrivingDataset
from datasets.sceneflow_flying_dataset import SceneFlowFlyingDataset
from datasets.kitti_stereo_2012_dataset import KittiStereo2012Dataset
from datasets.kitti_stereo_2015_dataset import KittiStereo2015Dataset
from datasets.kitti_stereo_full_dataset import KittiStereoFullDataset
from datasets.sceneflow_monkaa_dataset import SceneFlowMonkaaDataset
from evaluate_options import EvaluateOptions
from evaluation.metrics import compute_depth_errors_2012, compute_depth_errors_2015, disp_to_depth

MIN_DEPTH = 1e-3
MAX_DEPTH = 80


def get_save_filename_for_dataset(line, dataset, scale=0):
  frame_index_or_filename = line.split(" ")[-1]

  if dataset == "SceneFlowDrivingDataset":
    direction = line.split(" ")[1]
    save_filename = "{}_{:04d}_{}.pt".format(direction, int(frame_index_or_filename), scale)
  elif dataset == "SceneFlowFlyingDataset":
    _, lf, nf, frame_index = line.split(" ")
    save_filename = "{}_{}_{:04d}_{}.pt".format(lf, nf, int(frame_index), scale)
  elif dataset == "SceneFlowMonkaaDataset":
    sn, frame_index = line.split(" ")
    save_filename = "{}_{:04d}_{}.pt".format(sn, int(frame_index), scale)
  elif dataset in ["KittiStereo2012Dataset", "KittiStereo2015Dataset", "KittiStereoFullDataset"]:
    save_filename = str(scale) + "_" + frame_index_or_filename.replace(".png", ".pt")
  else:
    frame_index = int(frame_index_or_filename)
    save_filename = "{:04d}_{}_{}.pt".format(frame_index, scale)

  return save_filename


def evaluate_depth(opt):
  """
  Evaluate depth prediction accuracy of a model.
  """
  print("*** Running mode {} ***".format(opt.mode))

  dataset, lines = get_dataset(opt.dataset, opt.split, opt.dataset_path, opt.height, opt.width, opt.do_vertical_flip)
  device = torch.device("cuda")

  if opt.network == "MGC-Net":
    model = MGCNet(opt.radius_disp, img_height=opt.height, img_width=opt.width, device=device,
                   predict_variance=True, variance_mode=opt.variance_mode)
  elif opt.network == "MGC-Net-Guided":
    model = MGCNetGuided(opt.radius_disp, img_height=opt.height, img_width=opt.width, device=device,
                         predict_variance=True, variance_mode=opt.variance_mode)
  elif opt.network == "MAD-Net":
    model = MadNet(opt.radius_disp, img_height=opt.height, img_width=opt.width, device=device)

  pretrained_dict = load_model(model, None, opt.load_weights_folder, "madnet.pth", None)

  # Make sure loaded model was trained with same image height and width.
  model_height = pretrained_dict["height"]
  model_width = pretrained_dict["width"]
  print("Model loaded from disk trained with height={} and width={}".format(model_height, model_width))
  assert(opt.height == model_height and opt.width == model_width)

  save_folder_left = os.path.join(opt.load_weights_folder, "pred_disp_l/{}".format(opt.split))
  save_folder_right = os.path.join(opt.load_weights_folder, "pred_disp_r/{}".format(opt.split))

  save_folder_variance_left = os.path.join(opt.load_weights_folder, "pred_log_variance_l/{}".format(opt.split))
  save_folder_variance_right = os.path.join(opt.load_weights_folder, "pred_log_variance_r/{}".format(opt.split))

  # SAVE MODE: Save disparity predictions for all images in the test set.
  if opt.mode == "save":
    os.makedirs(save_folder_left, exist_ok=True)
    os.makedirs(save_folder_right, exist_ok=True)

    for i, inputs in enumerate(dataset):
      for key in inputs:
        inputs[key] = inputs[key].unsqueeze(0).to(device)

      outputs = process_batch(model.feature_extractor, model, inputs, output_scales=[0, 1, 2],
                              predict_variance=opt.network == "MGC-Net", variance_mode=opt.variance_mode,
                              leftright_consistency=opt.leftright_consistency, device=device)

      save_filename = get_save_filename_for_dataset(lines[i], opt.dataset, opt.scale)

      disp_l = outputs["pred_disp_l/{}".format(opt.scale)].cpu().squeeze(0)
      torch.save(disp_l, os.path.join(save_folder_left, save_filename))
      print("Saved {}".format(os.path.join(save_folder_left, save_filename)))

      if opt.leftright_consistency and "pred_disp_r/{}".format(opt.scale) in outputs:
        disp_r = outputs["pred_disp_r/{}".format(opt.scale)].cpu().squeeze(0)
        torch.save(disp_r, os.path.join(save_folder_right, save_filename))
        print("Saved {}".format(os.path.join(save_folder_right, save_filename)))

      if "pred_log_variance_l/{}".format(opt.scale) in outputs:
        os.makedirs(save_folder_variance_left, exist_ok=True)
        variance_save_path = os.path.join(save_folder_variance_left, save_filename)
        variance = outputs["pred_log_variance_l/{}".format(opt.scale)].cpu().squeeze(0)
        torch.save(variance, variance_save_path)
        print("Saved {}".format(variance_save_path))

      if "pred_log_variance_r/{}".format(opt.scale) in outputs:
        os.makedirs(save_folder_variance_right, exist_ok=True)
        variance_save_path = os.path.join(save_folder_variance_right, save_filename)
        variance = outputs["pred_log_variance_r/{}".format(opt.scale)].cpu().squeeze(0)
        torch.save(variance, variance_save_path)
        print("Saved {}".format(variance_save_path))

  # PLAYBACK MODE: View the input images, pred and gt disparity.
  elif opt.mode == "playback":
    cv.namedWindow("left", cv.WINDOW_NORMAL)
    cv.namedWindow("right", cv.WINDOW_NORMAL)
    cv.namedWindow("pred_disp_l", cv.WINDOW_NORMAL)
    cv.namedWindow("gt_disp", cv.WINDOW_NORMAL)
    cv.namedWindow("error", cv.WINDOW_NORMAL)
    cv.namedWindow("log_variance", cv.WINDOW_NORMAL)
    cv.namedWindow("sigma_e_approx", cv.WINDOW_NORMAL)

    for i in range(len(dataset)):
      inputs = dataset[i]
      save_filename = get_save_filename_for_dataset(lines[i], opt.dataset, scale=opt.scale)

      right_img = inputs["color_r/{}".format(opt.scale)]
      left_img = inputs["color_l/{}".format(opt.scale)]
      gt_disp = inputs["gt_disp_l/{}".format(opt.scale)]

      # NOTE: This will only work if this script has been run in SAVE MODE already.
      disp_path = os.path.join(save_folder_left, save_filename)
      print("Loading disp from {}".format(disp_path))
      disp = torch.load(disp_path)

      if os.path.exists(save_folder_variance_left):
        log_variance_path = os.path.join(save_folder_variance_left, save_filename)
        log_variance = torch.load(log_variance_path)
        log_variance_jet = visualize_disp_cv(log_variance)
        cv.imshow("log_variance", log_variance_jet)

      err = torch.abs(gt_disp - disp)
      err[gt_disp < 1e-3] = 0
      err_jet = visualize_disp_cv(err)

      disp_jet = visualize_disp_cv(disp, vmin=0, vmax=150)
      gt_disp_jet = visualize_disp_cv(gt_disp)

      sigma_e_approx = err - torch.exp(0.5 * log_variance)
      sigma_e_approx = sigma_e_approx.clamp(min=0)
      sigma_e_approx_jet = visualize_disp_cv(sigma_e_approx)

      cv.imshow("left", tensor_to_cv_rgb(left_img))
      cv.imshow("right", tensor_to_cv_rgb(right_img))
      cv.imshow("pred_disp_l", disp_jet)
      cv.imshow("gt_disp", gt_disp_jet)
      cv.imshow("error", err_jet)
      cv.imshow("sigma_e_approx", sigma_e_approx_jet)

      if opt.wait:
        cv.waitKey(0 if i == 0 else 1)
      else:
        cv.waitKey(0)

      if opt.save_disp_as_image:
        os.makedirs("/home/milo/Desktop/output_disp/", exist_ok=True)
        cv.imwrite("/home/milo/Desktop/output_disp/{}.png".format(i), disp_jet)

  # EVAL MODE: Compute KITTI depth evaluation metrics.
  elif opt.mode == "eval":
    with torch.no_grad():
      # NOTE: This will have to change once other datasets are used.
      camera_intrinsics = dataset.get_intrinsics(dataset.height, dataset.width)
      camera_baseline = dataset.get_baseline()

      metrics_each_image_2012 = []
      metrics_each_image_2015 = []

      for i, inputs in enumerate(dataset):
        print("Doing evaluation {}/{}".format(i, len(dataset)))
        save_filename = get_save_filename_for_dataset(lines[i], opt.dataset)

        gt_disp = inputs["gt_disp_l/0"]

        # NOTE: This will only work if this script has been run in SAVE MODE already.
        pred_disp_l = torch.load(os.path.join(save_folder_left, save_filename))

        gt_depth = disp_to_depth(gt_disp, camera_intrinsics[0,0], camera_baseline)
        pred_depth = disp_to_depth(pred_disp_l, camera_intrinsics[0,0], camera_baseline)

        gt_depth_valid = (gt_depth > MIN_DEPTH) * (gt_depth <  MAX_DEPTH)
        pred_depth = torch.clamp(pred_depth, MIN_DEPTH, MAX_DEPTH)

        gt_depth = gt_depth[gt_depth_valid]
        pred_depth = pred_depth[gt_depth_valid]

        # For each image, we get a tuple of (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3).
        all_errors = compute_depth_errors_2012(gt_depth, pred_depth)
        metrics_each_image_2012.append(list(all_errors))

        d1_all = compute_depth_errors_2015(gt_disp, pred_disp_l, 1e-3, 192)
        metrics_each_image_2015.append(list(d1_all))

      errors_2012 = torch.Tensor(metrics_each_image_2012)
      errors_2015 = torch.Tensor(metrics_each_image_2015)

      save_folder_metrics = os.path.join(opt.load_weights_folder, "metrics/{}".format(opt.split))
      os.makedirs(save_folder_metrics, exist_ok=True)

      torch.save(errors_2012, os.path.join(save_folder_metrics, "metrics_2012.pth"))
      torch.save(errors_2015, os.path.join(save_folder_metrics, "metrics_2015.pth"))

      with open(os.path.join(save_folder_metrics, "metrics_2012.csv"), "w") as f:
        f.write(",".join(["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]))
        f.write("\n")
        for m in errors_2012:
          f.write(",".join([str(s) for s in m.tolist()]))
          f.write("\n")

      with open(os.path.join(save_folder_metrics, "metrics_2015.csv"), "w") as f:
        f.write(",".join(["D1-all-3px", "D1-all-5px", "D1-all-10px"]))
        f.write("\n")
        for m in errors_2015:
          f.write(",".join([str(s) for s in m.tolist()]))
          f.write("\n")

      print("\n================ KITTI 2012 EVALUATION ====================")
      mean_errors_2012 = errors_2012.mean(axis=0)
      print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
      print(("&{: 8.3f}  " * 7).format(*mean_errors_2012.tolist()) + "\\\\")
      print("\n-> Done!")

      print("\n================ KITTI 2015 EVALUATION ====================")
      mean_errors_2015 = errors_2015.mean(axis=0)
      print("\n  " + ("{:>8} | " * 3).format("D1-all-3px", "D1-all-5px", "D1-all-10px"))
      print(("&{: 8.3f}  " * 3).format(*mean_errors_2015.tolist()) + "\\\\")

      with open(os.path.join(save_folder_metrics, "metrics_summary.csv"), "w") as f:
        f.write("\n================ KITTI 2012 EVALUATION ====================")
        f.write("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        f.write("\n")
        f.write(("&{: 8.3f}  " * 7).format(*mean_errors_2012.tolist()) + "\\\\")
        f.write("\n================ KITTI 2015 EVALUATION ====================")
        f.write("\n  " + ("{:>8} | " * 3).format("D1-all-3px", "D1-all-5px", "D1-all-10px"))
        f.write("\n")
        f.write(("&{: 8.3f}  " * 3).format(*mean_errors_2015.tolist()) + "\\\\")
        f.write("\n")

  else:
    raise ValueError("Invalid mode {} for evaluate_depth.py".format(opt.mode))


if __name__ == "__main__":
  opt = EvaluateOptions().parse()
  evaluate_depth(opt)

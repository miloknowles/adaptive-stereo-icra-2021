# Copyright 2020 Massachusetts Institute of Technology
#
# @file evaluate_variance.py
# @author Milo Knowles
# @date 2020-03-25 10:30:42 (Wed)

import os, math
import torch
from torch.distributions import Normal, Laplace
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from evaluate_options import EvaluateOptions
from evaluate_depth import get_save_filename_for_dataset

from utils.dataset_utils import read_lines, get_dataset
from utils.epistemic_error import *
from utils.loss_functions import *
from utils.visualization import *
from datasets.sceneflow_driving_dataset import SceneFlowDrivingDataset
from datasets.sceneflow_flying_dataset import SceneFlowFlyingDataset
from datasets.kitti_stereo_2015_dataset import KittiStereo2015Dataset
from datasets.kitti_stereo_full_dataset import KittiStereoFullDataset
from models.linear_warping import LinearWarping


def evaluate_variance(opt):
  """
  Evaluate variance prediction accuracy of a model.

  STEP 1: Run ./evaluate_flying save MGC-Net sceneflow_flying_240 ~/training_logs/mgc_flying_variance_08/models/weights_56
  """
  print("======== Running evaluation mode {} ========".format(opt.mode))
  dataset, lines = get_dataset(opt.dataset, opt.split, opt.dataset_path, opt.height, opt.width, opt.do_vertical_flip)

  load_model_filename = "madnet.pth"
  disp_save_folder_left = os.path.join(opt.load_weights_folder, "pred_disp_l/{}".format(opt.split))
  disp_save_folder_right = os.path.join(opt.load_weights_folder, "pred_disp_r/{}".format(opt.split))
  logvar_save_folder_left = os.path.join(opt.load_weights_folder, "pred_log_variance_l/{}".format(opt.split))
  logvar_save_folder_right = os.path.join(opt.load_weights_folder, "pred_log_variance_r/{}".format(opt.split))

  device = torch.device("cuda")

  # Compare the loss distributions for the "Supervised-SAD" (i.e MADNet loss) and "Supervised-NLL" loss.
  # This will process the split of images and save numpy files with the loss values for downstream plotting.
  if opt.mode == "save_loss_hist_supervised":
    with torch.no_grad():
      sup_mad_loss = torch.zeros(len(dataset))
      sup_nll_loss = torch.zeros(len(dataset))

      for i, inputs in enumerate(dataset):
        if i % 20 == 0:
          print("Processing {}/{}".format(i, len(dataset)))

        save_filename = get_save_filename_for_dataset(lines[i], opt.dataset, scale=opt.scale)

        gt_disp = inputs["gt_disp_l/{}".format(opt.scale)]
        pred_disp = torch.load(os.path.join(disp_save_folder_left, save_filename))
        pred_logvar = torch.load(os.path.join(logvar_save_folder_left, save_filename))
        pred_sigma  = torch.exp(0.5 * pred_logvar)

        has_valid_gt = (gt_disp > 0)

        # L = log(2) + log(sigma) + |x-u|/sigma
        sup_nll_loss_image = -1 * Laplace(pred_disp, pred_sigma).log_prob(gt_disp)

        # Need to collect loss for valid pixels only on KITTI.
        if "kitti" in opt.split:
          sup_nll_loss[i] = sup_nll_loss_image[has_valid_gt].mean()
          sup_mad_loss[i] = torch.abs(pred_disp - gt_disp)[has_valid_gt].mean()
        else:
          sup_nll_loss[i] = sup_nll_loss_image.mean()
          sup_mad_loss[i] = torch.abs(pred_disp - gt_disp).mean()

      output_folder = "../output/save_loss_hist_supervised/"
      os.makedirs(output_folder, exist_ok=True)
      np.save(os.path.join(output_folder, "sup_nll_loss_{}.npy".format(opt.split)), sup_nll_loss)
      np.save(os.path.join(output_folder, "sup_mad_loss_{}.npy".format(opt.split)), sup_mad_loss)
      print("Saved outputs to:", output_folder)

  # Compare the loss distributions for the "Monodepth" (unmodified) and "Monodepth-NLL" loss.
  # This will process the split of images and save numpy files with the loss values for downstream plotting.
  elif opt.mode == "save_loss_hist_monodepth":
    with torch.no_grad():
      warper = LinearWarping(opt.height, opt.width, device)

      cv.namedWindow("left", cv.WINDOW_NORMAL)
      cv.namedWindow("right", cv.WINDOW_NORMAL)
      cv.namedWindow("warped_l", cv.WINDOW_NORMAL)
      cv.namedWindow("disp_l", cv.WINDOW_NORMAL)
      cv.namedWindow("disp_err_l", cv.WINDOW_NORMAL)
      cv.namedWindow("logvar_l", cv.WINDOW_NORMAL)
      cv.namedWindow("photo_err_l", cv.WINDOW_NORMAL)

      monodepth_loss_all = []
      monodepth_ll_all = []

      md_loss = torch.zeros(len(dataset))
      md_nll_loss = torch.zeros(len(dataset))

      for i, inputs in enumerate(dataset):
        for key in inputs:
          inputs[key] = inputs[key].to(device).unsqueeze(0)
        if i % 20 == 0:
          print("Processing {}/{}".format(i, len(dataset)))

        save_filename = get_save_filename_for_dataset(lines[i], opt.dataset, scale=opt.scale)

        gt_disp_l = inputs["gt_disp_l/{}".format(opt.scale)]
        pred_disp_l = torch.load(os.path.join(disp_save_folder_left, save_filename)).to(device).unsqueeze(0)
        pred_disp_r = torch.load(os.path.join(disp_save_folder_right, save_filename)).to(device).unsqueeze(0)

        pred_logvar_l = torch.load(os.path.join(logvar_save_folder_left, save_filename)).to(device).unsqueeze(0)
        pred_logvar_r = torch.load(os.path.join(logvar_save_folder_right, save_filename)).to(device).unsqueeze(0)

        right_img = inputs["color_r/{}".format(opt.scale)]
        left_img = inputs["color_l/{}".format(opt.scale)]

        # Warp the left-centered disparity to the right and vice-versa. This allows us to detect
        # regions that are occluded in the left and right.
        warped_disp_l = warper(pred_disp_r, pred_disp_l, right_to_left=True)[0]
        warped_disp_r = warper(pred_disp_l, pred_disp_r, right_to_left=False)[0]

        # Each pixel is zero if there is an occlusion, one otherwise.
        occ_mask_l = pred_disp_l >= (0.9*warped_disp_l)
        occ_mask_r = pred_disp_r >= (0.9*warped_disp_r)

        left_warped, left_warped_mask = warper(right_img, pred_disp_l, right_to_left=True)
        right_warped, right_warped_mask = warper(left_img, pred_disp_r, right_to_left=False)

        loss_mask_left = occ_mask_l * left_warped_mask
        loss_mask_right = occ_mask_r * right_warped_mask

        # NOTE: Want to mask the photometric loss but NOT the smoothness loss!
        smoothness_weight = 1e-3
        consistency_weight = 1e-3

        L_total_left = monodepth_loss(pred_disp_l, left_img, left_warped, smoothness_weight=smoothness_weight)[0]
        L_total_right = monodepth_loss(pred_disp_r, right_img, right_warped, smoothness_weight=smoothness_weight)[0]

        l1_likelihood_left = laplace_likelihood_loss_image(left_img, left_warped, pred_logvar_l)
        l1_likelihood_right = laplace_likelihood_loss_image(right_img, right_warped, pred_logvar_r)

        L_consistency = (loss_mask_left*torch.abs(pred_disp_l - warped_disp_l)).mean() + \
                        (loss_mask_right*torch.abs(pred_disp_r - warped_disp_r)).mean()

        monodepth_loss_total = L_total_left[loss_mask_left].mean() + \
                              L_total_right[loss_mask_right].mean() + \
                              consistency_weight*L_consistency

        l1_likelihood_total = l1_likelihood_left[loss_mask_left].mean() + l1_likelihood_right[loss_mask_right].mean()

        md_loss[i] = monodepth_loss_total.mean().item()
        md_nll_loss[i] = l1_likelihood_total.mean().item()

        # disp_l_viz = visualize_disp_cv(pred_disp_l.cpu())
        # logvar_l_viz = visualize_disp_cv(pred_logvar_l.cpu())
        # disp_err_l_viz = visualize_disp_cv(torch.abs(pred_disp_l - gt_disp_l).cpu() * loss_mask_left.cpu())
        # warped_l_viz = tensor_to_cv_rgb((left_warped * loss_mask_left).squeeze(0).cpu())
        # photo_err_l_viz = visualize_disp_cv((torch.abs(left_img - left_warped) * loss_mask_left).mean(axis=1).cpu())

        # cv.imshow("left", tensor_to_cv_rgb(left_img.squeeze(0)))
        # cv.imshow("right", tensor_to_cv_rgb(right_img.squeeze(0)))
        # cv.imshow("disp_l", disp_l_viz)
        # cv.imshow("disp_err_l", disp_err_l_viz)
        # cv.imshow("warped_l", warped_l_viz)
        # cv.imshow("logvar_l", logvar_l_viz)
        # cv.imshow("photo_err_l", photo_err_l_viz)
        # cv.waitKey(0)

      os.makedirs("../output/save_loss_hist_monodepth", exist_ok=True)
      np.save("../output/save_loss_hist_monodepth/md_loss_{}.npy".format(opt.split), md_loss)
      np.save("../output/save_loss_hist_monodepth/md_nll_loss_{}.npy".format(opt.split), md_nll_loss)

  elif opt.mode == "analyze_photo_err":
    warper = LinearWarping(opt.height, opt.width, device)

    pixelwise_l1_loss = []
    pixelwise_ssim_loss = []
    pixelwise_combo_loss = []

    for i, inputs in enumerate(dataset):
      for key in inputs:
        inputs[key] = inputs[key].to(device).unsqueeze(0)
      if i % 10 == 0:
        print("Processing {}/{}".format(i, len(dataset)))

      save_filename = get_save_filename_for_dataset(lines[i], opt.dataset, scale=opt.scale)

      with torch.no_grad():
        gt_disp_l = inputs["gt_disp_l/{}".format(opt.scale)]
        pred_disp_l = torch.load(os.path.join(disp_save_folder_left, save_filename)).to(device).unsqueeze(0)
        pred_disp_r = torch.load(os.path.join(disp_save_folder_right, save_filename)).to(device).unsqueeze(0)

        pred_logvar_l = torch.load(os.path.join(logvar_save_folder_left, save_filename)).to(device).unsqueeze(0)
        pred_logvar_r = torch.load(os.path.join(logvar_save_folder_right, save_filename)).to(device).unsqueeze(0)

        right_img = inputs["color_r/{}".format(opt.scale)]
        left_img = inputs["color_l/{}".format(opt.scale)]

        # Warp the left-centered disparity to the right and vice-versa. This allows us to detect
        # regions that are occluded in the left and right.
        warped_disp_l = warper(pred_disp_r, pred_disp_l, right_to_left=True)[0]
        warped_disp_r = warper(pred_disp_l, pred_disp_r, right_to_left=False)[0]

        # Each pixel is zero if there is an occlusion, one otherwise.
        occ_mask_l = pred_disp_l >= (0.9*warped_disp_l)
        occ_mask_r = pred_disp_r >= (0.9*warped_disp_r)

        left_warped, left_warped_mask = warper(right_img, pred_disp_l, right_to_left=True)
        right_warped, right_warped_mask = warper(left_img, pred_disp_r, right_to_left=False)

        loss_mask_left = occ_mask_l * left_warped_mask
        loss_mask_right = occ_mask_r * right_warped_mask

        left_ssim = SSIM(left_img, left_warped).mean(axis=1, keepdims=True)
        right_ssim = SSIM(right_img, right_warped).mean(axis=1, keepdims=True)
        left_l1 = torch.abs(left_img - left_warped).mean(axis=1, keepdims=True)
        right_l1 = torch.abs(right_img - right_warped).mean(axis=1, keepdims=True)

        total_photo_left = 0.85*left_ssim + 0.15*left_l1
        total_photo_right = 0.85*right_ssim + 0.15*right_l1

        pixelwise_l1_loss.append(left_l1[loss_mask_left].cpu())
        pixelwise_l1_loss.append(right_l1[loss_mask_right].cpu())

        pixelwise_ssim_loss.append(left_ssim[loss_mask_left].cpu())
        pixelwise_ssim_loss.append(right_ssim[loss_mask_right].cpu())

        pixelwise_combo_loss.append(total_photo_left[loss_mask_left].cpu())
        pixelwise_combo_loss.append(total_photo_right[loss_mask_right].cpu())

    pixelwise_l1_loss = torch.cat(pixelwise_l1_loss).flatten()
    pixelwise_ssim_loss = torch.cat(pixelwise_ssim_loss).flatten()
    pixelwise_combo_loss = torch.cat(pixelwise_combo_loss).flatten()

    print("Got L1 loss values for {} pixels".format(len(pixelwise_l1_loss)))
    plt.hist(pixelwise_l1_loss, bins=100, density=True, facecolor="red", alpha=0.5)
    plt.show()

  elif opt.mode == "visualize_loss":
    cv.namedWindow("left", cv.WINDOW_NORMAL)
    cv.namedWindow("right", cv.WINDOW_NORMAL)
    cv.namedWindow("pred_disp_gauss", cv.WINDOW_NORMAL)
    cv.namedWindow("pred_disp_laplace", cv.WINDOW_NORMAL)
    cv.namedWindow("gt_disp", cv.WINDOW_NORMAL)
    cv.namedWindow("error_gauss", cv.WINDOW_NORMAL)
    cv.namedWindow("error_laplace", cv.WINDOW_NORMAL)
    cv.namedWindow("sigma_gauss", cv.WINDOW_NORMAL)
    cv.namedWindow("sigma_laplace", cv.WINDOW_NORMAL)
    cv.namedWindow("gauss_nll", cv.WINDOW_NORMAL)
    cv.namedWindow("laplace_nll", cv.WINDOW_NORMAL)

    save_folder_gauss_disp = os.path.join("/home/milo/training_logs/mgc_variance_flying_09/models/weights_126", "pred_disp/{}".format(opt.split))
    save_folder_laplace_disp = os.path.join("/home/milo/training_logs/mgc_var_fly_laplace_02/models/weights_122", "pred_disp/{}".format(opt.split))
    logvar_save_folder_left_gauss = os.path.join("/home/milo/training_logs/mgc_variance_flying_09/models/weights_126", "pred_log_variance/{}".format(opt.split))
    logvar_save_folder_left_laplace = os.path.join("/home/milo/training_logs/mgc_var_fly_laplace_02/models/weights_122", "pred_log_variance/{}".format(opt.split))

    for i, inputs in enumerate(dataset):
      if i % 10 == 0:
        print("Processing {}/{}".format(i, len(dataset)))
      save_filename = get_save_filename_for_dataset(lines[i], opt.dataset, scale=opt.scale)

      right_img = inputs["color_r/{}".format(opt.scale)]
      left_img = inputs["color_l/{}".format(opt.scale)]

      gt_disp = inputs["gt_disp_l/{}".format(opt.scale)]
      no_gt_disp = gt_disp == 0

      # Get predictions from the Gaussian model.
      pred_disp_gauss = torch.load(os.path.join(save_folder_gauss_disp, save_filename))
      pred_logvar_gauss = torch.load(os.path.join(logvar_save_folder_left_gauss, save_filename))
      pred_sigma_gauss = torch.exp(0.5 * pred_logvar_gauss)

      # Get predictions from the Laplacian model.
      pred_disp_laplace = torch.load(os.path.join(save_folder_laplace_disp, save_filename))
      pred_logvar_laplace = torch.load(os.path.join(logvar_save_folder_left_laplace, save_filename))
      pred_sigma_laplace = torch.exp(0.5 * pred_logvar_laplace)

      log_prob_gauss = -1 * Normal(pred_disp_gauss, pred_sigma_gauss).log_prob(gt_disp)
      log_prob_laplace = -1 * Laplace(pred_disp_laplace, pred_sigma_laplace).log_prob(gt_disp)
      print("Avg negative log-likelihood (Gaussian):\n  ", log_prob_gauss.mean())
      print("Avg negative log-likelihood (Laplacian):\n  ", log_prob_laplace.mean())

      # In order to compare the losses in a visually fair way, need to normalize both losses
      # by the same constants. Since the gaussian loss is worse, use its min and max for
      # normalization.
      min_loss, max_loss = log_prob_laplace.min(), log_prob_laplace.max()
      log_prob_gauss_jet = visualize_disp_cv(log_prob_gauss, vmin=min_loss, vmax=max_loss)

      log_prob_laplace[no_gt_disp] = 0
      log_prob_laplace_jet = visualize_disp_cv(log_prob_laplace, vmin=min_loss, vmax=max_loss)

      err_gauss = torch.abs(gt_disp - pred_disp_gauss)
      err_laplace = torch.abs(gt_disp - pred_disp_laplace)
      min_error = err_gauss.min()
      max_error = err_gauss.max()

      err_gauss_jet = visualize_disp_cv(err_gauss, vmin=None, vmax=None)
      err_laplace_jet = visualize_disp_cv(err_laplace, vmin=None, vmax=None)

      disp_gauss_jet = visualize_disp_cv(pred_disp_gauss)
      disp_laplace_jet = visualize_disp_cv(pred_disp_laplace)
      gt_disp_jet = visualize_disp_cv(gt_disp)

      sigma_gauss_jet = visualize_disp_cv(pred_sigma_gauss, vmin=0, vmax=0.4*max_error)
      sigma_laplace_jet = visualize_disp_cv(pred_sigma_laplace, vmin=0, vmax=0.4*max_error)

      cv.imshow("left", tensor_to_cv_rgb(left_img))
      cv.imshow("right", tensor_to_cv_rgb(right_img))
      cv.imshow("pred_disp_gauss", disp_gauss_jet)
      cv.imshow("pred_disp_laplace", disp_laplace_jet)
      cv.imshow("gt_disp", gt_disp_jet)
      cv.imshow("error_gauss", err_gauss_jet)
      cv.imshow("error_laplace", err_laplace_jet)
      cv.imshow("sigma_gauss", sigma_gauss_jet)
      cv.imshow("sigma_laplace", sigma_laplace_jet)
      cv.imshow("gauss_nll", log_prob_gauss_jet)
      cv.imshow("laplace_nll", log_prob_laplace_jet)
      cv.waitKey(0)

  elif opt.mode == "compare_loss":
    image_losses_gauss = torch.zeros(len(dataset))
    image_losses_laplace = torch.zeros(len(dataset))

    for i, inputs in enumerate(dataset):
      if i % 10 == 0:
        print("Processing {}/{}".format(i, len(dataset)))
      save_filename = get_save_filename_for_dataset(lines[i], opt.dataset, scale=opt.scale)

      gt_disp = inputs["gt_disp_l/{}".format(opt.scale)]
      pred_disp_l = torch.load(os.path.join(disp_save_folder_left, save_filename))

      logvar_path = os.path.join(logvar_save_folder_left, save_filename)
      pred_logvar = torch.load(logvar_path)
      pred_sigma = torch.exp(0.5 * pred_logvar)

      log_prob_gauss = -1 * Normal(pred_disp, pred_sigma).log_prob(gt_disp)
      log_prob_laplace = -1 * Laplace(pred_disp, pred_sigma).log_prob(gt_disp)

      image_losses_gauss[i] = log_prob_gauss.mean()
      image_losses_laplace[i] = log_prob_laplace.mean()

    print("Avg negative log-likelihood (Gaussian):\n  ", image_losses_gauss.mean())
    print("Avg negative log-likelihood (Laplacian):\n  ", image_losses_laplace.mean())

  elif opt.mode == "conditional_supervised":
    residuals = []
    stdevs = []

    with torch.no_grad():
      for i, inputs in enumerate(dataset):
        if i % 10 == 0:
          print("Processing {}/{}".format(i, len(dataset)))
        save_filename = get_save_filename_for_dataset(lines[i], opt.dataset, scale=opt.scale)

        gt_disp = inputs["gt_disp_l/{}".format(opt.scale)]
        pred_disp_l = torch.load(os.path.join(disp_save_folder_left, save_filename))

        logvar_path = os.path.join(logvar_save_folder_left, save_filename)
        pred_logvar = torch.load(logvar_path)

        residuals.append(pred_disp_l - gt_disp)
        stdevs.append(torch.exp(0.5 * pred_logvar))

      residuals = torch.cat(residuals).flatten()
      stdevs = torch.cat(stdevs).flatten()

      print("Standard deviation min={} max={}".format(stdevs.min(), stdevs.max()))

      os.makedirs("../output/supervised_variance_calibration/", exist_ok=True)
      print("Saving outputs to ../output/supervised_variance_calibration")

      epsilon = 0.005
      std_clamp_min = -5
      std_clamp_max = 5

      # sigma_range_gauss = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
      sigma_range_laplace = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0]

      # Generate Kolmogorov-Smirov plot.
      # candidate_dist = "laplace"
      # ks_stats = []
      # stdevs_to_test = np.linspace(0.5, 20, 500)
      # for sigma in stdevs_to_test:
      #   mask_in_sigma_bin = np.logical_and(stdevs >= (1-epsilon)*sigma, stdevs <= (1+epsilon)*sigma).bool()
      #   std_residuals = (residuals[mask_in_sigma_bin] / sigma)
      #   D, p = stats.kstest(std_residuals, candidate_dist, (0, 1))
      #   ks_stats.append(D)

      # ks_test_filename = "../output/supervised_variance_calibration/ks_test_stats_{}.npz".format(candidate_dist)
      # np.savez(ks_test_filename, K=np.array(ks_stats), stdev=stdevs_to_test)
      # print("Saved Kolmogorov-Smirnov values to {}".format(ks_test_filename))

      for sigma in sigma_range_laplace:
        mask_in_sigma_bin = np.logical_and(stdevs >= (1-epsilon)*sigma, stdevs <= (1+epsilon)*sigma).bool()
        num_datapoints = mask_in_sigma_bin.sum()
        print("Number of data points in bin (stdev={}):".format(sigma), num_datapoints)
        std_residuals = (residuals[mask_in_sigma_bin] / sigma)

        # Compute a Kolmogorov-Smirnov test.
        D_gauss, p_gauss = stats.kstest(std_residuals, 'norm', (0, 1))
        D_laplace, p_laplace = stats.kstest(std_residuals, 'laplace', (0, 1))
        print("Kolmogorov-Smirnov Test:")
        print("GAUSSIAN:  D={} p={}".format(D_gauss, p_gauss))
        print("LAPLACE:   D={} p={}".format(D_laplace, p_laplace))

        # std_residuals = std_residuals.clamp(std_clamp_min, std_clamp_max)
        std_residuals = std_residuals[std_residuals >= std_clamp_min * std_residuals <= std_clamp_max]

        plt.hist(std_residuals, 100, facecolor="red", alpha=0.5, density=True)
        x = np.linspace(std_clamp_min, std_clamp_max, 500)
        # plt.plot(x, stats.norm.pdf(x, loc=0, scale=1), linestyle="dashed", color="b", label="Ideal Gaussian")
        plt.plot(x, stats.laplace.pdf(x, loc=0, scale=1), linestyle="dashed", color="b", label="Ideal Laplacian")
        plt.title("Standardized Error Distribution (stdev={})".format(sigma))
        plt.xlabel("Standardized Error")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig("../output/supervised_variance_calibration/emp_err_sup_sigma_{}.png".format(sigma))
        plt.show()
        plt.clf()

  elif opt.mode == "conditional_monodepth":
    losses = []
    logvars = []

    warper = LinearWarping(opt.height, opt.width, device)

    for i, inputs in enumerate(dataset):
      for key in inputs:
        inputs[key] = inputs[key].to(device).unsqueeze(0)

      if i % 10 == 0:
        print("Processing {}/{}".format(i, len(dataset)))

      save_filename = get_save_filename_for_dataset(lines[i], opt.dataset, scale=opt.scale)

      with torch.no_grad():
        gt_disp_l = inputs["gt_disp_l/{}".format(opt.scale)]
        pred_disp_l = torch.load(os.path.join(disp_save_folder_left, save_filename)).to(device).unsqueeze(0)
        pred_disp_r = torch.load(os.path.join(disp_save_folder_right, save_filename)).to(device).unsqueeze(0)

        pred_logvar_l = torch.load(os.path.join(logvar_save_folder_left, save_filename)).to(device).unsqueeze(0)
        pred_logvar_r = torch.load(os.path.join(logvar_save_folder_right, save_filename)).to(device).unsqueeze(0)

        right_img = inputs["color_r/{}".format(opt.scale)]
        left_img = inputs["color_l/{}".format(opt.scale)]

        # Warp the left-centered disparity to the right and vice-versa. This allows us to detect
        # regions that are occluded in the left and right.
        warped_disp_l = warper(pred_disp_r, pred_disp_l, right_to_left=True)[0]
        warped_disp_r = warper(pred_disp_l, pred_disp_r, right_to_left=False)[0]

        # Each pixel is zero if there is an occlusion, one otherwise.
        occ_mask_l = pred_disp_l >= (0.9*warped_disp_l)
        occ_mask_r = pred_disp_r >= (0.9*warped_disp_r)

        left_warped, left_warped_mask = warper(right_img, pred_disp_l, right_to_left=True)
        right_warped, right_warped_mask = warper(left_img, pred_disp_r, right_to_left=False)

        loss_mask_left = occ_mask_l * left_warped_mask
        loss_mask_right = occ_mask_r * right_warped_mask

        # NOTE: Want to mask the photometric loss but NOT the smoothness loss!
        smoothness_weight = 1e-3
        consistency_weight = 1e-3

        l1_error_left = (left_img - left_warped).mean(axis=1, keepdim=True)
        l1_error_right = (right_img - right_warped).mean(axis=1, keepdim=True)

        logvars.append(pred_logvar_l[loss_mask_left].cpu())
        logvars.append(pred_logvar_r[loss_mask_right].cpu())
        losses.append(l1_error_left[loss_mask_left].cpu())
        losses.append(l1_error_right[loss_mask_right].cpu())

    losses = torch.cat(losses).flatten().numpy()
    logvars = torch.cat(logvars).flatten()

    print("Got loss values for {} pixels".format(len(losses)))
    print("Range of loss values: min={} max={}".format(losses.min(), losses.max()))

    epsilon = 0.005
    stdevs = torch.exp(0.5*logvars).numpy()

    candidate_dist = "norm"
    ks_stats = []
    stdevs_to_test = np.linspace(0.002, 0.05, 500)
    for sigma in stdevs_to_test:
      mask_in_sigma_bin = np.logical_and(stdevs >= (1-epsilon)*sigma, stdevs <= (1+epsilon)*sigma)
      std_losses = losses[mask_in_sigma_bin] / sigma
      D, p = stats.kstest(std_losses, candidate_dist, (0, 1))
      print(D, p)
      ks_stats.append(D)
    np.savez("../output/ks_test_stats_md_{}.npz".format(candidate_dist), K=np.array(ks_stats), stdev=stdevs_to_test)

    for sigma in [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05]:
      mask_in_sigma_bin = np.logical_and(stdevs >= (1-epsilon)*sigma, stdevs <= (1+epsilon)*sigma)

      num_datapoints = mask_in_sigma_bin.sum()
      print("Got {} datapoints for sigma={}".format(num_datapoints, sigma))

      std_losses = losses[mask_in_sigma_bin] / sigma
      std_losses = std_losses[np.logical_and(std_losses >= -5, std_losses <= 5)]
      # std_losses = std_losses.clamp(min=-5, max=5)
      plt.hist(std_losses, 50, facecolor="red", alpha=0.5, density=True, label="Empirical Error")
      x = np.linspace(std_losses.min(), std_losses.max(), 500)
      plt.plot(x, stats.laplace.pdf(x, loc=0, scale=1), color="b", linestyle="dashed", label="Ideal Laplace")
      plt.xlabel("Standardized Image Reconstruction Error")
      plt.ylabel("Frequency")
      plt.title("Standardized Error (stdev={})".format(sigma))
      plt.legend()
      plt.savefig("../output/emp_err_md_sigma_{}.png".format(sigma))
      plt.show()
      plt.clf()

  elif opt.mode == "plot_stdev_histogram":
    """
    Plots a histogram of the model's standard deviation predictions, aggregated across some split of images.
    """
    stdevs = []

    with torch.no_grad():
      for i, inputs in enumerate(dataset):
        if i % 20 == 0:
          print("Processing {}/{}".format(i, len(dataset)))
        save_filename = get_save_filename_for_dataset(lines[i], opt.dataset, scale=opt.scale)
        pred_logvar_l = torch.load(os.path.join(logvar_save_folder_left, save_filename)).cpu()
        pred_logvar_r = torch.load(os.path.join(logvar_save_folder_right, save_filename)).cpu()
        stdevs.append(torch.exp(0.5*pred_logvar_l))
        stdevs.append(torch.exp(0.5*pred_logvar_r))

      stdevs = torch.cat(stdevs).flatten()

    print("Standard deviation min={} max={}".format(stdevs.min(), stdevs.max()))

    plt.hist(stdevs.clamp(max=20), bins=100, density=True, facecolor="red", alpha=0.5)
    plt.title("Standard-Deviation Distribution (Laplace)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

  elif opt.mode == "priors":
    residuals = []
    stdevs = []

    for i, inputs in enumerate(dataset):
      if i % 10 == 0:
        print("Processing {}/{}".format(i, len(dataset)))
      save_filename = get_save_filename_for_dataset(lines[i], opt.dataset, scale=opt.scale)

      gt_disp = inputs["gt_disp_l/{}".format(opt.scale)]
      pred_disp_l = torch.load(os.path.join(disp_save_folder_left, save_filename))

      logvar_path = os.path.join(logvar_save_folder_left, save_filename)
      pred_logvar = torch.load(logvar_path)

      residuals.append(pred_disp_l - gt_disp)
      stdevs.append(torch.exp(0.5 * pred_logvar))

    residuals = torch.cat(residuals).flatten()
    stdevs = torch.cat(stdevs).flatten()

    std_clamp_min = -5
    std_clamp_max = 5
    std_residuals = (residuals / stdevs).clamp(std_clamp_min, std_clamp_max)

    N = len(residuals)
    assert(len(stdevs) == N)

    # Show the distribution of standardized residual errors.
    plt.hist(std_residuals, 500, facecolor="red", alpha=0.5, density=True, label="Residual Errors")
    x = np.linspace(std_clamp_min, std_clamp_max, 500)
    plt.plot(x, stats.norm.pdf(x, loc=0, scale=1), linestyle="dashed", color="b", label="Gaussian Prior")
    plt.plot(x, stats.laplace.pdf(x, loc=0, scale=1), linestyle="dashed", color="g", label="Laplacian Prior")
    plt.title("Distribution of Standardized Errors")
    plt.ylabel("Frequency")
    plt.xlabel("Disparity Error (px)")
    plt.legend()
    plt.show()

    # Show the distribution of standard deviations.
    stdev_clamp_min = 0
    stdev_clamp_max = 10
    plt.hist(stdevs.clamp(stdev_clamp_min, stdev_clamp_max), 500, facecolor="red", alpha=0.5, density=True, label="stdev")
    plt.title("Distribution of Predicted Standard Deviation")
    plt.ylabel("Frequency")
    plt.xlabel("Standard Deviation (px)")
    plt.show()

  else:
    raise ValueError("Invalid mode {} for evaluate_variance.py".format(opt.mode))


if __name__ == "__main__":
  opt = EvaluateOptions().parse()
  evaluate_variance(opt)

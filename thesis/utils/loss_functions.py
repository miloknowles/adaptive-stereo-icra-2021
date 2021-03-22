import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.laplace import Laplace


MATH_LOG_2 = math.log(2)
MATH_LOG_2PI = math.log(2 * math.pi)
MATH_LOG_SQRT_PI_2 = math.log(math.sqrt(math.pi) / math.sqrt(2))


def SSIM(x, y):
  """
  Structural similarity between two images x and y.

  Adapted from MonoDepth (https://github.com/nianticlabs/monodepth2/blob/master/layers.py).
  Written by @wng: https://github.com/robustrobotics/rrg/blob/master/src/perception/depthest/python/depthest/utils/losses.py
  """
  assert(len(x.shape) == 4) # (batch, channel, rows, cols)
  assert(len(y.shape) == 4) # (batch, channel, rows, cols)

  C1 = 0.01 ** 2
  C2 = 0.03 ** 2

  patch_size = 3
  padding = patch_size // 2

  mu_x = torch.nn.functional.avg_pool2d(x, patch_size, stride=1, padding=padding)
  mu_y = torch.nn.functional.avg_pool2d(y, patch_size, stride=1, padding=padding)

  sigma_x  = torch.nn.functional.avg_pool2d(x ** 2, patch_size, stride=1, padding=padding) - mu_x ** 2
  sigma_y  = torch.nn.functional.avg_pool2d(y ** 2, patch_size, stride=1, padding=padding) - mu_y ** 2
  sigma_xy = torch.nn.functional.avg_pool2d(x * y, patch_size, stride=1, padding=padding) - mu_x * mu_y

  ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
  ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

  ssim = ssim_n / ssim_d

  ssim = (1 - ssim) / 2
  ssim = ssim.clamp(min=0, max=1)

  return ssim


def edge_aware_smoothness_loss(disp, img):
  """
  Computes the smoothness loss for a disparity image. The color image is used for edge-aware smoothness.

  From MonoDepth: https://github.com/nianticlabs/monodepth2

  Args:
    disp (torch.Tensor) : Shape (b, 1, h, w).
    img (torch.Tensor) : Shape (b, 3, h, w).

  Returns:
    (torch.Tensor) of shape (b, 1, h, w)
  """
  grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
  grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

  grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
  grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

  grad_disp_x *= torch.exp(-grad_img_x)
  grad_disp_y *= torch.exp(-grad_img_y)

  # Pad the x-dim (last dim in tensor) with a column of zeros.
  grad_disp_x = F.pad(grad_disp_x, (0, 1), "constant", 0)

  # Pad the y-dim (second to last dim) with a row of zeros.
  grad_disp_y = F.pad(grad_disp_y, (0, 0, 0, 1), "constant", 0)

  return grad_disp_x + grad_disp_y


def monodepth_loss(pred_disp, true_img, warped_img, smoothness_weight=0.001):
  """
  From: https://github.com/nianticlabs/monodepth2

  Loss = 0.85*SSIM + 0.15*L1 + smoothness_weight*Edge_Aware_Smoothness

  Args:
    pred_disp (torch.Tensor) : Shape (b, 1, h, w).
    true_img (torch.Tensor) : Shape (b, 3, h, w).
    warped_img (torch.Tensor) : Shape (b, 3, h, w).
    smoothness_weight (float) : The smoothness loss weight (1e-3 in Monodepth paper).

  Returns:
    (torch.Tensor) of shape (b, 1, h, w) total loss
    (torch.Tensor) of shape (b, 1, h, w) l1 image reconstruction loss
    (torch.Tensor) of shape (b, 1, h, w) SSIM loss
    (torch.Tensor) of shape (b, 1, h, w) smoothness loss
  """
  # Take mean along the channel dim.
  photo_ssim = SSIM(true_img, warped_img).mean(axis=1, keepdims=True)
  photo_l1 = torch.abs(true_img - warped_img).mean(axis=1, keepdims=True)

  # Same weighting used in MADNet paper.
  L_photo = (0.85*photo_ssim + 0.15*photo_l1)

  # Take mean along the height and width dims.
  mean_disp = pred_disp.mean(2, True).mean(3, True)
  norm_disp = pred_disp / (mean_disp + 1e-7)
  L_smooth = edge_aware_smoothness_loss(norm_disp, true_img)

  L_total = L_photo + smoothness_weight*L_smooth

  return L_total, photo_l1, photo_ssim, L_smooth


def laplace_likelihood_loss_image(true_img, warped_img, pred_logvar_img):
  """
  Converts images to grayscale and then computes a Laplace negative log-likelihood loss.
  """
  return 0.5*pred_logvar_img + torch.exp(-0.5*pred_logvar_img)*torch.abs(true_img - warped_img).mean(axis=1, keepdims=True)


def madnet_loss(inputs, outputs):
  """
  Loss function described in MAD-Net (https://arxiv.org/abs/1810.05424).
  """
  losses = {"total_loss": 0}
  weights = [0.0001, 0, 0.005, 0.01, 0.02, 0.08, 0.32]
  scale_range = range(0, 7)

  for scale in scale_range:
    pred_disp_this_scale = outputs["pred_disp_l/{}".format(scale)]
    gt_disp_this_scale = inputs["gt_disp_l/{}".format(scale)]
    l1_loss_this_scale = weights[scale] * torch.abs(pred_disp_this_scale - gt_disp_this_scale)[gt_disp_this_scale > 1e-3].sum()
    losses["l1_loss/{}".format(scale)] = l1_loss_this_scale
    losses["total_loss"] += l1_loss_this_scale

  batch_size = inputs["gt_disp_l/0"].shape[0]
  losses["total_loss"] /= batch_size

  return losses


def mean_abs_diff_loss(inputs, outputs, scales=[0]):
  """
  Loss function descibed in GC-Net (https://arxiv.org/abs/1703.04309).
  """
  losses = {"total_loss": 0}
  weights = [0.5**scale for scale in range(7)]

  for scale in scales:
    pred_disp = outputs["pred_disp_l/{}".format(scale)]
    gt_disp = inputs["gt_disp_l/{}".format(scale)]

    gt_disp_valid = gt_disp > 0

    loss_this_scale = weights[scale] * torch.abs(pred_disp - gt_disp)[gt_disp_valid].mean()
    losses["mean_abs_diff_loss/{}".format(scale)] = loss_this_scale
    losses["total_loss"] += loss_this_scale

  return losses


def supervised_likelihood_loss(inputs, outputs, scale=2, distribution="laplacian"):
  """
  Compute the negative log-likelihood loss of the groundtruth disparity given predicted
  disparity mean and variance.

  NOTE(milo): I found that a Laplace likelihood matches the model's error distribution
  better than a Gaussian distribution.

  Gaussian:
    L(d; d_hat, sigma_hat) = constant + 1/2||d-d_hat||^2 / sigma_hat^2 + 1/2log(sigma_hat)

  Laplace:
    L(d; d_hat, sigma_hat) = constant + |d-d_hat| / sigma_hat + 1/2log(sigma_hat)
  """
  losses = {"total_loss": 0}

  # Compute likelihood loss at R2 to train variance predictions.
  # NOTE(milo): I found that this is more stable than R0 (which involves upsampling the
  # variance predictions), but haven't returned to R0 in a while to check if this is still the case.
  pred_disp = outputs["pred_disp_l/{}".format(scale)]
  pred_logvar = outputs["pred_log_variance_l/{}".format(scale)]
  gt_disp = inputs["gt_disp_l/{}".format(scale)]

  pdf = {"gaussian": Normal,
         "laplacian": Laplace}[distribution]

  outputs["likelihood_loss_l/{}".format(scale)] = pdf(gt_disp, pred_logvar).log_prob(pred_disp)
  losses["total_loss"] += outputs["likelihood_loss_l/{}".format(scale)].mean()

  return losses


def monodepth_leftright_loss(left_img, right_img, outputs, warper, scale):
  """
  Monodepth photometric loss with left-right consistency.
  From: https://github.com/nianticlabs/monodepth2

  Loss = 0.85*SSIM + 0.15*L1 + 0.001*Smoothness + 0.001*Consistency

  NOTE(milo): In Monodepth, they train with multiscale loss, but I found that after multiscale
  supervised pretraining, using Mondepth loss at only the highest resolution for fine-tuning
  works well.
  """
  losses = {"total_loss": 0}

  pred_disp_l = outputs["pred_disp_l/{}".format(scale)]
  pred_disp_r = outputs["pred_disp_r/{}".format(scale)]

  # Warp the left-centered disparity to the right and vice-versa. This allows us to detect
  # regions that are occluded in the left and right.
  warped_disp_l = warper(pred_disp_r, pred_disp_l, right_to_left=True)[0]
  warped_disp_r = warper(pred_disp_l, pred_disp_r, right_to_left=False)[0]

  # Each pixel is zero if there is an occlusion, one otherwise.
  occ_mask_l = pred_disp_l >= (0.95*warped_disp_l)
  occ_mask_r = pred_disp_r >= (0.95*warped_disp_r)

  left_warped, left_warped_mask = warper(right_img, pred_disp_l, right_to_left=True)
  right_warped, right_warped_mask = warper(left_img, pred_disp_r, right_to_left=False)

  loss_mask_left = occ_mask_l * left_warped_mask
  loss_mask_right = occ_mask_r * right_warped_mask
  loss_mask_left.detach_()
  loss_mask_right.detach_()
  del left_warped_mask, right_warped_mask, occ_mask_l, occ_mask_r

  # NOTE: Want to mask the photometric loss but NOT the smoothness loss!
  L_total_left = monodepth_loss(pred_disp_l, left_img, left_warped, smoothness_weight=1e-3)[0]
  L_total_right = monodepth_loss(pred_disp_r, right_img, right_warped, smoothness_weight=1e-3)[0]

  outputs["warp_l/0"] = left_warped * loss_mask_left
  outputs["warp_r/0"] = right_warped * loss_mask_right

  L_consistency = (loss_mask_left*torch.abs(pred_disp_l - warped_disp_l)).mean() + \
                  (loss_mask_right*torch.abs(pred_disp_r - warped_disp_r)).mean()
  losses["total_loss/0"] = L_total_left[loss_mask_left].mean() + \
                            L_total_right[loss_mask_right].mean() + \
                            1e-3*L_consistency

  losses["total_loss"] += losses["total_loss/0"]

  return losses, outputs


def monodepth_l1_likelihood_loss(pred_disp, true_img, warped_img, pred_logvar_img,
                                 smoothness_weight=0.001, likelihood_weight=0.02):
  """
  Uses the photometric loss from Monodepth, but with an L1 likelihood-loss.
  Loss = 0.85*SSIM + 0.15*L1_Likelihood + smoothness_weight*Edge_Aware_Smoothness
  Args:
    pred_disp (torch.Tensor) : Shape (b, 1, h, w).
    true_img (torch.Tensor) : Shape (b, 3, h, w).
    warped_img (torch.Tensor) : Shape (b, 3, h, w).
    pred_logvar_img (torch.Tensor) : Shape (b, 1, h, w), predicted log-variance over intensity.
    smoothness_weight (float) : The smoothness loss weight (1e-3 in MonoDepth paper).
  Returns:
    (torch.Tensor) of shape (b, 1, h, w) total loss
    (torch.Tensor) of shape (b, 1, h, w) image reconstruction likelihood loss
    (torch.Tensor) of shape (b, 1, h, w) SSIM loss
    (torch.Tensor) of shape (b, 1, h, w) smoothness loss
  """
  ssim_loss = SSIM(true_img, warped_img).mean(axis=1, keepdims=True)

  # NOTE: Using a Laplace likelihood here for now.
  nll_loss = 0.5*pred_logvar_img + torch.exp(-0.5*pred_logvar_img)*torch.abs(true_img - warped_img).mean(axis=1, keepdims=True)

  # Take mean along the height and width dims.
  mean_disp = pred_disp.mean(2, True).mean(3, True)
  norm_disp = pred_disp / (mean_disp + 1e-7)
  smooth_loss = edge_aware_smoothness_loss(norm_disp, true_img)

  # The SSIM loss is usually around 0.05 to 0.10, whereas the NLL is in the -5.0 to -1.0 range. The likelihood
  # weights should balance these losses.
  total_loss = (1-likelihood_weight)*ssim_loss + likelihood_weight*nll_loss + smoothness_weight*smooth_loss

  return total_loss, ssim_loss, nll_loss, smooth_loss


def monodepth_l1_likelihood_loss_leftright(inputs, outputs, warpers, opt):
  """
  Monodepth photometric loss with left-right consistency, but with an L1 image reconstruction loss
  likelihood term. The network predicts the log-variance of the intensity difference between the
  target image and the reconstructed target image.

  Loss = 0.85*SSIM + 0.15*L1_Likelihood + 0.001*Smoothness + 0.001*Consistency

  NOTE(milo): I've only tested this loss at the input resolution. I'm not sure how variance should be
  scaled when upsampling/downsampling to different pyramid levels. For disparity, it makes sense to
  scale by a factor of 2^2 since the disparity is scaled by a factor of 2, but I'm not sure what happens
  when you're resampling pixel intensities.
  """
  losses = {"total_loss": 0}
  scale = 0

  pred_disp_l = outputs["pred_disp_l/{}".format(scale)]
  pred_disp_r = outputs["pred_disp_r/{}".format(scale)]

  pred_logvar_l = outputs["pred_log_variance_l/{}".format(scale)]
  pred_logvar_r = outputs["pred_log_variance_r/{}".format(scale)]

  left_img = inputs["color_l/{}".format(scale)]
  right_img = inputs["color_r/{}".format(scale)]

  # Warp the left-centered disparity to the right and vice-versa. This allows us to detect
  # regions that are occluded in the left and right.
  warped_disp_l = warpers[scale](pred_disp_r, pred_disp_l, right_to_left=True)[0]
  warped_disp_r = warpers[scale](pred_disp_l, pred_disp_r, right_to_left=False)[0]

  # Each pixel is zero if there is an occlusion, one otherwise.
  occ_mask_l = pred_disp_l >= (0.9*warped_disp_l)
  occ_mask_r = pred_disp_r >= (0.9*warped_disp_r)

  left_warped, left_warped_mask = warpers[scale](right_img, pred_disp_l, right_to_left=True)
  right_warped, right_warped_mask = warpers[scale](left_img, pred_disp_r, right_to_left=False)

  loss_mask_left = occ_mask_l * left_warped_mask
  loss_mask_right = occ_mask_r * right_warped_mask
  loss_mask_left.detach_()
  loss_mask_right.detach_()
  del left_warped_mask, right_warped_mask, occ_mask_l, occ_mask_r

  # NOTE: Want to mask the photometric loss but NOT the smoothness loss!
  likelihood_loss_left = monodepth_l1_likelihood_loss(
      pred_disp_l, left_img, left_warped, pred_logvar_l,
      smoothness_weight=opt.smoothness_weight,
      likelihood_weight=opt.likelihood_weight)[0]
  likelihood_loss_right = monodepth_l1_likelihood_loss(
      pred_disp_r, right_img, right_warped, pred_logvar_r,
      smoothness_weight=opt.smoothness_weight,
      likelihood_weight=opt.likelihood_weight)[0]

  outputs["warp_l/{}".format(scale)] = left_warped * loss_mask_left
  outputs["warp_r/{}".format(scale)] = right_warped * loss_mask_right

  outputs["likelihood_loss_l/{}".format(scale)] = likelihood_loss_left * loss_mask_left
  outputs["likelihood_loss_r/{}".format(scale)] = likelihood_loss_right * loss_mask_right

  L_consistency = (loss_mask_left*torch.abs(pred_disp_l - warped_disp_l)).mean() + \
                  (loss_mask_right*torch.abs(pred_disp_r - warped_disp_r)).mean()

  losses["total_loss/{}".format(scale)] = likelihood_loss_left[loss_mask_left].mean() + \
                                          likelihood_loss_right[loss_mask_right].mean() + \
                                          opt.consistency_weight*L_consistency

  losses["total_loss"] += losses["total_loss/{}".format(scale)]

  return losses

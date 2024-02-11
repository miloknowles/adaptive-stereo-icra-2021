import torch
import torch.nn as nn
import torch.nn.functional as F


def khamis_robust_loss(pred_disp: torch.Tensor, gt_disp: torch.Tensor):
  """
  The two-parameter robust loss function used in StereoNet (Khamis et al., 2018).

  http://openaccess.thecvf.com/content_ECCV_2018/papers/Sameh_Khamis_StereoNet_Guided_Hierarchical_ECCV_2018_paper.pdf
  """
  mask = (gt_disp > 0)
  mask.detach_()
  num_valid = max(mask.sum(), 1)
  return torch.sum(torch.sqrt(torch.pow(gt_disp[mask] - pred_disp[mask], 2) + 4) / 2 - 1) / num_valid


def khamis_robust_loss_multiscale(inputs, outputs, scales=[0], gt_disp_scale=0):
  """
  Compute the multi-scale loss function used in StereoNet (Khamis et al., 2018).

  Each disparity map is upsampled to the original resolution to compute loss. In the paper,
  weighting seems to be equal for all predictions.

  http://openaccess.thecvf.com/content_ECCV_2018/papers/Sameh_Khamis_StereoNet_Guided_Hierarchical_ECCV_2018_paper.pdf
  """
  losses = {}
  losses["total_loss"] = 0

  for scale in scales:
    loss_this_scale = khamis_robust_loss(
      outputs["pred_disp_l/{}".format(scale)],
      inputs["gt_disp_l/{}".format(gt_disp_scale)]
    )
    losses["khamis_robust_loss/{}".format(scale)] = loss_this_scale
    losses["total_loss"] += loss_this_scale

  return losses


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


def monodepth_edge_aware_smoothness_loss(disp, img):
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
  L_smooth = monodepth_edge_aware_smoothness_loss(norm_disp, true_img)

  L_total = L_photo + smoothness_weight*L_smooth

  return L_total, photo_l1, photo_ssim, L_smooth


def monodepth_leftright_loss(left_img, right_img, outputs, warper, scale):
  """
  Monodepth photometric loss with left-right consistency.
  From: https://github.com/nianticlabs/monodepth2

  ```
  Loss = 0.85*SSIM + 0.15*L1 + 0.001*Smoothness + 0.001*Consistency
  ```
  
  NOTE(milo): In Monodepth, they train with multiscale loss, but I found that after multiscale
  supervised pretraining, using Mondepth loss at only the highest resolution for fine-tuning
  works well.
  """
  outputs = {}
  losses = {}

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

  outputs["left_warped/{}".format(scale)] = left_warped * loss_mask_left
  outputs["right_warped/{}".format(scale)] = right_warped * loss_mask_right

  L_consistency = (loss_mask_left*torch.abs(pred_disp_l - warped_disp_l)).mean() + \
                  (loss_mask_right*torch.abs(pred_disp_r - warped_disp_r)).mean()
  losses["Monodepth/total_loss"] = L_total_left[loss_mask_left].mean() + \
                                   L_total_right[loss_mask_right].mean() + \
                                   1e-3*L_consistency
  return losses, outputs

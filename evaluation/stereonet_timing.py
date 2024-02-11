import time

import torch
import torch.optim as optim

from adaptive_stereo.utils.loss_functions import monodepth_loss
from adaptive_stereo.models.stereo_net import StereoNet, FeatureExtractorNetwork
from adaptive_stereo.models.linear_warping import LinearWarping


def monodepth_single_loss(left_img, right_img, outputs, warper, scale):
  losses = {}
  left_warped, mask = warper(right_img, outputs["pred_disp_l/{}".format(scale)], right_to_left=True)
  losses["Monodepth/total_loss"] = monodepth_loss(
      outputs["pred_disp_l/{}".format(scale)],
      left_img, left_warped, smoothness_weight=1e-3)[0][mask].mean()

  outputs["left_warped/{}".format(scale)] = left_warped
  return losses


def run_inference_trials(N):
  feature_net = FeatureExtractorNetwork(4).cuda()
  stereo_net = StereoNet(4, 1, 0).cuda()

  feature_net.eval()
  stereo_net.eval()

  with torch.no_grad():
    iml = torch.zeros(1, 3, 320, 1216).cuda()
    imr = torch.zeros(1, 3, 320, 1216).cuda()

    t0 = time.time()
    for i in range(N):
      lf, rf = feature_net(iml), feature_net(imr)
      outputs = stereo_net(iml, lf, rf, "l", output_cost_volume=False)

    elap = time.time() - t0
    print("-------------------------------------------------------------------")
    print("-- Timing (inference only): %f sec (%f hz)" % (elap / N, N / elap))
    print("-------------------------------------------------------------------")


def run_backprop_trials(N):
  feature_net = FeatureExtractorNetwork(4).cuda()
  stereo_net = StereoNet(4, 1, 0).cuda()

  feature_net.train()
  stereo_net.train()

  parameters_to_train = [{"params": stereo_net.parameters()}, {"params": feature_net.parameters()}]
  optimizer = optim.Adam(parameters_to_train, lr=1e-4)

  warper = LinearWarping(320, 1216, torch.device("cuda"))

  iml = torch.zeros(1, 3, 320, 1216).cuda()
  imr = torch.zeros(1, 3, 320, 1216).cuda()

  t0 = time.time()
  for i in range(N):
    lf, rf = feature_net(iml), feature_net(imr)
    outputs = stereo_net(iml, lf, rf, "l", output_cost_volume=False)

    losses = monodepth_single_loss(iml, imr, outputs, warper, 0)
    optimizer.zero_grad()
    losses["Monodepth/total_loss"].backward()
    optimizer.step()

  elap = time.time() - t0
  print("-------------------------------------------------------------------")
  print("-- Timing (inference + backprop): %f sec (%f hz)" % (elap / N, N / elap))
  print("-------------------------------------------------------------------")



if __name__ == "__main__":
  run_inference_trials(100)
  run_backprop_trials(100)

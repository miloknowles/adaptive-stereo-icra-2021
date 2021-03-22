# Copyright 2020 Massachusetts Institute of Technology
#
# @file count_parameters.py
# @author Milo Knowles
# @date 2020-03-18 14:30:26 (Wed)

from models.madnet import MadNet
from models.mgc_net import MGCNet


# From: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
  nparams_mad = count_parameters(MadNet(2, 320, 1216))
  nparams_mgc = count_parameters(MGCNet(2, 320, 1216, predict_variance=False))
  nparams_mgc_v = count_parameters(MGCNet(2, 320, 1216, predict_variance=True))

  print("Number of trainable parameters:")
  print("MAD-Net ==> {}".format(nparams_mad))
  print("MGC-Net ==> {}".format(nparams_mgc))
  print("MGC-Net-V ==> {}".format(nparams_mgc_v))

# Copyright 2020 Massachusetts Institute of Technology
#
#	@file training.py
#	@author Milo Knowles
#	@date 2020-02-13 09:46:22 (Thu)

import os

import torch
import torch.nn.functional as F

from models.stereo_net import StereoNet


# From: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(model, optimizer, load_weights_folder, model_filename, optimizer_filename, ignore_prefixes=[]):
  """
  Load pretrained weights from a .pth file on disk.

  Args:
    model (PyTorch model) :
        Model to load pretrained weights for.
    optimizer (PyTorch optimizer) :
        Optimizer to (optionally) load state for.
    load_weights_folder (str) :
        Folder containing the .pth file.
    model_filename (str) :
        The model's filename (i.e madnet.pth).
    optimizer_filename (str) :
        If optimizer is not None, a filename to load the state from.
    ignore_prefixes (list of str) :
        If any of these strings are found in the name of pretrained parameters, don't load them.
  """
  load_weights_folder = os.path.expanduser(load_weights_folder)
  assert os.path.isdir(load_weights_folder), "Cannot find load_weights_folder {}".format(load_weights_folder)

  # Load the model weights.
  model_path = os.path.join(load_weights_folder, model_filename)
  model_dict = model.state_dict()
  pretrained_dict = torch.load(model_path)
  pretrained_dict_model = {k: v for k, v in pretrained_dict.items() if k in model_dict}
  model_dict.update(pretrained_dict_model)

  for pfx in ignore_prefixes:
    for key in list(model_dict.keys()):
      if pfx in key:
        del model_dict[key]
  model.load_state_dict(model_dict, strict=False)

  # Optionally load the optimizer state.
  if optimizer is not None:
    optimizer_path = os.path.join(load_weights_folder, optimizer_filename)
    optimizer_dict = torch.load(optimizer_path)
    optimizer.load_state_dict(optimizer_dict)

  return pretrained_dict

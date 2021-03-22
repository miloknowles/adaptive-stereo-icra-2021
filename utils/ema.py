# Copyright 2020 Massachusetts Institute of Technology
#
# @file ema.py
# @author Milo Knowles
# @date 2020-10-16 09:44:03 (Fri)


def online_ema(s_last, v_new, weight=0.999):
  """
  Updates an exponential moving average (EMA) with one new value. This is the same smoothing
  algorithm used in Tensorboard:

  https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar

  s_last (float) : The smoothed value at the last timestep.
  v_new (float) : The current value.
  """
  return s_last*weight + (1 - weight)*v_new

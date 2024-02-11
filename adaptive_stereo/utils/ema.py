def online_ema(s_last: float, v_new: float, weight=0.999) -> float:
  """
  Updates an exponential moving average (EMA) with one new value.
  
  This is the same smoothing algorithm used in Tensorboard:
  https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar

  Parameters
  ----------
  s_last (float) : The smoothed value at the last timestep.
  v_new (float) : The current value.
  """
  return s_last*weight + (1 - weight)*v_new

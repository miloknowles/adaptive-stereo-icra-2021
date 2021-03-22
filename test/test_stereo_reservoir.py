# Copyright 2020 Massachusetts Institute of Technology
#
# @file test_stereo_reservoir.py
# @author Milo Knowles
# @date 2020-10-06 12:56:20 (Tue)

import unittest, random
import numpy as np

from utils.stereo_reservoir import StereoReservoir


class StereoReservoirTest(unittest.TestCase):
  def test_01(self):
    random.seed(123)
    num_trials = 1000
    avgs = np.zeros(num_trials)

    for trial in range(num_trials):
      r = StereoReservoir(10) # Holds 10 items.
      for i in range(1000):
        r.add(None, None, i, i)
      avgs[trial] = r.average_value()

    print("Average index:", avgs.mean())

    # In expectation, the average value stored should be 500. Over a lot of trials, we should come
    # close to the expectation.
    self.assertTrue(abs(avgs.mean() - 500) < 5)


if __name__ == "__main__":
  unittest.main()

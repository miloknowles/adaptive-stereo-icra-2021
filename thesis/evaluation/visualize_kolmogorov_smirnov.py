# Copyright 2020 Massachusetts Institute of Technology
#
# @file visualize_kolmogorov_smirnov.py
# @author Milo Knowles
# @date 2020-05-18 10:19:14 (Mon)

import numpy as np
import matplotlib.pyplot as plt


def compare_ks_gauss_laplace_supervised():
  gauss_dict = np.load("../output/ks_test_stats_norm.npz", allow_pickle=True)
  laplace_dict = np.load("../output/ks_test_stats_laplace.npz", allow_pickle=True)

  assert((gauss_dict["stdev"] == laplace_dict["stdev"]).all())

  sigma = gauss_dict["stdev"]
  plt.plot(sigma, gauss_dict["K"], label="Gaussian Model", color="red")
  plt.plot(sigma, laplace_dict["K"], label="Laplace Model", color="green")
  plt.title("Kolmogorov-Smirnov Test")
  plt.ylabel("Badness of Fit (KS Statistic)")
  plt.xlabel("stdev")
  plt.legend()
  plt.show()


def compare_ks_gauss_laplace_monodepth():
  gauss_dict = np.load("../output/ks_test_stats_md_norm.npz", allow_pickle=True)
  laplace_dict = np.load("../output/ks_test_stats_md_laplace.npz", allow_pickle=True)

  assert((gauss_dict["stdev"] == laplace_dict["stdev"]).all())

  sigma = laplace_dict["stdev"]
  plt.plot(sigma, gauss_dict["K"], label="Gaussian Model", color="red")
  plt.plot(sigma, laplace_dict["K"], label="Laplace Model", color="green")
  plt.title("Kolmogorov-Smirnov Test")
  plt.ylabel("Badness of Fit (KS Statistic)")
  plt.xlabel("stdev")
  plt.legend()
  plt.show()


if __name__ == "__main__":
  # compare_ks_gauss_laplace_supervised()
  compare_ks_gauss_laplace_monodepth()

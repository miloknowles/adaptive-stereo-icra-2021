# Copyright 2020 Massachusetts Institute of Technology
#
# @file evaluate_options.py
# @author Milo Knowles
# @date 2020-05-18 10:19:24 (Mon)

import argparse


class EvaluateOptions(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser(description="Options for evaluating depth prediction")

    self.parser.add_argument("--network", type=str, default="MGC-Net", choices=["MGC-Net", "MAD-Net", "MGC-Net-Guided"])

    self.parser.add_argument("--mode", type=str,
                             choices=["save", "playback", "eval",
                                      "save_loss_hist_supervised", "save_loss_hist_monodepth",
                                      "priors", "conditional_supervised", "compare_loss", "visualize_loss",
                                      "variance_error", "conditional_monodepth", "analyze_photo_err",
                                      "plot_stdev_histogram"])

    self.parser.add_argument("--dataset_path", type=str, help="Top level folder for the dataset being used")
    self.parser.add_argument("--dataset", type=str, default="SceneFlowDrivingDataset", help="Which dataset to train on")
    self.parser.add_argument("--split", type=str, help="The dataset split to train on")
    self.parser.add_argument("--load_weights_folder", default=None, type=str, help="Path to load madnet.pth weights from")

    self.parser.add_argument("--radius_disp", type=int, default=2, help="Radius for disparity correction")
    self.parser.add_argument("--height", type=int, default=512, help="Image height (must be multiple of 64)")
    self.parser.add_argument("--width", type=int, default=960, help="Image width (must be multiple of 64)")

    self.parser.add_argument("--max_disp_viz", type=float, default=300, help="Divide the disparity by this to normalize it for viewing")
    self.parser.add_argument("--window_sf", type=float, default=1.5, help="Scale factor for OpenCV windows")
    self.parser.add_argument("--wait", default=False, action="store_true", help="If set, user must keypress through all images")

    self.parser.add_argument("--do_vertical_flip", default=False, action="store_true", help="Vertically flip input images")
    self.parser.add_argument("--predict_variance", default=False, action="store_true", help="Save variance predictions from the network (MGC-Net only)")

    self.parser.add_argument("--scale", type=int, default=0, help="Which scale to save network outputs at?")
    self.parser.add_argument("--leftright_consistency", action="store_true", default=False, help="Predict disparity for the left and right images")
    self.parser.add_argument("--variance_mode", default="disparity", choices=["disparity", "photometric"])
    self.parser.add_argument("--save_disp_as_image", action="store_true", default=False, help="Write color-mapped disparity to disk")

  def parse(self):
    self.options = self.parser.parse_args()
    return self.options

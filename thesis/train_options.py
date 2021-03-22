# Copyright 2020 Massachusetts Institute of Technology
#
# @file train_options.py
# @author Milo Knowles
# @date 2020-07-08 19:09:16 (Wed)

import argparse, os

FILE_DIR = os.path.dirname(__file__)


class TrainOptions(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser(description="Options for training Depth Networks")

    # Stereo architecture options.
    self.parser.add_argument("--radius_disp", type=int, default=2, help="Radius for disparity correction")
    self.parser.add_argument("--height", type=int, default=512, help="Image height (must be multiple of 64)")
    self.parser.add_argument("--width", type=int, default=960, help="Image width (must be multiple of 64)")
    self.parser.add_argument("--model_name", type=str, help="The name for this training experiment")
    self.parser.add_argument("--network", type=str, default="MGC-Net", choices=["MGC-Net", "MAD-Net"])

    # Dataset options.
    self.parser.add_argument("--dataset_path", type=str, help="Top level folder for the dataset being used")
    self.parser.add_argument("--dataset_name", type=str, default="SceneFlowDriving", help="Which dataset to train on")
    self.parser.add_argument("--split", type=str, help="The dataset split to train on")
    self.parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    self.parser.add_argument("--do_hflip", action="store_true", default=False, help="Do horizontal flip augmentation during training")
    self.parser.add_argument("--no_shuffle", action="store_true", default=False, help="Turn off dataset shuffling (i.e for sequential adaptation)")
    self.parser.add_argument("--use_grayscale", action="store_true", help="Convert training images to grayscale")

    # Output and saving options.
    self.parser.add_argument("--log_dir", type=str, default="/home/milo/training_logs", help="Directory for saving tensorboard events and models")
    self.parser.add_argument("--load_weights_folder", default=None, type=str, help="Path to load pretrained weights from")
    self.parser.add_argument("--dont_load_variance", default=False, action="store_true", help="Force random initialization of variance module weights")
    self.parser.add_argument("--dont_load_refinement", default=False, action="store_true", help="Force random initialization of the refinement module weights")
    self.parser.add_argument("--load_adam", action="store_true", default=False, help="Load the Adam optimizer state to resume training")
    self.parser.add_argument("--learning_rate", default=1e-5, type=float, help="Learning rate for Adam optimizer")
    self.parser.add_argument("--scheduler_step_size", default=10, type=int, help="Reduce LR by 0.1 after this many epochs")
    self.parser.add_argument("--num_workers", type=int, default=12, help="Number of dataloader workers")
    self.parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    self.parser.add_argument("--log_frequency", type=int, default=250, help="Number of batches between tensorboard logging")
    self.parser.add_argument("--save_freq", type=int, default=1, help="Save model weights after every x epochs")
    self.parser.add_argument("--fast_eval", action="store_true", default=False, help="Speeds up evaluation by only doing the first few batches")
    self.parser.add_argument("--training_eval", action="store_true", default=False, help="Log evaluation metrics on training batches")

    # Loss function options.
    self.parser.add_argument("--leftright_consistency", action="store_true", default=False, help="Predict disparity for the left and right images")
    self.parser.add_argument("--do_block_matching", action="store_true", default=False, help="Compute block matching disparity in dataset inputs")
    self.parser.add_argument("--predict_variance", action="store_true", help="Predict pixelwise disparity variance and train with likelihood loss")
    self.parser.add_argument("--loss_type", type=str, choices=["madnet", "supervised_likelihood", "monodepth", "mean_abs_diff", "khamis_robust_loss",
                                                               "monodepth_l1_likelihood"],
                             help="Loss function to use for training")
    self.parser.add_argument("--distribution", default="gaussian", choices=["gaussian", "laplacian"])
    self.parser.add_argument("--smoothness_weight", type=float, default=1e-3, help="Smoothness loss coefficient, as in MonoDepth2")
    self.parser.add_argument("--consistency_weight", type=float, default=1e-3, help="Left-right consistency loss coefficient, as in Monodepth1")
    self.parser.add_argument("--likelihood_weight", type=float, default=0.02, help="Should be between 0 and 1, controls relative contribution of SSIM and reconstruction likelihood")
    self.parser.add_argument("--scales", type=int, nargs="+", default=[0], help="Image scales used for the loss function")

    # Optimizer options.
    self.parser.add_argument("--clip_grad_norm", action="store_true", default=False, help="Clip gradient norms to 1.0")
    self.parser.add_argument("--variance_module_lr", type=float, default=1e-5, help="Specific learning rate for the variance module (every other layer uses the default LR")
    self.parser.add_argument("--freeze_disp", action="store_true", default=False, help="Freeze all network weights besides the variance module")
    self.parser.add_argument("--freeze_variance", action="store_true", default=False, help="Freeze the variance module weights")

    self.parser.add_argument("--train_dataset_path", type=str, help="Path to the training dataset folder")
    self.parser.add_argument("--train_dataset_name", type=str, help="Name of the training set")
    self.parser.add_argument("--train_split", type=str, help="Split to loading training validation images from")

  def parse(self):
    self.options = self.parser.parse_args()
    return self.options

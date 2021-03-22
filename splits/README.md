# Training Splits

Each subfolder contains the configuration files for training/validating/testing on various splits. Where valid, the subfolder will have a `train_lines.txt`, `val_lines.txt`, and `test_lines.txt`.

## SceneFlow Driving and Monkaa

The Driving and Monkaa sections of SceneFlow don't have a train/test split; I divided them into 70% train, and 15% validation, and 15% test. For the driving set, I use the only `35mm_focallength` folder, but both the forwards and backwards images. For the Monkaa set, I randomly shuffle all of the images in the dataset, which come from small sequences of 10-30ish images.

## SceneFlow Flying

I use the provided train/test split that comes with the dataset.

## KITTI Stereo 2012

394 training, 39 validation, and 194 testing images.

## KITTI Stereo 2015

400 training, 40 validation, and 200 testing images.

# Adaptation Splits

For adaptation, `train_lines.txt` and `val_lines.txt` should contain the same lines! After adapting to a sequence, we want to evaluate how well the model performs on that sequence as a whole.

## KITTI Raw Campus/City/Residential/Road

These splits were used for adaptation in the [https://arxiv.org/pdf/1810.05424.pdf](MADNet paper). The relevant folders for each environment are listed on the KITTI Raw website. Note that the groundtruth disparity for the KITTI Raw dataset was obtained using the `export_gt_disp.py` script.

## SceneFlow Driving

The 800 forward, slow stereo pairs.

The KITTI Stereo 2015 training set, but without random shuffling. This dataset isn't one continuous sequence, but has small subsequences of images within it.

## Virtual KITTI 01/20

These two sequences are the longest of six in the Virtual KITTI dataset. The lines correspond to the "clone" sequence (fog, rain, morning, etc. also exist).

## Experience Replay Splits

Any splits with the `*er_1000` suffix are for experience replay. I randomly sampled 1000 training images and put them in `train_lines.txt`.

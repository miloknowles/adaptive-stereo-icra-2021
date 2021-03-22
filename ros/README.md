# ROS Demo of StereoNet

This code was used to demonstrate learned depth estimation for the AIDTR project.

## Setup

All nodes were run using Ubuntu 18.04, ROS Melodic, and Python2.7.

## Running the Demo

To loop one of the stereo datasets:
```bash
python -m ros.test_image_publisher
```

The dataset can be chosen by editing `test_image_publisher.py` (see `main` method at the bottom).

Next, to predict disparity maps using `StereoNet` and publish a voxel map:
```bash
python -m ros.stereo_depth_node
```

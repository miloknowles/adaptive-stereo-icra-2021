# Datasets

# Download from Google Cloud

To download the datasets listed in this folder, check out the **Storage browser** on Google Cloud.

There, you should find:
- kitti_data_raw
- kitti_stereo_2012
- kitti_stereo_2015
- sceneflow_driving
- sceneflow_flying_things_3d
- sceneflow_monkaa
- virtual_kitti
- synthia_sf

To download an individual dataset to the current folder, use the command:
```bash
gsutil -m cp -r gs://name_of_storage_bucket .
```

The download should be really fast for a Google Cloud VM, especially in the same region (us-east) as the dataset.

NOTE: If you run into 404 errors you may have to run `gcloud auth login`.

# Implementing a new dataset

Although I originally subclassed new datasets from `StereoDataset`, I found that I was duplicating too much code. Instead, just add a few functions in `stereo_dataset.py` and `dataset_utils.py` to implement the disparity loading, intrinsics, and baseline for the new dataset.

# Download all of the stereo datasets from GCP (assuming they still exist there)...
mkdir ~/datasets/
gsutil -m cp -R gs://sceneflow_driving ~/datasets/
gsutil -m cp -R gs://sceneflow_flying_things_3d ~/datasets/
gsutil -m cp -R gs://kitti_data_raw ~/datasets/
gsutil -m cp -R gs://kitti_stereo_2012 ~/datasets/
gsutil -m cp -R gs://kitti_stereo_2015 ~/datasets/

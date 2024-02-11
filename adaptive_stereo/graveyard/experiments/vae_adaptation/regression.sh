SF_MODEL_PATH="/home/milo/rrg/src/perception/adaptive_stereo/resources/pretrained_models/stereo_net/flying_384x960_16X"
SF_VAE_PATH="/home/milo/training_logs/vae_flying_320x960_L2_64_02/models/weights_20/"
OOD_THRESHOLD=0

# ./adapt_nonstop.sh av_sf_to_vk01_nonstop virtual_kitti_01_adapt /home/milo/datasets/virtual_kitti/ VirtualKitti \
#     /home/milo/datasets/sceneflow_flying_things_3d/ SceneFlowFlying sceneflow_flying_100 \
#     $SF_MODEL_PATH $SF_VAE_PATH
./adapt_vs.sh av_sf_to_vk01_vs virtual_kitti_01_adapt /home/milo/datasets/virtual_kitti/ VirtualKitti \
    /home/milo/datasets/sceneflow_flying_things_3d/ SceneFlowFlying sceneflow_flying_100 \
    $SF_MODEL_PATH $OOD_THRESHOLD $SF_VAE_PATH
# ./adapt_er.sh adapt_flying_to_vk01_er virtual_kitti_01_adapt /home/milo/datasets/virtual_kitti/ VirtualKitti \
#     /home/milo/datasets/sceneflow_flying_things_3d/ SceneFlowFlying sceneflow_flying_100 \
#     $SF_MODEL_PATH $SF_VAE_PATH
# ./adapt_vs_er.sh adapt_flying_to_vk01_vs+er virtual_kitti_01_adapt /home/milo/datasets/virtual_kitti/ VirtualKitti \
#     /home/milo/datasets/sceneflow_flying_things_3d/ SceneFlowFlying sceneflow_flying_100 \
#     $SF_MODEL_PATH $OOD_THRESHOLD $SF_VAE_PATH

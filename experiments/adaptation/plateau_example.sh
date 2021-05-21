# SF_MODEL_PATH="/home/milo/training_logs/stereonet_flying_320x960_L0_16X_01/models/weights_3"
SF_MODEL_PATH="/home/milo/adaptive_stereo/resources/pretrained_models/stereo_net/flying_384x960_16X"
OOD_THRESHOLD=12.760914523256847

./adapt_nonstop.sh adapt_flying_to_vk01_nonstop virtual_kitti_01_adapt /home/milo/datasets/virtual_kitti/ VirtualKitti \
    /home/milo/datasets/sceneflow_flying_things_3d/ SceneFlowFlying sceneflow_flying_er_1000 \
    $SF_MODEL_PATH

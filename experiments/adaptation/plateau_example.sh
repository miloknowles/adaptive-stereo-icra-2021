SF_MODEL_PATH="/home/milo/adaptive_stereo/resources/pretrained_models/stereo_net/flying_384x960_16X"

# ./adapt_nonstop.sh plateau_example_adapt virtual_kitti_01_adapt /home/milo/datasets/virtual_kitti/ VirtualKitti \
#     /home/milo/datasets/sceneflow_flying_things_3d/ SceneFlowFlying sceneflow_flying_er_1000 \
#     $SF_MODEL_PATH

./adapt_none.sh plateau_example_baseline virtual_kitti_01_adapt /home/milo/datasets/virtual_kitti/ VirtualKitti \
    /home/milo/datasets/sceneflow_flying_things_3d/ SceneFlowFlying sceneflow_flying_er_1000 \
    $SF_MODEL_PATH

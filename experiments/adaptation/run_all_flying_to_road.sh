SF_MODEL_PATH="/home/milo/training_logs/stereonet_flying_320x960_L0_16X_01/models/weights_3"
VK_MODEL_PATH="/home/milo/training_logs/stereonet_clone_320x960_L0_16X_01/models/weights_39"
OOD_THRESHOLD=12.760914523256847

./adapt_nonstop.sh adapt_flying_to_road_nonstop kitti_raw_road_adapt /home/milo/datasets/kitti_data_raw/ KittiRaw \
    /home/milo/datasets/sceneflow_flying_things_3d/ SceneFlowFlying sceneflow_flying_er_1000 \
    $SF_MODEL_PATH
./adapt_vs.sh adapt_flying_to_road_vs kitti_raw_road_adapt /home/milo/datasets/kitti_data_raw/ KittiRaw \
    /home/milo/datasets/sceneflow_flying_things_3d/ SceneFlowFlying sceneflow_flying_er_1000 \
    $SF_MODEL_PATH $OOD_THRESHOLD
./adapt_er.sh adapt_flying_to_road_er kitti_raw_road_adapt /home/milo/datasets/kitti_data_raw/ KittiRaw \
    /home/milo/datasets/sceneflow_flying_things_3d/ SceneFlowFlying sceneflow_flying_er_1000 \
    $SF_MODEL_PATH
./adapt_vs_er.sh adapt_flying_to_road_vs+er kitti_raw_road_adapt /home/milo/datasets/kitti_data_raw/ KittiRaw \
    /home/milo/datasets/sceneflow_flying_things_3d/ SceneFlowFlying sceneflow_flying_er_1000 \
    $SF_MODEL_PATH $OOD_THRESHOLD

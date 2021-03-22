SF_MODEL_PATH="/home/milo/training_logs/stereonet_flying_320x960_L0_16X_01/models/weights_3"
VK_MODEL_PATH="/home/milo/training_logs/stereonet_clone_320x960_L0_16X_01/models/weights_39"
OOD_THRESHOLD=11.898818196844609

./adapt_nonstop.sh adapt_clone_to_rain_nonstop virtual_kitti_rain /home/milo/datasets/virtual_kitti/ VirtualKitti \
    /home/milo/datasets/virtual_kitti VirtualKitti virtual_kitti_clone_er_1000 \
    $VK_MODEL_PATH
./adapt_vs.sh adapt_clone_to_rain_vs virtual_kitti_rain /home/milo/datasets/virtual_kitti/ VirtualKitti \
    /home/milo/datasets/virtual_kitti VirtualKitti virtual_kitti_clone_er_1000 \
    $VK_MODEL_PATH $OOD_THRESHOLD
./adapt_er.sh adapt_clone_to_rain_er virtual_kitti_rain /home/milo/datasets/virtual_kitti/ VirtualKitti \
    /home/milo/datasets/virtual_kitti VirtualKitti virtual_kitti_clone_er_1000 \
    $VK_MODEL_PATH
./adapt_vs_er.sh adapt_clone_to_rain_vs+er virtual_kitti_rain /home/milo/datasets/virtual_kitti/ VirtualKitti \
    /home/milo/datasets/virtual_kitti VirtualKitti virtual_kitti_clone_er_1000 \
    $VK_MODEL_PATH $OOD_THRESHOLD

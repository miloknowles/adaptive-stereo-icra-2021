# FLYING (SUPERVISED)
# ./evaluate_flying.sh save MGC-Net sceneflow_flying_1000 ../resources/pretrained_models/flying_320x1216_supervised/weights_55/ disparity
# ./evaluate_flying.sh eval MGC-Net sceneflow_flying_1000 ../resources/pretrained_models/flying_320x1216_supervised/weights_55/ disparity

# FLYING (SELF-SUPERVISED)
# ./evaluate_flying.sh save MGC-Net sceneflow_flying_240 ../resources/pretrained_models/flying_320x1216_monodepth_nll/weights_6/ photometric
# ./evaluate_flying.sh eval MGC-Net sceneflow_flying_240 ../resources/pretrained_models/flying_320x1216_monodepth_nll/weights_6/ photometric

# DRIVING (SUPERVISED)
# ./evaluate_driving.sh save MGC-Net sceneflow_driving ../resources/pretrained_models/driving_320x1216_supervised/weights_1260 disparity
# ./evaluate_driving.sh eval MGC-Net sceneflow_driving ../resources/pretrained_models/driving_320x1216_supervised/weights_1260 disparity

# DRIVING (SELF-SUPERVISED)
./evaluate_driving.sh save MGC-Net sceneflow_driving ~/training_logs/monodepth_loss_driving_01/models/weights_164/ disparity
./evaluate_driving.sh eval MGC-Net sceneflow_driving ~/training_logs/monodepth_loss_driving_01/models/weights_164/ disparity

# KITTI 2015 (SELF-SUPERVISED)
# ./evaluate_2015.sh save MGC-Net kitti_stereo_2015 ~/training_logs/mgc_finetune_kitti_stereo_full_02/models/weights_1060 disparity
# ./evaluate_2015.sh eval MGC-Net kitti_stereo_2015 ~/training_logs/mgc_finetune_kitti_stereo_full_02/models/weights_1060 disparity

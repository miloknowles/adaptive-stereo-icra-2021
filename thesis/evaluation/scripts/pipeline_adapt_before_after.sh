# Supervised-SAD Loss ==> DRIVING
# ./evaluate_driving.sh save MGC-Net sceneflow_driving_forward_slow /home/milo/training_logs/adapt_fly_drive_sup_01/models/weights_before/ disparity
# ./evaluate_driving.sh save MGC-Net sceneflow_driving_forward_slow /home/milo/training_logs/adapt_fly_drive_sup_01/models/weights_after/ disparity
./evaluate_driving.sh playback MGC-Net sceneflow_driving_forward_slow /home/milo/training_logs/adapt_fly_drive_sup_01/models/weights_before/ disparity
./evaluate_driving.sh playback MGC-Net sceneflow_driving_forward_slow /home/milo/training_logs/adapt_fly_drive_sup_01/models/weights_after/ disparity

# Supervised-NLL Loss ==> DRIVING
# ./evaluate_driving.sh save MGC-Net sceneflow_driving_forward_slow /home/milo/training_logs/adapt_fly_drive_sup_ll_01/models/weights_before/ disparity
# ./evaluate_driving.sh save MGC-Net sceneflow_driving_forward_slow /home/milo/training_logs/adapt_fly_drive_sup_ll_01/models/weights_after/ disparity
# ./evaluate_driving.sh playback MGC-Net sceneflow_driving_forward_slow /home/milo/training_logs/adapt_fly_drive_sup_ll_01/models/weights_before/ disparity
# ./evaluate_driving.sh playback MGC-Net sceneflow_driving_forward_slow /home/milo/training_logs/adapt_fly_drive_sup_ll_01/models/weights_after/ disparity

# Supervised-NLL Loss ==> MONKAA
# ./evaluate_monkaa.sh save MGC-Net sceneflow_monkaa_400 /home/milo/training_logs/adapt_fly_monkaa_sup_ll_02/models/weights_before disparity
# ./evaluate_monkaa.sh save MGC-Net sceneflow_monkaa_400 /home/milo/training_logs/adapt_fly_monkaa_sup_ll_02/models/weights_after disparity
# ./evaluate_monkaa.sh playback MGC-Net sceneflow_monkaa_400 /home/milo/training_logs/adapt_fly_monkaa_sup_ll_02/models/weights_before disparity
# ./evaluate_monkaa.sh playback MGC-Net sceneflow_monkaa_400 /home/milo/training_logs/adapt_fly_monkaa_sup_ll_02/models/weights_after disparity

# Monodepth-NLL Loss ==> DRIVING
# ./evaluate_driving.sh save MGC-Net sceneflow_driving /home/milo/training_logs/adapt_fly_drive_md_conf_01/models/weights_before disparity
# ./evaluate_driving.sh save MGC-Net sceneflow_driving /home/milo/training_logs/adapt_fly_drive_md_conf_01/models/weights_after disparity

# ./evaluate_monkaa.sh save MGC-Net sceneflow_monkaa_400 /home/milo/training_logs/adapt_fly_monkaa_md_conf_02/models/weights_before disparity
# ./evaluate_monkaa.sh save MGC-Net sceneflow_monkaa_400 /home/milo/training_logs/adapt_fly_monkaa_md_conf_02/models/weights_after disparity

# Monodepth-NLL Loss ==> KITTI STEREO 2015
# ./evaluate_2015.sh save MGC-Net kitti_stereo_2015 /home/milo/training_logs/adapt_fly_2015_md_conf_02/models/weights_before photometric
# ./evaluate_2015.sh save MGC-Net kitti_stereo_2015 /home/milo/training_logs/adapt_fly_2015_md_conf_02/models/weights_after photometric
# ./evaluate_2015.sh playback MGC-Net kitti_stereo_2015 /home/milo/training_logs/adapt_fly_2015_md_conf_02/models/weights_before photometric
# ./evaluate_2015.sh playback MGC-Net kitti_stereo_2015 /home/milo/training_logs/adapt_fly_2015_md_conf_02/models/weights_after photometric

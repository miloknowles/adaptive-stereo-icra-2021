#  Evaluation for FLYING ==> DRIVING
# MD_FOLDER="/home/milo/training_logs/adapt_fly_drive_md_01"
# MD_WEIGHTS_BEFORE=${MD_FOLDER}/models/weights_before
# MD_WEIGHTS_AFTER=${MD_FOLDER}/models/weights_after

# LL_FOLDER="/home/milo/training_logs/adapt_fly_drive_conf_01"
# LL_WEIGHTS_BEFORE=${LL_FOLDER}/models/weights_before
# LL_WEIGHTS_AFTER=${LL_FOLDER}/models/weights_after

# ./evaluate_driving.sh save MGC-Net sceneflow_driving_forward_slow ${MD_WEIGHTS_BEFORE}
# ./evaluate_driving.sh eval MGC-Net sceneflow_driving_forward_slow ${MD_WEIGHTS_BEFORE}

# ./evaluate_driving.sh save MGC-Net sceneflow_driving_forward_slow ${LL_WEIGHTS_AFTER}
# ./evaluate_driving.sh eval MGC-Net sceneflow_driving_forward_slow ${LL_WEIGHTS_AFTER}

# Evaluation for FLYING ==> KITTI 2015
MD_FOLDER="/home/milo/training_logs/adapt_fly_2015_md_01"
MD_WEIGHTS_BEFORE=${MD_FOLDER}/models/weights_before
MD_WEIGHTS_AFTER=${MD_FOLDER}/models/weights_after

LL_FOLDER="/home/milo/training_logs/adapt_fly_2015_md_conf_01"
LL_WEIGHTS_BEFORE=${LL_FOLDER}/models/weights_before
LL_WEIGHTS_AFTER=${LL_FOLDER}/models/weights_after

./evaluate_kitti.sh save MGC-Net kitti_stereo_full ${MD_WEIGHTS_BEFORE}
./evaluate_kitti.sh eval MGC-Net kitti_stereo_full ${MD_WEIGHTS_BEFORE}
./evaluate_kitti.sh save MGC-Net kitti_stereo_full ${MD_WEIGHTS_AFTER}
./evaluate_kitti.sh eval MGC-Net kitti_stereo_full ${MD_WEIGHTS_AFTER}

./evaluate_kitti.sh save MGC-Net kitti_stereo_full ${LL_WEIGHTS_BEFORE}
./evaluate_kitti.sh eval MGC-Net kitti_stereo_full ${LL_WEIGHTS_BEFORE}
./evaluate_kitti.sh save MGC-Net kitti_stereo_full ${LL_WEIGHTS_AFTER}
./evaluate_kitti.sh eval MGC-Net kitti_stereo_full ${LL_WEIGHTS_AFTER}

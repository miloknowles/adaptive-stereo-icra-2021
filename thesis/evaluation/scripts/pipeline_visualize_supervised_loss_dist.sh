# FOLDER="/home/milo/training_logs/mgc_var_fly_laplace_02"
# EPOCH="122"
FOLDER="/home/milo/training_logs/mgcnet_var_sup_sceneflow_01"
EPOCH="23"
LOAD_WEIGHTS_PATH=${FOLDER}/models/weights_${EPOCH}

# Save disparity and variance outputs from the model.
# ./evaluate_flying.sh save MGC-Net sceneflow_flying_train_240 ${LOAD_WEIGHTS_PATH} disparity
# ./evaluate_flying.sh save MGC-Net sceneflow_flying_240 ${LOAD_WEIGHTS_PATH} disparity
./evaluate_driving.sh save MGC-Net sceneflow_driving ${LOAD_WEIGHTS_PATH} disparity
./evaluate_kitti.sh save MGC-Net kitti_stereo_full ${LOAD_WEIGHTS_PATH} disparity

# Save loss values for each image so that we can generate histograms.
# ./evaluate_variance_flying.sh save_loss_hist_supervised sceneflow_flying_train_240 ${LOAD_WEIGHTS_PATH}
./evaluate_variance_flying.sh save_loss_hist_supervised sceneflow_flying_240 ${LOAD_WEIGHTS_PATH}
./evaluate_variance_kitti.sh save_loss_hist_supervised kitti_stereo_full ${LOAD_WEIGHTS_PATH}
./evaluate_variance_driving.sh save_loss_hist_supervised sceneflow_driving ${LOAD_WEIGHTS_PATH}

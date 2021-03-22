FOLDER="/home/milo/training_logs/monodepth_l1_ll_flying_05"
EPOCH="6"
LOAD_WEIGHTS_PATH=${FOLDER}/models/weights_${EPOCH}

# Save disparity and variance outputs from the model.
# ./evaluate_flying.sh save MGC-Net sceneflow_flying_train_240 ${LOAD_WEIGHTS_PATH} photometric
# ./evaluate_flying.sh save MGC-Net sceneflow_flying_240 ${LOAD_WEIGHTS_PATH} photometric
# ./evaluate_driving.sh save MGC-Net sceneflow_driving ${LOAD_WEIGHTS_PATH} photometric
# ./evaluate_kitti.sh save MGC-Net kitti_stereo_full ${LOAD_WEIGHTS_PATH} photometric

# Save loss values for each image so that we can generate histograms.
./evaluate_variance_flying.sh save_loss_hist_monodepth sceneflow_flying_train_240 ${LOAD_WEIGHTS_PATH}
./evaluate_variance_flying.sh save_loss_hist_monodepth sceneflow_flying_240 ${LOAD_WEIGHTS_PATH}
./evaluate_variance_kitti.sh save_loss_hist_monodepth kitti_stereo_full ${LOAD_WEIGHTS_PATH}
./evaluate_variance_driving.sh save_loss_hist_monodepth sceneflow_driving ${LOAD_WEIGHTS_PATH}

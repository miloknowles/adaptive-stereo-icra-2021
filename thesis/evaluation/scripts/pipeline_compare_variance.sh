SUP_LAPLACE_WEIGHTS_PATH=/home/milo/training_logs/mgc_var_fly_laplace_02/models/weights_122
MD_LAPLACE_WEIGHTS_PATH=/home/milo/training_logs/monodepth_l1_ll_flying_05/models/weights_6

# Assume saving has been done already.
./evaluate_flying.sh playback MGC-Net sceneflow_flying_240 ${SUP_LAPLACE_WEIGHTS_PATH} disparity
# ./evaluate_flying.sh playback MGC-Net sceneflow_flying_240 ${MD_LAPLACE_WEIGHTS_PATH} photometric



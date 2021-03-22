# Saves model predictions (disparity mean and variance) and then generates variance calibration plots.

# GAUSS_WEIGHTS_PATH=/home/milo/training_logs/mgc_variance_flying_09/models/weights_126
# LAPLACE_WEIGHTS_PATH=/home/milo/training_logs/mgc_var_fly_laplace_02/models/weights_122
LAPLACE_WEIGHTS_PATH=/home/milo/training_logs/mgcnet_var_sup_sceneflow_01/models/weights_23

# Save disparity and variance predictions on split.
# ./evaluate_flying.sh save MGC-Net sceneflow_flying_240 ${GAUSS_WEIGHTS_PATH}
# ./evaluate_flying.sh save MGC-Net-Guided sceneflow_flying_240 ${LAPLACE_WEIGHTS_PATH} disparity

# Run the evaluation script.
# ./evaluate_variance_flying.sh conditional_supervised sceneflow_flying_240 ${GAUSS_WEIGHTS_PATH}
./evaluate_variance_flying.sh conditional_supervised sceneflow_flying_240 ${LAPLACE_WEIGHTS_PATH} disparity

# ./evaluate_variance_flying.sh plot_stdev_histogram sceneflow_flying_240 ${GAUSS_WEIGHTS_PATH}
# ./evaluate_variance_flying.sh plot_stdev_histogram sceneflow_flying_240 ${LAPLACE_WEIGHTS_PATH} disparity

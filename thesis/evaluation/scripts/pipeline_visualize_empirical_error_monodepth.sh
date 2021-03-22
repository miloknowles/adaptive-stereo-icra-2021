# Download the latest set of weights.
FOLDER="/home/milo/training_logs/monodepth_l1_ll_flying_05"
EPOCH="6"

# mkdir ${FOLDER}
# mkdir ${FOLDER}/models/

LOAD_WEIGHTS_PATH=${FOLDER}/models/weights_${EPOCH}

# gcloud compute scp --recurse milo@milo-pytorch-p100-vm-2:${FOLDER}/models/weights_${EPOCH} ${FOLDER}/models/

# Save disparity and variance predictions on split.
# ./evaluate_flying.sh save MGC-Net sceneflow_flying_240 ${LOAD_WEIGHTS_PATH}

# Run the evaluation script.
# ./evaluate_variance_flying.sh plot_stdev_histogram sceneflow_flying_240 ${LOAD_WEIGHTS_PATH}
./evaluate_variance_flying.sh conditional_monodepth sceneflow_flying_240 ${LOAD_WEIGHTS_PATH}

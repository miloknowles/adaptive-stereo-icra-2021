# Generates all of the histograms and precision recall curves used in the ICRA paper.
# NOTE: Must be run from top-level project directory!

# V2 Models
# SF_MODEL_PATH="/home/milo/training_logs/decoder_flying_368x960_16X_03/models/weights_4/"
# VK_MODEL_PATH="/home/milo/training_logs/decoder_vk_clone_368x960_16X_05/models/weights_99"

# V3 Models
# SF_MODEL_PATH="/home/milo/training_logs/decoder_flying_368x960_16X_04/models/weights_4/"
# VK_MODEL_PATH="/home/milo/training_logs/decoder_vk_clone_368x960_16X_06/models/weights_19/"

# V4 Models
SF_MODEL_PATH="/home/milo/training_logs/decoder_flying_368x960_16X_L2_01/models/weights_2/"
VK_MODEL_PATH="/home/milo/training_logs/decoder_vk_clone_368x960_16X_L2_01/models/weights_14/"

# python -m evaluation.ae_ood_analysis --load_weights_folder $SF_MODEL_PATH --environment sf_to_kitti --mode save # --normalize stdev
# python -m evaluation.ae_ood_analysis --environment sf_to_kitti --mode histogram
# python -m evaluation.ae_ood_analysis --environment sf_to_kitti --mode pr

python -m evaluation.ae_ood_analysis --load_weights_folder $VK_MODEL_PATH --environment vk_to_sf --mode save # --normalize stdev
python -m evaluation.ae_ood_analysis --environment vk_to_sf --mode histogram
python -m evaluation.ae_ood_analysis --environment vk_to_sf --mode pr

# python -m evaluation.ae_ood_analysis --load_weights_folder $SF_MODEL_PATH --environment sf_to_vk --mode save # --normalize stdev
# python -m evaluation.ae_ood_analysis --environment sf_to_vk --mode histogram
# python -m evaluation.ae_ood_analysis --environment sf_to_vk --mode pr

python -m evaluation.ae_ood_analysis --load_weights_folder $VK_MODEL_PATH --environment vk_to_kitti --mode save # --normalize stdev
python -m evaluation.ae_ood_analysis --environment vk_to_kitti --mode histogram
python -m evaluation.ae_ood_analysis --environment vk_to_kitti --mode pr

python -m evaluation.ae_ood_analysis --load_weights_folder $VK_MODEL_PATH --environment vk_weather --mode save # --normalize stdev
python -m evaluation.ae_ood_analysis --environment vk_weather --mode histogram
python -m evaluation.ae_ood_analysis --environment vk_weather --mode pr

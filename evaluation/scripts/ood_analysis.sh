# Generates all of the histograms and precision recall curves used in the ICRA paper.
# Also prints out the OOD thresholds for running adaptation experiments.
# NOTE: Must be run from top-level project directory!

SF_MODEL_PATH="/home/milo/training_logs/stereonet_flying_320x960_L0_16X_01/models/weights_3"
VK_MODEL_PATH="/home/milo/training_logs/stereonet_clone_320x960_L0_16X_01/models/weights_39"

python -m evaluation.ood_analysis --load_weights_folder $SF_MODEL_PATH --environment sf_to_kitti --save
python -m evaluation.ood_analysis --environment sf_to_kitti --histogram
python -m evaluation.ood_analysis --environment sf_to_kitti --pr

python -m evaluation.ood_analysis --load_weights_folder $SF_MODEL_PATH --environment sf_to_vk --save
python -m evaluation.ood_analysis --environment sf_to_vk --histogram
python -m evaluation.ood_analysis --environment sf_to_vk --pr

python -m evaluation.ood_analysis --load_weights_folder $VK_MODEL_PATH --environment vk_to_sf --save
python -m evaluation.ood_analysis --environment vk_to_sf --histogram
python -m evaluation.ood_analysis --environment vk_to_sf --pr

python -m evaluation.ood_analysis --load_weights_folder $VK_MODEL_PATH --environment vk_to_kitti --save
python -m evaluation.ood_analysis --environment vk_to_kitti --histogram
python -m evaluation.ood_analysis --environment vk_to_kitti --pr

python -m evaluation.ood_analysis --load_weights_folder $VK_MODEL_PATH --environment vk_weather --save
python -m evaluation.ood_analysis --environment vk_weather --histogram
python -m evaluation.ood_analysis --environment vk_weather --pr

python -m evaluation.ood_analysis --load_weights_folder $SD_MODEL_PATH --environment sd_to_vk --save
python -m evaluation.ood_analysis --environment sd_to_vk --histogram
python -m evaluation.ood_analysis --environment sd_to_vk --pr

python -m evaluation.ood_analysis --load_weights_folder $SD_MODEL_PATH --environment sd_to_kitti --save
python -m evaluation.ood_analysis --environment sd_to_kitti --histogram
python -m evaluation.ood_analysis --environment sd_to_kitti --pr

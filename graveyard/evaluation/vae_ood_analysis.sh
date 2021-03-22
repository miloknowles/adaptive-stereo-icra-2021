# Generates all of the histograms and precision recall curves used in the ICRA paper.
# NOTE: Must be run from top-level project directory!
SF_MODEL_PATH="/home/milo/training_logs/vae_flying_320x960_L2_64_02/models/weights_20"
VK_MODEL_PATH="/home/milo/training_logs/vae_clone_320x960_L2_512_01/models/weights_59"

python -m evaluation.vae_ood_analysis --load_weights_folder $SF_MODEL_PATH --environment sf_to_kitti --mode save_loss --vae_bottleneck 128
python -m evaluation.vae_ood_analysis --environment sf_to_kitti --mode histogram --alpha 1 --beta 0
# python -m evaluation.vae_ood_analysis --environment sf_to_kitti --mode pr

python -m evaluation.vae_ood_analysis --load_weights_folder $VK_MODEL_PATH --environment vk_to_sf --mode save_loss --vae_bottleneck 512
python -m evaluation.vae_ood_analysis --environment vk_to_sf --mode histogram --alpha 1 --beta 0
# python -m evaluation.vae_ood_analysis --environment vk_to_sf --mode pr

python -m evaluation.vae_ood_analysis --load_weights_folder $SF_MODEL_PATH --environment sf_to_vk --mode save_loss --vae_bottleneck 128
python -m evaluation.vae_ood_analysis --environment sf_to_vk --mode histogram --alpha 1 --beta 0
# python -m evaluation.vae_ood_analysis --environment sf_to_vk --mode pr

python -m evaluation.vae_ood_analysis --load_weights_folder $VK_MODEL_PATH --environment vk_to_kitti --mode save_loss --vae_bottleneck 512
python -m evaluation.vae_ood_analysis --environment vk_to_kitti --mode histogram --alpha 1 --beta 0
# python -m evaluation.vae_ood_analysis --environment vk_to_kitti --mode pr

python -m evaluation.vae_ood_analysis --load_weights_folder $VK_MODEL_PATH --environment vk_weather --mode save_loss --vae_bottleneck 512
python -m evaluation.vae_ood_analysis --environment vk_weather --mode histogram --alpha 1 --beta 0
# python -m evaluation.vae_ood_analysis --environment vk_weather --mode pr

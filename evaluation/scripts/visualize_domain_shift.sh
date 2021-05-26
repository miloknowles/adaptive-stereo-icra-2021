#============================ SAVE DATA ===============================
# echo "Saving data for TRAINING DOMAIN"
# python ../../evaluate_model.py \
#   --mode save \
#   --dataset_path ~/datasets/virtual_kitti/ \
#   --dataset_name VirtualKitti \
#   --split virtual_kitti_clone \
#   --load_weights_folder ../../resources/pretrained_models/stereo_net/vk_clone_368x960_16X \
#   --height 368 \
#   --width 960 \
#   --stereonet_k 4 \
#   --subsplit train \
#   --scales 0 \
#   --batch_size 8 \

# echo "Saving data for NOVEL DOMAIN"
# python ../../evaluate_model.py \
#   --mode save \
#   --dataset_path ~/datasets/virtual_kitti/ \
#   --dataset_name VirtualKitti \
#   --split virtual_kitti_rain \
#   --load_weights_folder ../../resources/pretrained_models/stereo_net/vk_clone_368x960_16X \
#   --height 368 \
#   --width 960 \
#   --stereonet_k 4 \
#   --subsplit train \
#   --scales 0 \
#   --batch_size 8 \

# python ../../evaluate_model.py \
#   --mode video \
#   --frames 100 \
#   --dataset_path ~/datasets/virtual_kitti/ \
#   --dataset_name VirtualKitti \
#   --split virtual_kitti_clone \
#   --load_weights_folder ../../resources/pretrained_models/stereo_net/vk_clone_368x960_16X \
#   --height 368 \
#   --width 960 \
#   --stereonet_k 4 \
#   --subsplit train \
#   --scales 0 \
#   --batch_size 1 \

# python ../../evaluate_model.py \
#   --mode video \
#   --frames 100 \
#   --dataset_path ~/datasets/virtual_kitti/ \
#   --dataset_name VirtualKitti \
#   --split virtual_kitti_rain \
#   --load_weights_folder ../../resources/pretrained_models/stereo_net/vk_clone_368x960_16X \
#   --height 368 \
#   --width 960 \
#   --stereonet_k 4 \
#   --subsplit train \
#   --scales 0 \
#   --batch_size 1 \

# Run these commands to generate video:
# convert -delay 5 -loop 0 pred*.png 0pred.gif
# convert -delay 5 -loop 0 left*.png 0left.gif
# https://askubuntu.com/questions/648244/how-do-i-create-an-animated-gif-from-still-images-preferably-with-the-command-l


python ../../evaluate_model.py \
  --mode playback \
  --dataset_path ~/datasets/virtual_kitti/ \
  --dataset_name VirtualKitti \
  --split virtual_kitti_clone \
  --load_weights_folder ../../resources/pretrained_models/stereo_net/vk_clone_368x960_16X \
  --height 368 \
  --width 960 \
  --stereonet_k 4 \
  --subsplit train \
  --scales 0 \
  --batch_size 1 \

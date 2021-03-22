#============================ SAVE DATA ===============================
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

# python ../../evaluate_model.py \
#   --mode save \
#   --dataset_path ~/datasets/virtual_kitti/ \
#   --dataset_name VirtualKitti \
#   --split virtual_kitti_clone \
#   --load_weights_folder ~/training_logs/adapt_fog_nonstop/models/weights_4000 \
#   --height 368 \
#   --width 960 \
#   --stereonet_k 4 \
#   --subsplit train \
#   --scales 0 \
#   --batch_size 8 \

# python ../../evaluate_model.py \
#   --mode save \
#   --dataset_path ~/datasets/virtual_kitti/ \
#   --dataset_name VirtualKitti \
#   --split virtual_kitti_clone \
#   --load_weights_folder ~/training_logs/adapt_fog_val_erb/models/weights_4000/ \
#   --height 368 \
#   --width 960 \
#   --stereonet_k 4 \
#   --subsplit train \
#   --scales 0 \
#   --batch_size 8 \

#========================= PLAYBACK DATA ==============================
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
  --batch_size 8 \

python ../../evaluate_model.py \
  --mode playback \
  --dataset_path ~/datasets/virtual_kitti/ \
  --dataset_name VirtualKitti \
  --split virtual_kitti_clone \
  --load_weights_folder ~/training_logs/adapt_fog_nonstop/models/weights_4000 \
  --height 368 \
  --width 960 \
  --stereonet_k 4 \
  --subsplit train \
  --scales 0 \
  --batch_size 8 \

python ../../evaluate_model.py \
  --mode playback \
  --dataset_path ~/datasets/virtual_kitti/ \
  --dataset_name VirtualKitti \
  --split virtual_kitti_clone \
  --load_weights_folder ~/training_logs/adapt_fog_val_erb/models/weights_4000 \
  --height 368 \
  --width 960 \
  --stereonet_k 4 \
  --subsplit train \
  --scales 0 \
  --batch_size 8 \

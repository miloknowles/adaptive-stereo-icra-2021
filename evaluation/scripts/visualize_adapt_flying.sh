#============================ SAVE DATA ===============================
# python ../../evaluate_model.py \
#   --mode save \
#   --dataset_path ~/datasets/sceneflow_flying_things_3d/ \
#   --dataset_name SceneFlowFlying \
#   --split sceneflow_flying_100 \
#   --load_weights_folder ~/training_logs/demo/vk20_none/ \
#   --height 368 \
#   --width 960 \
#   --stereonet_k 4 \
#   --subsplit train \
#   --scales 0 \
#   --batch_size 8 \

# python ../../evaluate_model.py \
#   --mode save \
#   --dataset_path ~/datasets/sceneflow_flying_things_3d/ \
#   --dataset_name SceneFlowFlying \
#   --split sceneflow_flying_100 \
#   --load_weights_folder ~/training_logs/demo/vk20_nonstop/ \
#   --height 368 \
#   --width 960 \
#   --stereonet_k 4 \
#   --subsplit train \
#   --scales 0 \
#   --batch_size 8 \

# python ../../evaluate_model.py \
#   --mode save \
#   --dataset_path ~/datasets/sceneflow_flying_things_3d/ \
#   --dataset_name SceneFlowFlying \
#   --split sceneflow_flying_100 \
#   --load_weights_folder ~/training_logs/demo/vk20_val_erb/ \
#   --height 368 \
#   --width 960 \
#   --stereonet_k 4 \
#   --subsplit train \
#   --scales 0 \
#   --batch_size 8 \

#========================= PLAYBACK DATA ==============================
# python ../../evaluate_model.py \
#   --mode playback \
#   --dataset_path ~/datasets/sceneflow_flying_things_3d/ \
#   --dataset_name SceneFlowFlying \
#   --split sceneflow_flying_100 \
#   --load_weights_folder ~/training_logs/demo/vk20_none/ \
#   --height 368 \
#   --width 960 \
#   --stereonet_k 4 \
#   --subsplit train \
#   --scales 0 \
#   --batch_size 8 \

python ../../evaluate_model.py \
  --mode playback \
  --dataset_path ~/datasets/sceneflow_flying_things_3d/ \
  --dataset_name SceneFlowFlying \
  --split sceneflow_flying_100 \
  --load_weights_folder ~/training_logs/demo/vk20_nonstop/ \
  --height 368 \
  --width 960 \
  --stereonet_k 4 \
  --subsplit train \
  --scales 0 \
  --batch_size 8 \

# python ../../evaluate_model.py \
#   --mode playback \
#   --dataset_path ~/datasets/sceneflow_flying_things_3d/ \
#   --dataset_name SceneFlowFlying \
#   --split sceneflow_flying_100 \
#   --load_weights_folder ~/training_logs/demo/vk20_val_erb/ \
#   --height 368 \
#   --width 960 \
#   --stereonet_k 4 \
#   --subsplit train \
#   --scales 0 \
#   --batch_size 8 \

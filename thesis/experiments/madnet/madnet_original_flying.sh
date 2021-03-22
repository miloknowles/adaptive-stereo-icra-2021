python ../../train.py --model_name madnet_original_flying_01 \
  --network MAD-Net \
  --split sceneflow_flying \
  --dataset_path /home/milo/datasets/sceneflow_flying_things_3d \
  --dataset SceneFlowFlying \
  --height 320 \
  --width 960 \
  --radius_disp 2 \
  --batch_size 32 \
  --log_dir ~/training_logs \
  --learning_rate 1e-4 \
  --num_workers 10 \
  --scheduler_step_size 40000 \
  --num_epochs 2000 \
  --log_frequency 1000 \
  --loss_type madnet \
  --save_freq 5 \
  --scales 0 1 2 3 4 5 6 \
  # --load_weights_folder ../../resources/pretrained_models/madnet/madnet_original_flying_01/models/weights_95/ \

SF_MODEL_PATH="/home/milo/rrg/src/perception/adaptive_stereo/resources/pretrained_models/stereo_net/flying_320x960_L1_8X/"

python ../../adapt.py --model_name stereonet_adapt_regression \
  --split virtual_kitti_01_adapt \
  --dataset_path /home/milo/datasets/virtual_kitti \
  --dataset_name VirtualKitti \
  --height 320 \
  --width 960 \
  --batch_size 1 \
  --learning_rate 5e-5 \
  --num_workers 2 \
  --scheduler_step_size 100000 \
  --num_epochs 1 \
  --log_frequency 20 \
  --load_weights_folder $SF_MODEL_PATH \
  --stereonet_k 3 \
  --stereonet_input_scale 1 \
  --clip_grad_norm \
  --eval_hz 100 \
  --num_steps 4000 \
  --ovs_buffer_size 8 \
  --ovs_validate_hz 20 \
  --val_improve_retries 2 \
  --adapt_mode VS+ER \
  --train_dataset_path /home/milo/datasets/sceneflow_flying_things_3d/ \
  --train_dataset_name SceneFlowFlying \
  --train_split sceneflow_flying_er_1000 \
  --er_loss_weight 0.05 \
  --skip_initial_eval \

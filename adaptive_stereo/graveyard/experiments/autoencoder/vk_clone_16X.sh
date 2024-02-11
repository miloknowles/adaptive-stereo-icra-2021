FEATURE_WEIGHTS_FOLDER="/home/milo/rrg/src/perception/adaptive_stereo/resources/pretrained_models/stereo_net/vk_clone_368x960_16X_scratch"

python ../../train_autoencoder.py \
  --name decoder_vk_clone_368x960_16X_L2_01 \
  --height 368 \
  --width 960 \
  --dataset_path /home/milo/datasets/virtual_kitti/ \
  --dataset_name VirtualKitti \
  --split virtual_kitti_clone_aug \
  --batch_size 32 \
  --do_hflip \
  --learning_rate 1e-3 \
  --num_workers 6 \
  --log_frequency 100 \
  --decoder_loss_scale 2 \
  --stereonet_k 4 \
  --encoder_type FeatureExtractorNetwork \
  --feature_weights_folder $FEATURE_WEIGHTS_FOLDER \
  --num_epochs 30 \

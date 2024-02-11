# FEATURE_WEIGHTS_FOLDER="/home/milo/rrg/src/perception/adaptive_stereo/resources/pretrained_models/stereo_net/flying_384x960_16X"
FEATURE_WEIGHTS_FOLDER="/home/milo/training_logs/decoder_vk_clone_368x960_16X_01/models/weights_25"
DECODER_WEIGHTS_FOLDER="/home/milo/training_logs/decoder_vk_clone_368x960_16X_01/models/weights_25"

python ../../train_autoencoder.py \
  --name decoder_vk_regression \
  --height 368 \
  --width 960 \
  --dataset_path /home/milo/datasets/virtual_kitti/ \
  --dataset_name VirtualKitti \
  --split virtual_kitti_clone \
  --batch_size 8 \
  --do_hflip \
  --learning_rate 1e-3 \
  --num_workers 4 \
  --log_frequency 100 \
  --fast_eval \
  --decoder_loss_scale 1 \
  --stereonet_k 4 \
  --encoder_type FeatureExtractorNetwork \
  --feature_weights_folder $FEATURE_WEIGHTS_FOLDER \
  # --decoder_weights_folder $DECODER_WEIGHTS_FOLDER \

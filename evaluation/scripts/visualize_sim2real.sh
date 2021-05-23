echo "Saving data for NOVEL DOMAIN"
python ../../evaluate_model.py \
  --mode save \
  --dataset_path /media/milo/MILO_SSD1/kitti_stereo_2015 \
  --dataset_name KittiStereo2015 \
  --split kitti_stereo_2015 \
  --load_weights_folder ../../resources/pretrained_models/stereo_net/vk_clone_368x960_16X \
  --height 368 \
  --width 960 \
  --stereonet_k 4 \
  --subsplit train \
  --scales 0 \
  --batch_size 8 \


python ../../evaluate_model.py \
  --mode playback \
  --dataset_path /media/milo/MILO_SSD1/kitti_stereo_2015 \
  --dataset_name KittiStereo2015 \
  --split kitti_stereo_2015 \
  --load_weights_folder ../../resources/pretrained_models/stereo_net/vk_clone_368x960_16X \
  --height 368 \
  --width 960 \
  --stereonet_k 4 \
  --subsplit train \
  --scales 0 \
  --batch_size 8 \

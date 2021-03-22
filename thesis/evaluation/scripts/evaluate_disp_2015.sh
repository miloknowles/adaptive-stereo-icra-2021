# SAVE EXAMPLE:
#   ./evaluate_disp_2015.sh save MGC-Net kitti_stereo_2015 ../../resources/pretrained_models/mgcnet/driving_320x1216_supervised/weights_1260/
# PLAY EXAMPLE:
#   ./evaluate_disp_2015.sh playback MGC-Net kitti_stereo_2015 ../../resources/pretrained_models/mgcnet/driving_320x1216_supervised/weights_1260/

python ../../evaluate_model.py \
  --mode $1 \
  --network $2 \
  --dataset_path /home/milo/datasets/kitti_stereo_2015 \
  --dataset KittiStereo2015 \
  --split $3 \
  --subsplit val \
  --height 320 \
  --width 1216 \
  --radius_disp 2 \
  --load_weights_folder $4 \
  --scales 0 \

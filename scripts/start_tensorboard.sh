# NOTE: The CLI $1 should be a MODEL_NAME
python /home/milo/.local/lib/python3.7/site-packages/tensorboard/main.py \
  --logdir=~/training_logs/$1 \
  --bind_all --port 6006

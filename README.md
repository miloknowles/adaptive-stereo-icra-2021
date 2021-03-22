# Toward Robust and Efficient Online Adaptation for Deep Stereo Depth Estimation

This is the code used in the ICRA 2021 paper draft.
- Pretrained models are located in `resources/pretrained_models`
- More information on dataset splits is in `splits/README.md`
- The `experiments` folder contains scripts for running training/adaptation
- Scripts used to generate figures are in `evaluation`

## Setup

This repository was tested with Ubuntu 18.04, Python3.6, and Pytorch 1.4.0.
```bash
# Install the dependencies:
pip install -r requirements36.txt

# (Optional) Download datasets from GCP storage buckets:
./scripts/download_datasets.sh
```

## Tests

Tests need to be run from the top-level directory:
```bash
python -m unittest test.test_stereo_net
```

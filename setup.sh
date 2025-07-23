#!/bin/bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
export HF_HOME="/mnt/projects/mlmi/dmcaf/models/hf_home"
export TORCH_HOME="/mnt/projects/mlmi/dmcaf/models/torch_home"
export MONAI_DATA_DIRECTORY="/mnt/projects/mlmi/dmcaf/models/monai_home/datasets"
export YOLO_CONFIG_DIR="/mnt/projects/mlmi/dmcaf/models/yolo_config"
export OUTPUT_DIRECTORY="/mnt/projects/mlmi/dmcaf/data"

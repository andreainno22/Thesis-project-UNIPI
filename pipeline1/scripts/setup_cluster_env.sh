#!/usr/bin/env bash
# Re-provision the P1 (YOLO) deps on the GPU container after a reset.
# torch/torchvision are baked into the image (survive resets); this script
# only needs to reinstall the lightweight extras that live in the
# container's writable layer (wiped on reset).
#
# Usage: bash pipeline1/scripts/setup_cluster_env.sh
set -euo pipefail

echo "[check] torch / CUDA"
python3 -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, '- CUDA OK')"

echo "[check] nvidia-smi"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo "[install] system libs opencv needs on a headless image (libGL, libglib)"
apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 > /dev/null

echo "[install] ultralytics + scikit-learn"
pip3 install --quiet ultralytics==8.4.89 scikit-learn

echo "[check] ultralytics import"
python3 -c "import ultralytics; print('ultralytics', ultralytics.__version__)"

echo "Done. Ready for prepare_yolo_dataset.py / train_yolo.py."

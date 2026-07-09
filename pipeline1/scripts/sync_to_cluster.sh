#!/usr/bin/env bash
# Sync the minimal P1 (YOLO) dataset + code subset to the university cluster.
# Excludes P3-only data (light_test, shadow_test, oggetti_fine_tuning source
# images, ostruzioni_reali_v1_backup, roi_ingegneria) to keep the transfer
# small (~620 MB instead of the full 1.5 GB Dataset/).
#
# Usage:
#   CLUSTER=user@cluster.unipi.it REMOTE_DIR=~/tesi_p1 ./pipeline1/scripts/sync_to_cluster.sh
set -euo pipefail

: "${CLUSTER:?Set CLUSTER=user@host}"
: "${REMOTE_DIR:?Set REMOTE_DIR=/path/on/cluster}"

ssh "$CLUSTER" "mkdir -p '$REMOTE_DIR'/Dataset '$REMOTE_DIR'/Aggregated_dataset_db"

rsync -avzP \
  Dataset/ostruzioni_reali/ \
  "$CLUSTER:$REMOTE_DIR/Dataset/ostruzioni_reali/"

rsync -avzP \
  Dataset/non_ostruite/ \
  "$CLUSTER:$REMOTE_DIR/Dataset/non_ostruite/"

rsync -avzP \
  Dataset/ostruzioni_gemini/ \
  "$CLUSTER:$REMOTE_DIR/Dataset/ostruzioni_gemini/"

rsync -avzP \
  Dataset/ostruzioni_poli_ingegneria/ \
  "$CLUSTER:$REMOTE_DIR/Dataset/ostruzioni_poli_ingegneria/"

rsync -avzP \
  Aggregated_dataset_db/occlusion.db \
  "$CLUSTER:$REMOTE_DIR/Aggregated_dataset_db/"

rsync -avzP \
  --exclude 'runs/' --exclude 'results/' --exclude 'data/' \
  pipeline1/ \
  "$CLUSTER:$REMOTE_DIR/pipeline1/"

echo "Done. On the cluster:"
echo "  cd $REMOTE_DIR"
echo "  conda activate tesi_env"
echo "  pip install ultralytics==8.4.89   # only new dep vs P3, see pipeline1/requirements_cluster.txt"
echo "  python pipeline1/src/prepare_yolo_dataset.py --db Aggregated_dataset_db/occlusion.db \\"
echo "      --dataset-root Dataset --out-dir pipeline1/data --folds 5 --seed 42"

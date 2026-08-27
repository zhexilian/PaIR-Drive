#!/usr/bin/env bash
set -euo pipefail

PAIRDRIVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================ USER CONFIG ============================
export PAIRDRIVE_ENV_NAME="pairdrive"
export OPENSCENE_DATA_ROOT="./dataset"
export NUPLAN_MAPS_ROOT="./dataset/maps"
export PAIRDRIVE_METRIC_CACHE="./outputs/metric_cache"
export PAIRDRIVE_FEATURE_CACHE="./outputs/dd_parallel_ir_cache"
export PAIRDRIVE_DD_CHECKPOINT="./checkpoints/diffusiondrive_navsim_88p1_PDMS.pth"
export PAIRDRIVE_PARALLEL_IR_CHECKPOINT="./checkpoints/parallel_ir.ckpt"
export PAIRDRIVE_DATA_SPLIT="test"
export PAIRDRIVE_DEVICE="auto"
export PAIRDRIVE_SCORING_WORKER="process"
export PAIRDRIVE_WORKERS="64"
export PAIRDRIVE_SEED="0"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
# ===================================================================

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PAIRDRIVE_ENV_NAME"

cd "$PAIRDRIVE_ROOT"
python -m pairdrive.evaluation.build_candidate_cache \
  --navsim-log-path "$OPENSCENE_DATA_ROOT/navsim_logs/$PAIRDRIVE_DATA_SPLIT" \
  --sensor-path "$OPENSCENE_DATA_ROOT/sensor_blobs/$PAIRDRIVE_DATA_SPLIT" \
  --metric-cache "$PAIRDRIVE_METRIC_CACHE" \
  --output-dir "$PAIRDRIVE_FEATURE_CACHE" \
  --base-checkpoint "$PAIRDRIVE_DD_CHECKPOINT" \
  --parallel-ir-checkpoint "$PAIRDRIVE_PARALLEL_IR_CHECKPOINT" \
  --device "$PAIRDRIVE_DEVICE" \
  --scoring-worker "$PAIRDRIVE_SCORING_WORKER" \
  --workers "$PAIRDRIVE_WORKERS" \
  --seed "$PAIRDRIVE_SEED"

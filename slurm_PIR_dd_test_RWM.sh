#!/usr/bin/env bash
set -euo pipefail

PAIRDRIVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================ USER CONFIG ============================
export PAIRDRIVE_ENV_NAME="pairdrive"
export NUPLAN_MAPS_ROOT="./dataset/maps"
export PAIRDRIVE_RWM_CHECKPOINT="./checkpoints/RWM.ckpt"
export PAIRDRIVE_METRIC_CACHE="./outputs/metric_cache"
export PAIRDRIVE_FEATURE_CACHE="./outputs/dd_parallel_ir_cache"
export PAIRDRIVE_OUTPUT_DIR="./outputs/evaluation/pair_ddrive"
export PAIRDRIVE_DEVICE="cpu"
export PAIRDRIVE_NUM_RANKS="32"
export PAIRDRIVE_TORCH_THREADS="4"
export PAIRDRIVE_PDM_WORKER="process"
export PAIRDRIVE_PDM_WORKERS="2"
export PAIRDRIVE_INFERENCE_BATCH_SIZE="4"
export OMP_NUM_THREADS="$PAIRDRIVE_TORCH_THREADS"
export MKL_NUM_THREADS="$PAIRDRIVE_TORCH_THREADS"
# ===================================================================

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PAIRDRIVE_ENV_NAME"

cd "$PAIRDRIVE_ROOT"
torchrun --standalone --nproc_per_node="$PAIRDRIVE_NUM_RANKS" \
  -m pairdrive.evaluation.run_rwm_pdm_score \
  --checkpoint "$PAIRDRIVE_RWM_CHECKPOINT" \
  --metric-cache "$PAIRDRIVE_METRIC_CACHE" \
  --feature-cache "$PAIRDRIVE_FEATURE_CACHE" \
  --output-dir "$PAIRDRIVE_OUTPUT_DIR" \
  --expected-base-model diffusiondrive \
  --device "$PAIRDRIVE_DEVICE" \
  --worker "$PAIRDRIVE_PDM_WORKER" \
  --workers "$PAIRDRIVE_PDM_WORKERS" \
  --inference-batch-size "$PAIRDRIVE_INFERENCE_BATCH_SIZE" \
  --torch-threads "$PAIRDRIVE_TORCH_THREADS"

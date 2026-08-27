#!/usr/bin/env bash
set -euo pipefail

PAIRDRIVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================ USER CONFIG ============================
export PAIRDRIVE_ENV_NAME="pairdrive"
export OPENSCENE_DATA_ROOT="./dataset"
export NUPLAN_MAPS_ROOT="./dataset/maps"
export NUPLAN_MAP_VERSION="./dataset/maps/nuplan-maps-v1.0.json"
export NAVSIM_EXP_ROOT="./outputs"
export PAIRDRIVE_METRIC_CACHE="./outputs/metric_cache"
export PAIRDRIVE_SPLIT="navtest"
export PAIRDRIVE_WORKER="single_machine_thread_pool"
export PAIRDRIVE_WORKERS="32"
export PAIRDRIVE_FORCE_CACHE="false"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
# ===================================================================

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PAIRDRIVE_ENV_NAME"

cd "$PAIRDRIVE_ROOT"
python -m navsim.planning.script.run_metric_caching \
  "train_test_split=$PAIRDRIVE_SPLIT" \
  "worker=$PAIRDRIVE_WORKER" \
  "worker.use_process_pool=true" \
  "worker.max_workers=$PAIRDRIVE_WORKERS" \
  "metric_cache_path=$PAIRDRIVE_METRIC_CACHE" \
  "force_feature_computation=$PAIRDRIVE_FORCE_CACHE"

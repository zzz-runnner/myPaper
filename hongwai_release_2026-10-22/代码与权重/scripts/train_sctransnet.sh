#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/envs/irstd/bin/python"
DATASET_NAME="${1:-SIRST3}"
DATASET_DIR="${IRSTD_DATASET_DIR:-/data1/irstd/datasets}"
RUN_DIR="${IRSTD_RUN_DIR:-$ROOT/runs/sctransnet}"
GPU_ID="${GPU:-0}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PATCH_SIZE="${PATCH_SIZE:-256}"
THREADS="${THREADS:-4}"
BEGIN_TEST="${BEGIN_TEST:-500}"
EVERY_TEST="${EVERY_TEST:-1}"
EVERY_PRINT="${EVERY_PRINT:-10}"
EVERY_SAVE_PTH="${EVERY_SAVE_PTH:-1000}"
SEED="${SEED:-42}"

mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/tensorboard"
cd "$ROOT/external/SCTransNet"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" train.py \
  --dataset_dir "$DATASET_DIR" \
  --dataset_names "$DATASET_NAME" \
  --model_names SCTransNet \
  --save "$RUN_DIR/checkpoints" \
  --log_dir "$RUN_DIR/tensorboard/${DATASET_NAME}_gpu${GPU_ID}" \
  --epochs "$EPOCHS" \
  --batchSize "$BATCH_SIZE" \
  --patchSize "$PATCH_SIZE" \
  --threads "$THREADS" \
  --begin_test "$BEGIN_TEST" \
  --every_test "$EVERY_TEST" \
  --every_print "$EVERY_PRINT" \
  --every_save_pth "$EVERY_SAVE_PTH" \
  --seed "$SEED"

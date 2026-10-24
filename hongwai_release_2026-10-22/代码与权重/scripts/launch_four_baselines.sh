#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${IRSTD_DATASET_DIR:-$ROOT/datasets}"
RUN_ROOT="${IRSTD_RUN_ROOT:-$ROOT/runs/baseline_4gpu}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PATCH_SIZE="${PATCH_SIZE:-256}"
THREADS="${THREADS:-4}"
BEGIN_TEST="${BEGIN_TEST:-500}"
EVERY_TEST="${EVERY_TEST:-1}"
EVERY_PRINT="${EVERY_PRINT:-10}"
EVERY_SAVE_PTH="${EVERY_SAVE_PTH:-1000}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/pids"

launch_one() {
  local dataset="$1"
  local gpu="$2"
  local tag="${dataset}_gpu${gpu}"
  local log="$RUN_ROOT/logs/${tag}.log"
  local pid_file="$RUN_ROOT/pids/${tag}.pid"

  if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "already running: $dataset on GPU $gpu pid $(cat "$pid_file")"
    return
  fi

  echo "launching: $dataset on GPU $gpu"
  nohup env \
    IRSTD_DATASET_DIR="$DATASET_DIR" \
    IRSTD_RUN_DIR="$RUN_ROOT/$dataset" \
    GPU="$gpu" \
    EPOCHS="$EPOCHS" \
    BATCH_SIZE="$BATCH_SIZE" \
    PATCH_SIZE="$PATCH_SIZE" \
    THREADS="$THREADS" \
    BEGIN_TEST="$BEGIN_TEST" \
    EVERY_TEST="$EVERY_TEST" \
    EVERY_PRINT="$EVERY_PRINT" \
    EVERY_SAVE_PTH="$EVERY_SAVE_PTH" \
    PYTHONUNBUFFERED=1 \
    bash "$ROOT/scripts/train_sctransnet.sh" "$dataset" >"$log" 2>&1 &
  echo "$!" > "$pid_file"
  echo "pid $(cat "$pid_file") log $log"
}

launch_one "SIRST3" 0
launch_one "NUDT-SIRST" 1
launch_one "IRSTD-1K" 2
launch_one "NUAA-SIRST" 3

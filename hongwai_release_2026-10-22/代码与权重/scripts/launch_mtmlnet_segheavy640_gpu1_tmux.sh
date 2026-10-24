#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
MTML="$ROOT/external/MTMLNet"
PY="$ROOT/envs/irstd/bin/python"
DATA_ROOT="$ROOT/generated/mtmlnet_full"
RUN_ROOT="$ROOT/runs/mtmlnet_full_4gpu"
LOG_DIR="$RUN_ROOT/logs"
SESSION="mtmlnet_segheavy640_gpu1"

mkdir -p "$LOG_DIR" "$RUN_ROOT"

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n gpu1 \
  "cd '$MTML' && \
   export PYTHONPATH='$MTML' && \
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
   CUDA_VISIBLE_DEVICES=1 '$PY' '$MTML/train.py' \
     --cfg '$MTML/models/detect/P234-MEA-HPDown-HFF.yaml' \
     --data '$DATA_ROOT/mtmlnet_full.yaml' \
     --hyp '$ROOT/configs/mtmlnet_hyp_adamw.yaml' \
     --weights '' \
     --project '$RUN_ROOT' \
     --name mtm_segheavy_s640_b2_adamw \
     --exist-ok \
     --device '' \
     --epochs 160 \
     --batch-size 2 \
     --imgsz 640 \
     --optimizer AdamW \
     --workers 4 \
     --patience 40 \
     --close-mosaic 20 \
     --seed 6113 \
     --det-gain 0.35 \
     --seg-gain 0.65 \
     --det-val-batch-mult 1.0 \
     --seg-val-batch 1 \
     --cos-lr \
     --noplots \
     2>&1 | tee '$LOG_DIR/mtm_segheavy_s640_b2_adamw_gpu1.log'"

tmux ls | grep "$SESSION"

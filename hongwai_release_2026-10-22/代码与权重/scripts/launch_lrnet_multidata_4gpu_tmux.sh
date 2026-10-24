#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cx/hongwai/irstd-lab
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_lrnet_multidata.py"
RUN_ROOT="$ROOT/runs/lrnet_multidata_4gpu"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"
SESSION=lrnet_multidata_4gpu

launch() {
  local name=$1
  local gpu=$2
  local norm=$3
  local batch=$4
  local seed=$5
  local pos_weight=$6
  local pos_prob=$7
  tmux new-window -t "$SESSION" -n "$name" \
    "cd '$ROOT' && export CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && '$PY' '$SCRIPT' \
      --train_datasets 'IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3,SIRST-UAVB' \
      --eval_datasets 'IRSTD-1K,NUAA-SIRST' \
      --norm_dataset '$norm' \
      --epochs 8 \
      --batch_size '$batch' \
      --patch_size 512 \
      --lr 0.0009 \
      --threshold 0.50 \
      --pos_weight '$pos_weight' \
      --pos_prob '$pos_prob' \
      --eval_interval 2 \
      --eval_max_samples 80 \
      --seed '$seed' \
      --device cuda:0 \
      --output_dir '$RUN_ROOT/$name' \
      2>&1 | tee '$LOG_DIR/$name.log'"
}

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n placeholder "sleep 2"
launch lrnet_irstd_a 0 IRSTD-1K 24 5201 70 0.70
launch lrnet_irstd_b 1 IRSTD-1K 32 5203 90 0.75
launch lrnet_nuaa_a 2 NUAA-SIRST 24 5205 70 0.70
launch lrnet_nuaa_b 3 NUAA-SIRST 32 5207 90 0.75
tmux kill-window -t "$SESSION:placeholder" 2>/dev/null || true
tmux ls | grep "$SESSION" || true

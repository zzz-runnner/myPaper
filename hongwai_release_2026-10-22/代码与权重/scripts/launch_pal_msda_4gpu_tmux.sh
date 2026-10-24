#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_pal_msda_multidata.py"
RUN_ROOT="$ROOT/runs/pal_msda_4gpu"
LOG_ROOT="$ROOT/reports/pal_msda_logs"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

launch() {
  local session="$1"
  local gpu="$2"
  local run_name="$3"
  local seed="$4"
  local lr="$5"
  local batch="$6"
  local repeat="$7"
  local pos="$8"
  local out="$RUN_ROOT/$run_name"
  local log="$LOG_ROOT/$run_name.log"

  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" \
    "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && CUDA_VISIBLE_DEVICES=$gpu '$PY' '$SCRIPT' \
      --dataset_dir '$ROOT/datasets' \
      --dataset_name NUAA-SIRST \
      --train_datasets 'IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3' \
      --model_name MSDA \
      --epochs 12 \
      --batch_size '$batch' \
      --patch_size 320 \
      --target_repeat '$repeat' \
      --lr '$lr' \
      --pos_weight '$pos' \
      --thresholds '0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90' \
      --eval_interval 2 \
      --seed '$seed' \
      --device cuda:0 \
      --amp \
      --output_dir '$out' \
      2>&1 | tee '$log'"
}

launch pal_msda_s301 0 nuaa_msda_s301 202606301 0.00030 16 8 25
launch pal_msda_s302 1 nuaa_msda_s302 202606302 0.00022 16 10 30
launch pal_msda_s303 2 nuaa_msda_s303 202606303 0.00016 16 12 35
launch pal_msda_s304 3 nuaa_msda_s304 202606304 0.00012 16 14 40

tmux ls

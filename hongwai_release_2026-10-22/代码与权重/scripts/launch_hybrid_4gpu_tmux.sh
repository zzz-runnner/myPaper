#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
SESSION="${SESSION:-irstd_hybrid4}"
DATASET_DIR="${IRSTD_DATASET_DIR:-$ROOT/datasets}"
RUN_ROOT="${IRSTD_HYBRID_RUN_ROOT:-$ROOT/runs/hybrid_4gpu}"
LOG_DIR="$RUN_ROOT/logs"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

launch() {
  local window="$1"
  local gpu="$2"
  local dataset="$3"
  local model="$4"
  local method="$5"
  local epochs="$6"
  local batch="$7"
  local lr="$8"
  local seed="$9"
  local extra="${10:-}"
  local run_dir="$RUN_ROOT/${dataset}_${method}_gpu${gpu}"
  local log="$LOG_DIR/${dataset}_${method}_gpu${gpu}.log"
  local cmd="cd '$ROOT' && CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' '$SCRIPT' --dataset_dir '$DATASET_DIR' --dataset_name '$dataset' --run_dir '$run_dir' --method_name '$method' --model '$model' --epochs '$epochs' --batch_size '$batch' --threads 4 --lr '$lr' --seed '$seed' --begin_test 20 --every_test 5 --every_print 10 $extra 2>&1 | tee '$log'"
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n "$window" "$cmd"
  else
    tmux new-window -t "$SESSION" -n "$window" "$cmd"
  fi
}

# Full-data runs. Not SCTransNet parameter sweeps:
# 1) BGM official-style BasicUNet_plus background masking line.
# 2) TCF-BGM innovation line: BGM backbone + FeatUp-style projection + background normality bank.
launch "irstd_bgm" 0 "IRSTD-1K" "bgm_basic" "BGM-BasicUNet" 400 24 0.0005 301 "--bgm_prob 0.35 --bgm_mask_ratio 0.12 --sls_iou_weight 0.10 --bg_hf_weight 0.18 --bg_pred_weight 0.05"
launch "irstd_tcf" 1 "IRSTD-1K" "tcf_bgm" "TCF-BGM" 400 18 0.00035 307 "--bgm_prob 0.30 --bgm_mask_ratio 0.10 --sls_iou_weight 0.14 --bg_hf_weight 0.24 --bg_pred_weight 0.07"
launch "nuaa_bgm" 2 "NUAA-SIRST" "bgm_basic" "BGM-BasicUNet" 400 24 0.0005 311 "--bgm_prob 0.35 --bgm_mask_ratio 0.12 --sls_iou_weight 0.10 --bg_hf_weight 0.18 --bg_pred_weight 0.05"
launch "nuaa_tcf" 3 "NUAA-SIRST" "tcf_bgm" "TCF-BGM" 400 18 0.00035 313 "--bgm_prob 0.30 --bgm_mask_ratio 0.10 --sls_iou_weight 0.14 --bg_hf_weight 0.24 --bg_pred_weight 0.07"

echo "Started tmux session: $SESSION"
echo "Run root: $RUN_ROOT"
tmux list-windows -t "$SESSION"

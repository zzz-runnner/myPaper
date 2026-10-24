#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
SESSION="${SESSION:-irstd_pconv_tcf_next}"
DATASET_DIR="${IRSTD_DATASET_DIR:-$ROOT/datasets}"
RUN_ROOT="${IRSTD_PCONV_RUN_ROOT:-$ROOT/runs/pconv_tcf_full}"
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
  local seed="$4"
  local batch="$5"
  local lr="$6"
  local run_dir="$RUN_ROOT/${dataset}_PConv-TCF_gpu${gpu}"
  local log="$LOG_DIR/${dataset}_PConv-TCF_gpu${gpu}.log"
  local cmd="cd '$ROOT' && CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' '$SCRIPT' --dataset_dir '$DATASET_DIR' --dataset_name '$dataset' --run_dir '$run_dir' --method_name 'PConv-TCF' --model 'pconv_tcf' --epochs 400 --batch_size '$batch' --threads 4 --lr '$lr' --seed '$seed' --begin_test 20 --every_test 5 --every_print 10 --bgm_prob 0.30 --bgm_mask_ratio 0.10 --sls_iou_weight 0.16 --sls_iou_warm_epochs 5 --bg_hf_weight 0.24 --bg_pred_weight 0.07 --pos_prob 0.62 --noise_std 0.004 --gamma_prob 0.10 --intensity_jitter 0.020 2>&1 | tee '$log'"
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n "$window" "$cmd"
  else
    tmux new-window -t "$SESSION" -n "$window" "$cmd"
  fi
}

launch "irstd_pconv" 0 "IRSTD-1K" 421 12 0.00030
launch "nuaa_pconv" 1 "NUAA-SIRST" 423 12 0.00030

echo "Prepared and started PConv-TCF session: $SESSION"
echo "Run root: $RUN_ROOT"
tmux list-windows -t "$SESSION"

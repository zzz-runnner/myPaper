#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
SESSION="${SESSION:-irstd_sct_refiner_next}"
DATASET_DIR="${IRSTD_DATASET_DIR:-$ROOT/datasets}"
RUN_ROOT="${IRSTD_REF_RUN_ROOT:-$ROOT/runs/sct_refiner_full}"
LOG_DIR="$RUN_ROOT/logs"
IRSTD_CKPT="${IRSTD_CKPT:-$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar}"
NUAA_CKPT="${NUAA_CKPT:-$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar}"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

launch() {
  local window="$1"
  local gpu="$2"
  local dataset="$3"
  local ckpt="$4"
  local seed="$5"
  local batch="$6"
  local lr="$7"
  local run_dir="$RUN_ROOT/${dataset}_SCT-Refiner_gpu${gpu}"
  local log="$LOG_DIR/${dataset}_SCT-Refiner_gpu${gpu}.log"
  local cmd="cd '$ROOT' && CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' '$SCRIPT' --dataset_dir '$DATASET_DIR' --dataset_name '$dataset' --run_dir '$run_dir' --method_name 'SCT-Refiner' --model 'sct_refiner' --base_checkpoint '$ckpt' --epochs 120 --batch_size '$batch' --threads 4 --lr '$lr' --weight_decay 0.00005 --seed '$seed' --begin_test 1 --every_test 1 --every_print 10 --bgm_prob 0.12 --bgm_mask_ratio 0.08 --sls_iou_weight 0.08 --sls_iou_warm_epochs 1 --bg_hf_weight 0.18 --bg_pred_weight 0.05 --pos_prob 0.55 --noise_std 0.001 --gamma_prob 0.03 --intensity_jitter 0.006 2>&1 | tee '$log'"
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n "$window" "$cmd"
  else
    tmux new-window -t "$SESSION" -n "$window" "$cmd"
  fi
}

launch "irstd_ref" 0 "IRSTD-1K" "$IRSTD_CKPT" 501 8 0.00012
launch "nuaa_ref" 1 "NUAA-SIRST" "$NUAA_CKPT" 503 8 0.00012

echo "Prepared and started SCT-Refiner session: $SESSION"
echo "Run root: $RUN_ROOT"
tmux list-windows -t "$SESSION"

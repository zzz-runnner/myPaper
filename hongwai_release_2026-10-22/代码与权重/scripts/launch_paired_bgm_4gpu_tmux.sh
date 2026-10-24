#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/paired_bgm_4gpu"
LOG_DIR="$RUN_ROOT/logs"
SESSION="paired_bgm_4gpu"

mkdir -p "$LOG_DIR"
tmux kill-session -t "$SESSION" 2>/dev/null || true

common_args=(
  --dataset_dir "$ROOT/datasets"
  --train_datasets "IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3"
  --epochs 220
  --batch_size 18
  --threads 4
  --optimizer adamw
  --weight_decay 0.00005
  --warmup_epochs 3
  --min_lr_ratio 0.12
  --begin_test 10
  --every_test 10
  --every_print 5
  --paired_bgm
  --bgm_prob 0.0
  --bgm_region_size 16
  --bgm_mask_ratio 0.20
  --noise_std 0.0008
  --gamma_prob 0.018
  --intensity_jitter 0.004
  --bce_weight 0.35
  --dice_weight 0.15
  --soft_iou_weight 0.80
  --focal_weight 0.015
  --sls_iou_weight 0.020
  --sls_iou_warm_epochs 5
  --bg_hf_weight 0.035
  --bg_pred_weight 0.006
)

run_cmd() {
  local gpu="$1"
  local dataset="$2"
  local target_repeat="$3"
  local model="$4"
  local method="$5"
  local batch="$6"
  local lr="$7"
  local seed="$8"
  local pos_prob="$9"
  local run_dir="$RUN_ROOT/${dataset}_${method}_gpu${gpu}"

  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT" "${common_args[@]}" \
    --dataset_name "$dataset" \
    --target_repeat "$target_repeat" \
    --model "$model" \
    --run_dir "$run_dir" \
    --method_name "$method" \
    --batch_size "$batch" \
    --lr "$lr" \
    --seed "$seed" \
    --pos_prob "$pos_prob"
}

tmux new-session -d -s "$SESSION" -n irstd_bgm_pair "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 0 IRSTD-1K 5 bgm_basic BGM-PairedBasicUNet 8 0.00045 4111 0.55 2>&1 | tee '$LOG_DIR/IRSTD-1K_BGM-PairedBasicUNet_gpu0.log'"

tmux new-window -t "$SESSION" -n irstd_tcf_pair "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 1 IRSTD-1K 5 tcf_bgm TCF-PairedBGM 6 0.00030 4113 0.58 2>&1 | tee '$LOG_DIR/IRSTD-1K_TCF-PairedBGM_gpu1.log'"

tmux new-window -t "$SESSION" -n nuaa_bgm_pair "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 2 NUAA-SIRST 5 bgm_basic BGM-PairedBasicUNet 8 0.00045 4115 0.52 2>&1 | tee '$LOG_DIR/NUAA-SIRST_BGM-PairedBasicUNet_gpu2.log'"

tmux new-window -t "$SESSION" -n nuaa_tcf_pair "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 3 NUAA-SIRST 5 tcf_bgm TCF-PairedBGM 6 0.00030 4117 0.54 2>&1 | tee '$LOG_DIR/NUAA-SIRST_TCF-PairedBGM_gpu3.log'"

tmux ls | grep "$SESSION"

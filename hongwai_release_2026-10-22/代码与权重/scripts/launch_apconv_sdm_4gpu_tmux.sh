#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/apconv_sdm_4gpu"
LOG_DIR="$RUN_ROOT/logs"
SESSION="apconv_sdm_4gpu"

mkdir -p "$LOG_DIR"
tmux kill-session -t "$SESSION" 2>/dev/null || true

common_args=(
  --dataset_dir "$ROOT/datasets"
  --train_datasets "IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3"
  --epochs 260
  --threads 4
  --optimizer adamw
  --weight_decay 0.00006
  --warmup_epochs 4
  --min_lr_ratio 0.10
  --begin_test 10
  --every_test 10
  --every_print 5
  --noise_std 0.0015
  --gamma_prob 0.030
  --intensity_jitter 0.006
  --bce_weight 0.30
  --dice_weight 0.12
  --soft_iou_weight 0.55
  --focal_weight 0.015
  --sls_iou_weight 0.030
  --sls_iou_warm_epochs 5
  --sdm_iou_weight 0.22
  --sdm_warm_epochs 5
  --sdm_delta 0.5
  --sdm_with_distance
  --bg_hf_weight 0.050
  --bg_pred_weight 0.010
  --target_dilate 7
)

run_cmd() {
  local gpu="$1"
  local dataset="$2"
  local target_repeat="$3"
  local method="$4"
  local batch="$5"
  local lr="$6"
  local seed="$7"
  local pos_prob="$8"
  local paired="$9"
  local run_dir="$RUN_ROOT/${dataset}_${method}_gpu${gpu}"

  local paired_args=()
  if [[ "$paired" == "yes" ]]; then
    paired_args=(--paired_bgm --bgm_prob 0.0 --bgm_region_size 16 --bgm_mask_ratio 0.18)
  else
    paired_args=(--bgm_prob 0.12 --bgm_region_size 16 --bgm_mask_ratio 0.12)
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT" "${common_args[@]}" "${paired_args[@]}" \
    --dataset_name "$dataset" \
    --target_repeat "$target_repeat" \
    --model apconv_tcf \
    --run_dir "$run_dir" \
    --method_name "$method" \
    --batch_size "$batch" \
    --lr "$lr" \
    --seed "$seed" \
    --pos_prob "$pos_prob"
}

tmux new-session -d -s "$SESSION" -n irstd_apconv_sdm "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 0 IRSTD-1K 5 APConv-TCF-SDM 7 0.00034 5111 0.58 no 2>&1 | tee '$LOG_DIR/IRSTD-1K_APConv-TCF-SDM_gpu0.log'"

tmux new-window -t "$SESSION" -n irstd_apconv_pair "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 1 IRSTD-1K 5 APConv-TCF-PairedSDM 5 0.00028 5113 0.60 yes 2>&1 | tee '$LOG_DIR/IRSTD-1K_APConv-TCF-PairedSDM_gpu1.log'"

tmux new-window -t "$SESSION" -n nuaa_apconv_sdm "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 2 NUAA-SIRST 5 APConv-TCF-SDM 7 0.00034 5115 0.54 no 2>&1 | tee '$LOG_DIR/NUAA-SIRST_APConv-TCF-SDM_gpu2.log'"

tmux new-window -t "$SESSION" -n nuaa_apconv_pair "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 3 NUAA-SIRST 5 APConv-TCF-PairedSDM 5 0.00028 5117 0.56 yes 2>&1 | tee '$LOG_DIR/NUAA-SIRST_APConv-TCF-PairedSDM_gpu3.log'"

tmux ls | grep "$SESSION"

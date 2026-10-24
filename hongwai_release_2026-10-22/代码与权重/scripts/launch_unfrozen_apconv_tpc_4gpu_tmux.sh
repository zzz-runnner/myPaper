#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/unfrozen_apconv_tpc_4gpu"
LOG_DIR="$RUN_ROOT/logs"
SESSION="unfrozen_apconv_tpc_4gpu"

mkdir -p "$LOG_DIR"
tmux kill-session -t "$SESSION" 2>/dev/null || true

common_args=(
  --dataset_dir "$ROOT/datasets"
  --train_datasets "IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3"
  --model sct_apconv_tpc_unfrozen_refiner
  --epochs 80
  --threads 4
  --optimizer adamw
  --weight_decay 0.000015
  --warmup_epochs 2
  --min_lr_ratio 0.25
  --begin_test 1
  --every_test 1
  --every_print 1
  --eval_before_train
  --noise_std 0.0006
  --gamma_prob 0.012
  --intensity_jitter 0.003
  --bgm_prob 0.08
  --bgm_region_size 16
  --bgm_mask_ratio 0.08
  --bce_weight 0.14
  --dice_weight 0.06
  --soft_iou_weight 0.24
  --focal_weight 0.008
  --sls_iou_weight 0.012
  --sls_iou_warm_epochs 1
  --sdm_iou_weight 0.070
  --sdm_warm_epochs 1
  --sdm_delta 0.5
  --sdm_with_distance
  --bg_hf_weight 0.035
  --bg_pred_weight 0.008
  --target_dilate 9
  --base_anchor_weight 0.32
  --base_anchor_bg_weight 1.00
  --base_anchor_pos_weight 0.60
  --base_anchor_logit_weight 0.020
  --base_anchor_margin 0.003
  --base_anchor_hf_boost 1.00
  --base_anchor_target_dilate 11
  --teacher_weight 0.20
  --teacher_bg_weight 1.00
  --teacher_pos_weight 0.45
  --teacher_miss_weight 0.22
  --teacher_margin 0.004
  --teacher_hf_boost 1.00
  --teacher_target_dilate 11
  --thermal_spot_weight 0.045
  --thermal_spot_sigma_scale 0.75
  --thermal_spot_sigma_min 1.1
  --thermal_spot_sigma_max 4.5
  --thermal_spot_guard 11
  --thermal_spot_core_boost 1.2
  --thermal_spot_margin 0.10
)

run_cmd() {
  local gpu="$1"
  local dataset="$2"
  local target_repeat="$3"
  local method="$4"
  local base_ckpt="$5"
  local resume_ckpt="$6"
  local batch="$7"
  local lr="$8"
  local base_scale="$9"
  local seed="${10}"
  local pos_prob="${11}"
  local run_dir="$RUN_ROOT/${dataset}_${method}_gpu${gpu}"

  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT" "${common_args[@]}" \
    --dataset_name "$dataset" \
    --target_repeat "$target_repeat" \
    --method_name "$method" \
    --base_checkpoint "$base_ckpt" \
    --resume_model_checkpoint "$resume_ckpt" \
    --teacher_checkpoint "$resume_ckpt" \
    --teacher_model sct_refiner \
    --run_dir "$run_dir" \
    --batch_size "$batch" \
    --lr "$lr" \
    --base_lr_scale "$base_scale" \
    --seed "$seed" \
    --pos_prob "$pos_prob"
}

IRSTD_BASE="$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar"
IRSTD_FIXED="$ROOT/runs/sct_refiner_fixed/IRSTD-1K_SCT-Refiner-Fixed_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar"
NUAA_BASE="$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar"
NUAA_FIXED="$ROOT/runs/sct_refiner_fixed/NUAA-SIRST_SCT-Refiner-Fixed_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar"

tmux new-session -d -s "$SESSION" -n irstd_unfrozen_a "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 0 IRSTD-1K 5 SCT-UnfrozenAPConvTPC-A '$IRSTD_BASE' '$IRSTD_FIXED' 6 0.0000040 0.12 7111 0.58 2>&1 | tee '$LOG_DIR/IRSTD-1K_SCT-UnfrozenAPConvTPC-A_gpu0.log'"

tmux new-window -t "$SESSION" -n irstd_unfrozen_b "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 1 IRSTD-1K 6 SCT-UnfrozenAPConvTPC-B '$IRSTD_BASE' '$IRSTD_FIXED' 7 0.0000030 0.08 7113 0.60 2>&1 | tee '$LOG_DIR/IRSTD-1K_SCT-UnfrozenAPConvTPC-B_gpu1.log'"

tmux new-window -t "$SESSION" -n nuaa_unfrozen_a "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 2 NUAA-SIRST 5 SCT-UnfrozenAPConvTPC-A '$NUAA_BASE' '$NUAA_FIXED' 6 0.0000030 0.10 7115 0.54 2>&1 | tee '$LOG_DIR/NUAA-SIRST_SCT-UnfrozenAPConvTPC-A_gpu2.log'"

tmux new-window -t "$SESSION" -n nuaa_unfrozen_b "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 3 NUAA-SIRST 6 SCT-UnfrozenAPConvTPC-B '$NUAA_BASE' '$NUAA_FIXED' 7 0.0000022 0.07 7117 0.56 2>&1 | tee '$LOG_DIR/NUAA-SIRST_SCT-UnfrozenAPConvTPC-B_gpu3.log'"

tmux ls | grep "$SESSION"

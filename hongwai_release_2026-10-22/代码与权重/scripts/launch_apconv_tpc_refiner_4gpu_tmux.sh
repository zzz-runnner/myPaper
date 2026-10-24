#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/apconv_tpc_refiner_4gpu"
LOG_DIR="$RUN_ROOT/logs"
SESSION="apconv_tpc_refiner_4gpu"

mkdir -p "$LOG_DIR"
tmux kill-session -t "$SESSION" 2>/dev/null || true

common_args=(
  --dataset_dir "$ROOT/datasets"
  --train_datasets "IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3"
  --model sct_apconv_tpc_refiner
  --epochs 120
  --threads 4
  --optimizer adamw
  --weight_decay 0.00003
  --warmup_epochs 2
  --min_lr_ratio 0.20
  --begin_test 2
  --every_test 2
  --every_print 2
  --eval_before_train
  --noise_std 0.0008
  --gamma_prob 0.018
  --intensity_jitter 0.004
  --bgm_prob 0.10
  --bgm_region_size 16
  --bgm_mask_ratio 0.10
  --bce_weight 0.18
  --dice_weight 0.08
  --soft_iou_weight 0.30
  --focal_weight 0.010
  --sls_iou_weight 0.015
  --sls_iou_warm_epochs 2
  --sdm_iou_weight 0.10
  --sdm_warm_epochs 2
  --sdm_delta 0.5
  --sdm_with_distance
  --bg_hf_weight 0.050
  --bg_pred_weight 0.012
  --target_dilate 9
  --base_anchor_weight 0.18
  --base_anchor_bg_weight 1.00
  --base_anchor_pos_weight 0.50
  --base_anchor_logit_weight 0.010
  --base_anchor_margin 0.004
  --base_anchor_hf_boost 1.00
  --base_anchor_target_dilate 11
  --teacher_weight 0.12
  --teacher_bg_weight 1.00
  --teacher_pos_weight 0.35
  --teacher_miss_weight 0.20
  --teacher_margin 0.006
  --teacher_hf_boost 1.00
  --teacher_target_dilate 11
  --thermal_spot_weight 0.08
  --thermal_spot_sigma_scale 0.75
  --thermal_spot_sigma_min 1.1
  --thermal_spot_sigma_max 4.5
  --thermal_spot_guard 11
  --thermal_spot_core_boost 1.4
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
  local seed="$9"
  local pos_prob="${10}"
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
    --seed "$seed" \
    --pos_prob "$pos_prob"
}

IRSTD_BASE="$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar"
IRSTD_FIXED="$ROOT/runs/sct_refiner_fixed/IRSTD-1K_SCT-Refiner-Fixed_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar"
NUAA_BASE="$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar"
NUAA_FIXED="$ROOT/runs/sct_refiner_fixed/NUAA-SIRST_SCT-Refiner-Fixed_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar"

tmux new-session -d -s "$SESSION" -n irstd_apconv_tpc_a "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 0 IRSTD-1K 5 SCT-APConvTPC-A '$IRSTD_BASE' '$IRSTD_FIXED' 8 0.000010 6111 0.58 2>&1 | tee '$LOG_DIR/IRSTD-1K_SCT-APConvTPC-A_gpu0.log'"

tmux new-window -t "$SESSION" -n irstd_apconv_tpc_b "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 1 IRSTD-1K 6 SCT-APConvTPC-B '$IRSTD_BASE' '$IRSTD_FIXED' 10 0.000007 6113 0.60 2>&1 | tee '$LOG_DIR/IRSTD-1K_SCT-APConvTPC-B_gpu1.log'"

tmux new-window -t "$SESSION" -n nuaa_apconv_tpc_a "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 2 NUAA-SIRST 5 SCT-APConvTPC-A '$NUAA_BASE' '$NUAA_FIXED' 8 0.000006 6115 0.54 2>&1 | tee '$LOG_DIR/NUAA-SIRST_SCT-APConvTPC-A_gpu2.log'"

tmux new-window -t "$SESSION" -n nuaa_apconv_tpc_b "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 3 NUAA-SIRST 6 SCT-APConvTPC-B '$NUAA_BASE' '$NUAA_FIXED' 10 0.000004 6117 0.56 2>&1 | tee '$LOG_DIR/NUAA-SIRST_SCT-APConvTPC-B_gpu3.log'"

tmux ls | grep "$SESSION"

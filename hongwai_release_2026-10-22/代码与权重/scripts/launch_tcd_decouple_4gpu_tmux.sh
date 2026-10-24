#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/tcd_decouple_4gpu"
LOG_DIR="$RUN_ROOT/logs"
SESSION="tcd_decouple_4gpu"

mkdir -p "$LOG_DIR"
tmux kill-session -t "$SESSION" 2>/dev/null || true

common_args=(
  --dataset_dir "$ROOT/datasets"
  --train_datasets "IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3"
  --model sct_tcd_decouple_refiner
  --epochs 36
  --batch_size 24
  --threads 4
  --weight_decay 0.00001
  --warmup_epochs 1
  --min_lr_ratio 0.25
  --begin_test 2
  --every_test 2
  --every_print 1
  --eval_before_train
  --bgm_prob 0.045
  --bgm_mask_ratio 0.030
  --noise_std 0.00025
  --gamma_prob 0.004
  --intensity_jitter 0.0012
  --bce_weight 0.72
  --dice_weight 0.25
  --focal_weight 0.018
  --sls_iou_weight 0.006
  --sls_iou_warm_epochs 1
  --bg_hf_weight 0.024
  --bg_pred_weight 0.0025
  --base_anchor_bg_weight 1.0
  --base_anchor_pos_weight 0.72
  --base_anchor_logit_weight 0.070
  --base_anchor_bg_threshold 0.08
  --base_anchor_pos_threshold 0.35
  --base_anchor_margin 0.0025
  --base_anchor_hf_boost 1.4
  --base_anchor_target_dilate 11
  --tcd_target_gate_weight 1.0
  --tcd_clutter_gate_weight 0.35
  --tcd_anti_target_weight 0.70
  --tcd_target_hf_boost 1.2
  --tcd_target_dilate 9
  --thermal_spot_sigma_scale 0.70
  --thermal_spot_sigma_min 1.0
  --thermal_spot_sigma_max 4.0
  --thermal_spot_guard 11
  --thermal_spot_core_boost 1.4
  --thermal_spot_margin 0.07
  --thermal_spot_contrast_weight 0.18
  --thermal_spot_scale_weight 0.04
)

run_cmd() {
  local gpu="$1"
  local dataset="$2"
  local target_repeat="$3"
  local method="$4"
  local base_ckpt="$5"
  local resume_ckpt="$6"
  local lr="$7"
  local seed="$8"
  local anchor_weight="$9"
  local tcd_weight="${10}"
  local thermal_weight="${11}"
  local pos_prob="${12}"
  local run_dir="$RUN_ROOT/${dataset}_${method}_gpu${gpu}"

  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT" "${common_args[@]}" \
    --dataset_name "$dataset" \
    --target_repeat "$target_repeat" \
    --run_dir "$run_dir" \
    --method_name "$method" \
    --base_checkpoint "$base_ckpt" \
    --resume_model_checkpoint "$resume_ckpt" \
    --lr "$lr" \
    --seed "$seed" \
    --base_anchor_weight "$anchor_weight" \
    --tcd_gate_weight "$tcd_weight" \
    --thermal_spot_weight "$thermal_weight" \
    --pos_prob "$pos_prob"
}

IRSTD_BASE="$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar"
IRSTD_FIXED="$ROOT/runs/sct_refiner_fixed/IRSTD-1K_SCT-Refiner-Fixed_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar"
NUAA_BASE="$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar"
NUAA_FIXED="$ROOT/runs/sct_refiner_fixed/NUAA-SIRST_SCT-Refiner-Fixed_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar"

tmux new-session -d -s "$SESSION" -n irstd_dec_a "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 0 IRSTD-1K 6 SCT-TCD-Decouple-A '$IRSTD_BASE' '$IRSTD_FIXED' 0.0000060 3111 2.10 0.055 0.004 0.60 2>&1 | tee '$LOG_DIR/IRSTD-1K_SCT-TCD-Decouple-A_gpu0.log'"

tmux new-window -t "$SESSION" -n irstd_dec_b "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 1 IRSTD-1K 7 SCT-TCD-Decouple-B '$IRSTD_BASE' '$IRSTD_FIXED' 0.0000045 3113 2.60 0.075 0.003 0.62 2>&1 | tee '$LOG_DIR/IRSTD-1K_SCT-TCD-Decouple-B_gpu1.log'"

tmux new-window -t "$SESSION" -n nuaa_dec_a "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 2 NUAA-SIRST 5 SCT-TCD-Decouple-A '$NUAA_BASE' '$NUAA_FIXED' 0.0000035 3115 3.00 0.030 0.002 0.56 2>&1 | tee '$LOG_DIR/NUAA-SIRST_SCT-TCD-Decouple-A_gpu2.log'"

tmux new-window -t "$SESSION" -n nuaa_dec_b "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 3 NUAA-SIRST 6 SCT-TCD-Decouple-B '$NUAA_BASE' '$NUAA_FIXED' 0.0000028 3117 3.30 0.025 0.0015 0.58 2>&1 | tee '$LOG_DIR/NUAA-SIRST_SCT-TCD-Decouple-B_gpu3.log'"

tmux ls | grep "$SESSION"

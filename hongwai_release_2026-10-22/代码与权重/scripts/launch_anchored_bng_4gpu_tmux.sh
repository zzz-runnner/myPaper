#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/bng_featup_anchor_4gpu"
LOG_ROOT="$RUN_ROOT/logs"
DATA="$ROOT/datasets"
TRAIN_DATASETS="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3"
SESSION="bng_anchor_4gpu"

mkdir -p "$LOG_ROOT"
tmux kill-session -t "$SESSION" 2>/dev/null || true

common_args=(
  --dataset_dir "$DATA"
  --train_datasets "$TRAIN_DATASETS"
  --model sct_bng_featup_refiner
  --epochs 110
  --batch_size 28
  --threads 4
  --weight_decay 0.00005
  --begin_test 5
  --every_test 5
  --every_print 10
  --eval_before_train
  --bgm_prob 0.05
  --bgm_mask_ratio 0.035
  --sls_iou_weight 0.012
  --sls_iou_warm_epochs 1
  --bg_hf_weight 0.040
  --bg_pred_weight 0.006
  --teacher_weight 0.0
  --base_anchor_bg_weight 1.0
  --base_anchor_pos_weight 0.60
  --base_anchor_logit_weight 0.030
  --base_anchor_bg_threshold 0.08
  --base_anchor_pos_threshold 0.35
  --base_anchor_margin 0.004
  --base_anchor_hf_boost 1.3
  --base_anchor_target_dilate 11
  --thermal_spot_sigma_scale 0.70
  --thermal_spot_sigma_min 1.0
  --thermal_spot_sigma_max 4.2
  --thermal_spot_guard 11
  --thermal_spot_core_boost 1.5
  --thermal_spot_margin 0.08
  --thermal_spot_contrast_weight 0.22
  --thermal_spot_scale_weight 0.06
  --noise_std 0.0004
  --gamma_prob 0.010
  --intensity_jitter 0.002
)

run_cmd() {
  local gpu="$1"
  local dataset="$2"
  local target_repeat="$3"
  local method="$4"
  local run_dir="$5"
  local base_ckpt="$6"
  local resume_ckpt="$7"
  local lr="$8"
  local seed="$9"
  local anchor_weight="${10}"
  local thermal_weight="${11}"
  local pos_prob="${12}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT" \
    "${common_args[@]}" \
    --dataset_name "$dataset" \
    --target_repeat "$target_repeat" \
    --run_dir "$run_dir" \
    --method_name "$method" \
    --base_checkpoint "$base_ckpt" \
    --resume_model_checkpoint "$resume_ckpt" \
    --lr "$lr" \
    --seed "$seed" \
    --base_anchor_weight "$anchor_weight" \
    --thermal_spot_weight "$thermal_weight" \
    --pos_prob "$pos_prob"
}

IRSTD_BASE="$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar"
IRSTD_FIXED="$ROOT/runs/sct_refiner_fixed/IRSTD-1K_SCT-Refiner-Fixed_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar"
NUAA_BASE="$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar"
NUAA_FIXED="$ROOT/runs/sct_refiner_fixed/NUAA-SIRST_SCT-Refiner-Fixed_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar"

tmux new-session -d -s "$SESSION" -n irstd_anchor_a \
  "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && ROOT='$ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; $(declare -f run_cmd); common_args=($(printf '%q ' "${common_args[@]}")); run_cmd 0 IRSTD-1K 5 SCT-BNG-Anchor-A '$RUN_ROOT/IRSTD-1K_SCT-BNG-Anchor-A_gpu0' '$IRSTD_BASE' '$IRSTD_FIXED' 0.000006 911 1.20 0.018 0.58 2>&1 | tee '$LOG_ROOT/IRSTD-1K_SCT-BNG-Anchor-A_gpu0.log'"

tmux new-window -t "$SESSION" -n irstd_anchor_b \
  "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && ROOT='$ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; $(declare -f run_cmd); common_args=($(printf '%q ' "${common_args[@]}")); run_cmd 1 IRSTD-1K 6 SCT-BNG-Anchor-B '$RUN_ROOT/IRSTD-1K_SCT-BNG-Anchor-B_gpu1' '$IRSTD_BASE' '$IRSTD_FIXED' 0.000004 913 1.80 0.012 0.60 2>&1 | tee '$LOG_ROOT/IRSTD-1K_SCT-BNG-Anchor-B_gpu1.log'"

tmux new-window -t "$SESSION" -n nuaa_anchor_a \
  "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && ROOT='$ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; $(declare -f run_cmd); common_args=($(printf '%q ' "${common_args[@]}")); run_cmd 2 NUAA-SIRST 5 SCT-BNG-Anchor-A '$RUN_ROOT/NUAA-SIRST_SCT-BNG-Anchor-A_gpu2' '$NUAA_BASE' '$NUAA_FIXED' 0.000005 915 1.40 0.014 0.54 2>&1 | tee '$LOG_ROOT/NUAA-SIRST_SCT-BNG-Anchor-A_gpu2.log'"

tmux new-window -t "$SESSION" -n nuaa_anchor_b \
  "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && ROOT='$ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; $(declare -f run_cmd); common_args=($(printf '%q ' "${common_args[@]}")); run_cmd 3 NUAA-SIRST 6 SCT-BNG-Anchor-B '$RUN_ROOT/NUAA-SIRST_SCT-BNG-Anchor-B_gpu3' '$NUAA_BASE' '$NUAA_FIXED' 0.0000035 917 2.00 0.010 0.56 2>&1 | tee '$LOG_ROOT/NUAA-SIRST_SCT-BNG-Anchor-B_gpu3.log'"

tmux list-windows -t "$SESSION"

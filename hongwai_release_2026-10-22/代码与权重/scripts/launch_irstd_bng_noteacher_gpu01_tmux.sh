#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/bng_featup_irstd_gpu01_noteacher"
LOG_ROOT="$RUN_ROOT/logs"
DATA="$ROOT/datasets"
TRAIN_DATASETS="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3"
SESSION="irstd_bng_gpu01"

mkdir -p "$LOG_ROOT"
tmux kill-session -t "$SESSION" 2>/dev/null || true

common_args=(
  --dataset_dir "$DATA"
  --dataset_name IRSTD-1K
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
  --bgm_prob 0.07
  --bgm_mask_ratio 0.045
  --sls_iou_weight 0.020
  --sls_iou_warm_epochs 1
  --bg_hf_weight 0.060
  --bg_pred_weight 0.010
  --teacher_weight 0.0
  --thermal_spot_sigma_scale 0.70
  --thermal_spot_sigma_min 1.0
  --thermal_spot_sigma_max 4.2
  --thermal_spot_guard 11
  --thermal_spot_core_boost 1.7
  --thermal_spot_margin 0.10
  --thermal_spot_contrast_weight 0.28
  --thermal_spot_scale_weight 0.10
  --noise_std 0.0005
  --gamma_prob 0.014
  --intensity_jitter 0.003
  --base_checkpoint "$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar"
  --resume_model_checkpoint "$ROOT/runs/sct_refiner_fixed/IRSTD-1K_SCT-Refiner-Fixed_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar"
)

tmux new-session -d -s "$SESSION" -n irstd_bng_a \
  "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && CUDA_VISIBLE_DEVICES=0 '$PY' '$SCRIPT' $(printf '%q ' "${common_args[@]}") --target_repeat 5 --run_dir '$RUN_ROOT/IRSTD-1K_SCT-BNG-FeatUp-NT28-A_gpu0' --method_name SCT-BNG-FeatUp-NT28-A --lr 0.000012 --seed 901 --thermal_spot_weight 0.030 --pos_prob 0.60 2>&1 | tee '$LOG_ROOT/IRSTD-1K_SCT-BNG-FeatUp-NT28-A_gpu0.log'"

tmux new-window -t "$SESSION" -n irstd_bng_b \
  "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && CUDA_VISIBLE_DEVICES=1 '$PY' '$SCRIPT' $(printf '%q ' "${common_args[@]}") --target_repeat 6 --run_dir '$RUN_ROOT/IRSTD-1K_SCT-BNG-FeatUp-NT28-B_gpu1' --method_name SCT-BNG-FeatUp-NT28-B --lr 0.000009 --seed 903 --thermal_spot_weight 0.024 --pos_prob 0.62 2>&1 | tee '$LOG_ROOT/IRSTD-1K_SCT-BNG-FeatUp-NT28-B_gpu1.log'"

tmux list-windows -t "$SESSION"

#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/tpc_thermal_irstd_gpu0"
LOG_ROOT="$RUN_ROOT/logs"
DATA="$ROOT/datasets"
TRAIN_DATASETS="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3"

mkdir -p "$LOG_ROOT"
tmux kill-session -t irstd_tpc_thermal_gpu0 2>/dev/null || true

tmux new-session -d -s irstd_tpc_thermal_gpu0 -n irstd_tpc_thermal \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES=0 '$PY' '$SCRIPT' \
    --dataset_dir '$DATA' \
    --dataset_name 'IRSTD-1K' \
    --train_datasets '$TRAIN_DATASETS' \
    --target_repeat 5 \
    --run_dir '$RUN_ROOT/IRSTD-1K_SCT-TPC-Thermal_gpu0' \
    --method_name 'SCT-TPC-Thermal' \
    --model 'sct_tpc_dhif_refiner' \
    --base_checkpoint '$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar' \
    --resume_model_checkpoint '$ROOT/runs/sct_refiner_fixed/IRSTD-1K_SCT-Refiner-Fixed_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar' \
    --teacher_checkpoint '$ROOT/runs/sct_refiner_fixed/IRSTD-1K_SCT-Refiner-Fixed_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar' \
    --teacher_model 'sct_refiner' \
    --epochs 90 \
    --batch_size 48 \
    --threads 4 \
    --lr 0.000020 \
    --weight_decay 0.00005 \
    --seed 871 \
    --begin_test 5 \
    --every_test 5 \
    --every_print 10 \
    --eval_before_train \
    --bgm_prob 0.08 \
    --bgm_mask_ratio 0.05 \
    --sls_iou_weight 0.025 \
    --sls_iou_warm_epochs 1 \
    --bg_hf_weight 0.075 \
    --bg_pred_weight 0.012 \
    --teacher_weight 0.20 \
    --teacher_bg_weight 1.0 \
    --teacher_pos_weight 0.40 \
    --teacher_miss_weight 0.42 \
    --teacher_bg_threshold 0.08 \
    --teacher_pos_threshold 0.35 \
    --teacher_margin 0.010 \
    --teacher_hf_boost 1.2 \
    --teacher_target_dilate 11 \
    --thermal_spot_weight 0.050 \
    --thermal_spot_sigma_scale 0.70 \
    --thermal_spot_sigma_min 1.0 \
    --thermal_spot_sigma_max 4.2 \
    --thermal_spot_guard 11 \
    --thermal_spot_core_boost 1.8 \
    --thermal_spot_margin 0.10 \
    --thermal_spot_contrast_weight 0.30 \
    --thermal_spot_scale_weight 0.10 \
    --pos_prob 0.62 \
    --noise_std 0.0007 \
    --gamma_prob 0.018 \
    --intensity_jitter 0.004 \
    2>&1 | tee '$LOG_ROOT/IRSTD-1K_SCT-TPC-Thermal_gpu0.log'"

tmux list-windows -t irstd_tpc_thermal_gpu0

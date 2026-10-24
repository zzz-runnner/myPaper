#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/tpc_thermal_nuaa_gpu3"
LOG_ROOT="$RUN_ROOT/logs"
DATA="$ROOT/datasets"
TRAIN_DATASETS="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3"

mkdir -p "$LOG_ROOT"
tmux kill-session -t nuaa_tpc_thermal_gpu3 2>/dev/null || true

tmux new-session -d -s nuaa_tpc_thermal_gpu3 -n nuaa_tpc_thermal \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES=3 '$PY' '$SCRIPT' \
    --dataset_dir '$DATA' \
    --dataset_name 'NUAA-SIRST' \
    --train_datasets '$TRAIN_DATASETS' \
    --target_repeat 6 \
    --run_dir '$RUN_ROOT/NUAA-SIRST_SCT-TPC-Thermal_gpu3' \
    --method_name 'SCT-TPC-Thermal' \
    --model 'sct_tpc_dhif_refiner' \
    --base_checkpoint '$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar' \
    --resume_model_checkpoint '$ROOT/runs/sct_refiner_fixed/NUAA-SIRST_SCT-Refiner-Fixed_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar' \
    --teacher_checkpoint '$ROOT/runs/sct_refiner_fixed/NUAA-SIRST_SCT-Refiner-Fixed_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar' \
    --teacher_model 'sct_refiner' \
    --epochs 90 \
    --batch_size 48 \
    --threads 4 \
    --lr 0.000017 \
    --weight_decay 0.00005 \
    --seed 867 \
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
    --teacher_weight 0.18 \
    --teacher_bg_weight 1.0 \
    --teacher_pos_weight 0.38 \
    --teacher_miss_weight 0.40 \
    --teacher_bg_threshold 0.08 \
    --teacher_pos_threshold 0.35 \
    --teacher_margin 0.010 \
    --teacher_hf_boost 1.2 \
    --teacher_target_dilate 11 \
    --thermal_spot_weight 0.045 \
    --thermal_spot_sigma_scale 0.70 \
    --thermal_spot_sigma_min 1.0 \
    --thermal_spot_sigma_max 4.0 \
    --thermal_spot_guard 11 \
    --thermal_spot_core_boost 1.8 \
    --thermal_spot_margin 0.10 \
    --thermal_spot_contrast_weight 0.30 \
    --thermal_spot_scale_weight 0.10 \
    --pos_prob 0.58 \
    --noise_std 0.0006 \
    --gamma_prob 0.015 \
    --intensity_jitter 0.003 \
    2>&1 | tee '$LOG_ROOT/NUAA-SIRST_SCT-TPC-Thermal_gpu3.log'"

tmux list-windows -t nuaa_tpc_thermal_gpu3

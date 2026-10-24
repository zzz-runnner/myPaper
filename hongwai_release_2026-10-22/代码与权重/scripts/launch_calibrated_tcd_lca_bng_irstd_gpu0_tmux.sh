#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/calibrated_tcd_lca_bng_gpu0"
LOG_ROOT="$RUN_ROOT/logs"
SESSION="cal_tcd_lca_bng_irstd_gpu0"

mkdir -p "$LOG_ROOT"
tmux kill-session -t "$SESSION" 2>/dev/null || true
rm -rf "$RUN_ROOT/IRSTD-1K_CalTCD-LCA-BNG_gpu0"

DATASETS="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3"
IRSTD_BASE="$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar"
IRSTD_TPC="$ROOT/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-B_gpu1/checkpoints/IRSTD-1K/best_mIoU.pth.tar"

tmux new-session -d -s "$SESSION" -n irstd_cal_tcd_lca_bng \
  "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && CUDA_VISIBLE_DEVICES=0 '$PY' '$SCRIPT' \
    --dataset_dir '$ROOT/datasets' \
    --dataset_name 'IRSTD-1K' \
    --train_datasets '$DATASETS' \
    --target_repeat 6 \
    --run_dir '$RUN_ROOT/IRSTD-1K_CalTCD-LCA-BNG_gpu0' \
    --method_name 'CalTCD-LCA-BNG' \
    --model 'sct_tcd_lca_bng_refiner' \
    --base_checkpoint '$IRSTD_BASE' \
    --resume_model_checkpoint '$IRSTD_TPC' \
    --teacher_model 'sct_tpc_dhif_refiner' \
    --teacher_base_checkpoint '$IRSTD_BASE' \
    --teacher_checkpoint '$IRSTD_TPC' \
    --epochs 60 \
    --batch_size 10 \
    --threads 4 \
    --optimizer adamw \
    --lr 0.000035 \
    --weight_decay 0.00002 \
    --warmup_epochs 1 \
    --min_lr_ratio 0.20 \
    --seed 8011 \
    --begin_test 1 \
    --every_test 1 \
    --every_print 1 \
    --eval_before_train \
    --pos_prob 0.68 \
    --noise_std 0.0004 \
    --gamma_prob 0.004 \
    --intensity_jitter 0.0015 \
    --bgm_prob 0.02 \
    --bgm_region_size 16 \
    --bgm_mask_ratio 0.06 \
    --bce_weight 0.08 \
    --dice_weight 0.035 \
    --soft_iou_weight 0.22 \
    --focal_weight 0.004 \
    --sls_iou_weight 0.010 \
    --sls_iou_warm_epochs 2 \
    --sls_location_weight 0.12 \
    --sdm_iou_weight 0.055 \
    --sdm_warm_epochs 2 \
    --sdm_delta 0.5 \
    --sdm_with_distance \
    --bg_hf_weight 0.018 \
    --bg_pred_weight 0.004 \
    --target_dilate 7 \
    --teacher_weight 0.55 \
    --teacher_bg_weight 1.0 \
    --teacher_pos_weight 0.55 \
    --teacher_miss_weight 0.18 \
    --teacher_logit_weight 0.18 \
    --teacher_bg_threshold 0.08 \
    --teacher_pos_threshold 0.35 \
    --teacher_margin 0.004 \
    --teacher_hf_boost 1.0 \
    --teacher_target_dilate 11 \
    --base_anchor_weight 0.45 \
    --base_anchor_bg_weight 1.0 \
    --base_anchor_pos_weight 0.55 \
    --base_anchor_logit_weight 0.080 \
    --base_anchor_bg_threshold 0.08 \
    --base_anchor_pos_threshold 0.35 \
    --base_anchor_margin 0.003 \
    --base_anchor_hf_boost 1.0 \
    --base_anchor_target_dilate 11 \
    --tcd_gate_weight 0.010 \
    --tcd_target_gate_weight 1.0 \
    --tcd_clutter_gate_weight 0.25 \
    --tcd_anti_target_weight 0.50 \
    --tcd_target_hf_boost 0.8 \
    --tcd_target_dilate 9 \
    --thermal_spot_weight 0.006 \
    --thermal_spot_sigma_scale 0.70 \
    --thermal_spot_sigma_min 1.0 \
    --thermal_spot_sigma_max 3.8 \
    --thermal_spot_guard 11 \
    --thermal_spot_core_boost 1.2 \
    --thermal_spot_margin 0.06 \
    --thermal_spot_contrast_weight 0.12 \
    --thermal_spot_scale_weight 0.03 \
    2>&1 | tee '$LOG_ROOT/IRSTD-1K_CalTCD-LCA-BNG_gpu0.log'"

tmux ls | grep "$SESSION"

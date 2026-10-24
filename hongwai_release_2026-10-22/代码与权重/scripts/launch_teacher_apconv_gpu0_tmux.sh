#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/teacher_apconv_gpu0"
LOG_ROOT="$RUN_ROOT/logs"
SESSION="teacher_apconv_gpu0"

mkdir -p "$LOG_ROOT"
tmux kill-session -t "$SESSION" 2>/dev/null || true
rm -rf "$RUN_ROOT/IRSTD-1K_APConvTCF-TPCDHiFTeacher_gpu0"

DATASETS="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3"
IRSTD_BASE="$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar"
IRSTD_TEACHER="$ROOT/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-B_gpu1/checkpoints/IRSTD-1K/best_mIoU.pth.tar"

tmux new-session -d -s "$SESSION" -n irstd_apconv_teacher \
  "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && CUDA_VISIBLE_DEVICES=0 '$PY' '$SCRIPT' \
    --dataset_dir '$ROOT/datasets' \
    --dataset_name 'IRSTD-1K' \
    --train_datasets '$DATASETS' \
    --target_repeat 6 \
    --run_dir '$RUN_ROOT/IRSTD-1K_APConvTCF-TPCDHiFTeacher_gpu0' \
    --method_name 'APConvTCF-TPCDHiFTeacher' \
    --model 'apconv_tcf' \
    --teacher_model 'sct_tpc_dhif_refiner' \
    --teacher_base_checkpoint '$IRSTD_BASE' \
    --teacher_checkpoint '$IRSTD_TEACHER' \
    --epochs 120 \
    --batch_size 10 \
    --threads 4 \
    --optimizer adamw \
    --lr 0.00024 \
    --weight_decay 0.00005 \
    --warmup_epochs 4 \
    --min_lr_ratio 0.08 \
    --seed 7311 \
    --begin_test 5 \
    --every_test 5 \
    --every_print 1 \
    --pos_prob 0.62 \
    --noise_std 0.0010 \
    --gamma_prob 0.020 \
    --intensity_jitter 0.004 \
    --bgm_prob 0.10 \
    --bgm_region_size 16 \
    --bgm_mask_ratio 0.10 \
    --bce_weight 0.28 \
    --dice_weight 0.14 \
    --soft_iou_weight 0.55 \
    --focal_weight 0.015 \
    --sls_iou_weight 0.040 \
    --sls_iou_warm_epochs 4 \
    --sls_location_weight 0.22 \
    --sdm_iou_weight 0.20 \
    --sdm_warm_epochs 4 \
    --sdm_delta 0.5 \
    --sdm_with_distance \
    --bg_hf_weight 0.060 \
    --bg_pred_weight 0.012 \
    --target_dilate 7 \
    --teacher_weight 0.30 \
    --teacher_bg_weight 1.0 \
    --teacher_pos_weight 0.45 \
    --teacher_miss_weight 0.30 \
    --teacher_bg_threshold 0.08 \
    --teacher_pos_threshold 0.35 \
    --teacher_margin 0.010 \
    --teacher_hf_boost 1.2 \
    --teacher_target_dilate 11 \
    2>&1 | tee '$LOG_ROOT/IRSTD-1K_APConvTCF-TPCDHiFTeacher_gpu0.log'"

tmux ls | grep "$SESSION"

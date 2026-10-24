#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/teacher_nuaa_gpu1"
LOG_ROOT="$RUN_ROOT/logs"
SESSION="teacher_nuaa_gpu1"

mkdir -p "$LOG_ROOT"
tmux kill-session -t "$SESSION" 2>/dev/null || true
rm -rf "$RUN_ROOT/NUAA-SIRST_APConvTCF-LCATeacher_gpu1"

DATASETS="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3"
NUAA_BASE="$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar"
NUAA_TEACHER="$ROOT/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar"

tmux new-session -d -s "$SESSION" -n nuaa_apconv_lca_teacher \
  "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && CUDA_VISIBLE_DEVICES=1 '$PY' '$SCRIPT' \
    --dataset_dir '$ROOT/datasets' \
    --dataset_name 'NUAA-SIRST' \
    --train_datasets '$DATASETS' \
    --target_repeat 8 \
    --run_dir '$RUN_ROOT/NUAA-SIRST_APConvTCF-LCATeacher_gpu1' \
    --method_name 'APConvTCF-LCATeacher' \
    --model 'apconv_tcf' \
    --teacher_model 'sct_lca_refiner' \
    --teacher_base_checkpoint '$NUAA_BASE' \
    --teacher_checkpoint '$NUAA_TEACHER' \
    --epochs 120 \
    --batch_size 10 \
    --threads 4 \
    --optimizer adamw \
    --lr 0.00024 \
    --weight_decay 0.00005 \
    --warmup_epochs 4 \
    --min_lr_ratio 0.08 \
    --seed 7411 \
    --begin_test 5 \
    --every_test 5 \
    --every_print 1 \
    --pos_prob 0.60 \
    --noise_std 0.0010 \
    --gamma_prob 0.018 \
    --intensity_jitter 0.004 \
    --bgm_prob 0.10 \
    --bgm_region_size 16 \
    --bgm_mask_ratio 0.10 \
    --bce_weight 0.30 \
    --dice_weight 0.14 \
    --soft_iou_weight 0.55 \
    --focal_weight 0.015 \
    --sls_iou_weight 0.035 \
    --sls_iou_warm_epochs 4 \
    --sls_location_weight 0.22 \
    --sdm_iou_weight 0.18 \
    --sdm_warm_epochs 4 \
    --sdm_delta 0.5 \
    --sdm_with_distance \
    --bg_hf_weight 0.055 \
    --bg_pred_weight 0.010 \
    --target_dilate 7 \
    --teacher_weight 0.32 \
    --teacher_bg_weight 1.0 \
    --teacher_pos_weight 0.48 \
    --teacher_miss_weight 0.32 \
    --teacher_bg_threshold 0.07 \
    --teacher_pos_threshold 0.34 \
    --teacher_margin 0.010 \
    --teacher_hf_boost 1.25 \
    --teacher_target_dilate 11 \
    2>&1 | tee '$LOG_ROOT/NUAA-SIRST_APConvTCF-LCATeacher_gpu1.log'"

tmux ls | grep "$SESSION"

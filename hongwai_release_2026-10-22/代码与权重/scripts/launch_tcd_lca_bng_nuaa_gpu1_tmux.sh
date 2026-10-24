#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/tcd_lca_bng_gpu1"
LOG_ROOT="$RUN_ROOT/logs"
SESSION="tcd_lca_bng_nuaa_gpu1"
PSEUDO_ROOT="$ROOT/generated/pseudo_nuaa_lca_tcd_train"

mkdir -p "$LOG_ROOT"
tmux kill-session -t "$SESSION" 2>/dev/null || true
rm -rf "$RUN_ROOT/NUAA-SIRST_TCD-LCA-BNG-LCATCDPseudo_gpu1"

DATASETS="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3"
NUAA_BASE="$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar"
NUAA_LCA="$ROOT/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar"
NUAA_TCD="$ROOT/runs/selected_checkpoints/tcd_decouple/NUAA-SIRST_SCT-TCD-Decouple-A_epoch2_best.pth.tar"

tmux new-session -d -s "$SESSION" -n nuaa_tcd_lca_bng \
  "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && CUDA_VISIBLE_DEVICES=1 '$PY' '$SCRIPT' \
    --dataset_dir '$ROOT/datasets' \
    --dataset_name 'NUAA-SIRST' \
    --train_datasets '$DATASETS' \
    --target_repeat 8 \
    --pseudo_dir '$PSEUDO_ROOT' \
    --run_dir '$RUN_ROOT/NUAA-SIRST_TCD-LCA-BNG-LCATCDPseudo_gpu1' \
    --method_name 'TCD-LCA-BNG-LCATCDPseudo' \
    --model 'sct_tcd_lca_bng_refiner' \
    --base_checkpoint '$NUAA_BASE' \
    --resume_model_checkpoint '$NUAA_TCD' \
    --teacher_model 'sct_lca_refiner' \
    --teacher_base_checkpoint '$NUAA_BASE' \
    --teacher_checkpoint '$NUAA_LCA' \
    --epochs 80 \
    --batch_size 8 \
    --threads 4 \
    --optimizer adamw \
    --lr 0.00016 \
    --weight_decay 0.00003 \
    --warmup_epochs 3 \
    --min_lr_ratio 0.10 \
    --seed 7811 \
    --begin_test 1 \
    --every_test 2 \
    --every_print 1 \
    --eval_before_train \
    --pos_prob 0.62 \
    --noise_std 0.0008 \
    --gamma_prob 0.012 \
    --intensity_jitter 0.003 \
    --bgm_prob 0.06 \
    --bgm_region_size 16 \
    --bgm_mask_ratio 0.08 \
    --bce_weight 0.22 \
    --dice_weight 0.10 \
    --soft_iou_weight 0.52 \
    --focal_weight 0.010 \
    --sls_iou_weight 0.030 \
    --sls_iou_warm_epochs 3 \
    --sls_location_weight 0.20 \
    --sdm_iou_weight 0.18 \
    --sdm_warm_epochs 3 \
    --sdm_delta 0.5 \
    --sdm_with_distance \
    --bg_hf_weight 0.045 \
    --bg_pred_weight 0.009 \
    --target_dilate 7 \
    --teacher_weight 0.22 \
    --teacher_bg_weight 1.0 \
    --teacher_pos_weight 0.42 \
    --teacher_miss_weight 0.25 \
    --teacher_bg_threshold 0.07 \
    --teacher_pos_threshold 0.34 \
    --teacher_margin 0.008 \
    --teacher_hf_boost 1.1 \
    --teacher_target_dilate 11 \
    --pseudo_weight 0.18 \
    --pseudo_bce_weight 1.0 \
    --pseudo_logit_weight 0.035 \
    --pseudo_iou_weight 0.20 \
    --pseudo_pos_weight 1.20 \
    --pseudo_bg_weight 0.70 \
    --pseudo_pos_threshold 0.34 \
    --pseudo_bg_threshold 0.06 \
    --pseudo_hf_boost 0.8 \
    --pseudo_target_dilate 11 \
    --base_anchor_weight 0.16 \
    --base_anchor_bg_weight 1.0 \
    --base_anchor_pos_weight 0.45 \
    --base_anchor_logit_weight 0.018 \
    --base_anchor_bg_threshold 0.07 \
    --base_anchor_pos_threshold 0.34 \
    --base_anchor_margin 0.006 \
    --base_anchor_hf_boost 1.0 \
    --base_anchor_target_dilate 11 \
    --tcd_gate_weight 0.035 \
    --tcd_target_gate_weight 1.0 \
    --tcd_clutter_gate_weight 0.35 \
    --tcd_anti_target_weight 0.65 \
    --tcd_target_hf_boost 1.0 \
    --tcd_target_dilate 9 \
    --thermal_spot_weight 0.020 \
    --thermal_spot_sigma_scale 0.75 \
    --thermal_spot_sigma_min 1.1 \
    --thermal_spot_sigma_max 4.0 \
    --thermal_spot_guard 11 \
    --thermal_spot_core_boost 1.4 \
    --thermal_spot_margin 0.08 \
    --thermal_spot_contrast_weight 0.20 \
    --thermal_spot_scale_weight 0.06 \
    2>&1 | tee '$LOG_ROOT/NUAA-SIRST_TCD-LCA-BNG-LCATCDPseudo_gpu1.log'"

tmux ls | grep "$SESSION"

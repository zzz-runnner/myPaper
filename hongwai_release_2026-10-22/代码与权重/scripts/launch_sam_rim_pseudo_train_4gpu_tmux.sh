#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/sam_rim_pseudo_4gpu"
LOG_ROOT="$RUN_ROOT/logs"
mkdir -p "$LOG_ROOT"

launch() {
  local session="$1"
  local gpu="$2"
  local dataset="$3"
  local method="$4"
  local model="$5"
  local seed="$6"
  local lr="$7"
  local batch="$8"
  local target_repeat="$9"
  local base_ckpt="${10}"
  local resume_ckpt="${11}"
  local teacher_model="${12}"
  local teacher_base="${13}"
  local teacher_ckpt="${14}"
  local pseudo_dir="${15}"
  local extra="${16}"
  local run_dir="$RUN_ROOT/${dataset}_${method}_gpu${gpu}"
  local log="$LOG_ROOT/${dataset}_${method}_gpu${gpu}.log"

  tmux new-session -d -s "$session" \
    "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && CUDA_VISIBLE_DEVICES=$gpu '$PY' '$SCRIPT' \
      --dataset_dir '$ROOT/datasets' \
      --dataset_name '$dataset' \
      --train_datasets 'IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3' \
      --target_repeat '$target_repeat' \
      --run_dir '$run_dir' \
      --method_name '$method' \
      --model '$model' \
      --base_checkpoint '$base_ckpt' \
      --resume_model_checkpoint '$resume_ckpt' \
      --teacher_model '$teacher_model' \
      --teacher_base_checkpoint '$teacher_base' \
      --teacher_checkpoint '$teacher_ckpt' \
      --pseudo_dir '$pseudo_dir' \
      --epochs 70 \
      --batch_size '$batch' \
      --threads 6 \
      --optimizer adamw \
      --lr '$lr' \
      --weight_decay 0.000012 \
      --warmup_epochs 1 \
      --min_lr_ratio 0.25 \
      --seed '$seed' \
      --begin_test 1 \
      --every_test 1 \
      --every_print 1 \
      --eval_before_train \
      --pos_prob 0.74 \
      --noise_std 0.00025 \
      --gamma_prob 0.0020 \
      --intensity_jitter 0.0010 \
      --bgm_prob 0.012 \
      --bgm_region_size 16 \
      --bgm_mask_ratio 0.045 \
      --bce_weight 0.052 \
      --dice_weight 0.028 \
      --soft_iou_weight 0.17 \
      --focal_weight 0.0025 \
      --sls_iou_weight 0.008 \
      --sls_iou_warm_epochs 2 \
      --sls_location_weight 0.10 \
      --sdm_iou_weight 0.040 \
      --sdm_warm_epochs 2 \
      --sdm_delta 0.5 \
      --sdm_with_distance \
      --bg_hf_weight 0.008 \
      --bg_pred_weight 0.0018 \
      --target_dilate 7 \
      --teacher_weight 0.66 \
      --teacher_bg_weight 1.0 \
      --teacher_pos_weight 0.58 \
      --teacher_miss_weight 0.16 \
      --teacher_logit_weight 0.26 \
      --teacher_bg_threshold 0.08 \
      --teacher_pos_threshold 0.35 \
      --teacher_margin 0.0025 \
      --teacher_hf_boost 1.0 \
      --teacher_target_dilate 11 \
      --base_anchor_weight 0.64 \
      --base_anchor_bg_weight 1.0 \
      --base_anchor_pos_weight 0.58 \
      --base_anchor_logit_weight 0.12 \
      --base_anchor_bg_threshold 0.08 \
      --base_anchor_pos_threshold 0.35 \
      --base_anchor_margin 0.0020 \
      --base_anchor_hf_boost 1.0 \
      --base_anchor_target_dilate 11 \
      --thermal_spot_weight 0.0025 \
      --thermal_spot_sigma_scale 0.66 \
      --thermal_spot_sigma_min 0.9 \
      --thermal_spot_sigma_max 3.5 \
      --thermal_spot_guard 11 \
      --thermal_spot_core_boost 1.12 \
      --thermal_spot_margin 0.045 \
      --thermal_spot_contrast_weight 0.08 \
      --thermal_spot_scale_weight 0.020 \
      --pseudo_weight 0.045 \
      --pseudo_bce_weight 1.0 \
      --pseudo_logit_weight 0.010 \
      --pseudo_iou_weight 0.055 \
      --pseudo_pos_weight 0.60 \
      --pseudo_bg_weight 0.35 \
      --pseudo_pos_threshold 0.50 \
      --pseudo_bg_threshold 0.06 \
      --pseudo_hf_boost 0.35 \
      --pseudo_target_dilate 11 \
      --pseudo_rim_weight 0.20 \
      --pseudo_rim_low 0.12 \
      --pseudo_rim_high 0.45 \
      --pseudo_rim_target 0.32 \
      $extra \
      2>&1 | tee '$log'"
}

IRSTD_BASE="$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar"
IRSTD_TPC_A="$ROOT/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-A_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar"
IRSTD_TPC_B="$ROOT/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-B_gpu1/checkpoints/IRSTD-1K/best_mIoU.pth.tar"
NUAA_BASE="$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar"
NUAA_LCA="$ROOT/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar"
NUAA_TCD="$ROOT/runs/selected_checkpoints/tcd_decouple/NUAA-SIRST_SCT-TCD-Decouple-A_epoch2_best.pth.tar"
PSEUDO_IRSTD="$ROOT/generated/sam_refined_irstd_tpc_rim_train"
PSEUDO_NUAA="$ROOT/generated/sam_refined_mix_for_nuaa_rim_train"

launch sam_rim_irstd_gpu0 0 IRSTD-1K SAMRim-TPCDHiF-IRSTD-A sct_tpc_dhif_refiner 9711 0.000014 28 7 \
  "$IRSTD_BASE" "$IRSTD_TPC_B" sct_tpc_dhif_refiner "$IRSTD_BASE" "$IRSTD_TPC_B" "$PSEUDO_IRSTD" \
  "--teacher_weight 0.72 --base_anchor_weight 0.70 --teacher_logit_weight 0.30 --base_anchor_logit_weight 0.15"

launch sam_rim_irstd_gpu1 1 IRSTD-1K SAMRim-TPCDHiF-IRSTD-B sct_tpc_dhif_refiner 9717 0.000012 28 8 \
  "$IRSTD_BASE" "$IRSTD_TPC_A" sct_tpc_dhif_refiner "$IRSTD_BASE" "$IRSTD_TPC_A" "$PSEUDO_IRSTD" \
  "--teacher_weight 0.78 --base_anchor_weight 0.76 --teacher_logit_weight 0.36 --base_anchor_logit_weight 0.18 --pseudo_weight 0.035 --pseudo_rim_weight 0.16"

launch sam_rim_nuaa_gpu2 2 NUAA-SIRST SAMRim-LCA-NUAA sct_lca_refiner 9723 0.000014 30 8 \
  "$NUAA_BASE" "$NUAA_LCA" sct_lca_refiner "$NUAA_BASE" "$NUAA_LCA" "$PSEUDO_NUAA" \
  "--teacher_bg_threshold 0.07 --teacher_pos_threshold 0.34 --base_anchor_bg_threshold 0.07 --base_anchor_pos_threshold 0.34 --pseudo_pos_threshold 0.50"

launch sam_rim_nuaa_gpu3 3 NUAA-SIRST SAMRim-TCD-NUAA sct_tcd_decouple_refiner 9729 0.000012 30 9 \
  "$NUAA_BASE" "$NUAA_TCD" sct_tcd_decouple_refiner "$NUAA_BASE" "$NUAA_TCD" "$PSEUDO_NUAA" \
  "--teacher_weight 0.72 --base_anchor_weight 0.70 --teacher_logit_weight 0.30 --base_anchor_logit_weight 0.15 --teacher_bg_threshold 0.07 --teacher_pos_threshold 0.34 --base_anchor_bg_threshold 0.07 --base_anchor_pos_threshold 0.34 --pseudo_weight 0.035 --pseudo_rim_weight 0.16 --pseudo_pos_threshold 0.50"

tmux ls

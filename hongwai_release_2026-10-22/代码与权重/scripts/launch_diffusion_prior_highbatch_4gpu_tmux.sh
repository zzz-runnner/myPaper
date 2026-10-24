#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/diffusion_prior_highbatch_4gpu"
LOG_ROOT="$RUN_ROOT/logs"
mkdir -p "$LOG_ROOT"

launch() {
  local session="$1"
  local gpu="$2"
  local dataset="$3"
  local method="$4"
  local seed="$5"
  local lr="$6"
  local batch="$7"
  local target_repeat="$8"
  local base_ckpt="$9"
  local resume_ckpt="${10}"
  local teacher_model="${11}"
  local teacher_base="${12}"
  local teacher_ckpt="${13}"
  local pseudo_dir="${14}"
  local extra="${15}"
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
      --model sct_diffusion_prior_refiner \
      --base_checkpoint '$base_ckpt' \
      --resume_model_checkpoint '$resume_ckpt' \
      --teacher_model '$teacher_model' \
      --teacher_base_checkpoint '$teacher_base' \
      --teacher_checkpoint '$teacher_ckpt' \
      --epochs 90 \
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
      --noise_std 0.00028 \
      --gamma_prob 0.0025 \
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
      --teacher_weight 0.62 \
      --teacher_bg_weight 1.0 \
      --teacher_pos_weight 0.58 \
      --teacher_miss_weight 0.16 \
      --teacher_logit_weight 0.22 \
      --teacher_bg_threshold 0.08 \
      --teacher_pos_threshold 0.35 \
      --teacher_margin 0.0025 \
      --teacher_hf_boost 1.0 \
      --teacher_target_dilate 11 \
      --base_anchor_weight 0.60 \
      --base_anchor_bg_weight 1.0 \
      --base_anchor_pos_weight 0.58 \
      --base_anchor_logit_weight 0.10 \
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
      --diffusion_prior_weight 0.014 \
      --diffusion_bg_weight 0.30 \
      --diffusion_align_weight 0.020 \
      --diffusion_conductance_weight 0.003 \
      --diffusion_pos_boost 0.70 \
      --diffusion_bg_boost 0.55 \
      --diffusion_target_dilate 9 \
      $([ -n "$pseudo_dir" ] && printf -- "--pseudo_dir '%s' --pseudo_weight 0.035 --pseudo_bce_weight 1.0 --pseudo_logit_weight 0.010 --pseudo_iou_weight 0.060 --pseudo_pos_weight 0.72 --pseudo_bg_weight 0.38 --pseudo_pos_threshold 0.34 --pseudo_bg_threshold 0.06 --pseudo_hf_boost 0.40 --pseudo_target_dilate 11" "$pseudo_dir") \
      $extra \
      2>&1 | tee '$log'"
}

IRSTD_BASE="$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar"
IRSTD_TPC_B="$ROOT/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-B_gpu1/checkpoints/IRSTD-1K/best_mIoU.pth.tar"
IRSTD_TPC_A="$ROOT/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-A_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar"
NUAA_BASE="$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar"
NUAA_LCA="$ROOT/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar"
NUAA_TCD="$ROOT/runs/selected_checkpoints/tcd_decouple/NUAA-SIRST_SCT-TCD-Decouple-A_epoch2_best.pth.tar"
PSEUDO="$ROOT/generated/pseudo_nuaa_lca_tcd_train"

launch diffusion_prior_irstd_gpu0 0 IRSTD-1K DiffusionPrior-IRSTD-A 9311 0.000018 22 7 \
  "$IRSTD_BASE" "$IRSTD_TPC_B" sct_tpc_dhif_refiner "$IRSTD_BASE" "$IRSTD_TPC_B" "" \
  ""

launch diffusion_prior_irstd_gpu1 1 IRSTD-1K DiffusionPrior-IRSTD-B 9317 0.000016 22 8 \
  "$IRSTD_BASE" "$IRSTD_TPC_A" sct_tpc_dhif_refiner "$IRSTD_BASE" "$IRSTD_TPC_A" "" \
  "--teacher_weight 0.68 --base_anchor_weight 0.66 --teacher_logit_weight 0.28 --base_anchor_logit_weight 0.14 --diffusion_prior_weight 0.012"

launch diffusion_prior_nuaa_gpu2 2 NUAA-SIRST DiffusionPrior-NUAA-LCA 9323 0.000018 22 8 \
  "$NUAA_BASE" "$NUAA_LCA" sct_lca_refiner "$NUAA_BASE" "$NUAA_LCA" "$PSEUDO" \
  "--teacher_bg_threshold 0.07 --teacher_pos_threshold 0.34 --base_anchor_bg_threshold 0.07 --base_anchor_pos_threshold 0.34 --diffusion_prior_weight 0.012 --diffusion_align_weight 0.018"

launch diffusion_prior_nuaa_gpu3 3 NUAA-SIRST DiffusionPrior-NUAA-TCD 9329 0.000016 22 9 \
  "$NUAA_BASE" "$NUAA_TCD" sct_tcd_decouple_refiner "$NUAA_BASE" "$NUAA_TCD" "$PSEUDO" \
  "--teacher_weight 0.68 --base_anchor_weight 0.66 --teacher_logit_weight 0.28 --base_anchor_logit_weight 0.14 --teacher_bg_threshold 0.07 --teacher_pos_threshold 0.34 --base_anchor_bg_threshold 0.07 --base_anchor_pos_threshold 0.34 --diffusion_prior_weight 0.012 --diffusion_align_weight 0.018"

tmux ls

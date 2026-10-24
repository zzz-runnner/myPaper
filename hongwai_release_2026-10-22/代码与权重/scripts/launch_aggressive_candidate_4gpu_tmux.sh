#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/aggressive_candidate_4gpu"
LOG_ROOT="$RUN_ROOT/logs"
mkdir -p "$LOG_ROOT"

COMMON_DATASETS="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3"

launch() {
  local session="$1"
  local gpu="$2"
  local dataset="$3"
  local method="$4"
  local model="$5"
  local seed="$6"
  local lr="$7"
  local batch="$8"
  local repeat="$9"
  local base_ckpt="${10}"
  local resume_ckpt="${11}"
  local teacher_model="${12}"
  local teacher_base="${13}"
  local teacher_ckpt="${14}"
  local pseudo_dir="${15}"
  local extra="${16}"
  local run_dir="$RUN_ROOT/${dataset}_${method}_gpu${gpu}"
  local log="$LOG_ROOT/${dataset}_${method}_gpu${gpu}.log"

  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" \
    "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && CUDA_VISIBLE_DEVICES=$gpu '$PY' '$SCRIPT' \
      --dataset_dir '$ROOT/datasets' \
      --dataset_name '$dataset' \
      --train_datasets '$COMMON_DATASETS' \
      --target_repeat '$repeat' \
      --run_dir '$run_dir' \
      --method_name '$method' \
      --model '$model' \
      --base_checkpoint '$base_ckpt' \
      --resume_model_checkpoint '$resume_ckpt' \
      --teacher_model '$teacher_model' \
      --teacher_base_checkpoint '$teacher_base' \
      --teacher_checkpoint '$teacher_ckpt' \
      --pseudo_dir '$pseudo_dir' \
      --epochs 24 \
      --batch_size '$batch' \
      --threads 5 \
      --optimizer adamw \
      --lr '$lr' \
      --weight_decay 0.000010 \
      --warmup_epochs 1 \
      --min_lr_ratio 0.40 \
      --seed '$seed' \
      --begin_test 1 \
      --every_test 1 \
      --every_print 1 \
      --eval_before_train \
      --pos_prob 0.78 \
      --noise_std 0.00030 \
      --gamma_prob 0.0030 \
      --intensity_jitter 0.0015 \
      --bgm_prob 0.020 \
      --bgm_region_size 16 \
      --bgm_mask_ratio 0.055 \
      --copy_paste_prob 0.22 \
      --copy_paste_max_targets 2 \
      --copy_paste_min_area 1 \
      --copy_paste_max_area 720 \
      --copy_paste_scale_min 0.70 \
      --copy_paste_scale_max 1.42 \
      --copy_paste_alpha_min 0.68 \
      --copy_paste_alpha_max 0.98 \
      --copy_paste_contrast_jitter 0.28 \
      --copy_paste_guard 11 \
      --bce_weight 0.075 \
      --dice_weight 0.040 \
      --soft_iou_weight 0.22 \
      --focal_weight 0.0040 \
      --small_weight 0.62 \
      --pos_weight 0.85 \
      --hard_neg_weight 1.45 \
      --hard_neg_threshold 0.18 \
      --sls_iou_weight 0.012 \
      --sls_iou_warm_epochs 1 \
      --sls_location_weight 0.12 \
      --sdm_iou_weight 0.050 \
      --sdm_warm_epochs 1 \
      --sdm_delta 0.5 \
      --sdm_with_distance \
      --bg_hf_weight 0.018 \
      --bg_pred_weight 0.0040 \
      --target_dilate 7 \
      --teacher_weight 0.42 \
      --teacher_bg_weight 1.0 \
      --teacher_pos_weight 0.55 \
      --teacher_miss_weight 0.26 \
      --teacher_logit_weight 0.08 \
      --teacher_bg_threshold 0.075 \
      --teacher_pos_threshold 0.33 \
      --teacher_margin 0.0060 \
      --teacher_hf_boost 1.15 \
      --teacher_target_dilate 11 \
      --base_anchor_weight 0.45 \
      --base_anchor_bg_weight 1.0 \
      --base_anchor_pos_weight 0.55 \
      --base_anchor_logit_weight 0.06 \
      --base_anchor_bg_threshold 0.075 \
      --base_anchor_pos_threshold 0.33 \
      --base_anchor_margin 0.0050 \
      --base_anchor_hf_boost 1.15 \
      --base_anchor_target_dilate 11 \
      --tcd_gate_weight 0.045 \
      --tcd_target_gate_weight 1.0 \
      --tcd_clutter_gate_weight 0.45 \
      --tcd_anti_target_weight 0.60 \
      --thermal_spot_weight 0.0060 \
      --thermal_spot_sigma_scale 0.66 \
      --thermal_spot_sigma_min 0.8 \
      --thermal_spot_sigma_max 3.8 \
      --thermal_spot_guard 11 \
      --thermal_spot_core_boost 1.28 \
      --thermal_spot_margin 0.060 \
      --thermal_spot_contrast_weight 0.14 \
      --thermal_spot_scale_weight 0.030 \
      --pseudo_weight 0.070 \
      --pseudo_bce_weight 1.0 \
      --pseudo_logit_weight 0.018 \
      --pseudo_iou_weight 0.120 \
      --pseudo_pos_weight 1.10 \
      --pseudo_bg_weight 0.50 \
      --pseudo_pos_threshold 0.42 \
      --pseudo_bg_threshold 0.055 \
      --pseudo_hf_boost 0.55 \
      --pseudo_target_dilate 11 \
      --pseudo_rim_weight 0.14 \
      --pseudo_rim_low 0.10 \
      --pseudo_rim_high 0.48 \
      --pseudo_rim_target 0.32 \
      $extra \
      2>&1 | tee '$log'"
}

IRSTD_BASE="$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar"
IRSTD_TPC_A="$ROOT/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-A_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar"
IRSTD_TPC_B="$ROOT/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-B_gpu1/checkpoints/IRSTD-1K/best_mIoU.pth.tar"
NUAA_BASE="$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar"
NUAA_LCA="$ROOT/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar"
NUAA_TCD_A="$ROOT/runs/selected_checkpoints/tcd_decouple/NUAA-SIRST_SCT-TCD-Decouple-A_epoch2_best.pth.tar"
PSEUDO_IRSTD="$ROOT/generated/sam_refined_irstd_tpc_rim_train"
PSEUDO_NUAA="$ROOT/generated/sam_refined_mix_for_nuaa_rim_train"

launch aggr_irstd_prompt 0 IRSTD-1K AggrPrompt-IRSTD sct_prompt_morph_refiner 14111 0.000014 16 8 \
  "$IRSTD_BASE" "$IRSTD_TPC_B" sct_tpc_dhif_refiner "$IRSTD_BASE" "$IRSTD_TPC_B" "$PSEUDO_IRSTD" \
  "--copy_paste_prob 0.26 --teacher_miss_weight 0.32 --base_anchor_weight 0.38 --teacher_weight 0.38"

launch aggr_irstd_diffusion 1 IRSTD-1K AggrDiffusion-IRSTD sct_diffusion_prior_refiner 14117 0.000012 14 8 \
  "$IRSTD_BASE" "$IRSTD_TPC_A" sct_tpc_dhif_refiner "$IRSTD_BASE" "$IRSTD_TPC_A" "$PSEUDO_IRSTD" \
  "--diffusion_prior_weight 0.065 --diffusion_bg_weight 0.42 --diffusion_align_weight 0.010 --diffusion_conductance_weight 0.016 --copy_paste_prob 0.24 --teacher_miss_weight 0.34 --base_anchor_weight 0.36 --teacher_weight 0.36"

launch aggr_nuaa_prompt 2 NUAA-SIRST AggrPrompt-NUAA sct_prompt_morph_refiner 14123 0.000014 16 8 \
  "$NUAA_BASE" "$NUAA_LCA" sct_lca_refiner "$NUAA_BASE" "$NUAA_LCA" "$PSEUDO_NUAA" \
  "--teacher_bg_threshold 0.065 --base_anchor_bg_threshold 0.065 --teacher_pos_threshold 0.32 --base_anchor_pos_threshold 0.32 --copy_paste_prob 0.24 --teacher_miss_weight 0.32 --base_anchor_weight 0.38 --teacher_weight 0.38"

launch aggr_nuaa_diffusion 3 NUAA-SIRST AggrDiffusion-NUAA sct_diffusion_prior_refiner 14129 0.000012 12 8 \
  "$NUAA_BASE" "$NUAA_TCD_A" sct_tcd_decouple_refiner "$NUAA_BASE" "$NUAA_TCD_A" "$PSEUDO_NUAA" \
  "--diffusion_prior_weight 0.065 --diffusion_bg_weight 0.42 --diffusion_align_weight 0.010 --diffusion_conductance_weight 0.016 --teacher_bg_threshold 0.065 --base_anchor_bg_threshold 0.065 --teacher_pos_threshold 0.32 --base_anchor_pos_threshold 0.32 --copy_paste_prob 0.22 --teacher_miss_weight 0.34 --base_anchor_weight 0.36 --teacher_weight 0.36"

tmux ls

#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
GEN="$ROOT/scripts/generate_teacher_pseudo_labels.py"
TRAIN="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/pseudo_nuaa_gpu3"
LOG_ROOT="$RUN_ROOT/logs"
SESSION="pseudo_nuaa_gpu3"
PSEUDO_ROOT="$ROOT/generated/pseudo_nuaa_lca_tcd_train"

mkdir -p "$LOG_ROOT"
tmux kill-session -t "$SESSION" 2>/dev/null || true
rm -rf "$RUN_ROOT/NUAA-SIRST_PConvTCF-LCATCDPseudo_gpu3"
rm -rf "$PSEUDO_ROOT"

DATASETS="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3"
NUAA_BASE="$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar"
NUAA_LCA="$ROOT/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar"
NUAA_TCD="$ROOT/runs/selected_checkpoints/tcd_decouple/NUAA-SIRST_SCT-TCD-Decouple-A_epoch2_best.pth.tar"

tmux new-session -d -s "$SESSION" -n nuaa_pseudo_train \
  "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && CUDA_VISIBLE_DEVICES=3 bash -lc '\
    set -euo pipefail; \
    \"$PY\" \"$GEN\" \
      --dataset_dir \"$ROOT/datasets\" \
      --datasets \"$DATASETS\" \
      --split train \
      --out_dir \"$PSEUDO_ROOT\" \
      --teacher \"name=nuaa_lca,model=sct_lca_refiner,base_checkpoint=$NUAA_BASE,checkpoint=$NUAA_LCA\" \
      --teacher \"name=nuaa_tcd,model=sct_tcd_decouple_refiner,base_checkpoint=$NUAA_BASE,checkpoint=$NUAA_TCD\" \
      --device cuda:0 \
      --consensus mean \
      --seed 7611; \
    \"$PY\" \"$TRAIN\" \
      --dataset_dir \"$ROOT/datasets\" \
      --dataset_name \"NUAA-SIRST\" \
      --train_datasets \"$DATASETS\" \
      --target_repeat 8 \
      --pseudo_dir \"$PSEUDO_ROOT\" \
      --run_dir \"$RUN_ROOT/NUAA-SIRST_PConvTCF-LCATCDPseudo_gpu3\" \
      --method_name \"PConvTCF-LCATCDPseudo\" \
      --model \"pconv_tcf\" \
      --epochs 120 \
      --batch_size 14 \
      --threads 4 \
      --optimizer adamw \
      --lr 0.00026 \
      --weight_decay 0.00005 \
      --warmup_epochs 4 \
      --min_lr_ratio 0.08 \
      --seed 7613 \
      --begin_test 5 \
      --every_test 5 \
      --every_print 1 \
      --pos_prob 0.60 \
      --noise_std 0.0012 \
      --gamma_prob 0.020 \
      --intensity_jitter 0.004 \
      --bgm_prob 0.11 \
      --bgm_region_size 16 \
      --bgm_mask_ratio 0.10 \
      --bce_weight 0.30 \
      --dice_weight 0.14 \
      --soft_iou_weight 0.56 \
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
      --pseudo_weight 0.34 \
      --pseudo_bce_weight 1.0 \
      --pseudo_logit_weight 0.055 \
      --pseudo_iou_weight 0.28 \
      --pseudo_pos_weight 1.45 \
      --pseudo_bg_weight 0.85 \
      --pseudo_pos_threshold 0.34 \
      --pseudo_bg_threshold 0.06 \
      --pseudo_hf_boost 1.0 \
      --pseudo_target_dilate 11' \
    2>&1 | tee '$LOG_ROOT/NUAA-SIRST_PConvTCF-LCATCDPseudo_gpu3.log'"

tmux ls | grep "$SESSION"

#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/sct_dhif_from_fixed_nuaa_gpu3"
LOG_ROOT="$RUN_ROOT/logs"
DATA="$ROOT/datasets"
TRAIN_DATASETS="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3"

mkdir -p "$LOG_ROOT"
tmux kill-session -t nuaa_dhif_fixed_gpu3 2>/dev/null || true

tmux new-session -d -s nuaa_dhif_fixed_gpu3 -n nuaa_dhif2 \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES=3 '$PY' '$SCRIPT' \
    --dataset_dir '$DATA' \
    --dataset_name 'NUAA-SIRST' \
    --train_datasets '$TRAIN_DATASETS' \
    --target_repeat 5 \
    --run_dir '$RUN_ROOT/NUAA-SIRST_SCT-DHiF-FromFixed2_gpu3' \
    --method_name 'SCT-DHiF-FromFixed2' \
    --model 'sct_dhif_refiner' \
    --base_checkpoint '$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar' \
    --resume_model_checkpoint '$ROOT/runs/sct_refiner_fixed/NUAA-SIRST_SCT-Refiner-Fixed_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar' \
    --epochs 80 \
    --batch_size 48 \
    --threads 4 \
    --lr 0.000015 \
    --weight_decay 0.00005 \
    --seed 843 \
    --begin_test 5 \
    --every_test 5 \
    --every_print 10 \
    --eval_before_train \
    --bgm_prob 0.08 \
    --bgm_mask_ratio 0.05 \
    --sls_iou_weight 0.020 \
    --sls_iou_warm_epochs 1 \
    --bg_hf_weight 0.08 \
    --bg_pred_weight 0.015 \
    --pos_prob 0.55 \
    --noise_std 0.0005 \
    --gamma_prob 0.01 \
    --intensity_jitter 0.003 \
    2>&1 | tee '$LOG_ROOT/NUAA-SIRST_SCT-DHiF-FromFixed2_gpu3.log'"

tmux list-windows -t nuaa_dhif_fixed_gpu3

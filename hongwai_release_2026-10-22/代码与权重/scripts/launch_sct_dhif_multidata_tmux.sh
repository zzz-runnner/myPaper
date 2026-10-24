#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/sct_dhif_multidata"
LOG_ROOT="$RUN_ROOT/logs"
DATA="$ROOT/datasets"
TRAIN_DATASETS="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3"

mkdir -p "$LOG_ROOT"
tmux kill-session -t irstd_sct_dhif_multi 2>/dev/null || true

tmux new-session -d -s irstd_sct_dhif_multi -n irstd_dhif \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES=0 '$PY' '$SCRIPT' \
    --dataset_dir '$DATA' \
    --dataset_name 'IRSTD-1K' \
    --train_datasets '$TRAIN_DATASETS' \
    --run_dir '$RUN_ROOT/IRSTD-1K_SCT-DHiF-Multi_gpu0' \
    --method_name 'SCT-DHiF-Multi' \
    --model 'sct_dhif_refiner' \
    --base_checkpoint '$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar' \
    --epochs 120 \
    --batch_size 48 \
    --threads 4 \
    --lr 0.000035 \
    --weight_decay 0.00005 \
    --seed 801 \
    --begin_test 5 \
    --every_test 5 \
    --every_print 10 \
    --eval_before_train \
    --bgm_prob 0.14 \
    --bgm_mask_ratio 0.08 \
    --sls_iou_weight 0.04 \
    --sls_iou_warm_epochs 1 \
    --bg_hf_weight 0.16 \
    --bg_pred_weight 0.035 \
    --pos_prob 0.55 \
    --noise_std 0.001 \
    --gamma_prob 0.03 \
    --intensity_jitter 0.006 \
    2>&1 | tee '$LOG_ROOT/IRSTD-1K_SCT-DHiF-Multi_gpu0.log'"

tmux new-window -t irstd_sct_dhif_multi -n nuaa_dhif \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES=2 '$PY' '$SCRIPT' \
    --dataset_dir '$DATA' \
    --dataset_name 'NUAA-SIRST' \
    --train_datasets '$TRAIN_DATASETS' \
    --run_dir '$RUN_ROOT/NUAA-SIRST_SCT-DHiF-Multi_gpu2' \
    --method_name 'SCT-DHiF-Multi' \
    --model 'sct_dhif_refiner' \
    --base_checkpoint '$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar' \
    --epochs 120 \
    --batch_size 48 \
    --threads 4 \
    --lr 0.000035 \
    --weight_decay 0.00005 \
    --seed 803 \
    --begin_test 5 \
    --every_test 5 \
    --every_print 10 \
    --eval_before_train \
    --bgm_prob 0.14 \
    --bgm_mask_ratio 0.08 \
    --sls_iou_weight 0.04 \
    --sls_iou_warm_epochs 1 \
    --bg_hf_weight 0.16 \
    --bg_pred_weight 0.035 \
    --pos_prob 0.53 \
    --noise_std 0.0007 \
    --gamma_prob 0.02 \
    --intensity_jitter 0.004 \
    2>&1 | tee '$LOG_ROOT/NUAA-SIRST_SCT-DHiF-Multi_gpu2.log'"

tmux list-windows -t irstd_sct_dhif_multi

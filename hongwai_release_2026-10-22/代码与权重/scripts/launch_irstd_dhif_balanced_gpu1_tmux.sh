#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/sct_dhif_balanced_irstd_gpu1"
LOG_ROOT="$RUN_ROOT/logs"
DATA="$ROOT/datasets"
TRAIN_DATASETS="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3"

mkdir -p "$LOG_ROOT"
tmux kill-session -t irstd_dhif_bal_gpu1 2>/dev/null || true

tmux new-session -d -s irstd_dhif_bal_gpu1 -n irstd_bal2 \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES=1 '$PY' '$SCRIPT' \
    --dataset_dir '$DATA' \
    --dataset_name 'IRSTD-1K' \
    --train_datasets '$TRAIN_DATASETS' \
    --target_repeat 4 \
    --run_dir '$RUN_ROOT/IRSTD-1K_SCT-DHiF-Balanced2_gpu1' \
    --method_name 'SCT-DHiF-Balanced2' \
    --model 'sct_dhif_refiner' \
    --base_checkpoint '$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar' \
    --epochs 100 \
    --batch_size 48 \
    --threads 4 \
    --lr 0.000025 \
    --weight_decay 0.00005 \
    --seed 821 \
    --begin_test 5 \
    --every_test 5 \
    --every_print 10 \
    --eval_before_train \
    --bgm_prob 0.10 \
    --bgm_mask_ratio 0.06 \
    --sls_iou_weight 0.025 \
    --sls_iou_warm_epochs 1 \
    --bg_hf_weight 0.10 \
    --bg_pred_weight 0.020 \
    --pos_prob 0.58 \
    --noise_std 0.0008 \
    --gamma_prob 0.02 \
    --intensity_jitter 0.004 \
    2>&1 | tee '$LOG_ROOT/IRSTD-1K_SCT-DHiF-Balanced2_gpu1.log'"

tmux list-windows -t irstd_dhif_bal_gpu1

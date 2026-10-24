#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/mshnet_irstd_gpu1"
LOG_ROOT="$RUN_ROOT/logs"
DATA="$ROOT/datasets"

mkdir -p "$LOG_ROOT"
tmux kill-session -t irstd_msh_gpu1 2>/dev/null || true

tmux new-session -d -s irstd_msh_gpu1 -n irstd_msh \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES=1 '$PY' '$SCRIPT' \
    --dataset_dir '$DATA' \
    --dataset_name 'IRSTD-1K' \
    --run_dir '$RUN_ROOT/IRSTD-1K_MSHNet-SLS_gpu1' \
    --method_name 'MSHNet-SLS' \
    --model 'mshnet_sls' \
    --epochs 500 \
    --batch_size 18 \
    --threads 4 \
    --optimizer adagrad \
    --lr 0.05 \
    --seed 701 \
    --begin_test 20 \
    --every_test 5 \
    --every_print 10 \
    --bgm_prob 0.20 \
    --bgm_mask_ratio 0.10 \
    --sls_iou_weight 0.18 \
    --sls_iou_warm_epochs 5 \
    --bg_hf_weight 0.12 \
    --bg_pred_weight 0.04 \
    --pos_prob 0.62 \
    --noise_std 0.003 \
    --gamma_prob 0.06 \
    --intensity_jitter 0.012 \
    2>&1 | tee '$LOG_ROOT/IRSTD-1K_MSHNet-SLS_gpu1.log'"

tmux list-windows -t irstd_msh_gpu1

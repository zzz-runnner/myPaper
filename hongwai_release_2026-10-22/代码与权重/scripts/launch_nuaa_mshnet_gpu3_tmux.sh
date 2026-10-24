#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/mshnet_nuaa_gpu3"
LOG_ROOT="$RUN_ROOT/logs"
DATA="$ROOT/datasets"

mkdir -p "$LOG_ROOT"
tmux kill-session -t nuaa_msh_gpu3 2>/dev/null || true

tmux new-session -d -s nuaa_msh_gpu3 -n nuaa_msh \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES=3 '$PY' '$SCRIPT' \
    --dataset_dir '$DATA' \
    --dataset_name 'NUAA-SIRST' \
    --run_dir '$RUN_ROOT/NUAA-SIRST_MSHNet-SLS_gpu3' \
    --method_name 'MSHNet-SLS' \
    --model 'mshnet_sls' \
    --epochs 500 \
    --batch_size 18 \
    --threads 4 \
    --optimizer adagrad \
    --lr 0.05 \
    --seed 703 \
    --begin_test 20 \
    --every_test 5 \
    --every_print 10 \
    --bgm_prob 0.20 \
    --bgm_mask_ratio 0.10 \
    --sls_iou_weight 0.18 \
    --sls_iou_warm_epochs 5 \
    --bg_hf_weight 0.12 \
    --bg_pred_weight 0.04 \
    --pos_prob 0.56 \
    --noise_std 0.002 \
    --gamma_prob 0.04 \
    --intensity_jitter 0.010 \
    2>&1 | tee '$LOG_ROOT/NUAA-SIRST_MSHNet-SLS_gpu3.log'"

tmux list-windows -t nuaa_msh_gpu3

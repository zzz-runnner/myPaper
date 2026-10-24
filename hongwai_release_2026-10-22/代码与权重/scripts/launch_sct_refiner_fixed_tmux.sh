#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/sct_refiner_fixed"
LOG_ROOT="$RUN_ROOT/logs"
DATA="$ROOT/datasets"

mkdir -p "$LOG_ROOT"
tmux kill-session -t irstd_sct_refiner_fixed 2>/dev/null || true

tmux new-session -d -s irstd_sct_refiner_fixed -n irstd_ref \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES=0 '$PY' '$SCRIPT' \
    --dataset_dir '$DATA' \
    --dataset_name 'IRSTD-1K' \
    --run_dir '$RUN_ROOT/IRSTD-1K_SCT-Refiner-Fixed_gpu0' \
    --method_name 'SCT-Refiner-Fixed' \
    --model 'sct_refiner' \
    --base_checkpoint '$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar' \
    --epochs 160 \
    --batch_size 64 \
    --threads 4 \
    --lr 0.00005 \
    --weight_decay 0.00005 \
    --seed 601 \
    --begin_test 1 \
    --every_test 5 \
    --every_print 10 \
    --eval_before_train \
    --bgm_prob 0.10 \
    --bgm_mask_ratio 0.06 \
    --sls_iou_weight 0.04 \
    --sls_iou_warm_epochs 1 \
    --bg_hf_weight 0.12 \
    --bg_pred_weight 0.025 \
    --pos_prob 0.54 \
    --noise_std 0.0008 \
    --gamma_prob 0.02 \
    --intensity_jitter 0.004 \
    2>&1 | tee '$LOG_ROOT/IRSTD-1K_SCT-Refiner-Fixed_gpu0.log'"

tmux new-window -t irstd_sct_refiner_fixed -n nuaa_ref \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES=2 '$PY' '$SCRIPT' \
    --dataset_dir '$DATA' \
    --dataset_name 'NUAA-SIRST' \
    --run_dir '$RUN_ROOT/NUAA-SIRST_SCT-Refiner-Fixed_gpu2' \
    --method_name 'SCT-Refiner-Fixed' \
    --model 'sct_refiner' \
    --base_checkpoint '$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar' \
    --epochs 160 \
    --batch_size 64 \
    --threads 4 \
    --lr 0.00005 \
    --weight_decay 0.00005 \
    --seed 603 \
    --begin_test 1 \
    --every_test 5 \
    --every_print 10 \
    --eval_before_train \
    --bgm_prob 0.10 \
    --bgm_mask_ratio 0.06 \
    --sls_iou_weight 0.04 \
    --sls_iou_warm_epochs 1 \
    --bg_hf_weight 0.12 \
    --bg_pred_weight 0.025 \
    --pos_prob 0.52 \
    --noise_std 0.0005 \
    --gamma_prob 0.01 \
    --intensity_jitter 0.003 \
    2>&1 | tee '$LOG_ROOT/NUAA-SIRST_SCT-Refiner-Fixed_gpu2.log'"

tmux list-windows -t irstd_sct_refiner_fixed

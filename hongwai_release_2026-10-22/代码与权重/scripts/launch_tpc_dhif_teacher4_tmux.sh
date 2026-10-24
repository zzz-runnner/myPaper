#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/tpc_dhif_teacher4"
LOG_ROOT="$RUN_ROOT/logs"
DATA="$ROOT/datasets"
TRAIN_DATASETS="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3"

mkdir -p "$LOG_ROOT"
tmux kill-session -t irstd_tpc_dhif_teacher4 2>/dev/null || true

COMMON="--dataset_dir '$DATA' \
  --train_datasets '$TRAIN_DATASETS' \
  --model 'sct_tpc_dhif_refiner' \
  --teacher_model 'sct_refiner' \
  --epochs 90 \
  --batch_size 48 \
  --threads 4 \
  --weight_decay 0.00005 \
  --begin_test 5 \
  --every_test 5 \
  --every_print 10 \
  --eval_before_train \
  --bgm_prob 0.10 \
  --bgm_mask_ratio 0.06 \
  --sls_iou_weight 0.025 \
  --sls_iou_warm_epochs 1 \
  --bg_hf_weight 0.08 \
  --bg_pred_weight 0.015 \
  --teacher_bg_weight 1.0 \
  --teacher_pos_weight 0.40 \
  --teacher_miss_weight 0.35 \
  --teacher_bg_threshold 0.08 \
  --teacher_pos_threshold 0.35 \
  --teacher_margin 0.012 \
  --teacher_hf_boost 1.2 \
  --teacher_target_dilate 11"

tmux new-session -d -s irstd_tpc_dhif_teacher4 -n irstd_tpc_a \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES=0 '$PY' '$SCRIPT' \
    $COMMON \
    --dataset_name 'IRSTD-1K' \
    --target_repeat 4 \
    --run_dir '$RUN_ROOT/IRSTD-1K_SCT-TPC-DHiF-A_gpu0' \
    --method_name 'SCT-TPC-DHiF-A' \
    --base_checkpoint '$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar' \
    --resume_model_checkpoint '$ROOT/runs/sct_refiner_fixed/IRSTD-1K_SCT-Refiner-Fixed_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar' \
    --teacher_checkpoint '$ROOT/runs/sct_refiner_fixed/IRSTD-1K_SCT-Refiner-Fixed_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar' \
    --lr 0.000018 \
    --seed 851 \
    --teacher_weight 0.35 \
    --pos_prob 0.60 \
    --noise_std 0.0006 \
    --gamma_prob 0.015 \
    --intensity_jitter 0.003 \
    2>&1 | tee '$LOG_ROOT/IRSTD-1K_SCT-TPC-DHiF-A_gpu0.log'"

tmux new-window -t irstd_tpc_dhif_teacher4 -n irstd_tpc_b \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES=1 '$PY' '$SCRIPT' \
    $COMMON \
    --dataset_name 'IRSTD-1K' \
    --target_repeat 5 \
    --run_dir '$RUN_ROOT/IRSTD-1K_SCT-TPC-DHiF-B_gpu1' \
    --method_name 'SCT-TPC-DHiF-B' \
    --base_checkpoint '$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar' \
    --resume_model_checkpoint '$ROOT/runs/sct_refiner_fixed/IRSTD-1K_SCT-Refiner-Fixed_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar' \
    --teacher_checkpoint '$ROOT/runs/sct_refiner_fixed/IRSTD-1K_SCT-Refiner-Fixed_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar' \
    --lr 0.000022 \
    --seed 853 \
    --teacher_weight 0.22 \
    --pos_prob 0.62 \
    --noise_std 0.0008 \
    --gamma_prob 0.02 \
    --intensity_jitter 0.004 \
    2>&1 | tee '$LOG_ROOT/IRSTD-1K_SCT-TPC-DHiF-B_gpu1.log'"

tmux new-window -t irstd_tpc_dhif_teacher4 -n nuaa_tpc_a \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES=2 '$PY' '$SCRIPT' \
    $COMMON \
    --dataset_name 'NUAA-SIRST' \
    --target_repeat 4 \
    --run_dir '$RUN_ROOT/NUAA-SIRST_SCT-TPC-DHiF-A_gpu2' \
    --method_name 'SCT-TPC-DHiF-A' \
    --base_checkpoint '$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar' \
    --resume_model_checkpoint '$ROOT/runs/sct_refiner_fixed/NUAA-SIRST_SCT-Refiner-Fixed_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar' \
    --teacher_checkpoint '$ROOT/runs/sct_refiner_fixed/NUAA-SIRST_SCT-Refiner-Fixed_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar' \
    --lr 0.000016 \
    --seed 855 \
    --teacher_weight 0.32 \
    --pos_prob 0.55 \
    --noise_std 0.0005 \
    --gamma_prob 0.012 \
    --intensity_jitter 0.003 \
    2>&1 | tee '$LOG_ROOT/NUAA-SIRST_SCT-TPC-DHiF-A_gpu2.log'"

tmux new-window -t irstd_tpc_dhif_teacher4 -n nuaa_tpc_b \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES=3 '$PY' '$SCRIPT' \
    $COMMON \
    --dataset_name 'NUAA-SIRST' \
    --target_repeat 6 \
    --run_dir '$RUN_ROOT/NUAA-SIRST_SCT-TPC-DHiF-B_gpu3' \
    --method_name 'SCT-TPC-DHiF-B' \
    --base_checkpoint '$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar' \
    --resume_model_checkpoint '$ROOT/runs/sct_refiner_fixed/NUAA-SIRST_SCT-Refiner-Fixed_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar' \
    --teacher_checkpoint '$ROOT/runs/sct_refiner_fixed/NUAA-SIRST_SCT-Refiner-Fixed_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar' \
    --lr 0.000019 \
    --seed 857 \
    --teacher_weight 0.20 \
    --pos_prob 0.58 \
    --noise_std 0.0007 \
    --gamma_prob 0.018 \
    --intensity_jitter 0.004 \
    2>&1 | tee '$LOG_ROOT/NUAA-SIRST_SCT-TPC-DHiF-B_gpu3.log'"

tmux list-windows -t irstd_tpc_dhif_teacher4

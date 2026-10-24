#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/eval_pixel_stacker.py"
LOGDIR="$ROOT/reports/pixel_stacker_logs"
mkdir -p "$LOGDIR"

COMMON="\
  --fit_datasets IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3 \
  --pos_per_image 192 \
  --neg_per_image 768 \
  --hard_neg_quantile 94.0 \
  --steps 1800 \
  --lr 0.06 \
  --l2 0.002 \
  --batch_pixels 65536 \
  --thresholds 0.35,0.40,0.45,0.50,0.55,0.60"

tmux kill-session -t pix_irstd_mixed 2>/dev/null || true
tmux kill-session -t pix_irstd_tpc 2>/dev/null || true
tmux kill-session -t pix_nuaa_tpc_lca 2>/dev/null || true
tmux kill-session -t pix_nuaa_lca_tcd 2>/dev/null || true

tmux new-session -d -s pix_irstd_mixed "cd '$ROOT' && CUDA_VISIBLE_DEVICES=0 '$PY' '$SCRIPT' \
  --dataset_name IRSTD-1K \
  --model_spec 'baseq|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-B_gpu1/checkpoints/IRSTD-1K/best_mIoU.pth.tar' \
  --model_spec 'refiner|sct_refiner|$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|$ROOT/runs/sct_refiner_fixed/IRSTD-1K_SCT-Refiner-Fixed_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar' \
  $COMMON --device cuda:0 \
  --output '$ROOT/reports/pixel_stacker_multidata_irstd_mixed.json' \
  > '$LOGDIR/irstd_mixed.log' 2>&1"

tmux new-session -d -s pix_irstd_tpc "cd '$ROOT' && CUDA_VISIBLE_DEVICES=1 '$PY' '$SCRIPT' \
  --dataset_name IRSTD-1K \
  --model_spec 'tpc_a|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-A_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar' \
  --model_spec 'tpc_b|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-B_gpu1/checkpoints/IRSTD-1K/best_mIoU.pth.tar' \
  $COMMON --device cuda:0 \
  --output '$ROOT/reports/pixel_stacker_multidata_irstd_tpc.json' \
  > '$LOGDIR/irstd_tpc.log' 2>&1"

tmux new-session -d -s pix_nuaa_tpc_lca "cd '$ROOT' && CUDA_VISIBLE_DEVICES=2 '$PY' '$SCRIPT' \
  --dataset_name NUAA-SIRST \
  --model_spec 'tpc_a|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/NUAA-SIRST_SCT-TPC-DHiF-A_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar' \
  --model_spec 'tpc_b|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/NUAA-SIRST_SCT-TPC-DHiF-B_gpu3/checkpoints/NUAA-SIRST/best_mIoU.pth.tar' \
  --model_spec 'lca|sct_lca_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar' \
  $COMMON --device cuda:0 \
  --output '$ROOT/reports/pixel_stacker_multidata_nuaa_tpc_lca.json' \
  > '$LOGDIR/nuaa_tpc_lca.log' 2>&1"

tmux new-session -d -s pix_nuaa_lca_tcd "cd '$ROOT' && CUDA_VISIBLE_DEVICES=3 '$PY' '$SCRIPT' \
  --dataset_name NUAA-SIRST \
  --model_spec 'lca|sct_lca_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar' \
  --model_spec 'tcd_a|sct_tcd_decouple_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/tcd_decouple/NUAA-SIRST_SCT-TCD-Decouple-A_epoch2_best.pth.tar' \
  --model_spec 'tcd_b|sct_tcd_decouple_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/tcd_decouple/NUAA-SIRST_SCT-TCD-Decouple-B_epoch2_best.pth.tar' \
  $COMMON --device cuda:0 \
  --output '$ROOT/reports/pixel_stacker_multidata_nuaa_lca_tcd.json' \
  > '$LOGDIR/nuaa_lca_tcd.log' 2>&1"

tmux ls | grep pix_

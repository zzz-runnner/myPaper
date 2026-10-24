#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cx/hongwai/irstd-lab
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/eval_component_fusion_learner.py"
LOG_DIR="$ROOT/reports/target_proto_gated_logs"
mkdir -p "$LOG_DIR"

COMMON=(--fit_datasets IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3
  --target_prototype_candidates
  --prototype_topk 8
  --prototype_threshold 0.82
  --prototype_teacher_suppress 0.25
  --seed_thresholds 0.40,0.45,0.50
  --keep_thresholds 0.10,0.15,0.20,0.30,0.40
  --output_thresholds 0.40,0.45,0.50
  --steps 1600
  --device cuda:0)

tmux new-session -d -s proto_irstd_mixed "cd '$ROOT' && CUDA_VISIBLE_DEVICES=0 '$PY' '$SCRIPT' \
  --dataset_name IRSTD-1K \
  --model_spec 'baseq|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-B_gpu1/checkpoints/IRSTD-1K/best_mIoU.pth.tar' \
  --model_spec 'refiner|sct_refiner|$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|$ROOT/runs/sct_refiner_fixed/IRSTD-1K_SCT-Refiner-Fixed_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar' \
  ${COMMON[*]} \
  --output '$ROOT/reports/target_proto_gated_multidata_irstd_mixed.json' \
  > '$LOG_DIR/irstd_mixed.log' 2>&1"

tmux new-session -d -s proto_irstd_tpc "cd '$ROOT' && CUDA_VISIBLE_DEVICES=1 '$PY' '$SCRIPT' \
  --dataset_name IRSTD-1K \
  --model_spec 'tpc_a|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-A_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar' \
  --model_spec 'tpc_b|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-B_gpu1/checkpoints/IRSTD-1K/best_mIoU.pth.tar' \
  ${COMMON[*]} \
  --output '$ROOT/reports/target_proto_gated_multidata_irstd_tpc.json' \
  > '$LOG_DIR/irstd_tpc.log' 2>&1"

tmux new-session -d -s proto_nuaa_tpc_lca "cd '$ROOT' && CUDA_VISIBLE_DEVICES=2 '$PY' '$SCRIPT' \
  --dataset_name NUAA-SIRST \
  --model_spec 'tpc_a|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/NUAA-SIRST_SCT-TPC-DHiF-A_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar' \
  --model_spec 'tpc_b|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/NUAA-SIRST_SCT-TPC-DHiF-B_gpu3/checkpoints/NUAA-SIRST/best_mIoU.pth.tar' \
  --model_spec 'lca|sct_lca_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar' \
  ${COMMON[*]} \
  --output '$ROOT/reports/target_proto_gated_multidata_nuaa_tpc_lca.json' \
  > '$LOG_DIR/nuaa_tpc_lca.log' 2>&1"

tmux new-session -d -s proto_nuaa_lca_tcd "cd '$ROOT' && CUDA_VISIBLE_DEVICES=3 '$PY' '$SCRIPT' \
  --dataset_name NUAA-SIRST \
  --model_spec 'lca|sct_lca_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar' \
  --model_spec 'tcd_a|sct_tcd_decouple_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/tcd_decouple/NUAA-SIRST_SCT-TCD-Decouple-A_epoch2_best.pth.tar' \
  --model_spec 'tcd_b|sct_tcd_decouple_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/tcd_decouple/NUAA-SIRST_SCT-TCD-Decouple-B_epoch2_best.pth.tar' \
  ${COMMON[*]} \
  --output '$ROOT/reports/target_proto_gated_multidata_nuaa_lca_tcd.json' \
  > '$LOG_DIR/nuaa_lca_tcd.log' 2>&1"

tmux ls | grep proto_

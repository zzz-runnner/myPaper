#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/build_bgm_rpcanet_fusion_cache.py"

MODEL_ARGS=(
  --model_spec "tpc_a|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/NUAA-SIRST_SCT-TPC-DHiF-A_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar"
  --model_spec "tpc_b|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/NUAA-SIRST_SCT-TPC-DHiF-B_gpu3/checkpoints/NUAA-SIRST/best_mIoU.pth.tar"
  --model_spec "lca|sct_lca_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar"
  --official_bgm_spec "bgm_nuaa|NUAA-SIRST|$ROOT/external/BGM/checkpoints_download/weights/BasicUNet_plus-2024-09-20-21-17-03-NUAA-SIRST/best_mIoU_on_NUAA-SIRST.pth.tar|0.25"
  --rpcanet_spec "rpcanet_sirst_s3|rpcanet_pp_s3|$ROOT/external/RPCANet_hf/result/ISTD/SIRST/RPCANet++_s3.pkl|0.50|256"
)

tmux kill-session -t build_fusion_cache_a 2>/dev/null || true
tmux kill-session -t build_fusion_cache_b 2>/dev/null || true

tmux new-session -d -s build_fusion_cache_a -n train_a "cd '$ROOT' && CUDA_VISIBLE_DEVICES=2 '$PY' '$SCRIPT' \
  --dataset_split IRSTD-1K:train \
  --dataset_split NUAA-SIRST:train \
  --device cuda:0 \
  --output_dir reports/fusion_cache_bgm_rpcanet_train_a \
  ${MODEL_ARGS[*]@Q} 2>&1 | tee reports/fusion_cache_bgm_rpcanet_train_a.log"

tmux new-session -d -s build_fusion_cache_b -n train_b_test "cd '$ROOT' && CUDA_VISIBLE_DEVICES=3 '$PY' '$SCRIPT' \
  --dataset_split NUDT-SIRST:train \
  --dataset_split SIRST3:train \
  --dataset_split NUAA-SIRST:test \
  --device cuda:0 \
  --output_dir reports/fusion_cache_bgm_rpcanet_train_b_test \
  ${MODEL_ARGS[*]@Q} 2>&1 | tee reports/fusion_cache_bgm_rpcanet_train_b_test.log"

tmux ls | rg 'build_fusion_cache|hfm_lk' || true

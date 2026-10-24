#!/usr/bin/env bash
set -euo pipefail
cd /home/cx/hongwai/irstd-lab
CUDA_VISIBLE_DEVICES=2 /home/cx/hongwai/irstd-lab/envs/irstd/bin/python scripts/build_prediction_cache.py \
  --dataset_dir datasets \
  --dataset_split IRSTD-1K:train --dataset_split NUAA-SIRST:train \
  --device cuda:0 --output_dir reports/prediction_cache_nuaa_tpc_lca_train_a \
  --model_spec 'tpc_a|sct_tpc_dhif_refiner|/home/cx/hongwai/irstd-lab/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|/home/cx/hongwai/irstd-lab/runs/tpc_dhif_teacher4/NUAA-SIRST_SCT-TPC-DHiF-A_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar' \
  --model_spec 'tpc_b|sct_tpc_dhif_refiner|/home/cx/hongwai/irstd-lab/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|/home/cx/hongwai/irstd-lab/runs/tpc_dhif_teacher4/NUAA-SIRST_SCT-TPC-DHiF-B_gpu3/checkpoints/NUAA-SIRST/best_mIoU.pth.tar' \
  --model_spec 'lca|sct_lca_refiner|/home/cx/hongwai/irstd-lab/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|/home/cx/hongwai/irstd-lab/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar'

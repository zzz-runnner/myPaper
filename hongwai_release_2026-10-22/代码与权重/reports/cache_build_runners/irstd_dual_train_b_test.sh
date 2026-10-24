#!/usr/bin/env bash
set -euo pipefail
cd /home/cx/hongwai/irstd-lab
CUDA_VISIBLE_DEVICES=1 /home/cx/hongwai/irstd-lab/envs/irstd/bin/python scripts/build_prediction_cache.py \
  --dataset_dir datasets \
  --dataset_split NUDT-SIRST:train --dataset_split SIRST3:train --dataset_split IRSTD-1K:test \
  --device cuda:0 --output_dir reports/prediction_cache_irstd_dual_train_b_test \
  --model_spec 'baseq|sct_tpc_dhif_refiner|/home/cx/hongwai/irstd-lab/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|/home/cx/hongwai/irstd-lab/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-B_gpu1/checkpoints/IRSTD-1K/best_mIoU.pth.tar' \
  --model_spec 'refiner|sct_refiner|/home/cx/hongwai/irstd-lab/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|/home/cx/hongwai/irstd-lab/runs/sct_refiner_fixed/IRSTD-1K_SCT-Refiner-Fixed_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar' \
  --official_pconv_mshnet_spec 'pconv_d05|mshnet|/home/cx/hongwai/irstd-lab/external/PConv-SDloss-Data/weight_extracted/MSHNet-IRSTD-1K-D-0.5/weight.pkl|official_mshnet|0.35|false|256' \
  --official_pconv_mshnet_spec 'pconv_p43|p43|/home/cx/hongwai/irstd-lab/external/PConv-SDloss-Data/weight_extracted/MSHNet-IRSTD-1K-P43/weight.pkl|official_mshnet|0.60|false|256'

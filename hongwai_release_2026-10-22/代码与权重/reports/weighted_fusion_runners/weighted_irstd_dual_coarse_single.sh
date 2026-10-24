#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/cx/hongwai/irstd-lab
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
$ROOT/envs/irstd/bin/python $ROOT/scripts/eval_weighted_candidate_fusion.py \
  --dataset_dir $ROOT/datasets \
  --fit_datasets IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3 \
  --dataset_name IRSTD-1K \
  --output_thresholds 0.40,0.45,0.50,0.55 \
  --gammas 0.9,1.0,1.1 \
  --top_fit 20 \
  --weight_mode coarse \
  --model_spec 'baseq|sct_tpc_dhif_refiner|/home/cx/hongwai/irstd-lab/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|/home/cx/hongwai/irstd-lab/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-B_gpu1/checkpoints/IRSTD-1K/best_mIoU.pth.tar' \
  --model_spec 'refiner|sct_refiner|/home/cx/hongwai/irstd-lab/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|/home/cx/hongwai/irstd-lab/runs/sct_refiner_fixed/IRSTD-1K_SCT-Refiner-Fixed_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar' \
  --official_pconv_mshnet_spec 'pconv_d05|mshnet|/home/cx/hongwai/irstd-lab/external/PConv-SDloss-Data/weight_extracted/MSHNet-IRSTD-1K-D-0.5/weight.pkl|official_mshnet|0.35|false|256' \
  --official_pconv_mshnet_spec 'pconv_p43|p43|/home/cx/hongwai/irstd-lab/external/PConv-SDloss-Data/weight_extracted/MSHNet-IRSTD-1K-P43/weight.pkl|official_mshnet|0.60|false|256' \
  --device cuda:0 \
  --output $ROOT/reports/weighted_fusion_irstd_dual_coarse_single.json 2>&1 | tee $ROOT/reports/weighted_fusion_logs/weighted_irstd_dual_coarse_single.log

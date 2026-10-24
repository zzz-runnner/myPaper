#!/usr/bin/env bash
set -euo pipefail
cd '/home/cx/hongwai/irstd-lab'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
'/home/cx/hongwai/irstd-lab/envs/irstd/bin/python' '/home/cx/hongwai/irstd-lab/scripts/train_local_candidate_mixer_from_cache.py' \
  --train_cache_spec 'IRSTD-1K|reports/prediction_cache_irstd_dual_train_a/IRSTD-1K__train.npz' \
  --train_cache_spec 'NUAA-SIRST|reports/prediction_cache_irstd_dual_train_a/NUAA-SIRST__train.npz' \
  --train_cache_spec 'NUDT-SIRST|reports/prediction_cache_irstd_dual_train_b_test/NUDT-SIRST__train.npz' \
  --train_cache_spec 'SIRST3|reports/prediction_cache_irstd_dual_train_b_test/SIRST3__train.npz' \
  --test_cache_spec 'IRSTD-1K|reports/prediction_cache_irstd_dual_train_b_test/IRSTD-1K__test.npz' \
  --epochs 6 \
  --batch_size 5 \
  --patch_size 512 \
  --base_channels 64 \
  --lr 0.00018 \
  --residual_scale 0.12 \
  --anchor_weight 0.025 \
  --entropy_weight 0.004 \
  --tv_weight 0.002 \
  --pos_weight 25.0 \
  --preferred 'avg' \
  --eval_interval 2 \
  --thresholds '0.30,0.35,0.368,0.38,0.40,0.42,0.45,0.50,0.55,0.60' \
  --device cuda:0 \
  --output_dir '/home/cx/hongwai/irstd-lab/runs/local_candidate_mixer_probe/irstd_mixer_bal_s121' \
  --seed 121

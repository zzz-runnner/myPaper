#!/usr/bin/env bash
set -euo pipefail
cd /home/cx/hongwai/irstd-lab
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec ./envs/irstd/bin/python scripts/train_teacher_anchor_refiner_from_cache.py --train_cache_spec 'NUAA-SIRST|reports/prediction_cache_nuaa_tpc_lca_train_a/NUAA-SIRST__train.npz' --test_cache_spec 'NUAA-SIRST|reports/prediction_cache_nuaa_tpc_lca_train_b_test/NUAA-SIRST__test.npz' --epochs 4 --batch_size 5 --patch_size 512 --base_channels 64 --lr 2e-4 --residual_scale 1.0 --anchor_weight 0.05 --eval_interval 2 --thresholds 0.30,0.35,0.38,0.40,0.42,0.45,0.50,0.55,0.60 --device cuda:0 --output_dir runs/anchor_refiner_probe/nuaa_r10_a005_s74 --seed 74

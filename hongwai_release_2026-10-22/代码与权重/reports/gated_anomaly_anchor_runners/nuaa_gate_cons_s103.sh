#!/usr/bin/env bash
set -euo pipefail
cd /home/cx/hongwai/irstd-lab
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec ./envs/irstd/bin/python scripts/train_gated_anomaly_anchor_from_cache.py \
  --train_cache_spec 'IRSTD-1K|reports/prediction_cache_nuaa_tpc_lca_train_a/IRSTD-1K__train.npz' \
  --train_cache_spec 'NUAA-SIRST|reports/prediction_cache_nuaa_tpc_lca_train_a/NUAA-SIRST__train.npz' \
  --train_cache_spec 'NUDT-SIRST|reports/prediction_cache_nuaa_tpc_lca_train_b_test/NUDT-SIRST__train.npz' \
  --train_cache_spec 'SIRST3|reports/prediction_cache_nuaa_tpc_lca_train_b_test/SIRST3__train.npz' \
  --test_cache_spec 'NUAA-SIRST|reports/prediction_cache_nuaa_tpc_lca_train_b_test/NUAA-SIRST__test.npz' \
  --epochs 4 --batch_size 3 --patch_size 512 --base_channels 56 --anomaly_base_channels 28 --lr 2e-4 \
  --residual_scale 0.42 --anomaly_scale 0.45 --anchor_weight 0.025 --anomaly_weight 0.18 --gate_weight 0.015 \
  --eval_interval 2 --thresholds 0.30,0.35,0.38,0.40,0.42,0.45,0.50,0.55,0.60 \
  --device cuda:0 --output_dir runs/gated_anomaly_anchor_probe/nuaa_gate_cons_s103 --seed 103

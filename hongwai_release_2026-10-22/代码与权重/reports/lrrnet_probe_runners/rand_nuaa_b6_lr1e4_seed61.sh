#!/usr/bin/env bash
set -euo pipefail
cd /home/cx/hongwai/irstd-lab
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec ./envs/irstd/bin/python scripts/train_lrrnet_multidata.py --epochs 6 --batch_size 6 --patch_size 512 --lr 1e-4 --eta_min 1e-6 --eval_interval 2 --thresholds 0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90 --device cuda:0 --output_dir runs/lrrnet_multidata_probe/rand_nuaa_b6_lr1e4_seed61 --seed 61 --norm_dataset NUAA-SIRST

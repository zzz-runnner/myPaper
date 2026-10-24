#!/usr/bin/env bash
set -euo pipefail
cd /home/cx/hongwai/irstd-lab
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec ./envs/irstd/bin/python scripts/train_lrrnet_multidata.py --epochs 6 --batch_size 6 --patch_size 512 --lr 5e-5 --eta_min 1e-6 --eval_interval 2 --thresholds 0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90 --device cuda:0 --output_dir runs/lrrnet_multidata_probe/finetune_irstd_b6_lr5e5_seed64 --seed 64 --norm_dataset IRSTD-1K --init_checkpoint external/LRRNet/checkpoints/IRDST_1K_miou_0.7239356450020329.pkl

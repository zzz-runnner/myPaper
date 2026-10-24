#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hfm_lk_anchor_refiner_from_cache.py"
OUT="$ROOT/runs/strong_fusion_hfmlk_nuaa_4gpu"

TRAIN_ARGS=(
  --train_cache_spec "IRSTD-1K|$ROOT/reports/fusion_cache_bgm_rpcanet_train_a/IRSTD-1K__train.npz"
  --train_cache_spec "NUAA-SIRST|$ROOT/reports/fusion_cache_bgm_rpcanet_train_a/NUAA-SIRST__train.npz"
  --train_cache_spec "NUDT-SIRST|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/NUDT-SIRST__train.npz"
  --train_cache_spec "SIRST3|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/SIRST3__train.npz"
  --test_cache_spec "NUAA-SIRST|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/NUAA-SIRST__test.npz"
  --epochs 8
  --patch_size 512
  --eval_interval 1
  --thresholds "0.46,0.48,0.50,0.52,0.54,0.56,0.58,0.60"
  --amp
  --batch_size 2
)

mkdir -p "$OUT"

tmux kill-session -t sf_hfmlk_nuaa_a 2>/dev/null || true
tmux kill-session -t sf_hfmlk_nuaa_b 2>/dev/null || true
tmux kill-session -t sf_hfmlk_nuaa_c 2>/dev/null || true
tmux kill-session -t sf_hfmlk_nuaa_d 2>/dev/null || true

tmux new-session -d -s sf_hfmlk_nuaa_a "cd '$ROOT' && CUDA_VISIBLE_DEVICES=0 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --teacher_mode weighted_raw \
  --base_channels 48 \
  --lr 0.00018 \
  --residual_scale 0.28 \
  --anchor_weight 0.035 \
  --pos_weight 30 \
  --hard_clutter_weight 1.2 \
  --hard_clutter_threshold 0.42 \
  --miss_weight 2.2 \
  --miss_threshold 0.18 \
  --uncertainty_weight 0.45 \
  --weighted_dice \
  --seed 701 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_weighted_raw_s701' 2>&1 | tee '$OUT/nuaa_weighted_raw_s701.log'"

tmux new-session -d -s sf_hfmlk_nuaa_b "cd '$ROOT' && CUDA_VISIBLE_DEVICES=1 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --teacher_mode weighted_raw \
  --base_channels 56 \
  --lr 0.00014 \
  --residual_scale 0.22 \
  --anchor_weight 0.050 \
  --pos_weight 34 \
  --hard_clutter_weight 1.8 \
  --hard_clutter_threshold 0.45 \
  --miss_weight 1.8 \
  --miss_threshold 0.16 \
  --uncertainty_weight 0.35 \
  --weighted_dice \
  --seed 702 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_weighted_raw_s702' 2>&1 | tee '$OUT/nuaa_weighted_raw_s702.log'"

tmux new-session -d -s sf_hfmlk_nuaa_c "cd '$ROOT' && CUDA_VISIBLE_DEVICES=2 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --teacher_mode strong_fusion \
  --base_channels 48 \
  --lr 0.00020 \
  --residual_scale 0.32 \
  --anchor_weight 0.030 \
  --pos_weight 28 \
  --hard_clutter_weight 0.9 \
  --hard_clutter_threshold 0.40 \
  --miss_weight 2.8 \
  --miss_threshold 0.20 \
  --uncertainty_weight 0.55 \
  --weighted_dice \
  --seed 703 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_strong_fusion_s703' 2>&1 | tee '$OUT/nuaa_strong_fusion_s703.log'"

tmux new-session -d -s sf_hfmlk_nuaa_d "cd '$ROOT' && CUDA_VISIBLE_DEVICES=3 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --teacher_mode strong_fusion \
  --base_channels 56 \
  --lr 0.00016 \
  --residual_scale 0.26 \
  --anchor_weight 0.045 \
  --pos_weight 32 \
  --hard_clutter_weight 1.5 \
  --hard_clutter_threshold 0.44 \
  --miss_weight 2.4 \
  --miss_threshold 0.18 \
  --uncertainty_weight 0.40 \
  --weighted_dice \
  --seed 704 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_strong_fusion_s704' 2>&1 | tee '$OUT/nuaa_strong_fusion_s704.log'"

tmux ls | rg 'sf_hfmlk_nuaa' || true

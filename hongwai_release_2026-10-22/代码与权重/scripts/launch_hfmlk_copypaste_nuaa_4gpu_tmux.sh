#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hfm_lk_anchor_refiner_from_cache.py"
OUT="$ROOT/runs/hfmlk_copypaste_nuaa_4gpu"

TRAIN_ARGS=(
  --train_cache_spec "IRSTD-1K|$ROOT/reports/fusion_cache_bgm_rpcanet_train_a/IRSTD-1K__train.npz"
  --train_cache_spec "NUAA-SIRST|$ROOT/reports/fusion_cache_bgm_rpcanet_train_a/NUAA-SIRST__train.npz"
  --train_cache_spec "NUDT-SIRST|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/NUDT-SIRST__train.npz"
  --train_cache_spec "SIRST3|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/SIRST3__train.npz"
  --test_cache_spec "NUAA-SIRST|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/NUAA-SIRST__test.npz"
  --epochs 5
  --patch_size 512
  --eval_interval 1
  --thresholds "0.42,0.44,0.46,0.472,0.48,0.50,0.52,0.54,0.56,0.58,0.60"
  --amp
  --batch_size 2
  --weighted_dice
  --copy_paste_max_area 128
)

mkdir -p "$OUT"

tmux kill-session -t hfmlk_cp_nuaa_a 2>/dev/null || true
tmux kill-session -t hfmlk_cp_nuaa_b 2>/dev/null || true
tmux kill-session -t hfmlk_cp_nuaa_c 2>/dev/null || true
tmux kill-session -t hfmlk_cp_nuaa_d 2>/dev/null || true

tmux new-session -d -s hfmlk_cp_nuaa_a "cd '$ROOT' && CUDA_VISIBLE_DEVICES=0 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --teacher_mode strong_fusion \
  --base_channels 48 \
  --lr 0.00016 \
  --residual_scale 0.24 \
  --anchor_weight 0.070 \
  --pos_weight 30 \
  --hard_clutter_weight 1.8 \
  --hard_clutter_threshold 0.42 \
  --miss_weight 1.6 \
  --miss_threshold 0.16 \
  --uncertainty_weight 0.35 \
  --copy_paste_prob 0.35 \
  --seed 931 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_cp_strong_s931' 2>&1 | tee '$OUT/nuaa_cp_strong_s931.log'"

tmux new-session -d -s hfmlk_cp_nuaa_b "cd '$ROOT' && CUDA_VISIBLE_DEVICES=1 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --teacher_mode strong_fusion \
  --base_channels 56 \
  --lr 0.00012 \
  --residual_scale 0.18 \
  --anchor_weight 0.095 \
  --pos_weight 32 \
  --hard_clutter_weight 2.6 \
  --hard_clutter_threshold 0.44 \
  --miss_weight 1.1 \
  --miss_threshold 0.14 \
  --uncertainty_weight 0.25 \
  --copy_paste_prob 0.55 \
  --seed 932 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_cp_strong_s932' 2>&1 | tee '$OUT/nuaa_cp_strong_s932.log'"

tmux new-session -d -s hfmlk_cp_nuaa_c "cd '$ROOT' && CUDA_VISIBLE_DEVICES=2 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --teacher_mode weighted_raw \
  --base_channels 48 \
  --lr 0.00014 \
  --residual_scale 0.22 \
  --anchor_weight 0.075 \
  --pos_weight 32 \
  --hard_clutter_weight 2.2 \
  --hard_clutter_threshold 0.43 \
  --miss_weight 1.3 \
  --miss_threshold 0.15 \
  --uncertainty_weight 0.30 \
  --copy_paste_prob 0.45 \
  --seed 933 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_cp_weighted_s933' 2>&1 | tee '$OUT/nuaa_cp_weighted_s933.log'"

tmux new-session -d -s hfmlk_cp_nuaa_d "cd '$ROOT' && CUDA_VISIBLE_DEVICES=3 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --teacher_mode weighted_raw \
  --base_channels 56 \
  --lr 0.00010 \
  --residual_scale 0.16 \
  --anchor_weight 0.110 \
  --pos_weight 36 \
  --hard_clutter_weight 3.0 \
  --hard_clutter_threshold 0.46 \
  --miss_weight 0.9 \
  --miss_threshold 0.13 \
  --uncertainty_weight 0.20 \
  --copy_paste_prob 0.70 \
  --seed 934 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_cp_weighted_s934' 2>&1 | tee '$OUT/nuaa_cp_weighted_s934.log'"

tmux ls | rg 'hfmlk_cp_nuaa' || true

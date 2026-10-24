#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hfm_lk_anchor_refiner_from_cache.py"
OUT="$ROOT/runs/hfmlk_focus_nuaa_4gpu"

TRAIN_ARGS=(
  --train_cache_spec "IRSTD-1K|$ROOT/reports/fusion_cache_bgm_rpcanet_train_a/IRSTD-1K__train.npz"
  --train_cache_spec "NUAA-SIRST|$ROOT/reports/fusion_cache_bgm_rpcanet_train_a/NUAA-SIRST__train.npz"
  --train_cache_spec "NUDT-SIRST|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/NUDT-SIRST__train.npz"
  --train_cache_spec "SIRST3|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/SIRST3__train.npz"
  --test_cache_spec "NUAA-SIRST|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/NUAA-SIRST__test.npz"
  --epochs 5
  --patch_size 512
  --eval_interval 1
  --thresholds "0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.56,0.58,0.60"
  --focus_dataset NUAA-SIRST
  --amp
  --batch_size 2
)

mkdir -p "$OUT"

tmux kill-session -t hfmlk_focus_nuaa_a 2>/dev/null || true
tmux kill-session -t hfmlk_focus_nuaa_b 2>/dev/null || true
tmux kill-session -t hfmlk_focus_nuaa_c 2>/dev/null || true
tmux kill-session -t hfmlk_focus_nuaa_d 2>/dev/null || true

tmux new-session -d -s hfmlk_focus_nuaa_a "cd '$ROOT' && CUDA_VISIBLE_DEVICES=0 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --teacher_mode strong_fusion \
  --base_channels 48 \
  --lr 0.00016 \
  --residual_scale 0.24 \
  --anchor_weight 0.060 \
  --pos_weight 30 \
  --hard_clutter_weight 2.2 \
  --hard_clutter_threshold 0.42 \
  --miss_weight 1.6 \
  --miss_threshold 0.16 \
  --uncertainty_weight 0.40 \
  --focus_patch_prob 0.75 \
  --focus_hard_threshold 0.42 \
  --focus_miss_threshold 0.16 \
  --weighted_dice \
  --seed 901 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_focus_strong_s901' 2>&1 | tee '$OUT/nuaa_focus_strong_s901.log'"

tmux new-session -d -s hfmlk_focus_nuaa_b "cd '$ROOT' && CUDA_VISIBLE_DEVICES=1 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --teacher_mode strong_fusion \
  --base_channels 48 \
  --lr 0.00012 \
  --residual_scale 0.18 \
  --anchor_weight 0.090 \
  --pos_weight 32 \
  --hard_clutter_weight 3.0 \
  --hard_clutter_threshold 0.44 \
  --miss_weight 1.2 \
  --miss_threshold 0.14 \
  --uncertainty_weight 0.25 \
  --focus_patch_prob 0.90 \
  --focus_hard_threshold 0.44 \
  --focus_miss_threshold 0.14 \
  --weighted_dice \
  --seed 902 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_focus_strong_s902' 2>&1 | tee '$OUT/nuaa_focus_strong_s902.log'"

tmux new-session -d -s hfmlk_focus_nuaa_c "cd '$ROOT' && CUDA_VISIBLE_DEVICES=2 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --teacher_mode weighted_raw \
  --base_channels 48 \
  --lr 0.00014 \
  --residual_scale 0.22 \
  --anchor_weight 0.070 \
  --pos_weight 32 \
  --hard_clutter_weight 2.6 \
  --hard_clutter_threshold 0.43 \
  --miss_weight 1.4 \
  --miss_threshold 0.15 \
  --uncertainty_weight 0.32 \
  --focus_patch_prob 0.80 \
  --focus_hard_threshold 0.43 \
  --focus_miss_threshold 0.15 \
  --weighted_dice \
  --seed 903 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_focus_weighted_s903' 2>&1 | tee '$OUT/nuaa_focus_weighted_s903.log'"

tmux new-session -d -s hfmlk_focus_nuaa_d "cd '$ROOT' && CUDA_VISIBLE_DEVICES=3 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --teacher_mode weighted_raw \
  --base_channels 56 \
  --lr 0.00010 \
  --residual_scale 0.16 \
  --anchor_weight 0.110 \
  --pos_weight 36 \
  --hard_clutter_weight 3.4 \
  --hard_clutter_threshold 0.46 \
  --miss_weight 1.0 \
  --miss_threshold 0.13 \
  --uncertainty_weight 0.20 \
  --focus_patch_prob 0.95 \
  --focus_hard_threshold 0.46 \
  --focus_miss_threshold 0.13 \
  --weighted_dice \
  --seed 904 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_focus_weighted_s904' 2>&1 | tee '$OUT/nuaa_focus_weighted_s904.log'"

tmux ls | rg 'hfmlk_focus_nuaa' || true

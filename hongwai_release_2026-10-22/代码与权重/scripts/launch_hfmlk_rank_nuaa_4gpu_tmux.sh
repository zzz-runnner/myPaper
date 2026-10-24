#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hfm_lk_anchor_refiner_from_cache.py"
OUT="$ROOT/runs/hfmlk_rank_nuaa_4gpu"

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
)

mkdir -p "$OUT"

tmux kill-session -t hfmlk_rank_nuaa_a 2>/dev/null || true
tmux kill-session -t hfmlk_rank_nuaa_b 2>/dev/null || true
tmux kill-session -t hfmlk_rank_nuaa_c 2>/dev/null || true
tmux kill-session -t hfmlk_rank_nuaa_d 2>/dev/null || true

tmux new-session -d -s hfmlk_rank_nuaa_a "cd '$ROOT' && CUDA_VISIBLE_DEVICES=0 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --teacher_mode strong_fusion \
  --base_channels 48 \
  --lr 0.00016 \
  --residual_scale 0.24 \
  --anchor_weight 0.065 \
  --pos_weight 30 \
  --hard_clutter_weight 2.0 \
  --hard_clutter_threshold 0.42 \
  --miss_weight 1.4 \
  --miss_threshold 0.16 \
  --uncertainty_weight 0.35 \
  --rank_weight 0.035 \
  --rank_margin 0.30 \
  --rank_topk 512 \
  --seed 921 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_rank_strong_s921' 2>&1 | tee '$OUT/nuaa_rank_strong_s921.log'"

tmux new-session -d -s hfmlk_rank_nuaa_b "cd '$ROOT' && CUDA_VISIBLE_DEVICES=1 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --teacher_mode strong_fusion \
  --base_channels 56 \
  --lr 0.00012 \
  --residual_scale 0.18 \
  --anchor_weight 0.090 \
  --pos_weight 32 \
  --hard_clutter_weight 2.8 \
  --hard_clutter_threshold 0.44 \
  --miss_weight 1.0 \
  --miss_threshold 0.14 \
  --uncertainty_weight 0.25 \
  --rank_weight 0.060 \
  --rank_margin 0.35 \
  --rank_topk 512 \
  --seed 922 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_rank_strong_s922' 2>&1 | tee '$OUT/nuaa_rank_strong_s922.log'"

tmux new-session -d -s hfmlk_rank_nuaa_c "cd '$ROOT' && CUDA_VISIBLE_DEVICES=2 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --teacher_mode weighted_raw \
  --base_channels 48 \
  --lr 0.00014 \
  --residual_scale 0.22 \
  --anchor_weight 0.070 \
  --pos_weight 32 \
  --hard_clutter_weight 2.4 \
  --hard_clutter_threshold 0.43 \
  --miss_weight 1.2 \
  --miss_threshold 0.15 \
  --uncertainty_weight 0.30 \
  --rank_weight 0.045 \
  --rank_margin 0.32 \
  --rank_topk 512 \
  --seed 923 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_rank_weighted_s923' 2>&1 | tee '$OUT/nuaa_rank_weighted_s923.log'"

tmux new-session -d -s hfmlk_rank_nuaa_d "cd '$ROOT' && CUDA_VISIBLE_DEVICES=3 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --teacher_mode weighted_raw \
  --base_channels 56 \
  --lr 0.00010 \
  --residual_scale 0.16 \
  --anchor_weight 0.105 \
  --pos_weight 36 \
  --hard_clutter_weight 3.2 \
  --hard_clutter_threshold 0.46 \
  --miss_weight 0.9 \
  --miss_threshold 0.13 \
  --uncertainty_weight 0.20 \
  --rank_weight 0.080 \
  --rank_margin 0.40 \
  --rank_topk 512 \
  --seed 924 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_rank_weighted_s924' 2>&1 | tee '$OUT/nuaa_rank_weighted_s924.log'"

tmux ls | rg 'hfmlk_rank_nuaa' || true

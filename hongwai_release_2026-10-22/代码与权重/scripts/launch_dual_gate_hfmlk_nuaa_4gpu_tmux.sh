#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_dual_gate_hfmlk_refiner_from_cache.py"
OUT="$ROOT/runs/dual_gate_hfmlk_nuaa_4gpu"

TRAIN_ARGS=(
  --train_cache_spec "IRSTD-1K|$ROOT/reports/fusion_cache_bgm_rpcanet_train_a/IRSTD-1K__train.npz"
  --train_cache_spec "NUAA-SIRST|$ROOT/reports/fusion_cache_bgm_rpcanet_train_a/NUAA-SIRST__train.npz"
  --train_cache_spec "NUDT-SIRST|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/NUDT-SIRST__train.npz"
  --train_cache_spec "SIRST3|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/SIRST3__train.npz"
  --test_cache_spec "NUAA-SIRST|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/NUAA-SIRST__test.npz"
  --epochs 5
  --patch_size 512
  --eval_interval 1
  --thresholds "0.44,0.46,0.468,0.48,0.50,0.52,0.54,0.56,0.58,0.60"
  --amp
  --batch_size 2
  --weighted_dice
)

mkdir -p "$OUT"

tmux kill-session -t dg_hfmlk_nuaa_a 2>/dev/null || true
tmux kill-session -t dg_hfmlk_nuaa_b 2>/dev/null || true
tmux kill-session -t dg_hfmlk_nuaa_c 2>/dev/null || true
tmux kill-session -t dg_hfmlk_nuaa_d 2>/dev/null || true

tmux new-session -d -s dg_hfmlk_nuaa_a "cd '$ROOT' && CUDA_VISIBLE_DEVICES=0 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --teacher_mode strong_fusion \
  --base_channels 48 \
  --lr 0.00020 \
  --add_scale 0.20 \
  --suppress_scale 0.16 \
  --anchor_weight 0.030 \
  --gate_weight 0.004 \
  --pos_weight 30 \
  --hard_clutter_weight 0.8 \
  --hard_clutter_threshold 0.40 \
  --miss_weight 3.2 \
  --miss_threshold 0.22 \
  --uncertainty_weight 0.55 \
  --seed 811 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_dual_gate_strong_s811' 2>&1 | tee '$OUT/nuaa_dual_gate_strong_s811.log'"

tmux new-session -d -s dg_hfmlk_nuaa_b "cd '$ROOT' && CUDA_VISIBLE_DEVICES=1 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --teacher_mode strong_fusion \
  --base_channels 48 \
  --lr 0.00016 \
  --add_scale 0.16 \
  --suppress_scale 0.22 \
  --anchor_weight 0.040 \
  --gate_weight 0.006 \
  --pos_weight 32 \
  --hard_clutter_weight 1.4 \
  --hard_clutter_threshold 0.42 \
  --miss_weight 2.6 \
  --miss_threshold 0.20 \
  --uncertainty_weight 0.45 \
  --seed 812 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_dual_gate_strong_s812' 2>&1 | tee '$OUT/nuaa_dual_gate_strong_s812.log'"

tmux new-session -d -s dg_hfmlk_nuaa_c "cd '$ROOT' && CUDA_VISIBLE_DEVICES=2 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --teacher_mode weighted_raw \
  --base_channels 48 \
  --lr 0.00018 \
  --add_scale 0.18 \
  --suppress_scale 0.18 \
  --anchor_weight 0.035 \
  --gate_weight 0.004 \
  --pos_weight 30 \
  --hard_clutter_weight 1.0 \
  --hard_clutter_threshold 0.42 \
  --miss_weight 2.8 \
  --miss_threshold 0.20 \
  --uncertainty_weight 0.50 \
  --seed 813 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_dual_gate_weighted_s813' 2>&1 | tee '$OUT/nuaa_dual_gate_weighted_s813.log'"

tmux new-session -d -s dg_hfmlk_nuaa_d "cd '$ROOT' && CUDA_VISIBLE_DEVICES=3 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --teacher_mode weighted_raw \
  --base_channels 56 \
  --lr 0.00014 \
  --add_scale 0.14 \
  --suppress_scale 0.24 \
  --anchor_weight 0.050 \
  --gate_weight 0.008 \
  --pos_weight 34 \
  --hard_clutter_weight 1.8 \
  --hard_clutter_threshold 0.45 \
  --miss_weight 2.2 \
  --miss_threshold 0.18 \
  --uncertainty_weight 0.35 \
  --seed 814 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_dual_gate_weighted_s814' 2>&1 | tee '$OUT/nuaa_dual_gate_weighted_s814.log'"

tmux ls | rg 'dg_hfmlk_nuaa' || true

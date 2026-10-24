#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hfm_lk_anchor_refiner_from_cache.py"
CKPT="$ROOT/runs/strong_fusion_hfmlk_nuaa_4gpu/nuaa_strong_fusion_s703/best.pth.tar"
OUT="$ROOT/runs/hfmlk_fp_suppress_nuaa_4gpu"

TRAIN_ARGS=(
  --train_cache_spec "IRSTD-1K|$ROOT/reports/fusion_cache_bgm_rpcanet_train_a/IRSTD-1K__train.npz"
  --train_cache_spec "NUAA-SIRST|$ROOT/reports/fusion_cache_bgm_rpcanet_train_a/NUAA-SIRST__train.npz"
  --train_cache_spec "NUDT-SIRST|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/NUDT-SIRST__train.npz"
  --train_cache_spec "SIRST3|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/SIRST3__train.npz"
  --test_cache_spec "NUAA-SIRST|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/NUAA-SIRST__test.npz"
  --teacher_mode strong_fusion
  --resume_checkpoint "$CKPT"
  --epochs 4
  --patch_size 512
  --eval_interval 1
  --thresholds "0.42,0.44,0.456,0.464,0.468,0.472,0.476,0.48,0.488,0.496,0.504,0.512,0.52,0.54,0.56"
  --amp
  --batch_size 2
  --weighted_dice
  --base_channels 48
  --residual_scale 0.32
)

mkdir -p "$OUT"

tmux kill-session -t hfmlk_fp_nuaa_a 2>/dev/null || true
tmux kill-session -t hfmlk_fp_nuaa_b 2>/dev/null || true
tmux kill-session -t hfmlk_fp_nuaa_c 2>/dev/null || true
tmux kill-session -t hfmlk_fp_nuaa_d 2>/dev/null || true

tmux new-session -d -s hfmlk_fp_nuaa_a "cd '$ROOT' && CUDA_VISIBLE_DEVICES=0 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --lr 0.000030 \
  --anchor_weight 0.120 \
  --pos_weight 30 \
  --hard_clutter_weight 2.4 \
  --hard_clutter_threshold 0.42 \
  --miss_weight 1.0 \
  --miss_threshold 0.15 \
  --uncertainty_weight 0.20 \
  --fp_topk_weight 0.018 \
  --fp_topk_ratio 0.0015 \
  --fp_topk_min 128 \
  --pos_peak_weight 0.012 \
  --pos_peak_margin 3.8 \
  --seed 941 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_fp_strong_s941' 2>&1 | tee '$OUT/nuaa_fp_strong_s941.log'"

tmux new-session -d -s hfmlk_fp_nuaa_b "cd '$ROOT' && CUDA_VISIBLE_DEVICES=1 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --lr 0.000020 \
  --anchor_weight 0.150 \
  --pos_weight 32 \
  --hard_clutter_weight 3.2 \
  --hard_clutter_threshold 0.44 \
  --miss_weight 0.8 \
  --miss_threshold 0.14 \
  --uncertainty_weight 0.15 \
  --fp_topk_weight 0.028 \
  --fp_topk_ratio 0.0010 \
  --fp_topk_min 128 \
  --pos_peak_weight 0.016 \
  --pos_peak_margin 4.0 \
  --seed 942 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_fp_strong_s942' 2>&1 | tee '$OUT/nuaa_fp_strong_s942.log'"

tmux new-session -d -s hfmlk_fp_nuaa_c "cd '$ROOT' && CUDA_VISIBLE_DEVICES=2 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --lr 0.000040 \
  --anchor_weight 0.100 \
  --pos_weight 30 \
  --hard_clutter_weight 2.0 \
  --hard_clutter_threshold 0.40 \
  --miss_weight 1.4 \
  --miss_threshold 0.16 \
  --uncertainty_weight 0.25 \
  --fp_topk_weight 0.012 \
  --fp_topk_ratio 0.0020 \
  --fp_topk_min 192 \
  --pos_peak_weight 0.010 \
  --pos_peak_margin 3.6 \
  --rank_weight 0.015 \
  --rank_margin 0.25 \
  --rank_topk 384 \
  --seed 943 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_fp_rank_s943' 2>&1 | tee '$OUT/nuaa_fp_rank_s943.log'"

tmux new-session -d -s hfmlk_fp_nuaa_d "cd '$ROOT' && CUDA_VISIBLE_DEVICES=3 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --lr 0.000025 \
  --anchor_weight 0.135 \
  --pos_weight 34 \
  --hard_clutter_weight 2.8 \
  --hard_clutter_threshold 0.43 \
  --miss_weight 1.0 \
  --miss_threshold 0.15 \
  --uncertainty_weight 0.18 \
  --fp_topk_weight 0.020 \
  --fp_topk_ratio 0.0015 \
  --fp_topk_min 160 \
  --pos_peak_weight 0.014 \
  --pos_peak_margin 3.9 \
  --focus_dataset NUAA-SIRST \
  --focus_patch_prob 0.45 \
  --focus_hard_threshold 0.43 \
  --focus_miss_threshold 0.16 \
  --seed 944 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_fp_focus_s944' 2>&1 | tee '$OUT/nuaa_fp_focus_s944.log'"

tmux ls | rg 'hfmlk_fp_nuaa' || true

#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hfm_lk_anchor_refiner_from_cache.py"
OUT="$ROOT/runs/hfmlk_s703_resume_nuaa_4gpu"
CKPT="$ROOT/runs/strong_fusion_hfmlk_nuaa_4gpu/nuaa_strong_fusion_s703/best.pth.tar"

TRAIN_ARGS=(
  --train_cache_spec "IRSTD-1K|$ROOT/reports/fusion_cache_bgm_rpcanet_train_a/IRSTD-1K__train.npz"
  --train_cache_spec "NUAA-SIRST|$ROOT/reports/fusion_cache_bgm_rpcanet_train_a/NUAA-SIRST__train.npz"
  --train_cache_spec "NUDT-SIRST|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/NUDT-SIRST__train.npz"
  --train_cache_spec "SIRST3|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/SIRST3__train.npz"
  --test_cache_spec "NUAA-SIRST|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/NUAA-SIRST__test.npz"
  --teacher_mode strong_fusion
  --base_channels 48
  --epochs 3
  --patch_size 512
  --eval_interval 1
  --thresholds "0.44,0.46,0.468,0.48,0.50,0.52,0.54,0.56,0.58"
  --amp
  --batch_size 2
  --resume_checkpoint "$CKPT"
)

mkdir -p "$OUT"

tmux kill-session -t hfmlk_resume_a 2>/dev/null || true
tmux kill-session -t hfmlk_resume_b 2>/dev/null || true
tmux kill-session -t hfmlk_resume_c 2>/dev/null || true
tmux kill-session -t hfmlk_resume_d 2>/dev/null || true

tmux new-session -d -s hfmlk_resume_a "cd '$ROOT' && CUDA_VISIBLE_DEVICES=0 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --lr 0.000020 \
  --residual_scale 0.32 \
  --anchor_weight 0.040 \
  --pos_weight 28 \
  --hard_clutter_weight 0.8 \
  --hard_clutter_threshold 0.40 \
  --miss_weight 2.4 \
  --miss_threshold 0.20 \
  --uncertainty_weight 0.45 \
  --weighted_dice \
  --seed 821 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_s703_resume_lr20_s821' 2>&1 | tee '$OUT/nuaa_s703_resume_lr20_s821.log'"

tmux new-session -d -s hfmlk_resume_b "cd '$ROOT' && CUDA_VISIBLE_DEVICES=1 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --lr 0.000010 \
  --residual_scale 0.32 \
  --anchor_weight 0.060 \
  --pos_weight 28 \
  --hard_clutter_weight 1.0 \
  --hard_clutter_threshold 0.42 \
  --miss_weight 2.0 \
  --miss_threshold 0.18 \
  --uncertainty_weight 0.35 \
  --weighted_dice \
  --seed 822 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_s703_resume_lr10_s822' 2>&1 | tee '$OUT/nuaa_s703_resume_lr10_s822.log'"

tmux new-session -d -s hfmlk_resume_c "cd '$ROOT' && CUDA_VISIBLE_DEVICES=2 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --lr 0.000005 \
  --residual_scale 0.30 \
  --anchor_weight 0.080 \
  --pos_weight 30 \
  --hard_clutter_weight 1.2 \
  --hard_clutter_threshold 0.42 \
  --miss_weight 1.8 \
  --miss_threshold 0.18 \
  --uncertainty_weight 0.30 \
  --weighted_dice \
  --seed 823 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_s703_resume_lr05_s823' 2>&1 | tee '$OUT/nuaa_s703_resume_lr05_s823.log'"

tmux new-session -d -s hfmlk_resume_d "cd '$ROOT' && CUDA_VISIBLE_DEVICES=3 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --lr 0.000003 \
  --residual_scale 0.28 \
  --anchor_weight 0.100 \
  --pos_weight 32 \
  --hard_clutter_weight 1.4 \
  --hard_clutter_threshold 0.44 \
  --miss_weight 1.6 \
  --miss_threshold 0.16 \
  --uncertainty_weight 0.25 \
  --weighted_dice \
  --seed 824 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_s703_resume_lr03_s824' 2>&1 | tee '$OUT/nuaa_s703_resume_lr03_s824.log'"

tmux ls | rg 'hfmlk_resume' || true

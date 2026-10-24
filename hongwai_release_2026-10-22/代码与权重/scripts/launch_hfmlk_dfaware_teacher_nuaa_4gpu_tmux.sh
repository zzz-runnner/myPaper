#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hfm_lk_anchor_refiner_from_cache.py"
CKPT="$ROOT/runs/strong_fusion_hfmlk_nuaa_4gpu/nuaa_strong_fusion_s703/best.pth.tar"
OUT="$ROOT/runs/hfmlk_dfaware_teacher_nuaa_4gpu"

TRAIN_ARGS=(
  --train_cache_spec "IRSTD-1K|$ROOT/reports/fusion_cache_bgm_rpcanet_train_a/IRSTD-1K__train.npz"
  --train_cache_spec "NUAA-SIRST|$ROOT/reports/fusion_cache_dfaware_teacher_nuaa/NUAA-SIRST__train.npz"
  --train_cache_spec "NUDT-SIRST|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/NUDT-SIRST__train.npz"
  --train_cache_spec "SIRST3|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/SIRST3__train.npz"
  --test_cache_spec "NUAA-SIRST|$ROOT/reports/fusion_cache_dfaware_teacher_nuaa/NUAA-SIRST__test.npz"
  --teacher_mode "dfaware_fused,strong_fusion"
  --resume_checkpoint "$CKPT"
  --epochs 4
  --patch_size 512
  --eval_interval 1
  --thresholds "0.42,0.44,0.456,0.464,0.468,0.472,0.476,0.48,0.488,0.496,0.504,0.512,0.52,0.54,0.56"
  --amp
  --batch_size 2
  --weighted_dice
  --base_channels 48
  --residual_scale 0.22
)

mkdir -p "$OUT"

for s in hfmlk_dfaware_nuaa_a hfmlk_dfaware_nuaa_b hfmlk_dfaware_nuaa_c hfmlk_dfaware_nuaa_d; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

tmux new-session -d -s hfmlk_dfaware_nuaa_a "cd '$ROOT' && CUDA_VISIBLE_DEVICES=0 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --lr 0.000020 \
  --anchor_weight 0.180 \
  --pos_weight 30 \
  --hard_clutter_weight 0.8 \
  --hard_clutter_threshold 0.48 \
  --miss_weight 0.8 \
  --miss_threshold 0.18 \
  --uncertainty_weight 0.12 \
  --seed 951 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_dfaware_teacher_s951' 2>&1 | tee '$OUT/nuaa_dfaware_teacher_s951.log'"

tmux new-session -d -s hfmlk_dfaware_nuaa_b "cd '$ROOT' && CUDA_VISIBLE_DEVICES=1 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --lr 0.000015 \
  --anchor_weight 0.240 \
  --pos_weight 28 \
  --hard_clutter_weight 0.6 \
  --hard_clutter_threshold 0.50 \
  --miss_weight 0.6 \
  --miss_threshold 0.18 \
  --uncertainty_weight 0.08 \
  --residual_scale 0.18 \
  --seed 952 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_dfaware_teacher_s952' 2>&1 | tee '$OUT/nuaa_dfaware_teacher_s952.log'"

tmux new-session -d -s hfmlk_dfaware_nuaa_c "cd '$ROOT' && CUDA_VISIBLE_DEVICES=2 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --lr 0.000030 \
  --anchor_weight 0.140 \
  --pos_weight 32 \
  --hard_clutter_weight 1.0 \
  --hard_clutter_threshold 0.46 \
  --miss_weight 1.0 \
  --miss_threshold 0.16 \
  --uncertainty_weight 0.15 \
  --fp_topk_weight 0.006 \
  --fp_topk_ratio 0.001 \
  --fp_topk_min 96 \
  --pos_peak_weight 0.006 \
  --pos_peak_margin 3.8 \
  --seed 953 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_dfaware_teacher_fp_s953' 2>&1 | tee '$OUT/nuaa_dfaware_teacher_fp_s953.log'"

tmux new-session -d -s hfmlk_dfaware_nuaa_d "cd '$ROOT' && CUDA_VISIBLE_DEVICES=3 '$PY' '$SCRIPT' \
  ${TRAIN_ARGS[*]@Q} \
  --lr 0.000025 \
  --anchor_weight 0.170 \
  --pos_weight 32 \
  --hard_clutter_weight 0.9 \
  --hard_clutter_threshold 0.47 \
  --miss_weight 1.2 \
  --miss_threshold 0.16 \
  --uncertainty_weight 0.18 \
  --focus_dataset NUAA-SIRST \
  --focus_patch_prob 0.35 \
  --focus_hard_threshold 0.47 \
  --focus_miss_threshold 0.16 \
  --seed 954 \
  --device cuda:0 \
  --output_dir '$OUT/nuaa_dfaware_teacher_focus_s954' 2>&1 | tee '$OUT/nuaa_dfaware_teacher_focus_s954.log'"

tmux ls | rg 'hfmlk_dfaware_nuaa' || true

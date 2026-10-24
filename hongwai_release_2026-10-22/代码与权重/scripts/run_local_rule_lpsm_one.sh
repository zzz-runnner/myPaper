#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 11 ]]; then
  echo "usage: $0 GPU NAME SEED STAGES RESIDUAL_SCALE ANCHOR_WEIGHT LR HARD_CLUTTER_WEIGHT MISS_WEIGHT UNCERTAINTY_WEIGHT OUT_ROOT [extra args...]" >&2
  exit 2
fi

GPU="$1"; shift
NAME="$1"; shift
SEED="$1"; shift
STAGES="$1"; shift
RESIDUAL="$1"; shift
ANCHOR="$1"; shift
LR="$1"; shift
HARD_CLUTTER="$1"; shift
MISS="$1"; shift
UNCERTAINTY="$1"; shift
OUT_ROOT="$1"; shift

ROOT="/home/cx/hongwai/irstd-lab"
RUN_DIR="$OUT_ROOT/$NAME"
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$ROOT/envs/irstd/bin/python" "$ROOT/scripts/train_lpsm_anchor_refiner_from_cache.py" \
  --train_cache_spec "IRSTD-1K|$ROOT/reports/fusion_cache_bgm_rpcanet_train_a/IRSTD-1K__train.npz" \
  --train_cache_spec "NUAA-SIRST|$ROOT/reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__train.npz" \
  --train_cache_spec "NUDT-SIRST|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/NUDT-SIRST__train.npz" \
  --train_cache_spec "SIRST3|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/SIRST3__train.npz" \
  --test_cache_spec "NUAA-SIRST|$ROOT/reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__test.npz" \
  --teacher_mode local_rule \
  --epochs 3 \
  --batch_size 3 \
  --patch_size 512 \
  --base_channels 56 \
  --state_stages "$STAGES" \
  --weighted_dice \
  --amp \
  --seed "$SEED" \
  --residual_scale "$RESIDUAL" \
  --anchor_weight "$ANCHOR" \
  --lr "$LR" \
  --hard_clutter_weight "$HARD_CLUTTER" \
  --miss_weight "$MISS" \
  --uncertainty_weight "$UNCERTAINTY" \
  "$@" \
  --device cuda:0 \
  --output_dir "$RUN_DIR"

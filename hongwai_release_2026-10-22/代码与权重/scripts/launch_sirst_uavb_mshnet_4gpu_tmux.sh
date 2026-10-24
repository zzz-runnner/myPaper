#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/sirst_uavb_mshnet_4gpu"
LOG_ROOT="$RUN_ROOT/logs"
SESSION="sirst_uavb_mshnet_4gpu"

mkdir -p "$LOG_ROOT"
tmux kill-session -t "$SESSION" 2>/dev/null || true

TRAIN_DATASETS="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3,SIRST-UAVB"

common_args=(
  --dataset_dir "$ROOT/datasets"
  --dataset_name "NUAA-SIRST"
  --train_datasets "$TRAIN_DATASETS"
  --target_repeat 5
  --image_preprocess official_mshnet
  --official_base_size 512
  --official_scale_aug
  --official_blur_prob 0.05
  --epochs 120
  --begin_test 1
  --every_test 1
  --every_print 20
  --patch_size 512
  --threads 4
  --optimizer adagrad
  --lr 0.035
  --warmup_epochs 3
  --min_lr_ratio 0.04
  --pos_prob 0.62
  --threshold 0.5
  --bce_weight 1.0
  --dice_weight 0.42
  --focal_weight 0.06
  --small_weight 0.70
  --pos_weight 0.95
  --hard_neg_weight 2.4
  --sls_iou_weight 0.18
  --sls_iou_warm_epochs 5
  --bg_hf_weight 0.12
  --bg_pred_weight 0.04
)

run_cmd() {
  local gpu="$1"; shift
  local name="$1"; shift
  local model="$1"; shift
  local batch="$1"; shift
  local seed="$1"; shift
  rm -rf "$RUN_ROOT/$name"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT" \
    "${common_args[@]}" \
    --run_dir "$RUN_ROOT/$name" \
    --method_name "$name" \
    --model "$model" \
    --batch_size "$batch" \
    --seed "$seed"
}

tmux new-session -d -s "$SESSION" -n msh3_a \
  "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 0 NUAA-UAVB-MSH3-A mshnet3_sls 10 6201 2>&1 | tee '$LOG_ROOT/NUAA-UAVB-MSH3-A_gpu0.log'"

tmux new-window -t "$SESSION" -n p43_a \
  "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 1 NUAA-UAVB-P43-A mshnet_p43_sls 10 6203 2>&1 | tee '$LOG_ROOT/NUAA-UAVB-P43-A_gpu1.log'"

tmux new-window -t "$SESSION" -n msh3_b \
  "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 2 NUAA-UAVB-MSH3-B mshnet3_sls 12 6205 2>&1 | tee '$LOG_ROOT/NUAA-UAVB-MSH3-B_gpu2.log'"

tmux new-window -t "$SESSION" -n p43_b \
  "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 3 NUAA-UAVB-P43-B mshnet_p43_sls 12 6207 2>&1 | tee '$LOG_ROOT/NUAA-UAVB-P43-B_gpu3.log'"

echo "launched tmux session: $SESSION"

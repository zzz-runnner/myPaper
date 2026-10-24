#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
MTML="$ROOT/external/MTMLNet"
PY="$ROOT/envs/irstd/bin/python"
DATA_ROOT="$ROOT/generated/mtmlnet_full"
RUN_ROOT="$ROOT/runs/mtmlnet_full_4gpu"
LOG_DIR="$RUN_ROOT/logs"
SESSION="mtmlnet_full_4gpu"

mkdir -p "$LOG_DIR" "$RUN_ROOT"

"$PY" "$ROOT/scripts/prepare_mtmlnet_dataset.py" \
  --dataset_root "$ROOT/datasets" \
  --out_root "$DATA_ROOT" \
  --datasets "IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3" \
  --target_datasets "IRSTD-1K,NUAA-SIRST" \
  --bbox_mode components \
  --min_area 1 \
  > "$LOG_DIR/prepare_dataset.log" 2>&1

tmux kill-session -t "$SESSION" 2>/dev/null || true

run_one() {
  local gpu="$1"
  local name="$2"
  local imgsz="$3"
  local batch="$4"
  local optimizer="$5"
  local hyp="$6"
  local close_mosaic="$7"
  local seed="$8"
  local extra="$9"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$MTML/train.py" \
    --cfg "$MTML/models/detect/P234-MEA-HPDown-HFF.yaml" \
    --data "$DATA_ROOT/mtmlnet_full.yaml" \
    --hyp "$hyp" \
    --weights "" \
    --project "$RUN_ROOT" \
    --name "$name" \
    --exist-ok \
    --device "" \
    --epochs 160 \
    --batch-size "$batch" \
    --imgsz "$imgsz" \
    --optimizer "$optimizer" \
    --workers 4 \
    --patience 40 \
    --close-mosaic "$close_mosaic" \
    --seed "$seed" \
    --noplots \
    $extra
}

tmux new-session -d -s "$SESSION" -n gpu0 "cd '$MTML' && export PYTHONPATH='$MTML' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -f run_one); PY='$PY'; MTML='$MTML'; DATA_ROOT='$DATA_ROOT'; RUN_ROOT='$RUN_ROOT'; run_one 0 mtm_full_s512_b3_sgd_a 512 3 SGD '$ROOT/configs/mtmlnet_hyp_sgd.yaml' 20 6101 '' 2>&1 | tee '$LOG_DIR/mtm_full_s512_b3_sgd_a_gpu0.log'"

tmux new-window -t "$SESSION" -n gpu1 "cd '$MTML' && export PYTHONPATH='$MTML' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -f run_one); PY='$PY'; MTML='$MTML'; DATA_ROOT='$DATA_ROOT'; RUN_ROOT='$RUN_ROOT'; run_one 1 mtm_full_s640_b2_sgd 640 2 SGD '$ROOT/configs/mtmlnet_hyp_sgd.yaml' 20 6103 '' 2>&1 | tee '$LOG_DIR/mtm_full_s640_b2_sgd_gpu1.log'"

tmux new-window -t "$SESSION" -n gpu2 "cd '$MTML' && export PYTHONPATH='$MTML' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -f run_one); PY='$PY'; MTML='$MTML'; DATA_ROOT='$DATA_ROOT'; RUN_ROOT='$RUN_ROOT'; run_one 2 mtm_full_s512_b3_adamw_a 512 3 AdamW '$ROOT/configs/mtmlnet_hyp_adamw.yaml' 20 6105 '--cos-lr' 2>&1 | tee '$LOG_DIR/mtm_full_s512_b3_adamw_a_gpu2.log'"

tmux new-window -t "$SESSION" -n gpu3 "cd '$MTML' && export PYTHONPATH='$MTML' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -f run_one); PY='$PY'; MTML='$MTML'; DATA_ROOT='$DATA_ROOT'; RUN_ROOT='$RUN_ROOT'; run_one 3 mtm_full_s512_b3_adamw_b 512 3 AdamW '$ROOT/configs/mtmlnet_hyp_adamw.yaml' 20 6107 '--cos-lr' 2>&1 | tee '$LOG_DIR/mtm_full_s512_b3_adamw_b_gpu3.log'"

tmux ls | grep "$SESSION"

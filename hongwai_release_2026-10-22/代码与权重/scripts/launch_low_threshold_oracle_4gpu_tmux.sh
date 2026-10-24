#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/eval_component_fusion_learner.py"
LOG_ROOT="$ROOT/reports/low_threshold_oracle_logs"
RUNNER_ROOT="$ROOT/reports/low_threshold_oracle_runners"
mkdir -p "$LOG_ROOT"
mkdir -p "$RUNNER_ROOT"

COMMON_ARGS=(
  --dataset_dir "$ROOT/datasets"
  --fit_datasets "IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3"
  --seed_thresholds "0.15,0.20,0.25,0.30,0.35,0.40,0.45"
  --keep_thresholds "0.05,0.10,0.15,0.20,0.30,0.40"
  --output_thresholds "0.20,0.25,0.30,0.35,0.40,0.45,0.50"
  --fill_modes "avg,max,binary"
  --max_components_per_image 256
  --steps 1200
)

launch() {
  local session="$1"
  local gpu="$2"
  shift 2
  local runner="$RUNNER_ROOT/${session}.sh"

  tmux kill-session -t "$session" 2>/dev/null || true
  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "$ROOT"
    printf 'CUDA_VISIBLE_DEVICES=%q %q %q' "$gpu" "$PY" "$SCRIPT"
    printf ' %q' "${COMMON_ARGS[@]}"
    printf ' %q' "$@"
    printf ' 2>&1 | tee %q\n' "$LOG_ROOT/${session}.log"
  } > "$runner"
  chmod +x "$runner"
  tmux new-session -d -s "$session" "$runner"
}

launch low_oracle_irstd_mixed 0 \
  --dataset_name IRSTD-1K \
  --model_spec "baseq|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-B_gpu1/checkpoints/IRSTD-1K/best_mIoU.pth.tar" \
  --model_spec "refiner|sct_refiner|$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|$ROOT/runs/sct_refiner_fixed/IRSTD-1K_SCT-Refiner-Fixed_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar" \
  --output "$ROOT/reports/low_threshold_oracle_irstd_mixed.json"

launch low_oracle_irstd_tpc 1 \
  --dataset_name IRSTD-1K \
  --model_spec "tpc_a|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-A_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar" \
  --model_spec "tpc_b|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-B_gpu1/checkpoints/IRSTD-1K/best_mIoU.pth.tar" \
  --output "$ROOT/reports/low_threshold_oracle_irstd_tpc.json"

launch low_oracle_nuaa_tpc_lca 2 \
  --dataset_name NUAA-SIRST \
  --model_spec "tpc_a|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/NUAA-SIRST_SCT-TPC-DHiF-A_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar" \
  --model_spec "tpc_b|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/NUAA-SIRST_SCT-TPC-DHiF-B_gpu3/checkpoints/NUAA-SIRST/best_mIoU.pth.tar" \
  --model_spec "lca|sct_lca_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar" \
  --output "$ROOT/reports/low_threshold_oracle_nuaa_tpc_lca.json"

launch low_oracle_nuaa_lca_tcd 3 \
  --dataset_name NUAA-SIRST \
  --model_spec "lca|sct_lca_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar" \
  --model_spec "tcd_a|sct_tcd_decouple_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/tcd_decouple/NUAA-SIRST_SCT-TCD-Decouple-A_epoch2_best.pth.tar" \
  --model_spec "tcd_b|sct_tcd_decouple_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/tcd_decouple/NUAA-SIRST_SCT-TCD-Decouple-B_epoch2_best.pth.tar" \
  --output "$ROOT/reports/low_threshold_oracle_nuaa_lca_tcd.json"

tmux ls | grep low_oracle

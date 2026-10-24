#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/eval_fast_component_rule_selector.py"
LOG_ROOT="$ROOT/reports/fast_rule_selector_logs"
RUNNER_ROOT="$ROOT/reports/fast_rule_selector_runners"
mkdir -p "$LOG_ROOT" "$RUNNER_ROOT"

COMMON_ARGS=(
  --dataset_dir "$ROOT/datasets"
  --fit_datasets "IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3"
  --seed_thresholds "0.35,0.45"
  --output_thresholds "0.45,0.50"
  --keep_percentiles "45,55,65,75,85"
  --max_components_per_image 160
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

launch fast_rule_irstd_p43 0 \
  --dataset_name IRSTD-1K \
  --model_spec "baseq|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-B_gpu1/checkpoints/IRSTD-1K/best_mIoU.pth.tar" \
  --model_spec "refiner|sct_refiner|$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|$ROOT/runs/sct_refiner_fixed/IRSTD-1K_SCT-Refiner-Fixed_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar" \
  --official_pconv_mshnet_spec "pconv_p43|p43|$ROOT/external/PConv-SDloss-Data/weight_extracted/MSHNet-IRSTD-1K-P43/weight.pkl|official_mshnet|0.60|false|256" \
  --output "$ROOT/reports/fast_rule_selector_irstd_p43.json"

launch fast_rule_irstd_dual 1 \
  --dataset_name IRSTD-1K \
  --model_spec "baseq|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-B_gpu1/checkpoints/IRSTD-1K/best_mIoU.pth.tar" \
  --model_spec "refiner|sct_refiner|$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|$ROOT/runs/sct_refiner_fixed/IRSTD-1K_SCT-Refiner-Fixed_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar" \
  --official_pconv_mshnet_spec "pconv_d05|mshnet|$ROOT/external/PConv-SDloss-Data/weight_extracted/MSHNet-IRSTD-1K-D-0.5/weight.pkl|official_mshnet|0.35|false|256" \
  --official_pconv_mshnet_spec "pconv_p43|p43|$ROOT/external/PConv-SDloss-Data/weight_extracted/MSHNet-IRSTD-1K-P43/weight.pkl|official_mshnet|0.60|false|256" \
  --output "$ROOT/reports/fast_rule_selector_irstd_dual_d05_p43.json"

launch fast_rule_nuaa_tpc_lca_p43 2 \
  --dataset_name NUAA-SIRST \
  --model_spec "tpc_a|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/NUAA-SIRST_SCT-TPC-DHiF-A_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar" \
  --model_spec "tpc_b|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/NUAA-SIRST_SCT-TPC-DHiF-B_gpu3/checkpoints/NUAA-SIRST/best_mIoU.pth.tar" \
  --model_spec "lca|sct_lca_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar" \
  --official_pconv_mshnet_spec "pconv_p43|p43|$ROOT/external/PConv-SDloss-Data/weight_extracted/MSHNet-IRSTD-1K-P43/weight.pkl|official_mshnet|0.60|false|256" \
  --output "$ROOT/reports/fast_rule_selector_nuaa_tpc_lca_p43.json"

launch fast_rule_nuaa_lca_tcd_p43 3 \
  --dataset_name NUAA-SIRST \
  --model_spec "lca|sct_lca_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar" \
  --model_spec "tcd_a|sct_tcd_decouple_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/tcd_decouple/NUAA-SIRST_SCT-TCD-Decouple-A_epoch2_best.pth.tar" \
  --model_spec "tcd_b|sct_tcd_decouple_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/tcd_decouple/NUAA-SIRST_SCT-TCD-Decouple-B_epoch2_best.pth.tar" \
  --official_pconv_mshnet_spec "pconv_p43|p43|$ROOT/external/PConv-SDloss-Data/weight_extracted/MSHNet-IRSTD-1K-P43/weight.pkl|official_mshnet|0.60|false|256" \
  --output "$ROOT/reports/fast_rule_selector_nuaa_lca_tcd_p43.json"

tmux ls | grep fast_rule

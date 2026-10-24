#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cx/hongwai/irstd-lab
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/eval_component_fusion_learner.py"
LOG_DIR="$ROOT/reports/nuaa_handcrafted_oracle_logs"
RUNNER_DIR="$ROOT/reports/nuaa_handcrafted_oracle_runners"
mkdir -p "$LOG_DIR" "$RUNNER_DIR"

FIT_DATASETS="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3,SIRST-UAVB"

BASE_SPECS=(
  --model_spec "tpc_a|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/NUAA-SIRST_SCT-TPC-DHiF-A_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar"
  --model_spec "tpc_b|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/NUAA-SIRST_SCT-TPC-DHiF-B_gpu3/checkpoints/NUAA-SIRST/best_mIoU.pth.tar"
  --model_spec "lca|sct_lca_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar"
)

PCONV_UAV=(
  --official_pconv_mshnet_spec "uav_d05|mshnet|$ROOT/external/PConv-SDloss-Data/weight_extracted/MSHNet-SIRST_UAV-D-0.5/weight.pkl|official_mshnet|0.35|false|256"
  --official_pconv_mshnet_spec "uav_p43|p43|$ROOT/external/PConv-SDloss-Data/weight_extracted/MSHNet-SIRST_UAV-P43/weight.pkl|official_mshnet|0.60|false|256"
)

COMMON=(
  --dataset_dir "$ROOT/datasets"
  --dataset_name NUAA-SIRST
  --fit_datasets "$FIT_DATASETS"
  --fit_split train
  --eval_split test
  --seed_thresholds 0.35,0.40
  --keep_thresholds 0.20,0.40
  --output_thresholds 0.35,0.40,0.45
  --fill_modes max,binary
  --max_fit_samples 180
  --max_eval_samples 0
  --max_components_per_image 80
  --steps 900
  --lr 0.08
  --l2 0.001
)

launch() {
  local session=$1
  local gpu=$2
  local output=$3
  shift 3
  local runner="$RUNNER_DIR/$session.sh"
  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\ncd %q\nexport CUDA_VISIBLE_DEVICES=%q\n' "$ROOT" "$gpu"
    printf '%q ' "$PY" "$SCRIPT" "${COMMON[@]}" "${BASE_SPECS[@]}" "$@" --device cuda:0 --output "$output"
    printf '2>&1 | tee %q\n' "$LOG_DIR/$session.log"
  } > "$runner"
  chmod +x "$runner"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" "$runner"
}

launch nuaa_hand_contrast_gpu0 0 "$ROOT/reports/nuaa_handcrafted_contrast_screen.json" \
  --contrast_candidates --contrast_topk 28 --contrast_threshold 0.72

launch nuaa_hand_sparse_gpu1 1 "$ROOT/reports/nuaa_handcrafted_sparse_screen.json" \
  --handcrafted_candidates --handcrafted_topk 32 --handcrafted_seed_threshold 0.84

launch nuaa_hand_pconv_contrast_gpu2 2 "$ROOT/reports/nuaa_pconv_contrast_screen.json" \
  "${PCONV_UAV[@]}" --contrast_candidates --contrast_topk 28 --contrast_threshold 0.70

launch nuaa_hand_pconv_bg_full_gpu3 3 "$ROOT/reports/nuaa_pconv_bg_handcrafted_full_screen.json" \
  "${PCONV_UAV[@]}" --handcrafted_candidates --handcrafted_topk 32 --handcrafted_seed_threshold 0.84 \
  --background_normality_candidates --background_as_candidate --background_topk 16 --background_threshold 0.86 --background_sample_pixels 1024

tmux ls | grep nuaa_hand || true

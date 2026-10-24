#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"

date -u '+%F %T UTC'
echo
echo "== GPU =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader
echo
echo "== tmux =="
tmux ls 2>/dev/null || true
echo
echo "== latest logs =="
for log in \
  "$ROOT/runs/teacher_apconv_gpu0/logs/IRSTD-1K_APConvTCF-TPCDHiFTeacher_gpu0.log" \
  "$ROOT/runs/teacher_nuaa_gpu1/logs/NUAA-SIRST_APConvTCF-LCATeacher_gpu1.log" \
  "$ROOT/runs/teacher_pconv_irstd_gpu2/logs/IRSTD-1K_PConvTCF-TPCDHiFTeacher_gpu2.log" \
  "$ROOT/runs/pseudo_nuaa_gpu3/logs/NUAA-SIRST_PConvTCF-LCATCDPseudo_gpu3.log"; do
  echo
  echo "--- $log ---"
  if [[ -f "$log" ]]; then
    tail -80 "$log" | tr '\r' '\n' | grep -E '^[A-Za-z]{3} [0-9]{2}|eval epoch=|best epoch=|final |train_datasets=|^[A-Z].*\t' || true
  else
    echo "missing"
  fi
done
echo
echo "== pseudo labels =="
find "$ROOT/generated/pseudo_nuaa_lca_tcd_train" -type f 2>/dev/null | wc -l || true
for dataset in IRSTD-1K NUAA-SIRST NUDT-SIRST SIRST3; do
  printf '%s ' "$dataset"
  find "$ROOT/generated/pseudo_nuaa_lca_tcd_train/$dataset/train" -type f 2>/dev/null | wc -l || true
done

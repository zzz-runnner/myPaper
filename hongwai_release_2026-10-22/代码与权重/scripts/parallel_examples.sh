#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${IRSTD_DATASET_DIR:-/data1/irstd/datasets}"

IRSTD_DATASET_DIR="$DATASET_DIR" GPU=0 "$ROOT/scripts/train_sctransnet.sh" SIRST3 &
IRSTD_DATASET_DIR="$DATASET_DIR" GPU=1 "$ROOT/scripts/train_sctransnet.sh" NUDT-SIRST &
IRSTD_DATASET_DIR="$DATASET_DIR" GPU=2 "$ROOT/scripts/train_sctransnet.sh" IRSTD-1K &

wait

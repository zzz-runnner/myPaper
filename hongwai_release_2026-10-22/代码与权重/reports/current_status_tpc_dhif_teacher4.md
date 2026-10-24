# Current Status: Teacher-Preserved TPC-DHiF

Generated: 2026-06-12 19:16 UTC

## What changed

- Added `sct_tpc_dhif_refiner` in `/home/cx/hongwai/irstd-lab/scripts/train_hybrid_methods.py`.
- Added a hard-clutter correction gate on top of the existing SCTransNet + refiner + DHiF path.
- Added teacher-preservation loss:
  - suppresses rises above the fixed-refiner teacher on teacher-clean background,
  - preserves teacher-positive target pixels,
  - gives extra correction pressure to target pixels where the teacher is weak.
- Added launch script `/home/cx/hongwai/irstd-lab/scripts/launch_tpc_dhif_teacher4_tmux.sh`.

This is not a pure SCTransNet parameter sweep. The current line combines:

- SCTransNet fixed-refiner as the strong teacher,
- DHiF/SET-style high-frequency target-clutter separation,
- BGM/UniVAD-inspired background normality and hard-clutter suppression,
- FeatUp-inspired feature up-projection already present in the refiner branch,
- teacher-preserved correction training to avoid destroying the current best checkpoint.

## Cleanup

Deleted stopped or clearly weak run directories after preserving reports:

- `/home/cx/hongwai/irstd-lab/runs/hybrid_4gpu`
- `/home/cx/hongwai/irstd-lab/runs/mshnet_irstd_gpu1`
- `/home/cx/hongwai/irstd-lab/runs/mshnet_nuaa_gpu3`
- `/home/cx/hongwai/irstd-lab/runs/sct_dhif_multidata`
- `/home/cx/hongwai/irstd-lab/runs/sct_dhif_balanced`
- `/home/cx/hongwai/irstd-lab/runs/sct_dhif_from_fixed`
- `/home/cx/hongwai/irstd-lab/runs/sct_dhif_balanced_irstd_gpu1`
- `/home/cx/hongwai/irstd-lab/runs/sct_dhif_from_fixed_nuaa_gpu3`
- `/home/cx/hongwai/irstd-lab/runs/smoke_tpc_dhif`

Kept required sources:

- `/home/cx/hongwai/irstd-lab/runs/sct_refiner_fixed`
- `/home/cx/hongwai/irstd-lab/runs/selected_checkpoints`
- all reports under `/home/cx/hongwai/irstd-lab/reports`

Current run storage is about 566M.

## Active runs

Sessions:

- `irstd_tpc_dhif_teacher4`
- `nuaa_tpc_thermal_gpu3`

Run roots:

- `/home/cx/hongwai/irstd-lab/runs/tpc_dhif_teacher4`
- `/home/cx/hongwai/irstd-lab/runs/tpc_thermal_nuaa_gpu3`

| GPU | Dataset | Method | Train datasets | Target repeat | Batch | Epochs | Teacher weight |
|---:|---|---|---|---:|---:|---:|---:|
| 0 | IRSTD-1K | SCT-TPC-DHiF-A | IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3 | 4 | 48 | 90 | 0.35 |
| 1 | IRSTD-1K | SCT-TPC-DHiF-B | IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3 | 5 | 48 | 90 | 0.22 |
| 2 | NUAA-SIRST | SCT-TPC-DHiF-A | IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3 | 4 | 48 | 90 | 0.32 |
| 3 | NUAA-SIRST | SCT-TPC-Thermal | IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3 | 6 | 48 | 90 | 0.18 |

GPU3 originally ran `SCT-TPC-DHiF-B`, but it was stopped after epoch 10 because latest eval dropped to 78.77 and did not improve over the inherited fixed-refiner teacher. It was replaced with the in-domain `SCT-TPC-Thermal` line using thermal-spot center/scale shape loss.

GPU state at launch confirmation:

| GPU | Memory | Utilization |
|---:|---:|---:|
| 0 | 10498 / 11264 MiB | 100% |
| 1 | 10498 / 11264 MiB | 100% |
| 2 | 10686 / 11264 MiB | 100% |
| 3 | 10686 / 11264 MiB | 100% |

## Current metrics

Current checked metrics:

| Dataset | Method | Best epoch | Best mIoU | README mIoU | Delta |
|---|---|---:|---:|---:|---:|
| IRSTD-1K | SCT-TPC-DHiF-A | 5 | 68.24 | 68.03 | +0.21 |
| IRSTD-1K | SCT-TPC-DHiF-B | 5 | 68.28 | 68.03 | +0.25 |
| NUAA-SIRST | SCT-TPC-DHiF-A | 0 | 79.12 | 77.50 | +1.62 |
| NUAA-SIRST | SCT-TPC-DHiF-B stopped | 0 | 79.12 | 77.50 | +1.62 |
| NUAA-SIRST | SCT-TPC-Thermal | 0 | 79.12 | 77.50 | +1.62 |

The +5 target is not achieved yet.

Latest observed non-best evaluations:

- `SCT-TPC-DHiF-A` on NUAA-SIRST: epoch 10 mIoU 79.04.
- `SCT-TPC-DHiF-B` on NUAA-SIRST: epoch 10 mIoU 78.77, then stopped.

## Monitoring files

- Results table: `/home/cx/hongwai/irstd-lab/reports/tpc_dhif_teacher4_results.md`
- Thermal line table: `/home/cx/hongwai/irstd-lab/reports/tpc_thermal_nuaa_gpu3_results.md`
- Logs:
  - `/home/cx/hongwai/irstd-lab/runs/tpc_dhif_teacher4/logs/IRSTD-1K_SCT-TPC-DHiF-A_gpu0.log`
  - `/home/cx/hongwai/irstd-lab/runs/tpc_dhif_teacher4/logs/IRSTD-1K_SCT-TPC-DHiF-B_gpu1.log`
  - `/home/cx/hongwai/irstd-lab/runs/tpc_dhif_teacher4/logs/NUAA-SIRST_SCT-TPC-DHiF-A_gpu2.log`
  - `/home/cx/hongwai/irstd-lab/runs/tpc_dhif_teacher4/logs/NUAA-SIRST_SCT-TPC-DHiF-B_gpu3.log`
  - `/home/cx/hongwai/irstd-lab/runs/tpc_thermal_nuaa_gpu3/logs/NUAA-SIRST_SCT-TPC-Thermal_gpu3.log`

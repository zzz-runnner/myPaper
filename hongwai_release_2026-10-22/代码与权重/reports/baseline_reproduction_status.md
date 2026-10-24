# SCTransNet Baseline Reproduction Status

## Dataset Preparation

Prepared from `/home/cx/hongwai/dataset` into `/home/cx/hongwai/irstd-lab/datasets`.

| Dataset | Images | Train | Test | Status |
|---|---:|---:|---:|---|
| NUAA-SIRST / SIRST-v1 | 427 | 213 | 214 | checked |
| IRSTD-1K | 1001 | 800 | 201 | checked |
| NUDT-SIRST | 1327 | 663 | 664 | checked |
| SIRST3 | 2755 | 1676 | 1079 | checked |

Note: SIRST-v1 sample `Misc_111` had an image/mask size mismatch in the raw release. The mask was resized with nearest-neighbor interpolation from `(592, 400)` to `(325, 220)`.

## Official Weight Verification

Official SCTransNet SIRST3 checkpoint:

```text
/home/cx/hongwai/irstd-lab/pretrained/SCTransNet_SIRST3_official.pth.tar
```

Evaluation report:

```text
/home/cx/hongwai/irstd-lab/reports/official_weight_eval.txt
```

| Dataset | README mIoU | Reproduced mIoU | README nIoU | Reproduced nIoU | README Pd | Reproduced Pd | README Fa | Reproduced Fa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SIRST-v1 / NUAA-SIRST | 77.50 | 77.64 | 81.08 | 81.22 | 96.95 | 96.96 | 13.92 | 13.93 |
| NUDT-SIRST | 94.09 | 94.11 | 94.38 | 94.40 | 98.62 | 98.62 | 4.29 | 4.30 |
| IRSTD-1K | 68.03 | 68.03 | 68.15 | 68.11 | 93.27 | 93.27 | 10.74 | 10.74 |

The official checkpoint matches the README within rounding noise, so the converted datasets and evaluation path are aligned with the authors' code.

## From-Scratch Runs

Four from-scratch SCTransNet runs were launched in tmux session `irstd_baseline4` and have completed.

| GPU | Dataset | Run Dir |
|---:|---|---|
| 0 | SIRST3 | `/home/cx/hongwai/irstd-lab/runs/baseline_4gpu/SIRST3` |
| 1 | NUDT-SIRST | `/home/cx/hongwai/irstd-lab/runs/baseline_4gpu/NUDT-SIRST` |
| 2 | IRSTD-1K | `/home/cx/hongwai/irstd-lab/runs/baseline_4gpu/IRSTD-1K` |
| 3 | NUAA-SIRST | `/home/cx/hongwai/irstd-lab/runs/baseline_4gpu/NUAA-SIRST` |

Final from-scratch report:

```text
/home/cx/hongwai/irstd-lab/reports/from_scratch_baseline_results.md
```

| Dataset | Best epoch | mIoU | README mIoU | Delta | Pd | README Pd | Fa | README Fa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SIRST3 | 678 | 82.96 | - | - | 96.94 | - | 7.83 | - |
| NUDT-SIRST | 547 | 93.76 | 94.09 | -0.33 | 98.94 | 98.62 | 4.37 | 4.29 |
| IRSTD-1K | 500 | 67.32 | 68.03 | -0.71 | 93.60 | 93.27 | 13.61 | 10.74 |
| SIRST-v1 / NUAA-SIRST | 823 | 77.52 | 77.50 | +0.02 | 96.20 | 96.95 | 18.04 | 13.92 |

SIRST3 is reported as the source training set in the authors' cross-dataset table rather than as a single README target. Its best checkpoint was evaluated on SIRST-v1, NUDT-SIRST, and IRSTD-1K in the final report.

# From-Scratch SCTransNet Baseline Results

Generated: 2026-06-10 00:02:52 UTC

## Training Log Summary

```text
/home/cx/hongwai/irstd-lab/runs/baseline_4gpu/logs/IRSTD-1K_gpu2.log
  latest_epoch=1000 latest_loss=0.003003
  best epoch=500 mIoU=67.32 Pd=93.60 Fa=13.61
/home/cx/hongwai/irstd-lab/runs/baseline_4gpu/logs/NUAA-SIRST_gpu3.log
  latest_epoch=1000 latest_loss=0.002671
  best epoch=823 mIoU=77.52 Pd=96.20 Fa=18.04
/home/cx/hongwai/irstd-lab/runs/baseline_4gpu/logs/NUDT-SIRST_gpu1.log
  latest_epoch=1000 latest_loss=0.00263
  best epoch=547 mIoU=93.76 Pd=98.94 Fa=4.37
/home/cx/hongwai/irstd-lab/runs/baseline_4gpu/logs/SIRST3_gpu0.log
  latest_epoch=1000 latest_loss=0.002862
  best epoch=678 mIoU=82.96 Pd=96.94 Fa=7.83
```

## Current Best vs README

| Dataset | Latest epoch | Latest loss | Best epoch | Best mIoU | README mIoU | Delta | Best Pd | README Pd | Best Fa | README Fa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SIRST3 | 1000 | 0.002862 | 678 | 82.96 | - | - | 96.94 | - | 7.83 | - |
| NUDT-SIRST | 1000 | 0.002630 | 547 | 93.76 | 94.09 | -0.33 | 98.94 | 98.62 | 4.37 | 4.29 |
| IRSTD-1K | 1000 | 0.003003 | 500 | 67.32 | 68.03 | -0.71 | 93.60 | 93.27 | 13.61 | 10.74 |
| SIRST-v1 / NUAA-SIRST | 1000 | 0.002671 | 823 | 77.52 | 77.50 | +0.02 | 96.20 | 96.95 | 18.04 | 13.92 |

SIRST3 has no single README target in the authors' table; after that run finishes, its best checkpoint is evaluated on SIRST-v1, NUDT-SIRST, and IRSTD-1K for the paper-style comparison.

## Best Checkpoints

| Dataset | Best checkpoint |
|---|---|
| SIRST3 | /home/cx/hongwai/irstd-lab/runs/baseline_4gpu/SIRST3/checkpoints/SIRST3/SCTransNet_678_best.pth.tar |
| NUDT-SIRST | /home/cx/hongwai/irstd-lab/runs/baseline_4gpu/NUDT-SIRST/checkpoints/NUDT-SIRST/SCTransNet_547_best.pth.tar |
| IRSTD-1K | /home/cx/hongwai/irstd-lab/runs/baseline_4gpu/IRSTD-1K/checkpoints/IRSTD-1K/SCTransNet_500_best.pth.tar |
| NUAA-SIRST | /home/cx/hongwai/irstd-lab/runs/baseline_4gpu/NUAA-SIRST/checkpoints/NUAA-SIRST/SCTransNet_823_best.pth.tar |

## SIRST3 Best Checkpoint Cross-Dataset Evaluation

```text
Deep-Supervision: True
NUAA-SIRST: mIoU=77.14, nIoU=79.19, Pd=96.58, Fa=15.78, pixAcc=85.06
Deep-Supervision: True
NUDT-SIRST: mIoU=93.72, nIoU=93.80, Pd=98.20, Fa=3.01, pixAcc=96.49
Deep-Supervision: True
IRSTD-1K: mIoU=67.10, nIoU=66.74, Pd=93.27, Fa=9.60, pixAcc=75.58
```

## Single-Dataset Best Checkpoint Evaluation

```text
Deep-Supervision: True
NUDT-SIRST: mIoU=93.76, nIoU=94.34, Pd=98.94, Fa=4.37, pixAcc=96.20
Deep-Supervision: True
IRSTD-1K: mIoU=67.32, nIoU=66.66, Pd=93.60, Fa=13.61, pixAcc=78.11
Deep-Supervision: True
NUAA-SIRST: mIoU=77.52, nIoU=78.79, Pd=96.20, Fa=18.04, pixAcc=86.35
```

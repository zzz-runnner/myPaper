# LR-Net External Candidate Attempt

Generated: 2026-06-14 UTC

## Purpose

NUAA-SIRST remains below the requested +5 mIoU target. This run tested a new lightweight external IRSTD architecture from the pasted method family:

```text
LR-Net: low-level lightweight robust IRSTD network
```

The motivation was to test whether LR-Net's low-level shallow evidence retention and lightweight edge/attention structure could produce a dense candidate different from the SCTransNet/TPC/LCA/PConv family.

## Setup

Cloned:

```text
external/LR-Net
```

Added local full-pool wrapper:

```text
scripts/train_lrnet_multidata.py
scripts/launch_lrnet_multidata_4gpu_tmux.sh
```

Training pool:

```text
IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3, SIRST-UAVB
```

Evaluation:

```text
IRSTD-1K, NUAA-SIRST
```

Input:

```text
512 patch, RGB-repeat grayscale input
```

## 4GPU Run

| GPU | Branch | Batch | Norm dataset | Peak memory |
| ---: | --- | ---: | --- | ---: |
| 0 | lrnet_irstd_a | 24 | IRSTD-1K | about 6.0G |
| 1 | lrnet_irstd_b | 32 | IRSTD-1K | about 8.0G |
| 2 | lrnet_nuaa_a | 24 | NUAA-SIRST | about 6.0G |
| 3 | lrnet_nuaa_b | 32 | NUAA-SIRST | about 8.0G |

LR-Net is extremely small, so even batch 32 with 512 patches did not reach 9G. The run was still fully using all four GPUs at about 99-100% utilization.

## Early Results

The run was stopped after epoch 6 because the signal was far below the existing candidate pool.

Best observed branch:

| Branch | Epoch | IRSTD-1K mIoU | NUAA-SIRST mIoU | Notes |
| --- | ---: | ---: | ---: | --- |
| lrnet_nuaa_a | 6 | 35.82 | 36.17 | best early signal, still far below current best |

Other epoch-6 signals:

| Branch | IRSTD-1K mIoU | NUAA-SIRST mIoU |
| --- | ---: | ---: |
| lrnet_irstd_a | 17.60 | 23.50 |
| lrnet_irstd_b | 0.00 | 0.00 |
| lrnet_nuaa_b | 11.18 | 15.05 |

Reference:

```text
Current reliable NUAA best: about 79.34 mIoU
NUAA target: 82.64 mIoU
IRSTD-1K current best: 73.0367 mIoU
```

## Decision

LR-Net is runnable and very cheap, but it is not a useful dense candidate in this full multi-dataset setting. Continuing this exact from-scratch LR-Net run is unlikely to close the gap to NUAA 82.64.

Decision:

- Stop the 4GPU run.
- Delete weak checkpoints.
- Keep small logs and scripts only.

## Cleanup

Deleted:

```text
runs/lrnet_multidata_4gpu/*/best.pth.tar
runs/lrnet_smoke/best.pth.tar
```

Retained size:

```text
runs/lrnet_multidata_4gpu: 56K
runs/lrnet_smoke: 12K
external/LR-Net: 4.6M
```

No active training process remains; GPUs are idle.

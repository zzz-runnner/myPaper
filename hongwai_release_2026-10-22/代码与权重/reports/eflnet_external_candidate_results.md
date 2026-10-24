# EFLNet Official Detection Weight Candidate Results

Generated: 2026-06-14 UTC

## Purpose

This run tested EFLNet as a new external-method candidate source, instead of further tuning SCTransNet training parameters.

EFLNet is a detection-box method, so the local test converts predicted boxes into several mask-like candidates and evaluates them with the same pixel-level IRSTD metrics.

## Setup

Repository:

```text
external/EFLNet
```

Official weights retained:

```text
external/EFLNet/runs/train/NUAA.pt
external/EFLNet/runs/train/IRSTD.pt
external/EFLNet/runs/train/NUDT.pt
```

Dependencies installed in the existing environment:

```text
mmengine 0.10.7
mmcv 2.2.0
```

The upstream `attempt_load()` path was not used because its `attempt_download()` lowercases `NUAA.pt` to `nuaa.pt`, which breaks on Linux. The local evaluator directly loads the existing checkpoint and calls `fuse().eval()`.

Added:

```text
scripts/eval_eflnet_box_candidate.py
```

Outputs:

```text
reports/eflnet_nuaa_smoke20.json
reports/eflnet_nuaa_full_box_candidate.json
reports/eflnet_irstd_full_box_candidate.json
```

No image dumps or large prediction caches were written.

## Box-to-Mask Modes

Tested:

```text
box
center0.50
center0.33
ellipse0.60
ellipse0.40
dot1
dot2
dot3
```

These are intentionally simple conversions:

1. full predicted box;
2. shrunk center rectangle;
3. shrunk center ellipse;
4. small center disk.

The purpose was to see whether EFLNet could provide a useful complementary target-location source for segmentation.

## NUAA-SIRST Full Test

Weight:

```text
external/EFLNet/runs/train/NUAA.pt
```

Detection stats:

```text
images = 214
images_with_det = 214
detections = 1400
detections_per_image = 6.54
```

Best rows:

| mode | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: |
| center0.50 | 0.50 | 50.0311 | 51.6857 | 95.0570 | 27.7833e-6 |
| box | 0.60 | 49.6983 | 48.6296 | 88.9734 | 61.3292e-6 |
| ellipse0.60 | 0.50 | 47.4114 | 46.5114 | 94.2966 | 31.4192e-6 |
| dot3 | 0.50 | 41.9963 | 44.4828 | 93.9163 | 23.4615e-6 |

Reference:

```text
current NUAA valid best = 79.5487
target = 82.64
```

EFLNet box-to-mask is therefore not competitive for NUAA segmentation.

## IRSTD-1K Full Test

Weight:

```text
external/EFLNet/runs/train/IRSTD.pt
```

Detection stats:

```text
images = 201
images_with_det = 201
detections = 1945
detections_per_image = 9.68
```

Best rows:

| mode | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: |
| box | 0.60 | 47.2063 | 45.1873 | 86.5320 | 35.4710e-6 |
| ellipse0.60 | 0.40 | 45.6267 | 44.9689 | 93.9394 | 27.6139e-6 |
| center0.50 | 0.40 | 45.6113 | 48.2347 | 93.6027 | 29.2460e-6 |
| dot3 | 0.40 | 30.4246 | 39.2266 | 94.6128 | 25.1466e-6 |

Reference:

```text
current IRSTD-1K valid best = 73.0367
target = 73.03
```

EFLNet box-to-mask is also not competitive for IRSTD-1K segmentation.

## Interpretation

EFLNet can detect many targets and has high Pd after box-to-mask conversion, but the predicted boxes are too coarse for the pixel-level segmentation mIoU objective. Shrinking boxes or using center disks improves false alarms in some cases but loses too much overlap.

Decision:

```text
Do not add EFLNet box-to-mask maps to the current segmentation candidate pool.
Do not use EFLNet as the next main training line for the +5 mIoU segmentation goal.
```

## Cleanup

Deleted:

```text
external/EFLNet/.git
external/EFLNet/bounding box version
```

Space:

```text
before cleanup: 527 MB
after cleanup: 249 MB
freed: about 279 MB
```

Kept the official weights and runnable code in case a later detection-assisted idea needs them, but this candidate is currently marked as failed for direct segmentation improvement.

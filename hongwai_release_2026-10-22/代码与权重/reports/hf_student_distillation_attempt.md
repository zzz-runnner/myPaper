# High-Frequency Student Distillation Attempt

Generated: 2026-06-14 UTC

## Purpose

This tested a new dense candidate source rather than another SCTransNet hyperparameter tweak. The idea was to distill the strongest cached teachers into a lightweight image-conditioned student.

Unlike the previous pixel MLP calibrator, this student sees the original infrared image through high-frequency / local-anomaly channels:

- robust-normalized infrared image
- local z-score
- top-hat response
- DoG response
- LoG response

It is supervised by both GT masks and cached teacher soft pseudo maps.

## New Script

Implemented:

```text
scripts/train_hf_student_from_cache.py
```

It reads existing prediction caches and original images, trains a small U-Net-like student, evaluates on cached test splits, and saves only one best checkpoint during the run.

## IRSTD Smoke

Teacher cache:

```text
baseq, refiner, pconv_d05, pconv_p43
```

Train pool:

```text
IRSTD-1K train
NUAA-SIRST train
NUDT-SIRST train
SIRST3 train
```

Limited smoke:

- 50 samples per train cache
- first 40 IRSTD test images
- 2 epochs
- base channels 16

Result:

| Epoch | IRSTD-1K mIoU | Pd | Fa |
|---:|---:|---:|---:|
| 1 | 0.15 | 7.69 | 7.09e-3 |
| 2 | 20.99 | 44.62 | 3.60e-5 |

## NUAA Limited Probe

Teacher cache:

```text
TPC-A, TPC-B, LCA
```

Train pool:

```text
IRSTD-1K train
NUAA-SIRST train
NUDT-SIRST train
SIRST3 train
```

Limited probe:

- 100 samples per train cache
- first 80 NUAA test images
- 12 epochs
- base channels 24
- pseudo weight 1.0

Observed results:

| Epoch | NUAA-SIRST mIoU | Pd | Fa |
|---:|---:|---:|---:|
| 2 | 26.11 | 61.47 | 1.50e-4 |
| 4 | 30.44 | 68.81 | 4.48e-4 |
| 6 | 22.23 | 66.97 | 7.10e-4 |
| 8 | 36.01 | 64.22 | 1.85e-4 |
| 10 | 35.99 | 67.89 | 2.05e-4 |
| 12 | 39.43 | 68.81 | 1.62e-4 |

## Cleanup

Weak checkpoints were deleted:

```text
runs/hf_student_smoke_irstd/best.pth.tar
runs/hf_student_smoke_nuaa/best.pth.tar
runs/hf_student_limited_nuaa/best.pth.tar
```

Only logs and summaries remain.

## Interpretation

The student trains and the implementation works, but it does not approach the teacher/fusion region. The best limited NUAA result is only 39.43 mIoU on the 80-image subset, far below the current reliable NUAA result of 79.34.

This suggests that a very small image-conditioned student cannot recover the teacher behavior from limited cached supervision. To make this route viable, the student would need either:

- a stronger backbone initialized from a good IRSTD checkpoint;
- full pseudo-label generation and longer training;
- or direct feature distillation from teacher internals, not only output maps.

Decision: do not expand this exact lightweight student to a full four-GPU run.

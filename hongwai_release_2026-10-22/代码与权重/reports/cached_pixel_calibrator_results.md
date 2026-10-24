# Cached Pixel Calibrator Results

Generated: 2026-06-13 UTC

## Purpose

This round tested a lightweight pixel-level calibrator trained from cached prediction maps. It is not SCTransNet parameter tuning: the neural teacher/PConv predictions are frozen, and the calibrator only learns a small MLP to combine candidate probability maps.

The motivation was to test whether a non-linear pixel fusion could outperform global averaging and hand component rules.

## Added Code

```text
scripts/train_cached_pixel_calibrator.py
```

The script samples positive target pixels and hard negative background pixels from the full multi-dataset train caches, trains a small MLP, then evaluates calibrated probability maps on the target test cache.

## IRSTD-1K Run

Output:

```text
reports/cached_pixel_calibrator_irstd_dual.json
```

Train caches:

```text
IRSTD-1K train
NUAA-SIRST train
NUDT-SIRST train
SIRST3 train
```

Candidate maps:

```text
baseq + fixed refiner + PConv-D05 + PConv-P43
```

Sampled pixels:

```text
7,056,323 total
194,000 positives
```

Best result:

| Pos scale | Threshold | mIoU | Pd | Fa |
|---:|---:|---:|---:|---:|
| 0.25 | 0.70 | 56.44 | 93.94 | 38.47 |

This is far below the current IRSTD best of 72.43.

## NUAA-SIRST Run

Output:

```text
reports/cached_pixel_calibrator_nuaa_tpc_lca.json
```

Train caches:

```text
IRSTD-1K train
NUAA-SIRST train
NUDT-SIRST train
SIRST3 train
```

Candidate maps:

```text
TPC-A + TPC-B + LCA
```

Sampled pixels:

```text
7,054,456 total
194,000 positives
```

Best result:

| Pos scale | Threshold | mIoU | Pd | Fa |
|---:|---:|---:|---:|---:|
| 0.25 | 0.70 | 74.23 | 97.34 | 35.12 |

This is far below the current NUAA best of 79.34.

## Interpretation

The pixel-level MLP calibrator failed on both datasets. It likely overfits sampled hard negatives/positives and loses the spatial/component-level structure that makes the original probability maps useful. It is also highly sensitive to class imbalance.

Do not continue this exact pixel-calibration route.

Current best remains:

| Dataset | Valid best | mIoU |
|---|---|---:|
| IRSTD-1K | dual-PConv average fusion region | 72.43 |
| NUAA-SIRST | background-normality component fusion | 79.34 |

No large model checkpoint was written; only small JSON reports were kept.

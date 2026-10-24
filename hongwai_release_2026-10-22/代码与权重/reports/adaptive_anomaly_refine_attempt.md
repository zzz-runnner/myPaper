# Adaptive Anomaly Refine Attempt

Generated: 2026-06-13 UTC

This was a lightweight external-domain transfer attempt inspired by anomaly detection / image-enhancement priors. It is not SCTransNet training-parameter tuning.

Implemented in:

```text
scripts/adaptive_anomaly_refine_cache.py
```

The script uses cached neural predictions plus original infrared images. It extracts local background and anomaly features per connected component:

- local z-score
- DoG response
- top-hat response
- ring contrast
- model agreement / uncertainty
- component area, side length, fill, and base probability

It selects suppression rules on a validation split from the full multi-dataset train cache:

```text
IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3
```

and evaluates the selected rule on the target test cache.

## NUAA-SIRST Smoke Result

The first run without a no-op baseline selected an over-aggressive suppression rule:

| Setting | Test subset | mIoU | Pd | Fa |
|---|---:|---:|---:|---:|
| anomaly suppression selected by val nIoU | first 80 NUAA test images | 70.02 | 93.58 | 13.77 |

The same 80-image subset baseline was:

| Baseline | Threshold | mIoU | Pd | Fa |
|---|---:|---:|---:|---:|
| LCA / avg / TPC teacher region | 0.50 | 78.43 | 95.41 | 14.90 |

So the anomaly rule was not useful; it damaged recall and mIoU.

After adding an explicit no-op candidate so validation can choose to keep all components, the selected result was:

| Selection | Test subset | mIoU | Pd | Fa |
|---|---:|---:|---:|---:|
| no-op baseline selected | first 80 NUAA test images | 78.43 | 95.41 | 14.90 |

## Interpretation

This route did not produce a positive NUAA signal. The local anomaly/image-enhancement features are too weak as a post-hoc suppression rule over the current teacher components. The result supports the earlier finding that NUAA needs a genuinely new dense candidate source, not more component suppression around the existing SCTransNet-family predictions.

No bulky artifacts were created. Only the small script and JSON smoke evidence remain.

# Cached PConv Component Gate Attempt

Generated: 2026-06-14 UTC

## Purpose

This tested whether the current best IRSTD-1K dual-PConv average fusion could be improved by a stricter component-level gate trained on full multi-dataset cached predictions.

This is a non-SCTransNet-parameter route. It uses cached predictions from:

```text
baseq, refiner, pconv_d05, pconv_p43
```

Fit data:

```text
IRSTD-1K train
NUAA-SIRST train
NUDT-SIRST train
SIRST3 train
```

Evaluation:

```text
IRSTD-1K test
```

## New Script

Implemented:

```text
scripts/eval_cached_pconv_component_gate.py
```

The script reads the existing compact `.npz` prediction caches, builds connected components from the averaged teacher/PConv map, extracts probability/geometric/model-disagreement features, trains a logistic keeper on the full multi-dataset train pool, and evaluates on IRSTD-1K test. It writes only a small JSON report and no checkpoint.

Output:

```text
reports/cached_pconv_component_gate_irstd.json
```

## Result

Best valid row:

| Kind | Threshold | mIoU | nIoU | Pd | Fa |
|---|---:|---:|---:|---:|---:|
| baseline avg | 0.42 | 72.48 | 66.61 | 94.95 | 5.54e-6 |

Best learned gate:

| Seed | Keep | Output | mIoU | nIoU | Pd | Fa |
|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | 0.20 | 0.30 | 67.10 | 65.70 | 90.57 | 2.43e-6 |

Best test oracle for this componentization:

| Seed | Output | mIoU | nIoU | Pd | Fa |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 0.38 | 68.46 | 67.99 | 94.61 | 3.80e-7 |

## Interpretation

This componentization is the wrong abstraction for the dual-PConv IRSTD gain. The current best comes from pixel-level probability averaging and thresholding. Once the average map is binarized into connected components and re-rendered, even the test oracle drops to 68.46 mIoU, far below the 72.43-72.48 average-fusion region.

Therefore the remaining IRSTD gap should not be attacked by "keep/delete avg components". The next IRSTD attempt should preserve the dense probability map and only calibrate local disagreement regions, especially where PConv-P43 contributes probability mass that base/refiner do not.

No large artifacts were produced.

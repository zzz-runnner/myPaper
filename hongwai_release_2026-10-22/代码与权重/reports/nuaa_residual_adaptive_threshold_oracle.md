# NUAA residual adaptive-threshold oracle diagnostic

Date: 2026-06-16

## Purpose

After the external-Mamba residual and mask-selection attempts failed to exceed the retained NUAA result, this diagnostic checked whether the current best residual probability map has enough hidden headroom under per-image threshold/min-area adaptation.

This is an oracle diagnostic, not a valid final method, because it chooses the best threshold/area per test image using ground truth.

## Script

Added:

```text
scripts/diagnose_nuaa_residual_adaptive_threshold.py
```

The script retrains the current tiny residual pixel calibrator in memory, predicts NUAA-SIRST test maps, and evaluates:

- fixed global threshold/min-area;
- per-image test oracle over a narrow threshold/area grid.

No checkpoint or probability map archive is saved.

## Valid baseline

The reproduced fixed setting matches the retained current best:

| threshold | min_area | mIoU | nIoU | Pd | Fa |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.498 | 3 | 82.2928 | 82.1643 | 96.9582 | 8.9867e-6 |

## Test Oracle

Fast oracle grid:

```text
thresholds = 0.492, 0.496, 0.498, 0.500, 0.502
min_area   = 2, 3, 4
```

Best per-image oracle:

| row | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: |
| per-image threshold/min-area oracle | 82.4280 | 82.2653 | 96.9582 | 9.2611e-6 |

Comparison:

```text
valid current best: 82.2928
test oracle:        82.4280
oracle gain:        +0.1352
NUAA target:        82.6400
oracle gap:         0.2120
```

## Stopped Slow Diagnostic

A broader train/test adaptive-rule run was started first, but it repeatedly evaluated connected components across the full train split and became CPU-bound. It was stopped before writing a result file. That stopped run is not counted as an experiment result.

The fast oracle above is sufficient for the main question: even per-image test-set threshold/area selection on the current residual map cannot reach the requested `82.64`.

## Interpretation

This rules out another local threshold/min-area tuning pass on the current residual probability map. The dense map itself does not contain enough recoverable headroom for the target:

```text
even test oracle remains below target by 0.2120 mIoU
```

The next useful NUAA work must change the dense prediction source itself, not just select thresholds, min-area values, or choose between the current residual masks.

## Outputs

```text
reports/nuaa_residual_adaptive_threshold_fast_oracle.json
reports/nuaa_residual_adaptive_threshold_oracle.md
```

Only compact files were saved.

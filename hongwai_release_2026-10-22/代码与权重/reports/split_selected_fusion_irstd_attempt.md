# Split-Selected IRSTD Dual-PConv Fusion Attempt

Generated: 2026-06-14 UTC

## Purpose

This revisited the strongest current IRSTD-1K line:

```text
baseq + fixed refiner + PConv-D05 + PConv-P43
```

The previous train-selected threshold using the full multi-dataset train pool selected threshold 0.38 and gave 72.31 mIoU on IRSTD-1K test. The test-sweep diagnostic could reach about 72.49. This run checked whether selecting weights/threshold on an IRSTD train internal validation split could legally select closer to the stronger 0.40-0.42 region.

## New Script

Implemented:

```text
scripts/search_cache_split_selected_fusion.py
```

It reads prediction caches, splits the train cache by stable hash, selects simple weighted fusion + threshold using validation nIoU, then evaluates once on the full test cache.

## Data

Validation source:

```text
reports/prediction_cache_irstd_dual_train_a/IRSTD-1K__train.npz
```

Test source:

```text
reports/prediction_cache_irstd_dual_train_b_test/IRSTD-1K__test.npz
```

Model maps:

```text
baseq, refiner, pconv_d05, pconv_p43
```

## Result

Validation-selected row:

| Weights | Threshold | Val nIoU | Test mIoU | Test nIoU | Pd | Fa |
|---|---:|---:|---:|---:|---:|---:|
| baseq 1.0, refiner 1.0, pconv_d05 0.9, pconv_p43 0.9 | 0.40 | 70.93 | 72.47 | 66.72 | 94.95 | 7.40e-6 |

Best test row inside validation top-20, diagnostic only:

| Weights | Threshold | Val nIoU | Test mIoU | Test nIoU | Pd | Fa |
|---|---:|---:|---:|---:|---:|---:|
| baseq 1.0, refiner 1.0, pconv_d05 0.9, pconv_p43 0.9 | 0.42 | 70.90 | 72.58 | 66.83 | 94.95 | 6.64e-6 |

## Interpretation

The IRSTD train-split selected result improves over the previous full-multi-dataset threshold-selected 72.31 and slightly improves over the conservative 72.43 region:

| Reference | IRSTD-1K mIoU |
|---|---:|
| SCTransNet official/reproduced | 68.03 |
| Previous strict full-train threshold selected | 72.31 |
| Previous conservative best region | 72.43 |
| This IRSTD split-selected row | 72.47 |
| Requested +5 target | 73.03 |

Remaining gap:

```text
73.03 - 72.47 = 0.56 mIoU
```

The diagnostic 72.58 row is close but was not the validation-selected top row, so it should not be promoted as the final strict number.

## Decision

This is a small but legitimate improvement on IRSTD under an internal train-split selection protocol. It still does not reach the requested +5 target. Further gains from frozen-map fusion appear very limited; the remaining gap likely needs a genuinely new candidate source or official external-method weights.

No checkpoint or large artifact was created.

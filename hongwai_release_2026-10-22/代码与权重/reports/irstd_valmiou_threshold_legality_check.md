# IRSTD Validation-mIoU Threshold Legality Check

Generated: 2026-06-14 UTC

## Purpose

The cached dual-PConv fusion had a diagnostic test row around 72.58-72.62 mIoU at thresholds 0.42-0.44, but previous strict reporting selected thresholds by train-split validation nIoU and stayed around 72.47-72.48.

This check asked whether the higher threshold row can be selected legally by validation mIoU instead of test-set tuning.

## Code Changes

Updated:

```text
scripts/search_cache_split_selected_fusion.py
```

The `--select_metric mIoU` option now actually rescans the validation top-k rows with full validation metrics before choosing a row. A separate experimental fast script was also added:

```text
scripts/search_cache_split_selected_fusion_fast.py
```

The fast all-weight grid was stopped because it was CPU-expensive and did not write result files. No large artifacts were created.

## Direct Threshold Check

Fixed weights:

```text
baseq 1.0, refiner 1.0, pconv_d05 0.9, pconv_p43 0.9
```

Output JSON:

```text
reports/irstd_threshold_valmiou_direct_check.json
```

Validation-mIoU selected thresholds:

| Split seed | Selected threshold by val mIoU | Val mIoU | Test mIoU |
|---:|---:|---:|---:|
| 20260614 | 0.40 | 74.92 | 72.47 |
| 20260615 | 0.38 | 75.32 | 72.48 |
| 20260616 | 0.38 | 75.17 | 72.48 |
| 20260617 | 0.38 | 74.15 | 72.48 |
| 20260618 | 0.38 | 73.90 | 72.48 |

Diagnostic rows:

| Threshold | Typical test mIoU | Validation behavior |
|---:|---:|---|
| 0.42 | 72.58 | not selected by val mIoU in any checked split |
| 0.44 | 72.62 | not selected by val mIoU in any checked split |

## Interpretation

The higher test rows at thresholds 0.42-0.44 remain diagnostic only. They are not supported by validation mIoU selection across the checked splits.

Strict current IRSTD-1K result remains about 72.47-72.48 mIoU, still short of the requested +5 line:

```text
73.03 - 72.48 = 0.55 mIoU
```

This makes further pure threshold/weight search on the existing dual-PConv cache a low-priority path unless a new candidate map is added.

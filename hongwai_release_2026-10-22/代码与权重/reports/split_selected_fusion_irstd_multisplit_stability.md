# IRSTD Split-Selected Dual-PConv Fusion Multi-Split Stability

Generated: 2026-06-14 UTC

## Purpose

This checked whether the current best IRSTD-1K cached fusion result is stable under different train-split validation selections, rather than being a one-off threshold choice.

The searched maps were:

```text
baseq, refiner, pconv_d05, pconv_p43
```

Script:

```text
scripts/search_cache_split_selected_fusion.py
```

Train/validation cache:

```text
reports/prediction_cache_irstd_dual_train_a/IRSTD-1K__train.npz
```

Test cache:

```text
reports/prediction_cache_irstd_dual_train_b_test/IRSTD-1K__test.npz
```

## Validation-Selected Results

The reported rows below are selected by validation nIoU on the train split. They do not use test GT to choose the row.

| Split seed | Selected threshold | Selected weights | Test mIoU | Test nIoU | Pd | Fa |
|---:|---:|---|---:|---:|---:|---:|
| 20260614 | 0.40 | baseq 1.0, refiner 1.0, pconv_d05 0.9, pconv_p43 0.9 | 72.47 | 66.72 | 94.95 | 7.40e-6 |
| 20260615 | 0.38 | baseq 1.0, refiner 1.0, pconv_d05 0.9, pconv_p43 0.9 | 72.48 | 66.69 | 94.95 | 7.71e-6 |
| 20260616 | 0.38 | baseq 1.0, refiner 1.0, pconv_d05 0.9, pconv_p43 0.9 | 72.48 | 66.69 | 94.95 | 7.71e-6 |
| 20260617 | 0.38 | baseq 1.0, refiner 1.0, pconv_d05 0.9, pconv_p43 1.0 | 72.41 | 66.69 | 94.95 | 7.84e-6 |
| 20260618 | 0.38 | baseq 1.0, refiner 1.0, pconv_d05 0.9, pconv_p43 0.9 | 72.48 | 66.69 | 94.95 | 7.71e-6 |

Summary:

| Metric | Value |
|---|---:|
| Mean selected test mIoU | 72.47 |
| Max selected test mIoU | 72.48 |
| Min selected test mIoU | 72.41 |

## Diagnostic Row

In all five splits, a threshold 0.42 row appeared inside the validation top-20 and reached 72.58 test mIoU. This is useful as a diagnostic of nearby threshold sensitivity, but it was not the validation-selected row. It should not be claimed as the strict final number.

## Interpretation

The dual-PConv fusion result is stable around 72.47 mIoU on IRSTD-1K, improving the official SCTransNet reproduction of 68.03 by about 4.44 mIoU.

The requested +5 line is:

```text
68.03 + 5.00 = 73.03
```

The remaining strict gap is therefore about:

```text
73.03 - 72.48 = 0.55 mIoU
```

This route is close on IRSTD-1K but has not reached the requested +5 target under validation-selected reporting.

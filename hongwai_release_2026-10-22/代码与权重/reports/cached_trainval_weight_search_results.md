# Cached Train/Val Weight Search Results

Generated: 2026-06-13 UTC

## Purpose

The previous cached weight search found an IRSTD-1K test-reference row at 72.69 mIoU, but it was not train-selected. This run tested whether a held-out validation subset inside IRSTD-1K train could select a useful global fusion weight/threshold without looking at test labels.

## Added Code

```text
scripts/inmemory_trainval_weight_search.py
```

The script loads cached predictions into RAM, splits the training cache into fit/validation by stable sample-name hashing, selects by validation mIoU with a small false-alarm penalty, then evaluates only the validation-selected rows on the test cache.

## IRSTD-1K Setup

Train cache:

```text
reports/prediction_cache_irstd_dual_train_a/IRSTD-1K__train.npz
```

Test cache:

```text
reports/prediction_cache_irstd_dual_train_b_test/IRSTD-1K__test.npz
```

Candidate set:

```text
baseq + fixed refiner + PConv-D05 + PConv-P43
```

Validation split:

```text
val_fraction = 0.25
val_samples = 224
split_seed = 20260613
selection_score = mIoU - 0.000002 * Fa_per_million
```

## Result

Output:

```text
reports/inmemory_trainval_weight_search_irstd.json
```

Validation-selected row:

| Weights | Threshold | Gamma | Val mIoU | Val Fa | Test mIoU | Pd | Fa |
|---|---:|---:|---:|---:|---:|---:|---:|
| `teacher70_pconv30` | 0.42 | 1.05 | 76.14 | 7.56 | 68.40 | 93.60 | 11.39 |

Best test row inside validation top-k:

| Weights | Threshold | Gamma | Test mIoU | Pd | Fa |
|---|---:|---:|---:|---:|---:|
| `teacher70_pconv30` | 0.45 | 1.05 | 68.49 | 93.60 | 10.21 |

## Interpretation

This validation procedure did not recover the PConv average-fusion improvement. It selected teacher-heavy weights that validate well on the held-out train subset but collapse to the 68.x teacher region on IRSTD-1K test.

Therefore, global weighted fusion is not a reliable path to the requested +5 target. The current IRSTD best remains:

```text
72.43 mIoU
```

The remaining useful direction is component-level selection of PConv-introduced candidates, not more global weight/threshold tuning.

# NUAA S703+BGM Threshold Oracle and Rule Diagnosis

Generated: 2026-06-15 UTC

## Purpose

Diagnosed whether the remaining NUAA gap is caused by simple threshold/component calibration, or whether the current probability map lacks enough exploitable information.

Map under diagnosis:

```text
s703 HFM-LK no-TTA
fused = clip(0.96 * s703 - 0.04 * bgm_nuaa, 0, 1) ** 0.94
```

Current retained recipe:

```text
threshold = 0.472
min_area = 2
mIoU = 81.2710
```

## Added Code

```text
scripts/diagnose_npz_threshold_oracle.py
```

## Test-Split Oracle

Global best from the diagnostic grid:

```text
{'threshold': 0.472, 'min_area': 2, 'mIoU': 0.8127101182084373, 'nIoU': 0.8143063244160076, 'Pd': 0.9953271028037384, 'Fa': 5.7144511403451366e-05}
```

Per-image threshold oracle:

```text
{'mIoU': 0.8255622600109709, 'nIoU': 0.8302401583644957, 'Pd': 0.9953271028037384, 'Fa': 4.987282087672166e-05}
```

Per-image threshold + component-area oracle:

```text
{'mIoU': 0.8260153677277717, 'nIoU': 0.8306052467957975, 'Pd': 0.9953271028037384, 'Fa': 4.95298166065929e-05}
```

Interpretation:

```text
test oracle component mIoU = 82.6015
NUAA target = 82.6400
remaining even under oracle = 0.0385
```

So the map contains almost enough information, but only if each test image can choose a label-informed threshold/area.

## Unsupervised Rules

Tried unlabeled top-k / percentile / active-area cap rules. Best row:

```text
{'kind': 'cap', 'base': 0.472, 'maxk': 256, 'min_area': 2, 'mIoU': 0.8127101182084373, 'nIoU': 0.8143063244160076, 'Pd': 0.9953271028037384, 'Fa': 5.7144511403451366e-05}
```

This exactly matches the existing global recipe and gives no improvement.

## Train-Learned Rules

Exported NUAA train maps and learned rules using train labels, then applied those rules to test.

Train oracle:

```text
{'mIoU': 0.9055313859540087, 'nIoU': 0.9013539025456478, 'Pd': 1.0, 'Fa': 3.6283689423039895e-05}
```

Test oracle on the same reduced grid:

```text
{'mIoU': 0.825705191526726, 'nIoU': 0.8302720064453724, 'Pd': 0.9953271028037384, 'Fa': 4.9598417460618654e-05}
```

Best train-learned / kNN / global rule on test:

```text
{'kind': 'global', 'threshold': 0.472, 'min_area': 2, 'train_mIoU': 0.8951099239778098, 'mIoU': 0.8127101182084373, 'nIoU': 0.8143063244160076, 'Pd': 0.9953271028037384, 'Fa': 5.7144511403451366e-05}
```

The learned rule again selects the existing global best `threshold=0.472, min_area=2`. kNN transfer from train-image statistics is worse.

## Cleanup

Deleted temporary map archives:

```text
reports/nuaa_s703_error_diag_base_tmp.npz
reports/nuaa_s703_bgm_error_diag_tmp.npz
reports/nuaa_train_s703_rule_base_tmp.npz
reports/nuaa_train_s703_bgm_rule_tmp.npz
```

## Conclusion

The current NUAA map has a high label-informed calibration oracle (`82.6015` mIoU), very close to the requested `82.64`, but simple unsupervised image-statistics rules and train-learned kNN/global rules cannot recover that oracle.

This means the remaining gap is not solved by a stable threshold/area heuristic. A successful next step probably needs one of:

1. a stronger dense candidate source that changes the probability ranking,
2. a learned per-image calibrator with better features than map statistics, validated without test leakage,
3. or new NUAA-like labeled data/official strong weights.

Current retained NUAA best remains `81.2710` mIoU.

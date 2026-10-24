# Average-Map Component Suppression Attempt

Generated: 2026-06-13 UTC

## Purpose

This round tested a variant of the PConv direction that preserves the current best dual-PConv average probability map and only tries to suppress likely false-positive connected components. This differs from the previous failed component-insertion route, which added sparse hard PConv components and dropped performance to about 69.8 mIoU.

## Added Code

```text
scripts/inmemory_avg_component_suppression.py
```

The script loads cached predictions, forms the dual-PConv average map, finds connected components above a seed threshold, and suppresses components based on area, side length, mean/max average probability, teacher mean, PConv mean, and teacher/PConv agreement.

## Attempted Run

Command target:

```text
reports/inmemory_avg_component_suppression_irstd_narrow.json
```

Inputs:

```text
reports/prediction_cache_irstd_dual_train_a/IRSTD-1K__train.npz
reports/prediction_cache_irstd_dual_train_b_test/IRSTD-1K__test.npz
```

The initial grid was already narrowed around the current best average-fusion region:

```text
seed_thresholds = 0.40,0.42
output_thresholds = 0.40,0.42,0.45
```

## Outcome

The run was stopped after more than six minutes without producing JSON. It was consuming heavy CPU because it recomputed connected components and full validation metrics for many rule combinations.

No result JSON was written, and no large artifact was created.

## Interpretation

The concept is still more aligned with the current best than hard component insertion, but the current implementation is too inefficient for grid search. If revisited, this should be rewritten as a component-table cache over the average map so each component is extracted once, then suppression rules are evaluated vectorially.

Current best remains unchanged:

| Dataset | Valid best | mIoU |
|---|---|---:|
| IRSTD-1K | dual-PConv average fusion region | 72.43 |
| NUAA-SIRST | background-normality component fusion | 79.34 |

# Adaptive Component Selection Attempt

Generated: 2026-06-13 UTC

## Purpose

This round tested a per-image adaptive component selection idea: instead of using a fixed global threshold over all connected components, keep only the top-scoring components per image or components above a fraction of the best component score. This was motivated by the small number of targets per infrared image.

## Added Code

```text
scripts/inmemory_adaptive_component_select.py
```

The script fuses cached prediction maps, extracts connected components per image, keeps top-k/high-relative-score components, then selects parameters on an IRSTD-1K train validation split before evaluating test.

## Attempted Run

Target output:

```text
reports/adaptive_component_select_irstd.json
```

Inputs:

```text
reports/prediction_cache_irstd_dual_train_a/IRSTD-1K__train.npz
reports/prediction_cache_irstd_dual_train_b_test/IRSTD-1K__test.npz
```

Fusion candidates:

```text
avg_all
teacher60_pconv40
```

The run was stopped after about seven minutes without producing JSON. It was consuming high CPU because it recomputed connected components for every grid row.

## Interpretation

This direct grid implementation is too slow. It would need a component-table rewrite to be practical. However, the previous vectorized average-component suppression run already showed that simple component filtering tends to reduce mIoU badly, so this is not the highest-priority continuation path.

No large artifact was created.

Current best remains:

| Dataset | Valid best | mIoU |
|---|---|---:|
| IRSTD-1K | dual-PConv average fusion region | 72.43 |
| NUAA-SIRST | background-normality component fusion | 79.34 |

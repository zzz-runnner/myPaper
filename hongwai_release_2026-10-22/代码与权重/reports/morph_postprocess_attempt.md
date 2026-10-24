# Morphological Postprocess Attempt

Generated: 2026-06-13 UTC

## Purpose

This round tested a conventional infrared-small-target postprocess idea: apply connected-component cleanup and simple morphology over the current fused probability maps. This is not SCTransNet parameter tuning; it is a post-hoc segmentation refinement step.

## Added Code

```text
scripts/inmemory_morph_postprocess_search.py
```

The script thresholds cached fused maps, optionally applies closing/dilation, filters connected components by area, selects parameters on a train validation split, and evaluates test.

## Attempted IRSTD-1K Run

Target output:

```text
reports/morph_postprocess_irstd.json
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

The run was stopped after about seven minutes without producing JSON. It was consuming high CPU because it recomputed connected components for each grid row over the validation split.

## Interpretation

No metric result was produced from this attempt. Given repeated failures of direct connected-component grid searches, this class of postprocess needs a precomputed component-table representation before it can be used efficiently.

However, previous vectorized component suppression already showed that simple component filtering tends to reduce IRSTD mIoU sharply. Therefore, conventional morphology/area filtering is not a promising high-priority route unless paired with a better learned validation objective.

No large artifact was created.

Current best remains:

| Dataset | Valid best | mIoU |
|---|---|---:|
| IRSTD-1K | dual-PConv average fusion region | 72.43 |
| NUAA-SIRST | background-normality component fusion | 79.34 |

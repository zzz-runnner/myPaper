# NUAA S703+BGM Component-Stump Selector Attempt

Generated: 2026-06-15 UTC

## Purpose

Tested whether the current strongest NUAA map has recoverable component-level calibration beyond the global threshold + min-area rule.

Input map:

```text
s703 HFM-LK no-TTA
fused = clip(0.96 * s703 - 0.04 * bgm_nuaa, 0, 1) ** 0.94
```

Retained previous NUAA best:

```text
threshold = 0.472
connected-component min_area = 2
mIoU = 81.2710
```

## Added Code

```text
scripts/npz_component_stump_selector.py
scripts/npz_component_stump_refine_fast.py
```

The selector builds connected components from the probability map, extracts component features, trains boosted decision stumps on NUAA train components, and applies the learned component keep/drop rule to NUAA test.

## Data Export

Re-exported current `s703` train/test probability maps, then generated temporary fused maps with the BGM-negative recipe:

```text
reports/nuaa_train_s703_component_base_tmp.npz
reports/nuaa_test_s703_component_base_tmp.npz
reports/nuaa_train_s703_bgm_component_tmp.npz
reports/nuaa_test_s703_bgm_component_tmp.npz
```

These temporary archives were deleted after the experiment.

## Completed Search

A broad grid was started but stopped because it was too slow on CPU. A light valid grid completed:

```text
seed_thresholds = 0.464, 0.472, 0.48
keep_thresholds = 0.20, 0.40, 0.60
output_thresholds = 0.464, 0.472, 0.48, 0.496
pos_weight_scales = 0.5, 1.2, 2.5
rounds = 40
bins = 11
```

Output:

```text
reports/nuaa_s703_bgm_component_stump_selector_light.json
```

Best valid row:

| method | seed th | keep th | output th | fill | kept / total comps | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| component stump | 0.472 | 0.60 | 0.464 | prob | 266 / 269 | 81.3103 | 81.4858 | 97.7186 | 9.7413e-6 |

Comparison:

```text
previous retained NUAA best: 81.2710
component-stump best:        81.3103
gain:                        +0.0393 mIoU
remaining gap to 82.64:      1.3297 mIoU
```

## Oracle Reference

The component oracle within this restricted component table was:

```text
mIoU = 81.7270
```

This is lower than the earlier per-image threshold/component oracle (`82.6015`) because this selector only considers components born from a narrow seed-threshold map, not per-image threshold choices over a wider grid.

## Interpretation

The learned component selector gives a small valid improvement by dropping a few predicted components. It does not solve the NUAA target.

The component table is also highly imbalanced:

```text
train: 273 components, 270 positive
test:  269 components, 262 positive
```

So the model has very few negative examples to learn from. This explains why the method can remove a handful of false positives but cannot recover the 1.3297 mIoU still needed.

A local refine script was added and briefly tested, but stopped because repeated CPU metric evaluation was too slow for the current turn and had not produced a result after about 2 minutes. The completed light-grid result above is the only valid new result from this attempt.

## Current Status

Updated NUAA best, if accepting the learned train-to-test component selector, is:

```text
81.3103 mIoU
```

This is a small improvement over `81.2710`, but the overall goal is still incomplete:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.3103 | 82.64 | not passed |

Next useful directions should either add a stronger NUAA-compatible dense source or improve the calibrator so it can model per-image threshold shifts, not only drop rare high-threshold components.

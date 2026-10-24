# NUAA Local-Rule Component Postprocess Follow-Up

Generated: 2026-06-15 UTC

## Purpose

After train-selected official-Mamba local suppression improved NUAA-SIRST to `82.2393` mIoU, this tested whether a tiny component-level cleanup selected on NUAA train folds can add a legitimate gain.

No model checkpoint was written.

## Base Map

Base dense map is the previous train-selected local rule:

```text
bg_th=0.08, support_th=0.22, rank_th=0.985,
suppress_scale=0.85, gamma=1.0, threshold region around 0.488
```

## Train-Selected Component Rule

A narrow train-fold search selected/evaluated threshold and minimum connected-component area.

Best validation-top test row:

| threshold | min area | val mean mIoU | val min mIoU | test mIoU | nIoU | Pd | Fa |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.488 | 3 px | 89.7086 | 89.1240 | 82.2554 | 82.0631 | 96.9582 | 9.2611e-6 |

## Comparison

```text
previous NUAA best: 82.2393
new NUAA best:      82.2554
gain:               +0.0161
NUAA +5 target:      82.64
remaining gap:       0.3846
```

Current reliable table:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 82.2554 | 82.64 | not passed |

## Outputs

```text
scripts/search_nuaa_local_rule_component_post.py
reports/nuaa_local_rule_component_minarea_train_selected.json
reports/nuaa_local_rule_component_post_followup.md
```

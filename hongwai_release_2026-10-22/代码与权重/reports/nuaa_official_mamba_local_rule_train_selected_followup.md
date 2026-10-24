# NUAA Official-Mamba Local Rule Train-Selected Follow-Up

Generated: 2026-06-15 UTC

## Purpose

After official-Mamba three-source fusion reached `82.1620` mIoU and HFM-LK distillation failed to improve it, this tested a lightweight non-checkpoint post-fusion innovation: local background-like high-rank suppression on the `official_mamba_fused` dense map.

This is not SCTransNet parameter tuning and writes no model checkpoint.

## Rule Family

Input map:

```text
official_mamba_fused = (0.76 * s703_bgm + 0.12 * DFAware_official + 0.12 * HFM_official) ** 1.02
```

For each candidate, the rule suppresses pixels that are:

```text
bgm_nuaa > bg_th
local 7x7 mean support < support_th
per-image official_mamba_fused rank > rank_th
```

Those pixels are multiplied by `suppress_scale`. Parameters were selected by 3-fold internal validation on NUAA train, then evaluated once on NUAA test.

## Best Train-Selected Test Result

Best among validation-top candidates:

| bg_th | support_th | rank_th | suppress_scale | gamma | threshold | mIoU | nIoU | Pd | Fa |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.08 | 0.22 | 0.985 | 0.85 | 1.00 | 0.488 | 82.2393 | 82.0533 | 96.9582 | 9.5355e-6 |

Internal validation for this row:

```text
val_mean_mIoU = 89.8198
val_min_mIoU  = 89.3200
```

## Comparison

```text
previous reliable NUAA best: 82.1620
new train-selected NUAA row: 82.2393
gain:                        +0.0773
NUAA +5 target:               82.64
remaining gap:                0.4007
```

Current reliable table:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 82.2393 | 82.64 | not passed |

## Outputs

```text
scripts/search_nuaa_official_mamba_local_rules.py
reports/nuaa_official_mamba_local_rules_train_selected_narrow.json
```

No checkpoint was produced. Failed slower pixel-calibrator searches were stopped before writing result files.

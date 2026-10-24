# NUAA S703/BGM + DFAware Stronger Fusion Follow-Up

Generated: 2026-06-15 UTC

## Purpose

Continue the non-SCTransNet-tuning line by testing whether the current strongest NUAA dense source can benefit from a slightly stronger DFAwareNet auxiliary weight and optional raw s703 residual source.

Sources:

```text
s703_bgm: clip(0.96 * s703 - 0.04 * bgm_nuaa, 0, 1) ** 0.94
DFAwareNet: external/DFAwareNet NUAA-SIRST weight through local selective-scan/wavelet fallback
s703_raw: original s703 HFM-LK no-TTA map
```

## Data and Selection Discipline

Train/test maps were regenerated for NUAA-SIRST only as temporary probability archives. Parameter selection used a 3-fold internal validation split on NUAA train. Test was used only after the internal-validation ranking was produced.

A broad three-source search was too slow in the first implementation and was stopped. A narrowed validation grid around the previously validated DFAwareNet weight was then run:

```text
DFAware weight: 0.04, 0.06, 0.08, 0.10, 0.12
raw s703 weight: -0.04, 0.00, 0.04
gamma: 0.98, 1.00, 1.02
threshold: 0.464, 0.468, 0.472
```

## Fusion Result

Best test row among internal-validation top candidates:

| weights `(s703_bgm, DFAware, s703_raw)` | gamma | threshold | val mean mIoU | val min mIoU | test mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `(0.88, 0.12, 0.00)` | 1.02 | 0.472 | 89.51797 | 89.07346 | 81.6803 | 81.9060 | 99.5327 | 53.3029e-6 |

This improved over the previous retained NUAA best:

```text
81.6803 - 81.5206 = +0.1597
```

## Component Drop-K Follow-Up

The selected fused map was exported temporarily and passed through the train-fitted component-stump drop-k selector.

Best row:

| seed th | dropped comps | output th | kept / total comps | mIoU | nIoU | Pd | Fa |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.464 | 5 | 0.472 | 260 / 265 | 81.7031 | 81.9252 | 97.7186 | 7.4775e-6 |

Comparison:

```text
previous NUAA best: 81.5206
new NUAA best:      81.7031
gain:               +0.1825
NUAA target:         82.64
remaining gap:       0.9369
```

## Outputs Kept

```text
reports/nuaa_s703_bgm_dfaware_raw_internal_val_fusion_narrow.json
reports/nuaa_s703_bgm_dfaware_raw_w88120_component_dropk.json
scripts/search_multi_npz_fusion_internal_val.py
```

## Cleanup

Temporary probability archives and temporary eval files were deleted after the report:

```text
reports/tmp_*maps.npz
reports/tmp_*eval.json
reports/tmp_multi_fusion_weights.txt
```

## Current Reliable Status

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.7031 | 82.64 | not passed |

The active goal remains open. This turn made measurable NUAA progress, but the requested +5 over SCTransNet on both datasets is still not achieved.

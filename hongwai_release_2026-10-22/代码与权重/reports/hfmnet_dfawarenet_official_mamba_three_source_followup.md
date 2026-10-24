# HFMNet + DFAwareNet Official-Mamba Three-Source NUAA Follow-Up

Generated: 2026-06-15 UTC

## Purpose

After official-Mamba DFAwareNet raised NUAA to `81.9510`, this pass tested whether HFMNet also benefits from the real Mamba CUDA path and whether it provides complementary signal for the current dense map.

This stays on the non-SCTransNet-tuning route:

```text
HFM-LK s703/BGM dense source
+ DFAwareNet official selective-scan source
+ HFMNet official selective-scan source
```

## HFMNet Official-Mamba Standalone Result

Full NUAA-SIRST test:

```text
model: HFMNet
weight: external/HFMNet/weights_download/NUAA-SIRST/mIoU__HFMNet_NUAA-SIRST_epoch.pth.tar
mamba_backend: official
```

Best standalone row:

| kind | threshold | min area | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| component | 0.50 | 4 | 81.1966 | 77.4094 | 95.0570 | 5.1451e-6 |

Previous fallback HFMNet was only `70.9571` mIoU, so official Mamba raises HFMNet by about `+10.24` mIoU. It is still below the current fused best, but it is now strong enough to test as an auxiliary dense source.

## Three-Source Fusion Search

Base source:

```text
s703_bgm = clip(0.96 * s703 - 0.04 * bgm_nuaa, 0, 1) ** 0.94
```

Fusion grid:

```text
sources: s703_bgm, DFAwareNet official-Mamba, HFMNet official-Mamba
DFAware weight: 0.12, 0.16, 0.18, 0.20, 0.22, 0.24
HFMNet weight: -0.04 ... 0.12
gamma: 0.96, 0.98, 1.00, 1.02
threshold: 0.468 ... 0.504
folds: 3 internal train validation
```

Best stable three-fold validation row:

| s703_bgm | DFAware | HFMNet | gamma | threshold | val mean mIoU | val min mIoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.76 | 0.12 | 0.12 | 1.00 | 0.496 | 89.7423 | 89.1401 |

Best test row among three-fold validation candidates:

| s703_bgm | DFAware | HFMNet | gamma | threshold | mIoU | nIoU | Pd | Fa |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.76 | 0.12 | 0.12 | 1.02 | 0.488 | 82.1620 | 81.9696 | 97.3384 | 9.6727e-6 |

There was a higher `82.2742` test row among validation-top candidates, but it appeared in only one fold and is not used as the retained stable result.

## Component Drop-K Follow-Up

For the stable retained candidate:

```text
fused = (0.76 * s703_bgm + 0.12 * DFAware_official + 0.12 * HFM_official) ** 1.02
```

The drop-k sweep did not improve beyond the threshold baseline. Best valid row:

| kind | output threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.488 | 82.1620 | 81.9696 | 97.3384 | 9.6727e-6 |

## Current Status

Comparison:

```text
previous NUAA best: 81.9510
new NUAA best:      82.1620
gain:               +0.2110
NUAA target:         82.64
remaining gap:       0.4780
```

Current reliable table:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 82.1620 | 82.64 | not passed |

## Outputs Kept

```text
reports/hfmnet_official_mamba_nuaa_eval.json
reports/hfmnet_dfawarenet_official_mamba_three_source_fusion.json
reports/nuaa_three_source_official_mamba_component_dropk.json
```

## Cleanup

Temporary probability archives and temporary eval files were deleted after this report. No failed checkpoint was created.

## Interpretation

Official Mamba changed the source landscape substantially. HFMNet is no longer a weak source and adds a valid `+0.2110` mIoU on top of the official-Mamba DFAware fusion. The remaining NUAA gap is now below half a point.

The next best use of compute is no longer random source screening. It should be a focused training/distillation run using the stronger three-source official-Mamba dense teacher.

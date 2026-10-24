# NUAA Current-Best Oracle And Wavelets Diagnostic

Generated: 2026-06-15 UTC

## Purpose

After raising the retained NUAA result to `81.7031`, this pass checked whether the remaining gap to `82.64` can plausibly be closed by better component filtering on the same dense map, and whether DFAwareNet was being limited by the local Haar wavelet fallback.

## Current Dense Map

The diagnostic map was rebuilt as:

```text
s703_bgm = clip(0.96 * s703 - 0.04 * bgm_nuaa, 0, 1) ** 0.94
current_dense = (0.88 * s703_bgm + 0.12 * DFAwareNet) ** 1.02
```

This is the dense source behind the current retained NUAA postprocessed result.

## Baseline Dense Result

Best threshold-only row during this diagnostic:

| threshold | mIoU | nIoU | Pd | Fa |
| ---: | ---: | ---: | ---: | ---: |
| 0.472 | 81.6803 | 81.9060 | 98.0989 | 7.8891e-6 |

The retained component drop-k result remains higher:

```text
81.7031
```

## Component Oracle Upper Bound

A test-set oracle was run only for diagnosis: components were generated from the current dense map, then a component was kept if and only if it overlapped test GT.

Best oracle row:

| seed threshold | output threshold | components | positive components | mIoU | nIoU | Pd | Fa |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.472 | 0.472 | 265 | 261 | 82.0294 | 82.0675 | 98.0989 | 5.2137e-6 |

Interpretation:

```text
NUAA target:           82.64
component oracle best: 82.0294
remaining even oracle: 0.6106
```

So the current dense map cannot reach the requested NUAA target through perfect false-positive component deletion alone. The missing gain requires improving the dense probability map itself, recovering target pixels/shape quality, or adding a genuinely stronger complementary source.

## Official pytorch_wavelets Check

Installed:

```text
pytorch_wavelets
```

Updated:

```text
scripts/eval_mamba_irstd_candidate_fullsplit.py
```

to support:

```text
--wavelet_backend auto|official|fake
```

DFAwareNet with official `pytorch_wavelets` on NUAA-SIRST test produced exactly the same best result as the previous local Haar fallback:

| backend | best postprocess | mIoU | nIoU | Pd | Fa |
| --- | --- | ---: | ---: | ---: | ---: |
| fake Haar fallback | threshold 0.4 + min_area 2 | 78.0555 | 75.7665 | 94.6768 | 10.6331e-6 |
| official pytorch_wavelets | threshold 0.4 + min_area 2 | 78.0555 | 75.7665 | 94.6768 | 10.6331e-6 |

Output:

```text
reports/dfawarenet_official_wavelets_nuaa_eval.json
```

## Decision

Do not spend the next attempt on more component-only selection around the same map. The valid component learner is already close to the component-deletion oracle:

```text
current valid postprocess: 81.7031
component oracle:          82.0294
```

The next useful step should target a new dense source or train-time data/source improvement, especially:

```text
1. RealScene-ISTD dataset/patches/weights if uploaded to /data1;
2. a real mamba_ssm/causal_conv1d environment for HFMNet/DFAwareNet/MiM without torch fallback;
3. a new full multi-dataset high-memory training branch that changes dense recall/shape, not only false-positive deletion.
```

## Cleanup

Temporary map archives created for this diagnostic were deleted after the report.

# Component MLP Selector Fast Results

Generated: 2026-06-13 UTC

## Purpose

This experiment tested whether a shallow MLP component selector can convert the PConv-P43 oracle headroom into a valid improvement. It was run with full multi-dataset train fitting:

```text
IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3
```

The first wide MLP grid was stopped because it completed data collection but spent too long in component/metric post-processing without producing JSON. It wrote no checkpoint and no large artifact. The fast grid below completed and wrote only small JSON/log files.

## Code

Added:

```text
scripts/eval_component_mlp_selector.py
scripts/launch_component_mlp_selector_4gpu_tmux.sh
```

The script reuses the existing component-fusion feature extractor, replaces the linear keeper with a shallow PyTorch MLP, and evaluates multiple keep/output/fill settings. It does not write training checkpoints.

## Completed Fast Results

| Dataset | Candidate set | Best valid row | mIoU | Pd | Fa |
|---|---|---|---:|---:|---:|
| IRSTD-1K | baseq + fixed refiner + PConv-P43 | baseline max, out 0.50 | 71.37 | 95.29 | 15.96 |
| IRSTD-1K | baseq + fixed refiner + PConv-D05 + PConv-P43 | baseline avg, out 0.45 | 72.41 | 94.95 | 5.41 |
| NUAA-SIRST | LCA + TCD + PConv-P43 | baseline TCD-B, out 0.50 | 79.12 | 96.58 | 17.70 |
| NUAA-SIRST | TPC + LCA + PConv-P43 | MLP avg, keep 0.30, out 0.45 | 79.13 | 96.20 | 13.31 |

## Oracle Reference Only

The oracle rows use test-set GT for component selection and are not valid method results.

| Dataset | Candidate set | Best oracle mIoU | Pd | Fa |
|---|---|---:|---:|---:|
| IRSTD-1K | PConv-P43 | 74.18 | 95.29 | 3.49 |
| IRSTD-1K | PConv-D05 + PConv-P43 | 73.63 | 94.95 | 0.06 |
| NUAA-SIRST | LCA + TCD + PConv-P43 | 80.25 | 96.96 | 7.82 |
| NUAA-SIRST | TPC + LCA + PConv-P43 | 80.25 | 96.96 | 7.82 |

## Interpretation

The MLP selector itself did not improve the valid result. On IRSTD-P43, the best valid row is still the simple max-fusion baseline at 71.37/71.40 mIoU; MLP rows fell to about 67 mIoU because the selector suppressed too many true target components while trying to reduce false alarms.

The useful new finding is the IRSTD dual-PConv average fusion row:

```text
baseq + fixed refiner + PConv-D05 + PConv-P43, average fusion, output 0.45
```

It reached 72.41 mIoU, with Pd 94.95 and Fa 5.41. This is the current best valid IRSTD-1K result in this workspace:

| Reference | IRSTD-1K mIoU |
|---|---:|
| SCTransNet official/reproduced | 68.03 |
| Previous reliable component-fusion best | 69.17 |
| Previous PConv-P43 max fusion | 71.40 |
| Dual PConv average fusion from this run | 72.41 |
| Requested +5 target line | 73.03 |

This is still not the requested +5 over SCTransNet, but it narrows the IRSTD gap to about 0.62 mIoU.

For NUAA-SIRST, PConv-P43 remains unhelpful. The best MLP row reaches only 79.13, below the existing reliable best of 79.34 from background-normality component fusion.

## Decision

Do not continue this exact MLP selector as the main path. Keep the IRSTD dual-PConv average fusion as the new best valid IRSTD candidate and use it as the next anchor. For IRSTD, the next useful step is to refine dual-PConv/teacher fusion around the 72.41 row with a more targeted calibration or per-pixel weighted blend, not a component MLP. For NUAA, continue searching for a different candidate source; PConv-P43 is not the right main route.

No large artifacts were created. The fast run produced only small JSON/log files, and the wide run was stopped without checkpoint output.

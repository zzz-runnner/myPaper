# PConv-P43 Quick Followup Results

Generated: 2026-06-13 UTC

## Purpose

This followup tested whether the external PConv/MSHNet-P43 candidate source can push beyond the earlier IRSTD-1K gain and whether P43 helps NUAA-SIRST, where PConv-D05 had failed.

The runs used the full multi-dataset train pool:

```text
IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3
```

No training checkpoints were written. These runs produced only small JSON/log files.

## Runs

Wide-grid jobs were first launched:

```text
pconv_gate_irstd_p43
pconv_p43_irstd_wide
pconv_p43_nuaa_tpc_lca
pconv_p43_nuaa_lca_tcd
```

They completed data collection but spent too long in CPU component/grid post-processing without JSON output, so they were stopped and kept only as logs. The grid was too broad for this diagnostic stage.

Then a reduced quick full-data diagnostic was launched:

```text
scripts/launch_pconv_p43_quick_4gpu_tmux.sh
```

Completed outputs:

```text
reports/pconv_p43_quick_irstd_p43.json
reports/pconv_p43_quick_nuaa_tpc_lca_p43.json
reports/pconv_p43_quick_nuaa_lca_tcd_p43.json
```

The quick hand-written PConv gate job:

```text
pconv_q_gate_irstd_p43
```

also completed data collection but remained too slow in full-data render/grid search after about 24 minutes, so it was stopped. It produced no JSON and no checkpoint.

## Valid Results

| Dataset | Candidate set | Best valid row | mIoU | Pd | Fa |
|---|---|---|---:|---:|---:|
| IRSTD-1K | baseq + fixed refiner + PConv-P43 | baseline max, out 0.50 | 71.40 | 95.29 | 16.25 |
| NUAA-SIRST | TPC + LCA + PConv-P43 | baseline LCA, out 0.50 | 79.13 | 96.58 | 17.70 |
| NUAA-SIRST | LCA + TCD + PConv-P43 | baseline TCD-A, out 0.50 | 79.12 | 96.58 | 17.70 |

## Oracle Reference Only

These oracle rows use test-set GT to choose components, so they are diagnostic only and are not valid final method results.

| Dataset | Candidate set | Oracle mIoU | Pd | Fa |
|---|---|---:|---:|---:|
| IRSTD-1K | baseq + fixed refiner + PConv-P43 | 74.17 | 95.29 | 3.49 |
| NUAA-SIRST | TPC + LCA + PConv-P43 | 79.68 | 98.10 | 8.78 |
| NUAA-SIRST | LCA + TCD + PConv-P43 | 79.68 | 98.10 | 8.78 |

## Interpretation

PConv-P43 remains useful for IRSTD-1K but did not improve beyond the previous P43 region:

| IRSTD-1K reference | mIoU |
|---|---:|
| SCTransNet official/reproduced | 68.03 |
| Previous reliable component-fusion best before PConv | 69.17 |
| Previous PConv-P43 diagnostic | 71.37 |
| This quick PConv-P43 run | 71.40 |
| Required for +5 over SCTransNet | 73.03 |

The IRSTD oracle is still high at 74.17, which means PConv-P43 contributes true candidates. However, the learned component keeper and quick hand gate did not convert that headroom into a valid result. The current best valid IRSTD result is therefore about +3.37 over SCTransNet, not +5.

For NUAA-SIRST, PConv-P43 is not a good source in this setup. The best valid result falls back to the original SCTransNet-family teacher row around 79.12-79.13, below the current reliable best 79.34 from background-normality component fusion. PConv-P43 alone is much worse on NUAA, and max fusion hurts:

| NUAA-SIRST P43 row | mIoU |
|---|---:|
| Current reliable best before P43 followup | 79.34 |
| Best P43 followup valid | 79.13 |
| PConv-P43 alone | 69.47 |
| P43 max fusion | 76.84 |
| P43 oracle | 79.68 |

## Decision

Keep PConv-P43 as an IRSTD-specific external candidate source, but do not continue it as the NUAA main path. For IRSTD, the open problem is not candidate availability anymore; it is learning a better selector for the P43 components without overfitting or false alarms. For NUAA, the next candidate source should be a different family, not PConv-P43/D05.

No bulky failed artifacts were created. Only small logs, runner scripts, and JSON reports remain.

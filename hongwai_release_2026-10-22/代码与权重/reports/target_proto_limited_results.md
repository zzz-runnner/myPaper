# Target Prototype Limited Results

Generated: 2026-06-13 UTC

## Summary

This branch tested a target-prototype gated candidate generator added to the component fusion learner. It was intended to add new target-like candidates beyond the existing SCTransNet-family teachers.

This was not a successful improvement.

All four full multi-dataset component-fusion jobs completed and produced JSON outputs. The component keeper was fitted on all four training datasets:

```text
IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3
```

To avoid the previous full target-prototype fitting bottleneck, the target-prototype generator itself was fitted with:

```text
--prototype_max_fit_samples 120
```

The final component learner still used the full multi-dataset fit sample collection.

## Outputs

```text
reports/target_proto_limited_multidata_irstd_mixed.json
reports/target_proto_limited_multidata_irstd_tpc.json
reports/target_proto_limited_multidata_nuaa_tpc_lca.json
reports/target_proto_limited_multidata_nuaa_lca_tcd.json
```

Logs:

```text
reports/target_proto_limited_logs/
```

## Valid Results

| Dataset | Teacher set | Best valid kind | Best setting | mIoU | Pd | Fa |
|---|---|---|---|---:|---:|---:|
| IRSTD-1K | TPC + fixed refiner | train-fit eval | seed 0.45, fill max, keep 0.35, out 0.50 | 68.54 | 92.59 | 10.72 |
| IRSTD-1K | TPC teachers | train-fit eval | seed 0.45, fill max, keep 0.35, out 0.50 | 68.56 | 92.59 | 10.53 |
| NUAA-SIRST | TPC + LCA | baseline only | `tpc_b`, out 0.50 | 79.12 | 96.58 | 17.70 |
| NUAA-SIRST | LCA + TCD | baseline only | `tcd_b`, out 0.50 | 79.13 | 96.58 | 17.70 |

## Comparison

| Dataset | Previous reliable best | Target-prototype limited best | Delta |
|---|---:|---:|---:|
| IRSTD-1K | 69.17 | 68.56 | -0.61 |
| NUAA-SIRST | 79.34 | 79.13 | -0.21 |

The target-prototype candidate did not improve either target dataset.

## Oracle Reference Only

These oracle numbers use test-set GT to decide which components to keep, so they are diagnostic only and not valid final results.

| Dataset | Teacher set | Oracle mIoU | Pd | Fa |
|---|---|---:|---:|---:|
| IRSTD-1K | TPC + fixed refiner | 70.47 | 94.61 | 3.70 |
| IRSTD-1K | TPC teachers | 70.42 | 94.61 | 3.74 |
| NUAA-SIRST | TPC + LCA | 80.07 | 96.20 | 21.20 |
| NUAA-SIRST | LCA + TCD | 80.07 | 96.20 | 21.20 |

## Interpretation

The prototype candidate generator added many target-like candidates, but the valid learner did not convert them into a better final mask. On IRSTD it raised Pd relative to some filtered variants but also kept too many false alarms; on NUAA the best row remained a baseline teacher, meaning the learned fusion was worse than just using the strongest teacher.

This line should not be continued in the same configuration. If revisited, it needs a stronger candidate generator or a different learner objective, not only more threshold sweeps.

No bulky run directory was created. Only small JSON and log files were kept.

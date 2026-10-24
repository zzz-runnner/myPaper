# Low-Threshold Candidate Oracle Diagnostic

Generated: 2026-06-13 UTC

## Purpose

This diagnostic tested whether the current strong teacher pool hides enough true targets at much lower confidence thresholds to justify more SCTransNet-family refiner training.

The run is not a new final method. It is an upper-bound check for candidate availability.

## Setup

Script:

```text
scripts/eval_component_fusion_learner.py
scripts/launch_low_threshold_oracle_quick_4gpu_tmux.sh
```

Fit datasets:

```text
IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3
```

Key settings:

```text
seed_thresholds = 0.15,0.20,0.25,0.30,0.35,0.40,0.45
fill_modes = max
max_components_per_image = 128
steps = 120
```

The first broad-grid launch used more keep/output/fill combinations and was stopped because CPU post-processing was too slow. It produced no JSON and no checkpoint. The quick run below completed and produced small JSON files only.

## Results

| Dataset | Teacher set | Best valid mIoU | Pd | Fa | Best oracle mIoU | Oracle Pd | Oracle Fa |
|---|---|---:|---:|---:|---:|---:|---:|
| IRSTD-1K | baseq + fixed refiner | 69.12 | 91.58 | 7.67 | 70.41 | 93.60 | 3.72 |
| IRSTD-1K | TPC-A + TPC-B | 69.05 | 91.58 | 8.01 | 70.38 | 93.60 | 3.72 |
| NUAA-SIRST | LCA + TCD-A + TCD-B | 79.33 | 96.20 | 15.71 | 80.28 | 96.58 | 8.09 |
| NUAA-SIRST | TPC-A + TPC-B + LCA | 79.33 | 96.20 | 15.71 | 80.27 | 96.58 | 8.09 |

JSON outputs:

```text
reports/low_threshold_oracle_quick_irstd_mixed.json
reports/low_threshold_oracle_quick_irstd_tpc.json
reports/low_threshold_oracle_quick_nuaa_lca_tcd.json
reports/low_threshold_oracle_quick_nuaa_tpc_lca.json
```

## Interpretation

Lowering teacher candidate thresholds did not expand the oracle ceiling.

The best oracle remains approximately:

| Dataset | Low-threshold oracle | Previous oracle region | Meaning |
|---|---:|---:|---|
| IRSTD-1K | 70.4 | 70.3-70.4 | No new +5 headroom |
| NUAA-SIRST | 80.3 | 80.3-80.4 | No new +5 headroom |

The best valid rows also do not beat the current reliable best:

| Dataset | Current reliable best | Low-threshold quick best | Delta |
|---|---:|---:|---:|
| IRSTD-1K | 69.17 | 69.12 | -0.05 |
| NUAA-SIRST | 79.34 | 79.33 | -0.01 |

Conclusion: the current SCTransNet-family teacher pool, even at low confidence thresholds, does not contain enough hidden true-positive candidates for the requested +5 mIoU jump. Further training on the same teacher/refiner family is unlikely to solve the objective.

Next work should switch to a genuinely different candidate/model source rather than another SCTransNet-style refinement round.

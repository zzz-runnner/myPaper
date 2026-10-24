# PConv Candidate Fusion Results

Generated: 2026-06-13 UTC

## Purpose

Previous low-threshold SCTransNet-family diagnostics showed no +5 mIoU headroom inside the existing teacher pool. This run tested whether the PConv + SD Loss / official MSHNet-PConv line can act as a genuinely different candidate source.

The official PConv-MSHNet outputs were added as candidate maps inside the component-fusion framework, then evaluated with full multi-dataset train fitting:

```text
IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3
```

No checkpoint was produced. This was a candidate/fusion diagnostic only.

## Code Changes

Updated:

```text
scripts/eval_component_fusion_learner.py
```

Added optional official PConv-MSHNet candidate specs:

```text
--official_pconv_mshnet_spec name|variant|weight|preprocess|threshold|warm|base_size
```

This lets the component-fusion learner combine SCTransNet-family teachers with official PConv-MSHNet probability/candidate maps.

Launcher:

```text
scripts/launch_pconv_candidate_fusion_4gpu_tmux.sh
```

Outputs:

```text
reports/pconv_candidate_fusion_irstd_d05.json
reports/pconv_candidate_fusion_irstd_p43.json
reports/pconv_candidate_fusion_nuaa_lca_tcd_d05.json
reports/pconv_candidate_fusion_nuaa_tpc_lca_d05.json
```

## Results

| Dataset | Candidate set | Best valid row | mIoU | Pd | Fa | Oracle mIoU | Oracle Pd | Oracle Fa |
|---|---|---|---:|---:|---:|---:|---:|---:|
| IRSTD-1K | baseq + fixed refiner + PConv-D05 | baseline max | 69.39 | 95.62 | 19.85 | 72.69 | 95.62 | 4.00 |
| IRSTD-1K | baseq + fixed refiner + PConv-P43 | baseline max | 71.37 | 95.29 | 16.38 | 74.18 | 95.29 | 3.51 |
| NUAA-SIRST | LCA + TCD-A + TCD-B + PConv-D05 | baseline LCA | 79.12 | 96.58 | 17.70 | 77.64 | 97.72 | 24.15 |
| NUAA-SIRST | TPC-A + TPC-B + LCA + PConv-D05 | baseline TPC-A | 79.12 | 96.58 | 17.70 | 77.64 | 97.72 | 24.15 |

## Interpretation

PConv-P43 is the first external candidate source that materially expands IRSTD-1K headroom:

| IRSTD-1K route | mIoU |
|---|---:|
| Current reliable best before this diagnostic | 69.17 |
| PConv-P43 + strong teacher max fusion | 71.37 |
| PConv-P43 + strong teacher oracle | 74.18 |

This is not yet the requested +5 over SCTransNet official 68.03, because 71.37 is +3.34. But it is a real new direction: the oracle rose from about 70.4 to 74.18, which means PConv-P43 contributes true candidates that the SCTransNet-family teachers did not contain.

NUAA-SIRST is different. Adding PConv-D05 hurts the candidate pool:

- best valid remains the original teacher baseline around 79.12;
- PConv-including oracle falls to 77.64 because the PConv candidates introduce too many false alarms under the current component rendering/threshold setup.

## Decision

Continue PConv-P43 on IRSTD as a candidate-source/gating line. The next useful experiment should learn a stricter gate for PConv-P43 components, especially to keep its new true positives while suppressing the high false-alarm baseline max row.

Do not spend more NUAA GPU time on this exact PConv-D05 candidate. For NUAA, the next source should be a different candidate family, such as a foundation-model prompt candidate, background/clutter-specific gate, or another external architecture.

All four GPUs are idle after completion, and no bulky artifacts were created.

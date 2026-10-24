# NUAA SIRST-UAVB Component Screen Results

Generated: 2026-06-14 UTC

## Purpose

NUAA-SIRST remains below the requested +5 mIoU target. After direct SIRST-UAVB training failed, this round tested SIRST-UAVB as an additional fitting domain for component-level candidate selection.

This is aligned with the non-plain-SCTransNet route:

```text
same-domain candidate sources: TPC/LCA/TCD + official SIRST-UAV PConv
cross-domain/transfer features: background normality, target prototype anomaly
extra data domain: SIRST-UAVB
```

## Experiments

### 1. Background + Target Prototype Fast Screen

Output:

```text
reports/nuaa_sirst_uavb_bg_proto_component_fast.json
```

Fit/eval caps:

```text
max_fit_samples = 180 per dataset
max_eval_samples = 220
```

Result:

| Row | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: |
| best valid | 79.1315 | 79.4226 | 96.5779 | 17.6990e-6 |
| test oracle, not valid | 79.9378 | 80.2083 | 96.1977 | 20.3059e-6 |

Interpretation:

The target-prototype candidate did not raise the NUAA candidate ceiling. It was slower and worse than the previous SIRST-UAV PConv oracle region.

### 2. Background Normality + SIRST-UAVB Fit Screen

Output:

```text
reports/nuaa_sirst_uavb_bg_component_capped.json
```

Fit/eval caps:

```text
max_fit_samples = 220 per dataset
max_eval_samples = 300
```

Result:

| Row | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: |
| best valid | 79.1658 | 79.5628 | 96.1977 | 28.8810e-6 |
| test oracle, not valid | 80.4862 | 80.4693 | 96.9582 | 8.3693e-6 |

Interpretation:

This recovers essentially the same oracle ceiling seen before with SIRST-UAV official PConv weights:

```text
about 80.49 mIoU oracle
```

It does not move toward the NUAA target:

```text
target = 82.64 mIoU
gap from oracle = about 2.15 mIoU
```

The valid result remains below the current reliable NUAA best:

```text
current reliable best = 79.34 mIoU
```

## Stopped Work

An uncapped background+prototype run and an uncapped background-only run were both stopped before output because CPU feature collection was too slow for the signal observed in capped screens.

No bulky artifact was kept.

## Cleanup / Current State

Persistent new artifacts are small:

```text
108K reports/nuaa_sirst_uavb_bg_component_capped.json
92K  reports/nuaa_sirst_uavb_bg_proto_component_fast.json
```

No active training/eval process remains. GPUs are idle.

## Conclusion

SIRST-UAVB has now been tested in three roles:

1. direct added training data for SCTransNet-family refiners;
2. from-scratch MSHNet/P43 official-style source training;
3. extra fitting domain for component/background/prototype selection.

None has moved NUAA-SIRST beyond the current 79.34 reliable best or raised the oracle ceiling beyond about 80.49. Therefore, the NUAA bottleneck is still lack of a genuinely stronger dense candidate source, not lack of selector fitting data.

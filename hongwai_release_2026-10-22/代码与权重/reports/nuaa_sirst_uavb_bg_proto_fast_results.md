# NUAA SIRST-UAVB Background/Prototype Fast Screen

Generated: 2026-06-14 UTC

## Purpose

NUAA-SIRST remains the blocker. This fast screen tested whether adding SIRST-UAVB into the component-fit pool plus background-normality and target-prototype anomaly cues raises the candidate-selection ceiling.

Fit datasets:

```text
IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3, SIRST-UAVB
```

Candidate sources:

```text
TPC-A, TPC-B, LCA, TCD
SIRST-UAV official PConv D-0.5, P43
background-normality candidate/feature
target-prototype candidate/feature
```

This was a capped screen:

```text
max_fit_samples = 180 per dataset
max_eval_samples = 220
```

It is not a final full-test result.

## Result

Output:

```text
reports/nuaa_sirst_uavb_bg_proto_component_fast.json
```

Best valid row on the capped NUAA eval subset:

| Kind | Mode | mIoU | nIoU | Pd | Fa |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | LCA | 79.1315 | 79.4226 | 96.5779 | 17.6990e-6 |

Best oracle row on the capped subset:

| Oracle | mIoU | Pd | Fa |
| --- | ---: | ---: | ---: |
| test oracle, not valid | 79.94 | 96.20 | 20.31e-6 |

## Interpretation

This screen did not show a useful NUAA signal. The valid row is below the current reliable full-test NUAA best:

```text
current reliable NUAA best: 79.34 mIoU
```

More importantly, the capped-subset oracle did not rise toward the required target:

```text
NUAA target: 82.64 mIoU
```

The target-prototype/background-as-candidate combination is therefore not worth expanding to a slow full-data run in its current form.

## Cleanup

The earlier uncapped prototype run was stopped before output because it spent too long in CPU feature fitting. No bulky artifact was kept.

This completed fast screen produced only a small JSON file:

```text
92K reports/nuaa_sirst_uavb_bg_proto_component_fast.json
```

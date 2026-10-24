# NUAA SIRST-UAVB Background-Fusion Attempt

Generated: 2026-06-15 UTC

## Purpose

This tested whether the newly converted SIRST-UAVB dataset can improve the current strongest NUAA BGM+RPCANet fusion through a wider background-normality fit distribution.

Unlike the earlier SIRST-UAVB SCTransNet-family refiner run, this did not retrain the deep model. It only added SIRST-UAVB to the multi-dataset background/anomaly candidate fitting pool used by the fixed fusion evaluator.

## Data

SIRST-UAVB is available in project layout:

```text
datasets/SIRST-UAVB
train: 2400
test: 600
```

Fit datasets used in this run:

```text
IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3, SIRST-UAVB
```

Fit cap:

```text
max_fit_samples = 220 per dataset
```

NUAA evaluation:

```text
NUAA-SIRST full test, 214 samples
```

## Command Output

Output JSON:

```text
reports/nuaa_fixed_bgm_rpcanet_fusion_sirst_uavb_bg.json
```

Best row:

| kind | threshold | gamma | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| with_bg_anomaly | 0.50 | 1.00 | 81.0136 | 81.5680 | 97.7186 | 8.8495e-6 |

Weights:

| Source | Weight |
| --- | ---: |
| `tpc_a` | 0.14 |
| `tpc_b` | 0.14 |
| `lca` | 0.14 |
| `bgm_nuaa` | 0.22 |
| `rpcanet_sirst_s3` | 0.28 |
| `bg_anomaly` | 0.08 |

## Comparison

Previous current best without SIRST-UAVB in the background fit pool:

```text
mIoU 81.0136
nIoU 81.5680
Pd   97.7186
Fa   8.8495e-6
```

This SIRST-UAVB background-fit run:

```text
mIoU 81.0136
nIoU 81.5680
Pd   97.7186
Fa   8.8495e-6
```

Decision:

```text
No measurable gain. This exactly reproduces the existing best row.
```

## Interpretation

Adding SIRST-UAVB to the background-normality fit pool does not improve the current BGM+RPCANet fixed fusion. Together with the earlier SIRST-UAVB refiner early-stop result, this suggests SIRST-UAVB is not useful here as simple extra training/background data.

If SIRST-UAVB is revisited, it should be used to train or evaluate a genuinely new candidate source, not only as more samples for the existing refiner/fusion machinery.

## Cleanup

No checkpoint or prediction archive was created.

Retained small evidence only:

```text
reports/nuaa_fixed_bgm_rpcanet_fusion_sirst_uavb_bg.json  ~24K
reports/nuaa_sirst_uavb_background_fusion_attempt.md      ~4K
```

Current reliable bests remain:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.0136 | 82.64 | not passed |


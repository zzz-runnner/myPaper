# NUAA SIRST-UAV PConv Candidate Results

Generated: 2026-06-14 UTC

## Purpose

NUAA-SIRST is still the blocker after IRSTD-1K passed the +5 target. The existing NUAA background-normality component pool has a valid best of 79.34 mIoU and a test oracle around 80.35 mIoU, which is far below the requested 82.64 target.

This run tested a new same-domain external candidate source from the pasted PConv/SD-loss material:

```text
SIRST-UAVB-trained MSHNet/PConv official weights
```

The goal was to see whether SIRST-UAVB weights introduce new NUAA true-target candidates beyond the TPC/LCA/TCD teacher family.

## Setup

Unpacked official weights:

```text
external/PConv-SDloss-Data/weight_extracted/MSHNet-SIRST_UAV-D-0.5/weight.pkl
external/PConv-SDloss-Data/weight_extracted/MSHNet-SIRST_UAV-P43/weight.pkl
```

Component-fusion candidates:

```text
tpc_a
tpc_b
lca
uav_d05
uav_p43
background-normality features
```

Training/fitting data:

```text
IRSTD-1K train
NUAA-SIRST train
NUDT-SIRST train
SIRST3 train
```

Test data:

```text
NUAA-SIRST test
```

Command output:

```text
reports/nuaa_sirst_uav_pconv_bg_component.json
```

No training checkpoint was written.

## Result

Current reference:

| Method | Valid mIoU | Oracle mIoU |
| --- | ---: | ---: |
| previous background-normality component pool | 79.34 | 80.35 |

SIRST-UAV PConv candidate result:

| row | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: |
| best valid train-fit eval | 79.2495 | 79.5726 | 96.5779 | 17.9734e-6 |
| best test oracle, not valid | 80.4862 | 80.4693 | 96.9582 | 8.3693e-6 |

## Interpretation

The SIRST-UAVB PConv weights add a small amount of new oracle headroom:

```text
80.35 -> 80.4862
```

But the valid learner does not convert this headroom:

```text
79.34 current best -> 79.2495 with SIRST-UAV PConv
```

More importantly, even the oracle upper bound is still far below the requested NUAA +5 target:

```text
82.64 - 80.4862 = about 2.15 mIoU
```

So SIRST-UAV PConv is useful evidence but not sufficient as the NUAA breakthrough source. The next NUAA step needs a stronger source than the current teacher/PConv/component pool: either additional NUAA-like data, stronger external pretrained weights, or a foundation-feature candidate generator that materially raises oracle headroom beyond 82.64.

## Cleanup

No failed bulky checkpoint was created. The only new persistent files are:

```text
reports/nuaa_sirst_uav_pconv_bg_component.json
reports/nuaa_sirst_uav_pconv_candidate_results.md
external/PConv-SDloss-Data/weight_extracted/MSHNet-SIRST_UAV-*/weight.pkl
```

The extracted SIRST-UAV official weights are retained as reusable external assets.

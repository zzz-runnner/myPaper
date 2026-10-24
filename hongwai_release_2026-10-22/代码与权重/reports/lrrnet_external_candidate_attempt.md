# LRRNet External Candidate Attempt

Generated: 2026-06-14 UTC

## Purpose

This tested a new same-domain but non-SCTransNet source: LRRNet, a TGRS 2025 low-rank background representation method. The motivation was that NUAA-SIRST currently appears limited by false alarms/background structure, and LRRNet explicitly models background rather than only target saliency.

External code:

```text
external/LRRNet
```

Official checkpoint included in the repository:

```text
external/LRRNet/checkpoints/IRDST_1K_miou_0.7239356450020329.pkl
```

New local wrapper:

```text
scripts/eval_lrrnet_local.py
```

## Smoke Result

First 20 test samples per dataset:

| Dataset | Best threshold | mIoU | Pd | Fa |
|---|---:|---:|---:|---:|
| IRSTD-1K | 0.20 | 73.75 | 94.12 | 4.77e-6 |
| NUAA-SIRST | 0.40 | 56.93 | 87.88 | 9.25e-5 |

The IRSTD smoke looked promising, so a full test evaluation was run.

## Full Test Result

| Dataset | Best threshold | mIoU | Pd | Fa |
|---|---:|---:|---:|---:|
| IRSTD-1K | 0.80 | 65.42 | 92.93 | 2.11e-5 |
| NUAA-SIRST | 0.50 | 67.25 | 94.68 | 4.52e-5 |

Outputs:

```text
reports/lrrnet_local_smoke.json
reports/lrrnet_local_full_test.json
```

## Interpretation

LRRNet is runnable and conceptually relevant, but the provided IRSTD-1K checkpoint did not transfer into a competitive full-test result in this local evaluation path.

Current reliable best remains:

| Dataset | Current reliable best | mIoU |
|---|---|---:|
| IRSTD-1K | validation-selected dual-PConv fusion | 72.47-72.48 |
| NUAA-SIRST | background-normality component fusion | 79.34 |

LRRNet does not improve either target dataset as a standalone candidate. It may still be useful as an architectural idea or a low-rank auxiliary module, but using its released IRSTD checkpoint as a direct dense candidate is not enough.

No failed training checkpoint was created. The 14 MB official LRRNet checkpoint is kept as a reusable external asset.

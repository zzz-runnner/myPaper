# LRRNet Multi-Dataset 4-GPU Probe Results

Generated: 2026-06-14 UTC

## Purpose

This followed up the LRRNet external-candidate check by training/finetuning LRRNet on the full local multi-dataset pool rather than only evaluating the released IRSTD checkpoint.

This is a same-domain non-SCTransNet line: low-rank background reconstruction plus target residual segmentation.

## New Script

Implemented:

```text
scripts/train_lrrnet_multidata.py
```

It trains on:

```text
IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3
```

and evaluates on:

```text
IRSTD-1K, NUAA-SIRST
```

The loss follows LRRNet's original structure:

```text
SoftIoU(T, mask) + SoftIoU-L1(T, mask) + mse_weight * MSE(D, image)
```

with an added optional weighted BCE term for small-target imbalance.

## GPU / Memory Use

The updated user requirement asked to use the four RTX 2080 Ti cards and try to occupy more than 9 GB VRAM.

Memory tests:

| Batch | Patch | Result |
|---:|---:|---|
| 8 | 512 | OOM at about 10.5 GB |
| 7 | 512 | OOM at about 10.5 GB |
| 6 | 512 | stable |

The final probe used batch size 6. During training all four cards reported approximately:

```text
10622 MiB used, 99% GPU utilization
```

## Runs

Four tmux jobs were launched:

| GPU | Run | Init | LR | Norm dataset |
|---:|---|---|---:|---|
| 0 | `rand_nuaa_b6_lr1e4_seed61` | random | 1e-4 | NUAA-SIRST |
| 1 | `rand_irstd_b6_lr3e4_seed62` | random | 3e-4 | IRSTD-1K |
| 2 | `finetune_nuaa_b6_lr2e5_seed63` | official LRRNet IRSTD checkpoint | 2e-5 | NUAA-SIRST |
| 3 | `finetune_irstd_b6_lr5e5_seed64` | official LRRNet IRSTD checkpoint | 5e-5 | IRSTD-1K |

The planned length was 6 epochs. The jobs were stopped after epoch 4 because all branches were far below the current best and the trend did not justify more GPU time.

## Results

| Run | Epoch | IRSTD-1K mIoU | NUAA-SIRST mIoU |
|---|---:|---:|---:|
| `finetune_irstd_b6_lr5e5_seed64` | 2 | 62.71 | 67.75 |
| `finetune_irstd_b6_lr5e5_seed64` | 4 | 61.95 | 68.02 |
| `finetune_nuaa_b6_lr2e5_seed63` | 2 | 64.59 | 68.36 |
| `finetune_nuaa_b6_lr2e5_seed63` | 4 | 63.30 | 69.73 |
| `rand_irstd_b6_lr3e4_seed62` | 2 | 47.73 | 59.60 |
| `rand_irstd_b6_lr3e4_seed62` | 4 | 56.09 | 70.84 |
| `rand_nuaa_b6_lr1e4_seed61` | 2 | 44.79 | 57.88 |
| `rand_nuaa_b6_lr1e4_seed61` | 4 | 52.69 | 66.92 |

Current reliable best for comparison:

| Dataset | Current reliable best | mIoU |
|---|---|---:|
| IRSTD-1K | validation-selected dual-PConv fusion | 72.47-72.48 |
| NUAA-SIRST | background-normality component fusion | 79.34 |

## Cleanup

The four weak checkpoints were deleted:

```text
runs/lrrnet_multidata_probe/*/best.pth.tar
```

Only small logs remain:

```text
runs/lrrnet_multidata_probe/*/train.log
reports/lrrnet_probe_logs/*.log
```

## Interpretation

LRRNet is runnable and the full multi-dataset training wrapper works. The run also satisfied the 4-GPU high-VRAM usage requirement during the probe.

However, this LRRNet training/finetuning line did not approach the existing best methods:

- best IRSTD-1K in this probe: 64.59, far below 72.48;
- best NUAA-SIRST in this probe: 70.84, far below 79.34.

Decision: do not continue this exact LRRNet standalone training setup. If the low-rank background idea is reused, it should be transplanted as an auxiliary reconstruction/background-consistency branch into a stronger existing teacher/fusion framework rather than used as a standalone detector.

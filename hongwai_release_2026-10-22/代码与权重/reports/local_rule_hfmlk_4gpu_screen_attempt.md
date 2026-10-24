# NUAA local-rule HFM-LK 4-GPU screen

Date: 2026-06-16

## Purpose

Run a four-branch screen of the HFM-LK large-kernel/frequency residual refiner using the current NUAA local-rule teacher. This continues the non-SCTransNet-parameter route:

- multi-dataset cache training;
- HFM/DFAware-style large-kernel context;
- high-frequency residual gating;
- NUAA official-Mamba local-rule teacher.

The current NUAA best before this run was:

```text
residual pixel calibrator + component filter
mIoU: 82.2928
threshold: 0.498
min_area: 3
```

The +5 target remains:

```text
82.6400 mIoU
```

## Launch

Added a simple wrapper to avoid shell quoting problems in tmux:

```text
scripts/run_local_rule_hfmlk_one.sh
```

The four-GPU launch script now calls that wrapper:

```text
scripts/launch_local_rule_hfmlk_nuaa_4gpu_screen.sh
```

Common settings:

```text
train caches:
  IRSTD-1K train
  NUAA-SIRST train
  NUDT-SIRST train
  SIRST3 train
test cache:
  NUAA-SIRST test
teacher_mode: local_rule
epochs: 3
batch_size: 2
patch_size: 512
base_channels: 48
max_train_samples_per_cache: 128
weighted_dice: true
amp: true
```

During training, each RTX 2080 Ti used about:

```text
8782-8784 MiB / 11264 MiB
GPU util: up to 98%
```

This is just under 9 GiB in binary units but about 9.2 GB in decimal units.

## Results

| branch | seed | residual | anchor | extra | best epoch | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| s1011_r004_a020 | 20260621 | 0.040 | 0.20 | none | 2 | 0.496 | 82.2703 | 82.0182 | 96.9582 | 9.3297e-6 |
| s1014_r005_a022 | 20260624 | 0.050 | 0.22 | uncertainty 0.20 | 2 | 0.498 | 82.2651 | 82.0234 | 96.9582 | 9.6041e-6 |
| s1012_r006_a018 | 20260622 | 0.060 | 0.18 | none | 2 | 0.500 | 82.2503 | 82.0186 | 96.9582 | 9.1239e-6 |
| s1013_r003_a024 | 20260623 | 0.030 | 0.24 | miss 0.10 | 2 | 0.484 | 82.2429 | 82.0599 | 96.5779 | 11.5935e-6 |

Best row:

```text
s1011_r004_a020 epoch 2
mIoU: 82.2703
```

Comparison:

```text
current NUAA best:       82.2928
best 4-GPU HFM-LK row:   82.2703
delta:                  -0.0225
remaining to +5 target:  0.3472
```

## Cleanup

All four branches failed to beat the retained NUAA best, so their checkpoints were deleted:

```text
runs/local_rule_hfmlk_nuaa_4gpu_screen/*/best.pth.tar
```

The run directory now keeps only logs and summary files and is about `88K`.

## Interpretation

This branch is close to the current NUAA residual-pixel result but still below it. The best rows again occur at epoch 2 and remain in the same narrow 82.24-82.27 band, suggesting that local-rule HFM-LK distillation is learning a similar dense correction but not adding enough new evidence to close the remaining 0.3472 mIoU gap.

Do not extend this exact screen blindly. The next NUAA attempt should either introduce a new dense source or change the training target/feature evidence, not only add epochs to this local-rule HFM-LK setup.

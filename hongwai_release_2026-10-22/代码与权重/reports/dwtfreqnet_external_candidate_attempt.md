# DWTFreqNet External Candidate Attempt

Generated: 2026-06-14 UTC

## Purpose

This tested a new non-SCTransNet infrared small-target method from the pasted method directions: frequency/high-frequency target-clutter modeling. DWTFreqNet is a 2025 TGRS method using wavelet-driven frequency matching and saliency-difference optimization.

External code:

```text
external/DWTFreqNet
```

New wrapper:

```text
scripts/train_dwtfreq_multidata.py
```

The wrapper adapts DWTFreqNet to the local prepared datasets and trains on the multi-dataset pool:

```text
IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3
```

It evaluates:

```text
IRSTD-1K, NUAA-SIRST
```

and saves only one best checkpoint during a run.

## Compatibility Smoke

A 1-epoch tiny smoke with 8 samples per dataset completed successfully. This verified:

- DWTFreqNet imports and forwards in the local environment.
- Backpropagation works.
- Evaluation through the local metric path works.

The smoke checkpoint was deleted because the metric was unusable and the checkpoint was 150M.

## Limited Multi-Dataset Probe

Command shape:

```text
python scripts/train_dwtfreq_multidata.py \
  --epochs 10 \
  --batch_size 2 \
  --lr 0.001 \
  --norm_dataset NUAA-SIRST \
  --eval_interval 2 \
  --eval_max_samples 80 \
  --max_train_samples_per_dataset 100 \
  --output_dir runs/dwtfreq_limited_probe/norm_nuaa_lr1e3_seed51 \
  --seed 51 \
  --device cuda:0
```

This used 100 train samples from each of the four datasets, so 400 training samples total. It was a feasibility probe before spending full multi-dataset GPU time.

Observed results:

| Epoch | IRSTD-1K mIoU | IRSTD Pd | IRSTD Fa | NUAA-SIRST mIoU | NUAA Pd | NUAA Fa |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | 10.43 | 66.94 | 8.36e-4 | 19.38 | 64.22 | 8.10e-4 |
| 6 | 19.09 | 79.03 | 3.75e-4 | 39.90 | 69.72 | 1.53e-4 |
| 8 | 21.33 | 87.90 | 3.87e-4 | 42.43 | 76.15 | 2.25e-4 |
| 10 | 21.65 | 90.32 | 4.69e-4 | 40.68 | 77.98 | 3.06e-4 |

Best observed:

| Dataset | Best probe mIoU | Current reliable best |
|---|---:|---:|
| IRSTD-1K | 21.65 | 72.43 |
| NUAA-SIRST | 42.43 | 79.34 |

## Runtime Notes

DWTFreqNet has about 37.4M parameters. With batch size 2 it used about 5.85G on one RTX 2080 Ti. A full epoch over all four datasets was very slow: the full probe was stopped after more than six minutes without completing epoch 1. The limited 400-sample probe ran at roughly 56-57 seconds per epoch.

Attempts to launch four background `nohup` jobs produced empty logs and no stable running jobs, while foreground execution worked. Because the limited foreground probe was far below current best, the launch issue was not pursued further.

## Cleanup

Deleted weak 150M checkpoints from:

```text
runs/dwtfreq_smoke
runs/dwtfreq_limited_probe
```

Only logs and summaries remain.

## Interpretation

DWTFreqNet is a runnable and relevant frequency-domain external candidate, but in this local multi-dataset adaptation it starts far below the current teacher/fusion region. The early learning curve does not justify a full multi-dataset long run right now.

Decision: keep the code wrapper and report, but do not spend more GPU time on this exact from-scratch DWTFreqNet setup unless official weights are obtained or its frequency modules are transplanted into a stronger existing teacher/student.

# Wavelet-Anchor Refiner 4GPU Results

Generated: 2026-06-14 UTC

## Purpose

This run tested a PAL-inspired frequency/wavelet module inside the stronger teacher-anchor residual refiner, instead of only tuning SCTransNet training parameters.

Idea:

```text
teacher-anchor residual refiner
+ Haar wavelet high/low-frequency gates after encoder blocks
+ full multi-dataset training
+ hard-clutter / missed-target weighting for NUAA variants
```

The goal was to check whether a frequency-aware module could improve the existing IRSTD-1K and NUAA-SIRST best results.

## Code

Added:

```text
scripts/train_wavelet_anchor_refiner_from_cache.py
scripts/launch_wavelet_anchor_4gpu_tmux.sh
```

The model adds fixed Haar-filter feature gates at multiple stages of the anchor residual U-Net. It reuses the local multi-dataset teacher caches and local full-test metric path.

## Training Setup

Hardware:

```text
4 x RTX 2080 Ti
about 9.1 GB per GPU
99-100% utilization while running
```

Common settings:

```text
patch_size = 512
batch_size = 4
base_channels = 56
AMP enabled
planned epochs = 8
eval interval = 2
```

The first batch-5 launch OOM'd near the 2080 Ti limit, so the retained run used batch size 4.

## Runs

| GPU | run | eval dataset | residual | anchor | lr | notes |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 0 | `irstd_wavelet_s401` | IRSTD-1K | 0.35 | 0.020 | 0.00020 | IRSTD-oriented teacher cache |
| 1 | `irstd_wavelet_s402` | IRSTD-1K | 0.45 | 0.030 | 0.00018 | IRSTD-oriented teacher cache |
| 2 | `nuaa_wavelet_s403` | NUAA-SIRST | 0.35 | 0.020 | 0.00018 | hard clutter 1.8, miss 2.8 |
| 3 | `nuaa_wavelet_s404` | NUAA-SIRST | 0.45 | 0.030 | 0.00020 | hard clutter 1.2, miss 3.6 |

Training data:

```text
IRSTD-1K train
NUAA-SIRST train
NUDT-SIRST train
SIRST3 train
```

Evaluation:

```text
IRSTD-1K full test for IRSTD runs
NUAA-SIRST full test for NUAA runs
```

## Results

Stopped after epoch 4 because all runs remained below the current retained bests.

| run | epoch | mIoU | nIoU | Pd | Fa | threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `irstd_wavelet_s401` | 2 | 72.3003 | 66.4624 | 94.6128 | 11.9945e-6 | 0.38 |
| `irstd_wavelet_s401` | 4 | 72.7490 | 67.1144 | 94.6128 | 11.9755e-6 | 0.38 |
| `irstd_wavelet_s402` | 2 | 72.7023 | 67.1411 | 94.6128 | 11.7857e-6 | 0.38 |
| `irstd_wavelet_s402` | 4 | 72.7745 | 67.2828 | 94.6128 | 11.4820e-6 | 0.38 |
| `nuaa_wavelet_s403` | 2 | 78.9281 | 79.3619 | 96.1977 | 30.1158e-6 | 0.58 |
| `nuaa_wavelet_s403` | 4 | 79.1014 | 79.5172 | 96.1977 | 30.5960e-6 | 0.58 |
| `nuaa_wavelet_s404` | 2 | 79.1792 | 79.6035 | 96.5779 | 16.6700e-6 | 0.62 |
| `nuaa_wavelet_s404` | 4 | 79.2212 | 79.6573 | 96.5779 | 16.3956e-6 | 0.62 |

Reference bests before this run:

| Dataset | retained best | target |
| --- | ---: | ---: |
| IRSTD-1K | 73.0367 | 73.03 |
| NUAA-SIRST | 79.5487 valid / 80.6827 oracle | 82.64 |

Best wavelet-anchor results:

```text
IRSTD-1K: 72.7745, below retained best 73.0367
NUAA-SIRST: 79.2212, below retained best 79.5487
```

## Cleanup

Stopped tmux sessions:

```text
wav_irstd_s401
wav_irstd_s402
wav_nuaa_s403
wav_nuaa_s404
```

Deleted failed checkpoints:

```text
runs/wavelet_anchor_4gpu/*/best.pth.tar
```

Each checkpoint was about 46 MB, about 184 MB total. Retained only lightweight logs:

```text
runs/wavelet_anchor_4gpu/*/train.log
reports/wavelet_anchor_logs/*.log
```

After cleanup, all GPUs were idle again.

## Interpretation

The PAL-inspired Haar frequency gates are runnable and fit the 2080 Ti memory target, but they did not improve either dataset. The NUAA result stayed in the same 79.0-79.2 region as prior teacher-anchor variants, which supports the current diagnosis: the NUAA blocker is not solved by another small residual/gating modification on the same teacher cache.

The next high-value work should add a genuinely stronger NUAA-compatible source: either a new labeled IR small-target dataset under `/data1`, or official checkpoints from a stronger method not yet evaluable locally.

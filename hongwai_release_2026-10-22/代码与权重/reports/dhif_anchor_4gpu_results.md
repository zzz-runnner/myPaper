# DHiF Dynamic High-Frequency Anchor Refiner Results

Generated: 2026-06-14 UTC

## Purpose

This run tested a stronger frequency-domain module from a recent external IRSTD method:

```text
Dynamic High-frequency Convolution for Infrared Small Target Detection
```

The motivation matches the current blocker: both true small targets and hard clutter are local high-frequency responses, so the useful module should distinguish target high-frequency from clutter high-frequency, not merely amplify all high-frequency content.

## Assets And Code

Cloned:

```text
external/DHiF
```

The repository is small, about 1.7 MB. It has no usable checkpoint locally, but exposes the `DHiF` module in:

```text
external/DHiF/model/basic.py
```

The repository's package init imports stale symbols, so the local wrapper loads `basic.py` directly with `importlib`.

Added:

```text
scripts/train_dhif_anchor_refiner_from_cache.py
scripts/launch_dhif_anchor_4gpu_tmux.sh
```

The model keeps the successful teacher-anchor residual framework and replaces only selected early encoder convolutions with DHiF:

```text
DHiF-anchor residual refiner
+ full multi-dataset cached teacher training
+ optional hard-clutter / missed-target weighting
```

This avoids fully replacing the calibrated teacher probability maps while still testing a genuine dynamic high-frequency module.

## Smoke

A small NUAA smoke run passed:

```text
max_train_samples_per_cache = 8
eval_max_samples = 8
patch_size = 128
base_channels = 16
dhif_stages = e1
```

Smoke checkpoint was deleted after verifying forward/backward/eval.

## Full 4GPU Setup

Training data for each run:

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

Common settings:

```text
patch_size = 512
batch_size = 4
base_channels = 56
AMP enabled
planned epochs = 8
eval interval = 2
```

GPU use:

```text
GPU0/GPU2 e1 variants: about 8.4-8.5 GB
GPU1/GPU3 e1,e2 variants: about 9.4-9.5 GB
utilization: 99-100%
```

The e1,e2 branches met the requested 9GB+ target. Batch size was not increased further to avoid OOM on the heavier DHiF branches.

## Runs

| GPU | run | eval dataset | DHiF stages | residual | anchor | lr |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 0 | `irstd_dhif_e1_s501` | IRSTD-1K | e1 | 0.35 | 0.020 | 0.00020 |
| 1 | `irstd_dhif_e12_s502` | IRSTD-1K | e1,e2 | 0.40 | 0.025 | 0.00018 |
| 2 | `nuaa_dhif_e1_s503` | NUAA-SIRST | e1 | 0.35 | 0.020 | 0.00018 |
| 3 | `nuaa_dhif_e12_s504` | NUAA-SIRST | e1,e2 | 0.45 | 0.030 | 0.00020 |

NUAA branches also used hard clutter / missed target / uncertainty weighted dice.

## Results

Stopped after epoch 4 because the full-test metrics stayed below the retained bests.

| run | epoch | mIoU | nIoU | Pd | Fa | threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `irstd_dhif_e1_s501` | 2 | 72.6255 | 67.0223 | 94.6128 | 11.9945e-6 | 0.38 |
| `irstd_dhif_e1_s501` | 4 | 72.8004 | 67.2113 | 94.6128 | 11.6529e-6 | 0.38 |
| `irstd_dhif_e12_s502` | 2 | 72.3660 | 66.7824 | 94.9495 | 12.8865e-6 | 0.38 |
| `irstd_dhif_e12_s502` | 4 | 72.0772 | 66.1807 | 94.9495 | 12.6967e-6 | 0.38 |
| `nuaa_dhif_e1_s503` | 2 | 79.1613 | 79.5054 | 96.5779 | 17.1502e-6 | 0.58 |
| `nuaa_dhif_e1_s503` | 4 | 78.9804 | 79.4916 | 96.1977 | 29.7042e-6 | 0.62 |
| `nuaa_dhif_e12_s504` | 2 | 79.2230 | 79.7525 | 96.5779 | 16.6700e-6 | 0.62 |
| `nuaa_dhif_e12_s504` | 4 | 79.2038 | 79.6625 | 96.5779 | 16.6014e-6 | 0.62 |

Reference retained bests:

| Dataset | retained best | +5 target |
| --- | ---: | ---: |
| IRSTD-1K | 73.0367 | 73.03 |
| NUAA-SIRST | 79.5487 valid / 80.6827 oracle | 82.64 |

Best DHiF-anchor:

```text
IRSTD-1K: 72.8004, below 73.0367
NUAA-SIRST: 79.2230, below 79.5487
```

## Cleanup

Stopped sessions:

```text
dhif_irstd_s501
dhif_irstd_s502
dhif_nuaa_s503
dhif_nuaa_s504
```

Deleted failed checkpoints:

```text
runs/dhif_anchor_4gpu/*/best.pth.tar
```

Each checkpoint was about 38 MB, about 154 MB total. Retained only lightweight logs:

```text
runs/dhif_anchor_4gpu/*/train.log
reports/dhif_anchor_logs/*.log
```

After cleanup, all GPUs were idle.

## Interpretation

DHiF is a real dynamic high-frequency module and is runnable inside the teacher-anchor framework, but it did not break the NUAA plateau. The best NUAA result, 79.2230, is close to the hard-clutter and wavelet-anchor variants but still below the current valid best 79.5487.

This strengthens the current diagnosis: within the existing teacher/cache distribution, replacing or augmenting early convolutions with high-frequency modules is not enough to reach the requested 82.64 NUAA target. The next useful progress likely requires a new strong candidate source or additional NUAA-like labeled data, not another small module on the same cached teachers.

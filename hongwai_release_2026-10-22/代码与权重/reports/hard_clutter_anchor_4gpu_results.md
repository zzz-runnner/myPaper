# Hard-Clutter Anchor Refiner 4GPU Results

Generated: 2026-06-14 UTC

## Purpose

NUAA-SIRST remained oracle-limited under the previous candidate pool. This run tested a more deliberate full multi-dataset training variant rather than SCTransNet hyperparameter tuning.

The idea was:

```text
teacher-anchored residual refiner
+ high-frequency/background-residual image cues
+ hard-clutter weighted loss
+ missed-target weighted loss
+ teacher-uncertainty weighting
```

The goal was to suppress teacher false positives on clutter while still allowing the residual branch to recover targets missed by the teacher.

## Code Change

Extended:

```text
scripts/train_teacher_anchor_refiner_from_cache.py
```

with optional arguments:

```text
--hard_clutter_weight
--hard_clutter_threshold
--miss_weight
--miss_threshold
--uncertainty_weight
--weighted_dice
```

Defaults keep the previous behavior unchanged.

Added launcher:

```text
scripts/launch_hard_clutter_anchor_4gpu_tmux.sh
```

## Training Setup

All four jobs trained on full NUAA-oriented multi-dataset caches:

```text
IRSTD-1K train
NUAA-SIRST train
NUDT-SIRST train
SIRST3 train
```

Evaluation:

```text
NUAA-SIRST full test
```

Common settings:

```text
patch_size = 512
batch_size = 5
base_channels = 64
epochs planned = 10
eval interval = 2
teacher mode = avg
```

GPU use:

```text
4 x RTX 2080 Ti
about 10.78 GB per GPU
100% utilization during training
```

This met the requested 9GB+ per-card utilization.

## Runs

| GPU | run | residual | anchor | hard clutter | miss | uncertainty | pos weight | lr |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `nuaa_hard_a_s201` | 0.40 | 0.025 | 1.6 | 2.4 | 0.6 | 28 | 0.00022 |
| 1 | `nuaa_hard_b_s202` | 0.35 | 0.020 | 2.2 | 3.0 | 0.9 | 32 | 0.00018 |
| 2 | `nuaa_hard_c_s203` | 0.45 | 0.030 | 1.2 | 3.6 | 0.7 | 30 | 0.00020 |
| 3 | `nuaa_hard_d_s204` | 0.30 | 0.018 | 2.8 | 2.2 | 1.1 | 34 | 0.00016 |

## Results

The jobs were stopped after epoch 5 because epoch 2 and epoch 4 full-test evaluations stayed clearly below the current NUAA best.

| run | epoch | mIoU | nIoU | Pd | Fa | threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nuaa_hard_a_s201` | 2 | 79.1324 | 79.4635 | 96.5779 | 17.4246e-6 | 0.58 |
| `nuaa_hard_a_s201` | 4 | 79.2317 | 79.6662 | 96.1977 | 29.9786e-6 | 0.62 |
| `nuaa_hard_b_s202` | 2 | 79.1172 | 79.3618 | 96.5779 | 17.6304e-6 | 0.58 |
| `nuaa_hard_b_s202` | 4 | 79.1003 | 79.5962 | 96.1977 | 29.9099e-6 | 0.62 |
| `nuaa_hard_c_s203` | 2 | 79.1376 | 79.4982 | 96.1977 | 30.3902e-6 | 0.62 |
| `nuaa_hard_c_s203` | 4 | 79.2212 | 79.6538 | 96.5779 | 16.6014e-6 | 0.62 |
| `nuaa_hard_d_s204` | 2 | 79.1627 | 79.5518 | 96.5779 | 17.2874e-6 | 0.58 |
| `nuaa_hard_d_s204` | 4 | 79.1564 | 79.5523 | 96.5779 | 17.2874e-6 | 0.58 |

Best in this run:

```text
79.2317 mIoU
```

Current NUAA best remains:

```text
79.5487 mIoU valid
80.6827 mIoU oracle
```

Requested target:

```text
82.64 mIoU
```

## Cleanup

Stopped all four tmux sessions:

```text
hca_nuaa_s201
hca_nuaa_s202
hca_nuaa_s203
hca_nuaa_s204
```

Deleted four failed checkpoints:

```text
runs/hard_clutter_anchor_4gpu/*/best.pth.tar
```

Retained only lightweight logs:

```text
runs/hard_clutter_anchor_4gpu/*/train.log
reports/hard_clutter_anchor_logs/*.log
```

After cleanup, all GPUs were idle again.

## Interpretation

Hard-clutter and missed-target weighted training did not overcome the NUAA candidate ceiling. It preserved the teacher region around 79.1-79.2 mIoU, but did not add the missing complementarity needed to approach 82.64.

The NUAA blocker remains unsolved. The evidence still points toward needing either a new strong pretrained source or additional NUAA-like labeled data, rather than more loss reweighting on the same teacher cache.

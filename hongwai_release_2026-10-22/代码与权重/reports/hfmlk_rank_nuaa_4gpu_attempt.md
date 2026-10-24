# HFM-LK Rank-Loss NUAA 4GPU Attempt

Generated: 2026-06-15 UTC

## Purpose

After output calibration on the retained `s703+BGM` map saturated around `81.3192` mIoU, this tested a training-objective change intended to alter the probability ranking itself.

This is not SCTransNet parameter tuning. It extends the existing HFM/LK + dynamic high-frequency residual refiner with a batch-level ranking objective:

```text
hard target pixels should score above high-teacher-confidence background pixels
```

## Code Changes

Modified:

```text
scripts/train_hfm_lk_anchor_refiner_from_cache.py
```

Added parameters:

```text
--rank_weight
--rank_margin
--rank_topk
```

Added script:

```text
scripts/launch_hfmlk_rank_nuaa_4gpu_tmux.sh
```

The rank loss samples low-scoring GT target pixels and high-teacher-score background pixels within each batch, then adds a softplus margin loss to push target logits above hard-clutter logits.

A single-GPU smoke test passed before the full run.

## 4GPU Full Multi-Dataset Setup

Training data:

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

Branches:

```text
nuaa_rank_strong_s921
nuaa_rank_strong_s922
nuaa_rank_weighted_s923
nuaa_rank_weighted_s924
```

Observed GPU utilization:

```text
GPU0: 9054 MB, 97%
GPU1: 10456 MB, 97-98%
GPU2: 9054 MB, 97%
GPU3: 10454 MB, 97%
```

This met the requested four-card / 9GB+ utilization target.

## Results

Stopped after epoch 1 because all branches were below the retained NUAA best `81.3192`, and also below the earlier non-rank HFM-LK starting region.

| run | epoch | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nuaa_rank_strong_s921` | 1 | 0.54 | 80.9926 | 81.5222 | 98.0989 | 7.3403e-6 |
| `nuaa_rank_strong_s922` | 1 | 0.56 | 81.0909 | 81.6355 | 98.0989 | 7.3403e-6 |
| `nuaa_rank_weighted_s923` | 1 | 0.54 | 81.0609 | 81.6004 | 98.0989 | 7.4089e-6 |
| `nuaa_rank_weighted_s924` | 1 | 0.54 | 81.0489 | 81.6157 | 98.0989 | 7.4775e-6 |

Best rank-loss branch:

```text
nuaa_rank_strong_s922 epoch 1 mIoU = 81.0909
```

Comparison:

```text
current retained NUAA best: 81.3192
best rank-loss result:     81.0909
delta:                     -0.2283
```

## Cleanup

Stopped tmux sessions:

```text
hfmlk_rank_nuaa_a
hfmlk_rank_nuaa_b
hfmlk_rank_nuaa_c
hfmlk_rank_nuaa_d
```

Deleted failed checkpoints:

```text
runs/hfmlk_rank_nuaa_4gpu/*/best.pth.tar
```

Retained only lightweight logs; run directory is about 52 KB.

## Conclusion

The rank-loss objective did not improve NUAA. It likely over-regularized toward conservative ranking and did not change the dense probability map in the needed direction.

Current reliable status remains:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.3192 | 82.64 | not passed |

Next useful work should prioritize a stronger dense candidate source or a different architecture/source pool, rather than more calibration or this rank-loss variant on the same teacher pool.

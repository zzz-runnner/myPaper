# Teacher-Anchored Residual Refiner Probe

Generated: 2026-06-14 UTC

## Purpose

This tested a module-combination idea rather than SCTransNet parameter tuning:

```text
strong teacher cache + image high-frequency cues + low-rank background residual proxy + bounded residual refiner
```

The model does not predict a mask from scratch. It takes the teacher prediction logit as an anchor and learns only a small bounded residual:

```text
final_logit = teacher_logit + residual_scale * tanh(refiner(features))
```

Input features include:

- robust-normalized infrared image;
- local contrast / local-z at multiple window sizes;
- top-hat and DoG/Laplacian high-frequency cues;
- a cheap low-rank/background proxy from downsample-reconstruct residual;
- teacher avg/max/min/std and individual teacher maps.

This combines the current strongest cached detector family with the low-rank-background idea tested through LRRNet, but keeps the strong teacher as the baseline anchor.

## New Script

Implemented:

```text
scripts/train_teacher_anchor_refiner_from_cache.py
```

## GPU / Memory Use

Memory tests with `base_channels=64`, `patch_size=512`:

| Batch | Result |
|---:|---|
| 6 | OOM around 10.8 GB |
| 5 without allocator setting | OOM around 10.7 GB |
| 5 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | stable |

The final 4-GPU probe used batch size 5 and occupied about:

```text
10.6-10.8 GB per RTX 2080 Ti, with high GPU utilization
```

## Runs

Four tmux jobs were launched:

| GPU | Run | Dataset | Teacher cache | residual scale | anchor weight |
|---:|---|---|---|---:|---:|
| 0 | `irstd_r05_a002_s71` | IRSTD-1K | baseq + refiner + PConv-D05 + PConv-P43 | 0.5 | 0.02 |
| 1 | `irstd_r10_a005_s72` | IRSTD-1K | baseq + refiner + PConv-D05 + PConv-P43 | 1.0 | 0.05 |
| 2 | `nuaa_r05_a002_s73` | NUAA-SIRST | TPC-A + TPC-B + LCA | 0.5 | 0.02 |
| 3 | `nuaa_r10_a005_s74` | NUAA-SIRST | TPC-A + TPC-B + LCA | 1.0 | 0.05 |

## Results

| Run | Epoch | mIoU | nIoU | Pd | Fa | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| `irstd_r05_a002_s71` | 2 | 72.49 | 67.12 | 94.61 | 1.23e-5 | 0.35 |
| `irstd_r05_a002_s71` | 4 | 72.62 | 67.33 | 94.61 | 1.23e-5 | 0.35 |
| `irstd_r10_a005_s72` | 2 | 68.16 | 67.91 | 93.60 | 1.10e-5 | 0.50 |
| `irstd_r10_a005_s72` | 4 | 68.44 | 67.86 | 93.60 | 9.77e-6 | 0.55 |
| `nuaa_r05_a002_s73` | 2 | 79.15 | 79.50 | 96.58 | 1.71e-5 | 0.60 |
| `nuaa_r05_a002_s73` | 4 | 79.20 | 79.51 | 96.58 | 1.72e-5 | 0.60 |
| `nuaa_r10_a005_s74` | 2 | 79.12 | 79.54 | 96.58 | 1.71e-5 | 0.55 |
| `nuaa_r10_a005_s74` | 4 | 79.05 | 79.26 | 96.58 | 1.73e-5 | 0.60 |

## Comparison

| Dataset | Previous strict best | This probe best | Delta | +5 target |
|---|---:|---:|---:|---:|
| IRSTD-1K | 72.48 | 72.62 | +0.14 | 73.03 |
| NUAA-SIRST | 79.34 | 79.20 | -0.14 | 82.64 |

## Cleanup

The three weak checkpoints were deleted:

```text
runs/anchor_refiner_probe/irstd_r10_a005_s72/best.pth.tar
runs/anchor_refiner_probe/nuaa_r05_a002_s73/best.pth.tar
runs/anchor_refiner_probe/nuaa_r10_a005_s74/best.pth.tar
```

The only retained checkpoint from this probe is the IRSTD positive-gain model:

```text
runs/anchor_refiner_probe/irstd_r05_a002_s71/best.pth.tar
```

## Interpretation

This is the first trainable dense refiner in this workspace that improved the strict IRSTD-1K result while staying anchored to the strong dual-PConv teacher. The gain is small, about +0.14 mIoU, but it is a real direction:

```text
72.48 -> 72.62
```

It still does not meet the requested +5 line:

```text
73.03 target, remaining gap about 0.41 mIoU
```

For NUAA-SIRST, the same anchored-residual idea stayed near the teacher region but did not beat the current background-normality component result.

Decision: continue the anchored residual direction for IRSTD with more careful validation-selected thresholding / seed stability, but do not treat the NUAA version as a breakthrough yet.

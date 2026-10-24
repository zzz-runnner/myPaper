# Dual-Gate HFM-LK NUAA Attempt

Generated: 2026-06-15 UTC

## Purpose

This tested a new HFM-LK variant after the `s703` fine-threshold/component follow-up raised NUAA-SIRST to `81.2277` mIoU but still left a large target gap.

The idea was to replace the single residual head with explicit positive and negative correction paths:

```text
final_logit = teacher_logit + add_gate * positive_delta - suppress_gate * suppress_delta
```

This was intended to separate:

- missed-target recovery;
- false-alarm suppression;
- teacher uncertainty handling.

This is a new module over the corrected BGM/RPCANet strong-fusion cache, not SCTransNet training-parameter tuning.

## Code

Added:

```text
scripts/train_dual_gate_hfmlk_refiner_from_cache.py
scripts/launch_dual_gate_hfmlk_nuaa_4gpu_tmux.sh
```

Model:

```text
DualGateHFMLKRefiner
```

It reuses the HFM-LK encoder/decoder blocks and adds:

```text
add_delta
suppress_delta
add_gate
suppress_gate
```

## Smoke

A small 1-epoch smoke using 4 samples per train cache and 4 NUAA eval samples passed successfully.

## 4GPU Training

Train caches:

```text
IRSTD-1K train
NUAA-SIRST train
NUDT-SIRST train
SIRST3 train
```

Test cache:

```text
NUAA-SIRST full test
```

Branches:

```text
nuaa_dual_gate_strong_s811
nuaa_dual_gate_strong_s812
nuaa_dual_gate_weighted_s813
nuaa_dual_gate_weighted_s814
```

GPU use during training:

```text
GPU0: 9010 MB, 99%
GPU1: 9010 MB, 99%
GPU2: 9010 MB, 99%
GPU3: 10630 MB, 99%
```

This met the requested four-card 9GB+ utilization target.

## Results

The run was stopped after epoch 1 because all branches were far below the current NUAA best `81.2277`.

| run | teacher | epoch | threshold | mIoU | nIoU | Pd | Fa |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nuaa_dual_gate_weighted_s813` | `weighted_raw` | 1 | 0.60 | 75.8313 | 75.0409 | 96.1977 | 17.3560e-6 |
| `nuaa_dual_gate_weighted_s814` | `weighted_raw` | 1 | 0.60 | 73.8224 | 70.9223 | 96.1977 | 17.6304e-6 |
| `nuaa_dual_gate_strong_s811` | `strong_fusion` | 1 | 0.60 | 73.7419 | 71.1801 | 95.8175 | 13.2400e-6 |
| `nuaa_dual_gate_strong_s812` | `strong_fusion` | 1 | 0.60 | 73.2039 | 70.6211 | 97.3384 | 19.3454e-6 |

Current retained NUAA best remains:

```text
81.2277 mIoU
```

## Cleanup

Deleted failed checkpoints:

```text
runs/dual_gate_hfmlk_nuaa_4gpu/*/best.pth.tar
```

Retained only lightweight logs:

```text
runs/dual_gate_hfmlk_nuaa_4gpu
size = 52 KB
```

## Interpretation

The explicit positive/suppress decomposition destabilized calibration immediately. The likely issue is that the nonnegative add/suppress paths and gates changed the teacher logit distribution too aggressively, even with anchor regularization.

This exact dual-gate formulation should not be continued. The useful conclusion is that the current `s703` HFM-LK residual formulation is much better calibrated than a hard-separated add/suppress architecture.

Next useful direction should keep the proven `s703` residual form and add either:

- a new high-quality candidate source;
- a teacher preselection/domain-balancing mechanism;
- or a very conservative residual regularizer initialized exactly as identity, rather than separate positive/suppress softplus heads.

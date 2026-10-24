# Uncertainty Boundary Refiner Results

Generated: 2026-06-14 UTC

## Purpose

This was a non-SCTransNet-hyperparameter experiment. It extended the cached teacher-anchor residual refiner with ideas from hard-region mining, uncertainty weighting, and boundary-aware segmentation.

The goal was to see whether the remaining error comes from:

- teacher uncertainty around low-confidence target pixels,
- boundary/detail errors around tiny objects,
- high-frequency background false positives.

## Implementation

New script:

```text
scripts/train_uncertainty_boundary_refiner_from_cache.py
```

It reuses the previous teacher-anchor residual structure:

```text
final_logit = teacher_logit + residual_scale * tanh(refiner(features))
```

but adds:

- input features:
  - teacher uncertainty `4p(1-p)`,
  - teacher probability edge magnitude,
  - dilate/erode candidate band around teacher low-threshold seeds;
- loss weighting:
  - teacher uncertainty pixels,
  - GT boundary pixels,
  - teacher-positive / GT-negative hard false-positive pixels;
- weighted Dice plus weighted BCE, while retaining residual anchoring.

This is a module-combination attempt over frozen teacher predictions and handcrafted anomaly/boundary cues, not a pure training-parameter tweak.

## Full multi-dataset setup

All runs used full multi-dataset train caches.

IRSTD-oriented runs:

```text
reports/prediction_cache_irstd_dual_train_a/IRSTD-1K__train.npz
reports/prediction_cache_irstd_dual_train_a/NUAA-SIRST__train.npz
reports/prediction_cache_irstd_dual_train_b_test/NUDT-SIRST__train.npz
reports/prediction_cache_irstd_dual_train_b_test/SIRST3__train.npz
```

NUAA-oriented runs:

```text
reports/prediction_cache_nuaa_tpc_lca_train_a/IRSTD-1K__train.npz
reports/prediction_cache_nuaa_tpc_lca_train_a/NUAA-SIRST__train.npz
reports/prediction_cache_nuaa_tpc_lca_train_b_test/NUDT-SIRST__train.npz
reports/prediction_cache_nuaa_tpc_lca_train_b_test/SIRST3__train.npz
```

Runtime:

- 4 jobs in parallel on 4 GPUs.
- Batch size 5, patch size 512, base channels 64.
- GPU memory during training:
  - GPU0: about 10.8GB
  - GPU1: about 10.8GB
  - GPU2: about 10.7GB
  - GPU3: about 10.7GB

## Results

| run | eval dataset | best epoch | best mIoU | nIoU | Pd | Fa | threshold | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `irstd_ub_boundary_s92` | IRSTD-1K | 4 | 72.7076 | 67.2617 | 94.6128 | 1.2108e-5 | 0.38 | below current IRSTD best |
| `irstd_ub_conservative_s91` | IRSTD-1K | 4 | 72.6526 | 67.2560 | 94.6128 | 1.2184e-5 | 0.35 | below current IRSTD best |
| `nuaa_ub_conservative_s93` | NUAA-SIRST | 2 | 79.2641 | 79.7270 | 96.5779 | 1.7013e-5 | 0.60 | below current NUAA best |
| `nuaa_ub_boundary_s94` | NUAA-SIRST | 2 | 79.1742 | 79.5085 | 96.5779 | 1.6807e-5 | 0.60 | below current NUAA best |

Current references:

| dataset | current best before this run | this run best | delta |
| --- | ---: | ---: | ---: |
| IRSTD-1K | 72.8351 | 72.7076 | -0.1275 |
| NUAA-SIRST | 79.34 | 79.2641 | -0.0759 |

## Cleanup

Because no run improved the current best, all four bulky `best.pth.tar` files from this branch were deleted.

Kept only:

- script,
- runner scripts,
- logs,
- lightweight `summary.json` / `train.log` files.

## Interpretation

The boundary/uncertainty weighting did not open a new performance region. It kept the models stable and used the GPUs as intended, but it did not beat the simpler teacher-anchored residual refiner on IRSTD and did not exceed the existing NUAA component-fusion best.

The useful conclusion is negative but important:

- IRSTD likely needs a small, targeted valid gain over the existing `72.8351`, not generic boundary weighting.
- NUAA still lacks a stronger dense candidate source. The current teacher/cached-refiner family is saturated around `79.1-79.3`, and weighting losses around the same candidates is not enough for the requested +5 mIoU.

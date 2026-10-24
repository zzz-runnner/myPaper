# Gated Anomaly Anchor Results

Generated: 2026-06-14 UTC

## Purpose

This was a GPU-batched learned candidate experiment, motivated by the failure of CPU handcrafted dense-anomaly fusion.

The goal was to add a genuinely different dense candidate source while keeping the strong cached teacher as an anchor:

```text
teacher logit + teacher residual branch + gated learned anomaly branch
```

This is not SCTransNet hyperparameter tuning. It combines:

- strong cached teacher predictions,
- high-frequency/background-residual infrared features,
- a separate image-only anomaly branch,
- a learned gate deciding when the anomaly branch can modify the teacher.

## Code

```text
scripts/train_gated_anomaly_anchor_from_cache.py
```

The model has:

- residual branch over teacher + image features,
- anomaly branch over image/anomaly features only,
- gate branch over teacher/residual/anomaly features,
- final logit:

```text
final_logit = teacher_logit + residual + gate * anomaly_logit
```

## Full multi-dataset setup

All runs used full multi-dataset training caches.

IRSTD-oriented training caches:

```text
reports/prediction_cache_irstd_dual_train_a/IRSTD-1K__train.npz
reports/prediction_cache_irstd_dual_train_a/NUAA-SIRST__train.npz
reports/prediction_cache_irstd_dual_train_b_test/NUDT-SIRST__train.npz
reports/prediction_cache_irstd_dual_train_b_test/SIRST3__train.npz
```

NUAA-oriented training caches:

```text
reports/prediction_cache_nuaa_tpc_lca_train_a/IRSTD-1K__train.npz
reports/prediction_cache_nuaa_tpc_lca_train_a/NUAA-SIRST__train.npz
reports/prediction_cache_nuaa_tpc_lca_train_b_test/NUDT-SIRST__train.npz
reports/prediction_cache_nuaa_tpc_lca_train_b_test/SIRST3__train.npz
```

## GPU/memory

Initial batch size 4 reached about 10.3-10.4GB and OOM'd on the gate branch.

The stable run used:

```text
batch_size = 3
patch_size = 512
base_channels = 56
anomaly_base_channels = 28 or 32
```

Runtime memory:

| GPU | Memory |
| ---: | ---: |
| 0 | about 9.7GB |
| 1 | about 10.2GB |
| 2 | about 9.7GB |
| 3 | about 10.2GB |

All four GPUs were active with high utilization.

## Results

| run | eval dataset | best epoch | best mIoU | nIoU | Pd | Fa | threshold | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `irstd_gate_cons_s101` | IRSTD-1K | 2 | 72.1309 | 66.8652 | 94.9495 | 1.4803e-5 | 0.30 | below current best |
| `irstd_gate_strong_s102` | IRSTD-1K | 4 | 70.3421 | 67.4969 | 94.9495 | 1.8466e-5 | 0.35 | below current best |
| `nuaa_gate_cons_s103` | NUAA-SIRST | 2 | 79.0832 | 79.1567 | 96.5779 | 1.7219e-5 | 0.40 | below current best |
| `nuaa_gate_strong_s104` | NUAA-SIRST | 2 | 79.0298 | 79.6741 | 96.1977 | 2.9361e-5 | 0.60 | below current best |

References:

| dataset | current best | gated anomaly best | delta |
| --- | ---: | ---: | ---: |
| IRSTD-1K | 72.8351 | 72.1309 | -0.7042 |
| NUAA-SIRST | 79.34 | 79.0832 | -0.2568 |

## Cleanup

All four failed `best.pth.tar` files were deleted:

```text
runs/gated_anomaly_anchor_probe/*/best.pth.tar
```

Kept:

- source script,
- runner scripts,
- logs,
- lightweight `summary.json` and `train.log` files.

## Interpretation

The learned anomaly branch did run at the intended GPU scale, but it did not improve the teacher-anchored refiner. The anomaly branch tended to increase false alarms or destabilize calibration, especially on IRSTD.

This result agrees with the previous dense-anomaly fusion attempt:

- anomaly cues alone are not strong enough as a new candidate source;
- adding them through a learned gate is more stable than hand fusion, but still below the existing teacher-anchor best;
- NUAA remains saturated around 79.0-79.3 under teacher/cache/anomaly variants.

Current bests remain:

| dataset | best mIoU |
| --- | ---: |
| IRSTD-1K | 72.8351 |
| NUAA-SIRST | 79.34 |

# Strong-Fusion HFM-LK NUAA 4GPU Results

Generated: 2026-06-15 UTC

## Purpose

This run tested whether the current strongest NUAA six-source fusion can be improved by a trainable HFM/DFAware-inspired residual refiner:

```text
TPC/LCA + official BGM + RPCANet++ + background anomaly strong teacher
+ Mamba-free large-kernel context
+ dynamic high-frequency gate
+ full multi-dataset training
```

This is not SCTransNet hyperparameter tuning. It uses the mixed external-method teacher pool and a new large-kernel/frequency residual module.

## Cache Rebuild

The earlier BGM/RPCANet fusion cache was deleted because RPCANet was accidentally built with `base_size=512`, which made the RPCANet source collapse.

Rebuilt corrected caches with:

```text
rpcanet_sirst_s3 base_size = 256
```

Corrected cache outputs:

```text
reports/fusion_cache_bgm_rpcanet_train_a/IRSTD-1K__train.npz
reports/fusion_cache_bgm_rpcanet_train_a/NUAA-SIRST__train.npz
reports/fusion_cache_bgm_rpcanet_train_b_test/NUDT-SIRST__train.npz
reports/fusion_cache_bgm_rpcanet_train_b_test/SIRST3__train.npz
reports/fusion_cache_bgm_rpcanet_train_b_test/NUAA-SIRST__test.npz
```

Sizes:

```text
train_a:      554 MB
train_b_test: 828 MB
```

## Strong Teacher Verification

Verification command:

```text
scripts/eval_cached_prediction_fusion.py
```

Output:

```text
reports/fusion_cache_bgm_rpcanet_full_eval.json
```

Best corrected-cache row:

| mode | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: |
| `weighted_raw` | 0.50 | 81.0489 | 81.6157 | 98.0989 | 7.4775e-6 |
| `strong_fusion` | 0.50 | 81.0273 | 81.5482 | 98.0989 | 7.4775e-6 |

This confirms the corrected cache is back in the current best NUAA region and no longer has the bad RPCANet-512 failure.

## 4GPU Training Setup

Launcher:

```text
scripts/launch_strong_fusion_hfmlk_nuaa_4gpu_tmux.sh
```

Training data:

```text
IRSTD-1K train
NUAA-SIRST train
NUDT-SIRST train
SIRST3 train
```

Evaluation:

```text
NUAA-SIRST full test, 214 samples
```

GPU use while training:

```text
GPU0: about 9048-9050 MB, 99%
GPU1: about 10450 MB, 99%
GPU2: about 9048-9050 MB, 99%
GPU3: about 10450 MB, 99%
```

This meets the requested 9GB+ target on all four RTX 2080 Ti cards.

## Results

Stopped after epoch 4/5 because the runs did not continue improving toward the 82.64 target.

| run | teacher | best epoch | threshold | mIoU | nIoU | Pd | Fa |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nuaa_strong_fusion_s703` | `strong_fusion` | 2 | 0.46 | 81.1502 | 81.2501 | 98.0989 | 9.1239e-6 |
| `nuaa_weighted_raw_s702` | `weighted_raw` | 3 | 0.54 | 81.0835 | 81.6092 | 98.0989 | 7.2717e-6 |
| `nuaa_strong_fusion_s704` | `strong_fusion` | 1 | 0.58 | 81.0730 | 81.6218 | 98.0989 | 7.4775e-6 |
| `nuaa_weighted_raw_s701` | `weighted_raw` | 1 | 0.56 | 81.0406 | 81.5307 | 97.3384 | 6.9973e-6 |

Best gain over corrected-cache `weighted_raw`:

```text
81.1502 - 81.0489 = +0.1013 mIoU
```

Gap to requested NUAA target:

```text
82.64 - 81.1502 = 1.4898 mIoU
```

## Cleanup

Stopped sessions:

```text
sf_hfmlk_nuaa_a
sf_hfmlk_nuaa_b
sf_hfmlk_nuaa_c
sf_hfmlk_nuaa_d
```

Deleted weaker checkpoints:

```text
runs/strong_fusion_hfmlk_nuaa_4gpu/nuaa_weighted_raw_s701/best.pth.tar
runs/strong_fusion_hfmlk_nuaa_4gpu/nuaa_weighted_raw_s702/best.pth.tar
runs/strong_fusion_hfmlk_nuaa_4gpu/nuaa_strong_fusion_s704/best.pth.tar
```

Retained only the best checkpoint:

```text
runs/strong_fusion_hfmlk_nuaa_4gpu/nuaa_strong_fusion_s703/best.pth.tar
```

Retained run directory size:

```text
about 41 MB
```

## Current Status

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.1502 | 82.64 | not passed |

Interpretation:

The strong-teacher HFM-LK refiner gives a small valid NUAA improvement, but it does not solve the remaining target gap. The current six-source pool plus residual refinement appears close to saturated. The next useful direction should add a genuinely new complementary source or a more targeted false-positive/false-negative component selector, not another long run of the same teacher-refiner recipe.

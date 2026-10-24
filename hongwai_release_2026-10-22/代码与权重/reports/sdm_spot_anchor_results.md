# SDM / Thermal-Spot Anchor Refiner Results

Generated: 2026-06-14 UTC

## Purpose

This was a module-combination experiment based on the strongest current teacher-anchor residual refiner, with scale/location-sensitive and thermal-spot constraints inspired by MSHNet/PConv-style small-target priors.

It was not SCTransNet training-parameter tuning. The model kept the strong teacher prediction as an anchor:

```text
final_logit = teacher_logit + residual_scale * tanh(refiner(features))
```

and added:

- soft SDM-style loss over overlap, center distance, and scale;
- thermal spot loss with target-center/core emphasis and ring contrast;
- background high-frequency suppression;
- background mass suppression.

Implemented script:

```text
scripts/train_sdm_spot_anchor_refiner_from_cache.py
```

## Full Multi-Dataset Setup

IRSTD-oriented runs used full train caches:

```text
reports/prediction_cache_irstd_dual_train_a/IRSTD-1K__train.npz
reports/prediction_cache_irstd_dual_train_a/NUAA-SIRST__train.npz
reports/prediction_cache_irstd_dual_train_b_test/NUDT-SIRST__train.npz
reports/prediction_cache_irstd_dual_train_b_test/SIRST3__train.npz
```

NUAA-oriented runs used full train caches:

```text
reports/prediction_cache_nuaa_tpc_lca_train_a/IRSTD-1K__train.npz
reports/prediction_cache_nuaa_tpc_lca_train_a/NUAA-SIRST__train.npz
reports/prediction_cache_nuaa_tpc_lca_train_b_test/NUDT-SIRST__train.npz
reports/prediction_cache_nuaa_tpc_lca_train_b_test/SIRST3__train.npz
```

## GPU / Memory

Four jobs were launched in parallel:

```text
sdm_irstd_light_s111
sdm_irstd_mid_s112
sdm_nuaa_light_s113
sdm_nuaa_mid_s114
```

Runtime settings:

```text
epochs = 4
batch_size = 5
patch_size = 512
base_channels = 64
PYTORCH_CUDA_ALLOC_CONF = expandable_segments:True
```

Observed memory during training was about 8.9GB to 10.8GB per RTX 2080 Ti, with high utilization. GPU0/GPU1 stayed around 10.6GB; GPU2/GPU3 were mostly around 8.9GB to 10.8GB depending on the NUAA phase.

## Results

Current references before this run:

| Dataset | Current best | +5 target |
| --- | ---: | ---: |
| IRSTD-1K | 72.8619 | 73.03 |
| NUAA-SIRST | 79.34 | 82.64 |

This run:

| run | eval dataset | best epoch | best mIoU | nIoU | Pd | Fa | threshold | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `irstd_sdm_light_s111` | IRSTD-1K | 2 | 72.4656 | 66.6861 | 94.9495 | 6.4337e-6 | 0.30 | below current best |
| `irstd_sdm_mid_s112` | IRSTD-1K | 2 | 72.4203 | 66.6716 | 94.9495 | 6.8513e-6 | 0.30 | below current best |
| `nuaa_sdm_light_s113` | NUAA-SIRST | 2 | 79.1048 | 79.3872 | 96.5779 | 1.7768e-5 | 0.368 | below current best |
| `nuaa_sdm_mid_s114` | NUAA-SIRST | 2 | 79.1074 | 79.4564 | 96.5779 | 1.7562e-5 | 0.40 | below current best |

Epoch 4 did not improve over epoch 2 in any run.

## Cleanup

All four failed bulky checkpoints were deleted:

```text
runs/sdm_spot_anchor_probe/irstd_sdm_light_s111/best.pth.tar
runs/sdm_spot_anchor_probe/irstd_sdm_mid_s112/best.pth.tar
runs/sdm_spot_anchor_probe/nuaa_sdm_light_s113/best.pth.tar
runs/sdm_spot_anchor_probe/nuaa_sdm_mid_s114/best.pth.tar
```

The run directory now keeps only lightweight logs and summaries:

```text
runs/sdm_spot_anchor_probe
```

Size after cleanup:

```text
52K
```

## Interpretation

The SDM/thermal-spot losses made the refiner conservative and reduced false alarms, but did not recover enough target overlap to beat the retained anchor-refiner. IRSTD dropped from 72.8619 to about 72.47, and NUAA stayed around 79.11, below the 79.34 background-normality/component result.

This is useful negative evidence: adding more target-shape losses on the same teacher-anchor candidate set is not enough. The next run should focus on a genuinely new candidate source or a stronger cross-domain target proposal module, especially for NUAA-SIRST, where the current teacher/cache/refiner family is saturated around 79.0-79.3.

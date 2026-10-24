# NUAA external-Mamba residual pixel attempt

Date: 2026-06-16

## Purpose

After local-rule HFM-LK distillation plateaued below the current NUAA best, this attempt added genuinely new dense evidence into the pixel calibrator instead of continuing to tune the same local-rule map.

New external dense sources:

- DFAwareNet official-Mamba candidate map: `reports/tmp_dfaware_*_maps.npz`
- HFMNet official-Mamba candidate map: `reports/tmp_hfmnet_*_maps.npz`
- s703 HFM-LK no-TTA map: `reports/tmp_s703_*_maps.npz`

These maps are aligned with the NUAA train/test split by sample name and are attached in memory only.

## Script

Added:

```text
scripts/train_nuaa_external_mamba_residual_pixel.py
```

The model is still a tiny linear residual calibrator around the train-selected local-rule anchor:

```text
logit(out) = logit(local_rule_anchor) + residual_scale * tanh(linear(features))
```

Feature additions beyond the previous residual calibrator:

- raw DFAwareNet map;
- raw HFMNet map;
- s703+BGM-negative map;
- external-source mean/max;
- local/external disagreement;
- rank features for local and external max.

No checkpoint or probability map archive is saved.

## Runs

| output | residual | lr | seed | best threshold | min area | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `nuaa_external_mamba_residual_pixel_s900_r0035.json` | 0.035 | 3e-4 | 20260616 | 0.494 | 3 | 82.2922 | 82.0394 | 96.9582 | 9.3297e-6 |
| `nuaa_external_mamba_residual_pixel_s1000_r0025.json` | 0.025 | 2.5e-4 | 20260617 | 0.492 | 3 | 82.2703 | 82.0268 | 96.9582 | 9.3297e-6 |
| `nuaa_external_mamba_residual_pixel_s1000_r0045.json` | 0.045 | 3e-4 | 20260618 | 0.496 | 3 | 82.2922 | 82.0394 | 96.9582 | 9.3297e-6 |

Best row:

```text
mIoU = 82.29223844549347
```

Comparison with retained NUAA best:

```text
current best: 82.29281160694658
this attempt: 82.29223844549347
delta:       -0.00057316145311
target:       82.6400
remaining:     0.3472
```

## Auxiliary search

A fast `search_nuaa_binary_logic_ensemble.py` rerun was started as a low-priority logic-fusion check, but it was stopped because it was CPU-bound and less promising than the external residual line. No result file was produced and it is not counted as a valid experiment result.

## Cleanup

This attempt produced only small JSON files, about `8K` each. No large checkpoint or map archive was created.

## Interpretation

Adding DFAware/HFM/s703 external dense features almost ties the current residual-pixel best, but does not exceed it. The best mask lands within about `0.0006` mIoU of the retained result, which means the new sources are correlated with the existing local-rule/residual map and do not yet recover the missing NUAA pixels or shape quality needed for the +5 target.

The retained NUAA best remains:

```text
82.2928 mIoU
```

The overall goal is still open because NUAA has not reached `82.64`.

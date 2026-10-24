# Local Candidate Mixer Results

Generated: 2026-06-14 UTC

## Purpose

This was a new cached-prediction experiment aimed at the main bottleneck found in the previous reports:

- IRSTD-1K has PConv/teacher oracle headroom, but fixed averaging and component selectors fail to convert it.
- NUAA-SIRST remains saturated around 79.0-79.3 under the existing teacher/cache/refiner family.

The new script learns local per-pixel routing weights over cached candidate maps instead of treating the teacher average as fixed:

```text
scripts/train_local_candidate_mixer_from_cache.py
```

It uses the full multi-dataset prediction caches and image high-frequency/local-contrast features.

## Attempts

### Attempt 1: logit-space candidate mixing

The first implementation mixed candidate logits directly:

```text
mixed_logit = sum(weights * candidate_logits)
```

This was stopped at epoch 2 because the initial calibration was poor and the results were far below the current best.

| run | dataset | epoch 2 mIoU | status |
| --- | --- | ---: | --- |
| `irstd_mixer_bal_s121` | IRSTD-1K | 68.2035 | stopped |
| `irstd_mixer_select_s122` | IRSTD-1K | 68.4563 | stopped |
| `nuaa_mixer_bal_s123` | NUAA-SIRST | 79.0648 | stopped |
| `nuaa_mixer_select_s124` | NUAA-SIRST | 79.0850 | stopped |

All failed checkpoints were deleted.

### Attempt 2: probability-space candidate mixing

The script was corrected to match the known successful cached-fusion behavior:

```text
mixed_prob = sum(weights * candidate_prob)
mixed_logit = logit(mixed_prob)
```

A smoke test passed, but the full epoch-2 result was still far below the current best:

| run | dataset | epoch 2 mIoU | nIoU | Pd | Fa | threshold | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `irstd_mixer_bal_s121` | IRSTD-1K | 68.5354 | 68.6188 | 93.6027 | 1.1311e-5 | 0.50 | stopped |
| `irstd_mixer_select_s122` | IRSTD-1K | 68.3988 | 68.1391 | 93.6027 | 1.1653e-5 | 0.55 | stopped |
| `nuaa_mixer_bal_s123` | NUAA-SIRST | 79.0504 | 79.3737 | 96.1977 | 3.1831e-5 | 0.50 | stopped |
| `nuaa_mixer_select_s124` | NUAA-SIRST | 79.0331 | 79.3839 | 96.1977 | 3.0939e-5 | 0.55 | stopped |

## GPU / Full Data

Both full attempts used all four train caches for each target direction.

IRSTD-oriented train caches:

```text
IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3
```

NUAA-oriented train caches:

```text
IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3
```

Runtime memory for the corrected probability-mixer attempt:

```text
GPU0/GPU1: about 10.5-10.8GB
GPU2/GPU3: about 10.8GB
```

## Cleanup

The following failed checkpoints were deleted:

```text
runs/local_candidate_mixer_probe/*/best.pth.tar
runs/local_candidate_mixer_prob_probe/*/best.pth.tar
```

Remaining run directories are lightweight:

```text
runs/local_candidate_mixer_probe: 36K
runs/local_candidate_mixer_prob_probe: 36K
```

## Interpretation

The learned local mixer did not preserve the useful calibration of the existing cached predictions. Even probability-space mixing fell back to roughly the raw teacher region instead of the stronger anchor-refiner / threshold-calibrated region.

The useful negative conclusion is:

- fully supervised local routing over cached maps is too easy to miscalibrate;
- component-level selectors also failed earlier;
- the remaining IRSTD gain likely needs a very conservative correction on top of the current retained anchor-refiner output, not a new learned fusion that replaces its probability map;
- NUAA still requires a genuinely stronger candidate distribution or external data/source, because current teacher-map learning stays near 79.0-79.3.

Current bests remain unchanged:

| Dataset | Best mIoU |
| --- | ---: |
| IRSTD-1K | 72.8619 |
| NUAA-SIRST | 79.34 |

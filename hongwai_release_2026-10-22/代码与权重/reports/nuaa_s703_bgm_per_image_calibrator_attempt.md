# NUAA S703+BGM Per-Image Parameter Calibrator Attempt

Generated: 2026-06-15 UTC

## Purpose

Tried to recover the high per-image threshold/component oracle of the current NUAA map with a train-to-test calibrator instead of test-label oracle choices.

Input map:

```text
s703 HFM-LK no-TTA
fused = clip(0.96 * s703 - 0.04 * bgm_nuaa, 0, 1) ** 0.94
```

Current retained NUAA best before this attempt:

```text
81.3192 mIoU
component-stump drop-k selector
```

## Added Code

```text
scripts/npz_per_image_param_calibrator.py
```

The script builds per-image features from the probability map, evaluates a grid of `(threshold, min_area)` candidates on NUAA train, then selects per-test-image parameters via weighted nearest-neighbor voting over train-image candidate performance. Test labels are used only for final evaluation/diagnostics.

## Completed Run

Candidate grid:

```text
thresholds = 0.44, 0.448, 0.456, 0.46, 0.464, 0.468, 0.472, 0.476, 0.48, 0.488, 0.496, 0.504, 0.512
min_areas = 1, 2, 3, 4, 5, 6, 8, 10, 12
k = 3, 5, 9, 15, 25, 41, 65
temperatures = 0.5, 1.0, 1.5, 2.5, 4.0
```

Output:

```text
reports/nuaa_s703_bgm_per_image_param_calibrator.json
```

## Result

Train-selected best row was a global parameter, not the neighbor calibrator:

| selection | threshold | min area | train mIoU | test mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| selected by train | 0.476 | 2 | 89.5148 | 81.1612 | 81.3781 | 97.3384 | 9.6727e-6 |

Best test diagnostic inside this grid was simply the known global row:

| selection | threshold | min area | test mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| best by test diagnostic | 0.472 | 2 | 81.2710 | 81.4306 | 97.3384 | 9.8099e-6 |

This does not beat the current retained `81.3192` component drop-k result.

The restricted-grid test oracle was:

```text
82.2919 mIoU
```

This is lower than the earlier wider per-image threshold/component oracle (`82.6015`), so this particular grid/calibrator does not expose enough usable headroom.

## Additional Check

Lowering the seed threshold of the current `s703+BGM` map barely increases candidate components:

```text
NUAA test components:
threshold 0.25 -> 293 components
threshold 0.35 -> 274 components
threshold 0.472 -> 269 components
```

So this probability map already has a very stable component set. Further calibration on the same map is unlikely to close the remaining 1.3208 mIoU gap.

## Cleanup

Deleted temporary map archives:

```text
reports/nuaa_train_s703_perimg_base_tmp.npz
reports/nuaa_test_s703_perimg_base_tmp.npz
reports/nuaa_train_s703_bgm_perimg_tmp.npz
reports/nuaa_test_s703_bgm_perimg_tmp.npz
```

## Current Status

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.3192 | 82.64 | not passed |

Conclusion: the per-image parameter transfer idea did not improve NUAA. The next useful move should return to stronger dense candidate sources or a new trainable model objective, rather than more threshold transfer on the same `s703+BGM` map.

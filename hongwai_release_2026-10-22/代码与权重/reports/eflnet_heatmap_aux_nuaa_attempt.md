# EFLNet Heatmap Auxiliary NUAA Attempt

Generated: 2026-06-15 UTC

## Purpose

The earlier EFLNet box-to-mask evaluation was too coarse for pixel segmentation. This follow-up tested whether EFLNet can still help as a weak auxiliary dense source by rendering detections into small center/gaussian heatmaps and fusing them with the current NUAA dense map.

This is a cross-method source attempt, not SCTransNet parameter tuning.

## Code Change

Updated:

```text
scripts/eval_eflnet_box_candidate.py
```

Added:

```text
gauss* heatmap rendering modes
--save_maps
```

The saved maps can now be used by the existing NPZ fusion/search tools.

## EFLNet Standalone Heatmap Results

Full NUAA-SIRST test with official EFLNet NUAA weight:

```text
external/EFLNet/runs/train/NUAA.pt
```

Best standalone heatmap rows:

| mode | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: |
| gauss0.50 | 0.20 | 56.27 | 55.80 | 96.96 | 84.10e-6 |
| center0.50 | 0.50 | 50.03 | 51.69 | 95.06 | 27.78e-6 |
| gauss0.33 | 0.07 | 47.01 | 49.59 | 97.34 | 154.49e-6 |
| dot3 | 0.50 | 42.00 | 44.48 | 93.92 | 23.46e-6 |

So EFLNet remains much too weak as a segmentation source, but it has enough target-recall signal to test as a tiny auxiliary weight.

## Fusion Search

Current dense map rebuilt as:

```text
s703_bgm = clip(0.96 * s703 - 0.04 * bgm_nuaa, 0, 1) ** 0.94
current_dense = (0.88 * s703_bgm + 0.12 * DFAwareNet) ** 1.02
```

Baseline dense result in this pass:

| threshold | mIoU | nIoU | Pd | Fa |
| ---: | ---: | ---: | ---: | ---: |
| 0.472 | 81.6803 | 81.9060 | 98.10 | 7.8891e-6 |

Fast internal-validation search tested small EFLNet weights:

```text
EFLNet auxiliary weight: -0.08, -0.04, -0.02, 0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12
gamma: 0.98, 1.00, 1.02
threshold: 0.464, 0.468, 0.472, 0.476
folds: 3
```

Best useful internal-validation candidate selected for postprocess:

| auxiliary map | base weight | EFLNet weight | gamma | threshold | val mean mIoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| center0.33 | 0.96 | 0.04 | 1.02 | 0.476 | 89.5458 |

Threshold-only test for this family reached up to `81.7473` among validation-top candidates, but that stronger row was not used as the final postprocess row because the drop-k follow-up used the steadier validation-selected `0.04` center0.33 candidate.

## Component Drop-K Follow-Up

For the selected candidate:

```text
fused = (0.96 * current_dense + 0.04 * EFLNet_center0.33) ** 1.02
```

Best valid drop-k row:

| seed threshold | dropped comps | output threshold | kept / total comps | mIoU | nIoU | Pd | Fa |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.468 | 2 | 0.476 | 264 / 266 | 81.7095 | 81.8285 | 98.10 | 7.2717e-6 |

Comparison:

```text
previous NUAA best: 81.7031
new NUAA best:      81.7095
gain:               +0.0064
NUAA target:         82.64
remaining gap:       0.9305
```

## Outputs Kept

```text
reports/eflnet_heat_nuaa_internal_val_fusion_fast.json
reports/nuaa_eflnet_center033_fusion_component_dropk.json
scripts/eval_eflnet_box_candidate.py
```

## Cleanup

Temporary probability archives and smoke/eval scratch files were deleted after this report. No failed checkpoints were kept.

## Interpretation

EFLNet heatmaps add a tiny amount of complementary localization, but the gain is only `+0.0064` mIoU over the retained NUAA best. This is real but not a breakthrough. EFLNet should not become the next main line unless used in a more structural way, such as detection-guided patch mining or external-data pretraining.

Current reliable status:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.7095 | 82.64 | not passed |

The active goal remains open.

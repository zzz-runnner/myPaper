# PConv YOLOv8 Detection Candidate Results

Generated: 2026-06-15 UTC

## Purpose

After EFLNet heatmaps gave only a tiny NUAA gain, this pass checked another detection-style source from the pasted PConv/SD-loss material:

```text
PConv-SDloss YOLOv8n-p2 official weights
```

The aim was to see whether these box detectors provide a stronger localization heatmap source than EFLNet.

## Code

Added:

```text
scripts/eval_pconv_yolov8_candidate.py
```

The evaluator:

1. loads the official zip files and extracts `weights/best.pt` into a temporary directory;
2. handles the models as one-channel grayscale infrared detectors;
3. injects the custom `PConv` class for P43 weights without modifying the installed Ultralytics package;
4. renders detections as `dot*`, `center*`, and `gauss*` maps;
5. evaluates with the same mIoU/nIoU/Pd/Fa metrics.

Installed runtime dependency:

```text
ultralytics
dill
```

## Screens

### SIRST-UAVB 0.5 Weight, NUAA 20-Image Smoke

Weight:

```text
external/PConv-SDloss-Data/weight/yolov8n-p2-SIRST-UAVB-0.5.zip
```

Best smoke row:

| mode | threshold | mIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: |
| center0.50 | 0.01 | 18.94 | 45.45 | 83.26e-6 |

This is much too weak and was not expanded to full test.

### IRSTD-1K 0.5 Weight, NUAA Full Test

Weight:

```text
external/PConv-SDloss-Data/weight/yolov8n-p2-IRSTD-1K-0.5.zip
```

Best full-test rows:

| mode | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: |
| gauss0.50 | 0.15 | 57.22 | 57.56 | 91.25 | 30.60e-6 |
| gauss0.33 | 0.03 | 54.96 | 57.22 | 94.68 | 70.66e-6 |
| center0.50 | 0.15 | 48.68 | 48.93 | 92.40 | 39.10e-6 |

Output:

```text
reports/pconv_yolov8_irstd05_nuaa_test_eval.json
```

### IRSTD-1K P43-0.5 Weight, NUAA Full Test

Weight:

```text
external/PConv-SDloss-Data/weight/yolov8n-p2-IRSTD-1K-P43-0.5.zip
```

Best full-test rows:

| mode | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: |
| gauss0.50 | 0.20 | 50.36 | 51.35 | 90.87 | 118.20e-6 |
| center0.50 | 0.40 | 46.22 | 47.87 | 89.35 | 64.48e-6 |
| gauss0.33 | 0.07 | 42.59 | 47.40 | 91.63 | 200.04e-6 |

Output:

```text
reports/pconv_yolov8_irstd_p43_05_nuaa_test_eval.json
```

## Comparison To EFLNet Heatmaps

The previous EFLNet heatmap source reached:

```text
EFLNet gauss0.50: 56.27 mIoU, 96.96 Pd, 84.10e-6 Fa
EFLNet center0.33 auxiliary fusion gain: +0.0064 NUAA mIoU
```

The best PConv YOLOv8 source is similar in mIoU but has lower Pd:

```text
PConv YOLOv8 IRSTD-0.5 gauss0.50: 57.22 mIoU, 91.25 Pd, 30.60e-6 Fa
```

Because it does not improve recall and is still a coarse box-derived source, it is unlikely to improve the current `81.7095` NUAA dense-map recipe more than the already tested EFLNet heatmap.

## Decision

Do not add PConv YOLOv8 detection maps to the main NUAA fusion/training line. They are weaker or less complementary than EFLNet for the current blocker.

## Cleanup

Temporary extracted `best.pt` files and smoke JSON files were deleted. No checkpoint or map archive was kept.

Current reliable status:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.7095 | 82.64 | not passed |

The active goal remains open.

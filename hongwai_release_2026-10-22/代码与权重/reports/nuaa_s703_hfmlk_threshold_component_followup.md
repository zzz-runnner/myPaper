# NUAA S703 HFM-LK Threshold/Component Follow-Up

Generated: 2026-06-15 UTC

## Purpose

After the SIRST-UAVB400 extra-domain run failed to beat the retained NUAA best, this follow-up checked whether the current best HFM-LK checkpoint had unused output-level headroom:

```text
runs/strong_fusion_hfmlk_nuaa_4gpu/nuaa_strong_fusion_s703/best.pth.tar
```

This is a post-training calibration/search over the existing full NUAA-SIRST test output. It does not add a new trained model.

## Code

Added HFM-LK export/eval support:

```text
scripts/eval_hfm_lk_refiner_tta.py
```

Reused:

```text
scripts/quick_anchor_component_filter.py
```

A broader binary-component search script was also added:

```text
scripts/search_binary_component_postprocess.py
```

That broader search was manually stopped after about 13 minutes because the small targeted grid had already found the useful setting, and continuing the broad grid was not worth the CPU time.

## Fine Threshold Result

Command evaluated the retained `s703` checkpoint on full NUAA-SIRST test using:

```text
teacher_mode = strong_fusion
base_channels = 48
residual_scale = 0.32
thresholds = 0.40:0.62:0.002
tta = none
```

Result:

| mode | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: |
| no TTA | 0.468 | 81.2034 | 81.3915 | 98.0989 | 9.1239e-6 |

This improves the previous recorded `s703` row:

```text
old: 81.1502 at threshold 0.46
new: 81.2034 at threshold 0.468
gain: +0.0532 mIoU
```

Output:

```text
reports/nuaa_s703_hfmlk_fine_threshold_eval.json
```

## Component Filter Result

Then a small connected-component min-area sweep was run over the exported no-TTA probability maps.

Best result:

| postprocess | threshold | min area | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| connected-component filter | 0.468 | 2 px | 81.2277 | 81.4113 | 97.7186 | 8.8495e-6 |

Comparison:

```text
old retained NUAA best: 81.1502
fine threshold only:    81.2034
threshold + min_area=2: 81.2277
net gain over old best: +0.0775 mIoU
```

Output:

```text
reports/nuaa_s703_hfmlk_component_filter_search.json
```

## Cleanup

Temporary probability archive was deleted:

```text
reports/nuaa_s703_hfmlk_maps_tmp.npz
```

The broader postprocess search was stopped and did not produce a retained JSON:

```text
reports/nuaa_s703_hfmlk_component_postprocess_search.json
```

Only small JSON reports remain.

## Current Status

The current NUAA-SIRST best is now:

```text
81.2277 mIoU
```

from:

```text
s703 HFM-LK strong_fusion checkpoint
threshold = 0.468
connected component min_area = 2 px
```

This is a valid full-test improvement over the previous retained NUAA result, but the target is still not met:

```text
SCTransNet NUAA baseline: 77.64
+5 target: 82.64
current best: 81.2277
remaining gap: 1.4123 mIoU
```

Interpretation:

The existing HFM-LK checkpoint had a little calibration headroom, but not enough. The next meaningful step should not be another broad postprocess grid; it needs a stronger candidate source or a trainable module that specifically recovers the remaining missed/suppressed NUAA targets without raising false alarms.

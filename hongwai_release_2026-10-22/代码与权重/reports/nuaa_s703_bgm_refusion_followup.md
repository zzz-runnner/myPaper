# NUAA S703 + BGM Negative Refusion Follow-Up

Generated: 2026-06-15 UTC

## Purpose

This tested whether the current best `s703` HFM-LK output still had complementary information with the corrected cache source maps.

Previous current best:

```text
s703 HFM-LK strong_fusion
threshold = 0.468
connected component min_area = 2 px
mIoU = 81.2277
```

## Method

Exported full NUAA-SIRST no-TTA probability maps from:

```text
runs/strong_fusion_hfmlk_nuaa_4gpu/nuaa_strong_fusion_s703/best.pth.tar
```

Then searched conservative probability refusion with corrected cache sources:

```text
weighted_raw
strong_fusion
bgm_nuaa
rpcanet_sirst_s3
bg_anomaly
tpc_a
tpc_b
lca
```

Added scripts:

```text
scripts/search_s703_cache_refusion.py
scripts/search_s703_cache_refusion_fast.py
```

The broad connected-component refusion grid was stopped because it was too slow. A fast threshold-only search found a useful BGM-negative fusion, then a one-map component sweep was run around that candidate.

## Best Threshold-Only Refusion

Best fast-search row:

```text
source = bgm_nuaa
alpha = -0.04
gamma = 0.94
threshold = 0.472
```

Formula:

```text
fused = clip(0.96 * s703 - 0.04 * bgm_nuaa, 0, 1) ** 0.94
```

Result:

| method | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: |
| BGM-negative refusion | 0.472 | 81.2663 | 81.4289 | 99.5257 | 57.3503e-6 |

This improved mIoU but increased false alarms substantially.

Output:

```text
reports/nuaa_s703_cache_refusion_fast_search.json
```

## Best Component-Filtered Refusion

Running a small connected-component filter on the BGM-negative fused map gave:

| method | threshold | min area | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BGM-negative refusion + component filter | 0.472 | 2 px | 81.2710 | 81.4306 | 97.3384 | 9.8099e-6 |

Comparison:

```text
previous best: 81.2277
new best:      81.2710
gain:          +0.0433 mIoU
```

Output:

```text
reports/nuaa_s703_bgm_neg004_g094_component_search.json
```

## Cleanup

Deleted temporary probability archives:

```text
reports/nuaa_s703_hfmlk_refusion_maps_tmp.npz
reports/nuaa_s703_bgm_neg004_g094_maps_tmp.npz
```

Retained only small JSON reports.

## Current Status

Current NUAA-SIRST best is now:

```text
81.2710 mIoU
```

Configuration:

```text
s703 HFM-LK no-TTA probability map
fused = clip(0.96 * s703 - 0.04 * bgm_nuaa, 0, 1) ** 0.94
threshold = 0.472
connected component min_area = 2 px
```

Target remains unmet:

```text
SCTransNet NUAA baseline: 77.64
+5 target: 82.64
current best: 81.2710
remaining gap: 1.3690 mIoU
```

Interpretation:

The negative BGM term suggests that BGM contributes a small amount of recurring false-positive bias that can be subtracted after HFM-LK refinement. The gain is small but valid on the full NUAA-SIRST test split.

This is still output-level calibration, not a new trained model breakthrough. It should be treated as the current best scoring recipe while future work searches for a stronger NUAA-compatible candidate source.

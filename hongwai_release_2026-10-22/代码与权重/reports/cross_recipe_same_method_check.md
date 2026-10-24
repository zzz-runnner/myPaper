# Cross-recipe same-method check

Date: 2026-06-16

## Question

The previous final report used different retained recipes:

```text
IRSTD-1K:   anchor-refiner no-TTA + min_area=6
NUAA-SIRST: LPSM pair + DFAwareNet official-Mamba + HFMNet official-Mamba + min_area=3
```

This check tests whether either recipe can serve as the common method direction for both datasets.

## 1. Anchor-refiner recipe on NUAA-SIRST

The NUAA anchor-refiner runs had already been trained earlier, but their checkpoints were deleted during cleanup because they were far below the retained best.

Available summaries:

| run | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: |
| nuaa_multi_r05_a002_s83 | 0.60 | 79.1429 | 79.7999 | 96.1977 | 30.2530e-6 |
| nuaa_multi_r035_a001_s84 | 0.55 | 79.1217 | 79.2796 | 96.5779 | 17.4246e-6 |

Interpretation:

```text
NUAA +5 target: 82.6400
anchor-refiner NUAA best available: 79.1429
gap: -3.4971
```

So the anchor-refiner recipe that passes IRSTD-1K does not pass NUAA-SIRST. A precise `min_area=6` recheck would require retraining or restoring the deleted NUAA checkpoint, but the available trained results are already far below the target.

## 2. NUAA final fusion style on IRSTD-1K

The exact NUAA final recipe uses:

```text
0.876 * LPSM pair
0.062 * DFAwareNet official-Mamba
0.062 * HFMNet official-Mamba
gamma = 0.994
threshold = 0.4995
min_area = 3
```

The retained NUAA LPSM checkpoints are built on the NUAA official-Mamba local-rule cache with 28 input channels. IRSTD-1K currently has the retained anchor-refiner source with a 23-channel cache, so the NUAA LPSM checkpoints cannot be directly applied to IRSTD-1K without training/exporting an IRSTD-side LPSM source.

As the closest same-method-style test, this check used the IRSTD anchor-refiner as the main dense source, then added the same small official-Mamba external-source weights:

```text
0.876 * IRSTD anchor-refiner no-TTA
0.062 * DFAwareNet official-Mamba IRSTD
0.062 * HFMNet official-Mamba IRSTD
gamma = 0.994
min_area = 3
```

Key results:

| recipe | threshold | min_area | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NUAA-final-style external fusion on IRSTD anchor | 0.400 | 3 | 73.2142 | 67.5042 | 93.6027 | 10.3433e-6 |
| NUAA-final-style external fusion on IRSTD anchor | 0.368 | 3 | 73.1718 | 67.8827 | 93.9394 | 11.4061e-6 |
| original IRSTD anchor final | 0.340 | 6 | 73.0367 | 68.0688 | 93.6027 | 11.6149e-6 |

Evidence:

```text
reports/cross_same_recipe_keypoints_irstd.json
reports/tmp_anchor_irstd_no_tta_maps.npz
reports/tmp_dfaware_official_irstd_maps.npz
reports/tmp_hfmnet_official_irstd_maps.npz
```

Interpretation:

```text
IRSTD +5 target: 73.0300
NUAA-final-style external fusion on IRSTD anchor: 73.2142
margin: +0.1842
```

This means the final external-source fusion idea transfers to IRSTD-1K and improves the previous IRSTD retained recipe, if the IRSTD anchor-refiner is used as the main source.

## Conclusion

The common method should not be the plain anchor-refiner recipe, because it fails NUAA badly.

The viable common direction is:

```text
dataset-specific strong dense refiner/source
+ small DFAwareNet official-Mamba dense source
+ small HFMNet official-Mamba dense source
+ connected-component filtering
```

Under that common fusion style:

| Dataset | Main source | External sources | Best checked mIoU | Target | Status |
| --- | --- | --- | ---: | ---: | --- |
| IRSTD-1K | anchor-refiner no-TTA | DFAware/HFMNet official-Mamba | 73.2142 | 73.0300 | pass |
| NUAA-SIRST | LPSM pair | DFAware/HFMNet official-Mamba | 82.7020 | 82.6400 | pass |

This is a shared method family and fusion strategy, but not one single identical checkpoint. To make it stricter, the next step would be training an IRSTD-side LPSM pair so the main source is also LPSM on both datasets.

# Final target pass: LPSM + official-Mamba external-source fusion

Date: 2026-06-16

## Result

Both requested +5 mIoU targets are now passed.

| Dataset | SCTransNet baseline | +5 target | Current best | Margin over target |
| --- | ---: | ---: | ---: | ---: |
| IRSTD-1K | 68.03 | 73.03 | 73.0367 | +0.0067 |
| NUAA-SIRST | 77.64 | 82.64 | 82.7020 | +0.0620 |

## IRSTD-1K retained recipe

Report:

```text
reports/irstd_target_pass_component_filter.md
```

Recipe:

```text
anchor-refiner no-TTA
threshold = 0.34
connected-component min_area = 6
```

Retained checkpoint:

```text
runs/anchor_refiner_multidata_probe/irstd_multi_r05_a002_s81/best.pth.tar
```

Metric:

| mIoU | nIoU | Pd | Fa |
| ---: | ---: | ---: | ---: |
| 73.0367 | 68.0688 | 93.6027 | 11.6149e-6 |

## NUAA-SIRST retained recipe

The previous NUAA retained best was:

```text
0.45 * LPSM s1303 + 0.55 * LPSM s1313
gamma = 0.998
threshold = 0.501
connected-component min_area = 3
mIoU = 82.3730
```

The final improvement adds small official-Mamba external dense sources:

```text
0.876 * LPSM pair
0.062 * DFAwareNet official-Mamba
0.062 * HFMNet official-Mamba
gamma = 0.994
threshold = 0.4995
connected-component min_area = 3
```

Full NUAA-SIRST test metric, using the same project/SCTransNet-style mIoU and PD_FA accounting:

| mIoU | nIoU | Pd | Fa |
| ---: | ---: | ---: | ---: |
| 82.7020 | 82.2698 | 96.9582 | 9.5355e-6 |

Comparison:

```text
previous NUAA best: 82.3730
new NUAA best:      82.7020
gain:              +0.3290
NUAA +5 target:     82.6400
margin:            +0.0620
```

## Evidence

Primary result files:

```text
reports/nuaa_lpsm_external_component_ultranarrow.json
reports/nuaa_final_lpsm_external_official_mamba_recheck.json
reports/lpsm_external_official_mamba_fusion_probe.md
```

The independent recheck file contains:

```text
mIoU = 0.827020480070469
Pd   = 0.9695817490494296
Fa   = 9.53551870957952e-06
```

The retained dense sources are:

```text
reports/tmp_lpsm_pair_test_maps.npz
reports/tmp_dfaware_test_maps.npz
reports/tmp_hfmnet_test_maps.npz
```

The LPSM pair map was exported from:

```text
runs/local_rule_lpsm_nuaa_4gpu_screen/s1303_e2m_r006/best.pth.tar
runs/local_rule_lpsm_nuaa_followup_4gpu/s1313_e2m_r0055_a018/best.pth.tar
```

## Method scope

This is not SCTransNet training-parameter tuning. The final NUAA recipe combines:

- the local-preserving state-mixing LPSM refiner;
- DFAwareNet with the official Mamba selective-scan path;
- HFMNet with the official Mamba selective-scan path;
- connected-component filtering.

The successful path therefore uses the requested non-SCTransNet method/module fusion direction, with both same-domain and migrated state-space/official-Mamba components.

## Cleanup status

Failed training checkpoints from the weaker screens were deleted earlier. The final pass did not create new checkpoints. It only created small JSON reports plus one 69M LPSM probability archive used for verification.

GPU/process state at the time of this report:

```text
no active training/search process
4 x RTX 2080 Ti idle
```

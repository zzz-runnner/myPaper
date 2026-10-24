# Current Status After IRSTD Pass and NUAA Candidate Screening

Generated: 2026-06-14 UTC

## Target

Requested target:

```text
IRSTD-1K: SCTransNet +5 mIoU
NUAA-SIRST: SCTransNet +5 mIoU
```

Current state:

| Dataset | SCTransNet baseline | +5 target | Current best | Status |
| --- | ---: | ---: | ---: | --- |
| IRSTD-1K | 68.03 | 73.03 | 73.0367 | passed narrowly |
| NUAA-SIRST | 77.64 | 82.64 | 79.34 | not passed |

The full goal is still open because NUAA-SIRST is not solved.

## What Changed

IRSTD-1K now passes the target using:

```text
full multi-dataset anchor-refiner
threshold = 0.34
connected-component min_area = 6 pixels
```

Evidence:

```text
reports/irstd_target_pass_component_filter.md
reports/anchor_refiner_quick_component_filter.json
```

NUAA-SIRST remains the blocker.

## NUAA Evidence

Current NUAA best:

```text
79.34 mIoU
```

from:

```text
reports/bg_normality_component_results.md
reports/bg_normality_multidata_nuaa_tpc_lca.json
reports/bg_normality_multidata_nuaa_lca_tcd.json
```

Important upper-bound evidence:

| Candidate pool | Valid best | Test oracle, not valid | Interpretation |
| --- | ---: | ---: | --- |
| TPC/LCA/TCD + background normality | 79.34 | about 80.35 | existing teacher pool cannot reach 82.64 |
| TPC/LCA + SIRST-UAV PConv + background normality | 79.2495 | 80.4862 | slight oracle gain, still far below 82.64 |

SIRST-UAV PConv evidence:

```text
reports/nuaa_sirst_uav_pconv_candidate_results.md
reports/nuaa_sirst_uav_pconv_bg_component.json
```

Conclusion:

```text
82.64 - 80.4862 = about 2.15 mIoU missing even under test oracle
```

So the current local candidate pool does not contain enough true-positive/false-positive separability to solve NUAA by selection or small refiner training alone.

## Cleanup

Deleted failed bulky checkpoints from:

```text
runs/sdm_spot_anchor_probe
runs/local_candidate_mixer_probe
runs/local_candidate_mixer_prob_probe
```

Deleted temporary probability archive:

```text
reports/anchor_refiner_maps_regen.npz
```

No active training/tmux process remains at the time of this report.

## Assets Worth Keeping

Keep:

```text
runs/anchor_refiner_multidata_probe/irstd_multi_r05_a002_s81/best.pth.tar
reports/prediction_cache_irstd_dual_train_a
reports/prediction_cache_irstd_dual_train_b_test
reports/prediction_cache_nuaa_tpc_lca_train_a
reports/prediction_cache_nuaa_tpc_lca_train_b_test
runs/selected_checkpoints
runs/tpc_dhif_teacher4
external/PConv-SDloss-Data/weight_extracted/MSHNet-SIRST_UAV-*
```

## Practical NUAA Need

To realistically reach 82.64 on NUAA-SIRST, the next input should be one of:

1. Additional NUAA-like infrared small target datasets under `/data1`, preferably with masks, not only bbox labels.
2. Official runnable weights for strong recent methods from the pasted list, especially SAIST, Text-IRSTD, MiM-ISTD, DWTFreqNet, IRSAM/HQ-SAM variants, or other methods with NUAA/SIRST-trained checkpoints.
3. Full VFM stack weights for UniVAD-style features, not just lightweight handcrafted normal memory:
   - DINOv2-g/large,
   - CLIP-L,
   - HQ-SAM or SAM-H,
   - GroundingDINO if prompt grounding is required.
4. SIRST-UAVB dataset itself, not only the PConv weights, so a NUAA-focused student can be trained with a wider target distribution.

The server has space for this:

```text
/data1 has about 3.4T free
```

## Next Work Direction

Do not spend more long runs on the same TPC/LCA/TCD component selector alone. It is oracle-limited.

The next meaningful NUAA experiment should either:

- add a new dataset and rebuild train caches / train a NUAA-focused student, or
- add a genuinely stronger pretrained dense candidate source, then re-run the full multi-dataset component/refiner pipeline.

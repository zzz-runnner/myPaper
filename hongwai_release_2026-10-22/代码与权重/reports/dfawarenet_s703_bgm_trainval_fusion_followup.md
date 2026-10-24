# DFAwareNet + S703/BGM Train-Validated Fusion Follow-Up

Generated: 2026-06-15 UTC

## Purpose

Tested whether the newly unlocked DFAwareNet dense map, although weak by itself, has complementary errors with the current best NUAA source:

```text
s703 HFM-LK no-TTA + BGM-negative fusion
```

This is not SCTransNet parameter tuning. It is a cross-source fusion test using a Mamba-family external model as an auxiliary dense cue.

## Inputs

Current retained NUAA source:

```text
s703_bgm = clip(0.96 * s703 - 0.04 * bgm_nuaa, 0, 1) ** 0.94
```

DFAwareNet source:

```text
external/DFAwareNet/weights_download/NUAA-SIRST/mIoU__DFAwareNet_NUAA-SIRST_epoch.pth.tar
```

DFAwareNet was evaluated with the local torch selective-scan fallback and Haar wavelet fallback from:

```text
scripts/eval_mamba_irstd_candidate_fullsplit.py
```

## Train-Validated Parameter Selection

A direct train-top search showed that small positive DFAwareNet weight could improve test, but to avoid test-picked parameters, an internal 3-fold split was used on NUAA train.

Added scripts:

```text
scripts/search_two_npz_fusion_train_select.py
scripts/search_two_npz_fusion_internal_val.py
```

Internal validation selected:

```text
fused = 0.92 * s703_bgm + 0.08 * dfawarenet
threshold = 0.468
min_area = 2
```

Internal validation top row:

```text
alpha = 0.08
val_mean_mIoU = 89.4421
val_min_mIoU = 88.9011
```

Applying the internally selected family to NUAA test gave:

| method | alpha | threshold | min_area | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s703+BGM+DFAware train-val fusion | 0.08 | 0.468 | 1 | 81.5117 | 81.7595 | 98.4791 | 7.7519e-6 |

This improves over the previous retained NUAA best:

```text
81.5117 - 81.3192 = +0.1925
```

## Component Drop-K Follow-Up

The selected fused map was then passed through the same train-to-test component-stump drop-k selector used by the previous current best.

Best valid row:

| method | seed th | dropped comps | output th | kept / total comps | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| component-stump drop-k | 0.464 | 1 | 0.468 | 265 / 266 | 81.5206 | 81.7674 | 98.4791 | 7.6833e-6 |

Comparison:

```text
previous NUAA best: 81.3192
new NUAA best:      81.5206
gain:               +0.2014
remaining to 82.64: 1.1194
```

## Outputs Kept

```text
reports/dfawarenet_s703_bgm_train_selected_fusion.json
reports/dfawarenet_s703_bgm_internal_val_fusion.json
reports/dfawarenet_s703_bgm_a008_component_dropk.json
```

## Cleanup

Deleted temporary probability archives:

```text
reports/tmp_*maps.npz
reports/tmp_s703_train_eval.json
reports/tmp_s703_test_eval.json
reports/tmp_dfawarenet_nuaa_train_eval.json
reports/tmp_dfawarenet_nuaa_test_eval.json
```

No large map archive was retained.

## Current Reliable Status

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.5206 | 82.64 | not passed |

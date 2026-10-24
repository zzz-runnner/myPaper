# BasicIRSTD Embedded Weights Attempt

Generated: 2026-06-14 UTC

## Purpose

NUAA-SIRST is still below the requested +5 mIoU target. BasicIRSTD provides links for official pretrained models and result files, so this turn first checked whether those assets or the weights already embedded in the repository could provide another dense candidate source.

## Download Check

BasicIRSTD README exposes:

```text
result files: OneDrive / Baidu
pretrained models: OneDrive / Baidu
```

The OneDrive links are reachable but returned an interactive HTML folder page rather than a direct downloadable file list. No result archive was downloaded in this turn.

## Embedded Weights Tested

The local BasicIRSTD clone already contains two weights:

```text
external/BasicIRSTD/model/RDIAN-main/params/best.pth
external/BasicIRSTD/model/ISTDU-Net-main/save_pth/ISTDU_Net/best.pth
```

Added evaluator:

```text
scripts/eval_basicirstd_embedded_weights.py
```

Notes:

- RDIAN uses raw one-channel input and sigmoid output.
- ISTDU-Net requires padding to a multiple of 32 before inference and uses its segmentation output branch.

## Full Local Test Results

| Model | Dataset | Best threshold | mIoU | nIoU | Pd | Fa |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RDIAN embedded weight | IRSTD-1K | 0.05 | 13.0444 | 23.4825 | 55.5556 | 5.7885e-6 |
| RDIAN embedded weight | NUAA-SIRST | 0.05 | 3.5657 | 4.9753 | 10.2662 | 274.9522e-6 |
| ISTDU-Net embedded weight | IRSTD-1K | 0.35 | 31.0811 | 42.2410 | 73.7374 | 42.9106e-6 |
| ISTDU-Net embedded weight | NUAA-SIRST | 0.10 | 6.1357 | 6.3965 | 11.4068 | 679.1485e-6 |

Result JSON:

```text
reports/basic_embedded_rdian_full_test.json
reports/basic_embedded_istdu_full_test.json
```

## Interpretation

These embedded weights are not useful dense candidate sources. They are far below the current reliable region:

| Dataset | Current reliable best | Best embedded BasicIRSTD weight |
| --- | ---: | ---: |
| IRSTD-1K | 73.0367 | 31.0811 |
| NUAA-SIRST | 79.5487 latest BGM candidate screen / 79.34 previous reliable | 6.1357 |

Do not add these two embedded weights to the component pool.

## Cleanup

No checkpoint, prediction map archive, or large output was created. Only small JSON/report files were added. The embedded weights remain inside `external/BasicIRSTD` because they were already part of the cloned repository.

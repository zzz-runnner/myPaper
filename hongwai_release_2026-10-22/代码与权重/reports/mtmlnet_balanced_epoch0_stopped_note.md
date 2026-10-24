# MTMLNet Balanced Epoch-0 Stop Note

Generated: 2026-06-13 02:45 UTC

These three MTMLNet balanced-target branches were stopped after epoch-0 evidence and independent IRSTD/NUAA evaluation. They were using the full multi-dataset MTMLNet-format training set with target balancing, but the segmentation performance was far below the verified SCTransNet baselines.

## Branches

| Branch | Init | det_gain | seg_gain | Official val mIoU | mAP50 | IRSTD-1K independent mIoU | NUAA-SIRST independent mIoU |
|---|---|---:|---:|---:|---:|---:|---:|
| `mtm_balanced_s512_b3_seg08_adamw` | scratch | 0.20 | 0.80 | 0.46814 | 0.60492 | 43.7144 | 60.6506 |
| `mtm_balanced_s512_b3_seg09_adamw` | scratch | 0.10 | 0.90 | 0.51374 | 0.62097 | 43.6373 | 54.8185 |
| `mtm_balanced_fromseg_s512_b3_seg08_adamw` | seg-heavy best | 0.20 | 0.80 | 0.24414 | 0.69492 | not separately improved | not separately improved |

## Decision

The strongest balanced branch was still about 24 mIoU below the IRSTD-1K SCTransNet baseline and about 17 mIoU below the NUAA-SIRST baseline. The detection head learned usable boxes, but the segmentation head did not show a plausible path to the requested +5 mIoU target. The GPU time is better spent on the teacher/pseudo-label innovation line.

## Preserved Evidence

Small evidence files were copied under:

```text
/home/cx/hongwai/irstd-lab/reports/mtmlnet_balanced_preserved/
```

The bulky run directories were deleted after preserving results, configs, logs, and independent evaluation JSON files.

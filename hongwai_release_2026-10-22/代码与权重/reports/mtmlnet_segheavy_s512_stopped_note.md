# MTMLNet Seg-Heavy 512 Stopped Note

Generated: 2026-06-13 02:13 UTC

Stopped branch:

```text
/home/cx/hongwai/irstd-lab/runs/mtmlnet_full_4gpu/mtm_segheavy_s512_b3_adamw
```

Reason:

- This branch was the best early MTMLNet signal, but it remained far below SCTransNet on the original IRSTD-1K and NUAA-SIRST test splits.
- Official MTMLNet validation mIoU peaked at epoch 1 and dropped at epoch 2:

| Epoch | Official MTMLNet mIoU | Detection mAP@0.5 |
|---:|---:|---:|
| 0 | 0.52381 | 0.50052 |
| 1 | 0.53051 | 0.60829 |
| 2 | 0.49701 | 0.62850 |

Independent original-split segmentation eval:

| Timing | Dataset | Best mIoU | Best threshold |
|---|---|---:|---:|
| epoch0 best | IRSTD-1K | 42.40 | 0.600 |
| epoch0 best | NUAA-SIRST | 60.91 | 0.300 |
| later best checkpoint | IRSTD-1K | 46.42 | 0.700 |
| later best checkpoint | NUAA-SIRST | 57.58 | 0.800 |

Preserved small evidence files:

```text
/home/cx/hongwai/irstd-lab/reports/mtmlnet_segheavy_s512_preserved/results.csv
/home/cx/hongwai/irstd-lab/reports/mtmlnet_segheavy_s512_preserved/opt.yaml
/home/cx/hongwai/irstd-lab/reports/mtmlnet_segheavy_s512_preserved/hyp.yaml
/home/cx/hongwai/irstd-lab/reports/mtmlnet_segheavy_s512_preserved/train.log
```

Decision:

- Stop this weak branch and delete the bulky run directory.
- Keep GPU0 available for a more promising teacher/pseudo-label/distillation line instead of continuing a branch that is not closing the gap.


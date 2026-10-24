# MTMLNet s640 SGD Stopped Note

Generated: 2026-06-13 01:10 UTC

Stopped branch:

```text
/home/cx/hongwai/irstd-lab/runs/mtmlnet_full_4gpu/mtm_full_s640_b2_sgd
```

Reason:

- This was the 640px, batch 2, SGD MTMLNet full-data branch on GPU1.
- It completed epoch 0 and wrote a checkpoint successfully.
- First validation was clearly failed:
  - detection precision 0.039583
  - detection recall 0.45179
  - mAP@0.5 0.027621
  - mAP@0.5:0.95 0.008664
  - val/cls_loss 1283.6, abnormal
  - segmentation mIoU 0.0059368
- It is much worse than the parallel 512 AdamW-A branch and should not keep GPU1 occupied.

Preserved one-line results below before deleting the bulky checkpoint directory:

```csv
               epoch,      train/box_loss,      train/cls_loss,      train/dfl_loss,   metrics/precision,      metrics/recall,     metrics/mAP_0.5,metrics/mAP_0.5:0.95,        val/box_loss,        val/cls_loss,        val/dfl_loss,               x/lr0,               x/lr1,               x/lr2,            msegloss,                mIoU
                   0,              2.8902,              3.2322,              2.1545,            0.039583,             0.45179,            0.027621,            0.008664,              2.9503,              1283.6,              1.9019,            0.070018,           0.0033313,           0.0033313,            0.065329,           0.0059368
```

Decision: stop GPU1 branch and remove its run directory to free GPU and disk.

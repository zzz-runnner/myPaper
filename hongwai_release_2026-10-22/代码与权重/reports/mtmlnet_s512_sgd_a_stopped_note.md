# MTMLNet s512 SGD-A Stopped Note

Generated: 2026-06-13 00:58 UTC

Stopped branch:

```text
/home/cx/hongwai/irstd-lab/runs/mtmlnet_full_4gpu/mtm_full_s512_b3_sgd_a
```

Reason:

- This was the 512px, batch 3, SGD MTMLNet full-data branch on GPU0.
- It completed epoch 0 and wrote a checkpoint successfully.
- First validation was clearly failed for segmentation:
  - train/box_loss 3.0281
  - train/cls_loss 3.2592
  - train/dfl_loss 2.0354
  - detection precision 0.024247
  - detection recall 0.46786
  - mAP@0.5 0.019105
  - mAP@0.5:0.95 0.005723
  - val/cls_loss 3.164e+05, abnormal
  - segmentation mIoU 0.00082194
- Compared with the parallel AdamW-A branch mIoU 0.33252 after epoch 0, this branch is not worth more GPU time.

Preserved one-line results below before deleting the bulky checkpoint directory:

```csv
               epoch,      train/box_loss,      train/cls_loss,      train/dfl_loss,   metrics/precision,      metrics/recall,     metrics/mAP_0.5,metrics/mAP_0.5:0.95,        val/box_loss,        val/cls_loss,        val/dfl_loss,               x/lr0,               x/lr1,               x/lr2,            msegloss,                mIoU
                   0,              3.0281,              3.2592,              2.0354,            0.024247,             0.46786,            0.019105,            0.005723,              2.9557,           3.164e+05,              1.5369,            0.070054,           0.0033274,           0.0033274,            0.067744,          0.00082194
```

Decision: stop GPU0 branch and remove its run directory to free GPU and disk.

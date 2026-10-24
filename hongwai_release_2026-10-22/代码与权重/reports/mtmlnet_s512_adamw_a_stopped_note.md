# MTMLNet s512 AdamW-A Stopped Note

Generated: 2026-06-13 02:04 UTC

Stopped branch:

```text
/home/cx/hongwai/irstd-lab/runs/mtmlnet_full_4gpu/mtm_full_s512_b3_adamw_a
```

Reason:

- This was the original 512px, batch 3, AdamW full multi-dataset MTMLNet branch on GPU2.
- It improved early, but official segmentation mIoU peaked at epoch 2 and then dropped at epoch 3:
  - epoch 0: mIoU `0.33252`
  - epoch 1: mIoU `0.45366`
  - epoch 2: mIoU `0.49843`
  - epoch 3: mIoU `0.39254`
- Independent evaluation at epoch 1 was still far below SCTransNet:
  - IRSTD-1K mIoU `39.76`
  - NUAA-SIRST mIoU `53.27`
- The segmentation-heavy 512 branch is stronger on independent target-set evaluation, so this generic branch should not continue occupying GPU2.

Preserved evidence:

```text
/home/cx/hongwai/irstd-lab/reports/mtmlnet_s512_adamw_a_preserved/results.csv
/home/cx/hongwai/irstd-lab/reports/mtmlnet_s512_adamw_a_preserved/opt.yaml
/home/cx/hongwai/irstd-lab/reports/mtmlnet_s512_adamw_a_preserved/hyp.yaml
/home/cx/hongwai/irstd-lab/reports/mtmlnet_s512_adamw_a_epoch0_irstd_eval.json
/home/cx/hongwai/irstd-lab/reports/mtmlnet_s512_adamw_a_epoch0_nuaa_eval.json
/home/cx/hongwai/irstd-lab/reports/mtmlnet_s512_adamw_a_epoch1_irstd_eval.json
/home/cx/hongwai/irstd-lab/reports/mtmlnet_s512_adamw_a_epoch1_nuaa_eval.json
```

Decision: stop GPU2 branch and remove the bulky run directory after preserving result artifacts.

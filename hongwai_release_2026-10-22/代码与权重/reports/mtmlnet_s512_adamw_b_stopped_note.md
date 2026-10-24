# MTMLNet s512 AdamW-B Stopped Note

Generated: 2026-06-13 01:55 UTC

Stopped branch:

```text
/home/cx/hongwai/irstd-lab/runs/mtmlnet_full_4gpu/mtm_full_s512_b3_adamw_b
```

Reason:

- This was the 512px, batch 3, AdamW, cosine-LR full multi-dataset MTMLNet branch on GPU3.
- It was the same recipe family as `mtm_full_s512_b3_adamw_a`, but a different seed.
- After three completed validations, its segmentation mIoU was consistently much lower than the better active branches:
  - epoch 0: mIoU `0.12247`
  - epoch 1: mIoU `0.27620`
  - epoch 2: mIoU `0.35976`
- At the same point, `mtm_full_s512_b3_adamw_a` had reached mIoU `0.49843`, and the segmentation-heavy 512 branch had already reached mIoU `0.53051` by epoch 1.
- Continuing this duplicate weak seed is a poor use of one RTX 2080 Ti while the goal is to explore stronger multi-dataset external/domain-transfer directions.

Preserved evidence:

```text
/home/cx/hongwai/irstd-lab/reports/mtmlnet_s512_adamw_b_preserved/results.csv
/home/cx/hongwai/irstd-lab/reports/mtmlnet_s512_adamw_b_preserved/opt.yaml
/home/cx/hongwai/irstd-lab/reports/mtmlnet_s512_adamw_b_preserved/hyp.yaml
```

Decision: stop GPU3 branch and remove the bulky run directory after preserving the report artifacts.

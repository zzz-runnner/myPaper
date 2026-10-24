# MTMLNet Seg-Heavy s640 Stopped Note

Generated: 2026-06-13 02:02 UTC

Stopped branch:

```text
/home/cx/hongwai/irstd-lab/runs/mtmlnet_full_4gpu/mtm_segheavy_s640_b2_adamw
```

Reason:

- This was the 640px, batch 2, AdamW, segmentation-heavy MTMLNet full multi-dataset branch on GPU1.
- First official validation was not competitive with the 512px segmentation-heavy branch:
  - official val mIoU `0.40480`
  - official mAP@0.5 `0.61033`
- Independent original-split evaluation was also weak:
  - IRSTD-1K mIoU `39.27` at threshold `0.900`
  - NUAA-SIRST mIoU `48.60` at threshold `0.900`
- The 512px segmentation-heavy branch already reached independent IRSTD-1K `46.42`, so the 640px branch is not worth another long epoch on an 11GB RTX 2080 Ti.

Preserved evidence:

```text
/home/cx/hongwai/irstd-lab/reports/mtmlnet_segheavy_s640_preserved/results.csv
/home/cx/hongwai/irstd-lab/reports/mtmlnet_segheavy_s640_preserved/opt.yaml
/home/cx/hongwai/irstd-lab/reports/mtmlnet_segheavy_s640_preserved/hyp.yaml
/home/cx/hongwai/irstd-lab/reports/mtmlnet_segheavy_s640_epoch0_irstd_eval.json
/home/cx/hongwai/irstd-lab/reports/mtmlnet_segheavy_s640_epoch0_nuaa_eval.json
```

Decision: stop GPU1 branch and remove the bulky run directory after preserving result artifacts.

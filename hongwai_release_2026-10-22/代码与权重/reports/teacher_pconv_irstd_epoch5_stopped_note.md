# IRSTD PConvTCF-TPCDHiF Teacher Stopped Note

Generated: 2026-06-13 03:26 UTC

Stopped branch:

```text
/home/cx/hongwai/irstd-lab/runs/teacher_pconv_irstd_gpu2/IRSTD-1K_PConvTCF-TPCDHiFTeacher_gpu2
```

Reason:

- This was a full multi-dataset PConvTCF student on IRSTD-1K with an SCT-TPC-DHiF teacher.
- First validation at epoch 5 was far below the verified baseline:
  - epoch 5 mIoU: `0.521185` / `52.12`
  - official IRSTD baseline: `68.03`
  - current IRSTD best: `68.28`
- The branch is about 16 mIoU below baseline, so it is not a good use of one RTX 2080 Ti.

Preserved evidence:

```text
/home/cx/hongwai/irstd-lab/reports/teacher_pconv_irstd_epoch5_preserved/
```

Decision: stop the branch, remove the bulky run directory, and replace GPU2 with a second TCD+LCA+BNG/FeatUp strong-checkpoint refiner variant.

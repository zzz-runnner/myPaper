# IRSTD APConvTCF-TPCDHiF Teacher Stopped Note

Generated: 2026-06-13 03:20 UTC

Stopped branch:

```text
/home/cx/hongwai/irstd-lab/runs/teacher_apconv_gpu0/IRSTD-1K_APConvTCF-TPCDHiFTeacher_gpu0
```

Reason:

- This was a full multi-dataset APConvTCF student on IRSTD-1K with an SCT-TPC-DHiF teacher.
- It recovered from epoch 5 but remained far below the verified SCTransNet baseline and current best:
  - epoch 5 mIoU: `0.489734` / `48.97`
  - epoch 10 mIoU: `0.577059` / `57.71`
  - official IRSTD baseline: `68.03`
  - current IRSTD best: `68.28`
- The gap is still about `-10.3 mIoU` to baseline and `-10.6 mIoU` to current best, so continuing to epoch 120 is not a good use of one RTX 2080 Ti.

Preserved evidence:

```text
/home/cx/hongwai/irstd-lab/reports/teacher_apconv_irstd_epoch10_preserved/
```

Decision: stop the branch, remove the bulky run directory, and replace GPU0 with the TCD+LCA+BNG/FeatUp strong-checkpoint refiner line.

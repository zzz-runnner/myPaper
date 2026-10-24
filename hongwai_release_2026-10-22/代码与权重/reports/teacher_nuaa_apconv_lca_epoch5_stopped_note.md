# NUAA APConvTCF-LCA Teacher Stopped Note

Generated: 2026-06-13 03:15 UTC

Stopped branch:

```text
/home/cx/hongwai/irstd-lab/runs/teacher_nuaa_gpu1/NUAA-SIRST_APConvTCF-LCATeacher_gpu1
```

Reason:

- This was a full multi-dataset APConvTCF student on NUAA-SIRST with an LCA SCTransNet-family teacher.
- First validation at epoch 5 was far below the verified SCTransNet/LCA baseline:
  - epoch 5 mIoU: `0.621322` / `62.13`
  - current NUAA best: `79.13`
  - gap: about `-17.8 mIoU`
- The goal requires exploring stronger full-data innovation lines and keeping the four GPUs useful. This branch is not on a plausible path to +5.

Preserved evidence:

```text
/home/cx/hongwai/irstd-lab/reports/teacher_nuaa_apconv_lca_epoch5_preserved/
```

Decision: stop the branch, remove the bulky run directory, and replace GPU1 with the TCD+LCA+BNG/FeatUp pseudo-label refiner line.

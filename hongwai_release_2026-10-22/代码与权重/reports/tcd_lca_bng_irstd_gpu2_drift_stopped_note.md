# IRSTD TCD-LCA-BNG SeedB Drift Stop Note

Generated: 2026-06-13 03:39 UTC

Stopped branch: `tcd_lca_bng_irstd_gpu2`

Reason:

- This was a second strong-checkpoint TCD/LCA/BNG IRSTD-1K variant.
- It started at epoch 0 mIoU `68.23`, then dropped to epoch 2 mIoU `65.88`.
- The drift confirmed the original composition loss/LR was still too aggressive even with a lower LR than GPU0.

Decision: preserve evidence, stop the run, and replace it with a stricter teacher-logit-anchored calibration variant.

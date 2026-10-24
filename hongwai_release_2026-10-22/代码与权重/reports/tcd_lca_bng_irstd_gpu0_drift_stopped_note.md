# IRSTD TCD-LCA-BNG Drift Stop Note

Generated: 2026-06-13 03:35 UTC

Stopped branch: `tcd_lca_bng_irstd_gpu0`

Reason:

- This was the first strong-checkpoint TCD/LCA/BNG composition branch for IRSTD-1K.
- It started safely at epoch 0 mIoU `68.23`, but training pulled it down to epoch 2 mIoU `64.96`.
- The branch proved the composition can load and evaluate, but the loss/LR recipe damages the strong base.

Decision: preserve evidence, stop the run, and replace it with a lower-LR, teacher-logit-anchored calibration variant.

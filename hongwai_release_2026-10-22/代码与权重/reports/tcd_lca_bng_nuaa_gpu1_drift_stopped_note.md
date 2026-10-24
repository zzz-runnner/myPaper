# NUAA TCD-LCA-BNG Drift Stop Note

Generated: 2026-06-13 03:35 UTC

Stopped branch: `tcd_lca_bng_nuaa_gpu1`

Reason:

- This was the first strong-checkpoint TCD/LCA/BNG + pseudo-label composition branch for NUAA-SIRST.
- It started at epoch 0 mIoU `79.11`, then dropped to epoch 2 `78.80` and epoch 4 `78.17`.
- It was not a collapse, but it was drifting away from the best known checkpoint instead of improving it.

Decision: preserve evidence, stop the run, and replace it with a lower-LR, teacher-logit-anchored calibration variant.

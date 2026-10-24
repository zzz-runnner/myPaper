# Next Innovation Conservative Batch Stop Note

Generated: 2026-06-13 UTC

Stopped conservative anchored full multi-dataset runs under `runs/next_innovation_4gpu`.

All runs used `IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3` as training datasets and combined external-domain modules such as spectral anomaly, heat diffusion prior, local-preserving state mixing, background normality, and SAM-refined pseudo labels.

Reason for stop:

- `NextSpectral-IRSTD`: best epoch 0 mIoU 68.2836, then flat/slightly down through epoch 5; below reliable best 69.17.
- `NextDiffusion-IRSTD`: improved from 68.1634 to 68.2655 by epoch 4, but still far below reliable best 69.17 and too conservative.
- `NextSpectral-NUAA`: best epoch 5 mIoU 79.1252, then drifted down; below reliable best 79.34.
- `NextLPSM-NUAA-RetryB12`: best epoch 4 mIoU 79.1235, essentially flat; below reliable best 79.34.
- First `NextLPSM-NUAA` attempt with batch 26 OOMed, then was relaunched with batch 12.

Decision:

These branches were not collapses, but they were teacher-anchored too tightly and did not create new candidates. All low-performing checkpoints from this batch were deleted; logs are retained as evidence.

Next step should use a more aggressive candidate-generation setup with weaker teacher anchoring and stronger prompt/diffusion/copy-paste pressure, then stop quickly if it drifts or fails to exceed current reliable best.

# Aggressive Candidate Batch Stop Note

Generated: 2026-06-13 UTC

Stopped aggressive full multi-dataset candidate-generation runs under `runs/aggressive_candidate_4gpu`.

All four runs used `IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3` and intentionally reduced teacher/base anchoring while increasing prompt morph, diffusion prior, SAM-refined pseudo-label, copy-paste, thermal spot shape, and high-frequency/background losses.

Reason for stop:

- `AggrPrompt-IRSTD`: best epoch 0 mIoU 68.2777; epoch 1/2 dropped to 68.2296/68.2219. Below reliable best 69.17.
- `AggrDiffusion-IRSTD`: best epoch 1 mIoU 68.2280; epoch 2 dropped to 68.1490. Below reliable best 69.17.
- `AggrPrompt-NUAA`: best epoch 1 mIoU 79.1252; epoch 2/3 dropped to 79.0852/79.0715. Below reliable best 79.34.
- `AggrDiffusion-NUAA`: best epoch 1 mIoU 79.1230; epoch 2/3 dropped to 79.0999/79.0857. Below reliable best 79.34.

Decision:

The aggressive branch did create train-time pressure, but it damaged precision/false-alarm balance instead of creating useful new validated candidates. All low-performing checkpoints from this batch were deleted; logs remain as evidence.

Next diagnostic: test whether much lower teacher candidate thresholds contain hidden true positives by running low-threshold component-fusion/oracle scans before spending more GPU training time.

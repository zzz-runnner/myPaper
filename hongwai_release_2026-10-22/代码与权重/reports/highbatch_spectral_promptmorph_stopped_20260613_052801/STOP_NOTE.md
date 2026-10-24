# High-batch Spectral / PromptMorph Stop Note

Stopped after early full multi-dataset evidence.

- SpectralAnomalyHB-IRSTD-A: best epoch2 mIoU 0.682817, a tiny +~0.00024 over the previous 0.682575/0.6828 region, but then drifted down to 0.682249 by epoch13. Not a meaningful route toward +5.
- PromptMorph-IRSTD-B: best epoch1 mIoU 0.682510, then drifted to about 0.6819-0.6820.
- PromptMorph-NUAA-LCA: best epoch0 mIoU 0.791208, then drifted to about 0.7906-0.7907.
- PromptMorph-NUAA-TCD: best epoch0 mIoU 0.791208, then drifted to about 0.7906-0.7907.

All lines used full train_datasets=IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3 and high-batch GPU utilization around 7-8GB/card. Evidence was preserved here before deleting bulky run dirs.

# DiffusionPrior High-Batch Stop Note

Stopped after early full multi-dataset evidence on 2026-06-13 05:48 UTC.

All four branches used `train_datasets=IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3` with high GPU utilization around 6.8 GB / RTX 2080 Ti.

| Branch | Best observed | Later signal | Decision |
|---|---:|---:|---|
| `DiffusionPrior-IRSTD-A` | epoch1 mIoU 0.682634 | epoch2 mIoU 0.681530 | did not beat the current IRSTD best region around 0.6828 and started drifting down |
| `DiffusionPrior-IRSTD-B` | epoch0 mIoU 0.682456 | epoch2 mIoU 0.682177 | below current best and drifting |
| `DiffusionPrior-NUAA-LCA` | epoch0 mIoU 0.791145 | epoch3 mIoU 0.790596 | below/tied current NUAA best and drifting |
| `DiffusionPrior-NUAA-TCD` | epoch0 mIoU 0.791208 | epoch3 mIoU 0.790680 | below/tied current NUAA best and drifting |

Conclusion: the heat-diffusion prior branch loads and trains correctly, but it follows the same plateau/drift pattern as the spectral and prompt-morph branches. It is not a useful route toward the requested +5 mIoU target in this configuration.

The bulky run directory was deleted after preserving logs and best checkpoints here.

# LPSSM High-Batch Stop Note

Stopped after early full multi-dataset evidence on 2026-06-13 06:04 UTC.

The branch was inspired by MambaIR/MSVMamba-style local-preserving state-space modeling. All four runs used `train_datasets=IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3` and reached high utilization around 9.0-9.2 GB / RTX 2080 Ti.

| Branch | Best observed | Later signal | Decision |
|---|---:|---:|---|
| `LPSM-IRSTD-A` | epoch0 mIoU 0.682535 | epoch1 mIoU 0.682457 | below current IRSTD best region around 0.6828 and not improving |
| `LPSM-IRSTD-B` | epoch0 mIoU 0.681751 | epoch1 mIoU 0.681732 | below current best and flat |
| `LPSM-NUAA-LCA` | epoch0/1 mIoU 0.791230 | epoch2 mIoU 0.791105 | tied/below current NUAA best and drifting |
| `LPSM-NUAA-TCD` | epoch2 mIoU 0.791177 | epoch2 best still below 0.7913 region | small rise but no breakthrough |

Conclusion: the local-preserving state-space branch fits 2080 Ti and trains correctly, but it does not open a new performance region. It was stopped to avoid wasting four GPUs on another plateau branch.

The bulky run directory was deleted after preserving logs and best checkpoints here.

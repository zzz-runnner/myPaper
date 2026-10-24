# Copy-Paste Hard-Mining Stop Note

Stopped after early full multi-dataset evidence on 2026-06-13 around 07:15 UTC.

This branch implemented a real-data thermal target instance copy-paste augmentation, inspired by weak/active/easy-hard data mechanisms rather than SCTransNet parameter tuning. All four runs used full `train_datasets=IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3` plus SAM soft pseudo labels and teacher anchoring.

| Branch | Best observed | Later signal | Decision |
|---|---:|---:|---|
| `CopyPaste-TPC-IRSTD-A` | epoch0 mIoU 0.682858 | epoch2 mIoU 0.682297 | no gain after training |
| `CopyPaste-TPC-IRSTD-B` | epoch1 mIoU 0.682199 | epoch2 mIoU 0.682101 | below current best |
| `CopyPaste-LCA-NUAA` | epoch0 mIoU 0.791230 | epoch4 mIoU 0.791128 | no gain |
| `CopyPaste-TCD-NUAA` | epoch2 mIoU 0.791230 | epoch3 mIoU 0.791168 | no gain |

Interpretation: high-probability copy-paste increases target diversity but perturbs the already-strong target-domain distribution too much. Next attempt should make copy-paste curriculum-like: light on the target dataset, stronger on auxiliary datasets only.

The bulky run directory was deleted after preserving logs and best checkpoints here.

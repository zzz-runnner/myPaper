# Copy-Paste Curriculum Stop Note

Stopped after early full multi-dataset evidence on 2026-06-13 around 07:36 UTC.

This was the conservative follow-up to high-probability target copy-paste. It used per-dataset copy-paste probabilities: light augmentation on the target dataset and stronger augmentation on auxiliary datasets, with full `train_datasets=IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3`.

| Branch | Best observed | Later signal | Decision |
|---|---:|---:|---|
| `CPCurr-TPC-IRSTD-A` | epoch0 mIoU 0.682485 | epoch5 mIoU 0.680961 | drifted down |
| `CPCurr-TPC-IRSTD-B` | epoch3 mIoU 0.682601 | epoch4 mIoU 0.682361 | below current best |
| `CPCurr-LCA-NUAA` | epoch7 mIoU 0.791471 | epoch9 mIoU 0.791346 | noise-level only |
| `CPCurr-TCD-NUAA` | epoch7 mIoU 0.791426 | no breakthrough | noise-level only |

Interpretation: curriculum copy-paste is safer than hard copy-paste, but it still does not open a new performance region. It is not a viable route toward +5 mIoU in this setup.

The bulky run directory was deleted after preserving logs and best checkpoints here.

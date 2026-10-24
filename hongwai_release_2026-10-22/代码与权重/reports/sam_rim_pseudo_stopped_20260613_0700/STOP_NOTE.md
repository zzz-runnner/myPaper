# SAM-Rim Pseudo Stop Note

Stopped after early full multi-dataset evidence on 2026-06-13 around 07:00 UTC.

This line used foundation-model/SAM-assisted soft-rim pseudo labels, not just SCTransNet parameter tuning. Pseudo labels covered the full training pool: `IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3`.

| Branch | Best observed | Later signal | Decision |
|---|---:|---:|---|
| `SAMRim-TPCDHiF-IRSTD-A` | epoch0 mIoU 0.682638 | epoch5 mIoU 0.675108 | drifted down strongly |
| `SAMRim-TPCDHiF-IRSTD-B` | epoch1 mIoU 0.683039 | epoch5 mIoU 0.679268 | tiny early bump, then drifted down |
| `SAMRim-LCA-NUAA` | epoch5 mIoU 0.791515 | epoch10 mIoU 0.791018 | only noise-level gain, then drifted |
| `SAMRim-TCD-NUAA` | epoch4 mIoU 0.791471 | epoch7 mIoU 0.790889 | only noise-level gain, then drifted |

Conclusion: hard SAM masks over-expanded tiny targets, so the final tried version used only conservative soft-rim supervision. It trains correctly and fits the 2080 Ti GPUs, but it does not create a meaningful improvement region. The bulky run directory was deleted after preserving logs and best checkpoints here.

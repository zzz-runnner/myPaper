# Current Status After TCD/LCA/BNG Replacement

Generated: 2026-06-13 03:22 UTC

## What Changed This Round

- Added a new `sct_tcd_lca_bng_refiner` model in `scripts/train_hybrid_methods.py`.
- The new branch combines:
  - in-field TCD target/clutter decoupling,
  - LCA local-contrast anomaly calibration,
  - FeatUp-style multi-scale high-resolution recovery,
  - background-normality memory guidance,
  - teacher preservation, base anchoring, pseudo-label loss, and thermal spot shape loss.
- Added launch scripts:
  - `scripts/launch_tcd_lca_bng_irstd_gpu0_tmux.sh`
  - `scripts/launch_tcd_lca_bng_nuaa_gpu1_tmux.sh`

## Stopped Weak Branches

### IRSTD APConvTCF-TPCDHiF Teacher

Stopped session: `teacher_apconv_gpu0`

Reason:

- epoch 5 mIoU: `48.97`
- epoch 10 mIoU: `57.71`
- official IRSTD baseline: `68.03`
- current IRSTD best: `68.28`

The branch was still more than 10 mIoU below baseline at epoch 10, so it was stopped and replaced.

Preserved evidence:

```text
reports/teacher_apconv_irstd_epoch10_preserved/
reports/teacher_apconv_irstd_epoch10_stopped_note.md
```

### NUAA APConvTCF-LCA Teacher

Stopped session: `teacher_nuaa_gpu1`

Reason:

- epoch 5 mIoU: `62.13`
- current NUAA best: `79.13`

The branch was about 17.8 mIoU below the current best on the first validation, so it was stopped and replaced.

Preserved evidence:

```text
reports/teacher_nuaa_apconv_lca_epoch5_preserved/
reports/teacher_nuaa_apconv_lca_epoch5_stopped_note.md
```

## Current Active Sessions

| GPU | Session | Dataset | Method | Current signal |
|---:|---|---|---|---|
| 0 | `tcd_lca_bng_irstd_gpu0` | IRSTD-1K | `TCD-LCA-BNG-TPCDHiFPseudo` | epoch 0 eval-before-train mIoU `68.23` |
| 1 | `tcd_lca_bng_nuaa_gpu1` | NUAA-SIRST | `TCD-LCA-BNG-LCATCDPseudo` | epoch 0 eval-before-train mIoU `79.11`, epoch 1 training done |
| 2 | `tcd_lca_bng_irstd_gpu2` | IRSTD-1K | `TCD-LCA-BNG-TPCDHiF-SeedB` | epoch 0 eval-before-train mIoU `68.23` |
| 3 | `pseudo_nuaa_gpu3` | NUAA-SIRST | `PConvTCF-LCATCDPseudo` | epoch 5 mIoU `68.74`, epoch 6 running |

## Interpretation

- The two APConv student branches were not competitive and were removed after preserving evidence.
- The PConv student branch on GPU2 was also removed after its first validation:
  - epoch 5 mIoU: `52.12`
  - preserved under `reports/teacher_pconv_irstd_epoch5_preserved/`
- The two new TCD/LCA/BNG strong-checkpoint branches have stable starting points:
  - IRSTD starts around the current best region but has not surpassed it yet.
  - NUAA starts near the current best region but has not surpassed it yet.
- A second IRSTD TCD/LCA/BNG seed was launched on GPU2 with a lower LR and stronger anchoring.
- GPU2 and GPU3 are still being allowed to reach their next evidence points:
  - GPU3 should be judged again at epoch 10 unless it collapses sooner.

## Next Decision Rules

- If GPU3 remains far below NUAA best by epoch 10, stop and preserve evidence.
- Keep GPU0/GPU1/GPU2 combination branches running through the first few short eval intervals because their epoch-0 checkpoints did not collapse.

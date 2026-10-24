# Current Full Status Report

Generated: 2026-06-13 03:53 UTC

## Executive Summary

The project environment and datasets are set up under `/home/cx/hongwai/irstd-lab`.
The current target remains to improve infrared small-target segmentation on IRSTD-1K and NUAA-SIRST, ideally by about +5 mIoU over SCTransNet.

That target has not been reached.

Best reliable results so far:

| Dataset | Best method so far | Best mIoU | Reference SCTransNet | Delta |
|---|---|---:|---:|---:|
| IRSTD-1K | `SCT-TPC-DHiF-B` | 68.28 | 68.03 | +0.25 |
| NUAA-SIRST | `SCT-LCA-B` / `SCT-TCD-Decouple` family | 79.13 | 77.50 | +1.63 |

Current live state is not idle. Four tmux training sessions are still running, one per RTX 2080 Ti.

## Disk / Runtime State

Latest disk check:

| Mount | Total | Used | Available | Use |
|---|---:|---:|---:|---:|
| `/` | 437G | 357G | 62G | 86% |
| `/data1` | 3.6T | 195G | 3.4T | 6% |

Current project sizes:

| Path | Size |
|---|---:|
| `/home/cx/hongwai/irstd-lab` | 8.6G |
| `runs` | 806M |
| `reports` | 176M |
| `generated` | 454M |
| `datasets` | 517M |
| `envs` | 5.8G |

Dataset storage recommendation remains: put large future datasets under `/data1`, not under `/`.

## Environment / Code Prepared

Python environment:

```text
/home/cx/hongwai/irstd-lab/envs/irstd
```

External repositories prepared:

```text
external/SCTransNet
external/BGM
external/MSHNet
external/PConv-SDloss-Data
external/MTMLNet
```

Key implemented scripts / changes:

- `scripts/train_hybrid_methods.py`: main hybrid training/evaluation driver.
- Added pseudo-label support in dataset/loss path.
- Added `sct_tcd_lca_bng_refiner`, combining TCD target/clutter decoupling, LCA local contrast anomaly, FeatUp/background normality, teacher/base anchoring, pseudo labels, and thermal spot losses.
- Added `--teacher_logit_weight` to constrain student logits near a strong teacher.
- `scripts/generate_teacher_pseudo_labels.py`: generated offline pseudo labels.
- `scripts/eval_selective_postprocess.py`: exploratory postprocessing sweep script; the first large sweep was stopped because it was too slow and had no result file yet.
- MTMLNet helper files:
  - `scripts/prepare_mtmlnet_dataset.py`
  - `scripts/eval_mtmlnet_segmentation.py`
  - `scripts/launch_mtmlnet_full_4gpu_tmux.sh`
  - `configs/mtmlnet_hyp_sgd.yaml`
  - `configs/mtmlnet_hyp_adamw.yaml`
- Patched `external/MTMLNet/val.py` to create metric output directory before writing segmentation CSV.

## Data Prepared

Datasets under `/home/cx/hongwai/irstd-lab/datasets`:

| Dataset | Total | Train | Test |
|---|---:|---:|---:|
| IRSTD-1K | 1001 | 800 | 201 |
| NUAA-SIRST | 427 | 213 | 214 |
| NUDT-SIRST | 1327 | 663 | 664 |
| SIRST3 | 2755 | 1676 | 1079 |

One NUAA raw-data issue was fixed earlier: `Misc_111` image/mask size mismatch; mask was resized with nearest-neighbor interpolation.

Pseudo labels:

```text
/home/cx/hongwai/irstd-lab/generated/pseudo_nuaa_lca_tcd_train
```

This directory has 6705 files including `.npz`, `.json`, and `summary.json`.
It was generated from strong NUAA LCA/TCD teachers over the full training pool.

## Baseline Reproduction

Official SCTransNet checkpoint reproduced the README metrics closely:

| Dataset | README mIoU | Reproduced mIoU | README Pd | Reproduced Pd | README Fa | Reproduced Fa |
|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | 77.50 | 77.64 | 96.95 | 96.96 | 13.92 | 13.93 |
| NUDT-SIRST | 94.09 | 94.11 | 98.62 | 98.62 | 4.29 | 4.30 |
| IRSTD-1K | 68.03 | 68.03 | 93.27 | 93.27 | 10.74 | 10.74 |

Conclusion: dataset conversion/evaluation pipeline is aligned with the official SCTransNet setup.

## Completed From-Scratch SCTransNet Runs

Four 1000-epoch from-scratch SCTransNet runs completed earlier:

| Dataset | Epochs | Best epoch | Best mIoU | Reference | Delta |
|---|---:|---:|---:|---:|---:|
| SIRST3 | 1000 | 678 | 82.96 | n/a | n/a |
| NUDT-SIRST | 1000 | 547 | 93.76 | 94.09 | -0.33 |
| IRSTD-1K | 1000 | 500 | 67.32 | 68.03 | -0.71 |
| NUAA-SIRST | 1000 | 823 | 77.52 | 77.50 | +0.02 |

These proved the training path works but did not provide the desired large gain.

## SCTransNet-Family Experiments Tried

Main SCTransNet-side refiners/gates tried:

| Line | IRSTD best | NUAA best | Result |
|---|---:|---:|---|
| `SCT-Refiner-Fixed` | 68.09 | 79.12 | Small gains only |
| `SCT-DHiF-Balanced` | 68.05 | 79.11 | No meaningful improvement |
| `SCT-DHiF-FromFixed` | 68.09 | 79.12 | Same plateau |
| `SCT-DHiF-Multi` | 68.06 | 79.11 | Same plateau |
| `SCT-TPC-DHiF` | 68.28 | 79.12 | Best IRSTD so far |
| `SCT-LCA` | 68.09 | 79.13 | Best NUAA so far |
| `BNG/FeatUp` variants | 68.09 | 79.12 | No meaningful gain |
| `ExpertGate` / `ExpertOracleGate` | 68.10 | 79.12 | Gating did not unlock gain |
| `TCD-Decouple` | 68.09 | 79.13 | Tied NUAA best |

Interpretation: these checkpoints are highly correlated. Complementarity/oracle checks showed limited headroom, especially on NUAA.

## External / Non-SCTransNet Lines Tried

### Official MSHNet / PConv-MSHNet

This was a genuine external-method line based on official MSHNet/PConv-MSHNet.

| Dataset | Method | Best mIoU | Existing best target | Delta |
|---|---|---:|---:|---:|
| IRSTD-1K | MSHNet-D05-FullData | 67.29 | 68.28 | -0.99 |
| IRSTD-1K | MSHNet-P43-FullData | 66.70 | 68.28 | -1.58 |
| NUAA-SIRST | MSHNet-D05-FullData | 71.40 | 79.13 | -7.73 |
| NUAA-SIRST | MSHNet-P43-FullData | 72.77 | 79.13 | -6.36 |

Decision: stopped and deleted bulky run directory after preserving reports.

### Paired BGM

| Dataset | Method | Progress | Best mIoU | Reference | Delta |
|---|---|---:|---:|---:|---:|
| NUAA-SIRST | BGM-PairedBasicUNet | epoch 10 | 71.86 | 77.50 | -5.64 |

Decision: stopped and deleted bulky run directory after preserving reports.

### APConv + SDM + TCF

| Dataset | Method | Progress | Best mIoU | Reference | Delta |
|---|---|---:|---:|---:|---:|
| NUAA-SIRST | APConv-TCF-SDM | epoch 10 | 73.21 | 77.50 | -4.29 |

Decision: stopped and deleted bulky run directory after preserving reports.

### APConv/TPC Refiner and Unfrozen APConv/TPC

Results:

- Frozen SCT-APConvTPC started near strong checkpoints but degraded after training.
- Semi-unfrozen APConv/TPC immediately dropped:
  - IRSTD around 68.01/68.02 at epoch 0, then to 66.83/66.45.
  - NUAA around 79.10/79.11 at epoch 0, then to 75.35/76.04.

Decision: stopped and deleted bulky run directories after preserving reports.

### MTMLNet

MTMLNet was also tried as an official external detection+segmentation line.
The first launch had a GPU-device mapping problem and caused OOM; that was documented and relaunched correctly.

Several variants were then evaluated/preserved:

- full-data 512/640 SGD/AdamW variants;
- segmentation-heavy variants;
- balanced variants.

Observed MTMLNet segmentation mIoU was far below the current SCTransNet-family best, so those bulky runs were stopped/preserved/deleted.

Evidence remains in:

```text
reports/mtmlnet_*_stopped_note.md
reports/mtmlnet_*_preserved/
reports/mtmlnet_*_eval.json
```

## Recent Weak Branches Stopped

| Branch | Dataset | Stop point | Result | Evidence |
|---|---|---:|---:|---|
| APConvTCF-TPCDHiFTeacher | IRSTD-1K | epoch 10 | 57.71 | `reports/teacher_apconv_irstd_epoch10_preserved/` |
| APConvTCF-LCATeacher | NUAA-SIRST | epoch 5 | 62.13 | `reports/teacher_nuaa_apconv_lca_epoch5_preserved/` |
| PConvTCF-TPCDHiFTeacher | IRSTD-1K | epoch 5 | 52.12 | `reports/teacher_pconv_irstd_epoch5_preserved/` |
| TCD/LCA/BNG first seed | IRSTD-1K | epoch 2 | 64.96 after starting 68.23 | `reports/tcd_lca_bng_irstd_gpu0_drift_preserved/` |
| TCD/LCA/BNG first seed | NUAA-SIRST | epoch 4 | 78.17 after starting 79.11 | `reports/tcd_lca_bng_nuaa_gpu1_drift_preserved/` |
| TCD/LCA/BNG second IRSTD seed | IRSTD-1K | epoch 2 | 65.88 after starting 68.23 | `reports/tcd_lca_bng_irstd_gpu2_drift_preserved/` |

## Current Active Runs

As of the latest check, four tmux sessions are active and occupy all four RTX 2080 Ti GPUs.

| GPU | tmux session | Dataset | Method | Latest signal |
|---:|---|---|---|---|
| 0 | `cal_tcd_lca_bng_irstd_gpu0` | IRSTD-1K | calibrated TCD/LCA/BNG | epoch 3 mIoU 68.199; best epoch 0 mIoU 68.238 |
| 1 | `cal_tcd_lca_bng_nuaa_gpu1` | NUAA-SIRST | calibrated TCD/LCA/BNG + pseudo | epoch 4 mIoU 78.478; best epoch 0 mIoU 79.121 |
| 2 | `cal_tcd_lca_bng_irstd_gpu2` | IRSTD-1K | stricter calibrated TCD/LCA/BNG | epoch 2 mIoU 67.973; best epoch 0 mIoU 68.239 |
| 3 | `pseudo_nuaa_gpu3` | NUAA-SIRST | PConvTCF + LCA/TCD pseudo labels | epoch 15 mIoU 75.565; epoch 16 running |

Interpretation:

- GPU0/GPU2 IRSTD calibrated lines are stable-ish but still below the existing best 68.28.
- GPU1 NUAA calibrated line started near the best but has drifted down by epoch 4.
- GPU3 pseudo-label PConv line is improving from 68.74 to 73.83 to 75.57, but remains well below the NUAA best 79.13.

## Postprocessing Sweep Attempt

I launched two exploratory selective postprocessing sweeps:

- IRSTD-1K on `SCT-TPC-DHiF-B`
- NUAA-SIRST on `SCT-LCA-B`

They tested threshold, connected-component area, local contrast, and Laplacian filters.
The first sweep grid was too large and slow: after about 7 minutes no JSON result was produced.
I stopped both ordinary sweep processes so they would not keep occupying GPU0/GPU1.

This should be treated as an attempted but not completed result. It does not support any performance claim yet.

## Honest Current Conclusion

1. The environment, datasets, baselines, and evaluation path are properly set up.
2. Official SCTransNet reproduction is trustworthy.
3. We have not achieved +5 mIoU.
4. Best real gains are modest:
   - IRSTD-1K: +0.25 mIoU.
   - NUAA-SIRST: +1.63 mIoU.
5. Many SCTransNet-family variants converge to almost the same prediction behavior.
6. Several external methods tried so far are weaker on these target datasets.
7. Current active runs are mostly checking whether stricter anchoring/pseudo-labeling can recover small extra gains. The early signal is not strong.

## Immediate Decision Points

- If GPU0/GPU2 remain below 68.28 after a few more evals, stop/preserve/delete those calibrated IRSTD branches.
- If GPU1 continues drifting below 79.13, stop/preserve/delete it.
- If GPU3 remains below about 77-78 by epoch 20 or 25, stop/preserve/delete it despite the upward trend.
- Next promising direction is not another large blind refiner. It should be a smaller, exact-metric selective postprocess or complementarity analysis first, then only train if that analysis shows recoverable pixels.

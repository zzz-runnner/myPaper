# Current Status After Spectral Anomaly Launch

Generated: 2026-06-13 04:20 UTC

## What Was Stopped and Cleaned

The previous calibrated TCD/LCA/BNG branches were stopped because they drifted below the current best checkpoints:

| Branch | Stop point | Best / latest signal | Evidence |
|---|---:|---|---|
| `cal_tcd_lca_bng_irstd_gpu0` | epoch 3 | best 68.24, epoch3 68.20; below IRSTD best 68.28 | `reports/cal_tcd_lca_bng_irstd_gpu0_epoch3_preserved/` |
| `cal_tcd_lca_bng_nuaa_gpu1` | epoch 5 | best 79.12, epoch5 78.27; below NUAA best 79.13 | `reports/cal_tcd_lca_bng_nuaa_gpu1_epoch5_preserved/` |
| `cal_tcd_lca_bng_irstd_gpu2` | epoch 2 | best 68.24, epoch2 67.97; below IRSTD best 68.28 | `reports/cal_tcd_lca_bng_irstd_gpu2_epoch2_preserved/` |
| `pseudo_nuaa_gpu3` | epoch 20 | best 75.57 at epoch15, epoch20 75.49; far below NUAA best 79.13 | `reports/pseudo_nuaa_pconv_epoch20_preserved/` |

The bulky run directories for these stopped branches were deleted after preserving logs/checkpoints.

## Fusion / Complementarity Evidence

Added:

```text
scripts/eval_fusion_postprocess.py
```

This evaluates strong-checkpoint fusion with official mIoU/PD_FA:

- individual checkpoints
- average/max/min probability fusion
- binary vote fusion
- sample oracle
- threshold sweep

Outputs:

```text
reports/fusion_quick_irstd_strong.json
reports/fusion_quick_nuaa_strong.json
```

Key conclusion:

| Dataset | Best real single/fusion region | Sample oracle upper bound | Interpretation |
|---|---:|---:|---|
| IRSTD-1K | about 68.28 | 68.85 | Some sample-level complementarity exists, but far below +5 target |
| NUAA-SIRST | about 79.13 | 79.13 | Strong checkpoints are almost identical; fusion/selector has no headroom |

This rules out spending more GPU time on simple checkpoint fusion or selector training as the main route.

## New Code Added

Added model branch in:

```text
scripts/train_hybrid_methods.py
```

New model:

```text
sct_spectral_anomaly_refiner
```

Concept:

- From the user's pasted external-method line:
  - SET / DHiF style spectral and high-frequency enhancement.
  - UniVAD / AnomalyCLIP style background normality versus local anomaly.
- Implementation:
  - multi-scale local contrast and DoG cues,
  - FFT high-pass map,
  - spectral residual saliency map,
  - background normality memory bank,
  - bounded residual on top of a strong SCTransNet/TPC-DHiF checkpoint,
  - teacher/base logit anchoring to avoid drift.

Smoke tests passed:

| Dataset | Checkpoint seed | Eval-only mIoU |
|---|---|---:|
| IRSTD-1K | TPC-DHiF-B anchored | 68.23 |
| NUAA-SIRST | LCA anchored | 79.12 |

These confirm the new module loads existing checkpoints and starts near the strong baseline instead of collapsing.

## New Active 4-GPU Line

Launcher:

```text
scripts/launch_spectral_anomaly_4gpu_tmux.sh
```

Active sessions:

| GPU | Session | Dataset | Method | Initial signal |
|---:|---|---|---|---:|
| 0 | `spectral_irstd_gpu0` | IRSTD-1K | `SpectralAnomaly-IRSTD-A` | epoch0 68.27 |
| 1 | `spectral_irstd_gpu1` | IRSTD-1K | `SpectralAnomaly-IRSTD-B` | epoch0 68.18 |
| 2 | `spectral_nuaa_gpu2` | NUAA-SIRST | `SpectralAnomaly-NUAA-LCA` | epoch0 79.11 |
| 3 | `spectral_nuaa_gpu3` | NUAA-SIRST | `SpectralAnomaly-NUAA-TCD` | epoch0 79.12 |

All four use full multi-dataset training:

```text
IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3
```

Current status at launch:

- all four tmux sessions active,
- all four GPUs occupied,
- waiting for epoch1/epoch2 validation to see whether the spectral anomaly branch holds or drifts.

## Decision Rules

- Stop any IRSTD spectral branch if it falls below 68.0 for two consecutive early evals.
- Keep an IRSTD branch only if it reaches or beats 68.28, or shows stable near-best behavior with lower false alarm.
- Stop any NUAA spectral branch if it falls below 78.8 for two consecutive early evals.
- Keep a NUAA branch only if it reaches or beats 79.13; fusion evidence shows there is no selector headroom otherwise.
- If all spectral branches plateau, the next major route should be a stronger external-mechanism implementation, not another SCTransNet-family selector.

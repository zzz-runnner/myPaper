# Current Status: APConv-SDM Prepared

Generated: 2026-06-12 UTC

## Active run

The current active tmux session is `paired_bgm_4gpu`.

Run root:

```text
/home/cx/hongwai/irstd-lab/runs/paired_bgm_4gpu
```

It is running four full multi-dataset paired-BGM experiments:

| GPU | Dataset | Method | Model | Batch | Train datasets |
|---:|---|---|---|---:|---|
| 0 | IRSTD-1K | BGM-PairedBasicUNet | `bgm_basic` | 8 | IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3 |
| 1 | IRSTD-1K | TCF-PairedBGM | `tcf_bgm` | 6 | IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3 |
| 2 | NUAA-SIRST | BGM-PairedBasicUNet | `bgm_basic` | 8 | IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3 |
| 3 | NUAA-SIRST | TCF-PairedBGM | `tcf_bgm` | 6 | IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3 |

Current observed output:

- `NUAA-SIRST / BGM-PairedBasicUNet`: epoch 5, loss 0.446813.
- No run has reached the first evaluation point yet. First evaluation is configured at epoch 10.

## New prepared method line

Added official PConv + SDM support to:

```text
/home/cx/hongwai/irstd-lab/scripts/train_hybrid_methods.py
```

The implementation now includes:

- `APConvTCFBasicUNet`, using the official asymmetric-padding PConv operator from:
  `/home/cx/hongwai/irstd-lab/external/PConv-SDloss-Data/model/APConv.py`
- model option `--model apconv_tcf`.
- mask-format `sdm_iou_loss`, adapted from the official PConv + SD loss for segmentation masks.
- CLI options:
  - `--sdm_iou_weight`
  - `--sdm_warm_epochs`
  - `--sdm_delta`
  - `--sdm_with_distance`
  - `--sdm_static`
- upgraded `BackgroundNormalityBank` from one global prototype to a multi-prototype background memory bank, inspired by anomaly-detection normal-memory methods such as UniVAD.
- checkpoint loading now skips shape-mismatched keys, so older selected checkpoints are not broken by the expanded background-memory module.

Validation already done:

- Python compile check passed.
- CPU forward/backward checks passed for `tcf_bgm` and `apconv_tcf`.
- `APConvTCFBasicUNet` has about 3.44M parameters, suitable for RTX 2080 Ti.

## Next launch script

Prepared next-batch launcher:

```text
/home/cx/hongwai/irstd-lab/scripts/launch_apconv_sdm_4gpu_tmux.sh
```

It launches four full multi-dataset runs:

| GPU | Dataset | Method | Model | Main idea |
|---:|---|---|---|---|
| 0 | IRSTD-1K | APConv-TCF-SDM | `apconv_tcf` | official PConv + SDM + multi-prototype background memory |
| 1 | IRSTD-1K | APConv-TCF-PairedSDM | `apconv_tcf` | same, with paired BGM view |
| 2 | NUAA-SIRST | APConv-TCF-SDM | `apconv_tcf` | official PConv + SDM + multi-prototype background memory |
| 3 | NUAA-SIRST | APConv-TCF-PairedSDM | `apconv_tcf` | same, with paired BGM view |

## Decision rule

Wait until the active paired-BGM line reaches at least the first evaluation at epoch 10.

If paired-BGM remains clearly below the prior direct BGM/TCF line or far below SCTransNet, stop it, preserve the report, delete the weak run directory, and launch `apconv_sdm_4gpu`.

If paired-BGM shows a credible trend, keep it until epoch 20/30 before replacing it.

# Current Status: APConv-SDM and Next Candidate

Generated: 2026-06-12 22:58 UTC

## Active run

Current tmux session:

```text
apconv_sdm_4gpu
```

Run root:

```text
/home/cx/hongwai/irstd-lab/runs/apconv_sdm_4gpu
```

Report:

```text
/home/cx/hongwai/irstd-lab/reports/apconv_sdm_4gpu_results.md
```

Current observed status:

| Dataset | Method | Latest epoch | Latest loss | Eval |
|---|---|---:|---:|---|
| IRSTD-1K | APConv-TCF-SDM | 5 | 0.477661 | not reached |
| IRSTD-1K | APConv-TCF-PairedSDM | - | - | not reached |
| NUAA-SIRST | APConv-TCF-SDM | 5 | 0.519005 | not reached |
| NUAA-SIRST | APConv-TCF-PairedSDM | - | - | not reached |

All four GPUs are still occupied, with about 5.6G on the non-paired APConv-SDM jobs and 7.8G on the paired APConv-SDM jobs. No OOM, traceback, or runtime error has appeared.

Decision: keep this line running until at least epoch 10 evaluation. It has not produced enough evidence to stop yet.

## Prepared next candidate

Added model option:

```text
sct_apconv_tpc_refiner
```

Implementation file:

```text
/home/cx/hongwai/irstd-lab/scripts/train_hybrid_methods.py
```

The model freezes the strong SCTransNet/TCD coarse predictor and adds a small residual branch combining:

- official asymmetric-padding APConv from PConv + SD Loss.
- DHiF-style dynamic high-frequency filtering.
- hard-clutter TPC residual routing.
- multi-prototype background normality memory.
- SDM IoU and thermal spot shape losses in the launcher.

Validation done:

- Python compile check passed.
- CPU forward/backward check passed with selected IRSTD checkpoint.
- Trainable parameter count is about `11.6K`, so this is a fast residual-calibration candidate rather than another full SCTransNet retrain.

Prepared launcher:

```text
/home/cx/hongwai/irstd-lab/scripts/launch_apconv_tpc_refiner_4gpu_tmux.sh
```

This launcher is not started yet. It is the ready fallback if the active APConv-SDM line is weak at the first evaluation.

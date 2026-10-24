# Current Status After APConv/TPC Attempts

Generated: 2026-06-12 23:37 UTC

## What was stopped and removed

The following weak run directories were removed after preserving reports:

```text
/home/cx/hongwai/irstd-lab/runs/apconv_sdm_4gpu
/home/cx/hongwai/irstd-lab/runs/apconv_tpc_refiner_4gpu
/home/cx/hongwai/irstd-lab/runs/unfrozen_apconv_tpc_4gpu
```

Preserved reports:

```text
/home/cx/hongwai/irstd-lab/reports/apconv_sdm_4gpu_results.md
/home/cx/hongwai/irstd-lab/reports/apconv_sdm_4gpu_stopped_note.md
/home/cx/hongwai/irstd-lab/reports/apconv_tpc_refiner_4gpu_results.md
/home/cx/hongwai/irstd-lab/reports/apconv_tpc_refiner_4gpu_stopped_note.md
/home/cx/hongwai/irstd-lab/reports/unfrozen_apconv_tpc_4gpu_results.md
/home/cx/hongwai/irstd-lab/reports/unfrozen_apconv_tpc_4gpu_stopped_note.md
```

## Results

| Line | Key result | Decision |
|---|---|---|
| APConv-TCF-SDM from scratch | NUAA epoch 10 mIoU `73.21`, delta `-4.29` vs README `77.50` | stopped |
| Frozen SCT-APConvTPC residual | epoch 0 was near strong baseline, but training dropped IRSTD to `67.67/67.62` and NUAA to `78.89/78.99` | stopped |
| Semi-unfrozen SCT-APConvTPC | epoch 1 dropped IRSTD to `66.83/66.45` and NUAA to `75.35/76.04` | stopped |

## Interpretation

The APConv/TPC/DHiF/background-memory modules are not useful here as naive training attachments:

- training a lightweight APConv-TCF model from scratch starts too far below SCTransNet;
- training a frozen residual branch around the strong SCTransNet checkpoint does not improve metrics;
- unfreezing the strong base destabilizes it immediately, even with teacher/base-anchor constraints.

Next step should avoid blindly training another small SCTransNet-side variant. A better next step is to run checkpoint complementarity/fusion analysis over preserved strong checkpoints and external-method candidates, then only train or distill if the analysis shows recoverable pixels beyond current SCTransNet-family outputs.

## Current machine state

At cleanup time:

- no active tmux training session;
- all four GPUs idle;
- run directory size back to about `653M`;
- reports retained for traceability.

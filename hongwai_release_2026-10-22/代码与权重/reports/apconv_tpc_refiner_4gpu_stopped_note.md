# APConv-TPC Refiner 4GPU Stopped Note

Generated: 2026-06-12 23:25 UTC

Stopped tmux session:

```text
apconv_tpc_refiner_4gpu
```

Removed run directory:

```text
/home/cx/hongwai/irstd-lab/runs/apconv_tpc_refiner_4gpu
```

Preserved report:

```text
/home/cx/hongwai/irstd-lab/reports/apconv_tpc_refiner_4gpu_results.md
```

Reason:

- This line combined frozen strong SCTransNet/TCD predictions with APConv, DHiF, TPC hard-clutter routing, multi-prototype normal-memory, SDM IoU, and thermal spot shape constraints.
- It correctly loaded the strong checkpoints and started from the expected baseline region:
  - IRSTD-1K epoch 0 around `68.02-68.03`
  - NUAA-SIRST epoch 0 around `79.09-79.10`
- Training did not improve the metrics:
  - IRSTD-1K dropped to `67.67` / `67.62` at epoch 2.
  - NUAA-SIRST dropped to `78.89` / `78.99` by epoch 4.
- The residual branch also used only about `2.3-2.8G` per GPU, so it was not a good long-running use of four RTX 2080 Ti cards.

Decision:

Stop and remove this run. Keep only the report and move the GPUs to a heavier/full-capacity experiment.

# Unfrozen APConv-TPC 4GPU Stopped Note

Generated: 2026-06-12 23:36 UTC

Stopped tmux session:

```text
unfrozen_apconv_tpc_4gpu
```

Removed run directory:

```text
/home/cx/hongwai/irstd-lab/runs/unfrozen_apconv_tpc_4gpu
```

Preserved report:

```text
/home/cx/hongwai/irstd-lab/reports/unfrozen_apconv_tpc_4gpu_results.md
```

Reason:

- This line unfroze the strong SCTransNet/TCD base with a very small base learning-rate scale and added APConv/TPC/DHiF/background-memory residual modules.
- It started from the expected strong checkpoints:
  - IRSTD-1K epoch 0 around `68.01-68.02`.
  - NUAA-SIRST epoch 0 around `79.10-79.11`.
- After only one training epoch, all variants degraded:
  - IRSTD-1K dropped to `66.83` / `66.45`.
  - NUAA-SIRST dropped to `75.35` / `76.04`.
- This shows that semi-unfreezing the base, even with teacher/base-anchor constraints, destabilized the strong checkpoint.

Decision:

Stop and remove this run. Continue with safer alternatives that do not move the calibrated base so aggressively.

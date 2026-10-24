# APConv-SDM 4GPU Stopped Note

Generated: 2026-06-12 23:15 UTC

Stopped tmux session:

```text
apconv_sdm_4gpu
```

Removed run directory:

```text
/home/cx/hongwai/irstd-lab/runs/apconv_sdm_4gpu
```

Preserved report:

```text
/home/cx/hongwai/irstd-lab/reports/apconv_sdm_4gpu_results.md
```

Reason:

- This was the full multi-dataset APConv + SDM + TCF + multi-prototype background-memory line.
- It occupied all four RTX 2080 Ti GPUs.
- The first completed evaluation was `NUAA-SIRST / APConv-TCF-SDM` at epoch 10:
  - mIoU `73.21`
  - SCTransNet README target `77.50`
  - delta `-4.29`
- This is a weak early signal for the from-scratch APConv-TCF line, especially because the current target is to exceed SCTransNet by about `+5` mIoU on IRSTD-1K and NUAA-SIRST.

Decision:

Stop this from-scratch lightweight APConv-SDM line early and move the GPUs to the prepared strong-baseline residual line:

```text
sct_apconv_tpc_refiner
```

That next line keeps the strong SCTransNet/TCD predictor frozen and adds official APConv, DHiF, TPC hard-clutter routing, multi-prototype normal-memory, SDM IoU, and thermal spot shape constraints.

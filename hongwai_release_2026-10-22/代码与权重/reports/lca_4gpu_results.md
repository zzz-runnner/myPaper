# Hybrid Method Results

Run root: `/home/cx/hongwai/irstd-lab/runs/lca_4gpu`

| Dataset | Method | Model | Latest epoch | Latest loss | Latest eval | Latest eval mIoU | Best epoch | Best mIoU | README mIoU | Delta | Best Pd | Best Fa |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IRSTD-1K | SCT-LCA-A | sct_lca_refiner | 15 | 0.925662 | 15 | 68.03 | 0 | 68.09 | 68.03 | +0.06 | 93.60 | 11.88 |
| IRSTD-1K | SCT-LCA-B | sct_lca_refiner | 15 | 0.623339 | 15 | 68.07 | 0 | 68.09 | 68.03 | +0.06 | 93.60 | 11.88 |
| NUAA-SIRST | SCT-LCA-A | sct_lca_refiner | 25 | 0.434454 | 25 | 79.09 | 0 | 79.12 | 77.50 | +1.62 | 96.58 | 17.70 |
| NUAA-SIRST | SCT-LCA-B | sct_lca_refiner | 25 | 0.252431 | 25 | 79.09 | 5 | 79.13 | 77.50 | +1.63 | 96.58 | 17.70 |

## Logs

- `/home/cx/hongwai/irstd-lab/runs/lca_4gpu/logs/IRSTD-1K_SCT-LCA-A_gpu0.log`
- `/home/cx/hongwai/irstd-lab/runs/lca_4gpu/logs/IRSTD-1K_SCT-LCA-B_gpu1.log`
- `/home/cx/hongwai/irstd-lab/runs/lca_4gpu/logs/NUAA-SIRST_SCT-LCA-A_gpu2.log`
- `/home/cx/hongwai/irstd-lab/runs/lca_4gpu/logs/NUAA-SIRST_SCT-LCA-B_gpu3.log`

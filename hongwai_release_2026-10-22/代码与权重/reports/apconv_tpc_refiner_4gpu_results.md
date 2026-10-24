# Hybrid Method Results

Run root: `/home/cx/hongwai/irstd-lab/runs/apconv_tpc_refiner_4gpu`

| Dataset | Method | Model | Latest epoch | Latest loss | Latest eval | Latest eval mIoU | Best epoch | Best mIoU | README mIoU | Delta | Best Pd | Best Fa |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IRSTD-1K | SCT-APConvTPC-A | sct_apconv_tpc_refiner | 2 | 10.869207 | 2 | 67.67 | 0 | 68.02 | 68.03 | -0.01 | 93.60 | 11.69 |
| IRSTD-1K | SCT-APConvTPC-B | sct_apconv_tpc_refiner | 2 | 10.870572 | 2 | 67.62 | 0 | 68.03 | 68.03 | +0.00 | 93.60 | 11.65 |
| NUAA-SIRST | SCT-APConvTPC-A | sct_apconv_tpc_refiner | 4 | 5.593175 | 4 | 78.89 | 0 | 79.09 | 77.50 | +1.59 | 96.58 | 17.70 |
| NUAA-SIRST | SCT-APConvTPC-B | sct_apconv_tpc_refiner | 4 | 5.462093 | 4 | 78.99 | 0 | 79.10 | 77.50 | +1.60 | 96.58 | 17.77 |

## Logs

- `/home/cx/hongwai/irstd-lab/runs/apconv_tpc_refiner_4gpu/logs/IRSTD-1K_SCT-APConvTPC-A_gpu0.log`
- `/home/cx/hongwai/irstd-lab/runs/apconv_tpc_refiner_4gpu/logs/IRSTD-1K_SCT-APConvTPC-B_gpu1.log`
- `/home/cx/hongwai/irstd-lab/runs/apconv_tpc_refiner_4gpu/logs/NUAA-SIRST_SCT-APConvTPC-A_gpu2.log`
- `/home/cx/hongwai/irstd-lab/runs/apconv_tpc_refiner_4gpu/logs/NUAA-SIRST_SCT-APConvTPC-B_gpu3.log`

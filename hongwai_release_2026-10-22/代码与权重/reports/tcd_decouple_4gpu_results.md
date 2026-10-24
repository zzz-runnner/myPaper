# Hybrid Method Results

Run root: `/home/cx/hongwai/irstd-lab/runs/tcd_decouple_4gpu`

| Dataset | Method | Model | Latest epoch | Latest loss | Latest eval | Latest eval mIoU | Best epoch | Best mIoU | README mIoU | Delta | Best Pd | Best Fa |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IRSTD-1K | SCT-TCD-Decouple-A | sct_tcd_decouple_refiner | 5 | 0.393844 | 4 | 68.03 | 0 | 68.09 | 68.03 | +0.06 | 93.60 | 11.88 |
| IRSTD-1K | SCT-TCD-Decouple-B | sct_tcd_decouple_refiner | 4 | 0.443328 | 4 | 68.06 | 0 | 68.09 | 68.03 | +0.06 | 93.60 | 11.88 |
| NUAA-SIRST | SCT-TCD-Decouple-A | sct_tcd_decouple_refiner | 9 | 0.167873 | 8 | 79.10 | 2 | 79.13 | 77.50 | +1.63 | 96.58 | 17.70 |
| NUAA-SIRST | SCT-TCD-Decouple-B | sct_tcd_decouple_refiner | 8 | 0.143262 | 8 | 79.11 | 2 | 79.13 | 77.50 | +1.63 | 96.58 | 17.70 |

## Logs

- `/home/cx/hongwai/irstd-lab/runs/tcd_decouple_4gpu/logs/IRSTD-1K_SCT-TCD-Decouple-A_gpu0.log`
- `/home/cx/hongwai/irstd-lab/runs/tcd_decouple_4gpu/logs/IRSTD-1K_SCT-TCD-Decouple-B_gpu1.log`
- `/home/cx/hongwai/irstd-lab/runs/tcd_decouple_4gpu/logs/NUAA-SIRST_SCT-TCD-Decouple-A_gpu2.log`
- `/home/cx/hongwai/irstd-lab/runs/tcd_decouple_4gpu/logs/NUAA-SIRST_SCT-TCD-Decouple-B_gpu3.log`

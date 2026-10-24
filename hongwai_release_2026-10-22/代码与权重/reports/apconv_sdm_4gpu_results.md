# Hybrid Method Results

Run root: `/home/cx/hongwai/irstd-lab/runs/apconv_sdm_4gpu`

| Dataset | Method | Model | Latest epoch | Latest loss | Latest eval | Latest eval mIoU | Best epoch | Best mIoU | README mIoU | Delta | Best Pd | Best Fa |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IRSTD-1K | APConv-TCF-PairedSDM | apconv_tcf |  |  |  |  |  |  | 68.03 |  |  |  |
| IRSTD-1K | APConv-TCF-SDM | apconv_tcf | 5 | 0.477661 |  |  |  |  | 68.03 |  |  |  |
| NUAA-SIRST | APConv-TCF-PairedSDM | apconv_tcf | 5 | 0.470769 |  |  |  |  | 77.50 |  |  |  |
| NUAA-SIRST | APConv-TCF-SDM | apconv_tcf | 10 | 0.446845 | 10 | 73.21 | 10 | 73.21 | 77.50 | -4.29 | 93.54 | 33.61 |

## Logs

- `/home/cx/hongwai/irstd-lab/runs/apconv_sdm_4gpu/logs/IRSTD-1K_APConv-TCF-PairedSDM_gpu1.log`
- `/home/cx/hongwai/irstd-lab/runs/apconv_sdm_4gpu/logs/IRSTD-1K_APConv-TCF-SDM_gpu0.log`
- `/home/cx/hongwai/irstd-lab/runs/apconv_sdm_4gpu/logs/NUAA-SIRST_APConv-TCF-PairedSDM_gpu3.log`
- `/home/cx/hongwai/irstd-lab/runs/apconv_sdm_4gpu/logs/NUAA-SIRST_APConv-TCF-SDM_gpu2.log`

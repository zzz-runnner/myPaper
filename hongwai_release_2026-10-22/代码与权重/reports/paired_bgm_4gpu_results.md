# Hybrid Method Results

Run root: `/home/cx/hongwai/irstd-lab/runs/paired_bgm_4gpu`

| Dataset | Method | Model | Latest epoch | Latest loss | Latest eval | Latest eval mIoU | Best epoch | Best mIoU | README mIoU | Delta | Best Pd | Best Fa |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IRSTD-1K | BGM-PairedBasicUNet | bgm_basic | 5 | 0.450490 |  |  |  |  | 68.03 |  |  |  |
| IRSTD-1K | TCF-PairedBGM | tcf_bgm | 5 | 0.442196 |  |  |  |  | 68.03 |  |  |  |
| NUAA-SIRST | BGM-PairedBasicUNet | bgm_basic | 10 | 0.330069 | 10 | 71.86 | 10 | 71.86 | 77.50 | -5.64 | 94.30 | 27.37 |
| NUAA-SIRST | TCF-PairedBGM | tcf_bgm | 5 | 0.421166 |  |  |  |  | 77.50 |  |  |  |

## Logs

- `/home/cx/hongwai/irstd-lab/runs/paired_bgm_4gpu/logs/IRSTD-1K_BGM-PairedBasicUNet_gpu0.log`
- `/home/cx/hongwai/irstd-lab/runs/paired_bgm_4gpu/logs/IRSTD-1K_TCF-PairedBGM_gpu1.log`
- `/home/cx/hongwai/irstd-lab/runs/paired_bgm_4gpu/logs/NUAA-SIRST_BGM-PairedBasicUNet_gpu2.log`
- `/home/cx/hongwai/irstd-lab/runs/paired_bgm_4gpu/logs/NUAA-SIRST_TCF-PairedBGM_gpu3.log`

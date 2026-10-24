# Hybrid Method Results

Run root: `/home/cx/hongwai/irstd-lab/runs/hybrid_4gpu`

| Dataset | Method | Model | Latest epoch | Latest loss | Best epoch | Best mIoU | README mIoU | Delta | Best Pd | Best Fa |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IRSTD-1K | BGM-BasicUNet | bgm_basic | 70 | 0.185194 | 55 | 60.58 | 68.03 | -7.45 | 90.91 | 56.63 |
| IRSTD-1K | TCF-BGM | tcf_bgm | 80 | 0.207442 | 65 | 61.41 | 68.03 | -6.62 | 93.94 | 32.68 |
| NUAA-SIRST | BGM-BasicUNet | bgm_basic | 230 | 0.069180 | 225 | 76.66 | 77.50 | -0.84 | 97.34 | 15.50 |
| NUAA-SIRST | TCF-BGM | tcf_bgm | 400 | 0.068372 | 335 | 77.52 | 77.50 | +0.02 | 96.20 | 27.78 |

## Logs

- `/home/cx/hongwai/irstd-lab/runs/hybrid_4gpu/logs/IRSTD-1K_BGM-BasicUNet_gpu0.log`
- `/home/cx/hongwai/irstd-lab/runs/hybrid_4gpu/logs/IRSTD-1K_TCF-BGM_gpu1.log`
- `/home/cx/hongwai/irstd-lab/runs/hybrid_4gpu/logs/NUAA-SIRST_BGM-BasicUNet_gpu2.log`
- `/home/cx/hongwai/irstd-lab/runs/hybrid_4gpu/logs/NUAA-SIRST_TCF-BGM_gpu3.log`

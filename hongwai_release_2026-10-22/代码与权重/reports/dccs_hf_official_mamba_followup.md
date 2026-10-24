# DCCS_hf official Mamba follow-up

Date: 2026-06-15

## What changed

- Updated `scripts/eval_dccs_hf_fullsplit.py` so `--mode official` loads `mamba_ssm.ops.selective_scan_interface` through an ops-only shim before importing DCCS.
- This avoids the top-level `mamba_ssm`/`transformers` import problem while still using the compiled official CUDA selective-scan op.
- No DCCS model files were edited.

## Runs

### NUAA-SIRST smoke

Command:

```bash
envs/irstd/bin/python scripts/eval_dccs_hf_fullsplit.py \
  --dataset_name NUAA-SIRST \
  --split test \
  --weight external/DCCS_hf/weight/nuaa_dccs.pkl \
  --mode official \
  --base_size 256 \
  --max_samples 5 \
  --device cuda:0 \
  --thresholds 0.05:0.95:0.15 \
  --min_areas 1,2,3 \
  --output reports/dccs_hf_official_nuaa_smoke.json
```

Result:

- best mIoU: 20.8661
- nIoU: 59.2017
- Pd: 57.1429
- Fa: 22.9785e-6

The smoke run proved the official CUDA path executes.

### NUAA-SIRST full test

Command:

```bash
envs/irstd/bin/python scripts/eval_dccs_hf_fullsplit.py \
  --dataset_name NUAA-SIRST \
  --split test \
  --weight external/DCCS_hf/weight/nuaa_dccs.pkl \
  --mode official \
  --base_size 256 \
  --device cuda:0 \
  --thresholds 0.02:0.98:0.02 \
  --min_areas 1,2,3,4,6,8 \
  --output reports/dccs_hf_official_nuaa_full_eval.json
```

Best result:

- kind: component
- threshold: 0.50
- min_area: 3
- mIoU: 75.4819
- nIoU: 75.8896
- Pd: 96.1977
- Fa: 29.7728e-6

This is much weaker than the current NUAA best of 82.2554 mIoU, so DCCS official is not useful as a direct fusion source yet.

### IRSTD-1K full test

Attempted with `external/DCCS_hf/weight/irstd1k_dccs.pkl`, but stopped after more than 7 minutes because the path used only about 434 MiB GPU memory and was dominated by CPU/synchronization overhead. Since the NUAA result was already far below the active best, no large prediction maps were saved.

## Storage cleanup

- No `.npz` prediction maps were saved for DCCS official.
- Only small JSON evidence files were kept:
  - `reports/dccs_hf_official_nuaa_smoke.json`
  - `reports/dccs_hf_official_nuaa_full_eval.json`

## Conclusion

DCCS_hf official Mamba is now technically runnable, but it is not a good next source for the current target:

- Accuracy is too low on NUAA-SIRST.
- Runtime is slow.
- GPU memory usage is far below the desired 9GB+ per 2080Ti.

Do not spend more time on this branch unless the evaluator is first optimized or new DCCS weights are provided.

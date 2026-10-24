# Official MSHNet / PConv-MSHNet Full-Data Finetune Results

Generated: 2026-06-13 00:10 UTC

Run root before cleanup: `/home/cx/hongwai/irstd-lab/runs/official_mshnet_finetune_4gpu`

This experiment used the official PConv/MSHNet weights as an external-method starting point, not a SCTransNet refiner:

- `mshnet3_sls`: official MSHNet 3-channel D-0.5 weight
- `mshnet_p43_sls`: reconstructed P43/PConv first-block MSHNet weight
- preprocessing: official MSHNet RGB + ImageNet normalization
- train datasets: `IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3`
- batch size: 24 per GPU; observed VRAM about 5.5G for D-0.5 and 7.3G for P43

| Dataset | Method | Init epoch | Init mIoU | Best epoch | Best mIoU | Existing best target | Delta to existing best | Stop reason |
|---|---|---:|---:|---:|---:|---:|---:|---|
| IRSTD-1K | MSHNet-D05-FullData | 0 | 67.29 | 0 | 67.29 | 68.28 | -0.99 | epoch2 dropped to 61.81; loss jumped after SDM warmup |
| IRSTD-1K | MSHNet-P43-FullData | 0 | 66.70 | 0 | 66.70 | 68.28 | -1.58 | epoch2 dropped to 65.92; loss jumped after SDM warmup |
| NUAA-SIRST | MSHNet-D05-FullData | 0 | 70.02 | 2 | 71.40 | 79.13 | -7.73 | far below current best; epoch4 fell to 70.10 |
| NUAA-SIRST | MSHNet-P43-FullData | 0 | 69.46 | 4 | 72.77 | 79.13 | -6.36 | far below current best despite small improvement |

Key observation:

The official MSHNet/PConv-MSHNet line is a valid external baseline, but it is not competitive with the current SCTransNet-family best on the two target datasets. It should not consume more GPU time in this form. The run directory was deleted after this report to avoid keeping failed bulky experiments.

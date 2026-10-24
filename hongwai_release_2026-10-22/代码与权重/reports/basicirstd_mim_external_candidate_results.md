# BasicIRSTD / MiM External Candidate Results

Generated: 2026-06-14 UTC

This round continued the non-SCTransNet direction. The goal was to find a genuinely different candidate source from the pasted method list and test it on the full multi-dataset pool.

## Source Check

Checked external candidates:

- Text-IRSTD / text-guided line: current public repository only exposes README-level material, so it is not directly runnable here.
- MiM-ISTD: official code was cloned to `external/MiM-ISTD`. The repository only contains the network file and requires `timm`, `mamba_ssm`, and `causal_conv1d`; the checkpoint is linked through Baidu. It is useful as a structural reference but was not immediately runnable in the current environment.
- BasicIRSTD: official toolbox was cloned to `external/BasicIRSTD`. It contains runnable implementations of ACM, DNANet, UIUNet, ISTDU-Net, RDIAN, ResUNet, U-Net, etc.

Minor compatibility fix:

```text
external/BasicIRSTD/net.py
```

removed an accidental import from `skimage.feature.tests.test_orb`, which required `pytest` and prevented model construction.

## New Training Script

Implemented:

```text
scripts/train_basicirstd_multidata.py
```

This trains BasicIRSTD models on the full pool:

```text
IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3
```

and evaluates on:

```text
IRSTD-1K, NUAA-SIRST
```

It saves only a best checkpoint during the run. Failed/weak checkpoints were removed after inspection.

## 4-GPU Short Run

Launched four different non-SCTransNet models:

| GPU | Model | Status |
|---:|---|---|
| 0 | ACM | completed 30 epochs |
| 1 | ISTDU-Net | stopped after epoch 20 eval |
| 2 | DNANet | stopped after epoch 10+ because signal was weak |
| 3 | UIUNet | stopped after epoch 10 because output was all empty |

Best observed metrics:

| Model | Epoch | IRSTD-1K mIoU | NUAA-SIRST mIoU | Notes |
|---|---:|---:|---:|---|
| ACM | 30 | 52.75 | 67.47 | far below current best |
| DNANet | 10 | 51.75 | 68.02 | far below current best |
| ISTDU-Net | 20 | 56.03 | 73.65 | best of this batch but still far below |
| UIUNet | 10 | 0.00 | 0.00 | all-empty output in this setting |

Current reliable best remains:

| Dataset | Current reliable best | mIoU |
|---|---|---:|
| IRSTD-1K | dual-PConv / SCTransNet-family fusion | 72.43 |
| NUAA-SIRST | background-normality component fusion | 79.34 |

## Cleanup

Removed weak checkpoints:

```text
runs/basicirstd_multidata_4gpu/*/best_mean_mIoU.pth.tar
```

The run directory now keeps only small logs/summaries and is about 64K.

## Interpretation

BasicIRSTD models are useful as external baselines and the new full-pool training wrapper works, but this batch did not produce a competitive new dense candidate. The best signal, ISTDU-Net, reached only 73.65 mIoU on NUAA and 56.03 on IRSTD after 20 epochs. This is too far below the current fusion region to justify continuing the same setup to long training.

The next promising route should not be another BasicIRSTD baseline run. It should either:

- bring in actual pretrained weights for a stronger recent method, or
- use a stronger text/VLM/foundation prior once runnable code/weights are available, or
- add new datasets under `/data1` to create a different candidate distribution.

# Paired-BGM 4GPU Stopped Note

Generated: 2026-06-12 22:32 UTC

Run root stopped and removed:

```text
/home/cx/hongwai/irstd-lab/runs/paired_bgm_4gpu
```

Reason:

- The run used official-style paired BGM augmentation: original view plus positive/background-masked view.
- An initial larger-batch attempt hit CUDA out-of-memory, so the run was relaunched with reduced batches.
- The reduced-batch run occupied all four RTX 2080 Ti GPUs but produced only one early evaluation after about 40 minutes.
- `NUAA-SIRST / BGM-PairedBasicUNet` reached epoch 10 with mIoU `71.86`, which is `-5.64` below the SCTransNet README target `77.50`.
- The paired line also had poor prior evidence from the direct BGM/TCF family, which did not exceed SCTransNet on the target datasets.

Preserved evidence:

```text
/home/cx/hongwai/irstd-lab/reports/paired_bgm_4gpu_results.md
```

Decision:

Stop this weak line early and move the four GPUs to the prepared APConv + SDM + TCF + multi-prototype background-memory line.

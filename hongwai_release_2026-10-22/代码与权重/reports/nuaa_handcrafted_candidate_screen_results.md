# NUAA Handcrafted / Contrast Candidate Screen

Generated: 2026-06-14 UTC

## Purpose

NUAA-SIRST remains below the requested +5 mIoU target. The current reliable best is about 79.34 mIoU, and the previous strongest candidate-pool oracle with SIRST-UAV PConv/background normality is about 80.49 mIoU, still far below the 82.64 target.

This screen tested whether non-SCTransNet handcrafted infrared small-target cues could provide a genuinely new dense candidate source:

- local contrast sparse candidate;
- morphology/top-hat/local-z handcrafted sparse candidate;
- SIRST-UAV PConv + local contrast;
- SIRST-UAV PConv + background normality + handcrafted sparse candidate.

All runs used the full multi-dataset fitting pool with a capped fit sample count for speed:

```text
IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3, SIRST-UAVB
```

Evaluation was on the full NUAA-SIRST test split.

## Outputs

```text
reports/nuaa_handcrafted_contrast_screen.json
reports/nuaa_handcrafted_sparse_screen.json
reports/nuaa_pconv_contrast_screen.json
reports/nuaa_pconv_bg_handcrafted_full_screen.json
reports/nuaa_handcrafted_oracle_logs/
reports/nuaa_handcrafted_oracle_runners/
```

These artifacts are small, about 160K total. No checkpoint or large probability archive was written.

## Results

| Candidate pool | Best valid mIoU | Best test oracle, not valid | Interpretation |
| --- | ---: | ---: | --- |
| TPC/LCA + contrast sparse | 79.01 | 23.92 | contrast components are mostly false alarms |
| TPC/LCA + handcrafted sparse | 79.01 | 65.96 | handcrafted cue is not complementary enough |
| TPC/LCA + SIRST-UAV PConv + contrast | 79.12 | 23.61 | PConv does not rescue contrast noise |
| TPC/LCA + SIRST-UAV PConv + background + handcrafted | 79.02 | 64.44 | still far below the existing 80.49 oracle |

Reference:

```text
NUAA target: 82.64 mIoU
previous best valid: about 79.34 mIoU
previous best oracle: about 80.49 mIoU
```

## Conclusion

This route does not solve the NUAA bottleneck. The local-contrast and morphology/top-hat candidates add many background false alarms and do not raise the candidate oracle above the existing PConv/background-normality pool.

Decision:

- Do not expand this exact handcrafted/contrast candidate route into long training.
- Keep the small JSON/log evidence only to avoid repeating the same failed search.
- Continue looking for a genuinely stronger NUAA dense candidate source, preferably official pretrained weights or a foundation-feature source that can raise oracle headroom beyond 82.64 before spending full 4-GPU training time.

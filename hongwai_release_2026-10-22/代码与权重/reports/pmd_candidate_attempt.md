# IRSAM-PMD Candidate Attempt

Generated: 2026-06-15 UTC

## Purpose

NUAA-SIRST remains below the requested +5 mIoU target, so this tested a lightweight cross-domain/foundation-segmentation-inspired module from IRSAM:

```text
PMD gradient / curvature edge cues
```

The goal was to see whether IRSAM's PMD-style edge/shape signal can produce a useful dense or sparse candidate map without training the full MobileSAM/IRSAM stack.

## Source Audit

Checked local sources:

```text
external/IRSAM
external/MiM-ISTD
external/IRSTD-Diff
```

Findings:

- `MiM-ISTD` still requires `mamba_ssm` / `causal_conv1d` and Baidu checkpoint assets. This has the same environment blocker as DCCS under the current no-nvcc environment.
- `IRSTD-Diff` has runnable train/sample code but no local checkpoint. From-scratch diffusion training is too expensive to launch blindly before seeing a candidate signal.
- `IRSAM` has no released WPMD encoder weight, but its PMD gradient/curvature modules are lightweight enough to screen directly.

## Code

Added:

```text
scripts/eval_pmd_candidate.py
```

The script evaluates PMD-inspired candidate maps:

```text
pmd_mix
tophat
gradient
curvature
```

It writes only small JSON output and no checkpoint or prediction archive.

## NUAA Tiny Probe

Output:

```text
reports/pmd_candidate_nuaa_tiny20.json
```

Scale:

```text
dataset: NUAA-SIRST test
samples: first 20
modes: pmd_mix, tophat
thresholds: 0.84, 0.90, 0.94
topk: 4, 8
```

Best row:

| mode | kind | threshold | mIoU | nIoU | Pd | Fa |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `pmd_mix` | dense | 0.94 | 16.0561 | 15.1183 | 93.9394 | 1943.4855e-6 |

Selected top rows:

| mode | kind | threshold | topk | mIoU | Pd | Fa |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `pmd_mix` | dense | 0.94 | - | 16.0561 | 93.9394 | 1943.4855e-6 |
| `pmd_mix` | dense | 0.90 | - | 13.7286 | 93.9394 | 2638.0831e-6 |
| `pmd_mix` | topk component | 0.90 | 8 | 10.8668 | 30.3030 | 849.5522e-6 |
| `tophat` | dense | 0.94 | - | 9.4065 | 93.9394 | 6068.6703e-6 |

## Stopped Wider Probe

An initial 40-sample wider grid over `pmd_mix,tophat,gradient,curvature` was stopped because repeated CPU connected-component sweeps were too slow. It produced no JSON and no bulky artifact.

## Decision

Do not add PMD candidates to the current NUAA fusion pool.

Reason:

```text
Best PMD tiny result is only 16.06 mIoU with very high false alarms.
Sparse top-k variants reduce false alarms but lose too much recall.
This is far below the current 81.0269 NUAA best and unlikely to provide useful complementarity.
```

Current reliable bests remain:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.0269 | 82.64 | not passed |

Next useful work should prioritize stronger released weights/data rather than another CPU handcrafted edge candidate.

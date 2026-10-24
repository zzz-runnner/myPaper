# Target Prototype Gated Stopped Note

Generated: 2026-06-13 UTC

## What Was Stopped

The original full target-prototype gated component fusion run was stopped.

Sessions:

```text
proto_irstd_mixed
proto_irstd_tpc
proto_nuaa_tpc_lca
proto_nuaa_lca_tcd
```

Expected outputs were:

```text
reports/target_proto_gated_multidata_irstd_mixed.json
reports/target_proto_gated_multidata_irstd_tpc.json
reports/target_proto_gated_multidata_nuaa_tpc_lca.json
reports/target_proto_gated_multidata_nuaa_lca_tcd.json
```

No JSON result was produced.

## Why It Was Stopped

All four jobs completed train/test sample collection and then spent a long time in CPU-heavy component postprocessing / threshold evaluation. GPU utilization dropped to 0%, while the Python processes continued to consume large CPU. Because no result JSON was written after about 1 hour, these runs are not valid completed experiments.

The logs are preserved in:

```text
reports/target_proto_gated_logs/
```

## Follow-Up Fix

The script was modified to make this line controllable:

```text
scripts/eval_component_fusion_learner.py
```

Added controls:

```text
--max_components_per_image
--fill_modes
--prototype_max_fit_samples
```

A second limited attempt initially still repeated full prototype fitting in all four processes, so it was also stopped before producing formal full results. The active replacement is:

```text
scripts/launch_target_proto_limited_4gpu_tmux.sh
```

This version keeps the full multi-dataset component keeper fit, but limits the target-prototype generator fit to a representative subset per dataset using:

```text
--prototype_max_fit_samples 120
```

Active output targets:

```text
reports/target_proto_limited_multidata_irstd_mixed.json
reports/target_proto_limited_multidata_irstd_tpc.json
reports/target_proto_limited_multidata_nuaa_tpc_lca.json
reports/target_proto_limited_multidata_nuaa_lca_tcd.json
```

Until those JSON files exist and are parsed, the target-prototype line must not be counted as a successful result.

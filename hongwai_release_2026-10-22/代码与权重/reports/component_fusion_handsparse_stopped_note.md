# Handcrafted Candidate Component Fusion Stop Note

Generated: 2026-06-13 UTC

Attempted route: add non-neural handcrafted local-anomaly candidate maps to the component-level teacher fusion learner. This was intended to bring in traditional IR small-target priors: top-hat / local contrast / LoG-style cues, then let a train-split component learner filter false alarms.

Code changes:

```text
scripts/eval_component_fusion_learner.py
```

Added:

- `--handcrafted_candidates`
- `--handcrafted_full`
- sparse handcrafted candidate map `hand_sparse`
- separate `--handcrafted_seed_threshold`

Smoke behavior:

- Full handcrafted maps caused component explosion and were stopped.
- Aggregated `hand_max` still produced too many background-texture components.
- Sparse `hand_sparse` completed a tiny smoke, but it showed the handcrafted candidate mainly increases false alarms if used as max/binary fill; avg fill mostly falls back to the neural teacher average.

Tiny smoke file retained:

```text
reports/component_fusion_handsparse_smoke.json
```

Full four-GPU multi-dataset attempt:

- Started four full-data branches for IRSTD/NUAA with `fit_datasets=IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3`.
- All branches completed prediction collection or reached CPU component postprocessing.
- CPU postprocessing became too slow because handcrafted candidates added too many components even after sparse filtering.
- Runs were stopped before result JSON was produced.

Decision:

This handcrafted-candidate route is not efficient in the current component learner form. It should not consume more GPU/CPU time unless rewritten to keep only top-K handcrafted candidates per image before component fusion, or moved into a lightweight train-time auxiliary channel rather than a dense component-union generator.

No bulky run directory was kept. Only small logs and this note remain.

# Handcrafted Top-K Candidate Component Fusion Stop Note

Generated: 2026-06-13 UTC

Attempted route: constrain traditional local-anomaly candidates to per-image top-K components before adding them to the component-level teacher fusion learner.

Code updated in:

```text
scripts/eval_component_fusion_learner.py
```

Added/updated:

- `--handcrafted_candidates`
- `--handcrafted_topk`
- `hand_sparse` generated from top-K components of top-hat / black-hat / local-z / LoG candidate maps

Smoke result:

```text
reports/component_fusion_handtopk_smoke.json
```

Key smoke observations:

- Top-K control fixed the extreme component explosion seen in the earlier handcrafted attempt.
- However, on the small smoke split the best valid result was still just the neural teacher baseline (`tpc_b`), not a learned handcrafted+teacher improvement.
- Oracle improved slightly on the tiny smoke, meaning handcrafted candidates may contain occasional missed targets, but the train-fitted learner did not convert that into a valid gain.

Full four-GPU multi-dataset attempt:

- Started four branches with `fit_datasets=IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3`.
- Even with top-K=16, full train prediction plus handcrafted candidate extraction was slow; after more than 10 minutes the jobs were still in SIRST3 train collection and had not reached final results.
- Given the smoke result did not beat the teacher baseline and prior handcrafted variants showed false-alarm risk, the full attempt was stopped to avoid wasting GPU/CPU time.

Decision:

Do not continue this handcrafted-candidate-as-component-union route in the current form. If revisited, use handcrafted priors as lightweight auxiliary input/loss during training, not as dense component candidates for post-hoc union.

No bulky run directory was kept. GPU sessions were killed and only small logs/JSON/notes remain.

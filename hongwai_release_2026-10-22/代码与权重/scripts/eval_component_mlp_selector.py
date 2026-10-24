#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from eval_component_fusion_learner import (  # noqa: E402
    build_component_cache,
    build_models,
    collect_fit_samples_multi,
    collect_samples,
    load_official_pconv_models,
    metric_from_maps,
    parse_model_spec,
    parse_official_pconv_spec,
    render_cache,
)


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


class ComponentMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def fit_mlp(
    x: np.ndarray,
    y: np.ndarray,
    hidden: int,
    dropout: float,
    steps: int,
    lr: float,
    pos_weight_scale: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(seed)
    mean, std = standardize_fit(x)
    xs = ((x - mean) / std).astype(np.float32)
    xt = torch.from_numpy(xs).to(device)
    yt = torch.from_numpy(y.astype(np.float32)).to(device)
    model = ComponentMLP(xs.shape[1], hidden, dropout).to(device)
    pos = max(float(y.sum()), 1.0)
    neg = max(float((1 - y).sum()), 1.0)
    pos_weight = torch.tensor([neg / pos * pos_weight_scale], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    return {"model": model, "mean": mean, "std": std, "pos_weight": float(pos_weight.item())}


@torch.no_grad()
def predict_mlp(learner: dict[str, object], x: np.ndarray, device: torch.device) -> np.ndarray:
    if x.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    mean = learner["mean"]
    std = learner["std"]
    xs = ((x - mean) / std).astype(np.float32)
    xt = torch.from_numpy(xs).to(device)
    probs = torch.sigmoid(learner["model"](xt))
    return probs.detach().cpu().numpy().astype(np.float32)


def render_cache_mlp(cache, names: list[str], learner: dict[str, object], keep_threshold: float, fill_mode: str, device: torch.device):
    maps = []
    total_kept = 0
    total_components = 0
    for sample, comps, x, _y in cache:
        scores = predict_mlp(learner, x, device)
        preds = sample["preds"]
        out = np.zeros_like(sample["mask"], dtype=np.float32)
        avg = np.stack([preds[name] for name in names], axis=0).mean(axis=0)
        maxp = np.stack([preds[name] for name in names], axis=0).max(axis=0)
        for comp, score in zip(comps, scores):
            if float(score) < keep_threshold:
                continue
            mask = comp["component"]
            if fill_mode == "avg":
                out[mask] = np.maximum(out[mask], avg[mask])
            elif fill_mode == "max":
                out[mask] = np.maximum(out[mask], maxp[mask])
            elif fill_mode == "binary":
                out[mask] = 1.0
            else:
                raise ValueError(fill_mode)
        maps.append(out)
        total_kept += int((scores >= keep_threshold).sum())
        total_components += int(len(comps))
    return maps, {"kept_components": total_kept, "total_components": total_components}


def baseline_maps(samples: list[dict[str, object]], names: list[str], mode: str) -> list[np.ndarray]:
    out = []
    for sample in samples:
        stack = np.stack([sample["preds"][name] for name in names], axis=0)
        if mode in names:
            out.append(sample["preds"][mode])
        elif mode == "avg":
            out.append(stack.mean(axis=0))
        elif mode == "max":
            out.append(stack.max(axis=0))
        else:
            raise ValueError(mode)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a shallow MLP component selector over multi-dataset IRSTD candidates.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--fit_datasets", default="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3")
    parser.add_argument("--model_spec", action="append", required=True)
    parser.add_argument("--official_pconv_mshnet_spec", action="append", default=[])
    parser.add_argument("--seed_thresholds", default="0.35,0.40,0.45,0.50")
    parser.add_argument("--keep_thresholds", default="0.20,0.30,0.40,0.50,0.60,0.70")
    parser.add_argument("--output_thresholds", default="0.35,0.40,0.45,0.50")
    parser.add_argument("--fill_modes", default="avg,max,binary")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--pos_weight_scales", default="0.35,0.60,1.00,1.60")
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--min_seed_area", type=int, default=1)
    parser.add_argument("--max_components_per_image", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_fit_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    specs = [parse_model_spec(spec) for spec in args.model_spec]
    names = [spec[0] for spec in specs]
    models = build_models(specs, device)
    official_specs = [parse_official_pconv_spec(spec) for spec in args.official_pconv_mshnet_spec]
    officials = load_official_pconv_models(official_specs, device)
    names = names + [str(spec["name"]) for spec in official_specs]
    fit_datasets = [x.strip() for x in args.fit_datasets.split(",") if x.strip()]
    fit_samples = collect_fit_samples_multi(
        models,
        officials,
        Path(args.dataset_dir),
        fit_datasets,
        "train",
        args.max_fit_samples,
        device,
    )
    eval_samples = collect_samples(models, officials, Path(args.dataset_dir), args.dataset_name, "test", args.max_eval_samples, device)

    output_thresholds = parse_float_list(args.output_thresholds)
    keep_thresholds = parse_float_list(args.keep_thresholds)
    seed_thresholds = parse_float_list(args.seed_thresholds)
    fill_modes = [x.strip() for x in args.fill_modes.split(",") if x.strip()]
    pos_weight_scales = parse_float_list(args.pos_weight_scales)
    run_seeds = parse_int_list(args.seeds)

    rows: list[dict[str, object]] = []
    for mode in [*names, "avg", "max"]:
        maps = baseline_maps(eval_samples, names, mode)
        for out_th in output_thresholds:
            rows.append({"kind": "baseline", "mode": mode, "output_threshold": out_th, **metric_from_maps(eval_samples, maps, out_th)})

    for seed_th in seed_thresholds:
        print(f"building component cache seed={seed_th}", flush=True)
        fit_cache, x_train, y_train = build_component_cache(
            fit_samples, names, seed_th, args.min_seed_area, 0.85, args.max_components_per_image
        )
        eval_cache, x_eval, y_eval = build_component_cache(
            eval_samples, names, seed_th, args.min_seed_area, 0.85, args.max_components_per_image
        )
        if x_train.shape[0] == 0:
            continue
        print(
            f"cache seed={seed_th} train_components={x_train.shape[0]} train_positive={int(y_train.sum())} "
            f"eval_components={x_eval.shape[0]} eval_positive={int(y_eval.sum())}",
            flush=True,
        )
        for fill_mode in fill_modes:
            oracle_maps, oracle_count = render_cache(eval_cache, names, None, 0.5, fill_mode, oracle=True)
            oracle_out = 0.5 if fill_mode == "binary" else seed_th
            rows.append(
                {
                    "kind": "test_oracle_not_valid",
                    "seed_threshold": seed_th,
                    "fill_mode": fill_mode,
                    "keep_threshold": 0.5,
                    "output_threshold": oracle_out,
                    "train_components": int(x_train.shape[0]),
                    "train_positive": int(y_train.sum()),
                    "eval_components": int(x_eval.shape[0]),
                    "eval_positive": int(y_eval.sum()),
                    **oracle_count,
                    **metric_from_maps(eval_samples, oracle_maps, oracle_out),
                }
            )
        for seed in run_seeds:
            for pos_scale in pos_weight_scales:
                print(f"fitting mlp seed_th={seed_th} run_seed={seed} pos_scale={pos_scale}", flush=True)
                learner = fit_mlp(
                    x_train,
                    y_train,
                    args.hidden,
                    args.dropout,
                    args.steps,
                    args.lr,
                    pos_scale,
                    args.weight_decay,
                    seed,
                    device,
                )
                train_scores = predict_mlp(learner, x_train, device)
                eval_scores = predict_mlp(learner, x_eval, device)
                print(
                    f"scores seed_th={seed_th} run_seed={seed} pos_scale={pos_scale} "
                    f"train_pos={float(train_scores[y_train > 0].mean()) if (y_train > 0).any() else 0.0:.3f} "
                    f"train_neg={float(train_scores[y_train == 0].mean()) if (y_train == 0).any() else 0.0:.3f} "
                    f"eval_pos={float(eval_scores[y_eval > 0].mean()) if (y_eval > 0).any() else 0.0:.3f} "
                    f"eval_neg={float(eval_scores[y_eval == 0].mean()) if (y_eval == 0).any() else 0.0:.3f}",
                    flush=True,
                )
                for fill_mode in fill_modes:
                    out_ths = [0.5] if fill_mode == "binary" else output_thresholds
                    for keep_th in keep_thresholds:
                        maps, counts = render_cache_mlp(eval_cache, names, learner, keep_th, fill_mode, device)
                        for out_th in out_ths:
                            rows.append(
                                {
                                    "kind": "mlp_train_fit_eval",
                                    "seed_threshold": seed_th,
                                    "fill_mode": fill_mode,
                                    "keep_threshold": keep_th,
                                    "output_threshold": out_th,
                                    "run_seed": seed,
                                    "pos_weight_scale": pos_scale,
                                    "pos_weight": learner["pos_weight"],
                                    "train_components": int(x_train.shape[0]),
                                    "train_positive": int(y_train.sum()),
                                    "eval_components": int(x_eval.shape[0]),
                                    "eval_positive": int(y_eval.sum()),
                                    "train_score_pos_mean": float(train_scores[y_train > 0].mean()) if (y_train > 0).any() else 0.0,
                                    "train_score_neg_mean": float(train_scores[y_train == 0].mean()) if (y_train == 0).any() else 0.0,
                                    "eval_score_pos_mean": float(eval_scores[y_eval > 0].mean()) if (y_eval > 0).any() else 0.0,
                                    "eval_score_neg_mean": float(eval_scores[y_eval == 0].mean()) if (y_eval == 0).any() else 0.0,
                                    **counts,
                                    **metric_from_maps(eval_samples, maps, out_th),
                                }
                            )
    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best_valid = next(row for row in rows if row["kind"] != "test_oracle_not_valid")
    best_oracle = next((row for row in rows if row["kind"] == "test_oracle_not_valid"), None)
    print(
        f"best_valid kind={best_valid['kind']} mode={best_valid.get('mode', '')} "
        f"seed={best_valid.get('seed_threshold', '')} fill={best_valid.get('fill_mode', '')} "
        f"keep={best_valid.get('keep_threshold', '')} out={best_valid.get('output_threshold', '')} "
        f"mIoU={best_valid['mIoU'] * 100:.2f} Pd={best_valid['Pd'] * 100:.2f} Fa={best_valid['Fa'] * 1e6:.2f}",
        flush=True,
    )
    if best_oracle is not None:
        print(
            f"best_oracle_not_valid seed={best_oracle.get('seed_threshold', '')} fill={best_oracle.get('fill_mode', '')} "
            f"mIoU={best_oracle['mIoU'] * 100:.2f} Pd={best_oracle['Pd'] * 100:.2f} Fa={best_oracle['Fa'] * 1e6:.2f}",
            flush=True,
        )
    for row in rows[:20]:
        print(
            f"{row['kind']} mode={row.get('mode', '')} seed={row.get('seed_threshold', '')} "
            f"fill={row.get('fill_mode', '')} keep={row.get('keep_threshold', '')} "
            f"out={row.get('output_threshold', '')} mIoU={row['mIoU'] * 100:.2f} "
            f"Pd={row['Pd'] * 100:.2f} Fa={row['Fa'] * 1e6:.2f}",
            flush=True,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "fit_datasets": fit_datasets,
                "models": names,
                "official_pconv_mshnet_specs": args.official_pconv_mshnet_spec,
                "best_valid": best_valid,
                "best_oracle_not_valid": best_oracle,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

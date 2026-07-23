import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from evaluate_conditional_diffusion import distribution_metrics
from train_conditional_diffusion import VehiclePointCloudDataset
from train_target_latent_retrieval import build_descriptors
from evaluate_rc_tlr_v2 import (
    apply_proto_quota,
    build_sample_rows,
    candidate_cd,
    collapse_metrics,
    hard_batch_ok,
    train_mapper_rc,
)


EPS = 1e-8
FEATURES = [
    "flat_nn_dist",
    "flat_margin",
    "tlr_nn_dist",
    "tlr_margin",
    "tlr_rel_margin",
    "tlr_proto_frac",
    "shape_disagree_cd",
    "same_retrieval",
]


def derived_features(row: Dict) -> List[float]:
    flat_rel_margin = row["flat_margin"] / (row["flat_nn_dist"] + EPS)
    return [
        row["flat_nn_dist"] - row["tlr_nn_dist"],
        row["tlr_nn_dist"] / (row["flat_nn_dist"] + EPS),
        row["tlr_rel_margin"] - flat_rel_margin,
        row["shape_disagree_cd"] / (row["flat_nn_dist"] + EPS),
    ]


def feature_matrix(rows: Sequence[Dict]) -> np.ndarray:
    x = []
    for row in rows:
        vals = [float(row[k]) for k in FEATURES] + derived_features(row)
        x.append(vals)
    return np.asarray(x, dtype=np.float64)


def fit_scaler(rows: Sequence[Dict]):
    x = feature_matrix(rows)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def meta_scores(train_rows: Sequence[Dict], query_rows: Sequence[Dict], k: int, weighted: bool):
    if not train_rows:
        return np.zeros(len(query_rows), dtype=np.float64), np.zeros(len(query_rows), dtype=np.float64)
    mean, std = fit_scaler(train_rows)
    x_train = (feature_matrix(train_rows) - mean) / std
    x_query = (feature_matrix(query_rows) - mean) / std
    deltas = np.asarray([r["delta_flat_minus_tlr"] for r in train_rows], dtype=np.float64)
    pos = (deltas > 0).astype(np.float64)
    kk = min(k, len(train_rows))
    pred_delta, pred_pos = [], []
    for x in x_query:
        dist = np.linalg.norm(x_train - x[None, :], axis=1)
        ids = np.argpartition(dist, kk - 1)[:kk]
        if weighted:
            w = 1.0 / (dist[ids] + 1e-4)
            w = w / w.sum()
            pred_delta.append(float((deltas[ids] * w).sum()))
            pred_pos.append(float((pos[ids] * w).sum()))
        else:
            pred_delta.append(float(deltas[ids].mean()))
            pred_pos.append(float(pos[ids].mean()))
    return np.asarray(pred_delta), np.asarray(pred_pos)


def meta_decisions(config: Dict, train_rows: Sequence[Dict], rows: List[Dict], batch: Dict, args) -> List[bool]:
    if not hard_batch_ok(batch, args):
        return [False] * len(rows)
    guard = config.get("batch_guard")
    if guard is not None:
        if not (
            batch["top_frac"] <= guard["top_frac"]
            and batch["unique_ratio"] >= guard["unique_ratio"]
            and batch["eff_proto_ratio"] >= guard["eff_proto_ratio"]
        ):
            return [False] * len(rows)
    pred_delta, pred_pos = meta_scores(train_rows, rows, config["k"], config["weighted"])
    decisions = [
        bool(d >= config["delta_threshold"] and p >= config["positive_frac"])
        for d, p in zip(pred_delta, pred_pos)
    ]
    quota = config.get("proto_quota_frac")
    if quota is not None:
        decisions = apply_proto_quota(rows, decisions, quota)
    return decisions


def batch_decisions(config: Dict, rows: List[Dict], batch: Dict, args) -> List[bool]:
    if config["kind"] == "flat":
        return [False] * len(rows)
    if not hard_batch_ok(batch, args):
        return [False] * len(rows)
    ok = batch["top_frac"] <= config["top_frac"] and batch["unique_ratio"] >= config["unique_ratio"] and batch["eff_proto_ratio"] >= config["eff_proto_ratio"]
    decisions = [bool(ok)] * len(rows)
    quota = config.get("proto_quota_frac")
    if quota is not None:
        decisions = apply_proto_quota(rows, decisions, quota)
    return decisions


def make_configs():
    configs = [{"kind": "flat", "alpha": 1.0}]
    for top in [0.35, 0.5, 0.65, 0.75, 0.85, 0.9]:
        for uniq in [0.025, 0.05, 0.075, 0.1]:
            for eff in [0.025, 0.05, 0.075]:
                configs.append({"kind": "batch_guard", "top_frac": top, "unique_ratio": uniq, "eff_proto_ratio": eff, "alpha": 1.0})
    guards = [
        None,
        {"top_frac": 0.9, "unique_ratio": 0.025, "eff_proto_ratio": 0.025},
        {"top_frac": 0.75, "unique_ratio": 0.05, "eff_proto_ratio": 0.05},
        {"top_frac": 0.65, "unique_ratio": 0.075, "eff_proto_ratio": 0.05},
    ]
    for k in [15, 25, 40, 60, 100, 160]:
        for threshold in [-0.0005, 0.0, 0.00025, 0.0005, 0.001, 0.0015]:
            for positive_frac in [0.45, 0.5, 0.55, 0.6, 0.65]:
                for weighted in [False, True]:
                    for quota in [None, 0.35, 0.5, 0.65]:
                        for guard in guards:
                            cfg = {
                                "kind": "meta_knn",
                                "k": k,
                                "delta_threshold": threshold,
                                "positive_frac": positive_frac,
                                "weighted": weighted,
                                "alpha": 1.0,
                            }
                            if quota is not None:
                                cfg["proto_quota_frac"] = quota
                            if guard is not None:
                                cfg["batch_guard"] = guard
                            configs.append(cfg)
    seen, out = set(), []
    for cfg in configs:
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out


def group_by(rows: Sequence[Dict], key: str):
    groups = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    return groups


def score_config(config: Dict, rows_by_group: Dict[str, List[Dict]], batch_by_group: Dict[str, Dict], all_rows: List[Dict], args):
    group_summaries, all_cd, all_flat, selected_names = [], [], [], []
    accepted = 0
    for group, rows in rows_by_group.items():
        if config["kind"] == "meta_knn":
            train_rows = [r for r in all_rows if r["group_id"] != group]
            decisions = meta_decisions(config, train_rows, rows, batch_by_group[group], args)
        else:
            decisions = batch_decisions(config, rows, batch_by_group[group], args)
        alpha = float(config.get("alpha", 1.0))
        cds = [candidate_cd(row, use, alpha) for row, use in zip(rows, decisions)]
        flats = [row["flat_cd"] for row in rows]
        accepted += int(sum(decisions))
        all_cd.extend(cds)
        all_flat.extend(flats)
        selected_names.extend([row["tlr_name"] for row, use in zip(rows, decisions) if use])
        group_summaries.append(
            {
                "group": group,
                "mean_cd": float(np.mean(cds)),
                "flat_cd": float(np.mean(flats)),
                "gain_flat_minus_policy": float(np.mean(flats) - np.mean(cds)),
                "tlr_count": int(sum(decisions)),
                "num_cases": len(rows),
            }
        )
    mean_cd = float(np.mean(all_cd))
    flat_mean = float(np.mean(all_flat))
    gains = [g["gain_flat_minus_policy"] for g in group_summaries]
    if 0 < accepted < args.min_inner_accept:
        objective = float("inf")
    else:
        harm_excess = [max(0.0, -g - args.max_inner_prefix_harm) for g in gains]
        mean_gain = flat_mean - mean_cd
        objective = mean_cd
        objective += args.risk_lambda * float(np.mean(harm_excess))
        objective += args.q10_lambda * max(0.0, -float(np.quantile(gains, 0.1)))
        objective += args.negative_mean_lambda * max(0.0, args.min_mean_gain - mean_gain)
    return {
        "config": config,
        "objective": objective,
        "inner_mean_cd": mean_cd,
        "inner_flat_mean_cd": flat_mean,
        "inner_mean_gain": float(flat_mean - mean_cd),
        "inner_min_prefix_gain": float(min(gains)) if gains else 0.0,
        "inner_q10_prefix_gain": float(np.quantile(gains, 0.1)) if gains else 0.0,
        "inner_harm_prefixes": int(sum(g < -args.max_inner_prefix_harm for g in gains)),
        "accepted": int(accepted),
        "tlr_fraction": float(accepted / max(len(all_rows), 1)),
        "selected_collapse": collapse_metrics(selected_names) if selected_names else collapse_metrics([]),
        "group_summaries": group_summaries,
    }


def select_config(scored: List[Dict], args):
    finite = [s for s in scored if np.isfinite(s["objective"])]
    if not finite:
        return next(s for s in scored if s["config"]["kind"] == "flat")
    finite.sort(key=lambda s: (s["objective"], s["inner_mean_cd"]))
    best = finite[0]
    if best["inner_mean_gain"] < args.min_mean_gain:
        return next(s for s in scored if s["config"]["kind"] == "flat")
    return best


def apply_config(config: Dict, inner_rows: List[Dict], rows: List[Dict], batch: Dict, args):
    if config["kind"] == "meta_knn":
        decisions = meta_decisions(config, inner_rows, rows, batch, args)
    else:
        decisions = batch_decisions(config, rows, batch, args)
    alpha = float(config.get("alpha", 1.0))
    out = []
    for row, use in zip(rows, decisions):
        out.append({**row, "source": "tlr" if use else "flat_l2", "alpha": alpha if use else 0.0, "cd": candidate_cd(row, use, alpha)})
    return out, decisions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split_dir", default="splits/prefix_all10")
    p.add_argument("--point_dir", default="models_pointcloud_npy")
    p.add_argument("--label_dir", default="models_labels_npy")
    p.add_argument("--output_dir", default="eval/rc_tlr_meta")
    p.add_argument("--inner_epochs", type=int, default=80)
    p.add_argument("--outer_epochs", type=int, default=200)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--lambda_nce", type=float, default=0.1)
    p.add_argument("--lambda_spread", type=float, default=0.02)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hard_top_frac", type=float, default=0.9)
    p.add_argument("--hard_unique_ratio", type=float, default=0.025)
    p.add_argument("--hard_eff_proto_ratio", type=float, default=0.025)
    p.add_argument("--min_inner_accept", type=int, default=20)
    p.add_argument("--min_mean_gain", type=float, default=0.0)
    p.add_argument("--max_inner_prefix_harm", type=float, default=0.0015)
    p.add_argument("--risk_lambda", type=float, default=2.0)
    p.add_argument("--q10_lambda", type=float, default=0.5)
    p.add_argument("--negative_mean_lambda", type=float, default=5.0)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VehiclePointCloudDataset(args.point_dir, args.label_dir)
    descs = build_descriptors(dataset, device)
    configs = make_configs()
    summaries, all_out_rows = [], []

    for fold, split_path in enumerate(sorted(Path(args.split_dir).glob("prefix_fold*.json"))):
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        train_idx, test_idx = split["train"], split["test"]
        train_prefixes = sorted({dataset.names[idx][:2] for idx in train_idx})
        inner_rows, batch_by_group = [], {}
        for pi, prefix in enumerate(train_prefixes):
            inner_val = [idx for idx in train_idx if dataset.names[idx][:2] == prefix]
            inner_train = [idx for idx in train_idx if dataset.names[idx][:2] != prefix]
            model, x_mean, x_std, y_mean, y_std = train_mapper_rc(
                descs,
                inner_train,
                device,
                hidden_dim=args.hidden_dim,
                epochs=args.inner_epochs,
                lambda_nce=args.lambda_nce,
                temperature=args.temperature,
                lambda_spread=args.lambda_spread,
                seed=args.seed + fold * 101 + pi,
            )
            rows, _, _, batch = build_sample_rows(dataset, descs, inner_train, inner_val, model, x_mean, x_std, y_mean, y_std, device, prefix)
            inner_rows.extend(rows)
            batch_by_group[prefix] = batch

        rows_by_group = group_by(inner_rows, "group_id")
        scored = [score_config(cfg, rows_by_group, batch_by_group, inner_rows, args) for cfg in configs]
        best = select_config(scored, args)

        model, x_mean, x_std, y_mean, y_std = train_mapper_rc(
            descs,
            train_idx,
            device,
            hidden_dim=args.hidden_dim,
            epochs=args.outer_epochs,
            lambda_nce=args.lambda_nce,
            temperature=args.temperature,
            lambda_spread=args.lambda_spread,
            seed=args.seed + fold * 1009,
        )
        outer_rows_raw, flat_pred, tlr_pred, outer_batch = build_sample_rows(
            dataset, descs, train_idx, test_idx, model, x_mean, x_std, y_mean, y_std, device, split.get("test_prefix", f"fold{fold}")
        )
        outer_rows, decisions = apply_config(best["config"], inner_rows, outer_rows_raw, outer_batch, args)
        alpha = float(best["config"].get("alpha", 1.0))
        alpha_mask = torch.tensor([alpha if use else 0.0 for use in decisions], device=device, dtype=flat_pred.dtype).view(-1, 1, 1)
        pred = flat_pred * (1.0 - alpha_mask) + tlr_pred * alpha_mask
        target = descs["targets"][test_idx]
        cond = descs["conds"][test_idx]
        selected_names = [row["tlr_name"] for row, use in zip(outer_rows_raw, decisions) if use]
        top_scored = sorted(
            [
                {
                    "config": s["config"],
                    "objective": s["objective"],
                    "inner_mean_cd": s["inner_mean_cd"],
                    "inner_mean_gain": s["inner_mean_gain"],
                    "inner_min_prefix_gain": s["inner_min_prefix_gain"],
                    "inner_q10_prefix_gain": s["inner_q10_prefix_gain"],
                    "inner_harm_prefixes": s["inner_harm_prefixes"],
                    "accepted": s["accepted"],
                    "tlr_fraction": s["tlr_fraction"],
                    "selected_collapse": s["selected_collapse"],
                }
                for s in scored
            ],
            key=lambda s: (s["objective"], s["inner_mean_cd"]),
        )[:20]
        summary = {
            "fold": fold,
            "test_prefix": split.get("test_prefix"),
            "val_prefix": split.get("val_prefix"),
            "mean_cd": float(np.mean([r["cd"] for r in outer_rows])),
            "flat_l2_cd": float(np.mean([r["flat_cd"] for r in outer_rows])),
            "tlr_all_cd": float(np.mean([r["tlr_cd"] for r in outer_rows])),
            "num_cases": len(outer_rows),
            "num_tlr_samples": int(sum(decisions)),
            "tlr_sample_fraction": float(sum(decisions) / max(len(decisions), 1)),
            "outer_batch": outer_batch,
            "outer_selected_collapse": collapse_metrics(selected_names) if selected_names else collapse_metrics([]),
            "hard_batch_ok": hard_batch_ok(outer_batch, args),
            "best_config": best["config"],
            "best_inner_objective": best["objective"],
            "best_inner_mean_cd": best["inner_mean_cd"],
            "best_inner_flat_mean_cd": best["inner_flat_mean_cd"],
            "best_inner_mean_gain": best["inner_mean_gain"],
            "best_inner_min_prefix_gain": best["inner_min_prefix_gain"],
            "best_inner_q10_prefix_gain": best["inner_q10_prefix_gain"],
            "best_inner_harm_prefixes": best["inner_harm_prefixes"],
            "best_inner_tlr_fraction": best["tlr_fraction"],
            "best_group_summaries": best["group_summaries"],
            "top_config_scores": top_scored,
            "split_path": str(split_path),
        }
        summary.update({f"gen_{key}": value for key, value in distribution_metrics(pred, target).items()})
        summary.update({f"condition_{key}": value for key, value in distribution_metrics(cond, target).items()})
        summaries.append(summary)
        all_out_rows.extend([{**r, "fold": fold} for r in outer_rows])
        print(json.dumps(summary), flush=True)

    aggregate = {
        "folds": len(summaries),
        "mean_cd": float(np.mean([s["mean_cd"] for s in summaries])),
        "std_cd": float(np.std([s["mean_cd"] for s in summaries], ddof=1)),
        "mean_flat_l2_cd": float(np.mean([s["flat_l2_cd"] for s in summaries])),
        "mean_tlr_all_cd": float(np.mean([s["tlr_all_cd"] for s in summaries])),
        "mean_coverage_cd": float(np.mean([s["gen_coverage_cd"] for s in summaries])),
        "mean_mmd_cd": float(np.mean([s["gen_mmd_cd"] for s in summaries])),
        "tlr_sample_fraction": float(np.sum([s["num_tlr_samples"] for s in summaries]) / np.sum([s["num_cases"] for s in summaries])),
        "tlr_enabled_prefixes": [s["test_prefix"] for s in summaries if s["num_tlr_samples"] > 0],
        "hard_batch_failed_prefixes": [s["test_prefix"] for s in summaries if not s["hard_batch_ok"]],
    }
    result = {"aggregate": aggregate, "splits": summaries}
    with open(out / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(out / "per_sample_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_out_rows, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

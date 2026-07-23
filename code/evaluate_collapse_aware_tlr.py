import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from evaluate_conditional_diffusion import distribution_metrics
from evaluate_knn_design_ensemble import FlatKnnIndex, mean_cd as flat_mean_cd
from train_conditional_diffusion import VehiclePointCloudDataset, chamfer_distance
from train_target_latent_retrieval import build_descriptors, retrieve_outputs
from evaluate_tlr_inner_policy import train_mapper, tlr_mean_cd


@torch.no_grad()
def tlr_predict(dataset, descs, train_idx, eval_idx, model, x_mean, x_std, y_mean, y_std):
    pred, nn_idx, pred_desc = retrieve_outputs(model, descs, train_idx, eval_idx, x_mean, x_std, y_mean, y_std)
    names = [dataset.names[int(i.item())] for i in nn_idx]
    counts = Counter(names)
    n = max(len(names), 1)
    unique_ratio = len(counts) / n
    top_frac = max(counts.values()) / n
    return pred, nn_idx, {"unique_ratio": unique_ratio, "top_frac": top_frac}


@torch.no_grad()
def flat_predict(index: FlatKnnIndex, cond):
    return index.predict(cond, k=1, tau=0.0)


@torch.no_grad()
def evaluate_mixed(dataset, descs, train_idx, test_idx, tlr_pack, policy, device):
    flat_index = FlatKnnIndex(dataset, train_idx, device)
    target = descs["targets"][test_idx]
    cond = descs["conds"][test_idx]
    flat = flat_index.predict(cond, k=1, tau=0.0)
    if policy["use_tlr"]:
        model, x_mean, x_std, y_mean, y_std = tlr_pack
        tlr, nn_idx, collapse = tlr_predict(dataset, descs, train_idx, test_idx, model, x_mean, x_std, y_mean, y_std)
        pred = tlr
        retrieved = [dataset.names[int(i.item())] for i in nn_idx]
        source = ["tlr"] * len(test_idx)
    else:
        pred = flat
        collapse = {"unique_ratio": "", "top_frac": ""}
        retrieved = [""] * len(test_idx)
        source = ["flat_l2"] * len(test_idx)
    rows = []
    for i, idx in enumerate(test_idx):
        rows.append(
            {
                "name": dataset.names[idx],
                "prefix": dataset.names[idx][:2],
                "source": source[i],
                "retrieved_name": retrieved[i],
                "cd": float(chamfer_distance(pred[i:i + 1], target[i:i + 1]).item()),
                "flat_cd": float(chamfer_distance(flat[i:i + 1], target[i:i + 1]).item()),
            }
        )
    summary = {
        "mean_cd": float(np.mean([r["cd"] for r in rows])),
        "flat_l2_cd": float(np.mean([r["flat_cd"] for r in rows])),
        "num_cases": len(rows),
        **collapse,
    }
    summary.update({f"gen_{k}": v for k, v in distribution_metrics(pred, target).items()})
    summary.update({f"condition_{k}": v for k, v in distribution_metrics(cond, target).items()})
    return summary, rows


def candidate_decision(candidate: Dict, metrics: Dict):
    name = candidate["name"]
    if name == "always_flat":
        return False
    if name == "always_tlr":
        return True
    if name == "top_frac":
        return metrics["top_frac"] <= candidate["top_frac"]
    if name == "unique_ratio":
        return metrics["unique_ratio"] >= candidate["unique_ratio"]
    if name == "collapse_guard":
        return metrics["top_frac"] <= candidate["top_frac"] and metrics["unique_ratio"] >= candidate["unique_ratio"]
    raise ValueError(name)


def make_candidates():
    candidates = [{"name": "always_flat"}, {"name": "always_tlr"}]
    for top in [0.35, 0.5, 0.65, 0.75, 0.85, 0.9]:
        candidates.append({"name": "top_frac", "top_frac": top})
    for uniq in [0.025, 0.05, 0.075, 0.1, 0.15]:
        candidates.append({"name": "unique_ratio", "unique_ratio": uniq})
    for top in [0.5, 0.65, 0.75, 0.85, 0.9]:
        for uniq in [0.025, 0.05, 0.075, 0.1]:
            candidates.append({"name": "collapse_guard", "top_frac": top, "unique_ratio": uniq})
    return candidates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split_dir", default="splits/prefix_all10")
    p.add_argument("--point_dir", default="models_pointcloud_npy")
    p.add_argument("--label_dir", default="models_labels_npy")
    p.add_argument("--output_dir", default="eval/collapse_aware_tlr")
    p.add_argument("--inner_epochs", type=int, default=80)
    p.add_argument("--outer_epochs", type=int, default=200)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--lambda_nce", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--safety_top_frac", type=float, default=0.9)
    p.add_argument("--safety_unique_ratio", type=float, default=0.03)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VehiclePointCloudDataset(args.point_dir, args.label_dir)
    descs = build_descriptors(dataset, device)
    candidates = make_candidates()
    summaries, all_rows = [], []

    for fold, split_path in enumerate(sorted(Path(args.split_dir).glob("prefix_fold*.json"))):
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        train_idx, test_idx = split["train"], split["test"]
        train_prefixes = sorted({dataset.names[idx][:2] for idx in train_idx})
        inner_tasks = []
        for pi, prefix in enumerate(train_prefixes):
            inner_val = [idx for idx in train_idx if dataset.names[idx][:2] == prefix]
            inner_train = [idx for idx in train_idx if dataset.names[idx][:2] != prefix]
            model, x_mean, x_std, y_mean, y_std = train_mapper(
                descs,
                inner_train,
                device,
                hidden_dim=args.hidden_dim,
                epochs=args.inner_epochs,
                lambda_nce=args.lambda_nce,
                seed=args.seed + fold * 101 + pi,
            )
            tlr_cd = tlr_mean_cd(model, descs, inner_train, inner_val, x_mean, x_std, y_mean, y_std)
            flat_index = FlatKnnIndex(dataset, inner_train, device)
            flat_cd = flat_mean_cd(dataset, flat_index, inner_val, 1, 0.0, device)
            _, _, metrics = tlr_predict(dataset, descs, inner_train, inner_val, model, x_mean, x_std, y_mean, y_std)
            inner_tasks.append({"prefix": prefix, "tlr_cd": tlr_cd, "flat_cd": flat_cd, **metrics})

        scored = []
        for cand in candidates:
            cds = [task["tlr_cd"] if candidate_decision(cand, task) else task["flat_cd"] for task in inner_tasks]
            scored.append({"candidate": cand, "inner_mean_cd": float(np.mean(cds))})
        best_cand = min(scored, key=lambda r: r["inner_mean_cd"])["candidate"]

        model, x_mean, x_std, y_mean, y_std = train_mapper(
            descs,
            train_idx,
            device,
            hidden_dim=args.hidden_dim,
            epochs=args.outer_epochs,
            lambda_nce=args.lambda_nce,
            seed=args.seed + fold * 1009,
        )
        _, _, outer_metrics = tlr_predict(dataset, descs, train_idx, test_idx, model, x_mean, x_std, y_mean, y_std)
        selected = candidate_decision(best_cand, outer_metrics)
        safety_ok = (
            outer_metrics["top_frac"] <= args.safety_top_frac
            and outer_metrics["unique_ratio"] >= args.safety_unique_ratio
        )
        use_tlr = bool(selected and safety_ok)
        summary, rows = evaluate_mixed(
            dataset,
            descs,
            train_idx,
            test_idx,
            (model, x_mean, x_std, y_mean, y_std),
            {"use_tlr": use_tlr},
            device,
        )
        summary.update(
            {
                "fold": fold,
                "test_prefix": split.get("test_prefix"),
                "val_prefix": split.get("val_prefix"),
                "policy_candidate": best_cand,
                "candidate_selected_tlr": selected,
                "safety_ok": safety_ok,
                "used_tlr": use_tlr,
                "outer_unique_ratio": outer_metrics["unique_ratio"],
                "outer_top_frac": outer_metrics["top_frac"],
                "inner_tasks": inner_tasks,
                "candidate_scores": scored,
                "split_path": str(split_path),
            }
        )
        summaries.append(summary)
        all_rows.extend([{**r, "fold": fold} for r in rows])
        print(json.dumps(summary), flush=True)

    aggregate = {
        "folds": len(summaries),
        "mean_cd": float(np.mean([s["mean_cd"] for s in summaries])),
        "std_cd": float(np.std([s["mean_cd"] for s in summaries], ddof=1)),
        "mean_flat_l2_cd": float(np.mean([s["flat_l2_cd"] for s in summaries])),
        "used_tlr_prefixes": [s["test_prefix"] for s in summaries if s["used_tlr"]],
        "mean_coverage_cd": float(np.mean([s["gen_coverage_cd"] for s in summaries])),
        "mean_mmd_cd": float(np.mean([s["gen_mmd_cd"] for s in summaries])),
    }
    result = {"aggregate": aggregate, "splits": summaries}
    with open(out / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(out / "per_sample_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

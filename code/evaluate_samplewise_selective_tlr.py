import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from evaluate_conditional_diffusion import distribution_metrics
from evaluate_knn_design_ensemble import FlatKnnIndex
from evaluate_tlr_inner_policy import train_mapper
from train_conditional_diffusion import VehiclePointCloudDataset, chamfer_distance
from train_target_latent_retrieval import build_descriptors, normalize


EPS = 1e-8


@torch.no_grad()
def flat_candidates(dataset, train_idx: Sequence[int], eval_idx: Sequence[int], descs, device):
    index = FlatKnnIndex(dataset, train_idx, device)
    cond = descs["conds"][eval_idx]
    target = descs["targets"][eval_idx]
    q = cond.reshape(cond.shape[0], -1).to(device)
    dist = torch.cdist(q, index.conds_flat, p=2)
    vals, pos = dist.topk(2, largest=False, dim=1)
    nn_pos = pos[:, 0]
    pred = index.targets[nn_pos]
    train_tensor = torch.tensor(train_idx, device=device)
    nn_idx = train_tensor[nn_pos]
    rows = []
    for i in range(len(eval_idx)):
        rows.append(
            {
                "flat_cd": float(chamfer_distance(pred[i:i + 1], target[i:i + 1]).item()),
                "flat_nn_dist": float(vals[i, 0].item()),
                "flat_margin": float((vals[i, 1] - vals[i, 0]).item()) if vals.shape[1] > 1 else 0.0,
                "flat_idx": int(nn_idx[i].item()),
            }
        )
    return pred, nn_idx, rows


@torch.no_grad()
def tlr_candidates(dataset, train_idx: Sequence[int], eval_idx: Sequence[int], descs, model, x_mean, x_std, y_mean, y_std, device):
    model.eval()
    train_targets_desc = normalize(descs["target_desc"][train_idx], y_mean, y_std)
    q = normalize(descs["cond_desc"][eval_idx], x_mean, x_std)
    pred_desc = model(q)
    dist = torch.cdist(pred_desc, train_targets_desc, p=2)
    vals, pos = dist.topk(2, largest=False, dim=1)
    nn_pos = pos[:, 0]
    train_tensor = torch.tensor(train_idx, device=device)
    nn_idx = train_tensor[nn_pos]
    pred = descs["targets"][nn_idx]
    target = descs["targets"][eval_idx]
    names = [dataset.names[int(i.item())] for i in nn_idx]
    counts = Counter(names)
    n = max(len(names), 1)
    batch_unique_ratio = len(counts) / n
    batch_top_frac = max(counts.values()) / n
    rows = []
    for i, name in enumerate(names):
        d1 = float(vals[i, 0].item())
        d2 = float(vals[i, 1].item()) if vals.shape[1] > 1 else d1
        rows.append(
            {
                "tlr_cd": float(chamfer_distance(pred[i:i + 1], target[i:i + 1]).item()),
                "tlr_nn_dist": d1,
                "tlr_margin": d2 - d1,
                "tlr_rel_margin": (d2 - d1) / (d1 + EPS),
                "tlr_proto_frac": counts[name] / n,
                "tlr_idx": int(nn_idx[i].item()),
                "tlr_name": name,
            }
        )
    return pred, nn_idx, rows, {"unique_ratio": batch_unique_ratio, "top_frac": batch_top_frac}


@torch.no_grad()
def build_sample_rows(dataset, descs, train_idx, eval_idx, model, x_mean, x_std, y_mean, y_std, device):
    flat_pred, flat_idx, flat_rows = flat_candidates(dataset, train_idx, eval_idx, descs, device)
    tlr_pred, tlr_idx, tlr_rows, batch_metrics = tlr_candidates(
        dataset, train_idx, eval_idx, descs, model, x_mean, x_std, y_mean, y_std, device
    )
    rows = []
    for i, idx in enumerate(eval_idx):
        disagree = float(chamfer_distance(flat_pred[i:i + 1], tlr_pred[i:i + 1]).item())
        row = {
            "name": dataset.names[idx],
            "prefix": dataset.names[idx][:2],
            "same_retrieval": int(flat_idx[i].item() == tlr_idx[i].item()),
            "shape_disagree_cd": disagree,
            **flat_rows[i],
            **tlr_rows[i],
        }
        row["delta_flat_minus_tlr"] = row["flat_cd"] - row["tlr_cd"]
        rows.append(row)
    return rows, flat_pred, tlr_pred, batch_metrics


def quantiles(values: Sequence[float], qs: Sequence[float]):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return []
    return sorted({float(np.quantile(arr, q)) for q in qs})


def make_sample_policies(inner_rows: List[Dict]):
    policies = [{"name": "always_flat"}, {"name": "always_tlr"}]
    keys_low = ["tlr_nn_dist", "tlr_proto_frac", "shape_disagree_cd"]
    keys_high = ["tlr_margin", "tlr_rel_margin", "flat_nn_dist", "flat_margin"]
    for key in keys_low:
        for value in quantiles([r[key] for r in inner_rows], [0.25, 0.4, 0.5, 0.6, 0.75, 0.9]):
            policies.append({"name": "threshold", "key": key, "op": "<=", "value": value})
    for key in keys_high:
        for value in quantiles([r[key] for r in inner_rows], [0.1, 0.25, 0.4, 0.5, 0.6, 0.75]):
            policies.append({"name": "threshold", "key": key, "op": ">=", "value": value})
    for fd in quantiles([r["flat_nn_dist"] for r in inner_rows], [0.25, 0.4, 0.5, 0.6, 0.75]):
        for pf in [0.35, 0.5, 0.65, 0.8, 0.9]:
            policies.append({"name": "and", "rules": [{"key": "flat_nn_dist", "op": ">=", "value": fd}, {"key": "tlr_proto_frac", "op": "<=", "value": pf}]})
    for fd in quantiles([r["flat_nn_dist"] for r in inner_rows], [0.25, 0.4, 0.5, 0.6, 0.75]):
        for mg in quantiles([r["tlr_rel_margin"] for r in inner_rows], [0.25, 0.5, 0.75]):
            policies.append({"name": "and", "rules": [{"key": "flat_nn_dist", "op": ">=", "value": fd}, {"key": "tlr_rel_margin", "op": ">=", "value": mg}]})
    return policies


def rule_decision(rule: Dict, row: Dict) -> bool:
    if rule["op"] == "<=":
        return row[rule["key"]] <= rule["value"]
    if rule["op"] == ">=":
        return row[rule["key"]] >= rule["value"]
    raise ValueError(rule["op"])


def policy_decision(policy: Dict, row: Dict) -> bool:
    if policy["name"] == "always_flat":
        return False
    if policy["name"] == "always_tlr":
        return True
    if policy["name"] == "threshold":
        return rule_decision(policy, row)
    if policy["name"] == "and":
        return all(rule_decision(rule, row) for rule in policy["rules"])
    raise ValueError(policy["name"])


def score_policy(policy: Dict, rows: List[Dict], min_accept: int):
    cds = []
    accepted = 0
    for row in rows:
        use_tlr = policy_decision(policy, row)
        accepted += int(use_tlr)
        cds.append(row["tlr_cd"] if use_tlr else row["flat_cd"])
    if 0 < accepted < min_accept:
        return float("inf"), accepted
    return float(np.mean(cds)), accepted


def apply_policy(policy: Dict, rows: List[Dict], min_inner_gain: float, flat_inner_mean: float, policy_inner_mean: float, batch_ok: bool):
    use_policy = batch_ok and (flat_inner_mean - policy_inner_mean) >= min_inner_gain
    out_rows = []
    for row in rows:
        use_tlr = bool(use_policy and policy_decision(policy, row))
        cd = row["tlr_cd"] if use_tlr else row["flat_cd"]
        out_rows.append({**row, "source": "tlr" if use_tlr else "flat_l2", "cd": cd})
    return out_rows, use_policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split_dir", default="splits/prefix_all10")
    p.add_argument("--point_dir", default="models_pointcloud_npy")
    p.add_argument("--label_dir", default="models_labels_npy")
    p.add_argument("--output_dir", default="eval/samplewise_selective_tlr")
    p.add_argument("--inner_epochs", type=int, default=80)
    p.add_argument("--outer_epochs", type=int, default=200)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--lambda_nce", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_inner_gain", type=float, default=0.0)
    p.add_argument("--min_inner_accept", type=int, default=20)
    p.add_argument("--safety_top_frac", type=float, default=0.9)
    p.add_argument("--safety_unique_ratio", type=float, default=0.03)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VehiclePointCloudDataset(args.point_dir, args.label_dir)
    descs = build_descriptors(dataset, device)
    summaries, all_rows = [], []

    for fold, split_path in enumerate(sorted(Path(args.split_dir).glob("prefix_fold*.json"))):
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        train_idx, test_idx = split["train"], split["test"]
        train_prefixes = sorted({dataset.names[idx][:2] for idx in train_idx})
        inner_rows = []
        inner_task_summaries = []
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
            rows, _, _, batch_metrics = build_sample_rows(dataset, descs, inner_train, inner_val, model, x_mean, x_std, y_mean, y_std, device)
            inner_rows.extend([{**r, "inner_prefix": prefix} for r in rows])
            inner_task_summaries.append(
                {
                    "prefix": prefix,
                    "flat_cd": float(np.mean([r["flat_cd"] for r in rows])),
                    "tlr_cd": float(np.mean([r["tlr_cd"] for r in rows])),
                    **batch_metrics,
                }
            )

        policies = make_sample_policies(inner_rows)
        scored = []
        for policy in policies:
            cd, accepted = score_policy(policy, inner_rows, args.min_inner_accept)
            scored.append({"policy": policy, "inner_mean_cd": cd, "accepted": accepted})
        best = min(scored, key=lambda r: r["inner_mean_cd"])
        flat_inner_mean = float(np.mean([r["flat_cd"] for r in inner_rows]))

        model, x_mean, x_std, y_mean, y_std = train_mapper(
            descs,
            train_idx,
            device,
            hidden_dim=args.hidden_dim,
            epochs=args.outer_epochs,
            lambda_nce=args.lambda_nce,
            seed=args.seed + fold * 1009,
        )
        outer_rows_raw, flat_pred, tlr_pred, batch_metrics = build_sample_rows(
            dataset, descs, train_idx, test_idx, model, x_mean, x_std, y_mean, y_std, device
        )
        batch_ok = batch_metrics["top_frac"] <= args.safety_top_frac and batch_metrics["unique_ratio"] >= args.safety_unique_ratio
        outer_rows, policy_enabled = apply_policy(
            best["policy"], outer_rows_raw, args.min_inner_gain, flat_inner_mean, best["inner_mean_cd"], batch_ok
        )
        mask = torch.tensor([r["source"] == "tlr" for r in outer_rows], device=device)
        pred = flat_pred.clone()
        pred[mask] = tlr_pred[mask]
        target = descs["targets"][test_idx]
        cond = descs["conds"][test_idx]
        summary = {
            "fold": fold,
            "test_prefix": split.get("test_prefix"),
            "val_prefix": split.get("val_prefix"),
            "mean_cd": float(np.mean([r["cd"] for r in outer_rows])),
            "flat_l2_cd": float(np.mean([r["flat_cd"] for r in outer_rows])),
            "tlr_all_cd": float(np.mean([r["tlr_cd"] for r in outer_rows])),
            "num_cases": len(outer_rows),
            "num_tlr_samples": int(mask.sum().item()),
            "policy_enabled": policy_enabled,
            "batch_safety_ok": batch_ok,
            "outer_unique_ratio": batch_metrics["unique_ratio"],
            "outer_top_frac": batch_metrics["top_frac"],
            "flat_inner_mean_cd": flat_inner_mean,
            "best_inner_mean_cd": best["inner_mean_cd"],
            "best_inner_accept": best["accepted"],
            "best_policy": best["policy"],
            "inner_tasks": inner_task_summaries,
            "top_policy_scores": sorted(scored, key=lambda r: r["inner_mean_cd"])[:10],
            "split_path": str(split_path),
        }
        summary.update({f"gen_{key}": value for key, value in distribution_metrics(pred, target).items()})
        summary.update({f"condition_{key}": value for key, value in distribution_metrics(cond, target).items()})
        summaries.append(summary)
        all_rows.extend([{**r, "fold": fold} for r in outer_rows])
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
        "policy_enabled_prefixes": [s["test_prefix"] for s in summaries if s["policy_enabled"]],
    }
    result = {"aggregate": aggregate, "splits": summaries}
    with open(out / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(out / "per_sample_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

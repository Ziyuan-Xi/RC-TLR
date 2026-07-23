import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from evaluate_conditional_diffusion import distribution_metrics
from train_conditional_diffusion import VehiclePointCloudDataset, chamfer_distance
from train_retrieval_residual_diffusion import RetrievalPriorIndex, condition_descriptor


@torch.no_grad()
def evaluate_split(dataset, split: Dict, metric: str, device: torch.device):
    train_idx = split["train"]
    test_idx = split["test"]
    index = RetrievalPriorIndex(dataset, train_idx, device, retrieval_metric=metric)
    per_rows: List[Dict] = []
    all_real, all_cond, all_prior = [], [], []
    for idx in test_idx:
        item = dataset[idx]
        cond = item["cond"].float().to(device).unsqueeze(0)
        target = item["target"].float().to(device).unsqueeze(0)
        prior = index.retrieve(cond, [item["name"]], k_exclude_self=False)
        row = {
            "name": item["name"],
            "prefix": item["name"][:2],
            "condition_copy_cd": float(chamfer_distance(cond, target).item()),
            "retrieval_cd": float(chamfer_distance(prior, target).item()),
            "metric": metric,
        }
        per_rows.append(row)
        all_real.append(target.cpu())
        all_cond.append(cond.cpu())
        all_prior.append(prior.cpu())
    real = torch.cat(all_real).to(device)
    cond = torch.cat(all_cond).to(device)
    prior = torch.cat(all_prior).to(device)
    summary = {
        "metric": metric,
        "test_prefix": split.get("test_prefix"),
        "val_prefix": split.get("val_prefix"),
        "train_prefixes": split.get("train_prefixes"),
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "mean_condition_copy_cd": float(np.mean([r["condition_copy_cd"] for r in per_rows])),
        "mean_retrieval_cd": float(np.mean([r["retrieval_cd"] for r in per_rows])),
    }
    summary.update({f"retrieval_{k}": v for k, v in distribution_metrics(prior, real).items()})
    summary.update({f"condition_{k}": v for k, v in distribution_metrics(cond, real).items()})
    return summary, per_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split_dir", default="splits/prefix")
    p.add_argument("--point_dir", default="models_pointcloud_npy")
    p.add_argument("--label_dir", default="models_labels_npy")
    p.add_argument("--output_dir", default="eval/prefix_retrieval_stress")
    p.add_argument("--normalize", choices=["none", "unit_sphere"], default="none")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = VehiclePointCloudDataset(args.point_dir, args.label_dir, normalize=args.normalize)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries, all_rows = [], []
    for split_path in sorted(Path(args.split_dir).glob("prefix_fold*.json")):
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        for metric in ["descriptor", "flat_l2"]:
            summary, rows = evaluate_split(dataset, split, metric, device)
            summary["split_path"] = str(split_path)
            summaries.append(summary)
            all_rows.extend([{**r, "split_path": str(split_path)} for r in rows])
    aggregate = {}
    for metric in ["descriptor", "flat_l2"]:
        vals = [s["mean_retrieval_cd"] for s in summaries if s["metric"] == metric]
        aggregate[f"{metric}_mean_retrieval_cd"] = float(np.mean(vals))
        aggregate[f"{metric}_std_retrieval_cd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        vals = [s["retrieval_coverage_cd"] for s in summaries if s["metric"] == metric]
        aggregate[f"{metric}_mean_coverage_cd"] = float(np.mean(vals))
        vals = [s["retrieval_mmd_cd"] for s in summaries if s["metric"] == metric]
        aggregate[f"{metric}_mean_mmd_cd"] = float(np.mean(vals))
    aggregate["num_splits"] = len({s["split_path"] for s in summaries})
    with open(out / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"aggregate": aggregate, "splits": summaries}, f, indent=2)
    with open(out / "per_sample_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)
    print(json.dumps({"aggregate": aggregate, "splits": summaries}, indent=2))


if __name__ == "__main__":
    main()

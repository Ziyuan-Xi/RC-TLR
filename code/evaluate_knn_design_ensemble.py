import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from evaluate_conditional_diffusion import distribution_metrics
from train_conditional_diffusion import VehiclePointCloudDataset, chamfer_distance


class FlatKnnIndex:
    def __init__(self, dataset: VehiclePointCloudDataset, train_idx: Sequence[int], device):
        self.device = device
        conds, targets = [], []
        for idx in train_idx:
            item = dataset[idx]
            conds.append(item["cond"].float())
            targets.append(item["target"].float())
        self.conds = torch.stack(conds).to(device)
        self.targets = torch.stack(targets).to(device)
        self.conds_flat = self.conds.reshape(self.conds.shape[0], -1)

    def predict(self, cond, k=1, tau=0.0):
        q = cond.reshape(cond.shape[0], -1).to(self.device)
        dist = torch.cdist(q, self.conds_flat, p=2)
        vals, idx = dist.topk(k, largest=False, dim=1)
        neigh = self.targets[idx]
        if k == 1:
            return neigh[:, 0]
        if tau <= 0:
            w = torch.ones_like(vals) / vals.shape[1]
        else:
            w = torch.softmax(-(vals - vals[:, :1]) / tau, dim=1)
        return (neigh * w[:, :, None, None]).sum(dim=1)


@torch.no_grad()
def mean_cd(dataset, index, indices, k, tau, device, batch_size=16):
    total = 0.0
    count = 0
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        items = [dataset[idx] for idx in batch_idx]
        cond = torch.stack([item["cond"].float() for item in items]).to(device)
        target = torch.stack([item["target"].float() for item in items]).to(device)
        pred = index.predict(cond, k=k, tau=tau)
        bs = cond.shape[0]
        total += chamfer_distance(pred, target).item() * bs
        count += bs
    return total / max(count, 1)


@torch.no_grad()
def evaluate(dataset, index, indices, k, tau, device):
    rows = []
    all_real, all_pred, all_cond = [], [], []
    for idx in indices:
        item = dataset[idx]
        cond = item["cond"].float().to(device).unsqueeze(0)
        target = item["target"].float().to(device).unsqueeze(0)
        pred = index.predict(cond, k=k, tau=tau)
        rows.append(
            {
                "name": item["name"],
                "prefix": item["name"][:2],
                "cd": float(chamfer_distance(pred, target).item()),
                "condition_copy_cd": float(chamfer_distance(cond, target).item()),
            }
        )
        all_real.append(target.cpu())
        all_pred.append(pred.cpu())
        all_cond.append(cond.cpu())
    real = torch.cat(all_real).to(device)
    pred = torch.cat(all_pred).to(device)
    cond = torch.cat(all_cond).to(device)
    summary = {
        "k": k,
        "tau": tau,
        "mean_cd": float(np.mean([r["cd"] for r in rows])),
        "mean_condition_copy_cd": float(np.mean([r["condition_copy_cd"] for r in rows])),
        "num_cases": len(rows),
    }
    summary.update({f"gen_{key}": value for key, value in distribution_metrics(pred, real).items()})
    summary.update({f"condition_{key}": value for key, value in distribution_metrics(cond, real).items()})
    return summary, rows


def parse_list(text, cast):
    return [cast(x) for x in text.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split_dir", default="splits/prefix_all10")
    p.add_argument("--point_dir", default="models_pointcloud_npy")
    p.add_argument("--label_dir", default="models_labels_npy")
    p.add_argument("--output_dir", default="eval/knn_design_ensemble")
    p.add_argument("--k_grid", default="1,2,3,5,9")
    p.add_argument("--tau_grid", default="0,0.01,0.05,0.1,0.2,0.5,1.0")
    p.add_argument("--normalize", choices=["none", "unit_sphere"], default="none")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = VehiclePointCloudDataset(args.point_dir, args.label_dir, normalize=args.normalize)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_grid = parse_list(args.k_grid, int)
    tau_grid = parse_list(args.tau_grid, float)
    summaries, all_rows = [], []
    for split_path in sorted(Path(args.split_dir).glob("prefix_fold*.json")):
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        index = FlatKnnIndex(dataset, split["train"], device)
        candidates = []
        for k in k_grid:
            for tau in ([0.0] if k == 1 else tau_grid):
                val_cd = mean_cd(dataset, index, split["val"], k, tau, device)
                candidates.append({"k": k, "tau": tau, "val_cd": val_cd})
        best = min(candidates, key=lambda r: r["val_cd"])
        summary, rows = evaluate(dataset, index, split["test"], best["k"], best["tau"], device)
        summary.update(
            {
                "split_path": str(split_path),
                "test_prefix": split.get("test_prefix"),
                "val_prefix": split.get("val_prefix"),
                "selected_k": best["k"],
                "selected_tau": best["tau"],
                "val_cd": best["val_cd"],
            }
        )
        summaries.append(summary)
        all_rows.extend([{**r, "split_path": str(split_path)} for r in rows])
    aggregate = {
        "folds": len(summaries),
        "mean_cd": float(np.mean([s["mean_cd"] for s in summaries])),
        "std_cd": float(np.std([s["mean_cd"] for s in summaries], ddof=1)) if len(summaries) > 1 else 0.0,
        "mean_coverage_cd": float(np.mean([s["gen_coverage_cd"] for s in summaries])),
        "mean_mmd_cd": float(np.mean([s["gen_mmd_cd"] for s in summaries])),
        "selected": [{"prefix": s["test_prefix"], "k": s["selected_k"], "tau": s["selected_tau"]} for s in summaries],
    }
    result = {"aggregate": aggregate, "splits": summaries}
    with open(out / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(out / "per_sample_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

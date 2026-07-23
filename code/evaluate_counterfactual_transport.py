import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from evaluate_conditional_diffusion import distribution_metrics
from train_conditional_diffusion import VehiclePointCloudDataset, chamfer_distance
from train_retrieval_residual_diffusion import condition_descriptor


class TransportIndex:
    def __init__(self, dataset: VehiclePointCloudDataset, train_idx: Sequence[int], device: torch.device, metric: str):
        self.device = device
        self.metric = metric
        self.names: List[str] = []
        conds, targets, descs = [], [], []
        for idx in train_idx:
            item = dataset[idx]
            self.names.append(item["name"])
            cond = item["cond"].float()
            target = item["target"].float()
            conds.append(cond)
            targets.append(target)
            descs.append(condition_descriptor(cond).float())
        self.conds = torch.stack(conds).to(device)
        self.targets = torch.stack(targets).to(device)
        self.conds_flat = self.conds.reshape(self.conds.shape[0], -1)
        self.descs = torch.stack(descs).to(device)

    def nearest(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.metric == "flat_l2":
            q = cond.reshape(cond.shape[0], -1).to(self.device)
            dist = torch.cdist(q, self.conds_flat, p=2)
        elif self.metric == "descriptor":
            q = condition_descriptor(cond).to(self.device)
            dist = torch.cdist(q, self.descs, p=2)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        idx = dist.argmin(dim=1)
        return self.conds[idx], self.targets[idx]


def transport(prior_target, prior_cond, cond, alpha: float, mode: str):
    if mode == "none":
        return prior_target
    if mode == "point_delta":
        return prior_target + alpha * (cond - prior_cond)
    if mode == "centroid_delta":
        delta = cond.mean(dim=1, keepdim=True) - prior_cond.mean(dim=1, keepdim=True)
        return prior_target + alpha * delta
    if mode == "diag_affine":
        c0 = prior_cond.mean(dim=1, keepdim=True)
        c1 = cond.mean(dim=1, keepdim=True)
        s0 = prior_cond.std(dim=1, keepdim=True).clamp_min(1e-6)
        s1 = cond.std(dim=1, keepdim=True).clamp_min(1e-6)
        scaled = c1 + (prior_target - c0) * (s1 / s0)
        return (1.0 - alpha) * prior_target + alpha * scaled
    raise ValueError(f"Unknown transport mode: {mode}")


@torch.no_grad()
def evaluate_mean_cd(dataset, index: TransportIndex, indices: Sequence[int], alpha: float, mode: str, device: torch.device, batch_size=16):
    total = 0.0
    count = 0
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        items = [dataset[idx] for idx in batch_idx]
        cond = torch.stack([item["cond"].float() for item in items]).to(device)
        target = torch.stack([item["target"].float() for item in items]).to(device)
        prior_cond, prior_target = index.nearest(cond)
        pred = transport(prior_target, prior_cond, cond, alpha, mode)
        bs = cond.shape[0]
        total += chamfer_distance(pred, target).item() * bs
        count += bs
    return total / max(count, 1)


@torch.no_grad()
def evaluate_indices(
    dataset,
    index: TransportIndex,
    indices: Sequence[int],
    alpha: float,
    mode: str,
    device: torch.device,
    with_distribution: bool = True,
):
    per_rows = []
    all_real, all_prior, all_pred = [], [], []
    for idx in indices:
        item = dataset[idx]
        cond = item["cond"].float().to(device).unsqueeze(0)
        target = item["target"].float().to(device).unsqueeze(0)
        prior_cond, prior_target = index.nearest(cond)
        pred = transport(prior_target, prior_cond, cond, alpha, mode)
        per_rows.append(
            {
                "name": item["name"],
                "prefix": item["name"][:2],
                "metric": index.metric,
                "mode": mode,
                "alpha": alpha,
                "retrieval_cd": float(chamfer_distance(prior_target, target).item()),
                "transport_cd": float(chamfer_distance(pred, target).item()),
                "condition_copy_cd": float(chamfer_distance(cond, target).item()),
            }
        )
        all_real.append(target.cpu())
        all_prior.append(prior_target.cpu())
        all_pred.append(pred.cpu())
    summary = {
        "metric": index.metric,
        "mode": mode,
        "alpha": alpha,
        "mean_retrieval_cd": float(np.mean([r["retrieval_cd"] for r in per_rows])),
        "mean_transport_cd": float(np.mean([r["transport_cd"] for r in per_rows])),
        "mean_condition_copy_cd": float(np.mean([r["condition_copy_cd"] for r in per_rows])),
        "num_cases": len(per_rows),
    }
    if with_distribution:
        real = torch.cat(all_real).to(device)
        prior = torch.cat(all_prior).to(device)
        pred = torch.cat(all_pred).to(device)
        summary.update({f"transport_{k}": v for k, v in distribution_metrics(pred, real).items()})
        summary.update({f"retrieval_{k}": v for k, v in distribution_metrics(prior, real).items()})
    return summary, per_rows


def parse_alphas(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split_dir", default="splits/prefix_all10")
    p.add_argument("--point_dir", default="models_pointcloud_npy")
    p.add_argument("--label_dir", default="models_labels_npy")
    p.add_argument("--output_dir", default="eval/counterfactual_transport")
    p.add_argument("--metrics", default="flat_l2,descriptor")
    p.add_argument("--modes", default="none,centroid_delta,diag_affine,point_delta")
    p.add_argument("--alpha_grid", default="-1.0,-0.5,0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0")
    p.add_argument("--skip_distribution", action="store_true")
    p.add_argument("--normalize", choices=["none", "unit_sphere"], default="none")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = VehiclePointCloudDataset(args.point_dir, args.label_dir, normalize=args.normalize)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    alphas = parse_alphas(args.alpha_grid)

    summaries, all_rows = [], []
    for split_path in sorted(Path(args.split_dir).glob("prefix_fold*.json")):
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        train_idx, val_idx, test_idx = split["train"], split["val"], split["test"]
        for metric in metrics:
            index = TransportIndex(dataset, train_idx, device, metric)
            for mode in modes:
                candidates = []
                grid = [0.0] if mode == "none" else alphas
                for alpha in grid:
                    val_cd = evaluate_mean_cd(dataset, index, val_idx, alpha, mode, device)
                    candidates.append({"alpha": alpha, "mean_transport_cd": val_cd})
                best = min(candidates, key=lambda r: r["mean_transport_cd"])
                if args.skip_distribution:
                    test_cd = evaluate_mean_cd(dataset, index, test_idx, best["alpha"], mode, device)
                    retrieval_cd = evaluate_mean_cd(dataset, index, test_idx, 0.0, "none", device)
                    test_summary = {
                        "metric": metric,
                        "mode": mode,
                        "alpha": best["alpha"],
                        "mean_retrieval_cd": retrieval_cd,
                        "mean_transport_cd": test_cd,
                        "mean_condition_copy_cd": "",
                        "num_cases": len(test_idx),
                    }
                    rows = []
                else:
                    test_summary, rows = evaluate_indices(dataset, index, test_idx, best["alpha"], mode, device, with_distribution=True)
                test_summary.update(
                    {
                        "split_path": str(split_path),
                        "test_prefix": split.get("test_prefix"),
                        "val_prefix": split.get("val_prefix"),
                        "train_size": len(train_idx),
                        "val_transport_cd": best["mean_transport_cd"],
                        "selected_alpha": best["alpha"],
                    }
                )
                summaries.append(test_summary)
                all_rows.extend([{**r, "split_path": str(split_path), "selected_alpha": best["alpha"]} for r in rows])

    aggregate = []
    for metric in metrics:
        for mode in modes:
            rows = [s for s in summaries if s["metric"] == metric and s["mode"] == mode]
            if not rows:
                continue
            aggregate.append(
                {
                    "metric": metric,
                    "mode": mode,
                    "folds": len(rows),
                    "mean_transport_cd": float(np.mean([r["mean_transport_cd"] for r in rows])),
                    "std_transport_cd": float(np.std([r["mean_transport_cd"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
                    "mean_retrieval_cd": float(np.mean([r["mean_retrieval_cd"] for r in rows])),
                    "mean_coverage_cd": float(np.mean([r["transport_coverage_cd"] for r in rows])) if "transport_coverage_cd" in rows[0] else "",
                    "mean_mmd_cd": float(np.mean([r["transport_mmd_cd"] for r in rows])) if "transport_mmd_cd" in rows[0] else "",
                    "selected_alphas": [r["selected_alpha"] for r in rows],
                }
            )
    result = {"aggregate": aggregate, "splits": summaries}
    with open(out / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(out / "per_sample_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

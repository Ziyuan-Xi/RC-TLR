import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_conditional_diffusion import distribution_metrics
from evaluate_knn_design_ensemble import FlatKnnIndex, mean_cd as flat_mean_cd
from train_conditional_diffusion import VehiclePointCloudDataset, chamfer_distance, set_seed
from train_target_latent_retrieval import TargetLatentMapper, build_descriptors, normalize, retrieve_outputs


def train_mapper(descs, train_idx: Sequence[int], device, hidden_dim=512, epochs=100, lr=1e-3, lambda_nce=0.1, temperature=0.1, seed=42):
    set_seed(seed)
    x_train = descs["cond_desc"][train_idx]
    y_train = descs["target_desc"][train_idx]
    x_mean, x_std = x_train.mean(dim=0), x_train.std(dim=0)
    y_mean, y_std = y_train.mean(dim=0), y_train.std(dim=0)
    x_train_n = normalize(x_train, x_mean, x_std)
    y_train_n = normalize(y_train, y_mean, y_std)
    model = TargetLatentMapper(x_train_n.shape[1], y_train_n.shape[1], hidden_dim, 0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(train_idx)
    batch_size = min(128, n)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            ids = perm[start:start + batch_size]
            pred = model(x_train_n[ids])
            yb = y_train_n[ids]
            mse = F.mse_loss(pred, yb)
            nce = torch.tensor(0.0, device=device)
            if lambda_nce > 0 and pred.shape[0] > 1:
                logits = F.normalize(pred, dim=-1) @ F.normalize(yb, dim=-1).T / temperature
                labels = torch.arange(pred.shape[0], device=device)
                nce = F.cross_entropy(logits, labels)
            loss = mse + lambda_nce * nce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model, x_mean, x_std, y_mean, y_std


@torch.no_grad()
def tlr_mean_cd(model, descs, train_idx, eval_idx, x_mean, x_std, y_mean, y_std):
    pred, _, _ = retrieve_outputs(model, descs, train_idx, eval_idx, x_mean, x_std, y_mean, y_std)
    target = descs["targets"][eval_idx]
    total = 0.0
    for i in range(len(eval_idx)):
        total += chamfer_distance(pred[i:i + 1], target[i:i + 1]).item()
    return total / max(len(eval_idx), 1)


@torch.no_grad()
def tlr_eval_full(dataset, descs, train_idx, eval_idx, model, x_mean, x_std, y_mean, y_std):
    pred, nn_idx, _ = retrieve_outputs(model, descs, train_idx, eval_idx, x_mean, x_std, y_mean, y_std)
    target = descs["targets"][eval_idx]
    cond = descs["conds"][eval_idx]
    rows = []
    for i, idx in enumerate(eval_idx):
        rows.append(
            {
                "name": dataset.names[idx],
                "prefix": dataset.names[idx][:2],
                "retrieved_name": dataset.names[int(nn_idx[i].item())],
                "cd": float(chamfer_distance(pred[i:i + 1], target[i:i + 1]).item()),
            }
        )
    summary = {
        "mean_cd": float(np.mean([r["cd"] for r in rows])),
        "num_cases": len(rows),
    }
    summary.update({f"gen_{key}": value for key, value in distribution_metrics(pred, target).items()})
    summary.update({f"condition_{key}": value for key, value in distribution_metrics(cond, target).items()})
    return summary, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split_dir", default="splits/prefix_all10")
    p.add_argument("--point_dir", default="models_pointcloud_npy")
    p.add_argument("--label_dir", default="models_labels_npy")
    p.add_argument("--output_dir", default="eval/tlr_inner_policy")
    p.add_argument("--inner_epochs", type=int, default=100)
    p.add_argument("--outer_epochs", type=int, default=300)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--lambda_nce", type=float, default=0.1)
    p.add_argument("--delta_threshold", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VehiclePointCloudDataset(args.point_dir, args.label_dir)
    descs = build_descriptors(dataset, device)
    summaries, rows_all = [], []
    for fold, split_path in enumerate(sorted(Path(args.split_dir).glob("prefix_fold*.json"))):
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        train_idx, test_idx = split["train"], split["test"]
        train_prefixes = sorted({dataset.names[idx][:2] for idx in train_idx})
        inner = []
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
            inner.append({"prefix": prefix, "tlr_cd": tlr_cd, "flat_cd": flat_cd, "delta_flat_minus_tlr": flat_cd - tlr_cd})
        mean_delta = float(np.mean([r["delta_flat_minus_tlr"] for r in inner]))
        policy = "tlr" if mean_delta > args.delta_threshold else "flat_l2"
        flat_index = FlatKnnIndex(dataset, train_idx, device)
        flat_cd = flat_mean_cd(dataset, flat_index, test_idx, 1, 0.0, device)
        if policy == "tlr":
            model, x_mean, x_std, y_mean, y_std = train_mapper(
                descs,
                train_idx,
                device,
                hidden_dim=args.hidden_dim,
                epochs=args.outer_epochs,
                lambda_nce=args.lambda_nce,
                seed=args.seed + fold * 1009,
            )
            summary, rows = tlr_eval_full(dataset, descs, train_idx, test_idx, model, x_mean, x_std, y_mean, y_std)
        else:
            summary = {"mean_cd": flat_cd, "num_cases": len(test_idx)}
            rows = []
        summary.update(
            {
                "fold": fold,
                "test_prefix": split.get("test_prefix"),
                "val_prefix": split.get("val_prefix"),
                "policy": policy,
                "inner_mean_delta_flat_minus_tlr": mean_delta,
                "flat_l2_cd": flat_cd,
                "inner": inner,
                "split_path": str(split_path),
            }
        )
        summaries.append(summary)
        rows_all.extend([{**r, "fold": fold, "policy": policy} for r in rows])
        print(json.dumps(summary), flush=True)
    aggregate = {
        "folds": len(summaries),
        "mean_cd": float(np.mean([s["mean_cd"] for s in summaries])),
        "std_cd": float(np.std([s["mean_cd"] for s in summaries], ddof=1)),
        "policies": [{"prefix": s["test_prefix"], "policy": s["policy"], "delta": s["inner_mean_delta_flat_minus_tlr"]} for s in summaries],
    }
    result = {"aggregate": aggregate, "splits": summaries}
    with open(out / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(out / "per_sample_metrics.json", "w", encoding="utf-8") as f:
        json.dump(rows_all, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

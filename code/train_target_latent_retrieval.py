import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluate_conditional_diffusion import distribution_metrics
from train_conditional_diffusion import VehiclePointCloudDataset, chamfer_distance, make_split, set_seed
from train_retrieval_residual_diffusion import condition_descriptor


class TargetLatentMapper(nn.Module):
    def __init__(self, in_dim=27, out_dim=27, hidden_dim=256, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_split(args, dataset):
    if args.split_json_in:
        with open(args.split_json_in, "r", encoding="utf-8") as f:
            split = json.load(f)
        return split["train"], split["val"], split["test"], split
    train_idx, val_idx, test_idx = make_split(len(dataset), args.seed)
    return train_idx, val_idx, test_idx, {}


@torch.no_grad()
def build_descriptors(dataset: VehiclePointCloudDataset, device):
    cond_desc, target_desc, conds, targets = [], [], [], []
    for idx in range(len(dataset)):
        item = dataset[idx]
        cond = item["cond"].float()
        target = item["target"].float()
        cond_desc.append(condition_descriptor(cond).float())
        target_desc.append(condition_descriptor(target).float())
        conds.append(cond)
        targets.append(target)
    return {
        "cond_desc": torch.stack(cond_desc).to(device),
        "target_desc": torch.stack(target_desc).to(device),
        "conds": torch.stack(conds).to(device),
        "targets": torch.stack(targets).to(device),
    }


def normalize(x, mean, std):
    return (x - mean) / std.clamp_min(1e-6)


@torch.no_grad()
def retrieve_outputs(model, descs, train_idx: Sequence[int], query_idx: Sequence[int], x_mean, x_std, y_mean, y_std):
    model.eval()
    train_targets_desc = normalize(descs["target_desc"][train_idx], y_mean, y_std)
    q = normalize(descs["cond_desc"][query_idx], x_mean, x_std)
    pred_target_desc = model(q)
    dist = torch.cdist(pred_target_desc, train_targets_desc, p=2)
    nn_pos = dist.argmin(dim=1)
    nn_idx = torch.tensor(train_idx, device=dist.device)[nn_pos]
    return descs["targets"][nn_idx], nn_idx, pred_target_desc


@torch.no_grad()
def evaluate(model, dataset, descs, train_idx, eval_idx, x_mean, x_std, y_mean, y_std, with_distribution=False):
    pred, nn_idx, _ = retrieve_outputs(model, descs, train_idx, eval_idx, x_mean, x_std, y_mean, y_std)
    target = descs["targets"][eval_idx]
    cond = descs["conds"][eval_idx]
    total_cd = 0.0
    total_cond_cd = 0.0
    rows = []
    for i, idx in enumerate(eval_idx):
        cd = chamfer_distance(pred[i:i + 1], target[i:i + 1]).item()
        cond_cd = chamfer_distance(cond[i:i + 1], target[i:i + 1]).item()
        total_cd += cd
        total_cond_cd += cond_cd
        rows.append(
            {
                "name": dataset.names[idx],
                "prefix": dataset.names[idx][:2],
                "retrieved_name": dataset.names[int(nn_idx[i].item())],
                "cd": float(cd),
                "condition_copy_cd": float(cond_cd),
            }
        )
    summary = {
        "mean_cd": float(total_cd / max(len(eval_idx), 1)),
        "mean_condition_copy_cd": float(total_cond_cd / max(len(eval_idx), 1)),
        "num_cases": len(eval_idx),
    }
    if with_distribution:
        summary.update({f"gen_{k}": v for k, v in distribution_metrics(pred, target).items()})
        summary.update({f"condition_{k}": v for k, v in distribution_metrics(cond, target).items()})
    return summary, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--point_dir", default="models_pointcloud_npy")
    p.add_argument("--label_dir", default="models_labels_npy")
    p.add_argument("--output_dir", default="runs/target_latent_retrieval")
    p.add_argument("--split_json_in", default=None)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lambda_nce", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--normalize", choices=["none", "unit_sphere"], default="none")
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    dataset = VehiclePointCloudDataset(args.point_dir, args.label_dir, normalize=args.normalize)
    train_idx, val_idx, test_idx, split_meta = load_split(args, dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    descs = build_descriptors(dataset, device)
    x_train = descs["cond_desc"][train_idx]
    y_train = descs["target_desc"][train_idx]
    x_mean, x_std = x_train.mean(dim=0), x_train.std(dim=0)
    y_mean, y_std = y_train.mean(dim=0), y_train.std(dim=0)
    x_train_n = normalize(x_train, x_mean, x_std)
    y_train_n = normalize(y_train, y_mean, y_std)

    model = TargetLatentMapper(x_train_n.shape[1], y_train_n.shape[1], args.hidden_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = float("inf")

    with open(out / "split.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train": train_idx,
                "val": val_idx,
                "test": test_idx,
                "names": dataset.names,
                "split_meta": {k: v for k, v in split_meta.items() if k not in {"train", "val", "test"}},
            },
            f,
            indent=2,
        )

    n = len(train_idx)
    print(f"Dataset pairs={len(dataset)}, train/val/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)}, device={device}, method=target_latent_retrieval", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        totals = {"loss": 0.0, "mse": 0.0, "nce": 0.0}
        count = 0
        for start in range(0, n, args.batch_size):
            ids = perm[start:start + args.batch_size]
            xb = x_train_n[ids]
            yb = y_train_n[ids]
            pred = model(xb)
            mse = F.mse_loss(pred, yb)
            nce = torch.tensor(0.0, device=device)
            if args.lambda_nce > 0 and pred.shape[0] > 1:
                pred_n = F.normalize(pred, dim=-1)
                yb_n = F.normalize(yb, dim=-1)
                logits = pred_n @ yb_n.T / args.temperature
                labels = torch.arange(pred.shape[0], device=device)
                nce = F.cross_entropy(logits, labels)
            loss = mse + args.lambda_nce * nce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = xb.shape[0]
            totals["loss"] += loss.item() * bs
            totals["mse"] += mse.item() * bs
            totals["nce"] += nce.item() * bs
            count += bs
        row = {"epoch": epoch, **{f"train_{k}": v / max(count, 1) for k, v in totals.items()}}
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            val_summary, _ = evaluate(model, dataset, descs, train_idx, val_idx, x_mean, x_std, y_mean, y_std)
            row.update({f"val_{k}": v for k, v in val_summary.items()})
            print(json.dumps(row), flush=True)
            if val_summary["mean_cd"] < best_val:
                best_val = val_summary["mean_cd"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": vars(args),
                        "epoch": epoch,
                        "x_mean": x_mean,
                        "x_std": x_std,
                        "y_mean": y_mean,
                        "y_std": y_std,
                    },
                    out / "best_model.pth",
                )
        with open(out / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    ckpt = torch.load(out / "best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model"])
    summary, rows = evaluate(model, dataset, descs, train_idx, test_idx, x_mean, x_std, y_mean, y_std, with_distribution=True)
    summary.update({"model": "target_latent_retrieval", "best_epoch": int(ckpt["epoch"])})
    with open(out / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out / "per_sample_metrics.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print("TEST", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_conditional_diffusion import VehiclePointCloudDataset, chamfer_distance, make_split, set_seed
from train_retrieval_residual_diffusion import RetrievalBatchDataset, RetrievalPriorIndex, make_collate


class RetrievalResidualRefiner(nn.Module):
    def __init__(self, global_dim=384, hidden_dim=384, dropout=0.05):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(9, 96), nn.SiLU(),
            nn.Linear(96, 192), nn.SiLU(),
            nn.Linear(192, global_dim), nn.SiLU(),
        )
        self.body = nn.Sequential(
            nn.Linear(9 + global_dim, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, cond, prior):
        b, n, _ = cond.shape
        local = torch.cat([cond, prior, prior - cond], dim=-1)
        global_feat = self.encoder(local).max(dim=1).values[:, None, :].expand(b, n, -1)
        residual = self.body(torch.cat([local, global_feat], dim=-1))
        return prior + residual


def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    total = {"cd": 0.0, "retrieval_cd": 0.0, "mse": 0.0, "cond_cd": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            target = batch["target"].to(device)
            cond = batch["cond"].to(device)
            prior = batch["prior"].to(device)
            gen = model(cond, prior)
            bs = target.shape[0]
            total["cd"] += chamfer_distance(gen, target).item() * bs
            total["retrieval_cd"] += chamfer_distance(prior, target).item() * bs
            total["mse"] += F.mse_loss(gen, target).item() * bs
            total["cond_cd"] += chamfer_distance(gen, cond).item() * bs
            count += bs
    return {k: v / max(count, 1) for k, v in total.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--point_dir", default="models_pointcloud_npy")
    p.add_argument("--label_dir", default="models_labels_npy")
    p.add_argument("--output_dir", default="runs/rr_refiner")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden_dim", type=int, default=384)
    p.add_argument("--global_dim", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lambda_cd", type=float, default=1.0)
    p.add_argument("--lambda_mse", type=float, default=10.0)
    p.add_argument("--lambda_prior", type=float, default=0.05)
    p.add_argument("--normalize", choices=["none", "unit_sphere"], default="none")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=100)
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    dataset = VehiclePointCloudDataset(args.point_dir, args.label_dir, normalize=args.normalize)
    train_idx, val_idx, test_idx = make_split(len(dataset), args.seed)
    with open(out / "split.json", "w", encoding="utf-8") as f:
        json.dump({"train": train_idx, "val": val_idx, "test": test_idx, "names": dataset.names}, f, indent=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior_index = RetrievalPriorIndex(dataset, train_idx, device)
    train_loader = DataLoader(
        RetrievalBatchDataset(dataset, train_idx, prior_index),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=make_collate(prior_index, exclude_self=True),
    )
    val_loader = DataLoader(
        RetrievalBatchDataset(dataset, val_idx, prior_index),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate(prior_index, exclude_self=False),
    )
    test_loader = DataLoader(
        RetrievalBatchDataset(dataset, test_idx, prior_index),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate(prior_index, exclude_self=False),
    )

    model = RetrievalResidualRefiner(args.global_dim, args.hidden_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = float("inf")
    print(f"Dataset pairs={len(dataset)}, train/val/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)}, device={device}, method=retrieval_refiner")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = {"loss": 0.0, "cd": 0.0, "mse": 0.0, "prior_cd": 0.0}
        count = 0
        for batch in train_loader:
            target = batch["target"].to(device)
            cond = batch["cond"].to(device)
            prior = batch["prior"].to(device)
            gen = model(cond, prior)
            cd = chamfer_distance(gen, target)
            mse = F.mse_loss(gen, target)
            prior_cd = chamfer_distance(gen, prior)
            loss = args.lambda_cd * cd + args.lambda_mse * mse + args.lambda_prior * prior_cd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = target.shape[0]
            total["loss"] += loss.item() * bs
            total["cd"] += cd.item() * bs
            total["mse"] += mse.item() * bs
            total["prior_cd"] += prior_cd.item() * bs
            count += bs
        row = {"epoch": epoch, **{f"train_{k}": v / max(count, 1) for k, v in total.items()}}
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            val = evaluate(model, val_loader, device)
            row.update({f"val_{k}": v for k, v in val.items()})
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if val["cd"] < best_val:
                best_val = val["cd"]
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, out / "best_model.pth")
        elif epoch % 5 == 0:
            print(json.dumps(row, ensure_ascii=False), flush=True)
        if epoch % args.save_every == 0:
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, out / f"checkpoint_{epoch}.pth")
        with open(out / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    torch.save({"model": model.state_dict(), "args": vars(args), "epoch": args.epochs}, out / "last_model.pth")
    test = evaluate(model, test_loader, device)
    with open(out / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2, ensure_ascii=False)
    print("TEST", json.dumps(test, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

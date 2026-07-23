import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader, Subset

from evaluate_conditional_diffusion import approx_emd_sorted, distribution_metrics
from train_conditional_diffusion import VehiclePointCloudDataset, chamfer_distance, make_split, set_seed


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.weight * grad_output, None


def gradient_reverse(x, weight):
    return GradientReverse.apply(x, weight)


def prefix_labels(names: List[str], prefix_to_id: Dict[str, int]) -> torch.Tensor:
    return torch.tensor([prefix_to_id[str(name)[:2]] for name in names], dtype=torch.long)


class FamilyInvariantResidualField(nn.Module):
    """Pointwise residual field with a prefix-adversarial global code.

    The model is deliberately smaller and more geometric than the original
    flattened generator: it predicts a deformation field y = c + r(c), where
    the global shape code is discouraged from encoding train-prefix identity.
    """

    def __init__(
        self,
        hidden_dim=256,
        global_dim=256,
        num_prefixes=10,
        fourier_bands=4,
        dropout=0.0,
    ):
        super().__init__()
        self.fourier_bands = fourier_bands
        if fourier_bands > 0:
            bands = torch.pow(2.0, torch.arange(fourier_bands).float())
            self.register_buffer("bands", bands)
        else:
            self.register_buffer("bands", torch.empty(0))
        local_dim = 3 + 1
        if fourier_bands > 0:
            local_dim += 2 * 3 * fourier_bands
        self.local_encoder = nn.Sequential(
            nn.Linear(local_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, global_dim), nn.SiLU(),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim + 9, global_dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(global_dim, global_dim), nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(local_dim + global_dim, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.prefix_head = nn.Sequential(
            nn.Linear(global_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, num_prefixes),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def local_features(self, cond):
        centered = cond - cond.mean(dim=1, keepdim=True)
        radius = torch.linalg.norm(centered, dim=-1, keepdim=True)
        feats = [centered, radius]
        if self.fourier_bands > 0:
            xb = centered[..., None] * self.bands.view(1, 1, 1, -1) * torch.pi
            feats.extend([torch.sin(xb).flatten(-2), torch.cos(xb).flatten(-2)])
        return torch.cat(feats, dim=-1)

    def forward(self, cond, grl_weight=0.0):
        local = self.local_features(cond)
        point_feat = self.local_encoder(local)
        pooled = point_feat.max(dim=1).values
        stats = torch.cat(
            [
                cond.mean(dim=1),
                cond.std(dim=1),
                cond.amin(dim=1),
            ],
            dim=-1,
        )
        global_code = self.global_proj(torch.cat([pooled, stats], dim=-1))
        global_expand = global_code[:, None, :].expand(cond.shape[0], cond.shape[1], -1)
        residual = self.decoder(torch.cat([local, global_expand], dim=-1))
        out = cond + residual
        logits = self.prefix_head(gradient_reverse(global_code, grl_weight)) if grl_weight > 0 else self.prefix_head(global_code.detach())
        return out, residual, logits, global_code


def load_split(args, dataset):
    if args.split_json_in:
        with open(args.split_json_in, "r", encoding="utf-8") as f:
            split = json.load(f)
        train_idx = split["train"]
        val_idx = split["val"]
        test_idx = split["test"]
    else:
        train_idx, val_idx, test_idx = make_split(len(dataset), args.seed)
        split = {}
    return train_idx, val_idx, test_idx, split


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = {"cd": 0.0, "mse": 0.0, "cond_cd": 0.0, "residual_l2": 0.0}
    count = 0
    for batch in loader:
        target = batch["target"].to(device)
        cond = batch["cond"].to(device)
        pred, residual, _, _ = model(cond, grl_weight=0.0)
        bs = target.shape[0]
        total["cd"] += chamfer_distance(pred, target).item() * bs
        total["mse"] += F.mse_loss(pred, target).item() * bs
        total["cond_cd"] += chamfer_distance(cond, target).item() * bs
        total["residual_l2"] += residual.pow(2).mean().sqrt().item() * bs
        count += bs
    return {k: v / max(count, 1) for k, v in total.items()}


@torch.no_grad()
def save_outputs(model, loader, device, out: Path):
    model.eval()
    sample_dir = out / "samples"
    sample_dir.mkdir(exist_ok=True)
    all_real, all_cond, all_pred = [], [], []
    per_rows = []
    for batch in loader:
        target = batch["target"].to(device)
        cond = batch["cond"].to(device)
        names = batch["name"]
        pred, residual, _, _ = model(cond, grl_weight=0.0)
        emd = approx_emd_sorted(pred, target)
        for i, name in enumerate(names):
            cd = chamfer_distance(pred[i:i + 1], target[i:i + 1]).item()
            cond_cd = chamfer_distance(cond[i:i + 1], target[i:i + 1]).item()
            np.save(sample_dir / f"{name}-firf.npy", pred[i].detach().cpu().numpy())
            per_rows.append(
                {
                    "name": name,
                    "prefix": str(name)[:2],
                    "cd": float(cd),
                    "cond_cd": float(cond_cd),
                    "residual_l2": float(residual[i].pow(2).mean().sqrt().item()),
                    "approx_emd": float(emd.item()),
                }
            )
        all_real.append(target.cpu())
        all_cond.append(cond.cpu())
        all_pred.append(pred.cpu())
    real = torch.cat(all_real).to(device)
    cond = torch.cat(all_cond).to(device)
    pred = torch.cat(all_pred).to(device)
    summary = {
        "mean_cd": float(np.mean([r["cd"] for r in per_rows])),
        "mean_cond_cd": float(np.mean([r["cond_cd"] for r in per_rows])),
        "mean_residual_l2": float(np.mean([r["residual_l2"] for r in per_rows])),
        "mean_approx_emd": float(np.mean([r["approx_emd"] for r in per_rows])),
        "num_test": len(per_rows),
        "model": "family_invariant_residual_field",
    }
    summary.update({f"gen_{k}": v for k, v in distribution_metrics(pred, real).items()})
    summary.update({f"condition_{k}": v for k, v in distribution_metrics(cond, real).items()})
    with open(out / "per_sample_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_rows, f, indent=2, ensure_ascii=False)
    with open(out / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--point_dir", default="models_pointcloud_npy")
    p.add_argument("--label_dir", default="models_labels_npy")
    p.add_argument("--output_dir", default="runs/firf")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--global_dim", type=int, default=256)
    p.add_argument("--fourier_bands", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lambda_cd", type=float, default=0.0)
    p.add_argument("--lambda_mse", type=float, default=1.0)
    p.add_argument("--lambda_residual", type=float, default=1e-4)
    p.add_argument("--lambda_adv", type=float, default=0.05)
    p.add_argument("--grl_warmup_epochs", type=int, default=50)
    p.add_argument("--split_json_in", default=None)
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
    train_idx, val_idx, test_idx, split_meta = load_split(args, dataset)
    prefixes = sorted({name[:2] for name in dataset.names})
    prefix_to_id = {prefix: i for i, prefix in enumerate(prefixes)}
    with open(out / "split.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train": train_idx,
                "val": val_idx,
                "test": test_idx,
                "names": dataset.names,
                "prefix_to_id": prefix_to_id,
                "split_meta": {k: v for k, v in split_meta.items() if k not in {"train", "val", "test"}},
            },
            f,
            indent=2,
        )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FamilyInvariantResidualField(
        hidden_dim=args.hidden_dim,
        global_dim=args.global_dim,
        num_prefixes=len(prefix_to_id),
        fourier_bands=args.fourier_bands,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = float("inf")

    print(
        f"Dataset pairs={len(dataset)}, train/val/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)}, "
        f"device={device}, method=family_invariant_residual_field, prefixes={prefixes}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = {"loss": 0.0, "cd": 0.0, "mse": 0.0, "residual": 0.0, "prefix_ce": 0.0, "prefix_acc": 0.0}
        count = 0
        grl_weight = args.lambda_adv * min(1.0, epoch / max(args.grl_warmup_epochs, 1))
        for batch in train_loader:
            target = batch["target"].to(device)
            cond = batch["cond"].to(device)
            y_prefix = prefix_labels(batch["name"], prefix_to_id).to(device)
            pred, residual, logits, _ = model(cond, grl_weight=grl_weight)
            cd = chamfer_distance(pred, target)
            mse = F.mse_loss(pred, target)
            residual_reg = residual.pow(2).mean()
            prefix_ce = F.cross_entropy(logits, y_prefix)
            loss = (
                args.lambda_cd * cd
                + args.lambda_mse * mse
                + args.lambda_residual * residual_reg
                + (args.lambda_adv * prefix_ce if args.lambda_adv > 0 else 0.0)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = target.shape[0]
            total["loss"] += loss.item() * bs
            total["cd"] += cd.item() * bs
            total["mse"] += mse.item() * bs
            total["residual"] += residual_reg.item() * bs
            total["prefix_ce"] += prefix_ce.item() * bs
            total["prefix_acc"] += (logits.argmax(dim=-1) == y_prefix).float().mean().item() * bs
            count += bs
        row = {
            "epoch": epoch,
            "grl_weight": grl_weight,
            **{f"train_{k}": v / max(count, 1) for k, v in total.items()},
        }
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
    best_path = out / "best_model.pth"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    summary = save_outputs(model, test_loader, device, out)
    print("TEST", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate_conditional_diffusion import approx_emd_sorted, distribution_metrics
from models import Generator
from train_conditional_diffusion import VehiclePointCloudDataset, chamfer_distance, make_split, set_seed
from train_retrieval_residual_diffusion import RetrievalBatchDataset, RetrievalPriorIndex, make_collate


class HybridFusionRefiner(nn.Module):
    def __init__(self, global_dim=384, hidden_dim=384, dropout=0.0, use_aux_prior=False):
        super().__init__()
        self.use_aux_prior = use_aux_prior
        local_dim = 30 if use_aux_prior else 18
        self.encoder = nn.Sequential(
            nn.Linear(local_dim, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU(),
            nn.Linear(256, global_dim), nn.SiLU(),
        )
        self.body = nn.Sequential(
            nn.Linear(local_dim + global_dim, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, cond, base, prior, aux_prior=None):
        if self.use_aux_prior:
            if aux_prior is None:
                raise ValueError("aux_prior is required when use_aux_prior=True")
            local = torch.cat(
                [
                    cond,
                    base,
                    prior,
                    aux_prior,
                    base - cond,
                    prior - cond,
                    aux_prior - cond,
                    prior - base,
                    aux_prior - base,
                    aux_prior - prior,
                ],
                dim=-1,
            )
        else:
            local = torch.cat([cond, base, prior, base - cond, prior - cond, prior - base], dim=-1)
        global_feat = self.encoder(local).max(dim=1).values[:, None, :].expand(cond.shape[0], cond.shape[1], -1)
        residual = self.body(torch.cat([local, global_feat], dim=-1))
        return base + residual


def select_train_subset(train_idx, seed, train_fraction=1.0, train_count=None):
    train_idx = list(train_idx)
    if train_count is not None:
        keep = max(1, min(int(train_count), len(train_idx)))
    else:
        keep = max(1, min(len(train_idx), int(round(len(train_idx) * float(train_fraction)))))
    if keep >= len(train_idx):
        return train_idx
    rng = torch.Generator().manual_seed(int(seed) + 104729)
    perm = torch.randperm(len(train_idx), generator=rng).tolist()
    return [train_idx[i] for i in perm[:keep]]


def make_base(generator, cond, latent_dim, noise_mode):
    if noise_mode == "zero":
        z = torch.zeros(cond.shape[0], latent_dim, device=cond.device)
    else:
        z = torch.randn(cond.shape[0], latent_dim, device=cond.device)
    return generator(z, cond)


def evaluate(generator, refiner, loader, latent_dim, device, noise_mode) -> Dict[str, float]:
    generator.eval()
    refiner.eval()
    total = {"cd": 0.0, "base_cd": 0.0, "retrieval_cd": 0.0, "mse": 0.0, "cond_cd": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            target = batch["target"].to(device)
            cond = batch["cond"].to(device)
            prior = batch["prior"].to(device)
            aux_prior = batch.get("aux_prior")
            aux_prior = aux_prior.to(device) if aux_prior is not None else None
            base = make_base(generator, cond, latent_dim, noise_mode)
            gen = refiner(cond, base, prior, aux_prior)
            bs = target.shape[0]
            total["cd"] += chamfer_distance(gen, target).item() * bs
            total["base_cd"] += chamfer_distance(base, target).item() * bs
            total["retrieval_cd"] += chamfer_distance(prior, target).item() * bs
            total["mse"] += F.mse_loss(gen, target).item() * bs
            total["cond_cd"] += chamfer_distance(gen, cond).item() * bs
            count += bs
    return {k: v / max(count, 1) for k, v in total.items()}


@torch.no_grad()
def save_test_outputs(generator, refiner, loader, latent_dim, device, noise_mode, out: Path, output_policy="refined"):
    generator.eval()
    refiner.eval()
    sample_dir = out / "samples"
    sample_dir.mkdir(exist_ok=True)
    all_real, all_base, all_prior, all_gen = [], [], [], []
    per_rows: List[Dict] = []
    for batch in loader:
        target = batch["target"].to(device)
        cond = batch["cond"].to(device)
        prior = batch["prior"].to(device)
        aux_prior = batch.get("aux_prior")
        aux_prior = aux_prior.to(device) if aux_prior is not None else None
        names = batch["name"]
        base = make_base(generator, cond, latent_dim, noise_mode)
        refined = refiner(cond, base, prior, aux_prior)
        if output_policy == "refined":
            gen = refined
        elif output_policy == "base":
            gen = base
        elif output_policy == "prior":
            gen = prior
        else:
            raise ValueError(f"Unknown output_policy: {output_policy}")
        cd = torch.stack([chamfer_distance(gen[i:i + 1], target[i:i + 1]) for i in range(target.shape[0])])
        bcd = torch.stack([chamfer_distance(base[i:i + 1], target[i:i + 1]) for i in range(target.shape[0])])
        rcd = torch.stack([chamfer_distance(prior[i:i + 1], target[i:i + 1]) for i in range(target.shape[0])])
        ccd = torch.stack([chamfer_distance(gen[i:i + 1], cond[i:i + 1]) for i in range(target.shape[0])])
        emd = approx_emd_sorted(gen, target)
        for i, name in enumerate(names):
            np.save(sample_dir / f"{name}-hybrid.npy", gen[i].detach().cpu().numpy())
            per_rows.append({
                "name": name,
                "cd": float(cd[i].item()),
                "base_cd": float(bcd[i].item()),
                "retrieval_cd": float(rcd[i].item()),
                "approx_emd": float(emd.item()),
                "cond_cd": float(ccd[i].item()),
            })
        all_real.append(target.cpu())
        all_base.append(base.cpu())
        all_prior.append(prior.cpu())
        all_gen.append(gen.cpu())
    real = torch.cat(all_real).to(device)
    base = torch.cat(all_base).to(device)
    prior = torch.cat(all_prior).to(device)
    gen = torch.cat(all_gen).to(device)
    summary = {
        "mean_cd": float(np.mean([r["cd"] for r in per_rows])),
        "mean_base_cd": float(np.mean([r["base_cd"] for r in per_rows])),
        "mean_retrieval_cd": float(np.mean([r["retrieval_cd"] for r in per_rows])),
        "mean_approx_emd": float(np.mean([r["approx_emd"] for r in per_rows])),
        "mean_cond_cd": float(np.mean([r["cond_cd"] for r in per_rows])),
        "num_test": len(per_rows),
        "model": "hybrid_cgan_retrieval_refiner",
        "output_policy": output_policy,
    }
    summary.update({f"gen_{k}": v for k, v in distribution_metrics(gen, real).items()})
    summary.update({f"base_{k}": v for k, v in distribution_metrics(base, real).items()})
    summary.update({f"retrieval_{k}": v for k, v in distribution_metrics(prior, real).items()})
    with open(out / "per_sample_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_rows, f, indent=2, ensure_ascii=False)
    with open(out / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def make_hybrid_collate(prior_index: RetrievalPriorIndex, exclude_self: bool, aux_index: RetrievalPriorIndex | None = None):
    def collate(batch):
        target = torch.stack([b["target"] for b in batch], dim=0)
        cond = torch.stack([b["cond"] for b in batch], dim=0)
        names = [b["name"] for b in batch]
        cond_device = cond.to(prior_index.device)
        prior = prior_index.retrieve(cond_device, names, k_exclude_self=exclude_self).detach().cpu()
        row = {"target": target, "cond": cond, "prior": prior, "name": names}
        if aux_index is not None:
            row["aux_prior"] = aux_index.retrieve(cond_device, names, k_exclude_self=exclude_self).detach().cpu()
        return row

    return collate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_generator", required=True)
    p.add_argument("--point_dir", default="models_pointcloud_npy")
    p.add_argument("--label_dir", default="models_labels_npy")
    p.add_argument("--output_dir", default="runs/hybrid_cgan_retrieval")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--latent_dim", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden_dim", type=int, default=384)
    p.add_argument("--global_dim", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lambda_cd", type=float, default=1.0)
    p.add_argument("--lambda_mse", type=float, default=0.0)
    p.add_argument("--lambda_prior", type=float, default=0.0)
    p.add_argument("--noise_mode", choices=["random", "zero"], default="random")
    p.add_argument("--train_fraction", type=float, default=1.0)
    p.add_argument("--train_count", type=int, default=None)
    p.add_argument("--split_json_in", default=None)
    p.add_argument(
        "--retrieval_metric",
        choices=["descriptor", "flat_l2", "prefix_descriptor", "prefix_flat_l2"],
        default="descriptor",
    )
    p.add_argument(
        "--aux_retrieval_metric",
        choices=["none", "descriptor", "flat_l2", "prefix_descriptor", "prefix_flat_l2"],
        default="none",
    )
    p.add_argument("--freeze_base", action="store_true")
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
    if args.split_json_in:
        with open(args.split_json_in, "r", encoding="utf-8") as f:
            split = json.load(f)
        full_train_idx = split["train"]
        val_idx = split["val"]
        test_idx = split["test"]
    else:
        full_train_idx, val_idx, test_idx = make_split(len(dataset), args.seed)
    train_idx = select_train_subset(full_train_idx, args.seed, args.train_fraction, args.train_count)
    with open(out / "split.json", "w", encoding="utf-8") as f:
        json.dump(
            {"train": train_idx, "full_train": full_train_idx, "val": val_idx, "test": test_idx, "names": dataset.names},
            f,
            indent=2,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior_index = RetrievalPriorIndex(dataset, train_idx, device, retrieval_metric=args.retrieval_metric)
    aux_index = None
    if args.aux_retrieval_metric != "none":
        aux_index = RetrievalPriorIndex(dataset, train_idx, device, retrieval_metric=args.aux_retrieval_metric)
    train_loader = DataLoader(
        RetrievalBatchDataset(dataset, train_idx, prior_index),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=make_hybrid_collate(prior_index, exclude_self=True, aux_index=aux_index),
    )
    val_loader = DataLoader(
        RetrievalBatchDataset(dataset, val_idx, prior_index),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_hybrid_collate(prior_index, exclude_self=False, aux_index=aux_index),
    )
    test_loader = DataLoader(
        RetrievalBatchDataset(dataset, test_idx, prior_index),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_hybrid_collate(prior_index, exclude_self=False, aux_index=aux_index),
    )

    generator = Generator(latent_dim=args.latent_dim).to(device)
    generator.load_state_dict(torch.load(args.base_generator, map_location=device))
    refiner = HybridFusionRefiner(
        args.global_dim,
        args.hidden_dim,
        args.dropout,
        use_aux_prior=args.aux_retrieval_metric != "none",
    ).to(device)
    params = list(refiner.parameters())
    if args.freeze_base:
        for p0 in generator.parameters():
            p0.requires_grad_(False)
    else:
        params += list(generator.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    best_val = float("inf")

    print(
        f"Dataset pairs={len(dataset)}, train/val/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)}, "
        f"device={device}, method=hybrid_cgan_retrieval, freeze_base={args.freeze_base}"
    )
    for epoch in range(1, args.epochs + 1):
        generator.train(not args.freeze_base)
        refiner.train()
        total = {"loss": 0.0, "cd": 0.0, "base_cd": 0.0, "retrieval_cd": 0.0, "mse": 0.0, "prior_cd": 0.0}
        count = 0
        for batch in train_loader:
            target = batch["target"].to(device)
            cond = batch["cond"].to(device)
            prior = batch["prior"].to(device)
            aux_prior = batch.get("aux_prior")
            aux_prior = aux_prior.to(device) if aux_prior is not None else None
            if args.freeze_base:
                with torch.no_grad():
                    base = make_base(generator, cond, args.latent_dim, args.noise_mode)
            else:
                base = make_base(generator, cond, args.latent_dim, args.noise_mode)
            gen = refiner(cond, base, prior, aux_prior)
            cd = chamfer_distance(gen, target)
            mse = F.mse_loss(gen, target)
            prior_cd = chamfer_distance(gen, prior)
            loss = args.lambda_cd * cd + args.lambda_mse * mse + args.lambda_prior * prior_cd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            bs = target.shape[0]
            total["loss"] += loss.item() * bs
            total["cd"] += cd.item() * bs
            total["base_cd"] += chamfer_distance(base.detach(), target).item() * bs
            total["retrieval_cd"] += chamfer_distance(prior, target).item() * bs
            total["mse"] += mse.item() * bs
            total["prior_cd"] += prior_cd.item() * bs
            count += bs
        row = {"epoch": epoch, **{f"train_{k}": v / max(count, 1) for k, v in total.items()}}
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            val = evaluate(generator, refiner, val_loader, args.latent_dim, device, args.noise_mode)
            row.update({f"val_{k}": v for k, v in val.items()})
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if val["cd"] < best_val:
                best_val = val["cd"]
                torch.save({
                    "generator": generator.state_dict(),
                    "refiner": refiner.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                }, out / "best_model.pth")
        elif epoch % 5 == 0:
            print(json.dumps(row, ensure_ascii=False), flush=True)
        if epoch % args.save_every == 0:
            torch.save({
                "generator": generator.state_dict(),
                "refiner": refiner.state_dict(),
                "args": vars(args),
                "epoch": epoch,
            }, out / f"checkpoint_{epoch}.pth")
        with open(out / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    torch.save({"generator": generator.state_dict(), "refiner": refiner.state_dict(), "args": vars(args), "epoch": args.epochs}, out / "last_model.pth")
    summary = save_test_outputs(generator, refiner, test_loader, args.latent_dim, device, args.noise_mode, out)
    print("TEST", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

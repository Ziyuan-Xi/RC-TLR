import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from train_conditional_diffusion import (
    SinusoidalTimeEmbedding,
    VehiclePointCloudDataset,
    chamfer_distance,
    make_split,
    set_seed,
)


def condition_descriptor(x: torch.Tensor) -> torch.Tensor:
    """Compact descriptor for retrieving similar engineering-condition point arrays."""
    # x can be (N,3) or (B,N,3)
    squeeze = False
    if x.dim() == 2:
        x = x.unsqueeze(0)
        squeeze = True
    q_levels = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=x.device, dtype=x.dtype)
    quantiles = torch.quantile(x, q_levels, dim=1).permute(1, 0, 2).reshape(x.shape[0], -1)
    desc = torch.cat(
        [
            x.mean(dim=1),
            x.std(dim=1),
            x.min(dim=1).values,
            x.max(dim=1).values,
            quantiles,
        ],
        dim=1,
    )
    return desc.squeeze(0) if squeeze else desc


class RetrievalPriorIndex:
    def __init__(
        self,
        dataset: VehiclePointCloudDataset,
        train_idx: Sequence[int],
        device: torch.device,
        retrieval_metric: str = "descriptor",
    ):
        self.device = device
        self.retrieval_metric = retrieval_metric
        self.names: List[str] = []
        descs, conds, targets = [], [], []
        for idx in train_idx:
            item = dataset[idx]
            self.names.append(item["name"])
            descs.append(condition_descriptor(item["cond"]).float())
            conds.append(item["cond"].float())
            targets.append(item["target"].float())
        self.desc = torch.stack(descs).to(device)
        self.conds = torch.stack(conds).to(device)
        self.conds_flat = self.conds.reshape(self.conds.shape[0], -1)
        self.targets = torch.stack(targets).to(device)

    def retrieve(self, cond: torch.Tensor, names: Sequence[str] | None = None, k_exclude_self: bool = True) -> torch.Tensor:
        if self.retrieval_metric in {"descriptor", "prefix_descriptor"}:
            q = condition_descriptor(cond).to(self.device)
            dist = torch.cdist(q, self.desc, p=2)
        elif self.retrieval_metric in {"flat_l2", "prefix_flat_l2"}:
            q = cond.reshape(cond.shape[0], -1).to(self.device)
            dist = torch.cdist(q, self.conds_flat, p=2)
        else:
            raise ValueError(f"Unknown retrieval_metric: {self.retrieval_metric}")
        if names is not None and k_exclude_self:
            name_to_pos = {name: i for i, name in enumerate(self.names)}
            for row, name in enumerate(names):
                pos = name_to_pos.get(str(name))
                if pos is not None and self.desc.shape[0] > 1:
                    dist[row, pos] = float("inf")
        if names is not None and self.retrieval_metric.startswith("prefix_"):
            prefixes = [str(name)[:2] for name in names]
            for row, prefix in enumerate(prefixes):
                mask = torch.tensor([name[:2] == prefix for name in self.names], device=self.device)
                if mask.any():
                    dist[row, ~mask] = float("inf")
        nn_idx = dist.argmin(dim=1)
        return self.targets[nn_idx]


class RetrievalBatchDataset(Dataset):
    def __init__(self, base: VehiclePointCloudDataset, indices: Sequence[int], prior_index: RetrievalPriorIndex):
        self.base = base
        self.indices = list(indices)
        self.prior_index = prior_index

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, pos: int):
        item = self.base[self.indices[pos]]
        # Retrieval is performed in collate_fn to keep CUDA tensors out of workers.
        return item


def make_collate(prior_index: RetrievalPriorIndex, exclude_self: bool):
    def collate(batch):
        target = torch.stack([b["target"] for b in batch], dim=0)
        cond = torch.stack([b["cond"] for b in batch], dim=0)
        names = [b["name"] for b in batch]
        prior = prior_index.retrieve(cond.to(prior_index.device), names, k_exclude_self=exclude_self).detach().cpu()
        return {"target": target, "cond": cond, "prior": prior, "name": names}

    return collate


class PriorConditionEncoder(nn.Module):
    def __init__(self, out_dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 96), nn.SiLU(),
            nn.Linear(96, 192), nn.SiLU(),
            nn.Linear(192, out_dim), nn.SiLU(),
        )

    def forward(self, cond: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([cond, prior, prior - cond], dim=-1)
        return self.net(feat).max(dim=1).values


class RetrievalResidualDenoiser(nn.Module):
    def __init__(self, cond_dim=384, time_dim=128, hidden_dim=384):
        super().__init__()
        self.cond_encoder = PriorConditionEncoder(cond_dim)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        in_dim = 3 + 3 + 3 + 3 + cond_dim + time_dim
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, prior: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b, n, _ = x_t.shape
        c = self.cond_encoder(cond, prior)[:, None, :].expand(b, n, -1)
        te = self.time_embed(t)[:, None, :].expand(b, n, -1)
        feats = torch.cat([x_t, cond, prior, prior - cond, c, te], dim=-1)
        return self.body(feats)


class RetrievalResidualDiffusion(nn.Module):
    def __init__(self, model: nn.Module, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        om = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        return a * x0 + om * noise

    def predict_x0(self, x_t, t, eps):
        a = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        om = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        return (x_t - om * eps) / a.clamp_min(1e-8)

    def training_loss(self, target, cond, prior, lambda_cd=0.0, lambda_prior=0.0, residual_scale=0.1):
        residual = (target - prior) / max(residual_scale, 1e-6)
        b = residual.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=residual.device)
        noise = torch.randn_like(residual)
        x_t = self.q_sample(residual, t, noise)
        eps_pred = self.model(x_t, cond, prior, t)
        eps_loss = F.mse_loss(eps_pred, noise)
        loss = eps_loss
        cd = torch.tensor(0.0, device=residual.device)
        prior_reg = torch.tensor(0.0, device=residual.device)
        if lambda_cd > 0 or lambda_prior > 0:
            pred_res = self.predict_x0(x_t, t, eps_pred).clamp(-3.0, 3.0) * residual_scale
            gen = prior + pred_res
            if lambda_cd > 0:
                cd = chamfer_distance(gen, target)
                loss = loss + lambda_cd * cd
            if lambda_prior > 0:
                prior_reg = chamfer_distance(gen, prior)
                loss = loss + lambda_prior * prior_reg
        return loss, {"eps_mse": eps_loss.detach(), "x0_cd": cd.detach(), "prior_cd": prior_reg.detach()}

    @torch.no_grad()
    def sample_residual(self, cond, prior, steps=None, sampler="ddim", residual_scale=0.1):
        self.eval()
        b, n, _ = cond.shape
        x = torch.randn(b, n, 3, device=cond.device)
        if steps is None or steps >= self.timesteps:
            time_seq = list(range(self.timesteps - 1, -1, -1))
            use_ddpm = sampler == "ddpm"
        else:
            time_seq = np.linspace(self.timesteps - 1, 0, steps, dtype=int).tolist()
            use_ddpm = False
        for i, t_int in enumerate(time_seq):
            t = torch.full((b,), int(t_int), device=cond.device, dtype=torch.long)
            eps = self.model(x, cond, prior, t)
            if use_ddpm:
                beta_t = self.betas[t].view(-1, 1, 1)
                sqrt_one_minus = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
                sqrt_recip_alpha = self.sqrt_recip_alphas[t].view(-1, 1, 1)
                mean = sqrt_recip_alpha * (x - beta_t * eps / sqrt_one_minus.clamp_min(1e-8))
                x = mean + torch.sqrt(beta_t) * torch.randn_like(x) if t_int > 0 else mean
            else:
                x0_pred = self.predict_x0(x, t, eps).clamp(-3.0, 3.0)
                if i == len(time_seq) - 1:
                    x = x0_pred
                else:
                    next_t = int(time_seq[i + 1])
                    a_next = self.sqrt_alpha_bar[next_t].view(1, 1, 1)
                    om_next = self.sqrt_one_minus_alpha_bar[next_t].view(1, 1, 1)
                    x = a_next * x0_pred + om_next * eps
        return x * residual_scale


@torch.no_grad()
def evaluate(model, loader, device, sample_steps=50, sampler="ddim", max_batches=None, residual_scale=0.1) -> Dict[str, float]:
    model.eval()
    total = {"eps_mse": 0.0, "sample_cd": 0.0, "retrieval_cd": 0.0, "cond_cd": 0.0}
    count = 0
    for bi, batch in enumerate(loader):
        target = batch["target"].to(device)
        cond = batch["cond"].to(device)
        prior = batch["prior"].to(device)
        loss, parts = model.training_loss(target, cond, prior, 0.0, 0.0, residual_scale=residual_scale)
        residual = model.sample_residual(cond, prior, steps=sample_steps, sampler=sampler, residual_scale=residual_scale)
        gen = prior + residual
        bs = target.shape[0]
        total["eps_mse"] += parts["eps_mse"].item() * bs
        total["sample_cd"] += chamfer_distance(gen, target).item() * bs
        total["retrieval_cd"] += chamfer_distance(prior, target).item() * bs
        total["cond_cd"] += chamfer_distance(gen, cond).item() * bs
        count += bs
        if max_batches and bi + 1 >= max_batches:
            break
    return {k: v / max(count, 1) for k, v in total.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--point_dir", default="models_pointcloud_npy")
    p.add_argument("--label_dir", default="models_labels_npy")
    p.add_argument("--output_dir", default="runs/rrdiff")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    p.add_argument("--hidden_dim", type=int, default=384)
    p.add_argument("--cond_dim", type=int, default=384)
    p.add_argument("--time_dim", type=int, default=128)
    p.add_argument("--lambda_cd", type=float, default=0.05)
    p.add_argument("--lambda_prior", type=float, default=0.0)
    p.add_argument("--residual_scale", type=float, default=0.1)
    p.add_argument("--normalize", choices=["none", "unit_sphere"], default="none")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=20)
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

    denoiser = RetrievalResidualDenoiser(args.cond_dim, args.time_dim, args.hidden_dim)
    diffusion = RetrievalResidualDiffusion(denoiser, timesteps=args.timesteps).to(device)
    opt = torch.optim.AdamW(diffusion.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    print(
        f"Dataset pairs={len(dataset)}, train/val/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)}, "
        f"device={device}, method=retrieval_residual_diffusion"
    )
    for epoch in range(1, args.epochs + 1):
        diffusion.train()
        accum = {"loss": 0.0, "eps_mse": 0.0, "x0_cd": 0.0, "prior_cd": 0.0}
        seen = 0
        for batch in train_loader:
            target = batch["target"].to(device)
            cond = batch["cond"].to(device)
            prior = batch["prior"].to(device)
            loss, parts = diffusion.training_loss(target, cond, prior, args.lambda_cd, args.lambda_prior, args.residual_scale)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            opt.step()
            bs = target.shape[0]
            accum["loss"] += loss.item() * bs
            accum["eps_mse"] += parts["eps_mse"].item() * bs
            accum["x0_cd"] += parts["x0_cd"].item() * bs
            accum["prior_cd"] += parts["prior_cd"].item() * bs
            seen += bs
        row = {"epoch": epoch, **{f"train_{k}": v / max(seen, 1) for k, v in accum.items()}}
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = evaluate(
                diffusion, val_loader, device,
                sample_steps=args.sample_steps,
                sampler=args.sampler,
                residual_scale=args.residual_scale,
                max_batches=None,
            )
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if val_metrics["sample_cd"] < best_val:
                best_val = val_metrics["sample_cd"]
                torch.save({"model": diffusion.state_dict(), "args": vars(args), "epoch": epoch}, out / "best_model.pth")
        elif epoch % 5 == 0:
            print(json.dumps(row, ensure_ascii=False), flush=True)
        if epoch % args.save_every == 0:
            torch.save({"model": diffusion.state_dict(), "args": vars(args), "epoch": epoch}, out / f"checkpoint_{epoch}.pth")
        with open(out / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    torch.save({"model": diffusion.state_dict(), "args": vars(args), "epoch": args.epochs}, out / "last_model.pth")
    test_metrics = evaluate(
        diffusion,
        test_loader,
        device,
        sample_steps=args.sample_steps,
        sampler=args.sampler,
        residual_scale=args.residual_scale,
    )
    with open(out / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    print("TEST", json.dumps(test_metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

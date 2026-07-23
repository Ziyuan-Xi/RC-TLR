import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


class VehiclePointCloudDataset(Dataset):
    """Paired condition/target point-cloud dataset used by the original IGD CGAN code."""

    def __init__(self, point_dir: str, label_dir: str, normalize: str = "none"):
        self.point_dir = Path(point_dir)
        self.label_dir = Path(label_dir)
        self.normalize = normalize
        point_files = {p.name: p for p in self.point_dir.glob("*.npy")}
        label_files = {p.name: p for p in self.label_dir.glob("*.npy")}
        common = sorted(set(point_files) & set(label_files))
        if not common:
            raise RuntimeError(
                f"No matched .npy basenames between {self.point_dir} and {self.label_dir}. "
                "The original repo expects paired files with identical names."
            )
        missing_points = sorted(set(label_files) - set(point_files))[:10]
        missing_labels = sorted(set(point_files) - set(label_files))[:10]
        if missing_points or missing_labels:
            print(
                f"[WARN] matched {len(common)} pairs; examples missing point={missing_points}, "
                f"missing label={missing_labels}"
            )
        self.names = common
        self.point_files = point_files
        self.label_files = label_files

    def __len__(self):
        return len(self.names)

    def _load(self, path: Path) -> torch.Tensor:
        arr = np.load(path).astype(np.float32)
        if arr.shape != (2048, 3):
            raise ValueError(f"Expected {path} to have shape (2048,3), got {arr.shape}")
        x = torch.from_numpy(arr)
        if self.normalize == "unit_sphere":
            x = x - x.mean(dim=0, keepdim=True)
            scale = torch.sqrt((x ** 2).sum(dim=1)).max().clamp_min(1e-6)
            x = x / scale
        return x

    def __getitem__(self, idx: int):
        name = self.names[idx]
        target = self._load(self.point_files[name])
        cond = self._load(self.label_files[name])
        return {"target": target, "cond": cond, "name": name}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_split(n: int, seed: int, train_ratio=0.8, val_ratio=0.1):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return idx[:n_train].tolist(), idx[n_train:n_train + n_val].tolist(), idx[n_train + n_val:].tolist()


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32) *
            (-math.log(10000.0) / max(half - 1, 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class PointNetConditionEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, out_dim), nn.ReLU(inplace=True),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        feat = self.net(cond)  # B,N,C
        return feat.max(dim=1).values


class ConditionalPointDenoiser(nn.Module):
    def __init__(self, cond_dim=256, time_dim=128, hidden_dim=256, injection="concat"):
        super().__init__()
        self.injection = injection
        self.cond_encoder = PointNetConditionEncoder(cond_dim)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        if injection == "film":
            self.point_in = nn.Linear(3 + time_dim, hidden_dim)
            self.film = nn.Linear(cond_dim, hidden_dim * 2)
            self.body = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3)
            )
        elif injection in {"concat", "concat_local"}:
            in_dim = 3 + cond_dim + time_dim + (3 if injection == "concat_local" else 0)
            self.body = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, 3),
            )
        else:
            raise ValueError("injection must be concat, concat_local or film")

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b, n, _ = x_t.shape
        c = self.cond_encoder(cond)
        te = self.time_embed(t)
        if self.injection in {"concat", "concat_local"}:
            c_expand = c[:, None, :].expand(b, n, -1)
            t_expand = te[:, None, :].expand(b, n, -1)
            feats = [x_t]
            if self.injection == "concat_local":
                feats.append(cond)
            feats.extend([c_expand, t_expand])
            return self.body(torch.cat(feats, dim=-1))
        h = self.point_in(torch.cat([x_t, te[:, None, :].expand(b, n, -1)], dim=-1))
        gamma, beta = self.film(c).chunk(2, dim=-1)
        h = h * (1 + gamma[:, None, :]) + beta[:, None, :]
        return self.body(h)


class GaussianDiffusion(nn.Module):
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

    def training_loss(self, x0, cond, lambda_x0_cd=0.0):
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = self.model(x_t, cond, t)
        eps_loss = F.mse_loss(eps_pred, noise)
        loss = eps_loss
        cd = torch.tensor(0.0, device=x0.device)
        if lambda_x0_cd > 0:
            x0_pred = self.predict_x0(x_t, t, eps_pred)
            cd = chamfer_distance(x0_pred, x0)
            loss = loss + lambda_x0_cd * cd
        return loss, {"eps_mse": eps_loss.detach(), "x0_cd": cd.detach()}

    @torch.no_grad()
    def sample(self, cond, num_points=2048, steps=None, sampler="ddpm"):
        self.eval()
        device = cond.device
        b = cond.shape[0]
        x = torch.randn(b, num_points, 3, device=device)
        if steps is None or steps >= self.timesteps:
            time_seq = list(range(self.timesteps - 1, -1, -1))
            use_ddpm = (sampler == "ddpm")
        else:
            # DDIM-style strided reverse process.  The previous implementation applied
            # the one-step DDPM posterior while skipping timesteps, which can severely
            # degrade quality at 50-500 steps.  Here x_{t_next} is reconstructed from
            # predicted x0 and eps with eta=0.
            time_seq = np.linspace(self.timesteps - 1, 0, steps, dtype=int).tolist()
            use_ddpm = False
        for i, t_int in enumerate(time_seq):
            t = torch.full((b,), int(t_int), device=device, dtype=torch.long)
            eps = self.model(x, cond, t)
            if use_ddpm:
                beta_t = self.betas[t].view(-1, 1, 1)
                sqrt_one_minus = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
                sqrt_recip_alpha = self.sqrt_recip_alphas[t].view(-1, 1, 1)
                mean = sqrt_recip_alpha * (x - beta_t * eps / sqrt_one_minus.clamp_min(1e-8))
                if t_int > 0:
                    noise = torch.randn_like(x)
                    x = mean + torch.sqrt(beta_t) * noise
                else:
                    x = mean
            else:
                x0_pred = self.predict_x0(x, t, eps).clamp(-2.5, 2.5)
                if i == len(time_seq) - 1:
                    x = x0_pred
                else:
                    next_t = int(time_seq[i + 1])
                    a_next = self.sqrt_alpha_bar[next_t].view(1, 1, 1)
                    om_next = self.sqrt_one_minus_alpha_bar[next_t].view(1, 1, 1)
                    x = a_next * x0_pred + om_next * eps
        return x


def chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(a, b, p=2)
    return 0.5 * (dist.min(dim=2).values.mean() + dist.min(dim=1).values.mean())


def evaluate(diffusion, loader, device, sample_steps=None, max_batches=None, target_mode="absolute"):
    diffusion.eval()
    total = {"eps_mse": 0.0, "sample_cd": 0.0, "cond_cd": 0.0}
    count = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            target = batch["target"].to(device)
            cond = batch["cond"].to(device)
            x0 = target - cond if target_mode == "residual" else target
            loss, parts = diffusion.training_loss(x0, cond, 0.0)
            sample = diffusion.sample(cond, x0.shape[1], steps=sample_steps)
            gen = cond + sample if target_mode == "residual" else sample
            bs = target.shape[0]
            total["eps_mse"] += parts["eps_mse"].item() * bs
            total["sample_cd"] += chamfer_distance(gen, target).item() * bs
            total["cond_cd"] += chamfer_distance(gen, cond).item() * bs
            count += bs
            if max_batches and bi + 1 >= max_batches:
                break
    return {k: v / max(count, 1) for k, v in total.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--point_dir", default="models_pointcloud_npy")
    p.add_argument("--label_dir", default="models_labels_npy")
    p.add_argument("--output_dir", default="runs/ddpm")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--cond_dim", type=int, default=256)
    p.add_argument("--time_dim", type=int, default=128)
    p.add_argument("--injection", choices=["concat", "concat_local", "film"], default="concat")
    p.add_argument("--lambda_x0_cd", type=float, default=0.0)
    p.add_argument("--target_mode", choices=["absolute", "residual"], default="absolute")
    p.add_argument("--normalize", choices=["none", "unit_sphere"], default="none")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=10)
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

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalPointDenoiser(args.cond_dim, args.time_dim, args.hidden_dim, args.injection)
    diffusion = GaussianDiffusion(model, timesteps=args.timesteps).to(device)
    opt = torch.optim.AdamW(diffusion.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    history: List[Dict] = []
    print(f"Dataset pairs={len(dataset)}, train/val/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)}, device={device}")
    for epoch in range(1, args.epochs + 1):
        diffusion.train()
        accum = {"loss": 0.0, "eps_mse": 0.0, "x0_cd": 0.0}
        seen = 0
        for batch in train_loader:
            target = batch["target"].to(device)
            cond = batch["cond"].to(device)
            x0 = target - cond if args.target_mode == "residual" else target
            loss, parts = diffusion.training_loss(x0, cond, args.lambda_x0_cd)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            opt.step()
            bs = x0.shape[0]
            accum["loss"] += loss.item() * bs
            accum["eps_mse"] += parts["eps_mse"].item() * bs
            accum["x0_cd"] += parts["x0_cd"].item() * bs
            seen += bs
        row = {"epoch": epoch, **{f"train_{k}": v / max(seen, 1) for k, v in accum.items()}}
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = evaluate(diffusion, val_loader, device, sample_steps=args.sample_steps, max_batches=2, target_mode=args.target_mode)
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            print(json.dumps(row, ensure_ascii=False))
            if val_metrics["eps_mse"] < best_val:
                best_val = val_metrics["eps_mse"]
                torch.save({"model": diffusion.state_dict(), "args": vars(args), "epoch": epoch}, out / "best_model.pth")
        elif epoch % 5 == 0:
            print(json.dumps(row, ensure_ascii=False))
        if epoch % args.save_every == 0:
            torch.save({"model": diffusion.state_dict(), "args": vars(args), "epoch": epoch}, out / f"checkpoint_{epoch}.pth")
        history.append(row)
        with open(out / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    torch.save({"model": diffusion.state_dict(), "args": vars(args), "epoch": args.epochs}, out / "last_model.pth")
    test_metrics = evaluate(diffusion, test_loader, device, sample_steps=args.sample_steps, max_batches=None, target_mode=args.target_mode)
    with open(out / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    print("TEST", json.dumps(test_metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()

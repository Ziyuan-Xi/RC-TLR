import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from train_conditional_diffusion import (
    ConditionalPointDenoiser,
    GaussianDiffusion,
    VehiclePointCloudDataset,
    chamfer_distance,
    make_split,
    set_seed,
)


def approx_emd_sorted(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cheap differentiability-free proxy: sort flattened coordinates then L1."""
    av = torch.sort(a.reshape(a.shape[0], -1), dim=1).values
    bv = torch.sort(b.reshape(b.shape[0], -1), dim=1).values
    return (av - bv).abs().mean()


def pairwise_cd(x: torch.Tensor, y: torch.Tensor, batch_y: int = 16) -> torch.Tensor:
    vals = []
    for i in range(x.shape[0]):
        row = []
        xi = x[i:i + 1].expand(min(batch_y, y.shape[0]), -1, -1)
        for j in range(0, y.shape[0], batch_y):
            yb = y[j:j + batch_y]
            xib = x[i:i + 1].expand(yb.shape[0], -1, -1)
            row.append(torch.stack([chamfer_distance(xib[k:k+1], yb[k:k+1]) for k in range(yb.shape[0])]))
        vals.append(torch.cat(row))
    return torch.stack(vals, dim=0)


def distribution_metrics(gen: torch.Tensor, real: torch.Tensor) -> Dict[str, float]:
    # gen/real are small test tensors on device. Pairwise loop is acceptable for 81 test samples.
    d_gr = pairwise_cd(gen, real)
    d_gg = pairwise_cd(gen, gen)
    d_rr = pairwise_cd(real, real)
    mmd = d_gr.min(dim=0).values.mean().item()
    cov = (d_gr.argmin(dim=0).unique().numel() / max(gen.shape[0], 1))
    # 1NN two-sample accuracy; diagonal self-distances ignored by setting to inf.
    n_g, n_r = gen.shape[0], real.shape[0]
    d_gg = d_gg.clone(); d_rr = d_rr.clone()
    d_gg.fill_diagonal_(float("inf")); d_rr.fill_diagonal_(float("inf"))
    correct = 0
    total = n_g + n_r
    for i in range(n_g):
        if d_gg[i].min() < d_gr[i].min():
            correct += 1
    d_rg = d_gr.t()
    for i in range(n_r):
        if d_rr[i].min() < d_rg[i].min():
            correct += 1
    diversity = d_gg[d_gg.isfinite()].mean().item() if n_g > 1 else 0.0
    return {"mmd_cd": mmd, "coverage_cd": cov, "one_nn_acc_cd": correct / total, "diversity_cd": diversity}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--point_dir", default="models_pointcloud_npy")
    p.add_argument("--label_dir", default="models_labels_npy")
    p.add_argument("--output_dir", default="eval/ddpm")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    p.add_argument("--num_samples_per_cond", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normalize", choices=["none", "unit_sphere"], default="none")
    p.add_argument("--target_mode", choices=["absolute", "residual"], default=None)
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = ckpt.get("args", {})
    dataset = VehiclePointCloudDataset(args.point_dir, args.label_dir, normalize=args.normalize)
    _, _, test_idx = make_split(len(dataset), args.seed)
    loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalPointDenoiser(
        train_args.get("cond_dim", 256), train_args.get("time_dim", 128),
        train_args.get("hidden_dim", 256), train_args.get("injection", "concat")
    )
    diffusion = GaussianDiffusion(model, timesteps=train_args.get("timesteps", 1000)).to(device)
    diffusion.load_state_dict(ckpt["model"])
    diffusion.eval()

    target_mode = args.target_mode or train_args.get("target_mode", "absolute")
    all_real, all_cond, all_gen = [], [], []
    per_rows: List[Dict] = []
    sample_dir = out / "samples"; sample_dir.mkdir(exist_ok=True)
    for batch in loader:
        real = batch["target"].to(device)
        cond = batch["cond"].to(device)
        names = batch["name"]
        gen_list = []
        for _ in range(args.num_samples_per_cond):
            sample = diffusion.sample(cond, real.shape[1], steps=args.sample_steps, sampler=args.sampler)
            gen_list.append(cond + sample if target_mode == "residual" else sample)
        gen = torch.stack(gen_list, dim=1).mean(dim=1)
        cd = torch.stack([chamfer_distance(gen[i:i+1], real[i:i+1]) for i in range(real.shape[0])])
        emd = approx_emd_sorted(gen, real)
        ccd = torch.stack([chamfer_distance(gen[i:i+1], cond[i:i+1]) for i in range(real.shape[0])])
        for i, name in enumerate(names):
            np.save(sample_dir / f"{name}-ddpm.npy", gen[i].detach().cpu().numpy())
            per_rows.append({"name": name, "cd": float(cd[i].item()), "approx_emd": float(emd.item()), "cond_cd": float(ccd[i].item())})
        all_real.append(real.cpu()); all_cond.append(cond.cpu()); all_gen.append(gen.cpu())

    real = torch.cat(all_real).to(device)
    cond = torch.cat(all_cond).to(device)
    gen = torch.cat(all_gen).to(device)
    summary = {
        "mean_cd": float(np.mean([r["cd"] for r in per_rows])),
        "mean_approx_emd": float(np.mean([r["approx_emd"] for r in per_rows])),
        "mean_cond_cd": float(np.mean([r["cond_cd"] for r in per_rows])),
        "num_test": len(per_rows),
        "sample_steps": args.sample_steps,
        "sampler": args.sampler,
        "target_mode": target_mode,
    }
    summary.update(distribution_metrics(gen, real))
    with open(out / "per_sample_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_rows, f, indent=2, ensure_ascii=False)
    with open(out / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

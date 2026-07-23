import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_conditional_diffusion import approx_emd_sorted, distribution_metrics
from train_conditional_diffusion import VehiclePointCloudDataset, chamfer_distance, make_split, set_seed
from train_retrieval_residual_diffusion import (
    RetrievalBatchDataset,
    RetrievalPriorIndex,
    RetrievalResidualDenoiser,
    RetrievalResidualDiffusion,
    make_collate,
)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--point_dir", default="models_pointcloud_npy")
    p.add_argument("--label_dir", default="models_labels_npy")
    p.add_argument("--output_dir", default="eval/rrdiff")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    p.add_argument("--num_samples_per_cond", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normalize", choices=["none", "unit_sphere"], default="none")
    p.add_argument("--residual_scale", type=float, default=None)
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = ckpt.get("args", {})
    dataset = VehiclePointCloudDataset(args.point_dir, args.label_dir, normalize=args.normalize)
    train_idx, _, test_idx = make_split(len(dataset), args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior_index = RetrievalPriorIndex(dataset, train_idx, device)
    loader = DataLoader(
        RetrievalBatchDataset(dataset, test_idx, prior_index),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate(prior_index, exclude_self=False),
    )

    denoiser = RetrievalResidualDenoiser(
        train_args.get("cond_dim", 384),
        train_args.get("time_dim", 128),
        train_args.get("hidden_dim", 384),
    )
    diffusion = RetrievalResidualDiffusion(denoiser, timesteps=train_args.get("timesteps", 1000)).to(device)
    diffusion.load_state_dict(ckpt["model"])
    diffusion.eval()
    residual_scale = args.residual_scale if args.residual_scale is not None else train_args.get("residual_scale", 0.1)

    sample_dir = out / "samples"
    sample_dir.mkdir(exist_ok=True)
    all_real, all_gen, all_prior = [], [], []
    per_rows: List[Dict] = []
    for batch in loader:
        real = batch["target"].to(device)
        cond = batch["cond"].to(device)
        prior = batch["prior"].to(device)
        names = batch["name"]
        gen_list = []
        for _ in range(args.num_samples_per_cond):
            residual = diffusion.sample_residual(cond, prior, steps=args.sample_steps, sampler=args.sampler, residual_scale=residual_scale)
            gen_list.append(prior + residual)
        gen = torch.stack(gen_list, dim=1).mean(dim=1)
        cd = torch.stack([chamfer_distance(gen[i:i + 1], real[i:i + 1]) for i in range(real.shape[0])])
        prior_cd = torch.stack([chamfer_distance(prior[i:i + 1], real[i:i + 1]) for i in range(real.shape[0])])
        cond_cd = torch.stack([chamfer_distance(gen[i:i + 1], cond[i:i + 1]) for i in range(real.shape[0])])
        emd = approx_emd_sorted(gen, real)
        for i, name in enumerate(names):
            np.save(sample_dir / f"{name}-rrdiff.npy", gen[i].detach().cpu().numpy())
            per_rows.append(
                {
                    "name": name,
                    "cd": float(cd[i].item()),
                    "retrieval_cd": float(prior_cd[i].item()),
                    "approx_emd": float(emd.item()),
                    "cond_cd": float(cond_cd[i].item()),
                }
            )
        all_real.append(real.cpu())
        all_gen.append(gen.cpu())
        all_prior.append(prior.cpu())

    real = torch.cat(all_real).to(device)
    gen = torch.cat(all_gen).to(device)
    prior = torch.cat(all_prior).to(device)
    summary = {
        "mean_cd": float(np.mean([r["cd"] for r in per_rows])),
        "mean_retrieval_cd": float(np.mean([r["retrieval_cd"] for r in per_rows])),
        "mean_approx_emd": float(np.mean([r["approx_emd"] for r in per_rows])),
        "mean_cond_cd": float(np.mean([r["cond_cd"] for r in per_rows])),
        "num_test": len(per_rows),
        "sample_steps": args.sample_steps,
        "sampler": args.sampler,
        "residual_scale": residual_scale,
        "model": "retrieval_residual_diffusion",
    }
    summary.update({f"gen_{k}": v for k, v in distribution_metrics(gen, real).items()})
    summary.update({f"retrieval_{k}": v for k, v in distribution_metrics(prior, real).items()})
    with open(out / "per_sample_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_rows, f, indent=2, ensure_ascii=False)
    with open(out / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

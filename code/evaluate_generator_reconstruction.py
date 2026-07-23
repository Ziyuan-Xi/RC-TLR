import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from evaluate_conditional_diffusion import approx_emd_sorted, distribution_metrics
from models import Generator
from train_conditional_diffusion import VehiclePointCloudDataset, chamfer_distance, make_split, set_seed


def make_noise(batch_size: int, latent_dim: int, device: torch.device, noise_mode: str) -> torch.Tensor:
    if noise_mode == "zero":
        return torch.zeros(batch_size, latent_dim, device=device)
    return torch.randn(batch_size, latent_dim, device=device)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generator", required=True)
    p.add_argument("--point_dir", default="models_pointcloud_npy")
    p.add_argument("--label_dir", default="models_labels_npy")
    p.add_argument("--output_dir", default="eval/generator_reconstruction")
    p.add_argument("--split_json", default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--latent_dim", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise_mode", choices=["random", "zero"], default="zero")
    p.add_argument("--normalize", choices=["none", "unit_sphere"], default="none")
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = VehiclePointCloudDataset(args.point_dir, args.label_dir, normalize=args.normalize)
    if args.split_json:
        with open(args.split_json, "r", encoding="utf-8") as f:
            split = json.load(f)
        test_idx = split["test"]
    else:
        _, _, test_idx = make_split(len(dataset), args.seed)
    loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(latent_dim=args.latent_dim).to(device)
    generator.load_state_dict(torch.load(args.generator, map_location=device))
    generator.eval()

    sample_dir = out / "samples"
    sample_dir.mkdir(exist_ok=True)
    all_real, all_cond, all_gen = [], [], []
    per_rows: List[Dict] = []
    for batch in loader:
        real = batch["target"].to(device)
        cond = batch["cond"].to(device)
        names = batch["name"]
        z = make_noise(cond.shape[0], args.latent_dim, device, args.noise_mode)
        gen = generator(z, cond)
        cd = torch.stack([chamfer_distance(gen[i:i + 1], real[i:i + 1]) for i in range(real.shape[0])])
        mse = torch.stack([F.mse_loss(gen[i:i + 1], real[i:i + 1]) for i in range(real.shape[0])])
        ccd = torch.stack([chamfer_distance(gen[i:i + 1], cond[i:i + 1]) for i in range(real.shape[0])])
        emd = approx_emd_sorted(gen, real)
        for i, name in enumerate(names):
            np.save(sample_dir / f"{name}-generator.npy", gen[i].detach().cpu().numpy())
            per_rows.append({
                "name": name,
                "cd": float(cd[i].item()),
                "mse": float(mse[i].item()),
                "approx_emd": float(emd.item()),
                "cond_cd": float(ccd[i].item()),
            })
        all_real.append(real.cpu())
        all_cond.append(cond.cpu())
        all_gen.append(gen.cpu())

    real = torch.cat(all_real).to(device)
    cond = torch.cat(all_cond).to(device)
    gen = torch.cat(all_gen).to(device)
    summary = {
        "mean_cd": float(np.mean([r["cd"] for r in per_rows])),
        "mean_mse": float(np.mean([r["mse"] for r in per_rows])),
        "mean_approx_emd": float(np.mean([r["approx_emd"] for r in per_rows])),
        "mean_cond_cd": float(np.mean([r["cond_cd"] for r in per_rows])),
        "num_test": len(per_rows),
        "model": "generator_reconstruction",
        "noise_mode": args.noise_mode,
    }
    summary.update({f"gen_{k}": v for k, v in distribution_metrics(gen, real).items()})
    summary.update({f"cond_{k}": v for k, v in distribution_metrics(cond, real).items()})
    with open(out / "per_sample_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_rows, f, indent=2, ensure_ascii=False)
    with open(out / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out / "eval_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

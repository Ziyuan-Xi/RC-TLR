import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from models import Generator
from train_conditional_diffusion import VehiclePointCloudDataset, chamfer_distance, make_split, set_seed
from evaluate_conditional_diffusion import approx_emd_sorted, distribution_metrics


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generator", default="gbest_model.pth")
    p.add_argument("--point_dir", default="models_pointcloud_npy")
    p.add_argument("--label_dir", default="models_labels_npy")
    p.add_argument("--output_dir", default="eval/cgan_gbest")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--latent_dim", type=int, default=100)
    p.add_argument("--num_samples_per_cond", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normalize", choices=["none", "unit_sphere"], default="none")
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dataset = VehiclePointCloudDataset(args.point_dir, args.label_dir, normalize=args.normalize)
    _, _, test_idx = make_split(len(dataset), args.seed)
    loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(latent_dim=args.latent_dim).to(device)
    state = torch.load(args.generator, map_location=device)
    generator.load_state_dict(state)
    generator.eval()

    sample_dir = out / "samples"; sample_dir.mkdir(exist_ok=True)
    all_real, all_cond, all_gen = [], [], []
    per_rows: List[Dict] = []
    for batch in loader:
        real = batch["target"].to(device)
        cond = batch["cond"].to(device)
        names = batch["name"]
        gen_list = []
        for _ in range(args.num_samples_per_cond):
            z = torch.randn(cond.shape[0], args.latent_dim, device=device)
            gen_list.append(generator(z, cond))
        gen = torch.stack(gen_list, dim=1).mean(dim=1)
        cd = torch.stack([chamfer_distance(gen[i:i+1], real[i:i+1]) for i in range(real.shape[0])])
        emd_proxy = approx_emd_sorted(gen, real)
        ccd = torch.stack([chamfer_distance(gen[i:i+1], cond[i:i+1]) for i in range(real.shape[0])])
        for i, name in enumerate(names):
            np.save(sample_dir / f"{name}-cgan.npy", gen[i].detach().cpu().numpy())
            per_rows.append({"name": name, "cd": float(cd[i].item()), "approx_emd": float(emd_proxy.item()), "cond_cd": float(ccd[i].item())})
        all_real.append(real.cpu()); all_cond.append(cond.cpu()); all_gen.append(gen.cpu())

    real = torch.cat(all_real).to(device)
    gen = torch.cat(all_gen).to(device)
    summary = {
        "mean_cd": float(np.mean([r["cd"] for r in per_rows])),
        "mean_approx_emd": float(np.mean([r["approx_emd"] for r in per_rows])),
        "mean_cond_cd": float(np.mean([r["cond_cd"] for r in per_rows])),
        "num_test": len(per_rows),
        "model": "cgan_gbest",
    }
    summary.update(distribution_metrics(gen, real))
    with open(out / "per_sample_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_rows, f, indent=2, ensure_ascii=False)
    with open(out / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

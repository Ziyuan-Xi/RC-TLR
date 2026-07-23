import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from models import Generator
from train_conditional_diffusion import VehiclePointCloudDataset, chamfer_distance, make_split, set_seed


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


def evaluate(generator, loader, latent_dim, device, noise_mode):
    generator.eval()
    total = {"cd": 0.0, "mse": 0.0, "cond_cd": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            real = batch["target"].to(device)
            cond = batch["cond"].to(device)
            if noise_mode == "zero":
                z = torch.zeros(real.shape[0], latent_dim, device=device)
            else:
                z = torch.randn(real.shape[0], latent_dim, device=device)
            gen = generator(z, cond)
            bs = real.shape[0]
            total["cd"] += chamfer_distance(gen, real).item() * bs
            total["mse"] += F.mse_loss(gen, real).item() * bs
            total["cond_cd"] += chamfer_distance(gen, cond).item() * bs
            count += bs
    return {k: v / max(count, 1) for k, v in total.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--point_dir", default="models_pointcloud_npy")
    p.add_argument("--label_dir", default="models_labels_npy")
    p.add_argument("--output_dir", default="runs/fair_cgan_reconstruction")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--latent_dim", type=int, default=100)
    p.add_argument("--init_generator", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda_cd", type=float, default=1.0)
    p.add_argument("--lambda_mse", type=float, default=0.0)
    p.add_argument("--noise_mode", choices=["random", "zero"], default="random")
    p.add_argument("--train_fraction", type=float, default=1.0)
    p.add_argument("--train_count", type=int, default=None)
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
    generator = Generator(latent_dim=args.latent_dim).to(device)
    if args.init_generator:
        generator.load_state_dict(torch.load(args.init_generator, map_location=device))
    opt = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    best_val = float("inf")

    print(
        f"Dataset pairs={len(dataset)}, train/val/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)}, "
        f"device={device}, method=fair_cgan_reconstruction"
    )
    for epoch in range(1, args.epochs + 1):
        generator.train()
        total = {"loss": 0.0, "cd": 0.0, "mse": 0.0}
        count = 0
        for batch in train_loader:
            real = batch["target"].to(device)
            cond = batch["cond"].to(device)
            if args.noise_mode == "zero":
                z = torch.zeros(real.shape[0], args.latent_dim, device=device)
            else:
                z = torch.randn(real.shape[0], args.latent_dim, device=device)
            gen = generator(z, cond)
            cd = chamfer_distance(gen, real)
            mse = F.mse_loss(gen, real)
            loss = args.lambda_cd * cd + args.lambda_mse * mse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt.step()
            bs = real.shape[0]
            total["loss"] += loss.item() * bs
            total["cd"] += cd.item() * bs
            total["mse"] += mse.item() * bs
            count += bs
        row = {"epoch": epoch, **{f"train_{k}": v / max(count, 1) for k, v in total.items()}}
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            val = evaluate(generator, val_loader, args.latent_dim, device, args.noise_mode)
            row.update({f"val_{k}": v for k, v in val.items()})
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if val["cd"] < best_val:
                best_val = val["cd"]
                torch.save(generator.state_dict(), out / "best_generator.pth")
        elif epoch % 5 == 0:
            print(json.dumps(row, ensure_ascii=False), flush=True)
        if epoch % args.save_every == 0:
            torch.save(generator.state_dict(), out / f"generator_{epoch}.pth")
        with open(out / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    torch.save(generator.state_dict(), out / "last_generator.pth")
    test = evaluate(generator, test_loader, args.latent_dim, device, args.noise_mode)
    with open(out / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2, ensure_ascii=False)
    print("TEST", json.dumps(test, ensure_ascii=False), flush=True)
    best_path = out / "best_generator.pth"
    if best_path.exists():
        generator.load_state_dict(torch.load(best_path, map_location=device))
        best_test = evaluate(generator, test_loader, args.latent_dim, device, args.noise_mode)
        with open(out / "test_best_metrics.json", "w", encoding="utf-8") as f:
            json.dump(best_test, f, indent=2, ensure_ascii=False)
        print("TEST_BEST", json.dumps(best_test, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

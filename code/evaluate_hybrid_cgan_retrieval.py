import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_conditional_diffusion import VehiclePointCloudDataset, make_split, set_seed
from train_hybrid_cgan_retrieval import HybridFusionRefiner, save_test_outputs
from train_hybrid_cgan_retrieval import make_hybrid_collate
from train_retrieval_residual_diffusion import RetrievalBatchDataset, RetrievalPriorIndex
from models import Generator


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--point_dir", default="models_pointcloud_npy")
    p.add_argument("--label_dir", default="models_labels_npy")
    p.add_argument("--output_dir", default="eval/hybrid_cgan_retrieval")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--noise_mode", choices=["random", "zero"], default=None)
    p.add_argument("--normalize", choices=["none", "unit_sphere"], default=None)
    p.add_argument(
        "--retrieval_metric",
        choices=["descriptor", "flat_l2", "prefix_descriptor", "prefix_flat_l2"],
        default=None,
    )
    p.add_argument(
        "--aux_retrieval_metric",
        choices=["none", "descriptor", "flat_l2", "prefix_descriptor", "prefix_flat_l2"],
        default=None,
    )
    p.add_argument("--output_policy", choices=["refined", "base", "prior", "adaptive"], default="refined")
    p.add_argument("--adaptive_fraction_threshold", type=float, default=0.5)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = ckpt.get("args", {})
    seed = args.seed if args.seed is not None else train_args.get("seed", 42)
    noise_mode = args.noise_mode or train_args.get("noise_mode", "random")
    normalize = args.normalize or train_args.get("normalize", "none")
    retrieval_metric = args.retrieval_metric or train_args.get("retrieval_metric", "descriptor")
    aux_retrieval_metric = args.aux_retrieval_metric
    if aux_retrieval_metric is None:
        aux_retrieval_metric = train_args.get("aux_retrieval_metric", "none")
    set_seed(seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = VehiclePointCloudDataset(args.point_dir, args.label_dir, normalize=normalize)
    split_path = Path(args.checkpoint).parent / "split.json"
    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        train_idx = split["train"]
        test_idx = split["test"]
    else:
        train_idx, _, test_idx = make_split(len(dataset), seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior_index = RetrievalPriorIndex(dataset, train_idx, device, retrieval_metric=retrieval_metric)
    aux_index = None
    if aux_retrieval_metric != "none":
        aux_index = RetrievalPriorIndex(dataset, train_idx, device, retrieval_metric=aux_retrieval_metric)
    test_loader = DataLoader(
        RetrievalBatchDataset(dataset, test_idx, prior_index),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_hybrid_collate(prior_index, exclude_self=False, aux_index=aux_index),
    )

    latent_dim = train_args.get("latent_dim", 100)
    generator = Generator(latent_dim=latent_dim).to(device)
    generator.load_state_dict(ckpt["generator"])
    refiner = HybridFusionRefiner(
        global_dim=train_args.get("global_dim", 384),
        hidden_dim=train_args.get("hidden_dim", 384),
        dropout=train_args.get("dropout", 0.0),
        use_aux_prior=aux_retrieval_metric != "none",
    ).to(device)
    refiner.load_state_dict(ckpt["refiner"])
    resolved_policy = args.output_policy
    if resolved_policy == "adaptive":
        train_fraction = float(train_args.get("train_fraction", 1.0))
        resolved_policy = "prior" if train_fraction <= args.adaptive_fraction_threshold else "refined"
    summary = save_test_outputs(
        generator,
        refiner,
        test_loader,
        latent_dim,
        device,
        noise_mode,
        out,
        output_policy=resolved_policy,
    )
    with open(out / "eval_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "seed": seed,
                "noise_mode": noise_mode,
                "normalize": normalize,
                "retrieval_metric": retrieval_metric,
                "aux_retrieval_metric": aux_retrieval_metric,
                "requested_output_policy": args.output_policy,
                "resolved_output_policy": resolved_policy,
                "adaptive_fraction_threshold": args.adaptive_fraction_threshold,
            },
            f,
            indent=2,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

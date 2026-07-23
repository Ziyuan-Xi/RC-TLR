import argparse
import json
from collections import Counter
from pathlib import Path

from train_conditional_diffusion import VehiclePointCloudDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--point_dir", default="models_pointcloud_npy")
    p.add_argument("--label_dir", default="models_labels_npy")
    p.add_argument("--output_dir", default="splits/prefix")
    p.add_argument("--folds", nargs="+", default=["dg:lx", "aj:cs", "dm:gd"])
    p.add_argument("--normalize", choices=["none", "unit_sphere"], default="none")
    args = p.parse_args()

    dataset = VehiclePointCloudDataset(args.point_dir, args.label_dir, normalize=args.normalize)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = [name[:2] for name in dataset.names]
    counts = Counter(prefix)
    manifest = {
        "num_pairs": len(dataset),
        "num_prefixes": len(counts),
        "prefix_counts": dict(sorted(counts.items())),
        "folds": [],
    }
    for fold_id, spec in enumerate(args.folds):
        test_prefix, val_prefix = spec.split(":")
        train = [i for i, pfx in enumerate(prefix) if pfx not in {test_prefix, val_prefix}]
        val = [i for i, pfx in enumerate(prefix) if pfx == val_prefix]
        test = [i for i, pfx in enumerate(prefix) if pfx == test_prefix]
        split = {
            "train": train,
            "val": val,
            "test": test,
            "names": dataset.names,
            "split_type": "prefix_holdout",
            "test_prefix": test_prefix,
            "val_prefix": val_prefix,
            "train_prefixes": sorted({prefix[i] for i in train}),
        }
        path = out / f"prefix_fold{fold_id}_{test_prefix}_test_{val_prefix}_val.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=2)
        manifest["folds"].append({
            "fold_id": fold_id,
            "path": str(path),
            "test_prefix": test_prefix,
            "val_prefix": val_prefix,
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
        })
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

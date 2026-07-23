#!/usr/bin/env python3
"""Generate the exact all-family prefix splits used by RC-TLR."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_FOLDS = (
    "aj:cs",
    "cs:dm",
    "dg:lx",
    "dm:gd",
    "gd:jc",
    "jc:kj",
    "kj:lt",
    "lt:lx",
    "lx:xd",
    "xd:aj",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--folds",
        nargs="+",
        default=list(DEFAULT_FOLDS),
        metavar="TEST:VAL",
    )
    args = parser.parse_args()

    condition_names = {path.name for path in args.conditions.glob("*.npy")}
    target_names = {path.name for path in args.targets.glob("*.npy")}
    names = sorted(condition_names & target_names)
    if not names:
        raise RuntimeError("No exact-basename condition/target pairs were found.")

    prefixes = [Path(name).stem[:2] for name in names]
    counts = Counter(prefixes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds = []

    for fold_id, specification in enumerate(args.folds):
        test_prefix, val_prefix = specification.split(":", maxsplit=1)
        if test_prefix not in counts or val_prefix not in counts:
            raise ValueError(
                f"Fold {specification!r} references a prefix absent from the data. "
                f"Available prefixes: {sorted(counts)}"
            )
        train = [
            index
            for index, prefix in enumerate(prefixes)
            if prefix not in {test_prefix, val_prefix}
        ]
        val = [index for index, prefix in enumerate(prefixes) if prefix == val_prefix]
        test = [index for index, prefix in enumerate(prefixes) if prefix == test_prefix]
        filename = (
            f"prefix_fold{fold_id}_{test_prefix}_test_{val_prefix}_val.json"
        )
        payload = {
            "train": train,
            "val": val,
            "test": test,
            "names": names,
            "split_type": "prefix_holdout",
            "test_prefix": test_prefix,
            "val_prefix": val_prefix,
            "train_prefixes": sorted(
                {prefixes[index] for index in train}
            ),
        }
        (args.output_dir / filename).write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        folds.append(
            {
                "fold_id": fold_id,
                "path": filename,
                "test_prefix": test_prefix,
                "val_prefix": val_prefix,
                "train_size": len(train),
                "val_size": len(val),
                "test_size": len(test),
            }
        )

    manifest = {
        "matched_pairs": len(names),
        "prefix_counts": dict(sorted(counts.items())),
        "folds": folds,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

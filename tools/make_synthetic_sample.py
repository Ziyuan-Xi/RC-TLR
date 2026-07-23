#!/usr/bin/env python3
"""Create a tiny synthetic RC-TLR-format dataset for public smoke checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PREFIXES = ("aj", "cs", "dg", "dm", "gd", "jc", "kj", "lt", "lx", "xd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synthetic"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    conditions = args.output_dir / "conditions"
    targets = args.output_dir / "targets"
    conditions.mkdir(parents=True, exist_ok=True)
    targets.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    for index, prefix in enumerate(PREFIXES):
        points = rng.normal(size=(2048, 3)).astype(np.float32)
        points /= np.linalg.norm(points, axis=1, keepdims=True).clip(1e-6)
        scale = np.asarray(
            [1.0 + index * 0.02, 0.55 + index * 0.01, 0.40],
            dtype=np.float32,
        )
        target = points * scale
        condition = target * np.asarray([0.96, 1.04, 1.0], dtype=np.float32)
        name = f"{prefix}_synthetic.npy"
        np.save(conditions / name, condition)
        np.save(targets / name, target)

    print(f"Wrote {len(PREFIXES)} synthetic pairs to {args.output_dir}")


if __name__ == "__main__":
    main()

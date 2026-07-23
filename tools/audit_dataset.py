#!/usr/bin/env python3
"""Audit RC-TLR condition/target pairs and optional OBJ meshes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_SHAPE = (2048, 3)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def npy_metadata(path: Path) -> tuple[str, str]:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return "x".join(str(value) for value in array.shape), str(array.dtype)
    except Exception as exc:  # pragma: no cover - reports corrupt user data
        return f"ERROR: {exc}", ""


def index_meshes(root: Path | None) -> tuple[dict[str, Path], dict[str, list[str]]]:
    if root is None:
        return {}, {}
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*.obj")):
        grouped[path.stem].append(path)
    unique = {stem: paths[0] for stem, paths in grouped.items()}
    duplicates = {
        stem: [str(path.relative_to(root)) for path in paths]
        for stem, paths in grouped.items()
        if len(paths) > 1
    }
    return unique, duplicates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--meshes", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--hash",
        action="store_true",
        help="Compute SHA-256 checksums. This is slower for the full dataset.",
    )
    args = parser.parse_args()

    for path in (args.conditions, args.targets):
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {path}")
    if args.meshes is not None and not args.meshes.is_dir():
        raise FileNotFoundError(f"Mesh directory not found: {args.meshes}")

    conditions = {path.name: path for path in args.conditions.glob("*.npy")}
    targets = {path.name: path for path in args.targets.glob("*.npy")}
    mesh_by_stem, duplicate_mesh_stems = index_meshes(args.meshes)
    all_names = sorted(set(conditions) | set(targets))
    common = sorted(set(conditions) & set(targets))

    rows: list[dict[str, Any]] = []
    shape_errors: list[dict[str, str]] = []
    for name in all_names:
        condition = conditions.get(name)
        target = targets.get(name)
        status = "matched"
        if condition is None:
            status = "missing_condition"
        elif target is None:
            status = "missing_target"

        condition_shape, condition_dtype = ("", "")
        target_shape, target_dtype = ("", "")
        if condition is not None:
            condition_shape, condition_dtype = npy_metadata(condition)
        if target is not None:
            target_shape, target_dtype = npy_metadata(target)
        for role, shape in (("condition", condition_shape), ("target", target_shape)):
            if shape and shape != "x".join(map(str, EXPECTED_SHAPE)):
                shape_errors.append({"name": name, "role": role, "shape": shape})

        stem = Path(name).stem
        mesh = mesh_by_stem.get(stem)
        rows.append(
            {
                "name": name,
                "prefix": stem[:2],
                "status": status,
                "condition_bytes": condition.stat().st_size if condition else "",
                "target_bytes": target.stat().st_size if target else "",
                "condition_shape": condition_shape,
                "target_shape": target_shape,
                "condition_dtype": condition_dtype,
                "target_dtype": target_dtype,
                "condition_sha256": sha256(condition) if args.hash and condition else "",
                "target_sha256": sha256(target) if args.hash and target else "",
                "mesh_path": (
                    str(mesh.relative_to(args.meshes)).replace("\\", "/")
                    if mesh is not None and args.meshes is not None
                    else ""
                ),
                "mesh_bytes": mesh.stat().st_size if mesh else "",
                "mesh_sha256": sha256(mesh) if args.hash and mesh else "",
            }
        )

    prefix_counts = Counter(Path(name).stem[:2] for name in common)
    summary = {
        "conditions": len(conditions),
        "targets": len(targets),
        "meshes": len(mesh_by_stem),
        "matched_pairs": len(common),
        "missing_condition": sorted(set(targets) - set(conditions)),
        "missing_target": sorted(set(conditions) - set(targets)),
        "prefix_pair_counts": dict(sorted(prefix_counts.items())),
        "expected_shape": list(EXPECTED_SHAPE),
        "shape_errors": shape_errors,
        "duplicate_mesh_stems": duplicate_mesh_stems,
        "checksums_included": args.hash,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["name"])
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if shape_errors:
        raise SystemExit("Dataset audit completed with shape/read errors.")


if __name__ == "__main__":
    main()

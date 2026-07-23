#!/usr/bin/env python3
"""Run lightweight checks before publishing the RC-TLR repository."""

from __future__ import annotations

import compileall
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "requirements.txt",
    "code/evaluate_rc_tlr_v2.py",
    "code/train_target_latent_retrieval.py",
    "data/DATASET_CARD.md",
    "data/dataset_audit.json",
    "splits/prefix_all10/manifest.json",
    "configs/rc_tlr_v2_conservative.json",
)


def main() -> None:
    errors: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"Missing required file: {relative}")

    if not compileall.compile_dir(ROOT / "code", quiet=1):
        errors.append("One or more Python source files failed to compile.")
    if not compileall.compile_dir(ROOT / "tools", quiet=1):
        errors.append("One or more tool files failed to compile.")

    audit_path = ROOT / "data/dataset_audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("matched_pairs") != 799:
            errors.append(
                "The committed full-dataset audit should report 799 matched pairs."
            )
        if audit.get("shape_errors"):
            errors.append("The committed dataset audit reports shape/read errors.")

    split_manifest = ROOT / "splits/prefix_all10/manifest.json"
    if split_manifest.is_file():
        splits = json.loads(split_manifest.read_text(encoding="utf-8"))
        if len(splits.get("folds", [])) != 10:
            errors.append("Expected ten prefix-holdout folds.")

    oversized = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or "_release_assets" in path.parts:
            continue
        if path.stat().st_size >= 100 * 1024 * 1024:
            oversized.append(str(path.relative_to(ROOT)))
    if oversized:
        errors.append(f"Files at or above GitHub's 100 MiB limit: {oversized}")

    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
        raise SystemExit(1)
    print("RC-TLR release checks passed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run RC-TLR v2 from a checked-in JSON configuration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "rc_tlr_v2_conservative.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    command = [sys.executable, str(ROOT / "code" / "evaluate_rc_tlr_v2.py")]
    for key, value in config.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                command.append(flag)
        elif value is not None:
            command.extend([flag, str(value)])

    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

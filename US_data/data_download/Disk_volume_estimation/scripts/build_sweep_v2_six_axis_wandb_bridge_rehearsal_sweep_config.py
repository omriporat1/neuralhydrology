"""Serialize the disposable v2 rehearsal W&B sweep configuration offline.

This tool only writes the authoritative configuration returned by
``build_wandb_bridge_rehearsal_sweep_config_v2``.  It neither imports W&B
nor registers a sweep.  Registration is a separate, explicitly authorized
step: this tool fixes the future strict rehearsal manifest's PATH before
registration, but the manifest itself is written only after registration
returns the real disposable sweep ID.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baseline.sweep_v2_six_axis_config import (  # noqa: E402
    build_wandb_bridge_rehearsal_sweep_config_v2,
)


def main() -> None:
    """Parse the serializer CLI and atomically create its JSON output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--program", default="scripts/run_sweep_v2_six_axis_wandb_bridge.py")
    parser.add_argument(
        "--manifest-path",
        type=str,
        required=True,
        help="Absolute path for the future strict v2 rehearsal manifest.",
    )
    args = parser.parse_args()

    if not (Path(args.manifest_path).is_absolute() or PurePosixPath(args.manifest_path).is_absolute()):
        parser.error("--manifest-path must be absolute")

    config = build_wandb_bridge_rehearsal_sweep_config_v2(
        program=args.program,
        manifest_path=args.manifest_path,
    )
    payload = json.dumps(config, indent=2) + "\n"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise SystemExit(f"--output {args.output} already exists") from exc


if __name__ == "__main__":
    main()

"""Write a disposable, non-scientific Sweep-v1 REHEARSAL W&B sweep config.

Sibling of ``build_sweep_v1_production_sweep_config.py``: same five-axis
parameter domain, but the disposable ``metric`` and the ``command``'s extra
positional manifest-path argument route a real ``wandb agent`` invocation to
``run_sweep_v1_wandb_bridge.main_from_manifest`` in rehearsal mode instead of
the production-only ``main()`` entry point. See
``src.baseline.sweep_v1_execution.build_wandb_bridge_rehearsal_sweep_config``
for the full rationale.

This script never creates the sweep against the real W&B service -- it only
serializes the config JSON. Registering it (``wandb.sweep(...)``) is a
separate, explicit step.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baseline.sweep_v1_execution import build_wandb_bridge_rehearsal_sweep_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--program", default="scripts/run_sweep_v1_wandb_bridge.py")
    parser.add_argument("--manifest-path", type=str, required=True,
                        help="Absolute path to a rehearsal launch manifest already validated by "
                             "src.baseline.sweep_v1_wandb_bridge_manifest with mode='rehearsal'.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing --output file. Without this flag, an existing "
                             "target is a hard error.")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"--output {args.output} already exists; pass --force to overwrite it deliberately")
    config = build_wandb_bridge_rehearsal_sweep_config(program=args.program, manifest_path=args.manifest_path)
    args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

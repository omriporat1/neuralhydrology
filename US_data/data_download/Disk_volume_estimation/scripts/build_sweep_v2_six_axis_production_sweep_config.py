"""Serialize the reusable v2 six-axis PRODUCTION W&B sweep configuration offline.

This tool writes exactly the authoritative configuration returned by
:func:`src.baseline.sweep_v2_six_axis_config.build_production_sweep_config_v2`.
It never imports W&B and never registers or contacts a sweep.

The production controller command is deliberately static
(``["${interpreter}", "${program}"]``) and embeds no proposal-specific
launch manifest: one reusable Bayesian controller serves every serialized
production proposal/attempt, and each separately launched one-agent job
selects its one immutable strict ``mode=production`` manifest through the
bridge's ``FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST`` operational input. Because
the command omits the ``${args}`` macro, W&B never appends swept
``--key=value`` flags to the bridge process.

Registration of the controller is a separate, explicitly authorized step
(``scripts/create_sweep_v2_six_axis_wandb_bridge_production_sweep.py``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baseline.sweep_v2_six_axis_config import (  # noqa: E402
    build_production_sweep_config_v2,
)


def main() -> None:
    """Parse the serializer CLI and atomically create its JSON output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--program", default="scripts/run_sweep_v2_six_axis_wandb_bridge.py")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing --output file. Without this flag, an existing target is a "
        "hard error: a real production Sweep-v2 config must never be silently replaced once written.",
    )
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise SystemExit(f"--output {args.output} already exists; pass --force to overwrite it deliberately")

    config = build_production_sweep_config_v2(program=args.program)
    payload = json.dumps(config, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

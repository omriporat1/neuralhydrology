"""Write the deterministic, unhosted production Sweep-v1 W&B config."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.baseline.sweep_v1_execution import build_production_sweep_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--program", default="scripts/run_sweep_v1_wandb_bridge.py")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing --output file. Without this flag, an "
                             "existing target is a hard error: a real production Sweep-v1 "
                             "config must never be silently replaced once written.")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"--output {args.output} already exists; pass --force to overwrite it deliberately")
    args.output.write_text(json.dumps(build_production_sweep_config(program=args.program), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

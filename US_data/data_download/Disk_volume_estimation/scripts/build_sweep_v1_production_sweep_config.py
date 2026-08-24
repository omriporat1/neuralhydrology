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
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_production_sweep_config(program=args.program), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

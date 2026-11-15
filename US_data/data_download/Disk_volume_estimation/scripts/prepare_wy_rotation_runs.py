#!/usr/bin/env python
"""Prepare the water-year rotation baseline run configs (CPU only; no training).

Subcommands:
  configs -- derive the five cycle configs + campaign_manifest.json + a
             verified copy of the frozen split contract from the frozen P19
             config (``wy_rotation_runs.prepare_campaign``).
  canary  -- derive one canary config (budget/identity changes only) from a
             verified campaign row (``wy_rotation_runs.prepare_canary``).

Run it on the machine that will train (paths in the configs are absolute).
``--runtime-project-root`` defaults to this checkout's project root.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baseline import wy_rotation_runs as rot  # noqa: E402


def _inside_root(path: Path) -> Path:
    resolved = path.resolve()
    if ROOT.resolve() not in (resolved, *resolved.parents):
        raise SystemExit(f"refusing to write outside the project root: {resolved}")
    return resolved


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("configs")
    c.add_argument("--base-config", type=Path, required=True, help="frozen P19 config.yaml")
    c.add_argument("--base-manifest", type=Path, required=True, help="P19 generation manifest JSON")
    c.add_argument("--output-root", type=Path, required=True)
    c.add_argument("--runtime-project-root", type=Path, default=ROOT)
    k = sub.add_parser("canary")
    k.add_argument("--campaign-manifest", type=Path, required=True)
    k.add_argument("--cycle", type=int, required=True)
    k.add_argument("--epochs", type=int, required=True)
    k.add_argument("--max-updates-per-epoch", type=int, required=True)
    k.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args(argv)
    out = _inside_root(args.output_root)
    if args.cmd == "configs":
        result = rot.prepare_campaign(base_config_path=args.base_config, base_manifest_path=args.base_manifest,
                                      output_root=out, runtime_project_root=args.runtime_project_root)
        print(json.dumps([{k: r[k] for k in ("index", "cycle", "run_id", "config_sha256")} for r in result["runs"]],
                         indent=1))
    else:
        result = rot.prepare_canary(campaign_manifest_path=args.campaign_manifest, cycle=args.cycle,
                                    epochs=args.epochs, max_updates_per_epoch=args.max_updates_per_epoch,
                                    output_root=out)
        print(json.dumps({k: result[k] for k in ("run_id", "config_dir", "config_sha256",
                                                 "config_differences_vs_campaign_row")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

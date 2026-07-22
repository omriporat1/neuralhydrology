#!/usr/bin/env python
"""Generate ONE rendered Stage 1 NH 1.13 integration-validation config
(local implementation increment).

Reads the committed scientific baseline policy, the certified Stage 1
Compact Scientific Package's manifest + attributes.csv, and the canonical
split-list files, then renders a single (lead, seq_length) config plus
matching train/validation/test basin-list files and a generation manifest
under --out-dir. Local-only: never touches h2o or Moriah, never writes a
Slurm script, never configures W&B, and never generates the full 16-config
matrix (one lead/seq_length pair per invocation).

Usage:
    python scripts/generate_stage1_nh_config.py \\
        --package-root /path/to/stage1_compact_scientific_package_v001 \\
        --lead-hours 6 --seq-length 24 \\
        --out-dir tmp/stage1_nh_config_lead06_seq24_v001
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline.nh_config_generation import NHConfigGenerationError, generate_stage1_nh_config, write_generated_config


def _fail(message: str) -> None:
    print(f"FATAL: {message}", file=sys.stderr)
    sys.exit(1)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--package-root", required=True,
                   help="Path to the certified stage1_compact_scientific_package_v001 root")
    p.add_argument("--policy", default=str(REPO_ROOT / "config" / "stage1_scientific_baseline_v001.yaml"),
                   help="Path to the Stage 1 scientific baseline policy YAML")
    p.add_argument("--splits-dir", default=str(REPO_ROOT / "config" / "stage1_baseline_splits_v001"),
                   help="Directory containing development_train.txt / spatial_holdout_nonca.txt / california_all.txt")
    p.add_argument("--static-column-manifest", default=None,
                   help="Optional external static column-role manifest JSON for an additional "
                        "independent re-derivation check (not required)")
    p.add_argument("--lead-hours", type=int, required=True, help="One of the policy-approved leads (hours)")
    p.add_argument("--seq-length", type=int, required=True, help="One of the policy-approved sequence lengths (hours)")
    p.add_argument("--experiment-name", default=None, help="Override the generated experiment_name")
    p.add_argument("--out-dir", required=True, help="Output directory for the rendered config + basin lists + manifest")
    p.add_argument("--force", action="store_true", help="Allow writing into a non-empty --out-dir")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        bundle = generate_stage1_nh_config(
            policy_path=args.policy,
            package_root=args.package_root,
            splits_dir=args.splits_dir,
            lead_hours=args.lead_hours,
            seq_length=args.seq_length,
            static_column_manifest_path=args.static_column_manifest,
        )
    except NHConfigGenerationError as exc:
        _fail(str(exc))
        return 1

    try:
        paths = write_generated_config(
            bundle, args.out_dir, experiment_name=args.experiment_name, force=args.force
        )
    except NHConfigGenerationError as exc:
        _fail(str(exc))
        return 1

    print(json.dumps({
        "out_dir": str(Path(args.out_dir)),
        "config_yaml": str(paths["config.yaml"]),
        "generation_manifest": str(paths["generation_manifest.json"]),
        "basin_count": len(bundle.basin_ids),
        "lead_hours": bundle.lead_hours,
        "seq_length": bundle.seq_length,
        "target_variable": bundle.target_variable,
        "static_attribute_count": bundle.static_attribute_result.count,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

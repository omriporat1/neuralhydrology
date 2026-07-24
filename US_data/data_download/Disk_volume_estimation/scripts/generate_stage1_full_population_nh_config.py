#!/usr/bin/env python
"""Generate the full-population Stage 1 NH 1.13 readiness config bundle pair
(local implementation increment).

Reads the committed scientific baseline policy, the certified full non-CA
population package's (``stage1_scientific_package_v002``) manifest +
attributes.csv, and the canonical split-list files, then renders exactly one
(lead, seq_length) scientific configuration as TWO strictly separated config
bundles under --out-dir:

  <out-dir>/development/     train == validation == temporal_test, the 2,307
                             development_train basins (different date
                             periods only -- see docs/decision_log.md).
  <out-dir>/spatial_holdout/ test-only, the 250 spatial_holdout_nonca
                             basins; its own train/validation basin lists are
                             the *development* population (never a holdout
                             basin), so this bundle's config can never be
                             misused to train or validate on holdout basins.

Local-only: never touches h2o or Moriah, never transfers the package, never
writes a Slurm script, never configures W&B, never trains or evaluates a
model, and never generates any other (lead, seq_length) combination.

--package-root must point at wherever the certified package actually lives
for this run (e.g. a Moriah mount path); it is never hard-coded here.

Usage:
    python scripts/generate_stage1_full_population_nh_config.py \\
        --package-root /sci/labs/efratmorin/omripo/Flash-NH/data/stage1_scientific_package_v002 \\
        --lead-hours 6 --seq-length 24 \\
        --out-dir tmp/stage1_full_population_nh_config_lead06_seq24_v001
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline.nh_config_generation import (
    NHConfigGenerationError,
    generate_stage1_full_population_nh_config_bundles,
    write_generated_config,
)


def _fail(message: str) -> None:
    print(f"FATAL: {message}", file=sys.stderr)
    sys.exit(1)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--package-root", required=True,
                   help="Path to the certified stage1_scientific_package_v002 root "
                        "(full non-CA population, 2,557 basins)")
    p.add_argument("--policy", default=str(REPO_ROOT / "config" / "stage1_scientific_baseline_v001.yaml"),
                   help="Path to the Stage 1 scientific baseline policy YAML")
    p.add_argument("--splits-dir", default=str(REPO_ROOT / "config" / "stage1_baseline_splits_v001"),
                   help="Directory containing development_train.txt / spatial_holdout_nonca.txt / california_all.txt")
    p.add_argument("--static-column-manifest", default=None,
                   help="Optional external static column-role manifest JSON for an additional "
                        "independent re-derivation check (not required)")
    p.add_argument("--lead-hours", type=int, required=True, help="One of the policy-approved leads (hours)")
    p.add_argument("--seq-length", type=int, required=True, help="One of the policy-approved sequence lengths (hours)")
    p.add_argument("--development-experiment-name", default=None,
                   help="Override the development bundle's generated experiment_name")
    p.add_argument("--spatial-holdout-experiment-name", default=None,
                   help="Override the spatial-holdout bundle's generated experiment_name")
    p.add_argument("--out-dir", required=True,
                   help="Output directory; development/ and spatial_holdout/ subdirectories are written under it")
    p.add_argument("--force", action="store_true", help="Allow writing into non-empty output subdirectories")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        bundles = generate_stage1_full_population_nh_config_bundles(
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

    out_dir = Path(args.out_dir)
    try:
        development_paths = write_generated_config(
            bundles.development, out_dir / "development",
            experiment_name=args.development_experiment_name, force=args.force,
        )
        spatial_holdout_paths = write_generated_config(
            bundles.spatial_holdout, out_dir / "spatial_holdout",
            experiment_name=args.spatial_holdout_experiment_name, force=args.force,
        )
    except NHConfigGenerationError as exc:
        _fail(str(exc))
        return 1

    print(json.dumps({
        "out_dir": str(out_dir),
        "lead_hours": bundles.development.lead_hours,
        "seq_length": bundles.development.seq_length,
        "target_variable": bundles.development.target_variable,
        "static_attribute_count": bundles.development.static_attribute_result.count,
        "development": {
            "config_yaml": str(development_paths["config.yaml"]),
            "generation_manifest": str(development_paths["generation_manifest.json"]),
            "basin_count": len(bundles.development.basin_ids),
        },
        "spatial_holdout": {
            "config_yaml": str(spatial_holdout_paths["config.yaml"]),
            "generation_manifest": str(spatial_holdout_paths["generation_manifest.json"]),
            "basin_count": len(bundles.spatial_holdout.basin_ids),
            "train_basin_count": len(bundles.spatial_holdout.train_basin_ids),
            "validation_basin_count": len(bundles.spatial_holdout.validation_basin_ids),
            "test_basin_count": len(bundles.spatial_holdout.test_basin_ids),
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

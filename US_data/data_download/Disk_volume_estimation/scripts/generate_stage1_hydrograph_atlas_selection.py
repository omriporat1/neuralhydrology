#!/usr/bin/env python
"""Generate the Stage 1 hydrograph-atlas basin selection (Part C).

Deterministic selection of a small (default 24, range 18-30) diagnostic
basin subset drawn strictly from the canonical non-California
``development_train`` pool, stratified by validation-skill quartile x
area-class tercile x east/west geographic side. This script does NOT build
NeuralHydrology packages, does NOT train models, does NOT select or freeze a
checkpoint, and does NOT modify any frozen or canonical artifact -- it only
reads canonical inputs plus a supplied per-basin NSE table and writes new
files under --out-dir.

See docs/stage1_validation_optimization_foundation.md ("Part C") and
config/stage1_hydrograph_atlas_selection_v001.yaml for the full method.

Usage:
    python scripts/generate_stage1_hydrograph_atlas_selection.py \\
        --split-assignment config/stage1_baseline_splits_v001/split_assignment.csv \\
        --per-basin-nse reports/seed_validation_review_v001/per_basin/epoch009/epoch009_per_basin_metrics.csv \\
        --policy config/stage1_hydrograph_atlas_selection_v001.yaml \\
        --out-dir tmp/stage1_hydrograph_atlas_selection_v001
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline.compact_selection import select_universe
from src.baseline.hydrograph_atlas_selection import (
    SelectionError,
    build_hydrograph_atlas_selection,
    load_atlas_policy,
    load_per_basin_nse,
    sha256_of,
    write_selection_artifacts,
)
from src.baseline.compact_selection import load_split_assignment

_REQUIRED_ASSIGNMENT_COLUMNS = ["STAID", "split_role", "STATE", "HUC02", "area_class", "hydro_class"]


def _fail(message: str) -> None:
    print(f"FATAL: {message}", file=sys.stderr)
    sys.exit(1)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-assignment", required=True,
                   help="Path to the canonical split_assignment.csv (development_train role + others)")
    p.add_argument("--per-basin-nse", required=True,
                   help="Path to a per-basin metrics CSV with basin_id/nse columns "
                        "(e.g. scripts/dump_per_basin_table.py output)")
    p.add_argument("--policy", default=str(REPO_ROOT / "config" / "stage1_hydrograph_atlas_selection_v001.yaml"),
                   help="Path to the hydrograph-atlas selection policy YAML")
    p.add_argument("--out-dir", default=None,
                   help="Output directory for generated artifacts (required unless --dry-run)")
    p.add_argument("--force", action="store_true", help="Allow writing into a non-empty --out-dir")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate inputs and compute the selection but do not write any artifacts")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    argv_for_record = argv if argv is not None else sys.argv[1:]

    if not args.dry_run and not args.out_dir:
        _fail("--out-dir is required unless --dry-run is set")

    try:
        policy = load_atlas_policy(args.policy)
    except SelectionError as exc:
        _fail(f"policy load/validate failed: {exc}")

    input_provenance = {
        "policy_path": str(Path(args.policy)),
        "policy_sha256": sha256_of(args.policy),
    }

    try:
        assignment_df = load_split_assignment(args.split_assignment, _REQUIRED_ASSIGNMENT_COLUMNS)
        universe = select_universe(assignment_df, policy)
        nse_series = load_per_basin_nse(args.per_basin_nse)
    except SelectionError as exc:
        _fail(str(exc))

    input_provenance["split_assignment_path"] = str(Path(args.split_assignment))
    input_provenance["split_assignment_sha256"] = sha256_of(args.split_assignment)
    input_provenance["split_assignment_development_train_count"] = int(len(universe))
    input_provenance["per_basin_nse_path"] = str(Path(args.per_basin_nse))
    input_provenance["per_basin_nse_sha256"] = sha256_of(args.per_basin_nse)
    input_provenance["per_basin_nse_finite_count"] = int(len(nse_series))

    try:
        selection_df, manifest_pieces = build_hydrograph_atlas_selection(universe, nse_series, policy)
    except SelectionError as exc:
        _fail(f"selection failed: {exc}")

    if selection_df["gauge_id"].duplicated().any():
        _fail("internal error: duplicate gauge_id in final selection")
    if len(selection_df) != policy["target_count"]:
        _fail(
            f"internal error: selected {len(selection_df)} basins, expected "
            f"{policy['target_count']}"
        )
    dev_set = set(universe.index)
    leaked = set(selection_df["gauge_id"]) - dev_set
    if leaked:
        _fail(f"internal error: selected basin(s) outside development pool: {sorted(leaked)[:10]}")

    manifest = {
        "created_by": "scripts/generate_stage1_hydrograph_atlas_selection.py",
        "status": "candidate",
        **input_provenance,
        **manifest_pieces,
    }

    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "counts": manifest_pieces["counts"],
            "cell_sizes": manifest_pieces["cell_sizes"],
            "cell_quota": manifest_pieces["cell_quota"],
            "empty_cells": manifest_pieces["empty_cells"],
            "skill_quartile_edges": manifest_pieces["skill_quartile_edges"],
            "distinct_huc02_soft_minimum_met": manifest_pieces["distinct_huc02_soft_minimum_met"],
        }, indent=2, default=str))
        return 0

    out_dir = Path(args.out_dir)
    try:
        paths = write_selection_artifacts(out_dir, selection_df, manifest, force=args.force)
    except SelectionError as exc:
        _fail(str(exc))

    run_command_path = out_dir / "run_command.txt"
    run_command_path.write_text(
        "python scripts/generate_stage1_hydrograph_atlas_selection.py "
        + " ".join(argv_for_record) + "\n",
        encoding="utf-8",
    )
    paths["run_command.txt"] = run_command_path

    print(json.dumps({
        "out_dir": str(out_dir),
        "counts": manifest_pieces["counts"],
        "empty_cells": manifest_pieces["empty_cells"],
        "distinct_huc02_soft_minimum_met": manifest_pieces["distinct_huc02_soft_minimum_met"],
        "artifacts": {name: str(p) for name, p in sorted(paths.items())},
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

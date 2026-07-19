#!/usr/bin/env python
"""Generate the Stage 1 Compact Scientific Package basin selection.

Deterministic selection of a small (default 32, range 20-50) diagnostic
basin subset drawn strictly from the canonical non-California
``development_train`` pool. This script does NOT build NeuralHydrology
packages, does NOT touch FlashNHDataset/the NH launcher, does NOT train
models, and does NOT modify any frozen or canonical artifact -- it only
reads canonical inputs and writes new files under --out-dir.

See docs/stage1_compact_package_selection.md for the full method, output
schema, and the exact future h2o command to generate the real selection.

Usage (local, synthetic/small inputs):
    python scripts/generate_stage1_compact_package_selection.py \\
        --split-assignment tmp/synthetic_split_assignment.csv \\
        --out-dir tmp/stage1_compact_package_selection_v001 \\
        --dry-run

Usage (real run, once canonical inputs are available):
    python scripts/generate_stage1_compact_package_selection.py \\
        --split-assignment config/stage1_baseline_splits_v001/split_assignment.csv \\
        --attributes-parquet <path to stage1_static_attributes_v001.parquet> \\
        --column-manifest <path to stage1_static_attributes_v001_column_manifest.json> \\
        --qobs-status <path to audit/target_status.csv> \\
        --policy config/stage1_compact_package_selection_v001.yaml \\
        --out-dir tmp/stage1_compact_package_selection_v001
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline.compact_selection import (
    SelectionError,
    build_compact_selection,
    load_qobs_status,
    load_selection_policy,
    load_split_assignment,
    load_static_missingness,
    select_universe,
    sha256_of,
    write_selection_artifacts,
)
from src.baseline.splits import SplitGenerationError, load_matrix_for_splits

_REQUIRED_ASSIGNMENT_COLUMNS = ["STAID", "split_role", "STATE", "HUC02", "area_class", "hydro_class"]


def _fail(message: str) -> None:
    print(f"FATAL: {message}", file=sys.stderr)
    sys.exit(1)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-assignment", required=True,
                   help="Path to the canonical split_assignment.csv (development_train role + others)")
    p.add_argument("--policy", default=str(REPO_ROOT / "config" / "stage1_compact_package_selection_v001.yaml"),
                   help="Path to the compact-selection policy YAML")
    p.add_argument("--attributes-parquet", default=None,
                   help="Optional path to the checksum-verified static-attribute matrix parquet")
    p.add_argument("--column-manifest", default=None,
                   help="Optional path to the static-attribute column-role manifest JSON "
                        "(required if --attributes-parquet is given)")
    p.add_argument("--qobs-status", default=None,
                   help="Optional path to a qobs completeness/target-status CSV "
                        "(e.g. audit/target_status.csv)")
    p.add_argument("--gap-manifest", default=None,
                   help="Optional path to a forcing-gap timestamp manifest, recorded for provenance "
                        "only (dimension 7 -- gaps are global, never used to differentiate basins)")
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
    if args.column_manifest and not args.attributes_parquet:
        _fail("--column-manifest was given without --attributes-parquet")
    if args.attributes_parquet and not args.column_manifest:
        _fail("--attributes-parquet was given without --column-manifest")

    try:
        policy = load_selection_policy(args.policy)
    except SelectionError as exc:
        _fail(f"policy load/validate failed: {exc}")
    policy_sha256 = sha256_of(args.policy)

    input_provenance: dict = {
        "policy_path": str(Path(args.policy)),
        "policy_sha256": policy_sha256,
    }

    try:
        assignment_df = load_split_assignment(args.split_assignment, _REQUIRED_ASSIGNMENT_COLUMNS)
        universe = select_universe(assignment_df, policy)
    except SelectionError as exc:
        _fail(str(exc))
    input_provenance["split_assignment_path"] = str(Path(args.split_assignment))
    input_provenance["split_assignment_sha256"] = sha256_of(args.split_assignment)
    input_provenance["split_assignment_development_train_count"] = int(len(universe))

    if args.attributes_parquet:
        try:
            raw_cols = load_matrix_for_splits(args.attributes_parquet, ["DRAIN_SQKM", "ari_ix_uav"])
            missing_series = load_static_missingness(args.attributes_parquet, args.column_manifest, policy)
        except (SelectionError, SplitGenerationError) as exc:
            _fail(str(exc))
        universe = universe.join(raw_cols[["DRAIN_SQKM", "ari_ix_uav"]], how="left")
        universe = universe.join(missing_series, how="left")
        input_provenance["attributes_parquet_path"] = str(Path(args.attributes_parquet))
        input_provenance["attributes_parquet_sha256"] = sha256_of(args.attributes_parquet)
        input_provenance["column_manifest_path"] = str(Path(args.column_manifest))
        input_provenance["column_manifest_sha256"] = sha256_of(args.column_manifest)
        n_matrix_covered = int(universe["DRAIN_SQKM"].notna().sum())
        input_provenance["development_pool_basins_found_in_attributes_matrix"] = n_matrix_covered

    qobs_df = None
    if args.qobs_status:
        try:
            qobs_df = load_qobs_status(args.qobs_status, policy)
        except SelectionError as exc:
            _fail(str(exc))
        input_provenance["qobs_status_path"] = str(Path(args.qobs_status))
        input_provenance["qobs_status_sha256"] = sha256_of(args.qobs_status)
        n_qobs_covered = int(sum(s in qobs_df.index for s in universe.index))
        input_provenance["development_pool_basins_found_in_qobs_status"] = n_qobs_covered

    if args.gap_manifest:
        gap_path = Path(args.gap_manifest)
        if not gap_path.is_file():
            _fail(f"gap manifest not found: {gap_path}")
        try:
            gap_timestamps = json.loads(gap_path.read_text(encoding="utf-8"))
            gap_count = len(gap_timestamps) if isinstance(gap_timestamps, list) else None
        except json.JSONDecodeError as exc:
            _fail(f"gap manifest {gap_path} is not valid JSON: {exc}")
        input_provenance["gap_manifest_path"] = str(gap_path)
        input_provenance["gap_manifest_sha256"] = sha256_of(gap_path)
        input_provenance["gap_manifest_timestamp_count"] = gap_count
        input_provenance["gap_manifest_note"] = (
            "Recorded for provenance only. Archive-gap hours are global timeline "
            "positions, not a per-basin property; not used to differentiate basins "
            "in this selection."
        )

    try:
        selection_df, manifest_pieces = build_compact_selection(universe, qobs_df, policy)
    except SelectionError as exc:
        _fail(f"selection failed: {exc}")

    # ---- post-hoc acceptance checks (belt-and-braces on top of build_compact_selection) ----
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
        "created_by": "scripts/generate_stage1_compact_package_selection.py",
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
            "reserved_category_log": manifest_pieces["reserved_category_log"],
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
        "python scripts/generate_stage1_compact_package_selection.py "
        + " ".join(argv_for_record) + "\n",
        encoding="utf-8",
    )
    paths["run_command.txt"] = run_command_path

    print(json.dumps({
        "out_dir": str(out_dir),
        "counts": manifest_pieces["counts"],
        "distinct_huc02_soft_minimum_met": manifest_pieces["distinct_huc02_soft_minimum_met"],
        "artifacts": {name: str(p) for name, p in sorted(paths.items())},
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

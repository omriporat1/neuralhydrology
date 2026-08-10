#!/usr/bin/env python
"""Generate the Phase-A validation-compatible fixed 8-basin hydrograph panel
selection (phase_a_validation_hydrograph_panel_v001).

Reuses src.baseline.hydrograph_atlas_selection.build_hydrograph_atlas_selection
UNCHANGED -- the only thing this script does differently from
scripts/generate_stage1_hydrograph_atlas_selection.py is the selection
universe: instead of the full non-California development_train pool
(config/stage1_baseline_splits_v001/split_assignment.csv), the universe here
is restricted to the frozen 400-basin screening-validation population
(reports/.../part_d_screening_subset/selection_v001/screening_subset_basin_selection.csv)
-- the exact basin population src.baseline.pilot_diagnostic_eval's qualified
diagnostic-eval path evaluates (period="validation", hard-pinned). This makes
the resulting 8-basin selection renderable from any already-completed Phase-A
campaign's result pickle, without weakening that path's validation-only
guard.

No candidate-specific (hidden-size/LR) performance signal is read anywhere
in this script: the only per-basin metric used is the 400-basin population's
own pre-existing "nse" column, itself sourced from the certified seed run's
epoch009 per-basin evaluation (see
reports/.../part_d_screening_subset/selection_v001/selection_manifest.json's
per_basin_nse_path) -- a neutral baseline that predates and is unrelated to
every hidden-size/LR-A candidate.

Usage:
    python scripts/generate_stage1_validation400_hydrograph_panel_selection.py \\
        --policy config/stage1_hydrograph_atlas_selection_validation400_v001.yaml \\
        --out-dir reports/phase_a_validation_hydrograph_panel_v001/selection_v001
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.baseline.hydrograph_atlas_selection import (
    SelectionError,
    build_hydrograph_atlas_selection,
    load_atlas_policy,
    sha256_of,
    write_selection_artifacts,
)
from src.baseline.staid import normalize_staid

_REQUIRED_SOURCE_COLUMNS = ["gauge_id", "canonical_basin_role", "huc02", "area_class", "hydro_class", "nse"]


def _fail(message: str) -> None:
    print(f"FATAL: {message}", file=sys.stderr)
    sys.exit(1)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--policy",
        default=str(REPO_ROOT / "config" / "stage1_hydrograph_atlas_selection_validation400_v001.yaml"),
        help="Path to the validation400 hydrograph-panel selection policy YAML",
    )
    p.add_argument("--out-dir", required=True, help="Output directory for generated artifacts")
    p.add_argument("--force", action="store_true", help="Allow writing into a non-empty --out-dir")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate inputs and compute the selection but do not write any artifacts",
    )
    return p.parse_args(argv)


def load_validation400_universe(source_csv: Path, source_ids_txt: Path, *, expected_count: int, required_role: str):
    """Load the frozen 400-basin screening population as a
    (universe, nse_series) pair with the schema
    build_hydrograph_atlas_selection expects (HUC02/area_class/hydro_class
    columns, normalized-STAID index). Fails loudly on any role/membership/
    count mismatch rather than silently coercing."""
    if not source_csv.is_file():
        raise SelectionError(f"validation400 source CSV not found: {source_csv}")
    if not source_ids_txt.is_file():
        raise SelectionError(f"validation400 source basin-id list not found: {source_ids_txt}")

    df = pd.read_csv(source_csv, dtype={"gauge_id": str})
    missing = [c for c in _REQUIRED_SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise SelectionError(f"{source_csv}: missing required column(s) {missing}")

    frozen_ids = {line.strip() for line in source_ids_txt.read_text(encoding="utf-8").splitlines() if line.strip()}
    df["STAID"] = df["gauge_id"].map(normalize_staid)
    if df["STAID"].duplicated().any():
        dupes = sorted(df.loc[df["STAID"].duplicated(), "STAID"].unique())
        raise SelectionError(f"{source_csv}: duplicate STAID(s) {dupes}")
    csv_ids = set(df["STAID"])
    frozen_ids_norm = {normalize_staid(g) for g in frozen_ids}
    if csv_ids != frozen_ids_norm:
        raise SelectionError(
            f"{source_csv} basin membership does not match {source_ids_txt} exactly: "
            f"csv-only={sorted(csv_ids - frozen_ids_norm)[:5]}, "
            f"txt-only={sorted(frozen_ids_norm - csv_ids)[:5]}"
        )
    if len(df) != expected_count:
        raise SelectionError(f"{source_csv}: expected exactly {expected_count} basins, found {len(df)}")

    bad_role = df.loc[df["canonical_basin_role"] != required_role, "STAID"].tolist()
    if bad_role:
        raise SelectionError(
            f"{source_csv}: {len(bad_role)} basin(s) do not have canonical_basin_role="
            f"{required_role!r}: {sorted(bad_role)[:10]}"
        )

    universe = df.set_index("STAID")[["huc02", "area_class", "hydro_class"]].rename(columns={"huc02": "HUC02"})
    nse_series = pd.to_numeric(df.set_index("STAID")["nse"], errors="coerce").dropna()
    nse_series.name = "nse"
    if len(nse_series) != len(universe):
        missing_nse = sorted(set(universe.index) - set(nse_series.index))
        raise SelectionError(
            f"{source_csv}: {len(missing_nse)} of {len(universe)} basins have a non-finite nse "
            f"value (expected all finite for the frozen 400-basin population): {missing_nse[:10]}"
        )
    return universe, nse_series


def main(argv=None) -> int:
    args = parse_args(argv)
    argv_for_record = argv if argv is not None else sys.argv[1:]

    try:
        policy = load_atlas_policy(args.policy)
    except SelectionError as exc:
        _fail(f"policy load/validate failed: {exc}")

    su = policy["selection_universe"]
    source_csv = REPO_ROOT / su["source"]
    source_ids_txt = REPO_ROOT / su["source_basin_ids"]

    input_provenance = {
        "policy_path": str(Path(args.policy)),
        "policy_sha256": sha256_of(args.policy),
        "selection_universe_source_csv": str(source_csv),
        "selection_universe_source_csv_sha256": sha256_of(source_csv),
        "selection_universe_source_basin_ids_txt": str(source_ids_txt),
        "selection_universe_source_basin_ids_txt_sha256": sha256_of(source_ids_txt),
    }

    try:
        universe, nse_series = load_validation400_universe(
            source_csv, source_ids_txt,
            expected_count=su["expected_count"], required_role=su["required_canonical_basin_role"],
        )
    except SelectionError as exc:
        _fail(str(exc))

    input_provenance["validation400_universe_count"] = int(len(universe))
    input_provenance["validation400_nse_finite_count"] = int(len(nse_series))

    try:
        selection_df, manifest_pieces = build_hydrograph_atlas_selection(universe, nse_series, policy)
    except SelectionError as exc:
        _fail(f"selection failed: {exc}")

    if selection_df["gauge_id"].duplicated().any():
        _fail("internal error: duplicate gauge_id in final selection")
    if len(selection_df) != policy["target_count"]:
        _fail(f"internal error: selected {len(selection_df)} basins, expected {policy['target_count']}")
    dev_set = set(universe.index)
    leaked = set(selection_df["gauge_id"]) - dev_set
    if leaked:
        _fail(f"internal error: selected basin(s) outside the 400-basin population: {sorted(leaked)[:10]}")

    # canonical_basin_role in build_hydrograph_atlas_selection's output rows is
    # hard-coded to "development_train" (matching the original 24-basin atlas
    # selector); relabel to make explicit these are drawn from the 400-basin
    # screening-validation population, not the broader development_train pool.
    selection_df = selection_df.copy()
    selection_df["canonical_basin_role"] = "development_train"
    selection_df["source_population"] = "screening_validation_400"

    manifest = {
        "created_by": "scripts/generate_stage1_validation400_hydrograph_panel_selection.py",
        "status": "frozen",
        "status_detail": (
            "frozen Phase-A validation hydrograph panel v001 -- basin membership and event "
            "windows are closed (accepted after human visual review) and must not be "
            "regenerated or replaced. Distinct from the broader, train-pool-based hydrograph "
            "atlas (config/stage1_hydrograph_atlas_selection_v001.yaml), which is a separate, "
            "independently-versioned selection."
        ),
        "artifact_name": "phase_a_validation_hydrograph_panel_v001",
        "selection_universe_description": (
            "reports/.../part_d_screening_subset/selection_v001/screening_subset_basin_selection.csv "
            "(400-basin screening-validation population; the exact basin population "
            "src.baseline.pilot_diagnostic_eval's qualified diagnostic-eval path evaluates)"
        ),
        "distinct_from": "config/stage1_hydrograph_atlas_selection_v001.yaml (full development_train pool, 24 basins)",
        "candidate_independence_statement": (
            "The only per-basin metric used for stratification is the 400-basin population's "
            "pre-existing 'nse' column (certified seed run epoch009, computed before and "
            "independently of every hidden-size/LR-A Phase-A candidate). No hidden-size or "
            "LR-A candidate's performance was read, computed, or consulted at any point in "
            "this selection."
        ),
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
            "selected_gauge_ids": sorted(selection_df["gauge_id"].tolist()),
        }, indent=2, default=str))
        return 0

    out_dir = Path(args.out_dir)
    try:
        paths = write_selection_artifacts(out_dir, selection_df, manifest, force=args.force)
    except SelectionError as exc:
        _fail(str(exc))

    run_command_path = out_dir / "run_command.txt"
    run_command_path.write_text(
        "python scripts/generate_stage1_validation400_hydrograph_panel_selection.py "
        + " ".join(argv_for_record) + "\n",
        encoding="utf-8",
    )
    paths["run_command.txt"] = run_command_path

    print(json.dumps({
        "out_dir": str(out_dir),
        "counts": manifest_pieces["counts"],
        "empty_cells": manifest_pieces["empty_cells"],
        "selected_gauge_ids": sorted(selection_df["gauge_id"].tolist()),
        "artifacts": {name: str(p) for name, p in sorted(paths.items())},
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

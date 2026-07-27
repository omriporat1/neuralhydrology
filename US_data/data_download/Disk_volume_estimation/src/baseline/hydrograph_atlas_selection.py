"""Stage 1 hydrograph-atlas basin selection (Part C, section 7.1/7.2 of
docs/stage1_validation_optimization_foundation.md).

Small, deterministic, testable functions, mirroring the style of
:mod:`src.baseline.compact_selection` and reusing its generic building blocks
directly (universe loading, macro-region mapping, cell-quota apportionment,
seeded within-cell sampling) rather than re-implementing them. The one new
piece this module adds is the *skill_stratum* dimension: a validation-NSE
quartile classification computed fresh from a supplied per-basin metrics
table, in the same spirit as :func:`src.baseline.splits.compute_tercile_edges`
but with 3 edges / 4 classes instead of 2 edges / 3 classes.

Scope (see config/stage1_hydrograph_atlas_selection_v001.yaml and
docs/stage1_validation_optimization_foundation.md, "Part C"): select a small
(default 24, range 18-30) diagnostic basin subset strictly from the canonical
non-California ``development_train`` pool
(config/stage1_baseline_splits_v001/split_assignment.csv), stratified by
skill_stratum x area_class x geo_side (east/west). This module does not
build NeuralHydrology packages, does not train models, and does not select or
freeze any checkpoint -- the supplied per-basin NSE table is diagnostic-only
evidence from one already-completed epoch of the certified seed run.

Method summary:
  1. Load the canonical split-assignment CSV and filter to
     ``development_train`` with fail-fast checks (no California, no holdout
     leakage, no duplicate normalized gauge IDs) -- reuses
     :func:`src.baseline.compact_selection.select_universe` unchanged.
  2. Compute skill-stratum quartile edges (p25/p50/p75) from the supplied
     per-basin raw-space NSE series, restricted to the selection universe,
     and classify every universe basin into one of four labels.
  3. Classify every universe basin's HUC02 into "east"/"west" via the
     policy's explicit macro-region mapping (reused unchanged from
     :mod:`src.baseline.compact_selection`); a basin resolving to "other" is
     excluded from the grid (hard requirement, not a soft check -- unlike the
     compact-package selector, geo_side is a required grid axis here).
  4. Stratify into a skill_stratum x area_class grid (area_class reused
     unchanged from the canonical split, like the compact-package selector's
     area_class x hydro_class grid), further split east/west, apportion the
     target quota across non-empty cells (largest-remainder method, floor of
     one per non-empty cell where the budget allows), and sample within each
     cell with an independent seeded RNG using HUC02 round-robin order.
  5. No cherry-picking (section 7.2): only validation NSE and canonical
     static metadata are used; no visual preference, no test-set/temporal-
     test/spatial-holdout signal, no manual "interesting basin" choice enters
     this module.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .compact_selection import (
    SelectionError,
    allocate_cell_quota,
    build_macro_region_map,
    macro_region_for_huc02,
    macro_region_side,
    make_cell_rngs,
    select_within_cell,
    sha256_of,
)
from .staid import normalize_staid

__all__ = [
    "SelectionError",
    "sha256_of",
    "load_atlas_policy",
    "load_per_basin_nse",
    "compute_skill_quartile_edges",
    "assign_skill_stratum",
    "classify_geo_side",
    "build_hydrograph_atlas_selection",
    "write_selection_artifacts",
]

_ALGORITHM_ID = "stage1_hydrograph_atlas_skill_area_geo_quota_selection_v1"
_ALGORITHM_VERSION = 1

_REQUIRED_TOP_LEVEL_KEYS = [
    "selection_name", "algorithm_id", "algorithm_version", "seed",
    "target_count", "min_target_count", "max_target_count",
    "selection_universe", "stratification", "skill_stratum", "geography",
]

_SKILL_LABELS_IN_QUARTILE_ORDER = ["severe_failure_lower_tail", "weak", "typical", "strong"]


# ---------------------------------------------------------------------------
# Policy loading (mirrors compact_selection.load_selection_policy, but with
# this module's own required-key schema -- deliberately does not force the
# compact-package's qobs_completeness/static_missingness/reserved_categories
# keys, which this simpler algorithm does not use).
# ---------------------------------------------------------------------------

def load_atlas_policy(path) -> dict:
    import yaml

    p = Path(path)
    if not p.is_file():
        raise SelectionError(f"hydrograph-atlas selection policy file not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise SelectionError(f"hydrograph-atlas selection policy {p} did not parse to a mapping")
    missing = [k for k in _REQUIRED_TOP_LEVEL_KEYS if k not in data]
    if missing:
        raise SelectionError(f"hydrograph-atlas selection policy {p} missing required key(s): {missing}")
    if not (data["min_target_count"] <= data["target_count"] <= data["max_target_count"]):
        raise SelectionError(
            f"target_count={data['target_count']} outside "
            f"[{data['min_target_count']}, {data['max_target_count']}]"
        )
    return data


# ---------------------------------------------------------------------------
# Per-basin NSE loading
# ---------------------------------------------------------------------------

def load_per_basin_nse(path, staid_column: str = "basin_id", nse_column: str = "nse") -> pd.Series:
    """Load a per-basin metrics CSV (e.g. scripts/dump_per_basin_table.py
    output) and return a Series of raw-space NSE indexed by normalized STAID.

    Rows with a non-finite (NaN) NSE value are dropped (a basin excluded from
    validation cannot be skill-stratified); this is recorded by the caller
    via the returned series' length, not raised as an error here.
    """
    p = Path(path)
    if not p.is_file():
        raise SelectionError(f"per-basin NSE table not found: {p}")
    df = pd.read_csv(p, dtype={staid_column: str})
    if staid_column not in df.columns or nse_column not in df.columns:
        raise SelectionError(
            f"per-basin NSE table {p} missing required column(s): "
            f"{staid_column!r} and/or {nse_column!r}"
        )
    try:
        df["STAID"] = df[staid_column].map(normalize_staid)
    except (TypeError, ValueError) as exc:
        raise SelectionError(f"malformed STAID in {p}: {exc}") from exc
    if df["STAID"].duplicated().any():
        dupes = sorted(df.loc[df["STAID"].duplicated(), "STAID"].unique())
        raise SelectionError(f"duplicate STAID(s) in per-basin NSE table {p}: {dupes[:10]}")
    series = df.set_index("STAID")[nse_column]
    series = pd.to_numeric(series, errors="coerce")
    series = series.dropna()
    series.name = "nse"
    return series


# ---------------------------------------------------------------------------
# Skill-stratum classification
# ---------------------------------------------------------------------------

def compute_skill_quartile_edges(values: pd.Series) -> tuple[float, float, float]:
    """Edges (p25, p50, p75) via numpy linear-interpolation quantiles.

    Mirrors src.baseline.splits.compute_tercile_edges's method (np.quantile,
    method="linear") generalized from 2 edges/3 classes to 3 edges/4 classes.
    """
    if values.isnull().any():
        raise SelectionError("compute_skill_quartile_edges received null values")
    v = values.to_numpy(dtype=float)
    if v.size == 0:
        raise SelectionError("compute_skill_quartile_edges received an empty population")
    edges = np.quantile(v, [0.25, 0.5, 0.75], method="linear")
    return float(edges[0]), float(edges[1]), float(edges[2])


def assign_skill_stratum(values: pd.Series, edges: tuple[float, float, float]) -> pd.Series:
    """severe_failure_lower_tail: v < p25; weak: p25 <= v < p50;
    typical: p50 <= v < p75; strong: v >= p75.

    If two or more edges coincide (a mass point covers >= 25% of the
    population), one or more middle classes become structurally empty --
    a known, accepted degeneracy, not an error (mirrors
    src.baseline.splits.assign_tercile_class's documented behavior).
    """
    if values.isnull().any():
        raise SelectionError("assign_skill_stratum received null values")
    p25, p50, p75 = edges
    v = values.to_numpy(dtype=float)
    out = np.where(
        v < p25, "severe_failure_lower_tail",
        np.where(v < p50, "weak", np.where(v < p75, "typical", "strong")),
    )
    return pd.Series(out, index=values.index)


# ---------------------------------------------------------------------------
# Geographic side classification (hard grid dimension here, unlike the
# compact-package selector's soft east/west breadth check)
# ---------------------------------------------------------------------------

def classify_geo_side(huc02_series: pd.Series, policy: dict) -> pd.Series:
    region_map = build_macro_region_map(policy)
    sides = []
    for huc02 in huc02_series:
        region = macro_region_for_huc02(huc02, region_map)
        sides.append(macro_region_side(region, policy))
    return pd.Series(sides, index=huc02_series.index, name="geo_side")


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def build_hydrograph_atlas_selection(
    universe: pd.DataFrame,
    nse_series: pd.Series,
    policy: dict,
) -> tuple[pd.DataFrame, dict]:
    """Assign the hydrograph-atlas selection.

    universe: output of compact_selection.select_universe (indexed by
    normalized STAID, columns STATE/HUC02/area_class/hydro_class).
    nse_series: output of load_per_basin_nse, indexed by normalized STAID.
    """
    target_count = policy["target_count"]
    seed = policy["seed"]
    area_valid = policy["stratification"]["area_class_valid_values"]
    min_per_cell = policy["stratification"]["min_quota_per_nonempty_cell"]

    if len(universe) < target_count:
        raise SelectionError(
            f"development pool has only {len(universe)} basin(s), cannot select "
            f"target_count={target_count}"
        )

    common = universe.index.intersection(nse_series.index)
    missing_nse = set(universe.index) - set(nse_series.index)
    pool = universe.loc[sorted(common)].copy()
    pool["nse"] = nse_series.loc[sorted(common)]

    if len(pool) < target_count:
        raise SelectionError(
            f"only {len(pool)} universe basin(s) have a finite per-basin NSE value "
            f"(missing for {len(missing_nse)}), cannot select target_count={target_count}"
        )

    edges = compute_skill_quartile_edges(pool["nse"])
    pool["skill_stratum"] = assign_skill_stratum(pool["nse"], edges).values
    pool["geo_side"] = classify_geo_side(pool["HUC02"], policy).values

    area_gap_mask = ~pool["area_class"].isin(area_valid)
    other_side_mask = ~pool["geo_side"].isin(["east", "west"])
    excluded_from_grid = set(pool.index[area_gap_mask]) | set(pool.index[other_side_mask])
    grid_pool = pool.drop(index=sorted(excluded_from_grid))

    cell_sizes = {}
    for skill in _SKILL_LABELS_IN_QUARTILE_ORDER:
        for area in area_valid:
            for side in ("east", "west"):
                n = int((
                    (grid_pool["skill_stratum"] == skill)
                    & (grid_pool["area_class"] == area)
                    & (grid_pool["geo_side"] == side)
                ).sum())
                if n > 0:
                    cell_sizes[(skill, area, side)] = n

    if not cell_sizes:
        raise SelectionError("hydrograph-atlas grid pool has zero eligible cells")

    empty_cells = [
        (skill, area, side)
        for skill in _SKILL_LABELS_IN_QUARTILE_ORDER
        for area in area_valid
        for side in ("east", "west")
        if (skill, area, side) not in cell_sizes
    ]

    quota_by_cell, quota_repair_log = allocate_cell_quota(cell_sizes, target_count, min_per_cell)
    rngs = make_cell_rngs(seed, list(quota_by_cell.keys()))

    picks: list[str] = []
    reason_by_staid: dict[str, str] = {}
    for cell_key in sorted(quota_by_cell.keys()):
        skill, area, side = cell_key
        q = quota_by_cell[cell_key]
        if q <= 0:
            continue
        cell_df = grid_pool.loc[
            (grid_pool["skill_stratum"] == skill)
            & (grid_pool["area_class"] == area)
            & (grid_pool["geo_side"] == side)
        ]
        cell_picks = select_within_cell(cell_df, q, rngs[cell_key])
        for s in cell_picks:
            picks.append(s)
            reason_by_staid[s] = f"skill={skill};area={area};geo_side={side}"

    all_ids = sorted(set(picks))
    if len(all_ids) != target_count:
        raise SelectionError(
            f"internal selection error: assembled {len(all_ids)} basin(s), expected {target_count}"
        )

    region_map = build_macro_region_map(policy)
    rows = []
    for staid in all_ids:
        urow = pool.loc[staid]
        macro_region = macro_region_for_huc02(urow["HUC02"], region_map)
        rows.append({
            "gauge_id": staid,
            "canonical_basin_role": "development_train",
            "huc02": urow["HUC02"],
            "macro_region": macro_region,
            "geo_side": urow["geo_side"],
            "area_class": urow["area_class"],
            "hydro_class": urow["hydro_class"],
            "nse": float(urow["nse"]),
            "skill_stratum": urow["skill_stratum"],
            "selection_reason": reason_by_staid[staid],
        })
    selection_df = pd.DataFrame(rows).sort_values("gauge_id").reset_index(drop=True)

    distinct_huc02 = int(selection_df["huc02"].nunique())
    soft_min = policy["geography"]["distinct_huc02_soft_minimum"]
    n_east = int((selection_df["geo_side"] == "east").sum())
    n_west = int((selection_df["geo_side"] == "west").sum())
    if n_east == 0 or n_west == 0:
        raise SelectionError(
            f"hydrograph-atlas selection lacks required east/west CONUS breadth: "
            f"n_east={n_east}, n_west={n_west}"
        )

    manifest_pieces = {
        "algorithm_id": _ALGORITHM_ID,
        "algorithm_version": _ALGORITHM_VERSION,
        "seed": seed,
        "target_count": target_count,
        "universe_size": int(len(universe)),
        "universe_basins_missing_nse": sorted(missing_nse),
        "skill_quartile_edges": {"p25": edges[0], "p50": edges[1], "p75": edges[2]},
        "cell_sizes": {f"{s}:{a}:{g}": n for (s, a, g), n in sorted(cell_sizes.items())},
        "cell_quota": {f"{s}:{a}:{g}": n for (s, a, g), n in sorted(quota_by_cell.items())},
        "cell_quota_repair_log": quota_repair_log,
        "empty_cells": [f"{s}:{a}:{g}" for (s, a, g) in empty_cells],
        "counts": {
            "n_selected": int(len(selection_df)),
            "distinct_huc02": distinct_huc02,
        },
        "huc02_counts": {str(k): int(v) for k, v in sorted(selection_df["huc02"].value_counts().items())},
        "area_class_counts": {
            str(k): int(v) for k, v in sorted(selection_df["area_class"].value_counts().items())
        },
        "skill_stratum_counts": {
            str(k): int(v) for k, v in sorted(selection_df["skill_stratum"].value_counts().items())
        },
        "geo_side_counts": {
            str(k): int(v) for k, v in sorted(selection_df["geo_side"].value_counts().items())
        },
        "macro_region_counts": {
            str(k): int(v) for k, v in sorted(selection_df["macro_region"].value_counts().items())
        },
        "distinct_huc02_soft_minimum": soft_min,
        "distinct_huc02_soft_minimum_met": distinct_huc02 >= soft_min,
        "east_west_breadth": {"n_east": n_east, "n_west": n_west},
    }
    return selection_df, manifest_pieces


# ---------------------------------------------------------------------------
# Artifact writing (mirrors compact_selection.write_selection_artifacts)
# ---------------------------------------------------------------------------

def _render_summary_md(manifest: dict) -> str:
    lines = [
        "# Stage 1 Hydrograph-Atlas Basin Selection Summary",
        "",
        f"- status: {manifest.get('status', 'unknown')}",
        f"- algorithm: {manifest['algorithm_id']} v{manifest['algorithm_version']}",
        f"- seed: {manifest['seed']}",
        f"- target_count: {manifest['target_count']}",
        f"- development pool size (input): {manifest['universe_size']}",
        f"- universe basins missing a per-basin NSE value: {len(manifest['universe_basins_missing_nse'])}",
        f"- basins selected: {manifest['counts']['n_selected']}",
        f"- skill quartile edges: p25={manifest['skill_quartile_edges']['p25']:.4f}, "
        f"p50={manifest['skill_quartile_edges']['p50']:.4f}, "
        f"p75={manifest['skill_quartile_edges']['p75']:.4f}",
        f"- distinct HUC02 regions covered: {manifest['counts']['distinct_huc02']} "
        f"(soft minimum {manifest['distinct_huc02_soft_minimum']}, "
        f"{'met' if manifest['distinct_huc02_soft_minimum_met'] else 'NOT MET -- advisory only'})",
        f"- east/west CONUS breadth: n_east={manifest['east_west_breadth']['n_east']}, "
        f"n_west={manifest['east_west_breadth']['n_west']} (required and met)",
        f"- empty grid cells (of 24): {len(manifest['empty_cells'])} "
        f"({', '.join(manifest['empty_cells']) if manifest['empty_cells'] else 'none'})",
        "",
    ]
    for title, key in [
        ("skill_stratum representation", "skill_stratum_counts"),
        ("area_class representation", "area_class_counts"),
        ("geo_side representation", "geo_side_counts"),
        ("HUC02 breadth", "huc02_counts"),
        ("macro-region representation", "macro_region_counts"),
    ]:
        lines += ["", f"## {title}", "", "| key | n |", "|---|---|"]
        for k, v in sorted(manifest[key].items()):
            lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## No-cherry-picking statement",
        "",
        "This selection uses only: (1) the canonical split assignment's "
        "area_class/HUC02/STATE fields, (2) validation-only raw-space NSE "
        "from one already-completed seed-run epoch, and (3) a fixed seed "
        "for within-cell sampling. No visual inspection, no prediction-error "
        "signal beyond the NSE value itself, no test-set/temporal-test/"
        "spatial-holdout behavior, and no manual basin choice entered this "
        "selection (docs/stage1_validation_optimization_foundation.md, "
        "Part C section 7.2).",
        "",
    ]
    return "\n".join(lines) + "\n"


def write_selection_artifacts(out_dir, selection_df: pd.DataFrame, manifest: dict, force: bool = False) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise SelectionError(
            f"output directory already exists and is non-empty: {out_dir} (use --force)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    csv_path = out_dir / "hydrograph_atlas_basin_selection.csv"
    selection_df.to_csv(csv_path, index=False)
    paths["hydrograph_atlas_basin_selection.csv"] = csv_path

    ids_path = out_dir / "hydrograph_atlas_basin_ids.txt"
    ids_path.write_text("\n".join(selection_df["gauge_id"].tolist()) + "\n", encoding="utf-8")
    paths["hydrograph_atlas_basin_ids.txt"] = ids_path

    summary_md_path = out_dir / "selection_summary.md"
    summary_md_path.write_text(_render_summary_md(manifest), encoding="utf-8")
    paths["selection_summary.md"] = summary_md_path

    summary_json_path = out_dir / "selection_summary.json"
    summary_json_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    paths["selection_summary.json"] = summary_json_path

    artifact_sha256 = {name: sha256_of(p) for name, p in sorted(paths.items())}
    manifest_to_write = {**manifest, "artifact_sha256": artifact_sha256}
    manifest_path = out_dir / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest_to_write, indent=2, default=str), encoding="utf-8")
    paths["selection_manifest.json"] = manifest_path

    return paths

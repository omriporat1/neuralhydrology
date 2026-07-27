"""Stage 1 frozen validation-screening subset selection (Part D, section 8 of
docs/stage1_validation_optimization_foundation.md).

Small, deterministic, testable functions, mirroring the style of
:mod:`src.baseline.hydrograph_atlas_selection` and reusing generic building
blocks directly rather than re-implementing them:
  - universe loading/filtering: :func:`src.baseline.compact_selection.select_universe`
  - macro-region mapping: :func:`src.baseline.compact_selection.build_macro_region_map`
    / :func:`macro_region_for_huc02`
  - cell-quota apportionment: :func:`src.baseline.compact_selection.allocate_cell_quota`
  - seeded within-cell sampling: :func:`src.baseline.compact_selection.make_cell_rngs`
    / :func:`select_within_cell`
  - flow-variability tercile: :func:`src.baseline.splits.compute_tercile_edges`
    / :func:`assign_tercile_class` (same method already used for area_class /
    hydro_class in the canonical split, applied here to a new covariate)
  - skill-stratum quartile: :func:`src.baseline.hydrograph_atlas_selection.
    compute_skill_quartile_edges` / :func:`assign_skill_stratum` (unchanged
    from Part C)

Scope (see config/stage1_screening_subset_v001.yaml and
docs/stage1_validation_optimization_foundation.md, "Part D"): select a fixed,
seeded subset (target ~400, range 300-500) of the canonical non-California
``development_train`` pool, proportionally stratified by a single composite
cell key -- macro_region x area_class x hydro_class x flow_var_class x
skill_stratum -- via largest-remainder apportionment with
``min_quota_per_nonempty_cell: 0`` (i.e. a pure proportional allocation, not
a "one guaranteed basin per cell" rule). This subset is for fast development-
validation feedback only; per section 8.4 the full 2,307-basin development
population remains authoritative for final run/checkpoint selection.
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
    make_cell_rngs,
    select_within_cell,
    sha256_of,
)
from .hydrograph_atlas_selection import (
    assign_skill_stratum,
    compute_skill_quartile_edges,
    load_per_basin_nse,
)
from .splits import assign_tercile_class, compute_tercile_edges
from .staid import normalize_staid

__all__ = [
    "SelectionError",
    "sha256_of",
    "load_per_basin_nse",
    "load_screening_subset_policy",
    "load_flow_variability",
    "build_screening_subset_selection",
    "write_selection_artifacts",
]

_ALGORITHM_ID = "stage1_screening_subset_proportional_composite_stratum_selection_v1"
_ALGORITHM_VERSION = 1

_REQUIRED_TOP_LEVEL_KEYS = [
    "selection_name", "algorithm_id", "algorithm_version", "seed",
    "target_count", "min_target_count", "max_target_count",
    "selection_universe", "stratification", "flow_variability",
    "skill_stratum", "geography",
]

_SKILL_LABELS_IN_QUARTILE_ORDER = ["severe_failure_lower_tail", "weak", "typical", "strong"]
_FLOW_VAR_LABELS_IN_TERCILE_ORDER = ["low", "middle", "high"]


# ---------------------------------------------------------------------------
# Policy loading (mirrors hydrograph_atlas_selection.load_atlas_policy)
# ---------------------------------------------------------------------------

def load_screening_subset_policy(path) -> dict:
    import yaml

    p = Path(path)
    if not p.is_file():
        raise SelectionError(f"screening-subset selection policy file not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise SelectionError(f"screening-subset selection policy {p} did not parse to a mapping")
    missing = [k for k in _REQUIRED_TOP_LEVEL_KEYS if k not in data]
    if missing:
        raise SelectionError(f"screening-subset selection policy {p} missing required key(s): {missing}")
    if not (data["min_target_count"] <= data["target_count"] <= data["max_target_count"]):
        raise SelectionError(
            f"target_count={data['target_count']} outside "
            f"[{data['min_target_count']}, {data['max_target_count']}]"
        )
    return data


# ---------------------------------------------------------------------------
# Flow-variability (flashiness) loading
# ---------------------------------------------------------------------------

def load_flow_variability(path, staid_column: str = "STAID", value_column: str = "rbi") -> pd.Series:
    """Load an observed flow-variability table (e.g. the WY2024 RBI screening
    results CSV) and return a Series indexed by normalized STAID.

    Rows with a non-finite value are dropped (a basin without an observed
    flashiness value cannot be flow-variability-stratified); this is recorded
    by the caller via the returned series' length, not raised as an error
    here.
    """
    p = Path(path)
    if not p.is_file():
        raise SelectionError(f"flow-variability table not found: {p}")
    df = pd.read_csv(p, dtype={staid_column: str})
    if staid_column not in df.columns or value_column not in df.columns:
        raise SelectionError(
            f"flow-variability table {p} missing required column(s): "
            f"{staid_column!r} and/or {value_column!r}"
        )
    try:
        df["STAID"] = df[staid_column].map(normalize_staid)
    except (TypeError, ValueError) as exc:
        raise SelectionError(f"malformed STAID in {p}: {exc}") from exc
    if df["STAID"].duplicated().any():
        dupes = sorted(df.loc[df["STAID"].duplicated(), "STAID"].unique())
        raise SelectionError(f"duplicate STAID(s) in flow-variability table {p}: {dupes[:10]}")
    series = df.set_index("STAID")[value_column]
    series = pd.to_numeric(series, errors="coerce")
    series = series.dropna()
    series.name = "rbi"
    return series


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def build_screening_subset_selection(
    universe: pd.DataFrame,
    nse_series: pd.Series,
    flow_var_series: pd.Series,
    policy: dict,
) -> tuple[pd.DataFrame, dict]:
    """Assign the frozen validation-screening subset.

    universe: output of compact_selection.select_universe (indexed by
    normalized STAID, columns STATE/HUC02/area_class/hydro_class).
    nse_series: output of load_per_basin_nse, indexed by normalized STAID.
    flow_var_series: output of load_flow_variability, indexed by normalized
    STAID.
    """
    target_count = policy["target_count"]
    seed = policy["seed"]
    strat = policy["stratification"]
    area_valid = strat["area_class_valid_values"]
    hydro_valid = strat["hydro_class_valid_values"]
    flow_var_valid = strat["flow_var_class_valid_values"]
    min_per_cell = strat["min_quota_per_nonempty_cell"]

    if len(universe) < target_count:
        raise SelectionError(
            f"development pool has only {len(universe)} basin(s), cannot select "
            f"target_count={target_count}"
        )

    common = universe.index.intersection(nse_series.index).intersection(flow_var_series.index)
    missing_nse = set(universe.index) - set(nse_series.index)
    missing_flow_var = set(universe.index) - set(flow_var_series.index)
    pool = universe.loc[sorted(common)].copy()
    pool["nse"] = nse_series.loc[sorted(common)]
    pool["rbi"] = flow_var_series.loc[sorted(common)]

    if len(pool) < target_count:
        raise SelectionError(
            f"only {len(pool)} universe basin(s) have both a finite per-basin NSE "
            f"and a finite flow-variability value (missing NSE for "
            f"{len(missing_nse)}, missing flow-variability for "
            f"{len(missing_flow_var)}), cannot select target_count={target_count}"
        )

    skill_edges = compute_skill_quartile_edges(pool["nse"])
    pool["skill_stratum"] = assign_skill_stratum(pool["nse"], skill_edges).values

    flow_var_edges = compute_tercile_edges(pool["rbi"])
    pool["flow_var_class"] = assign_tercile_class(pool["rbi"], flow_var_edges).values

    region_map = build_macro_region_map(policy)
    # macro_region_for_huc02 raises SelectionError for any HUC02 not covered
    # by the policy's explicit mapping -- so a resolved value is always one
    # of the named macro regions, never a fallback "other" bucket.
    pool["macro_region"] = [macro_region_for_huc02(h, region_map) for h in pool["HUC02"]]

    area_ok = pool["area_class"].isin(area_valid)
    hydro_ok = pool["hydro_class"].isin(hydro_valid)
    flow_var_ok = pool["flow_var_class"].isin(flow_var_valid)
    excluded_from_grid = set(pool.index[~(area_ok & hydro_ok & flow_var_ok)])
    grid_pool = pool.drop(index=sorted(excluded_from_grid))

    if len(grid_pool) < target_count:
        raise SelectionError(
            f"only {len(grid_pool)} pool basin(s) remain after grid-validity "
            f"exclusions (excluded {len(excluded_from_grid)}), cannot select "
            f"target_count={target_count}"
        )

    cell_sizes: dict[tuple, int] = {}
    grouped = grid_pool.groupby(
        ["macro_region", "area_class", "hydro_class", "flow_var_class", "skill_stratum"]
    ).size()
    for key, n in grouped.items():
        if n > 0:
            cell_sizes[key] = int(n)

    if not cell_sizes:
        raise SelectionError("screening-subset grid pool has zero eligible cells")

    quota_by_cell, quota_repair_log = allocate_cell_quota(cell_sizes, target_count, min_per_cell)
    rngs = make_cell_rngs(seed, list(quota_by_cell.keys()))

    picks: list[str] = []
    reason_by_staid: dict[str, str] = {}
    for cell_key in sorted(quota_by_cell.keys()):
        macro_region, area, hydro, flow_var, skill = cell_key
        q = quota_by_cell[cell_key]
        if q <= 0:
            continue
        cell_df = grid_pool.loc[
            (grid_pool["macro_region"] == macro_region)
            & (grid_pool["area_class"] == area)
            & (grid_pool["hydro_class"] == hydro)
            & (grid_pool["flow_var_class"] == flow_var)
            & (grid_pool["skill_stratum"] == skill)
        ]
        cell_picks = select_within_cell(cell_df, q, rngs[cell_key])
        for s in cell_picks:
            picks.append(s)
            reason_by_staid[s] = (
                f"macro_region={macro_region};area={area};hydro={hydro};"
                f"flow_var={flow_var};skill={skill}"
            )

    all_ids = sorted(set(picks))
    if len(all_ids) != target_count:
        raise SelectionError(
            f"internal selection error: assembled {len(all_ids)} basin(s), expected {target_count}"
        )

    rows = []
    for staid in all_ids:
        urow = pool.loc[staid]
        rows.append({
            "gauge_id": staid,
            "canonical_basin_role": "development_train",
            "huc02": urow["HUC02"],
            "macro_region": urow["macro_region"],
            "area_class": urow["area_class"],
            "hydro_class": urow["hydro_class"],
            "flow_var_class": urow["flow_var_class"],
            "rbi": float(urow["rbi"]),
            "nse": float(urow["nse"]),
            "skill_stratum": urow["skill_stratum"],
            "selection_reason": reason_by_staid[staid],
        })
    selection_df = pd.DataFrame(rows).sort_values("gauge_id").reset_index(drop=True)

    def _marginal_balance(dim_col: str, valid_values: list[str]) -> dict:
        pop_counts = grid_pool[dim_col].value_counts()
        sub_counts = selection_df[dim_col].value_counts()
        pop_n = len(grid_pool)
        sub_n = len(selection_df)
        out = {}
        for v in sorted(set(valid_values) | set(pop_counts.index) | set(sub_counts.index)):
            pop_frac = float(pop_counts.get(v, 0)) / pop_n if pop_n else 0.0
            sub_frac = float(sub_counts.get(v, 0)) / sub_n if sub_n else 0.0
            out[str(v)] = {
                "population_n": int(pop_counts.get(v, 0)),
                "population_frac": pop_frac,
                "subset_n": int(sub_counts.get(v, 0)),
                "subset_frac": sub_frac,
                "abs_frac_diff": abs(pop_frac - sub_frac),
            }
        return out

    balance = {
        "macro_region": _marginal_balance("macro_region", sorted(grid_pool["macro_region"].unique())),
        "area_class": _marginal_balance("area_class", area_valid),
        "hydro_class": _marginal_balance("hydro_class", hydro_valid),
        "flow_var_class": _marginal_balance("flow_var_class", flow_var_valid),
        "skill_stratum": _marginal_balance("skill_stratum", _SKILL_LABELS_IN_QUARTILE_ORDER),
    }
    max_abs_frac_diff = max(
        (cell["abs_frac_diff"] for dim in balance.values() for cell in dim.values()),
        default=0.0,
    )

    n_nonempty_cells = len(cell_sizes)
    n_cells_with_zero_quota = sum(1 for v in quota_by_cell.values() if v == 0)

    manifest_pieces = {
        "algorithm_id": _ALGORITHM_ID,
        "algorithm_version": _ALGORITHM_VERSION,
        "seed": seed,
        "target_count": target_count,
        "universe_size": int(len(universe)),
        "universe_basins_missing_nse": sorted(missing_nse),
        "universe_basins_missing_flow_variability": sorted(missing_flow_var),
        "basins_excluded_from_grid": sorted(excluded_from_grid),
        "skill_quartile_edges": {"p25": skill_edges[0], "p50": skill_edges[1], "p75": skill_edges[2]},
        "flow_var_tercile_edges": {"e1": flow_var_edges[0], "e2": flow_var_edges[1]},
        "n_nonempty_cells": n_nonempty_cells,
        "n_cells_with_zero_quota": n_cells_with_zero_quota,
        "cell_quota_repair_log": quota_repair_log,
        "counts": {
            "n_selected": int(len(selection_df)),
            "distinct_huc02": int(selection_df["huc02"].nunique()),
            "distinct_macro_region": int(selection_df["macro_region"].nunique()),
        },
        "balance": balance,
        "max_abs_marginal_frac_diff": max_abs_frac_diff,
    }
    return selection_df, manifest_pieces


# ---------------------------------------------------------------------------
# Artifact writing (mirrors hydrograph_atlas_selection.write_selection_artifacts)
# ---------------------------------------------------------------------------

def _render_summary_md(manifest: dict) -> str:
    lines = [
        "# Stage 1 Screening-Subset Selection Summary",
        "",
        f"- status: {manifest.get('status', 'unknown')}",
        f"- algorithm: {manifest['algorithm_id']} v{manifest['algorithm_version']}",
        f"- seed: {manifest['seed']}",
        f"- target_count: {manifest['target_count']}",
        f"- development pool size (input): {manifest['universe_size']}",
        f"- universe basins missing a per-basin NSE value: {len(manifest['universe_basins_missing_nse'])}",
        f"- universe basins missing a flow-variability value: "
        f"{len(manifest['universe_basins_missing_flow_variability'])}",
        f"- basins excluded from grid (invalid stratum value): {len(manifest['basins_excluded_from_grid'])}",
        f"- basins selected: {manifest['counts']['n_selected']}",
        f"- skill quartile edges: p25={manifest['skill_quartile_edges']['p25']:.4f}, "
        f"p50={manifest['skill_quartile_edges']['p50']:.4f}, "
        f"p75={manifest['skill_quartile_edges']['p75']:.4f}",
        f"- flow-variability (RBI) tercile edges: e1={manifest['flow_var_tercile_edges']['e1']:.4f}, "
        f"e2={manifest['flow_var_tercile_edges']['e2']:.4f}",
        f"- distinct HUC02 regions covered: {manifest['counts']['distinct_huc02']}",
        f"- distinct macro-regions covered: {manifest['counts']['distinct_macro_region']}",
        f"- non-empty composite cells: {manifest['n_nonempty_cells']}",
        f"- non-empty cells receiving zero quota (proportional rounding): "
        f"{manifest['n_cells_with_zero_quota']}",
        f"- max absolute marginal population-vs-subset fraction difference "
        f"(across all 5 dimensions): {manifest['max_abs_marginal_frac_diff']:.4f}",
        "",
    ]
    for title, key in [
        ("macro_region marginal balance", "macro_region"),
        ("area_class marginal balance", "area_class"),
        ("hydro_class marginal balance", "hydro_class"),
        ("flow_var_class marginal balance", "flow_var_class"),
        ("skill_stratum marginal balance", "skill_stratum"),
    ]:
        lines += [
            "", f"## {title}", "",
            "| key | population n | population frac | subset n | subset frac | abs diff |",
            "|---|---|---|---|---|---|",
        ]
        for k, v in sorted(manifest["balance"][key].items()):
            lines.append(
                f"| {k} | {v['population_n']} | {v['population_frac']:.4f} | "
                f"{v['subset_n']} | {v['subset_frac']:.4f} | {v['abs_frac_diff']:.4f} |"
            )
    lines += [
        "",
        "## No-cherry-picking statement",
        "",
        "This selection uses only: (1) the canonical split assignment's "
        "area_class/hydro_class/HUC02/STATE fields, (2) validation-only "
        "raw-space NSE from one already-completed seed-run epoch, "
        "(3) an observed-flow flashiness index (RBI) computed independently "
        "of any model prediction, and (4) a fixed seed for proportional "
        "within-cell sampling. No visual inspection, no prediction-error "
        "signal beyond the NSE value itself, no test-set/temporal-test/"
        "spatial-holdout behavior, and no manual basin choice entered this "
        "selection (docs/stage1_validation_optimization_foundation.md, "
        "Part D section 8.1). This subset is for fast development-validation "
        "feedback only -- the full 2,307-basin development population "
        "remains authoritative for final run/checkpoint selection "
        "(section 8.4).",
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

    csv_path = out_dir / "screening_subset_basin_selection.csv"
    selection_df.to_csv(csv_path, index=False)
    paths["screening_subset_basin_selection.csv"] = csv_path

    ids_path = out_dir / "screening_subset_basin_ids.txt"
    ids_path.write_text("\n".join(selection_df["gauge_id"].tolist()) + "\n", encoding="utf-8")
    paths["screening_subset_basin_ids.txt"] = ids_path

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

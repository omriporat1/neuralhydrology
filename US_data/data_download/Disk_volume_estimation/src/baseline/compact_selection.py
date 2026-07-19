"""Stage 1 Compact Scientific Package basin selection building blocks.

Small, deterministic, testable functions -- not a class hierarchy, mirroring
the style of :mod:`src.baseline.splits`. The top-level orchestration function
is :func:`build_compact_selection`.

Scope (see docs/stage1_compact_package_selection.md): select a small
(default 32, range 20-50) diagnostic basin subset strictly from the
canonical non-California ``development_train`` pool
(config/stage1_baseline_splits_v001/split_assignment.csv). This module does
not build NeuralHydrology packages, does not touch ``FlashNHDataset``, and
does not train models.

Method summary:
  1. Load the canonical split-assignment CSV and filter to
     ``development_train`` with fail-fast checks (no California, no
     holdout leakage, no duplicate normalized gauge IDs).
  2. Carve out a small set of reserved (forced-include) edge-case basins,
     evaluated in a fixed order: unusual (non-8-char) normalized STAIDs,
     basins missing the canonical hydro (aridity) stratifier, and (only
     when a static-attribute matrix is supplied) basins with one or more
     missing ``model_input`` columns.
  3. Stratify the remaining population into a 3x3 grid over the canonical
     split's own ``area_class`` x ``hydro_class`` tercile fields (reused
     unchanged -- no new tercile edges are computed here), apportion the
     remaining quota across non-empty cells (largest-remainder method,
     floor of one per non-empty cell where the budget allows), and sample
     within each cell with an independent seeded RNG, preferring HUC02
     round-robin order to spread the pick across distinct HUC02 regions.
  4. qobs completeness and static missingness are recorded as CSV
     annotations/bins (and can trigger a reserved forced-include for
     missingness); they are not additional stratification-grid dimensions,
     to keep the method simple and transparent (no optimizer).

Archive-gap hours (MRMS/RTMA) are global timeline positions, not a
per-basin property (see src/baseline/validity_mask.py) -- this module never
treats forcing-gap exposure as a per-basin selection criterion.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from .splits import SplitGenerationError, load_matrix_for_splits, sha256_of
from .staid import normalize_staid

__all__ = [
    "SelectionError",
    "sha256_of",
    "load_selection_policy",
    "load_split_assignment",
    "select_universe",
    "load_qobs_status",
    "load_static_missingness",
    "flag_unusual_identifier",
    "bin_qobs_completeness",
    "bin_static_missing",
    "build_reserved_selection",
    "allocate_cell_quota",
    "make_cell_rngs",
    "select_within_cell",
    "build_compact_selection",
    "write_selection_artifacts",
]


class SelectionError(ValueError):
    """Raised when compact-selection inputs, policy, or intermediate state are invalid."""


_ALGORITHM_ID = "stage1_compact_diversity_quota_selection_v1"
_ALGORITHM_VERSION = 2

_REQUIRED_TOP_LEVEL_KEYS = [
    "selection_name", "algorithm_id", "algorithm_version", "seed",
    "target_count", "min_target_count", "max_target_count",
    "selection_universe", "stratification", "geography",
    "qobs_completeness", "static_missingness", "reserved_categories",
]

_ABS_PATH_RE = re.compile(r"^([a-zA-Z]:[\\/]|/[^/])")
_CREDENTIAL_KEY_RE = re.compile(r"(password|secret|token|api[_-]?key|credential)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def _check_portability(data, source_path, prefix: str = "") -> None:
    if isinstance(data, dict):
        for k, v in data.items():
            if _CREDENTIAL_KEY_RE.search(str(k)):
                raise SelectionError(f"{source_path}: credential-like key {prefix}{k!r} is not allowed")
            _check_portability(v, source_path, prefix=f"{prefix}{k}.")
    elif isinstance(data, list):
        for i, v in enumerate(data):
            _check_portability(v, source_path, prefix=f"{prefix}[{i}].")
    elif isinstance(data, str):
        if _ABS_PATH_RE.match(data):
            raise SelectionError(f"{source_path}: absolute path found at {prefix.rstrip('.')}: {data!r}")


def load_selection_policy(path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise SelectionError(f"selection policy file not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise SelectionError(f"selection policy {p} did not parse to a mapping")
    missing = [k for k in _REQUIRED_TOP_LEVEL_KEYS if k not in data]
    if missing:
        raise SelectionError(f"selection policy {p} missing required key(s): {missing}")
    if not (data["min_target_count"] <= data["target_count"] <= data["max_target_count"]):
        raise SelectionError(
            f"target_count={data['target_count']} outside "
            f"[{data['min_target_count']}, {data['max_target_count']}]"
        )
    _check_portability(data, p)
    return data


# ---------------------------------------------------------------------------
# Selection-universe loading / validation
# ---------------------------------------------------------------------------

def load_split_assignment(path, required_columns) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise SelectionError(f"split assignment file not found: {p}")
    df = pd.read_csv(p, dtype=str)
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise SelectionError(f"split assignment {p} missing required column(s): {missing_cols}")
    try:
        df["STAID"] = df["STAID"].map(normalize_staid)
    except (TypeError, ValueError) as exc:
        raise SelectionError(f"malformed STAID in {p}: {exc}") from exc
    if df["STAID"].duplicated().any():
        dupes = sorted(df.loc[df["STAID"].duplicated(), "STAID"].unique())
        raise SelectionError(f"duplicate STAID(s) in split assignment {p}: {dupes[:10]}")
    return df


def select_universe(assignment_df: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """Filter to the canonical development role with fail-fast leakage checks.

    Returns a frame indexed by (normalized) STAID with columns
    STATE, HUC02, area_class, hydro_class.
    """
    cfg = policy["selection_universe"]
    required_role = cfg["required_split_role"]
    forbidden_state = cfg["forbidden_state"]
    forbidden_roles = set(cfg["forbidden_split_roles"])

    if required_role not in set(assignment_df["split_role"]):
        raise SelectionError(f"split assignment does not contain any {required_role!r} rows")

    universe = assignment_df.loc[assignment_df["split_role"] == required_role].copy()

    bad_state = universe.loc[universe["STATE"].astype(str) == forbidden_state]
    if len(bad_state):
        raise SelectionError(
            f"{len(bad_state)} {forbidden_state} basin(s) found under "
            f"split_role={required_role!r}: {sorted(bad_state['STAID'])[:10]}"
        )

    universe_staids = set(universe["STAID"])
    forbidden_rows = assignment_df.loc[assignment_df["split_role"].isin(forbidden_roles)]
    overlap = universe_staids & set(forbidden_rows["STAID"])
    if overlap:
        raise SelectionError(
            f"{len(overlap)} basin(s) appear under both {required_role!r} and a "
            f"forbidden role: {sorted(overlap)[:10]}"
        )

    if universe["STAID"].duplicated().any():
        dupes = sorted(universe.loc[universe["STAID"].duplicated(), "STAID"].unique())
        raise SelectionError(f"duplicate normalized STAID(s) in development pool: {dupes[:10]}")

    universe = universe.set_index("STAID")[["STATE", "HUC02", "area_class", "hydro_class"]]
    return universe


# ---------------------------------------------------------------------------
# Optional enrichment loaders
# ---------------------------------------------------------------------------

def load_qobs_status(path, policy: dict) -> pd.DataFrame:
    """Load a target-status/coverage table (e.g. audit/target_status.csv).

    Accepts either of policy['qobs_completeness']['staid_column_candidates']
    as the STAID column and either of ['coverage_column_candidates'] as the
    coverage-fraction column. Returns a frame indexed by normalized STAID
    with column qobs_coverage_fraction (and target_status, if present).
    """
    cfg = policy["qobs_completeness"]
    p = Path(path)
    if not p.is_file():
        raise SelectionError(f"qobs status file not found: {p}")
    header = pd.read_csv(p, nrows=0)

    staid_col = next((c for c in cfg["staid_column_candidates"] if c in header.columns), None)
    if staid_col is None:
        raise SelectionError(
            f"qobs status file {p} has none of the expected STAID column(s): "
            f"{cfg['staid_column_candidates']}"
        )
    coverage_col = next((c for c in cfg["coverage_column_candidates"] if c in header.columns), None)
    if coverage_col is None:
        raise SelectionError(
            f"qobs status file {p} has none of the expected coverage column(s): "
            f"{cfg['coverage_column_candidates']}"
        )

    df = pd.read_csv(p, dtype={staid_col: str})
    try:
        df["STAID"] = df[staid_col].map(normalize_staid)
    except (TypeError, ValueError) as exc:
        raise SelectionError(f"malformed STAID in {p}: {exc}") from exc
    if df["STAID"].duplicated().any():
        dupes = sorted(df.loc[df["STAID"].duplicated(), "STAID"].unique())
        raise SelectionError(f"duplicate STAID(s) in qobs status file {p}: {dupes[:10]}")

    keep_cols = ["STAID", coverage_col] + (["target_status"] if "target_status" in df.columns else [])
    out = df[keep_cols].rename(columns={coverage_col: "qobs_coverage_fraction"})
    return out.set_index("STAID")


def load_static_missingness(attributes_parquet, column_manifest_path, policy: dict) -> pd.Series:
    """Per-basin count of missing (NaN) model_input columns.

    Reuses splits.load_matrix_for_splits for gauge_id validation (no
    duplication of that logic). Returns a Series indexed by gauge_id
    (assumed already canonical, per splits.py convention) named
    static_missing_model_input_count.
    """
    manifest_path = Path(column_manifest_path)
    if not manifest_path.is_file():
        raise SelectionError(f"column-role manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    role_map = manifest.get("columns", {})
    model_input_role = policy["static_missingness"]["model_input_role"]
    model_input_cols = sorted(c for c, meta in role_map.items() if meta.get("role") == model_input_role)
    if not model_input_cols:
        raise SelectionError(
            f"no columns with role={model_input_role!r} found in column manifest {manifest_path}"
        )

    try:
        df = load_matrix_for_splits(attributes_parquet, model_input_cols)
    except SplitGenerationError as exc:
        raise SelectionError(str(exc)) from exc

    counts = df[model_input_cols].isna().sum(axis=1)
    counts.name = "static_missing_model_input_count"
    return counts


# ---------------------------------------------------------------------------
# Binning / flags
# ---------------------------------------------------------------------------

def flag_unusual_identifier(staid: str, standard_length: int) -> bool:
    return len(staid) != standard_length


def bin_qobs_completeness(value, policy: dict) -> str:
    if pd.isna(value):
        return "not_evaluated"
    edges = policy["qobs_completeness"]["bin_edges"]
    v = float(value)
    if v <= edges["low_max"]:
        return "low"
    if v <= edges["mid_max"]:
        return "mid"
    return "high"


def bin_static_missing(count, policy: dict) -> str:
    if pd.isna(count):
        return "not_evaluated"
    edges = policy["static_missingness"]["bin_edges"]
    c = float(count)
    if c <= edges["none_max"]:
        return "none"
    if c <= edges["some_max"]:
        return "some"
    return "high"


# ---------------------------------------------------------------------------
# Reserved (forced-include) edge-case categories
# ---------------------------------------------------------------------------

def build_reserved_selection(
    universe: pd.DataFrame, policy: dict
) -> tuple[list[str], dict[str, list[str]], list[dict]]:
    """Deterministically pick a small, parsimonious set of forced-include
    edge-case basins.

    Categories are evaluated in a fixed order. If a category's predicate is
    already satisfied by a basin reserved by an *earlier* category, no new
    basin is spent on it: the category is logged as ``"covered_by_overlap"``
    pointing at the covering basin, rather than forcing a second, often
    nearly-identical, reserved pick. Otherwise, up to ``forced_include_cap``
    new candidates are picked, preferring -- via a deterministic sort, not an
    optimizer -- whichever not-yet-reserved candidate also satisfies the most
    *other* predicates, so a single basin naturally absorbs overlapping edge
    cases instead of one basin per category.

    Returns (reserved_staids sorted, reasons_by_staid, category_log). A
    basin's reasons list includes every predicate it matches, even if only
    one category's cap "spent" a slot on it. Each category_log entry has a
    ``covered_by`` field (the STAID of the basin that already covers this
    category via overlap, else ``None``).
    """
    cfg = policy["reserved_categories"]
    valid_classes = set(policy["stratification"]["valid_class_values"])
    standard_len = cfg["unusual_identifier"]["standard_length"]
    has_missing_input = "static_missing_model_input_count" in universe.columns

    reasons: dict[str, set] = {}
    for staid in universe.index:
        if flag_unusual_identifier(staid, standard_len):
            reasons.setdefault(staid, set()).add("unusual_identifier")

    hydro_gap_mask = ~universe["hydro_class"].isin(valid_classes)
    for staid in universe.index[hydro_gap_mask]:
        reasons.setdefault(staid, set()).add("hydro_stratifier_gap")

    if has_missing_input:
        missing_mask = universe["static_missing_model_input_count"].fillna(0) > 0
        for staid in universe.index[missing_mask]:
            reasons.setdefault(staid, set()).add("static_missing_value_case")

    reserved: list[str] = []
    reserved_set: set = set()
    category_log: list[dict] = []
    for category in ("unusual_identifier", "hydro_stratifier_gap", "static_missing_value_case"):
        if category == "static_missing_value_case" and not has_missing_input:
            category_log.append({
                "category": category, "status": "not_evaluated_no_attributes_input",
                "n_candidates": 0, "cap": cfg[category]["forced_include_cap"],
                "picked": [], "covered_by": None,
            })
            continue

        covering = sorted(s for s in reserved if category in reasons.get(s, ()))
        if covering:
            category_log.append({
                "category": category, "status": "covered_by_overlap",
                "n_candidates": 0, "cap": cfg[category]["forced_include_cap"],
                "picked": [], "covered_by": covering[0],
            })
            continue

        cap = cfg[category]["forced_include_cap"]
        candidates = sorted(
            (s for s, rs in reasons.items() if category in rs and s not in reserved_set),
            key=lambda s: (-len(reasons[s]), s),
        )
        picked = candidates[:cap]
        for s in picked:
            reserved.append(s)
            reserved_set.add(s)
        category_log.append({
            "category": category,
            "status": "ok" if picked else "no_eligible_candidates",
            "n_candidates": len(candidates),
            "cap": cap,
            "picked": picked,
            "covered_by": None,
        })

    reserved_sorted = sorted(reserved_set)
    reasons_by_staid = {s: sorted(reasons[s]) for s in reserved_sorted}
    return reserved_sorted, reasons_by_staid, category_log


# ---------------------------------------------------------------------------
# Cell quota apportionment + seeded within-cell sampling
# ---------------------------------------------------------------------------

def allocate_cell_quota(
    cell_sizes: dict, remaining_quota: int, min_per_nonempty_cell: int
) -> tuple[dict, list[dict]]:
    """Largest-remainder apportionment of remaining_quota across non-empty cells.

    Guarantees min_per_nonempty_cell per cell where the total budget allows;
    trims deterministically (largest cells first) if the guaranteed minimums
    alone would exceed remaining_quota.
    """
    if remaining_quota < 0:
        raise SelectionError(f"remaining_quota is negative: {remaining_quota}")
    keys = sorted(cell_sizes.keys())
    if not keys:
        if remaining_quota == 0:
            return {}, []
        raise SelectionError("no non-empty cells available to allocate remaining_quota")

    total_available = sum(cell_sizes.values())
    if remaining_quota > total_available:
        raise SelectionError(
            f"requested {remaining_quota} diversity-quota picks but only "
            f"{total_available} basin(s) available across non-empty cells"
        )

    quota = {k: min(min_per_nonempty_cell, cell_sizes[k]) for k in keys}
    over = sum(quota.values()) - remaining_quota
    repair_log: list[dict] = []
    if over > 0:
        trim_order = sorted(keys, key=lambda k: (-cell_sizes[k], k))
        i = 0
        trimmed = 0
        while trimmed < over:
            k = trim_order[i % len(trim_order)]
            if quota[k] > 0:
                quota[k] -= 1
                trimmed += 1
            i += 1
        repair_log.append({
            "step": "min_per_cell_guarantee_exceeded_remaining_quota",
            "trimmed": trimmed,
        })
        return quota, repair_log

    remaining = remaining_quota - sum(quota.values())
    while remaining > 0:
        spare = {k: cell_sizes[k] - quota[k] for k in keys if cell_sizes[k] - quota[k] > 0}
        if not spare:
            raise SelectionError(
                f"internal error: no spare cell capacity left with remaining={remaining}"
            )
        total_spare = sum(spare.values())
        ideal = {k: remaining * v / total_spare for k, v in spare.items()}
        add = {k: min(int(np.floor(v)), spare[k]) for k, v in ideal.items()}
        leftover = remaining - sum(add.values())
        if leftover > 0:
            fractional_order = sorted(
                spare.keys(), key=lambda k: (-(ideal[k] - int(np.floor(ideal[k]))), k)
            )
            for k in fractional_order:
                if leftover <= 0:
                    break
                if add[k] < spare[k]:
                    add[k] += 1
                    leftover -= 1
            if leftover > 0:
                repair_log.append({
                    "step": "cell_quota_redistribution_pass_unresolved",
                    "unresolved_remaining": leftover,
                })
        for k, v in add.items():
            quota[k] += v
        newly_assigned = sum(add.values())
        remaining -= newly_assigned
        if newly_assigned == 0:
            break

    if remaining > 0:
        raise SelectionError(
            f"internal error: could not fully apportion remaining_quota "
            f"({remaining} unit(s) left unassigned)"
        )
    return quota, repair_log


def make_cell_rngs(seed: int, cell_keys: list) -> dict:
    """One independent, deterministic Generator per cell, spawned in sorted-key order."""
    keys_sorted = sorted(cell_keys)
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(len(keys_sorted)) if keys_sorted else []
    return {k: np.random.default_rng(c) for k, c in zip(keys_sorted, children)}


def select_within_cell(cell_df: pd.DataFrame, quota: int, rng: np.random.Generator) -> list[str]:
    """HUC02 round-robin pick using a seeded *permuted* visitation order.

    HUC02 keys are sorted only to fix a stable input order for the
    permutation draw -- the round-robin visitation order itself is then a
    deterministic random permutation of that list (``rng.permutation``), so
    no HUC02 code is systematically favored just because it sorts first.
    Each round after the first revisits groups in the same permuted order.
    Candidates within each HUC02 group are independently seeded-shuffled.
    Per-group shuffles are drawn in sorted-key order (before the visitation
    permutation is drawn) so rng consumption is deterministic regardless of
    frame row order.
    """
    if quota <= 0:
        return []
    huc_keys_sorted = sorted(cell_df["HUC02"].unique())
    groups: dict[str, list[str]] = {}
    for huc02 in huc_keys_sorted:
        sub = cell_df.loc[cell_df["HUC02"] == huc02]
        order = rng.permutation(len(sub))
        groups[huc02] = [sub.index[i] for i in order]

    visit_order = rng.permutation(len(huc_keys_sorted))
    huc_visit_keys = [huc_keys_sorted[i] for i in visit_order]

    picked: list[str] = []
    pos = 0
    while len(picked) < quota:
        progressed = False
        for huc02 in huc_visit_keys:
            if pos < len(groups[huc02]):
                picked.append(groups[huc02][pos])
                progressed = True
                if len(picked) == quota:
                    break
        pos += 1
        if not progressed:
            break
    return picked


# ---------------------------------------------------------------------------
# Macro-region (HUC02) geographic diagnostics
# ---------------------------------------------------------------------------

def build_macro_region_map(policy: dict) -> dict[str, str]:
    """HUC02 -> macro-region name, from the explicit, versioned
    ``policy['geography']['macro_regions']`` mapping (never inferred).

    Fails loudly if a HUC02 code is listed under more than one macro
    region -- an ambiguous/contradictory policy, not a data problem.
    """
    regions = policy["geography"]["macro_regions"]
    huc_to_region: dict[str, str] = {}
    for region_name, huc_codes in regions.items():
        for huc in huc_codes:
            if huc in huc_to_region:
                raise SelectionError(
                    f"HUC02 {huc!r} is assigned to both macro regions "
                    f"{huc_to_region[huc]!r} and {region_name!r} in "
                    "policy geography.macro_regions"
                )
            huc_to_region[huc] = region_name
    return huc_to_region


def macro_region_for_huc02(huc02: str, region_map: dict[str, str]) -> str:
    if huc02 not in region_map:
        raise SelectionError(
            f"HUC02 {huc02!r} is not covered by policy geography.macro_regions "
            "(explicit versioned mapping); update the policy before selecting "
            "basins with this HUC02 code"
        )
    return region_map[huc02]


def macro_region_side(macro_region: str, policy: dict) -> str:
    """'east', 'west', or 'other', per policy['geography']'s explicit
    east_macro_regions / west_macro_regions lists. Never inferred from the
    macro-region name itself."""
    geo = policy["geography"]
    if macro_region in geo.get("east_macro_regions", []):
        return "east"
    if macro_region in geo.get("west_macro_regions", []):
        return "west"
    return "other"


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def build_compact_selection(
    universe: pd.DataFrame,
    qobs_df: Optional[pd.DataFrame],
    policy: dict,
) -> tuple[pd.DataFrame, dict]:
    """Assign the compact selection. universe must come from select_universe(),
    optionally enriched with DRAIN_SQKM / ari_ix_uav / static_missing_model_input_count
    columns before calling this function.
    """
    target_count = policy["target_count"]
    seed = policy["seed"]
    valid_classes = policy["stratification"]["valid_class_values"]
    min_per_cell = policy["stratification"]["min_quota_per_nonempty_cell"]

    if len(universe) < target_count:
        raise SelectionError(
            f"development pool has only {len(universe)} basin(s), cannot select "
            f"target_count={target_count}"
        )

    reserved_ids, reserved_reasons, category_log = build_reserved_selection(universe, policy)
    if len(reserved_ids) > target_count:
        raise SelectionError(
            f"reserved forced-include categories alone need {len(reserved_ids)} slot(s), "
            f"exceeding target_count={target_count}"
        )

    hydro_gap_mask = ~universe["hydro_class"].isin(valid_classes)
    excluded_from_grid = set(universe.index[hydro_gap_mask]) | set(reserved_ids)
    grid_pool = universe.drop(index=sorted(excluded_from_grid))

    remaining_quota = target_count - len(reserved_ids)

    cell_sizes = {}
    for area in valid_classes:
        for hydro in valid_classes:
            n = int(((grid_pool["area_class"] == area) & (grid_pool["hydro_class"] == hydro)).sum())
            if n > 0:
                cell_sizes[(area, hydro)] = n

    if remaining_quota > 0 and not cell_sizes:
        raise SelectionError("grid pool has zero eligible (area_class, hydro_class) cells")

    quota_by_cell, quota_repair_log = allocate_cell_quota(cell_sizes, remaining_quota, min_per_cell)
    rngs = make_cell_rngs(seed, list(quota_by_cell.keys()))

    grid_picks: list[str] = []
    grid_reason_by_staid: dict[str, str] = {}
    for cell_key in sorted(quota_by_cell.keys()):
        area, hydro = cell_key
        q = quota_by_cell[cell_key]
        if q <= 0:
            continue
        cell_df = grid_pool.loc[(grid_pool["area_class"] == area) & (grid_pool["hydro_class"] == hydro)]
        picks = select_within_cell(cell_df, q, rngs[cell_key])
        for s in picks:
            grid_picks.append(s)
            grid_reason_by_staid[s] = f"diversity_quota:area={area};hydro={hydro}"

    all_ids = sorted(set(reserved_ids) | set(grid_picks))
    if len(all_ids) != target_count:
        raise SelectionError(
            f"internal selection error: assembled {len(all_ids)} basin(s), expected {target_count}"
        )

    region_map = build_macro_region_map(policy)

    rows = []
    for staid in all_ids:
        urow = universe.loc[staid]
        reasons = list(reserved_reasons.get(staid, []))
        if staid in grid_reason_by_staid:
            reasons.append(grid_reason_by_staid[staid])
        if not reasons:
            reasons.append("diversity_quota")

        qobs_coverage = np.nan
        target_status = ""
        if qobs_df is not None and staid in qobs_df.index:
            qobs_coverage = qobs_df.loc[staid, "qobs_coverage_fraction"]
            if "target_status" in qobs_df.columns:
                target_status = qobs_df.loc[staid, "target_status"]

        static_missing = urow.get("static_missing_model_input_count", np.nan)
        macro_region = macro_region_for_huc02(urow["HUC02"], region_map)

        rows.append({
            "gauge_id": staid,
            "canonical_basin_role": "development_train",
            "huc02": urow["HUC02"],
            "macro_region": macro_region,
            "macro_region_side": macro_region_side(macro_region, policy),
            "drain_sqkm": urow.get("DRAIN_SQKM", np.nan),
            "area_class": urow["area_class"],
            "aridity_value": urow.get("ari_ix_uav", np.nan),
            "hydro_class": urow["hydro_class"],
            "qobs_coverage_fraction": qobs_coverage,
            "qobs_completeness_bin": bin_qobs_completeness(qobs_coverage, policy),
            "target_status": target_status,
            "static_missing_model_input_count": static_missing,
            "static_missing_bin": bin_static_missing(static_missing, policy),
            "unusual_identifier_flag": flag_unusual_identifier(
                staid, policy["reserved_categories"]["unusual_identifier"]["standard_length"]
            ),
            "selection_reason": ";".join(reasons),
        })

    selection_df = pd.DataFrame(rows).sort_values("gauge_id").reset_index(drop=True)

    distinct_huc02 = int(selection_df["huc02"].nunique())
    soft_min = policy["geography"]["distinct_huc02_soft_minimum"]

    distinct_macro_regions = int(selection_df["macro_region"].nunique())
    macro_soft_min = policy["geography"].get("distinct_macro_region_soft_minimum", 1)
    n_east = int((selection_df["macro_region_side"] == "east").sum())
    n_west = int((selection_df["macro_region_side"] == "west").sum())
    require_east_west = policy["geography"].get("require_east_west_spread", True)
    if require_east_west and (n_east == 0 or n_west == 0):
        raise SelectionError(
            "compact selection lacks required east/west CONUS breadth: "
            f"n_east={n_east}, n_west={n_west} (macro_region_counts="
            f"{selection_df['macro_region'].value_counts().sort_index().to_dict()}); "
            "this defeats the purpose of a scientifically representative compact "
            "package -- adjust the diversity-quota cells, seed, or macro-region "
            "policy so both sides of CONUS are represented"
        )

    manifest_pieces = {
        "algorithm_id": _ALGORITHM_ID,
        "algorithm_version": _ALGORITHM_VERSION,
        "seed": seed,
        "target_count": target_count,
        "universe_size": int(len(universe)),
        "reserved_category_log": category_log,
        "cell_sizes": {f"{a}:{h}": n for (a, h), n in sorted(cell_sizes.items())},
        "cell_quota": {f"{a}:{h}": n for (a, h), n in sorted(quota_by_cell.items())},
        "cell_quota_repair_log": quota_repair_log,
        "counts": {
            "n_selected": int(len(selection_df)),
            "n_reserved": len(reserved_ids),
            "n_grid_picks": len(grid_picks),
            "distinct_huc02": distinct_huc02,
        },
        "huc02_counts": {str(k): int(v) for k, v in sorted(selection_df["huc02"].value_counts().items())},
        "area_class_counts": {
            str(k): int(v) for k, v in sorted(selection_df["area_class"].value_counts().items())
        },
        "hydro_class_counts": {
            str(k): int(v) for k, v in sorted(selection_df["hydro_class"].value_counts().items())
        },
        "qobs_completeness_bin_counts": {
            str(k): int(v) for k, v in sorted(selection_df["qobs_completeness_bin"].value_counts().items())
        },
        "static_missing_bin_counts": {
            str(k): int(v) for k, v in sorted(selection_df["static_missing_bin"].value_counts().items())
        },
        "distinct_huc02_soft_minimum": soft_min,
        "distinct_huc02_soft_minimum_met": distinct_huc02 >= soft_min,
        "macro_region_counts": {
            str(k): int(v) for k, v in sorted(selection_df["macro_region"].value_counts().items())
        },
        "macro_region_side_counts": {
            str(k): int(v) for k, v in sorted(selection_df["macro_region_side"].value_counts().items())
        },
        "distinct_macro_regions": distinct_macro_regions,
        "macro_region_soft_minimum": macro_soft_min,
        "macro_region_soft_minimum_met": distinct_macro_regions >= macro_soft_min,
        "east_west_breadth": {
            "n_east": n_east,
            "n_west": n_west,
            "required": require_east_west,
        },
    }
    return selection_df, manifest_pieces


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------

def _render_summary_md(manifest: dict) -> str:
    lines = [
        "# Stage 1 Compact Package Basin Selection Summary",
        "",
        f"- status: {manifest.get('status', 'unknown')} "
        "(candidate = not yet the final accepted Compact Scientific Package; "
        "see docs/stage1_compact_package_selection.md, \"Two selection runs, not one\")",
        f"- algorithm: {manifest['algorithm_id']} v{manifest['algorithm_version']}",
        f"- seed: {manifest['seed']}",
        f"- target_count: {manifest['target_count']}",
        f"- development pool size (input): {manifest['universe_size']}",
        f"- basins selected: {manifest['counts']['n_selected']}",
        f"- reserved (forced-include) basins: {manifest['counts']['n_reserved']}",
        f"- diversity-quota basins: {manifest['counts']['n_grid_picks']}",
        f"- distinct HUC02 regions covered: {manifest['counts']['distinct_huc02']} "
        f"(soft minimum {manifest['distinct_huc02_soft_minimum']}, "
        f"{'met' if manifest['distinct_huc02_soft_minimum_met'] else 'NOT MET -- advisory only'})",
        f"- distinct macro-regions covered: {manifest['distinct_macro_regions']} "
        f"(soft minimum {manifest['macro_region_soft_minimum']}, "
        f"{'met' if manifest['macro_region_soft_minimum_met'] else 'NOT MET -- advisory only'})",
        f"- east/west CONUS breadth: n_east={manifest['east_west_breadth']['n_east']}, "
        f"n_west={manifest['east_west_breadth']['n_west']} "
        f"({'required and met' if manifest['east_west_breadth']['required'] else 'not required'})",
        "",
        "## Reserved edge-case categories",
        "",
        "| category | status | n_candidates | cap | picked | covered_by |",
        "|---|---|---|---|---|---|",
    ]
    for entry in manifest["reserved_category_log"]:
        lines.append(
            f"| {entry['category']} | {entry['status']} | "
            f"{entry.get('n_candidates', '')} | {entry.get('cap', '')} | "
            f"{', '.join(entry.get('picked', []))} | {entry.get('covered_by') or ''} |"
        )
    for title, key in [
        ("area_class representation", "area_class_counts"),
        ("hydro_class (wet/dry) representation", "hydro_class_counts"),
        ("qobs completeness bin representation", "qobs_completeness_bin_counts"),
        ("static-attribute missingness bin representation", "static_missing_bin_counts"),
        ("HUC02 breadth", "huc02_counts"),
        ("macro-region representation", "macro_region_counts"),
        ("macro-region east/west side representation", "macro_region_side_counts"),
    ]:
        lines += ["", f"## {title}", "", "| key | n |", "|---|---|"]
        for k, v in sorted(manifest[key].items()):
            lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## Forcing-gap note",
        "",
        "Archive-gap hours (MRMS/RTMA) are global timeline positions, not a "
        "per-basin property (src/baseline/validity_mask.py) -- this selection "
        "does not claim any basin has more or fewer forcing gaps than another. "
        "The selected package will exercise issue times around the shared "
        "global gap intervals when later built/audited (see "
        "docs/stage1_compact_package_selection.md).",
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

    csv_path = out_dir / "compact_basin_selection.csv"
    selection_df.to_csv(csv_path, index=False)
    paths["compact_basin_selection.csv"] = csv_path

    ids_path = out_dir / "compact_basin_ids.txt"
    ids_path.write_text("\n".join(selection_df["gauge_id"].tolist()) + "\n", encoding="utf-8")
    paths["compact_basin_ids.txt"] = ids_path

    summary_md_path = out_dir / "selection_summary.md"
    summary_md_path.write_text(_render_summary_md(manifest), encoding="utf-8")
    paths["selection_summary.md"] = summary_md_path

    summary_json = {
        "status": manifest.get("status", "unknown"),
        "n_selected": int(len(selection_df)),
        "counts": manifest["counts"],
        "huc02_counts": manifest["huc02_counts"],
        "area_class_counts": manifest["area_class_counts"],
        "hydro_class_counts": manifest["hydro_class_counts"],
        "qobs_completeness_bin_counts": manifest["qobs_completeness_bin_counts"],
        "static_missing_bin_counts": manifest["static_missing_bin_counts"],
        "reserved_category_log": manifest["reserved_category_log"],
        "distinct_huc02_soft_minimum": manifest["distinct_huc02_soft_minimum"],
        "distinct_huc02_soft_minimum_met": manifest["distinct_huc02_soft_minimum_met"],
        "macro_region_counts": manifest["macro_region_counts"],
        "macro_region_side_counts": manifest["macro_region_side_counts"],
        "distinct_macro_regions": manifest["distinct_macro_regions"],
        "macro_region_soft_minimum": manifest["macro_region_soft_minimum"],
        "macro_region_soft_minimum_met": manifest["macro_region_soft_minimum_met"],
        "east_west_breadth": manifest["east_west_breadth"],
    }
    summary_json_path = out_dir / "selection_summary.json"
    summary_json_path.write_text(json.dumps(summary_json, indent=2, default=str), encoding="utf-8")
    paths["selection_summary.json"] = summary_json_path

    artifact_sha256 = {name: sha256_of(p) for name, p in sorted(paths.items())}
    manifest_to_write = {**manifest, "artifact_sha256": artifact_sha256}
    manifest_path = out_dir / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest_to_write, indent=2, default=str), encoding="utf-8")
    paths["selection_manifest.json"] = manifest_path

    return paths

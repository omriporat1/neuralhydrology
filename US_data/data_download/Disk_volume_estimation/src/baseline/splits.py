"""Stage 1 spatial split generator building blocks (Milestone 2K-G-I I-A2).

Small, deterministic, testable functions -- not a class hierarchy. The
top-level orchestration function is :func:`build_split_assignment`; everything
else is a building block it (or a test) calls directly.

Design intent (docs/stage1_scientific_baseline_design.md §8b, plan §7,
2K-G-I I-A2 sign-off): a reproducible, approximately balanced random split,
not an optimized allocation algorithm.

Stratification: HUC02 x area tercile x aridity tercile for non-California,
area tercile x aridity tercile (HUC02 diagnostic only) for California.
Terciles: edges = numpy linear-interpolation quantiles at (1/3, 2/3) on the
population with observed values of that field; class assignment is
low: value <= edge_1; middle: edge_1 < value <= edge_2; high: value > edge_2.

Sparse-stratum fallback (single level, no intermediate HUC02 x area rung):
an initial stratum with n < min_stratum_size has its basins pooled with every
other sparse stratum sharing the same pool key (HUC02 for non-CA; one
statewide pool for CA). If the resulting pool has >= min_stratum_size basins,
it is sampled like any other allocation group; otherwise every basin in the
pool is forced into the training role, never into a holdout.

Missing-aridity handling (Option B, adopted 2026-07-13): any basin missing
the aridity stratifier never enters stratification or sampling at all -- it
is assigned directly to the training role (development_train for non-CA,
california_finetune_train for CA), and can therefore never be selected into
a holdout. No "missing" tercile category is created and ari_ix_uav is never
imputed for split-generation purposes.

Randomness: one Generator per population (non-CA, CA), derived from a single
policy seed via numpy.random.SeedSequence.spawn(2) so the two draws are
statistically independent but fully deterministic from one seed and never
depend on each other's group counts. Within a population, allocation groups
are processed in sorted group_id order and each group's STAID list is sorted
before sampling, so identical seed + identical inputs always produce
byte-identical output, and Python's process-randomized hash() is never used.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .staid import normalize_staid

__all__ = [
    "SplitGenerationError",
    "AllocationGroup",
    "US_STATE_ABBR",
    "HUC02_RE",
    "REASON_MISSING_STRATIFIER",
    "REASON_DIRECT_STRATUM_SAMPLE",
    "REASON_SPARSE_POOL_SAMPLE",
    "REASON_SPARSE_POOL_FORCED_TRAINING",
    "sha256_of",
    "load_eligible_basins",
    "load_matrix_for_splits",
    "join_eligible_with_matrix",
    "validate_split_support_fields",
    "compute_tercile_edges",
    "assign_tercile_class",
    "build_allocation_groups",
    "make_split_rngs",
    "select_holdout",
    "build_split_assignment",
    "write_split_artifacts",
]


class SplitGenerationError(ValueError):
    """Raised when split-generator inputs or intermediate state are invalid."""


US_STATE_ABBR = frozenset({
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC", "PR", "VI", "GU", "AS", "MP",
})
HUC02_RE = re.compile(r"^\d{2}[LU]?$")

_MIN_STRATUM_SIZE_DEFAULT = 10
_ALGORITHM_ID = "stage1_seeded_random_stratified_split_v1"
_ALGORITHM_VERSION = 1

# Concise, consistent per-basin assignment_reason values -- one per code path,
# not an elaborate framework. "missing_hydroatlas_stratifier" is the binding
# value pinned in config/stage1_scientific_baseline_v001.yaml's
# spatial_split.missing_hydroclimate_policy.assignment_reason.
REASON_MISSING_STRATIFIER = "missing_hydroatlas_stratifier"
REASON_DIRECT_STRATUM_SAMPLE = "direct_stratum_sample"
REASON_SPARSE_POOL_SAMPLE = "sparse_pool_sample"
REASON_SPARSE_POOL_FORCED_TRAINING = "sparse_pool_forced_training"


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Loading / validation
# ---------------------------------------------------------------------------

def load_eligible_basins(path) -> list[str]:
    """Read a one-STAID-per-line file, strictly normalize, reject duplicates.

    Returns the STAIDs sorted (the file's on-disk order is not load-bearing).
    """
    p = Path(path)
    if not p.is_file():
        raise SplitGenerationError(f"eligible basins file not found: {p}")
    raw_lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    raw_lines = [ln for ln in raw_lines if ln]
    if not raw_lines:
        raise SplitGenerationError(f"eligible basins file is empty: {p}")

    staids: list[str] = []
    for ln in raw_lines:
        try:
            staids.append(normalize_staid(ln))
        except (TypeError, ValueError) as exc:
            raise SplitGenerationError(f"malformed STAID {ln!r} in {p}: {exc}") from exc

    if len(staids) != len(set(staids)):
        dupes = sorted({s for s in staids if staids.count(s) > 1})
        raise SplitGenerationError(f"duplicate STAIDs in eligible list: {dupes}")
    return sorted(staids)


def load_matrix_for_splits(parquet_path, required_columns) -> pd.DataFrame:
    """Load the static-attribute parquet, indexed by gauge_id (string).

    Validates: gauge_id present as index or column and non-numeric dtype
    (pandas>=3.0's dedicated string dtype is accepted -- only int/float
    coercion is rejected); no duplicate gauge_id; every required column
    present. Does not renormalize gauge_id values -- the matrix is assumed to
    already carry canonical STAIDs from its own builder.
    """
    df = pd.read_parquet(parquet_path)
    if df.index.name == "gauge_id":
        gauge_id = df.index.to_series()
    elif "gauge_id" in df.columns:
        df = df.set_index("gauge_id")
        gauge_id = df.index.to_series()
    else:
        raise SplitGenerationError("gauge_id not found as column or index name in matrix")

    if pd.api.types.is_numeric_dtype(gauge_id):
        raise SplitGenerationError(
            f"gauge_id dtype is {gauge_id.dtype} (numeric) -- coercion suspected"
        )
    if gauge_id.isnull().any():
        raise SplitGenerationError("gauge_id contains null values")
    if gauge_id.duplicated().any():
        dupes = gauge_id[gauge_id.duplicated()].tolist()[:10]
        raise SplitGenerationError(f"duplicate gauge_id values in matrix: {dupes}")

    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise SplitGenerationError(f"matrix missing required columns: {missing_cols}")
    return df


def join_eligible_with_matrix(matrix_df: pd.DataFrame, eligible_staids: list[str]) -> pd.DataFrame:
    """Exact one-to-one join: every eligible STAID must match exactly one row."""
    if len(eligible_staids) != len(set(eligible_staids)):
        raise SplitGenerationError("duplicate STAIDs passed to join_eligible_with_matrix")
    eligible_set = set(eligible_staids)
    missing = sorted(eligible_set - set(matrix_df.index))
    if missing:
        raise SplitGenerationError(
            f"{len(missing)} eligible basin(s) missing from matrix: {missing[:10]}"
        )
    joined = matrix_df.loc[matrix_df.index.isin(eligible_set)]
    if len(joined) != len(eligible_set):
        dup = joined.index[joined.index.duplicated()].unique().tolist()
        raise SplitGenerationError(
            f"join is not one-to-one: matrix has multiple rows for {dup[:10]}"
        )
    return joined.loc[sorted(eligible_set)].copy()


def validate_split_support_fields(frame: pd.DataFrame) -> None:
    """Fail loudly on malformed STATE/HUC02 or invalid DRAIN_SQKM/ari_ix_uav.

    ari_ix_uav is allowed to be NaN (Option B handles missing aridity
    explicitly downstream); it may never be +/-inf.
    """
    bad_state = sorted(set(frame["STATE"].astype(str)) - US_STATE_ABBR)
    if bad_state:
        raise SplitGenerationError(f"malformed STATE values: {bad_state}")

    bad_huc02 = sorted(v for v in set(frame["HUC02"].astype(str)) if HUC02_RE.match(v) is None)
    if bad_huc02:
        raise SplitGenerationError(f"malformed HUC02 values: {bad_huc02}")

    area = frame["DRAIN_SQKM"].astype(float)
    if area.isnull().any():
        bad = frame.index[area.isnull()].tolist()[:10]
        raise SplitGenerationError(f"DRAIN_SQKM missing for basin(s): {bad}")
    if not np.isfinite(area.to_numpy()).all():
        bad = frame.index[~np.isfinite(area.to_numpy())].tolist()[:10]
        raise SplitGenerationError(f"DRAIN_SQKM non-finite for basin(s): {bad}")
    if (area <= 0).any():
        bad = frame.index[(area <= 0).to_numpy()].tolist()[:10]
        raise SplitGenerationError(f"DRAIN_SQKM nonpositive for basin(s): {bad}")

    ari = frame["ari_ix_uav"].astype(float)
    ari_values = ari.to_numpy()
    bad_ari = np.isnan(ari_values) | np.isfinite(ari_values)
    if not bad_ari.all():
        bad = frame.index[~bad_ari].tolist()[:10]
        raise SplitGenerationError(f"ari_ix_uav non-finite (inf) for basin(s): {bad}")


# ---------------------------------------------------------------------------
# Tercile binning
# ---------------------------------------------------------------------------

def compute_tercile_edges(values: pd.Series) -> tuple[float, float]:
    """Edges (e1, e2) via numpy linear-interpolation quantiles at (1/3, 2/3)."""
    if values.isnull().any():
        raise SplitGenerationError("compute_tercile_edges received null values")
    v = values.to_numpy(dtype=float)
    if v.size == 0:
        raise SplitGenerationError("compute_tercile_edges received an empty population")
    edges = np.quantile(v, [1 / 3, 2 / 3], method="linear")
    return float(edges[0]), float(edges[1])


def assign_tercile_class(values: pd.Series, edges: tuple[float, float]) -> pd.Series:
    """low: v <= e1; middle: e1 < v <= e2; high: v > e2.

    If e1 == e2 (a mass point covers >= 1/3 of the population), the middle
    class becomes structurally empty -- this is a known, accepted degeneracy,
    not an error.
    """
    if values.isnull().any():
        raise SplitGenerationError("assign_tercile_class received null values")
    e1, e2 = edges
    v = values.to_numpy(dtype=float)
    out = np.where(v <= e1, "low", np.where(v <= e2, "middle", "high"))
    return pd.Series(out, index=values.index)


# ---------------------------------------------------------------------------
# Allocation groups + seeded selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AllocationGroup:
    group_id: str
    kind: str  # "stratum" or "sparse_pool"
    staids: tuple[str, ...]


def build_allocation_groups(
    frame: pd.DataFrame,
    staid_col: str,
    stratum_cols: list[str],
    pool_key_col: Optional[str],
    min_stratum_size: int = _MIN_STRATUM_SIZE_DEFAULT,
) -> tuple[list[AllocationGroup], list[dict]]:
    """Stratify, then pool every sparse (n < min_stratum_size) stratum once.

    pool_key_col groups sparse basins for pooling (HUC02 for non-CA); pass
    None to pool every sparse basin into a single statewide pool (CA). If the
    resulting pool is itself still below min_stratum_size, its basins are
    NOT turned into an allocation group -- they are returned in fallback_log
    for the caller to force into the training role directly. There is no
    further escalation (no intermediate HUC02 x area rung, no whole-HUC02
    downgrade of otherwise-sufficient strata).
    """
    groups: list[AllocationGroup] = []
    fallback_log: list[dict] = []
    sparse_rows = []

    for key, sub in frame.groupby(stratum_cols, sort=True, observed=True):
        key_parts = key if isinstance(key, tuple) else (key,)
        if len(sub) >= min_stratum_size:
            group_id = "stratum:" + ":".join(str(p) for p in key_parts)
            staids = tuple(sorted(sub[staid_col].tolist()))
            groups.append(AllocationGroup(group_id=group_id, kind="stratum", staids=staids))
        else:
            sparse_rows.append(sub)

    if sparse_rows:
        sparse = pd.concat(sparse_rows)
        if pool_key_col is None:
            pool_iter = [("ALL", sparse)]
        else:
            pool_iter = list(sparse.groupby(pool_key_col, sort=True, observed=True))
        for pool_key, sub in sorted(pool_iter, key=lambda kv: str(kv[0])):
            staids = sorted(sub[staid_col].tolist())
            if len(staids) >= min_stratum_size:
                group_id = f"sparse_pool:{pool_key}"
                groups.append(
                    AllocationGroup(group_id=group_id, kind="sparse_pool", staids=tuple(staids))
                )
            else:
                for s in staids:
                    fallback_log.append({
                        "staid": s,
                        "reason": "sparse_pool_below_min_stratum_size",
                        "pool_key": str(pool_key),
                        "pool_size": len(staids),
                    })

    groups.sort(key=lambda g: g.group_id)
    return groups, fallback_log


def make_split_rngs(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    """Two independent, deterministic Generators (non-CA, CA) from one seed."""
    ss = np.random.SeedSequence(seed)
    nonca_seed, ca_seed = ss.spawn(2)
    return np.random.default_rng(nonca_seed), np.random.default_rng(ca_seed)


def select_holdout(
    groups: list[AllocationGroup], rng: np.random.Generator, fraction: float
) -> dict[str, dict]:
    """Seeded random ~fraction holdout within each group, sorted-id order."""
    assignments: dict[str, dict] = {}
    for group in sorted(groups, key=lambda g: g.group_id):
        staids_sorted = sorted(group.staids)
        n = len(staids_sorted)
        k = int(round(fraction * n))
        k = max(0, min(k, n))
        if k > 0:
            idx = rng.choice(n, size=k, replace=False)
            holdout_set = {staids_sorted[i] for i in idx}
        else:
            holdout_set = set()
        for s in staids_sorted:
            assignments[s] = {
                "is_holdout": s in holdout_set,
                "group_id": group.group_id,
                "group_kind": group.kind,
                "group_size": n,
            }
    return assignments


# ---------------------------------------------------------------------------
# Population-level role assignment
# ---------------------------------------------------------------------------

def _assign_population_roles(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    holdout_fraction: float,
    holdout_role: str,
    training_role: str,
    use_geography: bool,
    min_stratum_size: int,
) -> tuple[list[dict], dict, list[dict]]:
    records: list[dict] = []
    fallback_log: list[dict] = []

    with_aridity = frame.loc[frame["ari_ix_uav"].notnull()].copy()
    missing_aridity = frame.loc[frame["ari_ix_uav"].isnull()].copy()

    area_edges: Optional[tuple[float, float]] = None
    aridity_edges: Optional[tuple[float, float]] = None

    if len(with_aridity) > 0:
        area_edges = compute_tercile_edges(with_aridity["DRAIN_SQKM"])
        aridity_edges = compute_tercile_edges(with_aridity["ari_ix_uav"])
        with_aridity["area_class"] = assign_tercile_class(with_aridity["DRAIN_SQKM"], area_edges)
        with_aridity["hydro_class"] = assign_tercile_class(with_aridity["ari_ix_uav"], aridity_edges)

    for staid, row in missing_aridity.iterrows():
        area_class = None
        if area_edges is not None:
            area_class = assign_tercile_class(
                pd.Series([row["DRAIN_SQKM"]], index=[staid]), area_edges
            ).iloc[0]
        records.append({
            "STAID": staid,
            "split_role": training_role,
            "STATE": row["STATE"],
            "HUC02": row["HUC02"],
            "area_class": area_class,
            "hydro_class": "missing",
            "stratum_id": REASON_MISSING_STRATIFIER,
            "assignment_reason": REASON_MISSING_STRATIFIER,
        })

    if len(with_aridity) == 0:
        edges = {"area_edges": area_edges, "aridity_edges": aridity_edges}
        return records, edges, fallback_log

    stratum_cols = ["HUC02", "area_class", "hydro_class"] if use_geography else ["area_class", "hydro_class"]
    pool_key_col = "HUC02" if use_geography else None

    frame_for_groups = with_aridity.reset_index()
    frame_for_groups = frame_for_groups.rename(columns={frame_for_groups.columns[0]: "STAID"})
    groups, sparse_fallback = build_allocation_groups(
        frame_for_groups,
        staid_col="STAID",
        stratum_cols=stratum_cols,
        pool_key_col=pool_key_col,
        min_stratum_size=min_stratum_size,
    )

    lookup = with_aridity[["STATE", "HUC02", "area_class", "hydro_class"]]

    for entry in sparse_fallback:
        staid = entry["staid"]
        row = lookup.loc[staid]
        records.append({
            "STAID": staid,
            "split_role": training_role,
            "STATE": row["STATE"],
            "HUC02": row["HUC02"],
            "area_class": row["area_class"],
            "hydro_class": row["hydro_class"],
            "stratum_id": f"sparse_pool:{entry['pool_key']}",
            "assignment_reason": REASON_SPARSE_POOL_FORCED_TRAINING,
        })
        fallback_log.append(entry)

    assignments = select_holdout(groups, rng, holdout_fraction)
    for staid, meta in assignments.items():
        row = lookup.loc[staid]
        role = holdout_role if meta["is_holdout"] else training_role
        reason = (
            REASON_DIRECT_STRATUM_SAMPLE if meta["group_kind"] == "stratum"
            else REASON_SPARSE_POOL_SAMPLE
        )
        records.append({
            "STAID": staid,
            "split_role": role,
            "STATE": row["STATE"],
            "HUC02": row["HUC02"],
            "area_class": row["area_class"],
            "hydro_class": row["hydro_class"],
            "stratum_id": meta["group_id"],
            "assignment_reason": reason,
        })

    edges = {"area_edges": area_edges, "aridity_edges": aridity_edges}
    return records, edges, fallback_log


def build_split_assignment(
    joined: pd.DataFrame,
    seed: int,
    nonca_holdout_fraction: float,
    ca_holdout_fraction: float,
    min_stratum_size: int = _MIN_STRATUM_SIZE_DEFAULT,
) -> tuple[pd.DataFrame, dict]:
    """Assign every basin in `joined` a Stage 1 split role.

    `joined` must already be exactly the eligible-basin population (one row
    per basin, indexed by STAID) with STATE/HUC02/DRAIN_SQKM/ari_ix_uav
    columns validated by validate_split_support_fields. Returns
    (assignment_df, manifest_pieces) where assignment_df has one row per
    basin with columns STAID, split_role, STATE, HUC02, area_class,
    hydro_class, stratum_id, assignment_reason.
    """
    validate_split_support_fields(joined)

    ca_mask = joined["STATE"].astype(str) == "CA"
    nonca = joined.loc[~ca_mask].copy()
    ca = joined.loc[ca_mask].copy()

    nonca_rng, ca_rng = make_split_rngs(seed)

    nonca_records, nonca_edges, nonca_fallback = _assign_population_roles(
        nonca, nonca_rng, nonca_holdout_fraction,
        holdout_role="spatial_holdout_nonca", training_role="development_train",
        use_geography=True, min_stratum_size=min_stratum_size,
    )
    ca_records, ca_edges, ca_fallback = _assign_population_roles(
        ca, ca_rng, ca_holdout_fraction,
        holdout_role="california_holdout", training_role="california_finetune_train",
        use_geography=False, min_stratum_size=min_stratum_size,
    )

    assignment_df = pd.DataFrame(nonca_records + ca_records).sort_values("STAID").reset_index(drop=True)

    if len(assignment_df) != len(joined):
        raise SplitGenerationError(
            f"assignment produced {len(assignment_df)} rows for {len(joined)} eligible basins"
        )
    if assignment_df["STAID"].duplicated().any():
        dup = assignment_df.loc[assignment_df["STAID"].duplicated(), "STAID"].tolist()
        raise SplitGenerationError(f"basin(s) received multiple roles: {dup}")
    if set(assignment_df["STAID"]) != set(joined.index):
        raise SplitGenerationError("assignment basin set does not match input basin set exactly")

    def _counts():
        return assignment_df["split_role"].value_counts().to_dict()

    counts = _counts()
    dev_n = counts.get("development_train", 0)
    holdout_n = counts.get("spatial_holdout_nonca", 0)
    ca_train_n = counts.get("california_finetune_train", 0)
    ca_holdout_n = counts.get("california_holdout", 0)

    huc02_role_counts = (
        assignment_df.groupby(["HUC02", "split_role"]).size().unstack(fill_value=0).sort_index()
    )
    huc02_role_counts_dict = {
        str(huc02): {str(role): int(n) for role, n in row.items() if n > 0}
        for huc02, row in huc02_role_counts.iterrows()
    }

    missing_stratifier_staids = sorted(
        assignment_df.loc[
            assignment_df["assignment_reason"] == REASON_MISSING_STRATIFIER, "STAID"
        ].tolist()
    )

    manifest_pieces = {
        "algorithm_id": _ALGORITHM_ID,
        "algorithm_version": _ALGORITHM_VERSION,
        "seed": seed,
        "min_stratum_size": min_stratum_size,
        "nonca_holdout_fraction_target": nonca_holdout_fraction,
        "ca_holdout_fraction_target": ca_holdout_fraction,
        "tercile_edges": {
            "nonca_area": nonca_edges["area_edges"],
            "nonca_aridity": nonca_edges["aridity_edges"],
            "ca_area": ca_edges["area_edges"],
            "ca_aridity": ca_edges["aridity_edges"],
        },
        "fallback_log": nonca_fallback + ca_fallback,
        "missing_stratifier_basins": {
            "reason": REASON_MISSING_STRATIFIER,
            "staids": missing_stratifier_staids,
            "count": len(missing_stratifier_staids),
        },
        "huc02_role_counts": huc02_role_counts_dict,
        "counts": {
            "development_train": int(dev_n),
            "validation": int(dev_n),
            "temporal_test": int(dev_n),
            "spatial_holdout_nonca": int(holdout_n),
            "california_all": int(len(ca)),
            "california_finetune_train": int(ca_train_n),
            "california_holdout": int(ca_holdout_n),
        },
        "resulting_fractions": {
            "nonca_holdout_of_nonca": (holdout_n / len(nonca)) if len(nonca) else 0.0,
            "ca_holdout_of_ca": (ca_holdout_n / len(ca)) if len(ca) else 0.0,
        },
    }
    return assignment_df, manifest_pieces


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------

def write_split_artifacts(
    out_dir,
    eligible_all: list[str],
    assignment_df: pd.DataFrame,
    manifest: dict,
    force: bool = False,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise SplitGenerationError(
            f"output directory already exists and is non-empty: {out_dir} (use --force)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_list(name: str, staids) -> Path:
        p = out_dir / name
        p.write_text("\n".join(sorted(staids)) + "\n", encoding="utf-8")
        return p

    paths: dict[str, Path] = {}
    paths["eligible_basins_v001.txt"] = write_list("eligible_basins_v001.txt", eligible_all)

    dev = assignment_df.loc[assignment_df["split_role"] == "development_train", "STAID"].tolist()
    holdout = assignment_df.loc[assignment_df["split_role"] == "spatial_holdout_nonca", "STAID"].tolist()
    ca_all = assignment_df.loc[assignment_df["STATE"] == "CA", "STAID"].tolist()
    ca_train = assignment_df.loc[assignment_df["split_role"] == "california_finetune_train", "STAID"].tolist()
    ca_holdout = assignment_df.loc[assignment_df["split_role"] == "california_holdout", "STAID"].tolist()

    paths["development_train.txt"] = write_list("development_train.txt", dev)
    paths["validation.txt"] = write_list("validation.txt", dev)
    paths["temporal_test.txt"] = write_list("temporal_test.txt", dev)
    paths["spatial_holdout_nonca.txt"] = write_list("spatial_holdout_nonca.txt", holdout)
    paths["california_all.txt"] = write_list("california_all.txt", ca_all)
    paths["california_finetune_train.txt"] = write_list("california_finetune_train.txt", ca_train)
    paths["california_holdout.txt"] = write_list("california_holdout.txt", ca_holdout)

    assignment_path = out_dir / "split_assignment.csv"
    assignment_df.sort_values("STAID").to_csv(assignment_path, index=False)
    paths["split_assignment.csv"] = assignment_path

    # Checksums of every generated artifact except the manifest itself
    # (recorded inside the manifest so consumers can verify the bundle
    # without a separate sidecar file; non-circular by construction since
    # the manifest is written last and is not included in its own hash set).
    artifact_sha256 = {name: sha256_of(p) for name, p in sorted(paths.items())}

    manifest_path = out_dir / "split_manifest.json"
    manifest_to_write = {**manifest, "artifact_sha256": artifact_sha256}
    manifest_path.write_text(json.dumps(manifest_to_write, indent=2, default=str), encoding="utf-8")
    paths["split_manifest.json"] = manifest_path

    return paths

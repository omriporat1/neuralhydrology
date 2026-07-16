"""Independent auditor for the Stage 1 candidate spatial split (Milestone 2K-G-I I-A3).

This module is deliberately NOT built on top of src/baseline/splits.py. It
re-reads the canonical inputs (static-attribute matrix, eligible-basin list)
and the candidate artifact bundle from disk and independently recomputes every
fact it checks: population membership, California/non-California partition,
tercile edges and labels, initial-stratum construction, sparse-pool routing,
counts/fractions, and HUC02 summaries. It then compares those independent
results against the candidate's basin lists, assignment table, and
`split_manifest.json`, and separately verifies checksums/provenance and the
byte-level credibility of the existing repeat-generation directory.

Independence boundary (docs/stage1_baseline_package_implementation_plan.md
§8, refined 2K-G-I I-A3):
  - reused as-is: `src.baseline.staid.normalize_staid` (generic STAID string
    parsing, not split logic) and, at the CLI layer only,
    `src.baseline.policy.load_stage1_baseline_policy` (schema validation of
    the policy file, which does not compute any expected split assignment).
  - independently reimplemented here: matrix/eligible-list loading and
    joining, California/non-California classification, tercile-edge fitting
    and labeling, initial-stratum construction, sparse-pool pooling and
    forced-training routing, all counts/fractions/HUC02 summaries, and
    checksum/manifest reconciliation.
  - deliberately NOT done: replaying the generator's `numpy.random.Generator`
    draw to reproduce the exact holdout membership. Doing so would mean
    calling the same seeded RNG API the generator calls with the same seed --
    it would prove determinism (already proven more directly by the
    byte-identical repeat-directory comparison below) without proving
    scientific correctness of the route. Instead this auditor proves (a)
    every basin's route/pool membership was computed correctly from the
    source population under the binding policy, (b) every basin's assignment
    reason matches that independently recomputed route, (c) sparse/missing
    basins never appear in a holdout role, and (d) each stratum/pool's
    *count* of holdout members equals the policy's deterministic
    `round(fraction * n)` rule -- a non-random, independently checkable
    invariant that a wrong random draw size (as opposed to a wrong specific
    member) would violate.

v001-specific constants (known facts about this eligible universe, not
computed from the policy file): the five basins missing `ari_ix_uav`, the six
non-8-character STAIDs, the two excluded target gauges, and the static-matrix
sha256. These are the same values pinned in the Milestone 2K-G-I task
brief and in `docs/stage1_scientific_baseline_design.md`.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .staid import normalize_staid

__all__ = [
    "SplitAuditError",
    "CheckRecord",
    "AuditReport",
    "ExpectedRoute",
    "sha256_file",
    "read_staid_list_independent",
    "load_matrix_independent",
    "reconstruct_population",
    "fit_terciles",
    "tercile_label",
    "reconstruct_routes",
    "read_assignment_table_independent",
    "run_audit",
    "write_audit_outputs",
    "REASON_MISSING_STRATIFIER",
    "REASON_DIRECT_STRATUM_SAMPLE",
    "REASON_SPARSE_POOL_SAMPLE",
    "REASON_SPARSE_POOL_FORCED_TRAINING",
    "ALL_REASON_CODES",
    "V001_KNOWN_MISSING_ARIDITY_STAIDS",
    "V001_KNOWN_NONSTANDARD_STAIDS",
    "V001_EXCLUDED_TARGET_STAIDS",
    "V001_STATIC_MATRIX_SHA256",
    "EXPECTED_ARTIFACT_NAMES",
]


class SplitAuditError(ValueError):
    """Raised for auditor-input problems that make an audit impossible to run."""


# ---------------------------------------------------------------------------
# Reason vocabulary (fixed by the binding policy; re-declared here rather than
# imported from src.baseline.splits so the comparison is not tautological).
# ---------------------------------------------------------------------------
REASON_MISSING_STRATIFIER = "missing_hydroatlas_stratifier"
REASON_DIRECT_STRATUM_SAMPLE = "direct_stratum_sample"
REASON_SPARSE_POOL_SAMPLE = "sparse_pool_sample"
REASON_SPARSE_POOL_FORCED_TRAINING = "sparse_pool_forced_training"
ALL_REASON_CODES = frozenset({
    REASON_MISSING_STRATIFIER,
    REASON_DIRECT_STRATUM_SAMPLE,
    REASON_SPARSE_POOL_SAMPLE,
    REASON_SPARSE_POOL_FORCED_TRAINING,
})

# ---------------------------------------------------------------------------
# v001-specific known facts (docs/stage1_scientific_baseline_design.md;
# Milestone 2K-G-I I-A3 task brief). Not derived from the policy file.
# ---------------------------------------------------------------------------
V001_KNOWN_MISSING_ARIDITY_STAIDS = frozenset({
    "393109104464500",
    "394839104570300",
    "401733105392404",
    "402114105350101",
    "402913084285400",
})
V001_KNOWN_NONSTANDARD_STAIDS = frozenset({
    "103366092",
    "393109104464500",
    "394839104570300",
    "401733105392404",
    "402114105350101",
    "402913084285400",
})
V001_EXCLUDED_TARGET_STAIDS = frozenset({"02299472", "04073468"})
V001_STATIC_MATRIX_SHA256 = (
    "eb17aaa07c786a25291ceaf69e770bd54bda4bc22fbd1216a81734fa6882f464"
)

EXPECTED_ARTIFACT_NAMES = frozenset({
    "eligible_basins_v001.txt",
    "development_train.txt",
    "validation.txt",
    "temporal_test.txt",
    "spatial_holdout_nonca.txt",
    "california_all.txt",
    "california_finetune_train.txt",
    "california_holdout.txt",
    "split_assignment.csv",
    "split_manifest.json",
})

_HOLDOUT_ROLE_NONCA = "spatial_holdout_nonca"
_TRAIN_ROLE_NONCA = "development_train"
_HOLDOUT_ROLE_CA = "california_holdout"
_TRAIN_ROLE_CA = "california_finetune_train"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

@dataclass
class CheckRecord:
    severity: str  # "OK" | "WARNING" | "ERROR"
    check_id: str
    message: str = ""


@dataclass
class AuditReport:
    records: list = field(default_factory=list)

    def ok(self, check_id: str, message: str = "") -> None:
        self.records.append(CheckRecord("OK", check_id, message))

    def warn(self, check_id: str, message: str) -> None:
        self.records.append(CheckRecord("WARNING", check_id, message))

    def error(self, check_id: str, message: str) -> None:
        self.records.append(CheckRecord("ERROR", check_id, message))

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.records if r.severity == "ERROR")

    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.records if r.severity == "WARNING")

    @property
    def ok_count(self) -> int:
        return sum(1 for r in self.records if r.severity == "OK")

    @property
    def status(self) -> str:
        return "FAIL" if self.error_count > 0 else "PASS"

    def failed_messages(self) -> list:
        return [f"{r.check_id}: {r.message}" for r in self.records if r.severity == "ERROR"]


# ---------------------------------------------------------------------------
# Generic, independent low-level helpers
# ---------------------------------------------------------------------------

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pget(d: dict, dotted: str):
    node = d
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise SplitAuditError(f"policy dict missing required key: {dotted}")
        node = node[part]
    return node


def read_staid_list_independent(path) -> list:
    """Read a one-STAID-per-line file, in file order, normalized. No dedup."""
    p = Path(path)
    if not p.is_file():
        raise SplitAuditError(f"file not found: {p}")
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    out = []
    for ln in lines:
        try:
            out.append(normalize_staid(ln))
        except (TypeError, ValueError) as exc:
            raise SplitAuditError(f"malformed STAID {ln!r} in {p}: {exc}") from exc
    return out


def load_matrix_independent(path, required_columns) -> pd.DataFrame:
    """Independently load the static-attribute parquet indexed by gauge_id."""
    df = pd.read_parquet(path)
    if df.index.name == "gauge_id":
        idx = df.index
    elif "gauge_id" in df.columns:
        df = df.set_index("gauge_id")
        idx = df.index
    else:
        raise SplitAuditError("gauge_id not found as column or index name in matrix")

    if pd.api.types.is_numeric_dtype(idx):
        raise SplitAuditError(f"gauge_id dtype is {idx.dtype} (numeric) -- coercion suspected")

    normalized = []
    for v in idx:
        try:
            normalized.append(normalize_staid(str(v)))
        except (TypeError, ValueError) as exc:
            raise SplitAuditError(f"malformed gauge_id {v!r} in matrix: {exc}") from exc

    df = df.copy()
    df.index = pd.Index(normalized, name="gauge_id")
    if df.index.duplicated().any():
        dupes = df.index[df.index.duplicated()].unique().tolist()[:10]
        raise SplitAuditError(f"duplicate gauge_id values in matrix: {dupes}")

    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise SplitAuditError(f"matrix missing required columns: {missing_cols}")
    return df


def reconstruct_population(matrix_df: pd.DataFrame, eligible_ids) -> pd.DataFrame:
    """Independent one-to-one join of the eligible list against the matrix."""
    if len(eligible_ids) != len(set(eligible_ids)):
        dupes = sorted({s for s in eligible_ids if eligible_ids.count(s) > 1})
        raise SplitAuditError(f"duplicate STAIDs in eligible list: {dupes}")
    eligible_set = set(eligible_ids)
    missing = sorted(eligible_set - set(matrix_df.index))
    if missing:
        raise SplitAuditError(f"{len(missing)} eligible basin(s) missing from matrix: {missing[:10]}")
    joined = matrix_df.loc[matrix_df.index.isin(eligible_set)]
    if len(joined) != len(eligible_set):
        dup = joined.index[joined.index.duplicated()].unique().tolist()
        raise SplitAuditError(f"join is not one-to-one: matrix has multiple rows for {dup[:10]}")
    return joined.loc[sorted(eligible_set)].copy()


def read_assignment_table_independent(path) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise SplitAuditError(f"assignment table not found: {p}")
    rows = []
    with open(p, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"STAID", "split_role", "STATE", "HUC02", "assignment_reason"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SplitAuditError(f"assignment table missing columns: {sorted(missing)}")
        for row in reader:
            rows.append(row)
    df = pd.DataFrame(rows)
    if df["STAID"].duplicated().any():
        dup = df.loc[df["STAID"].duplicated(), "STAID"].tolist()
        raise SplitAuditError(f"duplicate STAID rows in assignment table: {dup[:10]}")
    return df.set_index("STAID", drop=False)


# ---------------------------------------------------------------------------
# Tercile fitting/labeling (independent numeric reimplementation)
# ---------------------------------------------------------------------------

def fit_terciles(values: pd.Series) -> tuple:
    if values.isnull().any():
        raise SplitAuditError("fit_terciles received null values")
    v = values.to_numpy(dtype=float)
    if v.size == 0:
        raise SplitAuditError("fit_terciles received an empty population")
    edges = np.quantile(v, [1 / 3, 2 / 3], method="linear")
    return float(edges[0]), float(edges[1])


def tercile_label(v: float, edges: tuple) -> str:
    e1, e2 = edges
    if v <= e1:
        return "low"
    if v <= e2:
        return "middle"
    return "high"


# ---------------------------------------------------------------------------
# Route reconstruction (independent stratification + sparse-pool routing)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExpectedRoute:
    staid: str
    route: str  # "missing" | "direct" | "sparse_pool" | "sparse_forced"
    expected_reason: str
    allowed_roles: frozenset
    stratum_key: Optional[tuple]
    stratum_size: Optional[int]
    pool_key: Optional[str]
    pool_size: Optional[int]


def reconstruct_routes(
    frame: pd.DataFrame,
    use_geography: bool,
    min_stratum_size: int,
    holdout_role: str,
    training_role: str,
):
    """Independently reconstruct every basin's expected route in one population.

    `frame` must already be restricted to exactly one population (non-CA or
    CA), indexed by STAID, with HUC02/DRAIN_SQKM/ari_ix_uav columns.
    Returns (routes: dict[str, ExpectedRoute], area_edges, aridity_edges).
    """
    routes: dict = {}

    has_aridity = frame["ari_ix_uav"].notnull()
    with_a = frame.loc[has_aridity]
    missing = frame.loc[~has_aridity]

    for staid in missing.index:
        routes[staid] = ExpectedRoute(
            staid=staid, route="missing",
            expected_reason=REASON_MISSING_STRATIFIER,
            allowed_roles=frozenset({training_role}),
            stratum_key=None, stratum_size=None, pool_key=None, pool_size=None,
        )

    if len(with_a) == 0:
        return routes, None, None

    area_edges = fit_terciles(with_a["DRAIN_SQKM"])
    aridity_edges = fit_terciles(with_a["ari_ix_uav"])

    area_class = {s: tercile_label(float(v), area_edges) for s, v in with_a["DRAIN_SQKM"].items()}
    hydro_class = {s: tercile_label(float(v), aridity_edges) for s, v in with_a["ari_ix_uav"].items()}

    strata = defaultdict(list)
    for staid in with_a.index:
        if use_geography:
            key = (str(frame.loc[staid, "HUC02"]), area_class[staid], hydro_class[staid])
        else:
            key = (area_class[staid], hydro_class[staid])
        strata[key].append(staid)

    sparse_items = []
    for key, staids in strata.items():
        n = len(staids)
        if n >= min_stratum_size:
            for s in staids:
                routes[s] = ExpectedRoute(
                    staid=s, route="direct",
                    expected_reason=REASON_DIRECT_STRATUM_SAMPLE,
                    allowed_roles=frozenset({holdout_role, training_role}),
                    stratum_key=key, stratum_size=n, pool_key=None, pool_size=None,
                )
        else:
            for s in staids:
                sparse_items.append((s, key))

    if sparse_items:
        if use_geography:
            pools = defaultdict(list)
            for s, key in sparse_items:
                pools[key[0]].append((s, key))
            pool_iter = list(pools.items())
        else:
            pool_iter = [("ALL", sparse_items)]

        for pool_key, items in pool_iter:
            pool_size = len(items)
            for s, key in items:
                if pool_size >= min_stratum_size:
                    routes[s] = ExpectedRoute(
                        staid=s, route="sparse_pool",
                        expected_reason=REASON_SPARSE_POOL_SAMPLE,
                        allowed_roles=frozenset({holdout_role, training_role}),
                        stratum_key=key, stratum_size=len(strata[key]),
                        pool_key=str(pool_key), pool_size=pool_size,
                    )
                else:
                    routes[s] = ExpectedRoute(
                        staid=s, route="sparse_forced",
                        expected_reason=REASON_SPARSE_POOL_FORCED_TRAINING,
                        allowed_roles=frozenset({training_role}),
                        stratum_key=key, stratum_size=len(strata[key]),
                        pool_key=str(pool_key), pool_size=pool_size,
                    )

    return routes, area_edges, aridity_edges


def _group_holdout_count_checks(report: AuditReport, routes: dict, actual_role_by_staid: dict,
                                 holdout_role: str, fraction: float, label: str) -> None:
    """Verify each direct/sparse_pool group's holdout COUNT (not membership)
    equals the policy's deterministic round(fraction * n) rule."""
    groups = defaultdict(list)
    for staid, r in routes.items():
        if r.route == "direct":
            groups[("direct", r.stratum_key)].append(staid)
        elif r.route == "sparse_pool":
            groups[("sparse_pool", r.pool_key)].append(staid)
    for gkey, staids in groups.items():
        n = len(staids)
        expected_k = int(round(fraction * n))
        actual_k = sum(1 for s in staids if actual_role_by_staid.get(s) == holdout_role)
        if actual_k != expected_k:
            report.error(
                "sparse_and_direct_group_holdout_count",
                f"{label} group {gkey}: expected holdout count round({fraction}*{n})={expected_k}, "
                f"observed {actual_k}",
            )
        else:
            report.ok("sparse_and_direct_group_holdout_count", f"{label} group {gkey}: n={n} k={actual_k}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_audit(
    *,
    policy: dict,
    policy_path,
    candidate_dir,
    repeat_dir,
    attributes_parquet,
    eligible_basins,
    forcing_basins=None,
):
    """Run the full independent audit; returns (AuditReport, diagnostics dict).

    `policy` is an already-loaded policy mapping (the CLI loads/validates it
    via src.baseline.policy.load_stage1_baseline_policy; tests may pass a
    hand-built minimal dict with just the keys this function reads).
    """
    report = AuditReport()
    diagnostics: dict = {}

    candidate_dir = Path(candidate_dir)
    repeat_dir = Path(repeat_dir) if repeat_dir is not None else None
    policy_path = Path(policy_path)
    attributes_parquet = Path(attributes_parquet)
    eligible_basins = Path(eligible_basins)

    required_matrix_columns = ["STATE", "HUC02", "DRAIN_SQKM", "ari_ix_uav"]

    # ---- checksums / provenance (inputs) ----
    policy_sha256 = sha256_file(policy_path)
    matrix_sha256 = sha256_file(attributes_parquet)
    eligible_sha256 = sha256_file(eligible_basins)
    diagnostics["policy_path"] = str(policy_path)
    diagnostics["policy_sha256"] = policy_sha256
    diagnostics["attributes_parquet_sha256"] = matrix_sha256
    diagnostics["eligible_basins_sha256"] = eligible_sha256

    expected_matrix_sha256 = _pget(policy, "static_attributes.sha256")
    if matrix_sha256 != expected_matrix_sha256:
        report.error(
            "static_matrix_checksum_vs_policy",
            f"attributes parquet sha256 mismatch: expected {expected_matrix_sha256}, got {matrix_sha256}",
        )
    else:
        report.ok("static_matrix_checksum_vs_policy")
    # Advisory only: the authoritative correctness gate is
    # static_matrix_checksum_vs_policy above. This is a redundant sanity
    # cross-check against the known-good v001 production checksum and is
    # expected to differ for non-v001 inputs (e.g. synthetic test fixtures
    # or a future v002 dataset), so it never fails the audit outright.
    if matrix_sha256 != V001_STATIC_MATRIX_SHA256:
        report.warn(
            "static_matrix_checksum_vs_v001_constant",
            f"attributes parquet sha256 does not match known v001 constant: expected {V001_STATIC_MATRIX_SHA256}, got {matrix_sha256}",
        )
    else:
        report.ok("static_matrix_checksum_vs_v001_constant")

    # ---- independent loading ----
    try:
        eligible_ids = read_staid_list_independent(eligible_basins)
        matrix_df = load_matrix_independent(attributes_parquet, required_matrix_columns)
        population = reconstruct_population(matrix_df, eligible_ids)
    except SplitAuditError as exc:
        report.error("input_loading", str(exc))
        diagnostics["fatal_error"] = str(exc)
        return report, diagnostics

    eligible_set = set(eligible_ids)
    expected_eligible_count = _pget(policy, "basin_universe.expected_eligible_count")
    expected_nonca_count = _pget(policy, "basin_universe.expected_nonca_count")
    expected_ca_count = _pget(policy, "basin_universe.expected_ca_count")
    excluded_staids = set(_pget(policy, "basin_universe.excluded_staids"))

    # ---- universe and identity ----
    if len(eligible_set) != expected_eligible_count:
        report.error(
            "eligible_universe_count",
            f"eligible count {len(eligible_set)} != policy expected {expected_eligible_count}",
        )
    else:
        report.ok("eligible_universe_count", str(len(eligible_set)))

    ca_mask = population["STATE"].astype(str) == "CA"
    nonca_pop = population.loc[~ca_mask]
    ca_pop = population.loc[ca_mask]

    if len(nonca_pop) != expected_nonca_count:
        report.error("nonca_universe_count", f"{len(nonca_pop)} != policy expected {expected_nonca_count}")
    else:
        report.ok("nonca_universe_count", str(len(nonca_pop)))
    if len(ca_pop) != expected_ca_count:
        report.error("ca_universe_count", f"{len(ca_pop)} != policy expected {expected_ca_count}")
    else:
        report.ok("ca_universe_count", str(len(ca_pop)))

    bad_excluded = excluded_staids & eligible_set
    if bad_excluded:
        report.error("excluded_target_gauges_absent", f"excluded gauge(s) present in eligible set: {sorted(bad_excluded)}")
    else:
        report.ok("excluded_target_gauges_absent")

    v001_bad_excluded = V001_EXCLUDED_TARGET_STAIDS & eligible_set
    if v001_bad_excluded:
        report.error("v001_excluded_target_gauges_absent", f"v001 excluded gauge(s) present: {sorted(v001_bad_excluded)}")
    else:
        report.ok("v001_excluded_target_gauges_absent")

    missing_nonstd = V001_KNOWN_NONSTANDARD_STAIDS - eligible_set
    if missing_nonstd:
        report.error("v001_nonstandard_staids_present", f"missing/altered known nonstandard STAID(s): {sorted(missing_nonstd)}")
    else:
        report.ok("v001_nonstandard_staids_present")

    if forcing_basins is not None:
        forcing_ids = read_staid_list_independent(forcing_basins)
        forcing_set = set(forcing_ids)
        if forcing_set != eligible_set:
            sym_diff = sorted(forcing_set ^ eligible_set)[:10]
            report.error("eligible_forcing_agreement", f"eligible/forcing set mismatch, sample diff: {sym_diff}")
        else:
            report.ok("eligible_forcing_agreement")
    else:
        report.warn("eligible_forcing_agreement", "forcing basin list not supplied; check skipped")

    # ---- missing aridity (independent) ----
    missing_aridity_set = set(population.index[population["ari_ix_uav"].isnull()])
    if missing_aridity_set != V001_KNOWN_MISSING_ARIDITY_STAIDS:
        report.error(
            "missing_aridity_set_matches_known_v001",
            f"expected {sorted(V001_KNOWN_MISSING_ARIDITY_STAIDS)}, got {sorted(missing_aridity_set)}",
        )
    else:
        report.ok("missing_aridity_set_matches_known_v001")
    ca_missing_aridity = missing_aridity_set & set(ca_pop.index)
    if ca_missing_aridity:
        report.error("missing_aridity_all_nonca", f"missing-aridity basin(s) found in CA: {sorted(ca_missing_aridity)}")
    else:
        report.ok("missing_aridity_all_nonca")

    # ---- route reconstruction ----
    min_stratum_size = _pget(policy, "spatial_split.min_composite_stratum_size")
    seed = _pget(policy, "spatial_split.seed")
    nonca_holdout_fraction = _pget(policy, "spatial_split.nonca_holdout_fraction")
    ca_holdout_fraction = 1.0 - _pget(policy, "spatial_split.california_finetune_fraction")

    if seed != 42:
        report.error("seed_is_42", f"policy seed is {seed}, expected 42")
    else:
        report.ok("seed_is_42")

    nonca_routes, nonca_area_edges, nonca_aridity_edges = reconstruct_routes(
        nonca_pop, use_geography=True, min_stratum_size=min_stratum_size,
        holdout_role=_HOLDOUT_ROLE_NONCA, training_role=_TRAIN_ROLE_NONCA,
    )
    ca_routes, ca_area_edges, ca_aridity_edges = reconstruct_routes(
        ca_pop, use_geography=False, min_stratum_size=min_stratum_size,
        holdout_role=_HOLDOUT_ROLE_CA, training_role=_TRAIN_ROLE_CA,
    )
    all_routes = {**nonca_routes, **ca_routes}

    # ---- read candidate artifacts independently ----
    try:
        cand_eligible = read_staid_list_independent(candidate_dir / "eligible_basins_v001.txt")
        cand_dev = read_staid_list_independent(candidate_dir / "development_train.txt")
        cand_val = read_staid_list_independent(candidate_dir / "validation.txt")
        cand_test = read_staid_list_independent(candidate_dir / "temporal_test.txt")
        cand_holdout = read_staid_list_independent(candidate_dir / "spatial_holdout_nonca.txt")
        cand_ca_all = read_staid_list_independent(candidate_dir / "california_all.txt")
        cand_ca_train = read_staid_list_independent(candidate_dir / "california_finetune_train.txt")
        cand_ca_holdout = read_staid_list_independent(candidate_dir / "california_holdout.txt")
        assignment_df = read_assignment_table_independent(candidate_dir / "split_assignment.csv")
        manifest = json.loads((candidate_dir / "split_manifest.json").read_text(encoding="utf-8"))
    except SplitAuditError as exc:
        report.error("candidate_artifact_loading", str(exc))
        diagnostics["fatal_error"] = str(exc)
        return report, diagnostics

    dev_set, val_set, test_set = set(cand_dev), set(cand_val), set(cand_test)
    holdout_set = set(cand_holdout)
    ca_all_set, ca_train_set, ca_holdout_set = set(cand_ca_all), set(cand_ca_train), set(cand_ca_holdout)
    cand_eligible_set = set(cand_eligible)

    # ---- basin-list algebra ----
    if cand_eligible_set != eligible_set:
        report.error("candidate_eligible_matches_independent", "candidate eligible_basins_v001.txt does not match independently loaded eligible list")
    else:
        report.ok("candidate_eligible_matches_independent")

    dev_sha = sha256_file(candidate_dir / "development_train.txt")
    val_sha = sha256_file(candidate_dir / "validation.txt")
    test_sha = sha256_file(candidate_dir / "temporal_test.txt")
    if not (dev_sha == val_sha == test_sha):
        report.error("temporal_lists_byte_identical", f"dev={dev_sha} val={val_sha} test={test_sha}")
    else:
        report.ok("temporal_lists_byte_identical")
    if not (dev_set == val_set == test_set):
        report.error("temporal_lists_set_identical", "parsed sets of development/validation/temporal_test differ")
    else:
        report.ok("temporal_lists_set_identical")

    nonca_id_set = set(nonca_pop.index)
    ca_id_set = set(ca_pop.index)

    for name, s in [("development_train", dev_set), ("validation", val_set), ("temporal_test", test_set)]:
        leaked = s & ca_id_set
        if leaked:
            report.error("stage1_3_no_california_leakage", f"{name} contains California basin(s): {sorted(leaked)[:10]}")
        else:
            report.ok("stage1_3_no_california_leakage", name)

    if not dev_set.isdisjoint(holdout_set):
        report.error("dev_holdout_disjoint", f"overlap: {sorted(dev_set & holdout_set)[:10]}")
    else:
        report.ok("dev_holdout_disjoint")
    if (dev_set | holdout_set) != nonca_id_set:
        missing_ = sorted(nonca_id_set - (dev_set | holdout_set))[:10]
        extra_ = sorted((dev_set | holdout_set) - nonca_id_set)[:10]
        report.error("dev_holdout_union_is_nonca", f"missing={missing_} extra={extra_}")
    else:
        report.ok("dev_holdout_union_is_nonca")

    if not ca_train_set.isdisjoint(ca_holdout_set):
        report.error("ca_train_holdout_disjoint", f"overlap: {sorted(ca_train_set & ca_holdout_set)[:10]}")
    else:
        report.ok("ca_train_holdout_disjoint")
    if (ca_train_set | ca_holdout_set) != ca_all_set:
        report.error("ca_train_holdout_union_is_ca_all", "union of CA fine-tune/holdout != california_all")
    else:
        report.ok("ca_train_holdout_union_is_ca_all")
    if ca_all_set != ca_id_set:
        missing_ = sorted(ca_id_set - ca_all_set)[:10]
        extra_ = sorted(ca_all_set - ca_id_set)[:10]
        report.error("ca_all_matches_independent_ca_population", f"missing={missing_} extra={extra_}")
    else:
        report.ok("ca_all_matches_independent_ca_population")

    nonca_leak_into_ca = (dev_set | holdout_set) & ca_all_set
    if nonca_leak_into_ca:
        report.error("no_nonca_leakage_into_ca_lists", f"basin(s): {sorted(nonca_leak_into_ca)[:10]}")
    else:
        report.ok("no_nonca_leakage_into_ca_lists")

    # ---- assignment-table agreement ----
    table_ids = set(assignment_df.index)
    if table_ids != eligible_set:
        missing_ = sorted(eligible_set - table_ids)[:10]
        extra_ = sorted(table_ids - eligible_set)[:10]
        report.error("assignment_table_one_row_per_basin", f"missing={missing_} extra={extra_}")
    else:
        report.ok("assignment_table_one_row_per_basin")

    role_by_staid = assignment_df["split_role"].to_dict()
    list_role_map = {}
    for s in dev_set:
        list_role_map[s] = _TRAIN_ROLE_NONCA
    for s in holdout_set:
        list_role_map[s] = _HOLDOUT_ROLE_NONCA
    for s in ca_train_set:
        list_role_map[s] = _TRAIN_ROLE_CA
    for s in ca_holdout_set:
        list_role_map[s] = _HOLDOUT_ROLE_CA

    role_mismatches = []
    for staid, table_role in role_by_staid.items():
        list_role = list_role_map.get(staid)
        if list_role is None:
            role_mismatches.append((staid, table_role, None))
        elif list_role != table_role:
            role_mismatches.append((staid, table_role, list_role))
    if role_mismatches:
        report.error("table_role_matches_list_membership", f"{len(role_mismatches)} mismatch(es), e.g. {role_mismatches[:5]}")
    else:
        report.ok("table_role_matches_list_membership")

    bad_state_designation = []
    for staid, row in assignment_df.iterrows():
        expected_state_is_ca = staid in ca_id_set
        actual_is_ca = str(row["STATE"]) == "CA"
        if expected_state_is_ca != actual_is_ca:
            bad_state_designation.append(staid)
    if bad_state_designation:
        report.error("table_state_matches_source", f"{len(bad_state_designation)} basin(s), e.g. {bad_state_designation[:5]}")
    else:
        report.ok("table_state_matches_source")

    reason_values = set(assignment_df["assignment_reason"])
    bad_reasons = reason_values - ALL_REASON_CODES
    if bad_reasons:
        report.error("reason_codes_from_signed_set", f"unknown reason code(s): {sorted(bad_reasons)}")
    else:
        report.ok("reason_codes_from_signed_set")

    # ---- route/reason agreement (the scientific core of the audit) ----
    reason_mismatches = []
    role_not_allowed = []
    for staid, table_role in role_by_staid.items():
        route = all_routes.get(staid)
        if route is None:
            continue  # already reported as an omission above
        table_reason = assignment_df.loc[staid, "assignment_reason"]
        if table_reason != route.expected_reason:
            reason_mismatches.append((staid, table_reason, route.expected_reason))
        if table_role not in route.allowed_roles:
            role_not_allowed.append((staid, table_role, sorted(route.allowed_roles)))
    if reason_mismatches:
        report.error("assignment_reason_matches_recomputed_route", f"{len(reason_mismatches)} mismatch(es), e.g. {reason_mismatches[:5]}")
    else:
        report.ok("assignment_reason_matches_recomputed_route")
    if role_not_allowed:
        report.error("role_within_route_allowed_roles", f"{len(role_not_allowed)} violation(s), e.g. {role_not_allowed[:5]}")
    else:
        report.ok("role_within_route_allowed_roles")

    _group_holdout_count_checks(report, nonca_routes, role_by_staid, _HOLDOUT_ROLE_NONCA, nonca_holdout_fraction, "non-CA")
    _group_holdout_count_checks(report, ca_routes, role_by_staid, _HOLDOUT_ROLE_CA, ca_holdout_fraction, "CA")

    # sparse/missing never in holdout (explicit, redundant with role_not_allowed but
    # reported under its own check_id per the task's required-checks list)
    forbidden_holdout = [
        s for s, r in all_routes.items()
        if r.route in ("sparse_forced", "missing")
        and role_by_staid.get(s) in (_HOLDOUT_ROLE_NONCA, _HOLDOUT_ROLE_CA)
    ]
    if forbidden_holdout:
        report.error("sparse_and_missing_never_in_holdout", f"basin(s): {forbidden_holdout[:10]}")
    else:
        report.ok("sparse_and_missing_never_in_holdout")

    # HUC02 09 zero-holdout special case
    huc02_09_ids = set(nonca_pop.index[nonca_pop["HUC02"].astype(str) == "09"])
    huc02_09_holdout = [s for s in huc02_09_ids if role_by_staid.get(s) == _HOLDOUT_ROLE_NONCA]
    if huc02_09_holdout:
        report.error("huc02_09_zero_holdout", f"HUC02 09 basin(s) in holdout: {huc02_09_holdout}")
    else:
        report.ok("huc02_09_zero_holdout")

    # ---- counts, fractions, HUC02 summaries ----
    recomputed_counts = {
        "development_train": len(dev_set),
        "validation": len(val_set),
        "temporal_test": len(test_set),
        "spatial_holdout_nonca": len(holdout_set),
        "california_all": len(ca_all_set),
        "california_finetune_train": len(ca_train_set),
        "california_holdout": len(ca_holdout_set),
    }
    diagnostics["recomputed_counts"] = recomputed_counts

    nonca_holdout_frac = len(holdout_set) / len(nonca_id_set) if nonca_id_set else 0.0
    ca_holdout_frac = len(ca_holdout_set) / len(ca_id_set) if ca_id_set else 0.0
    diagnostics["recomputed_fractions"] = {
        "nonca_holdout_of_nonca": nonca_holdout_frac,
        "ca_holdout_of_ca": ca_holdout_frac,
    }
    if not (0.08 <= nonca_holdout_frac <= 0.12):
        report.error("nonca_holdout_fraction_in_band", f"{nonca_holdout_frac:.4f} outside [0.08, 0.12]")
    else:
        report.ok("nonca_holdout_fraction_in_band", f"{nonca_holdout_frac:.4f}")
    if not (0.08 <= ca_holdout_frac <= 0.12):
        report.error("ca_holdout_fraction_in_band", f"{ca_holdout_frac:.4f} outside [0.08, 0.12]")
    else:
        report.ok("ca_holdout_fraction_in_band", f"{ca_holdout_frac:.4f}")

    reason_counts = assignment_df["assignment_reason"].value_counts().to_dict()
    diagnostics["reason_counts"] = reason_counts

    huc02_role_counts = defaultdict(lambda: defaultdict(int))
    for staid, row in assignment_df.iterrows():
        huc02_role_counts[str(row["HUC02"])][row["split_role"]] += 1
    huc02_role_counts = {k: dict(v) for k, v in huc02_role_counts.items()}
    diagnostics["huc02_role_counts"] = huc02_role_counts

    zero_holdout_huc02 = sorted(
        h for h, roles in huc02_role_counts.items()
        if _HOLDOUT_ROLE_NONCA not in roles and _TRAIN_ROLE_NONCA in roles
    )
    diagnostics["zero_holdout_huc02_nonca"] = zero_holdout_huc02
    unexpected_zero_holdout = [h for h in zero_holdout_huc02 if h != "09"]
    if unexpected_zero_holdout:
        report.warn("zero_holdout_huc02_surfaced", f"HUC02(s) besides '09' with zero non-CA holdout: {unexpected_zero_holdout}")
    else:
        report.ok("zero_holdout_huc02_surfaced", str(zero_holdout_huc02))

    diagnostics["missing_stratifier_staids"] = sorted(missing_aridity_set)

    # ---- manifest reconciliation ----
    if manifest.get("seed") != 42:
        report.error("manifest_seed", f"manifest seed {manifest.get('seed')!r} != 42")
    else:
        report.ok("manifest_seed")
    if manifest.get("algorithm_version") != 1:
        report.error("manifest_algorithm_version", f"manifest algorithm_version {manifest.get('algorithm_version')!r} != 1 (recognized)")
    else:
        report.ok("manifest_algorithm_version")

    manifest_policy_sha = manifest.get("policy_sha256")
    if manifest_policy_sha != policy_sha256:
        report.error("manifest_policy_sha256", f"manifest {manifest_policy_sha!r} != recomputed {policy_sha256!r}")
    else:
        report.ok("manifest_policy_sha256")
    manifest_matrix_sha = manifest.get("attributes_parquet_sha256")
    if manifest_matrix_sha != matrix_sha256:
        report.error("manifest_matrix_sha256", f"manifest {manifest_matrix_sha!r} != recomputed {matrix_sha256!r}")
    else:
        report.ok("manifest_matrix_sha256")
    manifest_eligible_sha = manifest.get("eligible_basins_sha256")
    if manifest_eligible_sha != eligible_sha256:
        report.error("manifest_eligible_sha256", f"manifest {manifest_eligible_sha!r} != recomputed {eligible_sha256!r}")
    else:
        report.ok("manifest_eligible_sha256")

    manifest_counts = manifest.get("counts", {})
    if manifest_counts != recomputed_counts:
        report.error("manifest_counts_match_recomputed", f"manifest={manifest_counts} recomputed={recomputed_counts}")
    else:
        report.ok("manifest_counts_match_recomputed")

    manifest_fracs = manifest.get("resulting_fractions", {})
    frac_ok = (
        math.isclose(manifest_fracs.get("nonca_holdout_of_nonca", float("nan")), nonca_holdout_frac, rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(manifest_fracs.get("ca_holdout_of_ca", float("nan")), ca_holdout_frac, rel_tol=1e-9, abs_tol=1e-12)
    )
    if not frac_ok:
        report.error("manifest_fractions_match_recomputed", f"manifest={manifest_fracs} recomputed={diagnostics['recomputed_fractions']}")
    else:
        report.ok("manifest_fractions_match_recomputed")

    manifest_edges = manifest.get("tercile_edges", {})

    def _edges_close(a, b) -> bool:
        if a is None or b is None:
            return a == b
        return math.isclose(a[0], b[0], rel_tol=1e-9, abs_tol=1e-9) and math.isclose(a[1], b[1], rel_tol=1e-9, abs_tol=1e-9)

    edge_pairs = [
        ("nonca_area", nonca_area_edges), ("nonca_aridity", nonca_aridity_edges),
        ("ca_area", ca_area_edges), ("ca_aridity", ca_aridity_edges),
    ]
    edge_mismatches = []
    for name, recomputed_edges in edge_pairs:
        manifest_edge = manifest_edges.get(name)
        manifest_edge_t = tuple(manifest_edge) if manifest_edge is not None else None
        if not _edges_close(manifest_edge_t, recomputed_edges):
            edge_mismatches.append((name, manifest_edge_t, recomputed_edges))
    if edge_mismatches:
        report.error("manifest_tercile_edges_match_recomputed", str(edge_mismatches))
    else:
        report.ok("manifest_tercile_edges_match_recomputed")

    manifest_missing = manifest.get("missing_stratifier_basins", {})
    if set(manifest_missing.get("staids", [])) != missing_aridity_set or manifest_missing.get("count") != len(missing_aridity_set):
        report.error("manifest_missing_stratifier_matches_recomputed", f"manifest={manifest_missing} recomputed={sorted(missing_aridity_set)}")
    else:
        report.ok("manifest_missing_stratifier_matches_recomputed")

    manifest_huc02 = manifest.get("huc02_role_counts", {})
    if manifest_huc02 != huc02_role_counts:
        report.error("manifest_huc02_role_counts_match_recomputed", "mismatch between manifest huc02_role_counts and recomputation")
    else:
        report.ok("manifest_huc02_role_counts_match_recomputed")

    manifest_fallback = manifest.get("fallback_log", [])
    expected_forced = {
        s: {"pool_key": r.pool_key, "pool_size": r.pool_size}
        for s, r in all_routes.items() if r.route == "sparse_forced"
    }
    manifest_forced_staids = {e["staid"] for e in manifest_fallback}
    if manifest_forced_staids != set(expected_forced.keys()):
        report.error(
            "manifest_fallback_log_matches_recomputed_sparse_forced",
            f"missing={sorted(set(expected_forced) - manifest_forced_staids)[:10]} "
            f"extra={sorted(manifest_forced_staids - set(expected_forced))[:10]}",
        )
    else:
        pool_mismatches = []
        for e in manifest_fallback:
            exp = expected_forced[e["staid"]]
            if str(e.get("pool_key")) != str(exp["pool_key"]) or e.get("pool_size") != exp["pool_size"]:
                pool_mismatches.append(e["staid"])
        if pool_mismatches:
            report.error("manifest_fallback_log_matches_recomputed_sparse_forced", f"pool mismatch for: {pool_mismatches[:10]}")
        else:
            report.ok("manifest_fallback_log_matches_recomputed_sparse_forced")

    # ---- artifact checksum inventory ----
    actual_files = {p.name for p in candidate_dir.iterdir() if p.is_file()}
    if actual_files != EXPECTED_ARTIFACT_NAMES:
        report.error(
            "candidate_artifact_inventory_complete",
            f"missing={sorted(EXPECTED_ARTIFACT_NAMES - actual_files)} extra={sorted(actual_files - EXPECTED_ARTIFACT_NAMES)}",
        )
    else:
        report.ok("candidate_artifact_inventory_complete")

    manifest_artifact_sha = manifest.get("artifact_sha256", {})
    expected_artifact_set = EXPECTED_ARTIFACT_NAMES - {"split_manifest.json"}
    if set(manifest_artifact_sha.keys()) != expected_artifact_set:
        report.error(
            "manifest_artifact_inventory_complete_noncircular",
            f"missing={sorted(expected_artifact_set - set(manifest_artifact_sha))} "
            f"extra_or_self={sorted(set(manifest_artifact_sha) - expected_artifact_set)}",
        )
    else:
        report.ok("manifest_artifact_inventory_complete_noncircular")

    checksum_mismatches = []
    for name, recorded_sha in manifest_artifact_sha.items():
        fpath = candidate_dir / name
        if not fpath.is_file():
            checksum_mismatches.append((name, "file_missing", recorded_sha))
            continue
        actual_sha = sha256_file(fpath)
        if actual_sha != recorded_sha:
            checksum_mismatches.append((name, actual_sha, recorded_sha))
    if checksum_mismatches:
        report.error("candidate_artifact_checksums_match_manifest", str(checksum_mismatches[:10]))
    else:
        report.ok("candidate_artifact_checksums_match_manifest")

    # ---- repeat-generation evidence ----
    if repeat_dir is not None:
        repeat_result = _audit_repeat_directory(report, candidate_dir, repeat_dir)
        diagnostics["repeat_comparison"] = repeat_result
    else:
        report.warn("repeat_directory_comparison", "no repeat directory supplied; check skipped")
        diagnostics["repeat_comparison"] = None

    diagnostics["candidate_dir"] = str(candidate_dir)
    diagnostics["repeat_dir"] = str(repeat_dir) if repeat_dir is not None else None
    return report, diagnostics


def _audit_repeat_directory(report: AuditReport, candidate_dir: Path, repeat_dir: Path) -> dict:
    result = {"per_file_match": {}, "all_match": False}
    if not repeat_dir.is_dir():
        report.error("repeat_directory_exists", f"repeat directory not found: {repeat_dir}")
        return result

    repeat_files = {p.name for p in repeat_dir.iterdir() if p.is_file()}
    if repeat_files != EXPECTED_ARTIFACT_NAMES:
        report.error(
            "repeat_artifact_inventory_complete",
            f"missing={sorted(EXPECTED_ARTIFACT_NAMES - repeat_files)} extra={sorted(repeat_files - EXPECTED_ARTIFACT_NAMES)}",
        )
    else:
        report.ok("repeat_artifact_inventory_complete")

    empty_files = [name for name in EXPECTED_ARTIFACT_NAMES if (repeat_dir / name).is_file() and (repeat_dir / name).stat().st_size == 0]
    if empty_files:
        report.error("repeat_artifacts_nonempty", f"empty file(s) in repeat dir: {empty_files}")
    else:
        report.ok("repeat_artifacts_nonempty")

    all_match = True
    for name in sorted(EXPECTED_ARTIFACT_NAMES):
        cand_path = candidate_dir / name
        rep_path = repeat_dir / name
        if not cand_path.is_file() or not rep_path.is_file():
            result["per_file_match"][name] = False
            all_match = False
            continue
        match = sha256_file(cand_path) == sha256_file(rep_path)
        result["per_file_match"][name] = match
        if not match:
            all_match = False
    result["all_match"] = all_match
    if not all_match:
        mismatched = [n for n, ok in result["per_file_match"].items() if not ok]
        report.error("repeat_candidate_byte_identical", f"mismatched file(s): {mismatched}")
    else:
        report.ok("repeat_candidate_byte_identical")

    if (candidate_dir / "split_manifest.json").is_file() and (repeat_dir / "split_manifest.json").is_file():
        manifest_match = sha256_file(candidate_dir / "split_manifest.json") == sha256_file(repeat_dir / "split_manifest.json")
        if not manifest_match:
            report.error("repeat_manifest_byte_identical", "split_manifest.json differs between candidate and repeat (possible volatile field)")
        else:
            report.ok("repeat_manifest_byte_identical")

    return result


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_audit_outputs(out_dir, report: AuditReport, diagnostics: dict) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "status": report.status,
        "error_count": report.error_count,
        "warning_count": report.warning_count,
        "ok_count": report.ok_count,
        "candidate_path": diagnostics.get("candidate_dir"),
        "repeat_path": diagnostics.get("repeat_dir"),
        "policy_path": diagnostics.get("policy_path"),
        "policy_sha256": diagnostics.get("policy_sha256"),
        "attributes_parquet_sha256": diagnostics.get("attributes_parquet_sha256"),
        "eligible_basins_sha256": diagnostics.get("eligible_basins_sha256"),
        "recomputed_counts": diagnostics.get("recomputed_counts"),
        "recomputed_fractions": diagnostics.get("recomputed_fractions"),
        "reason_counts": diagnostics.get("reason_counts"),
        "missing_stratifier_staids": diagnostics.get("missing_stratifier_staids"),
        "huc02_role_counts": diagnostics.get("huc02_role_counts"),
        "zero_holdout_huc02_nonca": diagnostics.get("zero_holdout_huc02_nonca"),
        "repeat_comparison": diagnostics.get("repeat_comparison"),
        "failed_checks": report.failed_messages(),
    }
    summary_path = out_dir / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    csv_path = out_dir / "audit_checks.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["severity", "check_id", "message"])
        for r in report.records:
            writer.writerow([r.severity, r.check_id, r.message])

    md_lines = [
        f"# Stage 1 baseline split audit ({report.status})",
        "",
        f"- errors: {report.error_count}",
        f"- warnings: {report.warning_count}",
        f"- ok: {report.ok_count}",
        f"- candidate: {diagnostics.get('candidate_dir')}",
        f"- repeat: {diagnostics.get('repeat_dir')}",
        "",
        "## Recomputed counts",
        "```json",
        json.dumps(diagnostics.get("recomputed_counts"), indent=2),
        "```",
        "",
        "## Recomputed fractions",
        "```json",
        json.dumps(diagnostics.get("recomputed_fractions"), indent=2),
        "```",
        "",
    ]
    if report.error_count:
        md_lines.append("## Failed checks")
        for msg in report.failed_messages():
            md_lines.append(f"- {msg}")
    md_path = out_dir / "audit_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {"audit_summary.json": summary_path, "audit_checks.csv": csv_path, "audit_summary.md": md_path}

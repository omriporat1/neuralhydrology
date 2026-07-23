"""Independent auditor for the Stage 1 Compact Scientific Package (Gate 4).

This module re-derives and re-verifies every scientific and structural claim
made by the Compact Scientific Package builder, from raw sources, without
calling the builder's own code. The point of an independent auditor is that a
bug shared between construction and verification cannot pass unnoticed: if the
same buggy formula produced both the package and its own self-check, both
would agree and the bug would ship. This module therefore re-expresses the
package's scientific contract (unit conversion, lead-target shifting, gap
timeline reconstruction) using direct pandas/numpy/xarray/netCDF4 operations,
never by importing the modules under audit.

Independence boundary
----------------------
NOT imported, anywhere in this module (by design):
  - src.baseline.package_builder (and its private validation helpers)
  - src.baseline.package_assembly
  - src.baseline.package_netcdf
  - src.baseline.units (the m3/s -> mm/h conversion is re-derived locally,
    see :func:`independent_discharge_to_runoff_mm_per_h`)
  - src.baseline.lead_targets (the lead-shift logic is re-derived locally,
    see :func:`independent_lead_shift`)
  - src.baseline.gap_mask_io (the gap-inventory CSV is re-parsed and
    re-filtered locally, see :func:`load_gap_inventory_independent` /
    :func:`reconstruct_gap_timestamps_independent`)
  - src.baseline.splits (sha256_of is re-declared locally as
    :func:`sha256_file`)

Reused, with justification (neutral, non-scientific helpers only):
  - src.baseline.staid.normalize_staid -- strict basin-ID string validation
    (zero-padding / length rules). This is a format-validation utility, not a
    computation whose correctness this audit exists to check, and reusing it
    keeps basin-ID acceptance rules consistent across the codebase rather than
    risking silent drift between two hand-written copies.

Everything else -- directory layout, checksums, NetCDF dimensions/dtypes/
units/timeline, forcing/qobs value comparison, the m3/s -> mm/h conversion,
the lead-target shift and its tail-NaN behavior, gap-timestamp
reconstruction, gap-flag validity, static-attribute membership/order/values/
imputation placement, and QC-CSV-to-NetCDF cross-checking -- is implemented
directly in this module using pandas, numpy, xarray, netCDF4, json, csv,
pathlib, and hashlib.

This module never modifies, rebuilds, or promotes a package. It only reads.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .staid import normalize_staid

__all__ = [
    "PackageAuditError",
    "CheckRecord",
    "NumericCheckResult",
    "AuditReport",
    "sha256_file",
    "independent_discharge_to_runoff_mm_per_h",
    "independent_lead_shift",
    "compare_float_arrays",
    "read_basin_ids_independent",
    "read_area_csv_independent",
    "read_static_column_manifest_independent",
    "read_forcing_parquet_independent",
    "read_qobs_series_independent",
    "read_package_basin_netcdf_independent",
    "read_run_provenance_json_independent",
    "resolve_expected_netcdf_package_schema_independent",
    "package_declares_historical_v001_compatibility_lineage",
    "load_gap_inventory_independent",
    "reconstruct_gap_timestamps_independent",
    "read_gap_timestamps_json_independent",
    "derive_expected_index_independent",
    "resolve_gap_product_scope_independent",
    "check_package_layout",
    "check_basin_membership",
    "check_checksums_and_manifest",
    "audit_basin_netcdf",
    "audit_static_attributes",
    "audit_gap_reconstruction",
    "audit_qc_csv",
    "build_audit_manifest",
    "run_preflight",
    "run_audit",
    "write_audit_outputs",
]

SCHEMA_NAME = "stage1_compact_scientific_package_independent_audit_v001"
SCHEMA_VERSION = 1

# Re-declared independently of src.baseline.package_assembly.DYNAMIC_INPUTS /
# RAW_TARGET_VARIABLE and src.baseline.package_netcdf.EXPECTED_VARIABLES:
# this is the audited package's *contract*, not a value borrowed from the
# code that is supposed to satisfy it.
DYNAMIC_INPUTS = (
    "mrms_qpe_1h_mm",
    "rtma_2t_K",
    "rtma_2d_K",
    "rtma_2sh_kgkg",
    "rtma_10u_ms",
    "rtma_10v_ms",
    "mrms_qpe_1h_mm_gap",
    "rtma_gap",
)
RAW_TARGET_VARIABLE = "qobs_m3s"
LEADS_HOURS = (1, 3, 6, 12)
LEAD_VARIABLE_NAME_TEMPLATE = "qobs_mm_per_h_lead{lead:02d}"
LEAD_VARIABLES = tuple(LEAD_VARIABLE_NAME_TEMPLATE.format(lead=h) for h in LEADS_HOURS)
EXPECTED_VARIABLES = DYNAMIC_INPUTS + (RAW_TARGET_VARIABLE,) + LEAD_VARIABLES
GAP_FLAG_VARIABLES = frozenset(name for name in DYNAMIC_INPUTS if name.endswith("_gap"))

UNITS = {
    "mrms_qpe_1h_mm": "mm",
    "rtma_2t_K": "K",
    "rtma_2d_K": "K",
    "rtma_2sh_kgkg": "kg kg-1",
    "rtma_10u_ms": "m s-1",
    "rtma_10v_ms": "m s-1",
    "mrms_qpe_1h_mm_gap": "1",
    "rtma_gap": "1",
    RAW_TARGET_VARIABLE: "m3 s-1",
}
for _name in LEAD_VARIABLES:
    UNITS[_name] = "mm h-1"
del _name

MRMS_PRODUCT = "mrms_qpe_1h_pass1"
RTMA_PRODUCT = "rtma_conus_aws_2p5km"

# Re-declared independently of src.baseline.package_netcdf.SCHEMA_NAME /
# SCHEMA_VERSION: this is the *package's* on-disk dataset-level schema
# identity (written as ds.attrs["package_schema_name"/"package_schema_version"]
# by the builder), not this module's own audit-manifest schema identity
# (SCHEMA_NAME/SCHEMA_VERSION above). The two must never be confused.
PACKAGE_SCHEMA_NAME = "stage1_compact_scientific_package_v001"
PACKAGE_SCHEMA_VERSION = 1

# The scientific v002 counterpart, independently declared here alongside the
# legacy identity above.
SCIENTIFIC_V002_SCHEMA_NAME = "stage1_scientific_package_v002"
SCIENTIFIC_V002_SCHEMA_VERSION = 2

# Independently redeclared registry of every recognized on-disk NetCDF
# package schema identity -> {version, coordinate_name}. Mirrors (does not
# import) src.baseline.package_netcdf.REGISTERED_PACKAGE_NETCDF_SCHEMAS --
# re-declared here, from the certified/approved schema contract, so a bug in
# that module's own registry cannot also blind this audit. A basin file, a
# package manifest, or a run-provenance record is only ever "recognized" if
# its declared identity appears in this dict AND its version/coordinate
# agree with the entry below.
AUDIT_RECOGNIZED_PACKAGE_NETCDF_SCHEMAS = {
    PACKAGE_SCHEMA_NAME: {"version": PACKAGE_SCHEMA_VERSION, "coordinate_name": "time"},
    SCIENTIFIC_V002_SCHEMA_NAME: {"version": SCIENTIFIC_V002_SCHEMA_VERSION, "coordinate_name": "date"},
}
AUDIT_RECOGNIZED_TEMPORAL_COORDINATE_NAMES = frozenset(
    entry["coordinate_name"] for entry in AUDIT_RECOGNIZED_PACKAGE_NETCDF_SCHEMAS.values()
)

# Narrow, exact-lineage compatibility contract for the real, already-built,
# already-certified Compact Scientific Package v001: that package's manifest
# and run_provenance.json predate the netcdf_package_schema_*/
# netcdf_time_coordinate fields entirely, and the package must never be
# rewritten just to re-audit it. This is deliberately NOT a generic "missing
# metadata is acceptable" fallback -- it only applies when a source
# identifies itself, independently, through the exact frozen historical
# compact-v001 builder/manifest lineage below. Re-declared independently of
# src.baseline.package_builder.SCHEMA_NAME/SCHEMA_VERSION (never imported).
HISTORICAL_BUILDER_MANIFEST_SCHEMA_NAME = "stage1_compact_scientific_package_builder_v001"
HISTORICAL_BUILDER_MANIFEST_SCHEMA_VERSION = 1
HISTORICAL_BUILDER_MODULE = "src.baseline.package_builder"
HISTORICAL_PACKAGE_ROLE = "stage1_compact_scientific_package"

# The exact NetCDF package schema independently applied under the historical
# v001 compatibility contract -- identical in content to the
# AUDIT_RECOGNIZED_PACKAGE_NETCDF_SCHEMAS[PACKAGE_SCHEMA_NAME] entry, spelled
# out on its own so the fallback's intent reads standalone.
HISTORICAL_V001_EXPECTED_NETCDF_SCHEMA = {
    "name": PACKAGE_SCHEMA_NAME,
    "version": PACKAGE_SCHEMA_VERSION,
    "coordinate_name": "time",
}


def _package_manifest_declares_historical_v001_lineage(manifest: Mapping) -> bool:
    """True only for the exact frozen historical compact-v001 builder-manifest identity.

    Checked against fields that existed in the manifest before this schema
    patch (``schema_name``, ``schema_version``, ``package_role``) -- never
    against the newly introduced ``netcdf_package_schema_*``/
    ``netcdf_time_coordinate`` fields, since a real historical manifest
    cannot have those.
    """
    return (
        manifest.get("schema_name") == HISTORICAL_BUILDER_MANIFEST_SCHEMA_NAME
        and manifest.get("schema_version") == HISTORICAL_BUILDER_MANIFEST_SCHEMA_VERSION
        and manifest.get("package_role") == HISTORICAL_PACKAGE_ROLE
    )


def _run_provenance_declares_historical_v001_lineage(run_provenance: Mapping) -> bool:
    """True only for the exact frozen historical compact-v001 provenance identity.

    Checked against fields that existed in run_provenance.json before this
    schema patch (``builder_module``, the deprecated ``package_schema_name``)
    -- never against the newly introduced ``builder_manifest_schema_*``/
    ``netcdf_package_schema_*``/``netcdf_time_coordinate`` fields.
    """
    return (
        run_provenance.get("builder_module") == HISTORICAL_BUILDER_MODULE
        and run_provenance.get("package_schema_name") == HISTORICAL_BUILDER_MANIFEST_SCHEMA_NAME
    )


def package_declares_historical_v001_compatibility_lineage(
    manifest: Mapping, run_provenance: Mapping
) -> bool:
    """True only when BOTH the manifest and run_provenance independently agree

    that this package is the exact frozen historical compact-v001 lineage.
    Requiring agreement from both sources keeps the fallback restricted to
    the real historical package rather than a partially-tampered one.
    """
    return _package_manifest_declares_historical_v001_lineage(
        manifest
    ) and _run_provenance_declares_historical_v001_lineage(run_provenance)

# Re-declared independently of the attrs written by
# src.baseline.package_netcdf.build_basin_dataset -- the actual Gate 2
# serialization contract, inspected read-only, never imported.
GAP_FLAG_FLAG_VALUES = "0, 1"
GAP_FLAG_FLAG_MEANINGS = "no_gap gap"
RAW_TARGET_ROLE = "audit_provenance_not_training_target"
LEAD_TARGET_ROLE = "training_target"

FORBIDDEN_STATIC_COLUMNS = ("STATE", "HUC02", "LAT_GAGE", "LNG_GAGE")

# Mirrors (does not import) the package builder's own pinned CSV float64
# text-roundtrip tolerance: writing a float64 value to CSV text and reading
# it back should reproduce that same float64 value almost exactly. This is
# NOT used for QC-CSV-to-NetCDF comparison (see
# ``compare_qc_csv_against_netcdf_storage``): the QC CSV holds
# pre-quantization float64 values while the NetCDF stores float32, so that
# comparison instead casts through the on-disk storage dtype and requires
# exact agreement. This tolerance remains in use only for
# ``attributes_csv_values_match_prepared``, which compares two float64
# representations (the package's `attributes.csv` and the prepared static
# parquet) with no float32 quantization involved.
QC_CSV_ROUNDTRIP_RTOL = 1e-9
QC_CSV_ROUNDTRIP_ATOL = 1e-12

_HOUR = pd.Timedelta(hours=1)
_GAP_CSV_REQUIRED_COLUMNS = ("chunk_label", "product", "valid_time_utc", "reason")
_AREA_CSV_REQUIRED_COLUMNS = ("gauge_id", "DRAIN_SQKM")
_REQUIRED_TOP_LEVEL_PATHS = ("time_series", "attributes", "basins", "masks", "manifests", "run_provenance.json")
_REQUIRED_FILES = (
    Path("attributes/attributes.csv"),
    Path("basins/basin_ids.txt"),
    Path("masks/gap_timestamps.json"),
    Path("manifests/package_manifest.json"),
    Path("manifests/file_checksums.csv"),
    Path("run_provenance.json"),
)


class PackageAuditError(RuntimeError):
    """Raised when the audit cannot proceed (missing/unreadable/malformed input)."""


# ---------------------------------------------------------------------------
# Report primitives
# ---------------------------------------------------------------------------


@dataclass
class CheckRecord:
    severity: str
    check_id: str
    message: str = ""


@dataclass
class NumericCheckResult:
    check_id: str
    compared_count: int
    nan_mismatch_count: int
    finite_mismatch_count: int
    max_abs_diff: float
    max_rel_diff: float
    rtol: float
    atol: float
    passed: bool

    def as_dict(self) -> dict:
        return {
            "check_id": self.check_id,
            "compared_count": self.compared_count,
            "nan_mismatch_count": self.nan_mismatch_count,
            "finite_mismatch_count": self.finite_mismatch_count,
            "max_abs_diff": self.max_abs_diff,
            "max_rel_diff": self.max_rel_diff,
            "rtol": self.rtol,
            "atol": self.atol,
            "passed": self.passed,
        }


@dataclass
class AuditReport:
    records: list = field(default_factory=list)
    numeric_results: list = field(default_factory=list)

    def ok(self, check_id: str, message: str = "") -> None:
        self.records.append(CheckRecord("OK", check_id, message))

    def warn(self, check_id: str, message: str) -> None:
        self.records.append(CheckRecord("WARNING", check_id, message))

    def error(self, check_id: str, message: str) -> None:
        self.records.append(CheckRecord("ERROR", check_id, message))

    def numeric(self, result: NumericCheckResult, *, extra_message: str = "") -> None:
        self.numeric_results.append(result)
        msg = (
            f"compared={result.compared_count} nan_mismatch={result.nan_mismatch_count} "
            f"finite_mismatch={result.finite_mismatch_count} max_abs_diff={result.max_abs_diff:.6g} "
            f"max_rel_diff={result.max_rel_diff:.6g} rtol={result.rtol} atol={result.atol}"
        )
        if extra_message:
            msg = f"{extra_message}; {msg}"
        if result.passed:
            self.ok(result.check_id, msg)
        else:
            self.error(result.check_id, msg)

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
# Generic, self-contained helpers
# ---------------------------------------------------------------------------


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_source_checksums_independent(root, accepted_basin_ids: Sequence[str], *, kind: str) -> dict:
    """Recompute sha256 (from disk bytes) for every accepted basin's source file.

    ``kind`` is ``"forcing"`` (``<root>/time_series/<id>.parquet``) or
    ``"qobs"`` (``<root>/time_series/<id>.nc``). Returns a basin-id-keyed
    mapping; a basin whose source file is missing is simply omitted (the
    corresponding per-basin numeric/value checks already fail independently
    when a source file cannot be read).
    """
    if kind == "forcing":
        suffix = "parquet"
    elif kind == "qobs":
        suffix = "nc"
    else:
        raise ValueError(f"unknown kind: {kind!r}")

    root = Path(root)
    checksums = {}
    for basin_id in accepted_basin_ids:
        path = root / "time_series" / f"{basin_id}.{suffix}"
        if path.is_file():
            checksums[basin_id] = sha256_file(path)
    return checksums


def compute_package_artifact_checksums_independent(package_root: Path, accepted_basin_ids: Sequence[str]) -> dict:
    """Recompute sha256 (from disk bytes) for every authoritative package artifact.

    Covers the 32 (or however many accepted) ``time_series/<id>.nc`` files
    plus the 6 fixed metadata files in ``_REQUIRED_FILES``. Every value is
    computed independently from bytes on disk -- never copied from the
    package's own declared manifest -- so this mapping can be used to
    provenance-bind the package independently of its own self-reported
    checksums.
    """
    package_root = Path(package_root)
    checksums = {}
    for basin_id in accepted_basin_ids:
        rel = f"time_series/{basin_id}.nc"
        path = package_root / rel
        if path.is_file():
            checksums[rel] = sha256_file(path)
    for rel in _REQUIRED_FILES:
        path = package_root / rel
        if path.is_file():
            checksums[rel.as_posix()] = sha256_file(path)
    return checksums


def _validate_hourly_index(index: pd.DatetimeIndex, label: str = "timeline") -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise PackageAuditError(f"{label} must be a pandas.DatetimeIndex, got {type(index)!r}")
    if index.tz is not None:
        raise PackageAuditError(f"{label} must be timezone-naive")
    if index.has_duplicates:
        raise PackageAuditError(f"{label} contains duplicate timestamps")
    if not index.is_monotonic_increasing:
        raise PackageAuditError(f"{label} is not strictly increasing")
    if len(index) >= 2:
        deltas = index[1:] - index[:-1]
        if np.any(np.asarray(deltas) != _HOUR):
            raise PackageAuditError(f"{label} is not strictly hourly")


def _to_utc_naive_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


# ---------------------------------------------------------------------------
# Independent scientific re-derivations (the audit's core independence)
# ---------------------------------------------------------------------------


def independent_discharge_to_runoff_mm_per_h(q_m3s: np.ndarray, area_km2: float) -> np.ndarray:
    """Independent re-derivation of q_mm_per_h = q_m3s * 3.6 / area_km2.

    Deliberately re-expressed here rather than importing
    ``src.baseline.units.discharge_m3s_to_runoff_mm_per_h``, so that a shared
    bug in that module cannot pass both package construction and this audit.
    """
    q = np.asarray(q_m3s, dtype=np.float64)
    if not np.isfinite(area_km2) or area_km2 <= 0:
        raise PackageAuditError(f"area_km2 must be finite and positive, got {area_km2!r}")
    return q * 3.6 / float(area_km2)


def independent_lead_shift(mm_per_h: np.ndarray, lead_hours: int) -> np.ndarray:
    """Independent re-derivation of the lead-target shift.

    ``target[t] = mm_per_h[t + lead_hours]``, with the trailing ``lead_hours``
    rows NaN. Implemented via plain array slicing (not ``pandas.Series.shift``)
    so it is not simply the same code path as
    ``src.baseline.lead_targets.build_lead_target``.
    """
    mm_per_h = np.asarray(mm_per_h, dtype=np.float64)
    n = mm_per_h.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if lead_hours < n:
        out[: n - lead_hours] = mm_per_h[lead_hours:]
    return out


def compare_float_arrays(check_id: str, expected, actual, *, rtol: float, atol: float = 0.0) -> NumericCheckResult:
    """Full-array float comparison recording every field the audit must track.

    NaN positions must match exactly (a NaN/non-NaN mismatch at any position
    is always a failure, regardless of tolerance); finite values are compared
    with the supplied rtol/atol using the same convention as
    ``numpy.isclose``: ``|actual - expected| <= atol + rtol * |expected|``.

    +/-infinity is a categorically disallowed value in this package's
    scientific contract (NaN is scientifically meaningful in some fields;
    infinity never is). It is never treated as a match, even when ``expected``
    and ``actual`` hold the identical infinity at the same position -- naive
    finite-value comparison would otherwise silently pass such a pair
    (``inf - inf`` is ``nan``, and ``nan > threshold`` is ``False``).
    """
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if expected.shape != actual.shape:
        raise PackageAuditError(f"{check_id}: shape mismatch {expected.shape} vs {actual.shape}")

    n = int(expected.size)
    exp_nan = np.isnan(expected)
    act_nan = np.isnan(actual)
    nan_mismatch = int(np.count_nonzero(exp_nan != act_nan))

    exp_inf = np.isinf(expected)
    act_inf = np.isinf(actual)
    inf_count = int(np.count_nonzero(exp_inf | act_inf))

    finite_mask = ~exp_nan & ~act_nan & ~exp_inf & ~act_inf
    if finite_mask.any():
        diff = np.abs(actual[finite_mask] - expected[finite_mask])
        denom = np.abs(expected[finite_mask])
        max_abs = float(np.max(diff))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(denom > 0, diff / denom, 0.0)
        max_rel = float(np.max(rel))
        finite_mismatch = int(np.count_nonzero(diff > (atol + rtol * denom)))
    else:
        max_abs = 0.0
        max_rel = 0.0
        finite_mismatch = 0

    finite_mismatch += inf_count
    if inf_count:
        max_abs = float("inf")
        max_rel = float("inf")

    passed = (nan_mismatch == 0) and (finite_mismatch == 0)
    return NumericCheckResult(check_id, n, nan_mismatch, finite_mismatch, max_abs, max_rel, float(rtol), float(atol), passed)


def compare_qc_csv_against_netcdf_storage(check_id: str, netcdf_values, csv_values, *, storage_dtype) -> NumericCheckResult:
    """Compare non-authoritative QC CSV values to the authoritative NetCDF.

    The QC CSV is written from the builder's pre-quantization float64
    values; the NetCDF stores the same values quantized to ``storage_dtype``
    (independently confirmed per-variable from the NetCDF itself, not
    hard-coded -- expected to be ``float32`` for continuous variables). The
    question this check answers is "does the CSV evidence reproduce the
    exact stored representation after the documented quantization", not
    "are the CSV and NetCDF numerically close" -- so this projects the
    CSV's finite values through the same dtype cast the package applies and
    requires exact agreement, rather than a broader rtol/atol tolerance
    (which would only paper over quantization it is this check's job to
    verify). This is distinct from ``QC_CSV_ROUNDTRIP_RTOL``/``ATOL`` (a
    float64-to-float64 CSV text round-trip) and from
    ``package_float32_rtol`` (authoritative source-to-package comparison).

    NaN positions must match exactly, as in :func:`compare_float_arrays`.
    +/-infinity is likewise never treated as a match, even at matching
    positions.
    """
    csv_values = np.asarray(csv_values, dtype=np.float64)
    netcdf_values = np.asarray(netcdf_values, dtype=np.float64)
    if csv_values.shape != netcdf_values.shape:
        raise PackageAuditError(f"{check_id}: shape mismatch {csv_values.shape} vs {netcdf_values.shape}")

    n = int(csv_values.size)
    csv_nan = np.isnan(csv_values)
    nc_nan = np.isnan(netcdf_values)
    nan_mismatch = int(np.count_nonzero(csv_nan != nc_nan))

    csv_inf = np.isinf(csv_values)
    nc_inf = np.isinf(netcdf_values)
    inf_count = int(np.count_nonzero(csv_inf | nc_inf))

    finite_mask = ~csv_nan & ~nc_nan & ~csv_inf & ~nc_inf
    if finite_mask.any():
        projected = csv_values[finite_mask].astype(storage_dtype).astype(np.float64)
        actual = netcdf_values[finite_mask].astype(storage_dtype).astype(np.float64)
        mismatch = projected != actual
        finite_mismatch = int(np.count_nonzero(mismatch))
        if finite_mismatch:
            diff = np.abs(actual[mismatch] - projected[mismatch])
            max_abs = float(np.max(diff))
            denom = np.abs(projected[mismatch])
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.where(denom > 0, diff / denom, 0.0)
            max_rel = float(np.max(rel))
        else:
            max_abs = 0.0
            max_rel = 0.0
    else:
        finite_mismatch = 0
        max_abs = 0.0
        max_rel = 0.0

    finite_mismatch += inf_count
    if inf_count:
        max_abs = float("inf")
        max_rel = float("inf")

    passed = (nan_mismatch == 0) and (finite_mismatch == 0)
    return NumericCheckResult(check_id, n, nan_mismatch, finite_mismatch, max_abs, max_rel, 0.0, 0.0, passed)


# ---------------------------------------------------------------------------
# Independent readers of raw sources and package artifacts
# ---------------------------------------------------------------------------


def read_basin_ids_independent(path) -> list:
    p = Path(path)
    if not p.is_file():
        raise PackageAuditError(f"basin ids file not found: {p}")
    ids = []
    for line_number, raw_line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if raw_line == "":
            continue
        if raw_line != raw_line.strip():
            raise PackageAuditError(f"{p}:{line_number}: whitespace-padded basin id: {raw_line!r}")
        try:
            normalized = normalize_staid(raw_line)
        except (TypeError, ValueError) as exc:
            raise PackageAuditError(f"{p}:{line_number}: invalid basin id {raw_line!r}: {exc}") from exc
        if normalized != raw_line:
            raise PackageAuditError(
                f"{p}:{line_number}: basin id {raw_line!r} is not in canonical form (would be {normalized!r})"
            )
        ids.append(raw_line)
    return ids


def read_area_csv_independent(path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise PackageAuditError(f"area CSV not found: {p}")
    df = pd.read_csv(p, dtype={"gauge_id": str})
    missing = [c for c in _AREA_CSV_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise PackageAuditError(f"{p}: missing required column(s) {missing}")
    area_by_basin = {}
    for gauge_id, area in zip(df["gauge_id"], df["DRAIN_SQKM"]):
        try:
            normalized = normalize_staid(str(gauge_id))
        except (TypeError, ValueError) as exc:
            raise PackageAuditError(f"{p}: invalid gauge_id {gauge_id!r}: {exc}") from exc
        area_value = float(area)
        if not np.isfinite(area_value) or area_value <= 0:
            raise PackageAuditError(f"{p}: DRAIN_SQKM for {gauge_id!r} must be finite and positive, got {area_value!r}")
        area_by_basin[normalized] = area_value
    return area_by_basin


def read_static_column_manifest_independent(path, role: str = "model_input") -> list:
    p = Path(path)
    if not p.is_file():
        raise PackageAuditError(f"static column manifest not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    columns = data.get("columns")
    if not isinstance(columns, dict):
        raise PackageAuditError(f"{p}: missing/malformed top-level 'columns' mapping")
    cols = sorted(name for name, meta in columns.items() if isinstance(meta, dict) and meta.get("role") == role)
    if not cols:
        raise PackageAuditError(f"{p}: no columns with role={role!r}")
    return cols


def read_forcing_parquet_independent(forcing_root, basin_id: str, dynamic_inputs: Sequence[str] = DYNAMIC_INPUTS) -> pd.DataFrame:
    path = Path(forcing_root) / "time_series" / f"{basin_id}.parquet"
    if not path.is_file():
        raise PackageAuditError(f"forcing file not found for basin {basin_id!r}: {path}")
    df = pd.read_parquet(path)
    missing = [c for c in dynamic_inputs if c not in df.columns]
    if missing:
        raise PackageAuditError(f"forcing file for basin {basin_id!r} missing column(s) {missing}: {path}")
    df = df[list(dynamic_inputs)].copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    inf_columns = [c for c in dynamic_inputs if np.isinf(df[c].to_numpy(dtype=np.float64)).any()]
    if inf_columns:
        raise PackageAuditError(
            f"forcing file for basin {basin_id!r} contains +/-infinity value(s) in column(s) "
            f"{inf_columns}: {path}"
        )
    return df


def read_qobs_series_independent(qobs_root, basin_id: str) -> pd.Series:
    path = Path(qobs_root) / "time_series" / f"{basin_id}.nc"
    if not path.is_file():
        raise PackageAuditError(f"qobs file not found for basin {basin_id!r}: {path}")
    with xr.open_dataset(path) as ds:
        time_coord = "time" if "time" in ds.coords else "date"
        if time_coord not in ds.coords:
            raise PackageAuditError(f"{path}: no 'time' or 'date' coordinate found")
        if RAW_TARGET_VARIABLE not in ds.variables:
            raise PackageAuditError(f"{path}: no {RAW_TARGET_VARIABLE!r} variable found")
        idx = pd.DatetimeIndex(ds.coords[time_coord].values)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        values = np.asarray(ds[RAW_TARGET_VARIABLE].values, dtype=np.float64)
    if np.isinf(values).any():
        raise PackageAuditError(f"{path}: qobs source contains +/-infinity value(s)")
    return pd.Series(values, index=idx, name=RAW_TARGET_VARIABLE)


def read_package_basin_netcdf_independent(path) -> dict:
    """Independently read one package basin NetCDF file's contents.

    Uses xarray purely as a neutral I/O library; does not call any function
    from ``src.baseline.package_netcdf``. Does not assume a coordinate name:
    ``temporal_dims_present`` reports every recognized (``time``/``date``)
    coordinate actually found as a dimension, and ``time_index`` is only
    populated when exactly one is present -- the strict "exactly one, and
    it's the right one" judgment is made by the caller
    (:func:`audit_basin_netcdf`), not here, so both/neither/wrong-coordinate
    cases can each be reported as their own distinct check.
    """
    p = Path(path)
    if not p.is_file():
        raise PackageAuditError(f"package NetCDF file not found: {p}")
    with xr.open_dataset(p, engine="netcdf4") as ds:
        ds.load()
        gauge_id = ds.attrs.get("gauge_id")
        package_schema_name = ds.attrs.get("package_schema_name")
        package_schema_version = ds.attrs.get("package_schema_version")
        temporal_dims_present = sorted(
            str(name) for name in ds.dims if str(name) in AUDIT_RECOGNIZED_TEMPORAL_COORDINATE_NAMES
        )
        time_index = None
        if len(temporal_dims_present) == 1:
            coord_name = temporal_dims_present[0]
            time_index = pd.DatetimeIndex(ds[coord_name].values)
            if time_index.tz is not None:
                time_index = time_index.tz_convert("UTC").tz_localize(None)
        dataset_dims = {str(name): int(size) for name, size in dict(ds.sizes).items()}
        variable_order = list(ds.data_vars)
        variables = {name: np.asarray(ds[name].values) for name in variable_order}
        dtypes = {name: str(ds[name].values.dtype) for name in variable_order}
        units = {name: ds[name].attrs.get("units") for name in variable_order}
        variable_dims = {name: tuple(ds[name].dims) for name in variable_order}
        variable_attrs = {name: dict(ds[name].attrs) for name in variable_order}
    return {
        "gauge_id": gauge_id,
        "package_schema_name": package_schema_name,
        "package_schema_version": package_schema_version,
        "temporal_dims_present": temporal_dims_present,
        "dataset_dims": dataset_dims,
        "time_index": time_index,
        "variable_order": variable_order,
        "variables": variables,
        "dtypes": dtypes,
        "units": units,
        "variable_dims": variable_dims,
        "variable_attrs": variable_attrs,
    }


def read_run_provenance_json_independent(report: "AuditReport", package_root) -> dict:
    """Independently read ``run_provenance.json``'s raw content.

    Records a ``run_provenance_readable`` check and returns ``{}`` (never
    raises) on any read/parse failure, mirroring
    :func:`check_checksums_and_manifest`'s tolerant-read-then-report pattern
    so a missing/corrupt provenance file fails the audit cleanly instead of
    crashing the run.
    """
    path = Path(package_root) / "run_provenance.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.error("run_provenance_readable", str(exc))
        return {}
    report.ok("run_provenance_readable")
    return data


def resolve_expected_netcdf_package_schema_independent(
    report: "AuditReport",
    *,
    source_label: str,
    declared: Mapping,
    historical_lineage_recognized: bool = False,
) -> Optional[dict]:
    """Independently validate one source's declared NetCDF package schema.

    ``declared`` is either the package manifest or the run-provenance
    record; ``source_label`` (e.g. ``"package_manifest"``/``"run_provenance"``)
    namespaces the emitted check IDs. Checks, in order: the name/version/
    coordinate fields are all present; the declared name is a recognized
    schema identity (independently redeclared in
    ``AUDIT_RECOGNIZED_PACKAGE_NETCDF_SCHEMAS`` -- never imported from
    ``src.baseline.package_netcdf``); the declared version and coordinate
    agree with that registry entry. Returns the resolved
    ``{"name", "version", "coordinate_name"}`` only if every check passes,
    else ``None``.

    If all three fields are absent AND ``historical_lineage_recognized`` is
    True (the caller has independently confirmed this source belongs to the
    exact frozen historical compact-v001 lineage -- see
    :func:`package_declares_historical_v001_compatibility_lineage`), the
    frozen ``HISTORICAL_V001_EXPECTED_NETCDF_SCHEMA`` is applied instead of
    failing. This is a narrow, exact-lineage compatibility path, not a
    generic "missing metadata is fine" fallback: any other missing-field
    combination, or an unrecognized lineage, still fails as before.
    """
    name = declared.get("netcdf_package_schema_name")
    version = declared.get("netcdf_package_schema_version")
    coordinate_name = declared.get("netcdf_time_coordinate")

    if name is None and version is None and coordinate_name is None and historical_lineage_recognized:
        report.ok(
            f"{source_label}_netcdf_schema_historical_v001_compatibility_used",
            "explicit netcdf_package_schema_name/version/netcdf_time_coordinate fields are absent from "
            f"{source_label}; recognized the exact frozen historical compact-v001 builder/manifest lineage, "
            "so the historical NetCDF schema is applied independently without requiring the package to be "
            f"rewritten: {HISTORICAL_V001_EXPECTED_NETCDF_SCHEMA}",
        )
        return dict(HISTORICAL_V001_EXPECTED_NETCDF_SCHEMA)

    if name is not None:
        report.ok(f"{source_label}_netcdf_schema_name_present")
    else:
        report.error(f"{source_label}_netcdf_schema_name_present", "netcdf_package_schema_name is missing")

    if version is not None:
        report.ok(f"{source_label}_netcdf_schema_version_present")
    else:
        report.error(f"{source_label}_netcdf_schema_version_present", "netcdf_package_schema_version is missing")

    if coordinate_name is not None:
        report.ok(f"{source_label}_netcdf_schema_coordinate_present")
    else:
        report.error(f"{source_label}_netcdf_schema_coordinate_present", "netcdf_time_coordinate is missing")

    if name is None or version is None or coordinate_name is None:
        return None

    registry_entry = AUDIT_RECOGNIZED_PACKAGE_NETCDF_SCHEMAS.get(name)
    if registry_entry is None:
        report.error(
            f"{source_label}_netcdf_schema_identity_recognized",
            f"unrecognized netcdf_package_schema_name {name!r}; recognized: "
            f"{sorted(AUDIT_RECOGNIZED_PACKAGE_NETCDF_SCHEMAS)}",
        )
        return None
    report.ok(f"{source_label}_netcdf_schema_identity_recognized")

    if version == registry_entry["version"]:
        report.ok(f"{source_label}_netcdf_schema_version_matches_registry")
    else:
        report.error(
            f"{source_label}_netcdf_schema_version_matches_registry",
            f"declared version {version!r} != registry version {registry_entry['version']!r} for {name!r}",
        )
        return None

    if coordinate_name == registry_entry["coordinate_name"]:
        report.ok(f"{source_label}_netcdf_schema_coordinate_matches_registry")
    else:
        report.error(
            f"{source_label}_netcdf_schema_coordinate_matches_registry",
            f"declared coordinate {coordinate_name!r} != registry coordinate "
            f"{registry_entry['coordinate_name']!r} for {name!r}",
        )
        return None

    return {"name": name, "version": version, "coordinate_name": coordinate_name}


def load_gap_inventory_independent(csv_path) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.is_file():
        raise PackageAuditError(f"gap inventory CSV not found: {p}")
    df = pd.read_csv(p, dtype=str)
    missing = [c for c in _GAP_CSV_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise PackageAuditError(f"{p}: missing required column(s) {missing}")
    return df


def reconstruct_gap_timestamps_independent(df: pd.DataFrame, products: Sequence[str]) -> list:
    unknown = sorted(set(products) - {MRMS_PRODUCT, RTMA_PRODUCT})
    if unknown:
        raise PackageAuditError(f"unknown gap product(s) requested: {unknown}")
    subset = df.loc[df["product"].isin(products)]
    timestamps = sorted({_to_utc_naive_timestamp(v) for v in subset["valid_time_utc"]})
    return timestamps


def read_gap_timestamps_json_independent(path) -> list:
    p = Path(path)
    if not p.is_file():
        raise PackageAuditError(f"gap timestamps artifact not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise PackageAuditError(f"{p}: expected a JSON list, got {type(raw).__name__}")
    out = []
    for value in raw:
        if not isinstance(value, str):
            raise PackageAuditError(f"{p}: expected ISO timestamp strings, got {value!r}")
        out.append(_to_utc_naive_timestamp(value))
    return out


def derive_expected_index_independent(policy: Mapping) -> pd.DatetimeIndex:
    """Independent re-derivation of the canonical hourly timeline from policy.

    Deliberately re-expressed here rather than importing
    ``src.baseline.package_builder.derive_expected_index_from_policy``.
    """
    period = policy["period"]
    start = pd.Timestamp(period["start_utc"])
    end = pd.Timestamp(period["end_utc"])
    if start.tzinfo is not None:
        start = start.tz_convert("UTC").tz_localize(None)
    if end.tzinfo is not None:
        end = end.tz_convert("UTC").tz_localize(None)
    index = pd.date_range(start, end, freq="h")
    _validate_hourly_index(index, label="policy-derived timeline")
    expected_hours = period["expected_hours"]
    if len(index) != expected_hours:
        raise PackageAuditError(f"policy period does not produce the pinned {expected_hours} rows, got {len(index)}")
    return index


def resolve_gap_product_scope_independent(policy: Mapping) -> tuple:
    """Independent re-derivation of which gap products feed the history mask.

    Deliberately re-expressed here rather than importing
    ``src.baseline.package_builder.resolve_gap_product_scope``.
    """
    include_rtma = policy["gap_policy"]["include_rtma_in_history_mask"]
    if not isinstance(include_rtma, bool):
        raise PackageAuditError("gap_policy.include_rtma_in_history_mask must be a bool")
    if include_rtma:
        return (MRMS_PRODUCT, RTMA_PRODUCT)
    return (MRMS_PRODUCT,)


# ---------------------------------------------------------------------------
# Layout / membership / checksum checks
# ---------------------------------------------------------------------------


def check_package_layout(report: AuditReport, package_root: Path) -> None:
    missing_paths = [name for name in _REQUIRED_TOP_LEVEL_PATHS if not (package_root / name).exists()]
    if missing_paths:
        report.error("package_layout_complete", f"missing required path(s): {missing_paths}")
    else:
        report.ok("package_layout_complete")

    missing_files = [str(rel) for rel in _REQUIRED_FILES if not (package_root / rel).is_file()]
    if missing_files:
        report.error("package_required_files_present", f"missing file(s): {missing_files}")
    else:
        report.ok("package_required_files_present")


def check_exact_package_layout(report: AuditReport, package_root: Path, accepted_basin_ids: Sequence[str]) -> None:
    """Closed-world package layout check.

    Unlike :func:`check_package_layout` (which only checks that required
    paths/files are present -- an open-world check), this recursively
    enumerates every file actually on disk under ``package_root`` and fails
    on anything missing from, or extra beyond, the exactly-defined expected
    set: the 6 fixed metadata files plus one ``time_series/<id>.nc`` per
    accepted basin. Also rejects any unexpected top-level package entry.
    """
    expected_relative_files = {rel.as_posix() for rel in _REQUIRED_FILES}
    expected_relative_files |= {f"time_series/{basin_id}.nc" for basin_id in accepted_basin_ids}

    actual_relative_files = set()
    if package_root.is_dir():
        for p in package_root.rglob("*"):
            if p.is_file():
                actual_relative_files.add(p.relative_to(package_root).as_posix())

    missing = sorted(expected_relative_files - actual_relative_files)
    extra = sorted(actual_relative_files - expected_relative_files)
    if missing or extra:
        report.error(
            "package_exact_file_set",
            f"missing={missing[:10]} extra={extra[:10]} (total missing={len(missing)}, extra={len(extra)})",
        )
    else:
        report.ok("package_exact_file_set", str(len(actual_relative_files)))

    expected_top_level = set(_REQUIRED_TOP_LEVEL_PATHS)
    actual_top_level = {p.name for p in package_root.iterdir()} if package_root.is_dir() else set()
    unexpected_top_level = sorted(actual_top_level - expected_top_level)
    if unexpected_top_level:
        report.error("package_no_unexpected_top_level_entries", f"unexpected: {unexpected_top_level}")
    else:
        report.ok("package_no_unexpected_top_level_entries")


def check_basin_membership(report: AuditReport, package_root: Path, accepted_basin_ids: Sequence[str]) -> list:
    package_basin_ids = read_basin_ids_independent(package_root / "basins" / "basin_ids.txt")

    if list(package_basin_ids) == list(accepted_basin_ids):
        report.ok("basin_membership_and_order_match_selection", str(len(package_basin_ids)))
    elif set(package_basin_ids) == set(accepted_basin_ids):
        report.error("basin_order_matches_selection", "same membership, different order")
    else:
        missing = sorted(set(accepted_basin_ids) - set(package_basin_ids))
        extra = sorted(set(package_basin_ids) - set(accepted_basin_ids))
        report.error("basin_membership_matches_selection", f"missing={missing} extra={extra}")

    nc_dir = package_root / "time_series"
    actual_nc_names = {p.name for p in nc_dir.iterdir() if p.is_file()} if nc_dir.is_dir() else set()
    expected_nc_names = {f"{b}.nc" for b in accepted_basin_ids}
    if actual_nc_names != expected_nc_names:
        report.error(
            "netcdf_file_membership",
            f"missing={sorted(expected_nc_names - actual_nc_names)} extra={sorted(actual_nc_names - expected_nc_names)}",
        )
    else:
        report.ok("netcdf_file_membership", str(len(actual_nc_names)))

    return package_basin_ids


def check_checksums_and_manifest(report: AuditReport, package_root: Path) -> dict:
    manifest_path = package_root / "manifests" / "package_manifest.json"
    checksums_csv_path = package_root / "manifests" / "file_checksums.csv"

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.error("package_manifest_readable", str(exc))
        return {}
    report.ok("package_manifest_readable")

    declared = {}
    try:
        with open(checksums_csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                declared[row["relative_path"]] = row
    except OSError as exc:
        report.error("file_checksums_csv_readable", str(exc))
        return manifest
    report.ok("file_checksums_csv_readable")

    missing_on_disk = []
    mismatched = []
    for rel_path, row in declared.items():
        abs_path = package_root / rel_path
        if not abs_path.is_file():
            missing_on_disk.append(rel_path)
            continue
        actual_sha256 = sha256_file(abs_path)
        actual_size = abs_path.stat().st_size
        if actual_sha256 != row["sha256"] or str(actual_size) != str(row["size_bytes"]):
            mismatched.append(rel_path)
    if missing_on_disk or mismatched:
        report.error(
            "file_checksums_recomputed_match_declared",
            f"missing={missing_on_disk[:10]} mismatched={mismatched[:10]} "
            f"(total missing={len(missing_on_disk)}, mismatched={len(mismatched)})",
        )
    else:
        report.ok("file_checksums_recomputed_match_declared", str(len(declared)))

    actual_files = set()
    for sub in ("time_series", "attributes", "basins", "masks"):
        subdir = package_root / sub
        if subdir.is_dir():
            for p in subdir.rglob("*"):
                if p.is_file():
                    actual_files.add(p.relative_to(package_root).as_posix())
    undeclared = sorted(actual_files - set(declared.keys()))
    if undeclared:
        report.error("no_undeclared_authoritative_files", f"undeclared file(s): {undeclared[:10]}")
    else:
        report.ok("no_undeclared_authoritative_files")

    manifest_mismatches = []
    for entry in manifest.get("per_basin_time_series", []):
        rel = entry.get("relative_path")
        abs_path = package_root / rel if rel else None
        if abs_path is None or not abs_path.is_file() or sha256_file(abs_path) != entry.get("sha256"):
            manifest_mismatches.append(rel)
    if manifest_mismatches:
        report.error("package_manifest_per_basin_checksums_match", f"mismatched: {manifest_mismatches[:10]}")
    else:
        report.ok("package_manifest_per_basin_checksums_match", str(len(manifest.get("per_basin_time_series", []))))

    gap_entry = manifest.get("gap_timestamp_artifact") or {}
    gap_path = package_root / "masks" / "gap_timestamps.json"
    if gap_path.is_file() and gap_entry.get("sha256") == sha256_file(gap_path):
        report.ok("package_manifest_gap_artifact_checksum_match")
    else:
        report.error("package_manifest_gap_artifact_checksum_match", "gap_timestamps.json checksum mismatch vs manifest")

    return manifest


# ---------------------------------------------------------------------------
# Per-basin NetCDF scientific audit
# ---------------------------------------------------------------------------


def audit_basin_netcdf(
    report: AuditReport,
    *,
    basin_id: str,
    nc_path: Path,
    expected_index: pd.DatetimeIndex,
    forcing_root,
    qobs_root,
    area_km2: float,
    package_float32_rtol: float,
    expected_schema: Optional[dict] = None,
    expected_schema_from_provenance: Optional[dict] = None,
) -> Optional[tuple]:
    """Audit one basin NetCDF file. Returns the file's own independently-read
    ``(package_schema_name, package_schema_version, coordinate_name)``
    identity tuple (or ``None`` if the file could not be read, or the
    coordinate could not be determined) so the caller can additionally check
    every basin file in a package shares the same identity.

    ``expected_schema``/``expected_schema_from_provenance`` are the resolved,
    independently-validated schema declarations from the package manifest
    and run-provenance record respectively (see
    :func:`resolve_expected_netcdf_package_schema_independent`); either may
    be ``None`` if that source's declaration failed to resolve, in which
    case the corresponding cross-check is skipped here (already reported as
    a failure at the source-resolution level).
    """
    try:
        disk = read_package_basin_netcdf_independent(nc_path)
    except PackageAuditError as exc:
        report.error(f"netcdf_readable[{basin_id}]", str(exc))
        return None

    if disk["gauge_id"] == basin_id:
        report.ok(f"netcdf_gauge_id[{basin_id}]")
    else:
        report.error(f"netcdf_gauge_id[{basin_id}]", f"gauge_id {disk['gauge_id']!r} != {basin_id!r}")

    if list(disk["variable_order"]) == list(EXPECTED_VARIABLES):
        report.ok(f"netcdf_variable_order[{basin_id}]")
    else:
        report.error(f"netcdf_variable_order[{basin_id}]", f"got {disk['variable_order']}")

    disk_schema_name = disk["package_schema_name"]
    disk_schema_version = disk["package_schema_version"]
    temporal_dims_present = disk["temporal_dims_present"]

    if disk_schema_name is not None:
        report.ok(f"netcdf_package_schema_name_present[{basin_id}]")
    else:
        report.error(f"netcdf_package_schema_name_present[{basin_id}]", "package_schema_name attribute missing")

    if disk_schema_version is not None:
        report.ok(f"netcdf_package_schema_version_present[{basin_id}]")
    else:
        report.error(f"netcdf_package_schema_version_present[{basin_id}]", "package_schema_version attribute missing")

    registry_entry = AUDIT_RECOGNIZED_PACKAGE_NETCDF_SCHEMAS.get(disk_schema_name)
    if registry_entry is not None:
        report.ok(f"netcdf_package_schema_identity_recognized[{basin_id}]")
    else:
        report.error(
            f"netcdf_package_schema_identity_recognized[{basin_id}]",
            f"unrecognized package_schema_name {disk_schema_name!r} on disk",
        )

    # Preserved check ID / comparison semantics: the basin file's own
    # embedded identity must match the package's declared (manifest) schema
    # exactly -- this is what previously only ever checked against the
    # frozen legacy v001 identity.
    if (
        expected_schema is not None
        and disk_schema_name == expected_schema["name"]
        and disk_schema_version == expected_schema["version"]
    ):
        report.ok(f"netcdf_package_schema[{basin_id}]")
    else:
        report.error(
            f"netcdf_package_schema[{basin_id}]",
            f"got name={disk_schema_name!r} version={disk_schema_version!r}, "
            f"expected name={(expected_schema or {}).get('name')!r} "
            f"version={(expected_schema or {}).get('version')!r}",
        )

    if expected_schema_from_provenance is not None:
        if (
            disk_schema_name == expected_schema_from_provenance["name"]
            and disk_schema_version == expected_schema_from_provenance["version"]
        ):
            report.ok(f"netcdf_matches_run_provenance_schema[{basin_id}]")
        else:
            report.error(
                f"netcdf_matches_run_provenance_schema[{basin_id}]",
                f"got name={disk_schema_name!r} version={disk_schema_version!r}, "
                f"run_provenance declares name={expected_schema_from_provenance['name']!r} "
                f"version={expected_schema_from_provenance['version']!r}",
            )

    if len(temporal_dims_present) == 1:
        report.ok(f"netcdf_temporal_coordinate_present[{basin_id}]", temporal_dims_present[0])
    else:
        report.error(
            f"netcdf_temporal_coordinate_present[{basin_id}]",
            f"expected exactly one of {sorted(AUDIT_RECOGNIZED_TEMPORAL_COORDINATE_NAMES)} as a dimension, "
            f"found {temporal_dims_present}",
        )

    coordinate_name = temporal_dims_present[0] if len(temporal_dims_present) == 1 else None
    if expected_schema is not None:
        if temporal_dims_present == [expected_schema["coordinate_name"]]:
            report.ok(f"netcdf_temporal_coordinate_matches_declared_schema[{basin_id}]")
        else:
            report.error(
                f"netcdf_temporal_coordinate_matches_declared_schema[{basin_id}]",
                f"declared schema {expected_schema['name']!r} requires coordinate "
                f"{expected_schema['coordinate_name']!r}, found {temporal_dims_present}",
            )

    if coordinate_name is not None:
        expected_dims = {coordinate_name: len(expected_index)}
        if disk["dataset_dims"] == expected_dims:
            report.ok(f"netcdf_dataset_dims[{basin_id}]", str(expected_dims))
        else:
            report.error(f"netcdf_dataset_dims[{basin_id}]", f"got {disk['dataset_dims']}, expected {expected_dims}")
    else:
        report.error(f"netcdf_dataset_dims[{basin_id}]", f"cannot determine dataset dims: dataset_dims={disk['dataset_dims']}")

    if disk["time_index"] is not None and disk["time_index"].equals(expected_index):
        report.ok(f"netcdf_timeline_exact[{basin_id}]", str(len(expected_index)))
    else:
        report.error(f"netcdf_timeline_exact[{basin_id}]", "time index does not equal expected canonical timeline")

    for name in EXPECTED_VARIABLES:
        expected_units = UNITS[name]
        actual_units = disk["units"].get(name)
        if actual_units == expected_units:
            report.ok(f"netcdf_units[{basin_id}][{name}]")
        else:
            report.error(f"netcdf_units[{basin_id}][{name}]", f"got {actual_units!r}, expected {expected_units!r}")

        expected_dtype = "int8" if name in GAP_FLAG_VARIABLES else "float32"
        actual_dtype = disk["dtypes"].get(name)
        if actual_dtype == expected_dtype:
            report.ok(f"netcdf_dtype[{basin_id}][{name}]")
        else:
            report.error(f"netcdf_dtype[{basin_id}][{name}]", f"got {actual_dtype!r}, expected {expected_dtype!r}")

        expected_var_dims = (coordinate_name,) if coordinate_name is not None else None
        actual_var_dims = disk["variable_dims"].get(name)
        if expected_var_dims is not None and actual_var_dims == expected_var_dims:
            report.ok(f"netcdf_variable_dims[{basin_id}][{name}]")
        else:
            report.error(
                f"netcdf_variable_dims[{basin_id}][{name}]",
                f"got {actual_var_dims!r}, expected {expected_var_dims!r}",
            )

        if name not in GAP_FLAG_VARIABLES:
            values = np.asarray(disk["variables"][name], dtype=np.float64)
            n_inf = int(np.count_nonzero(np.isinf(values)))
            if n_inf:
                report.error(f"netcdf_no_infinity[{basin_id}][{name}]", f"{n_inf} +/-infinity value(s) found")
            else:
                report.ok(f"netcdf_no_infinity[{basin_id}][{name}]")

    for name in GAP_FLAG_VARIABLES:
        values = np.asarray(disk["variables"][name], dtype=np.float64)
        n_nonfinite = int(np.count_nonzero(~np.isfinite(values)))
        finite_values = values[np.isfinite(values)]
        n_invalid = int(np.count_nonzero((finite_values != 0.0) & (finite_values != 1.0)))
        if n_nonfinite or n_invalid:
            report.error(
                f"netcdf_gap_flag_binary[{basin_id}][{name}]",
                f"{n_nonfinite} non-finite, {n_invalid} out-of-{{0,1}} value(s)",
            )
        else:
            report.ok(f"netcdf_gap_flag_binary[{basin_id}][{name}]")

        attrs = disk["variable_attrs"].get(name, {})
        if attrs.get("flag_values") == GAP_FLAG_FLAG_VALUES and attrs.get("flag_meanings") == GAP_FLAG_FLAG_MEANINGS:
            report.ok(f"netcdf_gap_flag_attrs[{basin_id}][{name}]")
        else:
            report.error(
                f"netcdf_gap_flag_attrs[{basin_id}][{name}]",
                f"got flag_values={attrs.get('flag_values')!r} flag_meanings={attrs.get('flag_meanings')!r}, "
                f"expected flag_values={GAP_FLAG_FLAG_VALUES!r} flag_meanings={GAP_FLAG_FLAG_MEANINGS!r}",
            )

    raw_attrs = disk["variable_attrs"].get(RAW_TARGET_VARIABLE, {})
    if raw_attrs.get("role") == RAW_TARGET_ROLE:
        report.ok(f"netcdf_raw_target_role[{basin_id}]")
    else:
        report.error(
            f"netcdf_raw_target_role[{basin_id}]",
            f"got role={raw_attrs.get('role')!r}, expected {RAW_TARGET_ROLE!r}",
        )

    try:
        forcing = read_forcing_parquet_independent(forcing_root, basin_id, DYNAMIC_INPUTS)
        qobs_series = read_qobs_series_independent(qobs_root, basin_id)
    except PackageAuditError as exc:
        report.error(f"raw_source_readable[{basin_id}]", str(exc))
        return (disk_schema_name, disk_schema_version, coordinate_name)

    if forcing.index.equals(expected_index):
        report.ok(f"forcing_source_timeline_exact[{basin_id}]")
    else:
        report.error(f"forcing_source_timeline_exact[{basin_id}]", "forcing source index != expected canonical timeline")

    if qobs_series.index.equals(expected_index):
        report.ok(f"qobs_source_timeline_exact[{basin_id}]")
    else:
        report.error(f"qobs_source_timeline_exact[{basin_id}]", "qobs source index != expected canonical timeline")

    for name in DYNAMIC_INPUTS:
        expected_values = forcing[name].to_numpy(dtype=np.float64)
        actual_values = np.asarray(disk["variables"][name], dtype=np.float64)
        if name in GAP_FLAG_VARIABLES:
            check_id = f"dynamic_input_matches_forcing_source[{basin_id}][{name}]"
            if np.array_equal(expected_values, actual_values):
                report.ok(check_id)
            else:
                n_mismatch = int(np.count_nonzero(expected_values != actual_values))
                report.error(check_id, f"{n_mismatch} value(s) differ")
            continue
        result = compare_float_arrays(
            f"dynamic_input_matches_forcing_source[{basin_id}][{name}]",
            expected_values,
            actual_values,
            rtol=package_float32_rtol,
        )
        report.numeric(result)

    raw_qobs_expected = qobs_series.to_numpy(dtype=np.float64)
    raw_qobs_actual = np.asarray(disk["variables"][RAW_TARGET_VARIABLE], dtype=np.float64)
    report.numeric(
        compare_float_arrays(
            f"raw_qobs_matches_source[{basin_id}]", raw_qobs_expected, raw_qobs_actual, rtol=package_float32_rtol
        )
    )

    mm_per_h_expected = independent_discharge_to_runoff_mm_per_h(raw_qobs_expected, area_km2)
    for lead_hours, name in zip(LEADS_HOURS, LEAD_VARIABLES):
        expected_target = independent_lead_shift(mm_per_h_expected, lead_hours)
        actual_target = np.asarray(disk["variables"][name], dtype=np.float64)
        report.numeric(
            compare_float_arrays(
                f"lead_target_matches_independent_recomputation[{basin_id}][{name}]",
                expected_target,
                actual_target,
                rtol=package_float32_rtol,
            )
        )

        tail = actual_target[-lead_hours:] if lead_hours > 0 else np.array([])
        if tail.size and np.all(np.isnan(tail)):
            report.ok(f"lead_target_tail_nan[{basin_id}][{name}]")
        elif tail.size:
            n_non_nan = int(np.count_nonzero(~np.isnan(tail)))
            report.error(
                f"lead_target_tail_nan[{basin_id}][{name}]",
                f"expected last {lead_hours} value(s) NaN, found {n_non_nan} non-NaN",
            )

        lead_attrs = disk["variable_attrs"].get(name, {})
        role_ok = lead_attrs.get("role") == LEAD_TARGET_ROLE
        lead_hours_ok = lead_attrs.get("lead_hours") == lead_hours
        if role_ok and lead_hours_ok:
            report.ok(f"netcdf_lead_target_attrs[{basin_id}][{name}]")
        else:
            report.error(
                f"netcdf_lead_target_attrs[{basin_id}][{name}]",
                f"got role={lead_attrs.get('role')!r} lead_hours={lead_attrs.get('lead_hours')!r}, "
                f"expected role={LEAD_TARGET_ROLE!r} lead_hours={lead_hours!r}",
            )

    return (disk_schema_name, disk_schema_version, coordinate_name)


# ---------------------------------------------------------------------------
# Static attribute audit
# ---------------------------------------------------------------------------


def audit_static_attributes(
    report: AuditReport,
    *,
    package_root: Path,
    accepted_basin_ids: Sequence[str],
    prepared_static_parquet_path,
    static_column_manifest_path,
    imputation_manifest_path=None,
    imputed_value_mask_path=None,
) -> None:
    try:
        model_input_columns = read_static_column_manifest_independent(static_column_manifest_path)
        prepared = pd.read_parquet(prepared_static_parquet_path)
    except (PackageAuditError, OSError) as exc:
        report.error("static_inputs_readable", str(exc))
        return
    report.ok("static_inputs_readable")

    if prepared.index.name != "gauge_id" and "gauge_id" in prepared.columns:
        prepared = prepared.set_index("gauge_id")
    prepared.index = [str(v) for v in prepared.index]

    missing_basins = sorted(set(accepted_basin_ids) - set(prepared.index))
    if missing_basins:
        report.error("prepared_static_covers_accepted_basins", f"missing: {missing_basins}")
        return
    report.ok("prepared_static_covers_accepted_basins")

    missing_cols = [c for c in model_input_columns if c not in prepared.columns]
    if missing_cols:
        report.error("prepared_static_has_model_input_columns", f"missing: {missing_cols}")
        return
    report.ok("prepared_static_has_model_input_columns")

    attrs_csv_path = package_root / "attributes" / "attributes.csv"
    if not attrs_csv_path.is_file():
        report.error("attributes_csv_readable", f"not found: {attrs_csv_path}")
        return
    package_attrs = pd.read_csv(attrs_csv_path, dtype={"gauge_id": str}).set_index("gauge_id")

    if list(package_attrs.index) == list(accepted_basin_ids):
        report.ok("attributes_csv_basin_order")
    else:
        report.error("attributes_csv_basin_order", "attributes.csv basin order does not match accepted basin selection")

    if list(package_attrs.columns) == list(model_input_columns):
        report.ok("attributes_csv_column_order_and_membership")
    else:
        missing_ = sorted(set(model_input_columns) - set(package_attrs.columns))
        extra_ = sorted(set(package_attrs.columns) - set(model_input_columns))
        if missing_ or extra_:
            report.error("attributes_csv_column_membership", f"missing={missing_} extra={extra_}")
        else:
            report.error("attributes_csv_column_order", f"got {list(package_attrs.columns)}")

    forbidden_present = sorted(set(package_attrs.columns) & set(FORBIDDEN_STATIC_COLUMNS))
    if forbidden_present:
        report.error("attributes_csv_no_forbidden_columns", f"forbidden column(s) present: {forbidden_present}")
    else:
        report.ok("attributes_csv_no_forbidden_columns")

    common_basins = [b for b in accepted_basin_ids if b in package_attrs.index]
    common_columns = [c for c in model_input_columns if c in package_attrs.columns]
    expected_values = prepared.loc[common_basins, common_columns].to_numpy(dtype=np.float64)
    actual_values = package_attrs.loc[common_basins, common_columns].to_numpy(dtype=np.float64)
    report.numeric(
        compare_float_arrays(
            "attributes_csv_values_match_prepared",
            expected_values,
            actual_values,
            rtol=QC_CSV_ROUNDTRIP_RTOL,
            atol=QC_CSV_ROUNDTRIP_ATOL,
        )
    )

    n_residual_nan = int(np.count_nonzero(np.isnan(actual_values)))
    if n_residual_nan:
        report.error("attributes_csv_no_residual_nan", f"{n_residual_nan} NaN value(s) remain in authoritative static attributes")
    else:
        report.ok("attributes_csv_no_residual_nan")

    if imputation_manifest_path is None:
        report.warn("imputation_manifest_readable", "no imputation manifest supplied; imputation-placement check skipped")
        return

    try:
        manifest = json.loads(Path(imputation_manifest_path).read_text(encoding="utf-8"))
    except OSError as exc:
        report.error("imputation_manifest_readable", str(exc))
        return
    report.ok("imputation_manifest_readable")
    per_column = manifest.get("per_column", {})

    if imputed_value_mask_path is None:
        report.warn("imputed_values_placed_correctly", "no imputed-value-mask supplied; check skipped")
        return

    try:
        mask_df = pd.read_parquet(imputed_value_mask_path)
    except OSError as exc:
        report.error("imputed_value_mask_readable", str(exc))
        return
    report.ok("imputed_value_mask_readable")

    if mask_df.index.name != "gauge_id" and "gauge_id" in mask_df.columns:
        mask_df = mask_df.set_index("gauge_id")
    mask_df.index = [str(v) for v in mask_df.index]

    if list(mask_df.index) == list(accepted_basin_ids):
        report.ok("imputed_value_mask_basin_order")
    else:
        report.error(
            "imputed_value_mask_basin_order",
            "imputed-value-mask basin index does not exactly match accepted basin selection (membership/order)",
        )

    if list(mask_df.columns) == list(model_input_columns):
        report.ok("imputed_value_mask_column_order")
    else:
        report.error(
            "imputed_value_mask_column_order",
            f"got {list(mask_df.columns)}, expected {list(model_input_columns)}",
        )

    common_mask_basins = [b for b in accepted_basin_ids if b in mask_df.index]
    common_mask_columns = [c for c in model_input_columns if c in mask_df.columns]
    mask_subset = mask_df.loc[common_mask_basins, common_mask_columns]

    if mask_subset.isna().any().any():
        n_missing_cells = int(mask_subset.isna().to_numpy().sum())
        report.error("imputed_value_mask_no_missing_cells", f"{n_missing_cells} missing (NaN) cell(s) in mask")
    else:
        report.ok("imputed_value_mask_no_missing_cells")

    mask_values = mask_subset.to_numpy()
    try:
        mask_numeric = mask_values.astype(np.float64)
        n_non_boolean = int(np.count_nonzero(~np.isin(mask_numeric, [0.0, 1.0])))
    except (TypeError, ValueError):
        n_non_boolean = int(mask_values.size)
    if n_non_boolean:
        report.error("imputed_value_mask_boolean_values_only", f"{n_non_boolean} non-binary value(s) found in mask")
    else:
        report.ok("imputed_value_mask_boolean_values_only")

    per_column_count_mismatches = []
    any_per_column_count_recorded = False
    for col in common_mask_columns:
        recorded = per_column.get(col, {}).get("n_missing_before_apply")
        if recorded is None:
            continue
        any_per_column_count_recorded = True
        actual_count = int(mask_subset[col].astype(bool).sum())
        if int(recorded) != actual_count:
            per_column_count_mismatches.append((col, recorded, actual_count))
    if per_column_count_mismatches:
        report.error(
            "imputed_value_mask_per_column_count_matches_manifest",
            f"{len(per_column_count_mismatches)} column(s) disagree with manifest: {per_column_count_mismatches[:10]}",
        )
    elif any_per_column_count_recorded:
        report.ok("imputed_value_mask_per_column_count_matches_manifest", str(len(common_mask_columns)))

    per_basin_recorded = manifest.get("per_basin")
    if isinstance(per_basin_recorded, dict):
        per_basin_mismatches = []
        for basin_id in common_mask_basins:
            basin_entry = per_basin_recorded.get(basin_id)
            recorded = basin_entry.get("n_imputed") if isinstance(basin_entry, dict) else None
            if recorded is None:
                continue
            actual_count = int(mask_subset.loc[basin_id].astype(bool).sum())
            if int(recorded) != actual_count:
                per_basin_mismatches.append((basin_id, recorded, actual_count))
        if per_basin_mismatches:
            report.error(
                "imputed_value_mask_per_basin_count_matches_manifest",
                f"{len(per_basin_mismatches)} basin(s) disagree with manifest: {per_basin_mismatches[:10]}",
            )
        else:
            report.ok("imputed_value_mask_per_basin_count_matches_manifest", str(len(common_mask_basins)))

    total_recorded = manifest.get("total_imputed_cells")
    if total_recorded is not None:
        actual_total = int(mask_subset.astype(bool).to_numpy().sum())
        if int(total_recorded) == actual_total:
            report.ok("imputed_value_mask_total_count_matches_manifest", str(actual_total))
        else:
            report.error(
                "imputed_value_mask_total_count_matches_manifest",
                f"manifest total={total_recorded}, mask actual total={actual_total}",
            )

    missing_fitted_value = []
    wrong_value = []
    n_checked = 0
    for basin_id in accepted_basin_ids:
        if basin_id not in mask_df.index or basin_id not in package_attrs.index:
            continue
        for col in model_input_columns:
            if col not in mask_df.columns or col not in package_attrs.columns:
                continue
            if not bool(mask_df.loc[basin_id, col]):
                continue
            n_checked += 1
            fitted_value = per_column.get(col, {}).get("fitted_value")
            if fitted_value is None or (isinstance(fitted_value, float) and np.isnan(fitted_value)):
                missing_fitted_value.append((basin_id, col))
                continue
            actual_value = float(package_attrs.loc[basin_id, col])
            if not np.isclose(actual_value, float(fitted_value), rtol=1e-9, atol=1e-12):
                wrong_value.append((basin_id, col))

    if missing_fitted_value:
        report.error(
            "imputed_values_have_manifest_fitted_value",
            f"{len(missing_fitted_value)} imputed cell(s) have no manifest fitted_value, e.g. {missing_fitted_value[:10]}",
        )
    else:
        report.ok("imputed_values_have_manifest_fitted_value", str(n_checked))

    if wrong_value:
        report.error(
            "imputed_values_placed_correctly",
            f"{len(wrong_value)} imputed cell(s) do not equal manifest fitted_value, e.g. {wrong_value[:10]}",
        )
    else:
        report.ok("imputed_values_placed_correctly", str(n_checked))


# ---------------------------------------------------------------------------
# Gap timestamp reconstruction audit
# ---------------------------------------------------------------------------


def audit_gap_reconstruction(
    report: AuditReport,
    *,
    package_root: Path,
    gap_inventory_csv_path,
    gap_product_scope: Sequence[str],
) -> list:
    try:
        gap_df = load_gap_inventory_independent(gap_inventory_csv_path)
        reconstructed = reconstruct_gap_timestamps_independent(gap_df, gap_product_scope)
    except PackageAuditError as exc:
        report.error("gap_inventory_readable", str(exc))
        return []
    report.ok("gap_inventory_readable")

    try:
        package_gap_timestamps = read_gap_timestamps_json_independent(package_root / "masks" / "gap_timestamps.json")
    except PackageAuditError as exc:
        report.error("package_gap_timestamps_readable", str(exc))
        return []
    report.ok("package_gap_timestamps_readable")

    expected_set = set(reconstructed)
    actual_set = set(package_gap_timestamps)
    if expected_set == actual_set:
        report.ok("gap_timestamps_reconstruction_matches_package", str(len(expected_set)))
    else:
        missing = sorted(str(t) for t in (expected_set - actual_set))[:10]
        extra = sorted(str(t) for t in (actual_set - expected_set))[:10]
        report.error(
            "gap_timestamps_reconstruction_matches_package",
            f"missing={missing} extra={extra} (expected {len(expected_set)}, got {len(actual_set)})",
        )

    if len(set(package_gap_timestamps)) == len(package_gap_timestamps):
        report.ok("package_gap_timestamps_no_duplicates")
    else:
        report.error("package_gap_timestamps_no_duplicates", "duplicate timestamp(s) found in masks/gap_timestamps.json")

    return sorted(reconstructed)


# ---------------------------------------------------------------------------
# QC CSV cross-check (non-authoritative)
# ---------------------------------------------------------------------------


def audit_qc_csv(
    report: AuditReport,
    *,
    package_root: Path,
    qc_evidence_root: Path,
    accepted_basin_ids: Sequence[str],
) -> None:
    manifest_path = qc_evidence_root / "csv_manifest.json"
    if not manifest_path.is_file():
        report.error("qc_csv_manifest_present", f"not found: {manifest_path}")
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.error("qc_csv_manifest_present", str(exc))
        return
    report.ok("qc_csv_manifest_present")

    for field in ("authoritative", "usable_for_training", "usable_for_package_reconstruction"):
        if manifest.get(field, True) is False:
            report.ok(f"qc_csv_declared_{field}_false")
        else:
            report.error(f"qc_csv_declared_{field}_false", f"csv_manifest.json does not declare {field}=false")

    files_entry = manifest.get("files", {}) if isinstance(manifest.get("files"), dict) else {}
    manifest_basin_ids = set(files_entry.keys())
    accepted_set = set(accepted_basin_ids)
    if manifest_basin_ids == accepted_set:
        report.ok("qc_csv_manifest_basin_membership", str(len(manifest_basin_ids)))
    else:
        report.error(
            "qc_csv_manifest_basin_membership",
            f"missing={sorted(accepted_set - manifest_basin_ids)} extra={sorted(manifest_basin_ids - accepted_set)}",
        )

    declared_count = manifest.get("basin_count")
    if declared_count is not None and int(declared_count) == len(manifest_basin_ids):
        report.ok("qc_csv_manifest_basin_count_matches_files", str(declared_count))
    elif declared_count is not None:
        report.error(
            "qc_csv_manifest_basin_count_matches_files",
            f"basin_count={declared_count}, len(files)={len(manifest_basin_ids)}",
        )

    csv_inspection_dir = qc_evidence_root / "csv_inspection"
    actual_csv_names = (
        {p.name for p in csv_inspection_dir.iterdir() if p.is_file() and p.suffix == ".csv"}
        if csv_inspection_dir.is_dir()
        else set()
    )
    expected_csv_names = {f"{basin_id}.csv" for basin_id in accepted_basin_ids}
    if actual_csv_names == expected_csv_names:
        report.ok("qc_csv_exact_file_membership", str(len(actual_csv_names)))
    else:
        report.error(
            "qc_csv_exact_file_membership",
            f"missing={sorted(expected_csv_names - actual_csv_names)} extra={sorted(actual_csv_names - expected_csv_names)}",
        )

    per_file_mismatches = []
    missing = []
    for basin_id in accepted_basin_ids:
        csv_path = qc_evidence_root / "csv_inspection" / f"{basin_id}.csv"
        nc_path = package_root / "time_series" / f"{basin_id}.nc"
        entry = files_entry.get(basin_id)
        if not csv_path.is_file() or not nc_path.is_file() or entry is None:
            missing.append(basin_id)
            continue

        actual_sha256 = sha256_file(csv_path)
        actual_size = csv_path.stat().st_size
        actual_row_count = int(len(pd.read_csv(csv_path)))
        if entry.get("sha256") != actual_sha256:
            per_file_mismatches.append((basin_id, "sha256", entry.get("sha256"), actual_sha256))
        if entry.get("size_bytes") is not None and int(entry.get("size_bytes")) != actual_size:
            per_file_mismatches.append((basin_id, "size_bytes", entry.get("size_bytes"), actual_size))
        if entry.get("row_count") is not None and int(entry.get("row_count")) != actual_row_count:
            per_file_mismatches.append((basin_id, "row_count", entry.get("row_count"), actual_row_count))
        for field in ("authoritative", "usable_for_training", "usable_for_package_reconstruction"):
            if entry.get(field, True) is not False:
                per_file_mismatches.append((basin_id, field, entry.get(field), False))

        csv_df = pd.read_csv(csv_path)
        disk = read_package_basin_netcdf_independent(nc_path)
        for name in EXPECTED_VARIABLES:
            if name not in csv_df.columns:
                missing.append(f"{basin_id}:{name}")
                continue
            csv_values = csv_df[name].to_numpy(dtype=np.float64)
            nc_values = np.asarray(disk["variables"][name], dtype=np.float64)
            check_id = f"qc_csv_matches_netcdf[{basin_id}][{name}]"
            if name in GAP_FLAG_VARIABLES:
                # Gap flags are exact binary/integer values in both the CSV
                # and the NetCDF (int8 on disk) -- no float32 quantization
                # is involved, so this is an exact comparison, not a
                # tolerance-based one.
                report.numeric(compare_float_arrays(check_id, nc_values, csv_values, rtol=0.0, atol=0.0))
            else:
                storage_dtype = np.dtype(disk["dtypes"][name])
                report.numeric(
                    compare_qc_csv_against_netcdf_storage(
                        check_id, nc_values, csv_values, storage_dtype=storage_dtype
                    )
                )
    if missing:
        report.error("qc_csv_files_present", f"missing: {missing[:10]}")
    else:
        report.ok("qc_csv_files_present")

    if per_file_mismatches:
        report.error(
            "qc_csv_manifest_entry_matches_disk",
            f"{len(per_file_mismatches)} mismatch(es), e.g. {per_file_mismatches[:10]}",
        )
    else:
        report.ok("qc_csv_manifest_entry_matches_disk", str(len(accepted_basin_ids)))


# ---------------------------------------------------------------------------
# Provenance manifest
# ---------------------------------------------------------------------------


def _get_auditor_git_commit(cwd=None) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd if cwd is not None else Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        commit = result.stdout.strip()
        return commit or None
    except Exception:
        return None


def resolve_auditor_git_commit(cwd) -> str:
    """Resolve the auditor's own executing-code git commit, or fail hard.

    Unlike :func:`_get_auditor_git_commit` (which returns ``None`` on any
    failure, used by non-full-audit callers), this raises
    :class:`PackageAuditError` if the commit cannot be resolved -- a full
    audit run must not silently proceed with an unidentified
    ``auditor_git_commit``.
    """
    commit = _get_auditor_git_commit(cwd)
    if not commit:
        raise PackageAuditError(f"could not resolve auditor git commit (cwd={cwd}); refusing to run full audit")
    return commit


def check_auditor_working_tree_clean(cwd) -> None:
    """Raise :class:`PackageAuditError` if the auditor's own repo is dirty.

    A resolvable ``auditor_git_commit`` only reliably identifies the code
    that executed if the working tree has no uncommitted changes on top of
    that commit. This is a hard failure for the full-audit path, not a
    warning.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except Exception as exc:
        raise PackageAuditError(f"could not check auditor working tree cleanliness (cwd={cwd}): {exc}") from exc
    if result.stdout.strip():
        raise PackageAuditError(
            f"auditor working tree is not clean (cwd={cwd}); refusing to run full audit with uncommitted "
            f"changes to the auditing code itself"
        )


def build_audit_manifest(
    *,
    build_git_commit: str,
    auditor_git_commit: Optional[str],
    policy_path,
    policy_sha256: str,
    basin_selection_path,
    basin_selection_sha256: str,
    prepared_static_sha256: Optional[str],
    static_column_manifest_sha256: Optional[str],
    imputation_manifest_sha256: Optional[str],
    imputed_value_mask_sha256: Optional[str],
    area_csv_sha256: Optional[str],
    forcing_root,
    qobs_root,
    gap_inventory_sha256: Optional[str],
    package_manifest_sha256: Optional[str],
    package_artifact_checksums: Mapping,
    forcing_source_checksums: Mapping,
    qobs_source_checksums: Mapping,
    audit_command: str,
    execution_timestamp_utc: str,
) -> dict:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "build_git_commit": build_git_commit,
        "auditor_git_commit": auditor_git_commit,
        "policy_path": str(policy_path),
        "policy_sha256": policy_sha256,
        "basin_selection_path": str(basin_selection_path),
        "basin_selection_sha256": basin_selection_sha256,
        "prepared_static_attributes_sha256": prepared_static_sha256,
        "static_column_manifest_sha256": static_column_manifest_sha256,
        "imputation_manifest_sha256": imputation_manifest_sha256,
        "imputed_value_mask_sha256": imputed_value_mask_sha256,
        "area_source_sha256": area_csv_sha256,
        "forcing_source_root": str(forcing_root) if forcing_root is not None else None,
        "qobs_source_root": str(qobs_root) if qobs_root is not None else None,
        "gap_inventory_sha256": gap_inventory_sha256,
        "package_manifest_sha256": package_manifest_sha256,
        "package_artifact_checksums": dict(package_artifact_checksums),
        "forcing_source_checksums": dict(forcing_source_checksums),
        "qobs_source_checksums": dict(qobs_source_checksums),
        "audit_command": audit_command,
        "execution_timestamp_utc": execution_timestamp_utc,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_preflight(
    *,
    package_root,
    policy_path=None,
    basin_selection_path=None,
    prepared_static_parquet_path=None,
    static_column_manifest_path=None,
    imputation_manifest_path=None,
    imputed_value_mask_path=None,
    forcing_root=None,
    qobs_root=None,
    area_csv_path=None,
    gap_inventory_csv_path=None,
    qc_evidence_root=None,
) -> tuple:
    """Read-only, no-computation existence/readability check.

    Verifies the package layout and that every supplied path exists, without
    opening NetCDF/parquet content or comparing any values. Intended to run
    quickly before the (expensive) full audit -- e.g. as a first h2o step.
    """
    report = AuditReport()
    package_root = Path(package_root)

    check_package_layout(report, package_root)

    candidates = {
        "policy_path": policy_path,
        "basin_selection_path": basin_selection_path,
        "prepared_static_parquet_path": prepared_static_parquet_path,
        "static_column_manifest_path": static_column_manifest_path,
        "imputation_manifest_path": imputation_manifest_path,
        "imputed_value_mask_path": imputed_value_mask_path,
        "forcing_root": forcing_root,
        "qobs_root": qobs_root,
        "area_csv_path": area_csv_path,
        "gap_inventory_csv_path": gap_inventory_csv_path,
        "qc_evidence_root": qc_evidence_root,
    }
    for label, value in candidates.items():
        if value is None:
            report.warn(f"preflight_path_supplied[{label}]", "not supplied")
            continue
        p = Path(value)
        if p.exists():
            report.ok(f"preflight_path_exists[{label}]", str(p))
        else:
            report.error(f"preflight_path_exists[{label}]", f"not found: {p}")

    diagnostics = {"candidates": {k: (str(v) if v is not None else None) for k, v in candidates.items()}}
    return report, diagnostics


def run_audit(
    *,
    package_root,
    policy: Mapping,
    policy_path,
    basin_selection_path,
    prepared_static_parquet_path,
    static_column_manifest_path,
    forcing_root,
    qobs_root,
    area_csv_path,
    gap_inventory_csv_path,
    imputation_manifest_path=None,
    imputed_value_mask_path=None,
    qc_evidence_root=None,
    build_git_commit: str,
    audit_command: str = "",
    dev_allow_missing_evidence: bool = False,
    auditor_repo_root=None,
) -> tuple:
    """Run the full independent Gate 4 audit against a real package + sources.

    ``imputation_manifest_path``, ``imputed_value_mask_path``, and
    ``qc_evidence_root`` are mandatory for a canonical full audit: a real
    Gate 4 run must not be able to PASS while skipping the imputation or QC
    evidence checks. Set ``dev_allow_missing_evidence=True`` only for
    isolated development/test use of this library function directly -- the
    production CLI (``--mode full``) never sets it, so canonical audits are
    always strict.

    ``auditor_repo_root`` identifies the git working tree whose commit is
    bound into the provenance manifest as ``auditor_git_commit``; it must be
    resolvable and clean or the run refuses to proceed (see
    :func:`resolve_auditor_git_commit` / :func:`check_auditor_working_tree_clean`).
    Defaults to this module's own repository, which is what the CLI uses.

    Returns ``(report, diagnostics)``. ``report.status`` is ``"PASS"`` iff no
    ERROR-severity check failed. ``diagnostics`` carries the JSON-serializable
    detail (numeric results, resolved basin lists, the provenance manifest)
    for :func:`write_audit_outputs`.
    """
    if not dev_allow_missing_evidence:
        missing_evidence = []
        if imputation_manifest_path is None:
            missing_evidence.append("imputation_manifest_path")
        if imputed_value_mask_path is None:
            missing_evidence.append("imputed_value_mask_path")
        if qc_evidence_root is None:
            missing_evidence.append("qc_evidence_root")
        if missing_evidence:
            raise PackageAuditError(
                f"canonical full audit requires {missing_evidence}; a real Gate 4 audit must not skip "
                f"imputation/QC evidence checks (pass dev_allow_missing_evidence=True only for isolated "
                f"development/test use)"
            )

    auditor_repo_root = Path(auditor_repo_root) if auditor_repo_root is not None else Path(__file__).resolve().parent
    auditor_git_commit = resolve_auditor_git_commit(auditor_repo_root)
    check_auditor_working_tree_clean(auditor_repo_root)

    report = AuditReport()
    package_root = Path(package_root)

    check_package_layout(report, package_root)

    accepted_basin_ids = read_basin_ids_independent(basin_selection_path)
    package_basin_ids = check_basin_membership(report, package_root, accepted_basin_ids)
    check_exact_package_layout(report, package_root, accepted_basin_ids)

    manifest = check_checksums_and_manifest(report, package_root)
    run_provenance = read_run_provenance_json_independent(report, package_root)

    historical_lineage_recognized = package_declares_historical_v001_compatibility_lineage(
        manifest, run_provenance
    )
    expected_schema_from_manifest = resolve_expected_netcdf_package_schema_independent(
        report,
        source_label="package_manifest",
        declared=manifest,
        historical_lineage_recognized=historical_lineage_recognized,
    )
    expected_schema_from_provenance = resolve_expected_netcdf_package_schema_independent(
        report,
        source_label="run_provenance",
        declared=run_provenance,
        historical_lineage_recognized=historical_lineage_recognized,
    )

    expected_index = derive_expected_index_independent(policy)
    area_by_basin = read_area_csv_independent(area_csv_path)
    package_float32_rtol = float(policy["audit"]["package_float32_rtol"])

    disk_schema_identities = []
    for basin_id in accepted_basin_ids:
        nc_path = package_root / "time_series" / f"{basin_id}.nc"
        if basin_id not in area_by_basin:
            report.error(f"area_available[{basin_id}]", "no DRAIN_SQKM area found in area source")
            continue
        identity = audit_basin_netcdf(
            report,
            basin_id=basin_id,
            nc_path=nc_path,
            expected_index=expected_index,
            forcing_root=forcing_root,
            qobs_root=qobs_root,
            area_km2=area_by_basin[basin_id],
            package_float32_rtol=package_float32_rtol,
            expected_schema=expected_schema_from_manifest,
            expected_schema_from_provenance=expected_schema_from_provenance,
        )
        if identity is not None:
            disk_schema_identities.append(identity)

    # Sorted via a string-coercing key rather than raw tuple comparison: a
    # basin file whose coordinate could not be determined (both/neither
    # present -- already reported as its own ERROR by
    # netcdf_temporal_coordinate_present) contributes an identity tuple with
    # a ``None`` coordinate_name, and Python cannot order ``None`` against
    # ``str`` when a plain tuple sort compares that position.
    distinct_disk_schema_identities = sorted(
        set(disk_schema_identities),
        key=lambda identity: tuple("" if v is None else str(v) for v in identity),
    )
    if len(distinct_disk_schema_identities) <= 1:
        report.ok("package_all_basins_same_netcdf_schema", str(distinct_disk_schema_identities))
    else:
        report.error(
            "package_all_basins_same_netcdf_schema",
            f"mixed NetCDF package schema identities across basin files: {distinct_disk_schema_identities}",
        )

    audit_static_attributes(
        report,
        package_root=package_root,
        accepted_basin_ids=accepted_basin_ids,
        prepared_static_parquet_path=prepared_static_parquet_path,
        static_column_manifest_path=static_column_manifest_path,
        imputation_manifest_path=imputation_manifest_path,
        imputed_value_mask_path=imputed_value_mask_path,
    )

    gap_product_scope = resolve_gap_product_scope_independent(policy)
    audit_gap_reconstruction(
        report,
        package_root=package_root,
        gap_inventory_csv_path=gap_inventory_csv_path,
        gap_product_scope=gap_product_scope,
    )

    if qc_evidence_root is not None:
        audit_qc_csv(
            report,
            package_root=package_root,
            qc_evidence_root=Path(qc_evidence_root),
            accepted_basin_ids=accepted_basin_ids,
        )
    else:
        report.warn("qc_csv_cross_check", "no QC evidence root supplied; check skipped")

    policy_sha256 = sha256_file(policy_path)
    basin_selection_sha256 = sha256_file(basin_selection_path)
    prepared_static_sha256 = sha256_file(prepared_static_parquet_path)
    static_column_manifest_sha256 = sha256_file(static_column_manifest_path)
    imputation_manifest_sha256 = sha256_file(imputation_manifest_path) if imputation_manifest_path else None
    imputed_value_mask_sha256 = sha256_file(imputed_value_mask_path) if imputed_value_mask_path else None
    area_csv_sha256 = sha256_file(area_csv_path)
    gap_inventory_sha256 = sha256_file(gap_inventory_csv_path)
    package_manifest_path = package_root / "manifests" / "package_manifest.json"
    package_manifest_sha256 = sha256_file(package_manifest_path) if package_manifest_path.is_file() else None

    package_artifact_checksums = compute_package_artifact_checksums_independent(package_root, accepted_basin_ids)
    forcing_source_checksums = compute_source_checksums_independent(forcing_root, accepted_basin_ids, kind="forcing")
    qobs_source_checksums = compute_source_checksums_independent(qobs_root, accepted_basin_ids, kind="qobs")

    audit_manifest = build_audit_manifest(
        build_git_commit=build_git_commit,
        auditor_git_commit=auditor_git_commit,
        policy_path=policy_path,
        policy_sha256=policy_sha256,
        basin_selection_path=basin_selection_path,
        basin_selection_sha256=basin_selection_sha256,
        prepared_static_sha256=prepared_static_sha256,
        static_column_manifest_sha256=static_column_manifest_sha256,
        imputation_manifest_sha256=imputation_manifest_sha256,
        imputed_value_mask_sha256=imputed_value_mask_sha256,
        area_csv_sha256=area_csv_sha256,
        forcing_root=forcing_root,
        qobs_root=qobs_root,
        gap_inventory_sha256=gap_inventory_sha256,
        package_manifest_sha256=package_manifest_sha256,
        package_artifact_checksums=package_artifact_checksums,
        forcing_source_checksums=forcing_source_checksums,
        qobs_source_checksums=qobs_source_checksums,
        audit_command=audit_command,
        execution_timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    diagnostics = {
        "accepted_basin_ids": list(accepted_basin_ids),
        "package_basin_ids": list(package_basin_ids),
        "numeric_results": [r.as_dict() for r in report.numeric_results],
        "audit_manifest": audit_manifest,
    }
    return report, diagnostics


def write_audit_outputs(out_dir, report: AuditReport, diagnostics: dict, *, overwrite: bool = False, log_lines: Optional[Sequence[str]] = None) -> dict:
    """Write the Gate 4 generated-output bundle.

    Writes ``audit_results.json``, ``audit_report.md``, ``audit_manifest.json``
    (patched with checksums of the generated result files), ``run.log``,
    ``file_checksums.csv`` (checksums of the generated files themselves), and
    a ``review_bundle/`` directory, all under ``out_dir``.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise PackageAuditError(f"output directory already exists and is non-empty: {out_dir} (pass overwrite=True)")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "status": report.status,
        "error_count": report.error_count,
        "warning_count": report.warning_count,
        "ok_count": report.ok_count,
        "checks": [
            {"severity": r.severity, "check_id": r.check_id, "message": r.message} for r in report.records
        ],
        "numeric_results": diagnostics.get("numeric_results", []),
        "failed_checks": report.failed_messages(),
    }
    results_path = out_dir / "audit_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    report_lines = [
        f"# Stage 1 Compact Scientific Package Independent Audit ({report.status})",
        "",
        f"- errors: {report.error_count}",
        f"- warnings: {report.warning_count}",
        f"- ok: {report.ok_count}",
        "",
    ]
    if report.error_count:
        report_lines.append("## Failed checks")
        for msg in report.failed_messages():
            report_lines.append(f"- {msg}")
        report_lines.append("")
    report_md_path = out_dir / "audit_report.md"
    report_md_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    log_path = out_dir / "run.log"
    log_text = "\n".join(log_lines) + "\n" if log_lines else ""
    log_path.write_text(log_text, encoding="utf-8")

    review_bundle_dir = out_dir / "review_bundle"
    review_bundle_dir.mkdir(parents=True, exist_ok=True)
    (review_bundle_dir / "audit_report.md").write_text(report_md_path.read_text(encoding="utf-8"), encoding="utf-8")
    (review_bundle_dir / "audit_results_summary.json").write_text(
        json.dumps(
            {
                "status": report.status,
                "error_count": report.error_count,
                "warning_count": report.warning_count,
                "ok_count": report.ok_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = dict(diagnostics.get("audit_manifest") or {})
    manifest["generated_output_checksums"] = {
        p.name: {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
        for p in (results_path, report_md_path, log_path)
    }
    manifest_path = out_dir / "audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    checksums_csv_path = out_dir / "file_checksums.csv"
    rows = ["relative_path,sha256,size_bytes"]
    for p in (results_path, report_md_path, manifest_path, log_path):
        rows.append(f"{p.name},{sha256_file(p)},{p.stat().st_size}")
    checksums_csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    return {
        "audit_results.json": results_path,
        "audit_report.md": report_md_path,
        "audit_manifest.json": manifest_path,
        "file_checksums.csv": checksums_csv_path,
        "run.log": log_path,
        "review_bundle": review_bundle_dir,
    }

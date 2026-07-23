"""Stage 1 local Compact Scientific Package builder (Gate 3A: filesystem
orchestration over synthetic/injectable fixtures).

Orchestrates the already-committed Gate 1 (:mod:`src.baseline.package_assembly`)
and Gate 2 (:mod:`src.baseline.package_netcdf`) APIs, plus the established
:mod:`src.baseline.gap_mask_io` writer/validator, into one filesystem
package build. This module performs no scientific transformation of its
own -- no unit conversion, no lead-target shifting, no gap-fill, no
imputation fitting -- all of that already lives in the layers it calls.

Two physically separate output trees:
- the authoritative package (``time_series/``, ``attributes/``, ``basins/``,
  ``masks/``, ``manifests/``, ``run_provenance.json``);
- a non-authoritative evidence tree (``csv_inspection/``, ``csv_manifest.json``,
  ``build_summary.json``) for human QC only, never read back into package
  construction and never referenced as an authoritative or training input.

Source loading is abstracted behind :class:`BasinSourceTables` and an
injectable ``load_basin_source`` callback (basin_id -> BasinSourceTables) so
this module has no hard-coded remote (h2o/Moriah), username, or Windows
paths, and so local synthetic tests never need real forcing/qobs archives.
:func:`default_local_basin_source_loader` is one concrete, thin loader for
the established local per-basin ``time_series/<id>.parquet`` /
``time_series/<id>.nc`` layout (see the frozen historical
``scripts/build_stage1_nh_package.py`` for that convention) -- it performs
no gap-fill, no reindexing, no interpolation; validation of the resulting
tables is entirely delegated to Gate 1.

Build is atomic at the directory level: everything is written into a
temporary sibling directory (same parent, same filesystem as the final
package root) and validated in place; only a fully validated package is
promoted with a directory rename. Any failure -- a bad source table, a
Gate 1/Gate 2 rejection, an I/O error -- removes the temporary directory and
leaves any pre-existing final package untouched.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .gap_mask_io import MRMS_PRODUCT, RTMA_PRODUCT, write_gap_timestamps_json
from .lead_targets import DEFAULT_LEADS_HOURS, DEFAULT_VARIABLE_NAME_TEMPLATE, variable_name_for_lead
from .package_assembly import DYNAMIC_INPUTS, RAW_TARGET_VARIABLE, assemble_basin_package_table
from .package_netcdf import (
    DEFAULT_PACKAGE_NETCDF_SCHEMA,
    LEGACY_COMPACT_V001_SCHEMA,
    SCIENTIFIC_V002_SCHEMA,
    PackageNetCDFSchema,
    build_basin_dataset,
    validate_basin_netcdf_file,
    write_basin_dataset_netcdf,
)
from .splits import sha256_of
from .staid import normalize_staid
from .static_preparation import model_input_columns_from_manifest
from .validity_mask import ValidityMaskError, bad_hour_mask_from_timestamps

__all__ = [
    "PackageBuilderError",
    "BasinSourceTables",
    "PackageBuildResult",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "DEFAULT_FORBIDDEN_STATIC_COLUMNS",
    "QC_CSV_ROLE",
    "QC_CSV_FLOAT_FORMAT",
    "build_compact_scientific_package",
    "derive_expected_index_from_policy",
    "resolve_gap_product_scope",
    "read_basin_ids_file",
    "read_area_csv",
    "default_local_basin_source_loader",
]

SCHEMA_NAME = "stage1_compact_scientific_package_builder_v001"
SCHEMA_VERSION = 1

#: Closed, explicit mapping from the selected registered NetCDF package
#: schema to the package's own role identity. Deliberately never inferred
#: from basin count, output path, basin selection, or scientific policy --
#: only from which fixed NetCDF schema was selected for the build.
_PACKAGE_ROLE_BY_NETCDF_SCHEMA_NAME = {
    LEGACY_COMPACT_V001_SCHEMA.name: "stage1_compact_scientific_package",
    SCIENTIFIC_V002_SCHEMA.name: "stage1_scientific_package",
}


def _resolve_package_role(package_netcdf_schema: PackageNetCDFSchema) -> str:
    try:
        return _PACKAGE_ROLE_BY_NETCDF_SCHEMA_NAME[package_netcdf_schema.name]
    except KeyError:
        raise PackageBuilderError(
            f"no package_role mapping registered for NetCDF schema {package_netcdf_schema.name!r}"
        ) from None

#: Mirrors config/stage1_scientific_baseline_v001.yaml
#: ``static_attributes.forbidden_model_inputs``.
DEFAULT_FORBIDDEN_STATIC_COLUMNS: tuple[str, ...] = ("STATE", "HUC02", "LAT_GAGE", "LNG_GAGE")

QC_CSV_ROLE = "human_readable_qc_non_authoritative_not_for_training_or_reconstruction"

#: 17 significant digits round-trips any float64 value with negligible loss;
#: verification below tolerates a tiny residual (formatting, not data, error).
QC_CSV_FLOAT_FORMAT = "%.17g"
QC_CSV_ROUNDTRIP_RTOL = 1e-9
QC_CSV_ROUNDTRIP_ATOL = 1e-12


class PackageBuilderError(RuntimeError):
    """Raised for any Gate 3A package-build contract violation or basin failure."""


@dataclass(frozen=True)
class BasinSourceTables:
    """One basin's already-loaded, in-memory Gate 1 inputs.

    forcing/qobs/area_km2 are passed through to
    :func:`src.baseline.package_assembly.assemble_basin_package_table`
    unchanged -- this dataclass is purely a loader-callback return contract,
    not a validation layer.
    """

    forcing: pd.DataFrame
    qobs: object
    area_km2: float


@dataclass(frozen=True)
class PackageBuildResult:
    """Outcome of a (possibly dry-run) :func:`build_compact_scientific_package` call."""

    package_root: Path
    evidence_root: Path | None
    basin_ids: tuple
    manifest: dict
    dry_run: bool


# ---------------------------------------------------------------------------
# Policy timeline derivation (production CLI path only -- tests use an
# explicit expected_index test seam instead, per the Gate 3A contract).
# ---------------------------------------------------------------------------


def derive_expected_index_from_policy(policy: Mapping) -> pd.DatetimeIndex:
    """Derive + validate the exact canonical Stage 1 hourly timeline from a
    validated policy mapping (``period.start_utc`` / ``end_utc`` /
    ``expected_hours``). Never reads or repairs -- fails loud if the
    resulting grid does not have the pinned row count."""
    start = pd.Timestamp(policy["period"]["start_utc"])
    end = pd.Timestamp(policy["period"]["end_utc"])
    if start.tzinfo is not None:
        start = start.tz_convert("UTC").tz_localize(None)
    if end.tzinfo is not None:
        end = end.tz_convert("UTC").tz_localize(None)
    index = pd.date_range(start, end, freq="h")
    expected_hours = policy["period"]["expected_hours"]
    if len(index) != expected_hours:
        raise PackageBuilderError(
            f"policy period does not produce the pinned {expected_hours} hourly rows: "
            f"got {len(index)} (start={start}, end={end})"
        )
    return index


# ---------------------------------------------------------------------------
# Gap-product scope (which gap_mask_io product name(s) belong in the runtime
# exclusion artifact) -- policy-driven, never hard-coded by a caller.
# ---------------------------------------------------------------------------


def resolve_gap_product_scope(policy: Mapping) -> tuple:
    """Derive the accepted gap-inventory product scope from a validated
    policy's ``gap_policy`` block.

    MRMS gaps are always in scope (the signed-off Policy B hard-exclusion
    driver). RTMA gaps are additionally in scope iff
    ``gap_policy.include_rtma_in_history_mask`` is exactly ``True``; any
    other value (missing key, non-bool, ``False`` would simply mean
    MRMS-only) that does not resolve to an unambiguous bool fails loudly
    rather than silently defaulting.
    """
    if "gap_policy" not in policy:
        raise PackageBuilderError("policy is missing the required 'gap_policy' block")
    gap_policy = policy["gap_policy"]
    if "include_rtma_in_history_mask" not in gap_policy:
        raise PackageBuilderError(
            "policy.gap_policy is missing required field 'include_rtma_in_history_mask'"
        )
    include_rtma = gap_policy["include_rtma_in_history_mask"]
    if not isinstance(include_rtma, bool):
        raise PackageBuilderError(
            "policy.gap_policy.include_rtma_in_history_mask must be an unambiguous bool; "
            f"got unsupported value {include_rtma!r} ({type(include_rtma).__name__})"
        )
    if include_rtma:
        return (MRMS_PRODUCT, RTMA_PRODUCT)
    return (MRMS_PRODUCT,)


# ---------------------------------------------------------------------------
# Basin-ID list validation (never pad/infer/strip -- staid.py is authoritative)
# ---------------------------------------------------------------------------


def _validate_basin_id_list(basin_ids: Sequence[str]) -> tuple:
    if not isinstance(basin_ids, (list, tuple)):
        raise PackageBuilderError(f"basin_ids must be a list/tuple, got {type(basin_ids)!r}")
    if len(basin_ids) == 0:
        raise PackageBuilderError("basin_ids must not be empty")
    seen = set()
    validated = []
    for raw in basin_ids:
        if not isinstance(raw, str):
            raise PackageBuilderError(f"basin id must be a string, got {raw!r} ({type(raw).__name__})")
        try:
            normalized = normalize_staid(raw)
        except (TypeError, ValueError) as exc:
            raise PackageBuilderError(f"invalid basin id {raw!r}: {exc}") from exc
        if normalized != raw:
            raise PackageBuilderError(
                f"basin id {raw!r} is not already in canonical station-ID form "
                f"(would normalize to {normalized!r}); this builder never pads or repairs IDs"
            )
        if raw in seen:
            raise PackageBuilderError(f"duplicate basin id: {raw!r}")
        seen.add(raw)
        validated.append(raw)
    return tuple(validated)


# ---------------------------------------------------------------------------
# Static-attribute validation (already-prepared compact matrix only -- no
# imputation fitting here)
# ---------------------------------------------------------------------------


def _validate_static_attributes(
    static_attributes: pd.DataFrame,
    basin_ids: tuple,
    model_input_columns: Sequence[str],
    forbidden_columns: Sequence[str],
) -> pd.DataFrame:
    if not isinstance(static_attributes, pd.DataFrame):
        raise PackageBuilderError(
            f"static_attributes must be a pandas.DataFrame, got {type(static_attributes)!r}"
        )

    model_input_columns = list(model_input_columns)
    if len(model_input_columns) != len(set(model_input_columns)):
        raise PackageBuilderError("static_model_input_columns must not contain duplicate column names")
    forbidden_overlap = sorted(set(model_input_columns) & set(forbidden_columns))
    if forbidden_overlap:
        raise PackageBuilderError(
            f"static_model_input_columns contains forbidden field(s): {forbidden_overlap}"
        )

    index_values = list(static_attributes.index)
    if len(index_values) != len(set(index_values)):
        dupes = sorted({v for v in index_values if index_values.count(v) > 1})
        raise PackageBuilderError(f"static_attributes contains duplicate basin id(s) in its index: {dupes}")
    for v in index_values:
        if not isinstance(v, str):
            raise PackageBuilderError(
                f"static_attributes index must contain string basin IDs, got {v!r} ({type(v).__name__})"
            )
        try:
            normalized = normalize_staid(v)
        except (TypeError, ValueError) as exc:
            raise PackageBuilderError(f"static_attributes has invalid basin id {v!r}: {exc}") from exc
        if normalized != v:
            raise PackageBuilderError(
                f"static_attributes basin id {v!r} is not already in canonical form "
                f"(would normalize to {normalized!r})"
            )

    expected_set = set(basin_ids)
    actual_set = set(index_values)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing or extra:
        raise PackageBuilderError(
            f"static_attributes basin membership mismatch: missing {missing}, unexpected extra {extra}"
        )

    actual_columns = list(static_attributes.columns)
    if actual_columns != model_input_columns:
        missing_cols = sorted(set(model_input_columns) - set(actual_columns))
        extra_cols = sorted(set(actual_columns) - set(model_input_columns))
        if missing_cols or extra_cols:
            raise PackageBuilderError(
                f"static_attributes column contract mismatch: missing {missing_cols}, "
                f"unapproved extra {extra_cols}"
            )
        raise PackageBuilderError(
            "static_attributes column order does not match the required model-input "
            f"contract exactly; expected {model_input_columns}, got {actual_columns}"
        )

    ordered = static_attributes.loc[list(basin_ids)].copy()

    values = ordered.to_numpy(dtype=np.float64)
    n_nan = int(np.count_nonzero(np.isnan(values)))
    if n_nan:
        raise PackageBuilderError(
            f"static_attributes contains {n_nan} NaN value(s); imputation must already be "
            "complete before this builder is called"
        )
    n_inf = int(np.count_nonzero(np.isinf(values)))
    if n_inf:
        raise PackageBuilderError(f"static_attributes contains {n_inf} infinite value(s)")

    return ordered


# ---------------------------------------------------------------------------
# Production static v002 contract (only enforced when a validated production
# policy is supplied; the small synthetic-schema test seam is unaffected).
# ---------------------------------------------------------------------------

_STATIC_IDENTITY_PROVENANCE_FIELDS = ("matrix_name", "sha256")


def _validate_production_static_contract(
    policy: Mapping,
    static_model_input_columns: Sequence[str],
    static_column_manifest: Mapping | None,
    static_attributes_provenance: Mapping | None,
) -> None:
    if "static_attributes" not in policy:
        raise PackageBuilderError("policy is missing the required 'static_attributes' block")
    spec = policy["static_attributes"]

    expected_count = spec.get("expected_model_input_columns")
    if expected_count is None:
        raise PackageBuilderError("policy.static_attributes.expected_model_input_columns is required")
    if len(static_model_input_columns) != expected_count:
        raise PackageBuilderError(
            f"static_model_input_columns has {len(static_model_input_columns)} column(s); "
            f"policy.static_attributes.expected_model_input_columns requires exactly {expected_count}"
        )

    if static_column_manifest is None:
        raise PackageBuilderError(
            "a validated production policy was supplied but static_column_manifest is missing; "
            "it is required to cross-check the model-input column contract"
        )
    allowed_role = spec.get("allowed_role", "model_input")
    manifest_columns = model_input_columns_from_manifest(static_column_manifest, role=allowed_role)
    if list(static_model_input_columns) != list(manifest_columns):
        raise PackageBuilderError(
            "static_model_input_columns does not exactly match the supplied static column "
            "manifest's model-input contract (name and/or order mismatch); this rejects, "
            "among other cases, an older/incompatible column manifest"
        )

    if static_attributes_provenance is None:
        raise PackageBuilderError(
            "a validated production policy was supplied but static_attributes_provenance is "
            f"missing; it must record the required identity field(s) {_STATIC_IDENTITY_PROVENANCE_FIELDS}"
        )
    missing_fields = [f for f in _STATIC_IDENTITY_PROVENANCE_FIELDS if f not in static_attributes_provenance]
    if missing_fields:
        raise PackageBuilderError(
            f"static_attributes_provenance is missing required identity field(s) {missing_fields} "
            "for a production policy build"
        )
    for field in _STATIC_IDENTITY_PROVENANCE_FIELDS:
        expected_value = spec.get(field)
        if expected_value is not None and static_attributes_provenance[field] != expected_value:
            raise PackageBuilderError(
                f"static_attributes_provenance[{field!r}] = {static_attributes_provenance[field]!r} "
                f"does not match policy.static_attributes.{field} = {expected_value!r}"
            )


# ---------------------------------------------------------------------------
# Prepared (compact/imputed) static artifact identity -- independent of the
# canonical population-matrix identity checked above. The artifact actually
# supplied to this builder (e.g. a 32-row compact imputed matrix) is a
# legitimately different file from the canonical matrix it was derived from;
# their SHA-256 values are never required to agree. When an imputation
# preparation manifest (as written by
# ``static_preparation.write_imputation_artifacts``) is supplied, its
# declared ``artifact_sha256["imputed_static_attributes.parquet"]`` must
# agree with the actual file's checksum.
# ---------------------------------------------------------------------------

_PREPARED_STATIC_IDENTITY_FIELDS = ("sha256",)
_PREPARED_STATIC_MANIFEST_ARTIFACT_KEY = "imputed_static_attributes.parquet"


def _validate_prepared_static_artifact(
    prepared_static_attributes_provenance: Mapping | None,
    static_preparation_manifest: Mapping | None,
) -> None:
    if prepared_static_attributes_provenance is None:
        raise PackageBuilderError(
            "a validated production policy was supplied but "
            "prepared_static_attributes_provenance is missing; it must record the actually "
            "supplied (prepared/compact) static-attributes file's own identity field(s) "
            f"{_PREPARED_STATIC_IDENTITY_FIELDS}, independent of the canonical population "
            "matrix identity"
        )
    missing_fields = [
        f for f in _PREPARED_STATIC_IDENTITY_FIELDS if f not in prepared_static_attributes_provenance
    ]
    if missing_fields:
        raise PackageBuilderError(
            f"prepared_static_attributes_provenance is missing required identity field(s) "
            f"{missing_fields} for a production policy build"
        )

    if static_preparation_manifest is None:
        return

    artifact_sha256 = static_preparation_manifest.get("artifact_sha256")
    if not isinstance(artifact_sha256, Mapping) or _PREPARED_STATIC_MANIFEST_ARTIFACT_KEY not in artifact_sha256:
        raise PackageBuilderError(
            "static_preparation_manifest is missing the expected "
            f"artifact_sha256[{_PREPARED_STATIC_MANIFEST_ARTIFACT_KEY!r}] field"
        )
    manifest_sha256 = artifact_sha256[_PREPARED_STATIC_MANIFEST_ARTIFACT_KEY]
    actual_sha256 = prepared_static_attributes_provenance["sha256"]
    if manifest_sha256 != actual_sha256:
        raise PackageBuilderError(
            "prepared static artifact checksum mismatch: the actual "
            f"{_PREPARED_STATIC_MANIFEST_ARTIFACT_KEY} file's SHA-256 ({actual_sha256!r}) does "
            f"not match the preparation manifest's declared checksum ({manifest_sha256!r})"
        )


# ---------------------------------------------------------------------------
# Gap-timestamp validation (reuses validity_mask.py rather than inventing a
# second schema)
# ---------------------------------------------------------------------------


def _normalize_gap_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _validate_and_sort_gap_timestamps(gap_timestamps: Sequence, expected_index: pd.DatetimeIndex) -> list:
    normalized = [_normalize_gap_timestamp(v) for v in gap_timestamps]
    if len(normalized) != len(set(normalized)):
        dupes = sorted({str(t) for t in normalized if normalized.count(t) > 1})
        raise PackageBuilderError(f"gap_timestamps contains duplicate timestamp(s): {dupes[:10]}")
    try:
        bad_hour_mask_from_timestamps(expected_index, normalized, on_out_of_range="error")
    except ValidityMaskError as exc:
        raise PackageBuilderError(
            f"gap_timestamps failed validation against the package timeline: {exc}"
        ) from exc
    return sorted(normalized)


# ---------------------------------------------------------------------------
# QC CSV export (derived evidence only -- never re-read into the package)
# ---------------------------------------------------------------------------


def _write_and_verify_qc_csv(table: pd.DataFrame, gauge_id: str, evidence_tmp_dir: Path) -> dict:
    csv_dir = evidence_tmp_dir / "csv_inspection"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{gauge_id}.csv"

    export_df = table.reset_index()
    export_df.columns = ["time"] + list(table.columns)
    export_df["time"] = pd.DatetimeIndex(export_df["time"]).strftime("%Y-%m-%dT%H:%M:%SZ")
    export_df.to_csv(csv_path, index=False, float_format=QC_CSV_FLOAT_FORMAT)

    # Self-check read-back: verification only, never fed back into the package.
    reread = pd.read_csv(csv_path)
    if len(reread) != len(table):
        raise PackageBuilderError(
            f"QC CSV row-count self-check failed for basin {gauge_id!r}: "
            f"wrote {len(table)}, read back {len(reread)}"
        )
    if list(reread.columns) != ["time"] + list(table.columns):
        raise PackageBuilderError(f"QC CSV column self-check failed for basin {gauge_id!r}")

    expected_time = export_df["time"].to_numpy()
    actual_time = reread["time"].to_numpy()
    if not np.array_equal(expected_time, actual_time):
        n_mismatched = int(np.count_nonzero(expected_time != actual_time))
        raise PackageBuilderError(
            f"QC CSV timestamp self-check failed for basin {gauge_id!r}: "
            f"{n_mismatched} row(s) do not match the source table's timestamps"
        )

    for col in table.columns:
        expected_values = table[col].to_numpy(dtype=np.float64)
        actual_values = reread[col].to_numpy(dtype=np.float64)
        expected_nan_mask = np.isnan(expected_values)
        actual_nan_mask = np.isnan(actual_values)
        if not np.array_equal(expected_nan_mask, actual_nan_mask):
            raise PackageBuilderError(
                f"QC CSV NaN-mask self-check failed for basin {gauge_id!r}, column {col!r}: "
                f"expected {int(expected_nan_mask.sum())} NaN(s), got {int(actual_nan_mask.sum())}"
            )
        finite = ~expected_nan_mask
        if finite.any() and not np.allclose(
            expected_values[finite], actual_values[finite],
            rtol=QC_CSV_ROUNDTRIP_RTOL, atol=QC_CSV_ROUNDTRIP_ATOL,
        ):
            raise PackageBuilderError(
                f"QC CSV finite-value round-trip self-check failed for basin {gauge_id!r}, "
                f"column {col!r}: exceeds documented tolerance "
                f"(rtol={QC_CSV_ROUNDTRIP_RTOL}, atol={QC_CSV_ROUNDTRIP_ATOL}, "
                f"float_format={QC_CSV_FLOAT_FORMAT!r})"
            )

    return {
        "relative_path": (Path("csv_inspection") / f"{gauge_id}.csv").as_posix(),
        "sha256": sha256_of(csv_path),
        "size_bytes": csv_path.stat().st_size,
        "row_count": int(len(table)),
        "role": QC_CSV_ROLE,
        "authoritative": False,
        "usable_for_training": False,
        "usable_for_package_reconstruction": False,
    }


# ---------------------------------------------------------------------------
# Package-level artifact writers (operate inside the temp build directory)
# ---------------------------------------------------------------------------


def _write_basin_ids_txt(tmp_package_dir: Path, basin_ids: Sequence[str]) -> Path:
    basins_dir = tmp_package_dir / "basins"
    basins_dir.mkdir(parents=True, exist_ok=True)
    path = basins_dir / "basin_ids.txt"
    path.write_text("\n".join(basin_ids) + "\n", encoding="utf-8")
    return path


def _write_attributes_csv(tmp_package_dir: Path, ordered_static_attributes: pd.DataFrame) -> Path:
    attrs_dir = tmp_package_dir / "attributes"
    attrs_dir.mkdir(parents=True, exist_ok=True)
    path = attrs_dir / "attributes.csv"
    out = ordered_static_attributes.reset_index()
    out = out.rename(columns={out.columns[0]: "gauge_id"})
    out.to_csv(path, index=False)
    return path


def _write_gap_timestamps(tmp_package_dir: Path, sorted_gap_timestamps: list) -> Path:
    path = tmp_package_dir / "masks" / "gap_timestamps.json"
    write_gap_timestamps_json(sorted_gap_timestamps, path)
    return path


def _collect_checksums(tmp_package_dir: Path, per_basin_files, attrs_path: Path, basins_path: Path, gap_path: Path) -> dict:
    checksums = {}
    for _basin_id, rel_path, abs_path in per_basin_files:
        checksums[rel_path.as_posix()] = {
            "sha256": sha256_of(abs_path),
            "size_bytes": abs_path.stat().st_size,
            "artifact_role": "authoritative_time_series",
        }
    checksums["attributes/attributes.csv"] = {
        "sha256": sha256_of(attrs_path),
        "size_bytes": attrs_path.stat().st_size,
        "artifact_role": "authoritative_static_attributes",
    }
    checksums["basins/basin_ids.txt"] = {
        "sha256": sha256_of(basins_path),
        "size_bytes": basins_path.stat().st_size,
        "artifact_role": "authoritative_basin_list",
    }
    checksums["masks/gap_timestamps.json"] = {
        "sha256": sha256_of(gap_path),
        "size_bytes": gap_path.stat().st_size,
        "artifact_role": "authoritative_gap_mask",
    }
    return checksums


def _build_manifest(
    *,
    basin_ids: tuple,
    expected_index: pd.DatetimeIndex,
    static_model_input_columns: list,
    per_basin_files,
    checksums: dict,
    n_gap_timestamps: int,
    policy: dict | None,
    policy_provenance: dict | None,
    static_attributes_provenance: dict | None,
    basin_selection_provenance: dict | None,
    gap_inventory_provenance: dict | None,
    gap_product_scope: tuple | None,
    write_qc_csv: bool,
    prepared_static_attributes_provenance: dict | None = None,
    package_netcdf_schema: PackageNetCDFSchema = DEFAULT_PACKAGE_NETCDF_SCHEMA,
) -> dict:
    lead_variable_names = [
        variable_name_for_lead(h, DEFAULT_VARIABLE_NAME_TEMPLATE) for h in DEFAULT_LEADS_HOURS
    ]
    lead_targets = [
        {"name": name, "lead_hours": h} for name, h in zip(lead_variable_names, DEFAULT_LEADS_HOURS)
    ]
    static_columns_sha256 = hashlib.sha256(
        "\n".join(static_model_input_columns).encode("utf-8")
    ).hexdigest()

    policy_identity = dict(policy_provenance) if policy_provenance else {}
    if policy is not None:
        policy_identity.setdefault("policy_name", policy.get("policy_name"))
        policy_identity.setdefault("policy_version", policy.get("policy_version"))

    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "package_role": _resolve_package_role(package_netcdf_schema),
        "netcdf_package_schema_name": package_netcdf_schema.name,
        "netcdf_package_schema_version": package_netcdf_schema.version,
        "netcdf_time_coordinate": package_netcdf_schema.coordinate_name,
        "basin_count": len(basin_ids),
        "basin_ids": list(basin_ids),
        "timeline": {
            "start": expected_index[0].isoformat(),
            "end": expected_index[-1].isoformat(),
            "rows": int(len(expected_index)),
            "frequency": "hourly",
        },
        "dynamic_variables": list(DYNAMIC_INPUTS),
        "raw_target_variable": RAW_TARGET_VARIABLE,
        "lead_targets": lead_targets,
        "static_model_input_count": len(static_model_input_columns),
        "static_model_input_columns": list(static_model_input_columns),
        "static_model_input_columns_sha256": static_columns_sha256,
        "policy_identity": policy_identity,
        "canonical_static_source": dict(static_attributes_provenance) if static_attributes_provenance else None,
        "prepared_static_artifact": (
            dict(prepared_static_attributes_provenance) if prepared_static_attributes_provenance else None
        ),
        "basin_selection_source": dict(basin_selection_provenance) if basin_selection_provenance else None,
        "gap_inventory_source": dict(gap_inventory_provenance) if gap_inventory_provenance else None,
        "gap_product_scope": list(gap_product_scope) if gap_product_scope is not None else None,
        "gap_timestamp_artifact": {
            "relative_path": "masks/gap_timestamps.json",
            "sha256": checksums["masks/gap_timestamps.json"]["sha256"],
            "count": n_gap_timestamps,
        },
        "per_basin_time_series": [
            {
                "basin_id": basin_id,
                "relative_path": rel_path.as_posix(),
                "sha256": checksums[rel_path.as_posix()]["sha256"],
                "size_bytes": checksums[rel_path.as_posix()]["size_bytes"],
            }
            for basin_id, rel_path, _abs_path in per_basin_files
        ],
        "artifact_roles": {
            "time_series/*.nc": "authoritative",
            "attributes/attributes.csv": "authoritative",
            "basins/basin_ids.txt": "authoritative",
            "masks/gap_timestamps.json": "authoritative",
            "manifests/*": "authoritative_metadata",
        },
        "qc_csv_enabled": bool(write_qc_csv),
        "qc_csv_role": QC_CSV_ROLE if write_qc_csv else None,
        "qc_csv_authoritative": False,
        "qc_csv_usable_for_training": False,
    }


def _write_manifests(tmp_package_dir: Path, manifest: dict, checksums: dict) -> None:
    manifests_dir = tmp_package_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    rows = ["relative_path,sha256,size_bytes,artifact_role"]
    for rel_path in sorted(checksums):
        entry = checksums[rel_path]
        rows.append(f"{rel_path},{entry['sha256']},{entry['size_bytes']},{entry['artifact_role']}")
    (manifests_dir / "file_checksums.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_run_provenance(tmp_package_dir: Path, manifest: dict, *, dry_run: bool) -> None:
    provenance = {
        "builder_module": "src.baseline.package_builder",
        "builder_schema_version": SCHEMA_VERSION,
        # Deprecated: this historically named field actually identifies the
        # builder-manifest schema (manifest["schema_name"]), never the
        # on-disk NetCDF package schema/coordinate. Kept, unchanged in
        # meaning, for backward compatibility with existing readers -- use
        # builder_manifest_schema_name/netcdf_package_schema_name below for
        # new code.
        "package_schema_name": manifest["schema_name"],
        "builder_manifest_schema_name": manifest["schema_name"],
        "builder_manifest_schema_version": manifest["schema_version"],
        "netcdf_package_schema_name": manifest["netcdf_package_schema_name"],
        "netcdf_package_schema_version": manifest["netcdf_package_schema_version"],
        "netcdf_time_coordinate": manifest["netcdf_time_coordinate"],
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dry_run": dry_run,
        "basin_count": manifest["basin_count"],
        "qc_csv_enabled": manifest["qc_csv_enabled"],
    }
    (tmp_package_dir / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )


def _write_evidence_manifests(tmp_evidence_dir: Path, qc_csv_entries: dict, *, package_schema: str) -> None:
    csv_manifest = {
        "role": QC_CSV_ROLE,
        "authoritative": False,
        "usable_for_training": False,
        "usable_for_package_reconstruction": False,
        "package_schema_name": package_schema,
        "basin_count": len(qc_csv_entries),
        "files": qc_csv_entries,
    }
    (tmp_evidence_dir / "csv_manifest.json").write_text(
        json.dumps(csv_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    build_summary = {
        "role": "non_authoritative_build_summary",
        "basin_count": len(qc_csv_entries),
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (tmp_evidence_dir / "build_summary.json").write_text(
        json.dumps(build_summary, indent=2, sort_keys=True), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Atomic (coordinated) directory promotion
# ---------------------------------------------------------------------------


def _atomic_promote_all(promotions: Sequence[tuple], *, overwrite: bool) -> None:
    """Promote every ``(tmp_dir, destination)`` pair as one logical
    transaction: either all destinations end up replaced by their freshly
    built ``tmp_dir``, or none of them do.

    Used with a single pair when there is no QC-evidence tree, and with two
    pairs (package, then evidence) when ``write_qc_csv=True`` so the
    authoritative package and its evidence tree are never left with one
    promoted and the other not.
    """
    backups: list = []  # (destination, backup_or_None) for already-promoted items, in order
    promoted: list = []

    def _rollback() -> None:
        for destination, backup in reversed(backups):
            if destination not in promoted:
                continue
            shutil.rmtree(destination, ignore_errors=True)
            if backup is not None:
                os.rename(str(backup), str(destination))

    try:
        for tmp_dir, destination in promotions:
            existed = destination.exists()
            backup = None
            if existed:
                if not overwrite:
                    raise PackageBuilderError(f"destination already exists and overwrite=False: {destination}")
                backup = destination.parent / f".{destination.name}.pre-overwrite-backup"
                if backup.exists():
                    shutil.rmtree(backup)
                os.rename(str(destination), str(backup))
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.rename(str(tmp_dir), str(destination))
            except Exception:
                if backup is not None:
                    os.rename(str(backup), str(destination))
                raise
            backups.append((destination, backup))
            promoted.append(destination)
    except PackageBuilderError:
        _rollback()
        raise
    except Exception as exc:
        _rollback()
        raise PackageBuilderError(f"atomic promotion failed and was rolled back: {exc}") from exc

    for _destination, backup in backups:
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)


# ---------------------------------------------------------------------------
# Public API: main orchestration
# ---------------------------------------------------------------------------


def build_compact_scientific_package(
    *,
    basin_ids: Sequence[str],
    load_basin_source: Callable[[str], BasinSourceTables],
    static_attributes: pd.DataFrame,
    static_model_input_columns: Sequence[str],
    gap_timestamps: Sequence,
    expected_index: pd.DatetimeIndex,
    output_package_root,
    evidence_root=None,
    policy: dict | None = None,
    static_column_manifest: Mapping | None = None,
    gap_product_scope: Sequence[str] | None = None,
    forbidden_static_columns: Sequence[str] = DEFAULT_FORBIDDEN_STATIC_COLUMNS,
    write_qc_csv: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    policy_provenance: dict | None = None,
    static_attributes_provenance: dict | None = None,
    prepared_static_attributes_provenance: dict | None = None,
    static_preparation_manifest: Mapping | None = None,
    basin_selection_provenance: dict | None = None,
    gap_inventory_provenance: dict | None = None,
    package_netcdf_schema: PackageNetCDFSchema = DEFAULT_PACKAGE_NETCDF_SCHEMA,
) -> PackageBuildResult:
    """Build one Stage 1 Compact Scientific Package from already-loaded or
    injectable-loader-backed per-basin inputs.

    basin_ids: ordered, exact package membership -- also the deterministic
        NetCDF/attribute/basin-list write order.
    load_basin_source: ``basin_id -> BasinSourceTables`` callback; the only
        source-loading seam this module defines. For local synthetic tests,
        pass a small closure over in-memory fixtures. For the real local
        per-basin parquet/NetCDF layout, see
        :func:`default_local_basin_source_loader`.
    static_attributes: already-imputed compact static matrix, indexed by
        exact basin-ID strings, columns exactly ``static_model_input_columns``
        in that order. No imputation fitting happens here.
    expected_index: the exact canonical timeline every basin's forcing must
        equal (Gate 1 ``expected_index``). Production callers must derive
        this from the validated policy via
        :func:`derive_expected_index_from_policy`; tests may supply a short
        explicit index directly (the test seam this contract requires).
    output_package_root / evidence_root: destination directories for the
        authoritative package and the non-authoritative QC evidence tree
        respectively. Built into temporary sibling directories and promoted
        together as one transaction only after full validation.
    policy: when supplied, additionally enforces the production contract:
        the policy-derived gap-product scope (see
        :func:`resolve_gap_product_scope`) and the production static v002
        column-count/manifest/provenance contract (see
        ``static_column_manifest`` / ``static_attributes_provenance``).
        Omit for the small synthetic-schema test seam.
    static_column_manifest: the parsed static column-role manifest (as
        returned by ``static_preparation.load_column_manifest``); required
        whenever ``policy`` is supplied so ``static_model_input_columns``
        can be cross-checked against it exactly.
    static_attributes_provenance: canonical population-matrix identity
        (``matrix_name`` / ``sha256``) checked against
        ``policy.static_attributes``; required whenever ``policy`` is
        supplied. This is deliberately independent of
        ``prepared_static_attributes_provenance`` below -- the canonical
        matrix and the actually-supplied prepared/compact ``static_attributes``
        are legitimately different files and their checksums are never
        required to agree.
    prepared_static_attributes_provenance: identity (at minimum ``sha256``)
        of the actually-supplied ``static_attributes`` artifact itself
        (e.g. a compact imputed matrix derived from the canonical
        population matrix); required whenever ``policy`` is supplied.
    static_preparation_manifest: optional parsed imputation preparation
        manifest (as written by
        ``static_preparation.write_imputation_artifacts``, i.e. the
        on-disk ``imputation_manifest.json``); when supplied, its declared
        ``artifact_sha256["imputed_static_attributes.parquet"]`` must equal
        ``prepared_static_attributes_provenance["sha256"]``.
    gap_product_scope: optional explicit gap-inventory product scope used to
        select ``gap_timestamps`` upstream; when ``policy`` is also given it
        must equal :func:`resolve_gap_product_scope`'s result for that
        policy. Recorded in the manifest either way.
    package_netcdf_schema: which registered on-disk NetCDF package schema
        (see :mod:`src.baseline.package_netcdf`) every basin file in this
        build uses -- resolved once by the caller and applied identically
        to every basin. Defaults to the frozen legacy
        ``stage1_compact_scientific_package_v001``/``time`` schema for
        backward compatibility with existing direct callers; production
        CLI use must pass this explicitly (see
        ``scripts/build_stage1_baseline_nh_package.py --package-schema``)
        so a future scientific package can never silently fall back to the
        legacy schema.
    overwrite: refuse an existing destination unless True; even when True,
        the previous complete destination is preserved until the freshly
        built replacement has fully passed validation.
    dry_run: run every validation and build step (including a full
        into-temp-directory build) but never promote; temp directories are
        always cleaned up.
    """
    package_root = Path(output_package_root)
    evidence_root_path = Path(evidence_root) if evidence_root is not None else None

    if package_root.exists() and not overwrite:
        raise PackageBuilderError(f"output_package_root already exists and overwrite=False: {package_root}")
    if write_qc_csv and evidence_root_path is not None and evidence_root_path.exists() and not overwrite:
        raise PackageBuilderError(f"evidence_root already exists and overwrite=False: {evidence_root_path}")
    if write_qc_csv and evidence_root_path is None:
        raise PackageBuilderError("write_qc_csv=True requires an evidence_root")

    validated_basin_ids = _validate_basin_id_list(basin_ids)
    ordered_static_attributes = _validate_static_attributes(
        static_attributes, validated_basin_ids, static_model_input_columns, forbidden_static_columns
    )
    sorted_gap_timestamps = _validate_and_sort_gap_timestamps(gap_timestamps, expected_index)

    resolved_gap_product_scope = None
    if policy is not None:
        resolved_gap_product_scope = resolve_gap_product_scope(policy)
        if gap_product_scope is not None and tuple(gap_product_scope) != resolved_gap_product_scope:
            raise PackageBuilderError(
                f"gap_product_scope {tuple(gap_product_scope)!r} does not match the "
                f"policy-derived scope {resolved_gap_product_scope!r}; no extra (or missing) "
                "gap product may be folded in without the validated policy requiring it"
            )
        _validate_production_static_contract(
            policy, static_model_input_columns, static_column_manifest, static_attributes_provenance
        )
        _validate_prepared_static_artifact(prepared_static_attributes_provenance, static_preparation_manifest)
    elif gap_product_scope is not None:
        resolved_gap_product_scope = tuple(gap_product_scope)

    package_root.parent.mkdir(parents=True, exist_ok=True)
    tmp_package_dir = Path(
        tempfile.mkdtemp(prefix=f".{package_root.name}.building.", dir=str(package_root.parent))
    )
    tmp_evidence_dir = None
    if write_qc_csv:
        evidence_root_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_evidence_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{evidence_root_path.name}.building.", dir=str(evidence_root_path.parent)
            )
        )

    try:
        per_basin_files = []
        qc_csv_entries = {}
        for basin_id in validated_basin_ids:
            try:
                source = load_basin_source(basin_id)
                if not isinstance(source, BasinSourceTables):
                    raise PackageBuilderError(
                        f"load_basin_source({basin_id!r}) must return a BasinSourceTables, "
                        f"got {type(source)!r}"
                    )
                table = assemble_basin_package_table(
                    source.forcing,
                    source.qobs,
                    source.area_km2,
                    policy=policy,
                    expected_index=expected_index,
                )
                dataset = build_basin_dataset(table, basin_id, schema=package_netcdf_schema)
                nc_relative = Path("time_series") / f"{basin_id}.nc"
                nc_path = tmp_package_dir / nc_relative
                write_basin_dataset_netcdf(
                    dataset, nc_path, overwrite=False, create_parent=True, schema=package_netcdf_schema
                )
                validate_basin_netcdf_file(nc_path, table, basin_id, schema=package_netcdf_schema)
                per_basin_files.append((basin_id, nc_relative, nc_path))

                if write_qc_csv:
                    qc_csv_entries[basin_id] = _write_and_verify_qc_csv(table, basin_id, tmp_evidence_dir)
            except Exception as exc:
                raise PackageBuilderError(f"basin {basin_id!r} failed package build: {exc}") from exc

        attrs_path = _write_attributes_csv(tmp_package_dir, ordered_static_attributes)
        basins_path = _write_basin_ids_txt(tmp_package_dir, validated_basin_ids)
        gap_path = _write_gap_timestamps(tmp_package_dir, sorted_gap_timestamps)

        checksums = _collect_checksums(tmp_package_dir, per_basin_files, attrs_path, basins_path, gap_path)
        manifest = _build_manifest(
            basin_ids=validated_basin_ids,
            expected_index=expected_index,
            static_model_input_columns=list(static_model_input_columns),
            per_basin_files=per_basin_files,
            checksums=checksums,
            n_gap_timestamps=len(sorted_gap_timestamps),
            policy=policy,
            policy_provenance=policy_provenance,
            static_attributes_provenance=static_attributes_provenance,
            prepared_static_attributes_provenance=prepared_static_attributes_provenance,
            basin_selection_provenance=basin_selection_provenance,
            gap_inventory_provenance=gap_inventory_provenance,
            gap_product_scope=resolved_gap_product_scope,
            write_qc_csv=write_qc_csv,
            package_netcdf_schema=package_netcdf_schema,
        )
        _write_manifests(tmp_package_dir, manifest, checksums)
        _write_run_provenance(tmp_package_dir, manifest, dry_run=dry_run)

        if write_qc_csv:
            _write_evidence_manifests(tmp_evidence_dir, qc_csv_entries, package_schema=SCHEMA_NAME)

        # Package self-validation before promotion: membership, file
        # presence, and checksum agreement (independent of the future
        # Gate 4 whole-package auditor).
        written_nc_names = {p.name for p in (tmp_package_dir / "time_series").iterdir()}
        expected_nc_names = {f"{b}.nc" for b in validated_basin_ids}
        if written_nc_names != expected_nc_names:
            raise PackageBuilderError(
                f"post-build NetCDF membership mismatch: expected {expected_nc_names}, "
                f"got {written_nc_names}"
            )
        for rel_path, entry in checksums.items():
            abs_path = tmp_package_dir / rel_path
            if not abs_path.is_file():
                raise PackageBuilderError(f"manifest-listed file missing before promotion: {rel_path}")
            if sha256_of(abs_path) != entry["sha256"]:
                raise PackageBuilderError(f"checksum mismatch before promotion: {rel_path}")
        if write_qc_csv:
            csv_files = sorted((tmp_evidence_dir / "csv_inspection").glob("*.csv"))
            if len(csv_files) != len(validated_basin_ids):
                raise PackageBuilderError(
                    f"QC CSV count mismatch before promotion: expected "
                    f"{len(validated_basin_ids)}, got {len(csv_files)}"
                )

        if dry_run:
            return PackageBuildResult(
                package_root=package_root,
                evidence_root=evidence_root_path if write_qc_csv else None,
                basin_ids=validated_basin_ids,
                manifest=manifest,
                dry_run=True,
            )

        promotions = [(tmp_package_dir, package_root)]
        if write_qc_csv:
            promotions.append((tmp_evidence_dir, evidence_root_path))
        _atomic_promote_all(promotions, overwrite=overwrite)
        tmp_package_dir = None
        tmp_evidence_dir = None

        return PackageBuildResult(
            package_root=package_root,
            evidence_root=evidence_root_path if write_qc_csv else None,
            basin_ids=validated_basin_ids,
            manifest=manifest,
            dry_run=False,
        )
    finally:
        if tmp_package_dir is not None and tmp_package_dir.exists():
            shutil.rmtree(tmp_package_dir, ignore_errors=True)
        if tmp_evidence_dir is not None and tmp_evidence_dir.exists():
            shutil.rmtree(tmp_evidence_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI-facing strict readers (basin-ID list / area table) -- never silently
# repair a malformed input; fail loudly instead.
# ---------------------------------------------------------------------------


def read_basin_ids_file(path) -> list:
    """Read one basin ID per nonblank line, preserving each ID exactly as
    written.

    Truly empty lines are skipped. Any nonblank line carrying
    leading/trailing whitespace is rejected outright rather than silently
    trimmed -- a whitespace-padded ID is a data-quality signal, not
    something this reader is allowed to repair.
    """
    p = Path(path)
    if not p.is_file():
        raise PackageBuilderError(f"basin ids file not found: {p}")
    ids = []
    for line_number, raw_line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if raw_line == "":
            continue
        if raw_line != raw_line.strip():
            raise PackageBuilderError(
                f"{p}:{line_number}: basin id line has leading/trailing whitespace and will "
                f"not be silently trimmed: {raw_line!r}"
            )
        ids.append(raw_line)
    return ids


_AREA_CSV_REQUIRED_COLUMNS = ("gauge_id", "DRAIN_SQKM")


def read_area_csv(path, basin_ids: Sequence[str] | None = None) -> dict:
    """Read a ``gauge_id,DRAIN_SQKM`` area table with no silent repair.

    Requires both required columns; ``gauge_id`` is read as a string and
    validated via :func:`src.baseline.staid.normalize_staid` (already-canonical
    form only); duplicate ``gauge_id`` rows fail loudly rather than the last
    row silently winning; every area must be finite and strictly positive.
    When ``basin_ids`` is given, also requires an area for every one of them.
    """
    p = Path(path)
    if not p.is_file():
        raise PackageBuilderError(f"area CSV not found: {p}")
    df = pd.read_csv(p, dtype={"gauge_id": str})
    missing_columns = [c for c in _AREA_CSV_REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise PackageBuilderError(f"{p}: missing required column(s) {missing_columns}; have {list(df.columns)}")

    raw_ids = list(df["gauge_id"])
    if len(raw_ids) != len(set(raw_ids)):
        dupes = sorted({v for v in raw_ids if raw_ids.count(v) > 1})
        raise PackageBuilderError(f"{p}: duplicate gauge_id value(s): {dupes}")

    area_by_basin: dict = {}
    for gauge_id, area in zip(df["gauge_id"], df["DRAIN_SQKM"]):
        if not isinstance(gauge_id, str):
            raise PackageBuilderError(f"{p}: gauge_id must be a string, got {gauge_id!r}")
        try:
            normalized = normalize_staid(gauge_id)
        except (TypeError, ValueError) as exc:
            raise PackageBuilderError(f"{p}: invalid gauge_id {gauge_id!r}: {exc}") from exc
        if normalized != gauge_id:
            raise PackageBuilderError(
                f"{p}: gauge_id {gauge_id!r} is not already in canonical station-ID form "
                f"(would normalize to {normalized!r})"
            )
        area_value = float(area)
        if not np.isfinite(area_value) or area_value <= 0:
            raise PackageBuilderError(
                f"{p}: DRAIN_SQKM for {gauge_id!r} must be finite and strictly positive, got {area!r}"
            )
        area_by_basin[gauge_id] = area_value

    if basin_ids is not None:
        missing_areas = sorted(set(basin_ids) - set(area_by_basin))
        if missing_areas:
            raise PackageBuilderError(f"{p}: missing DRAIN_SQKM area for selected basin(s): {missing_areas}")

    return area_by_basin


# ---------------------------------------------------------------------------
# Default local per-basin source loader (established parquet/NetCDF layout;
# no gap-fill, no reindexing -- Gate 1 owns all structural validation)
# ---------------------------------------------------------------------------


def _validate_dynamic_inputs(dynamic_inputs: Sequence[str]) -> tuple:
    if not isinstance(dynamic_inputs, (list, tuple)):
        raise PackageBuilderError(f"dynamic_inputs must be a list/tuple, got {type(dynamic_inputs)!r}")
    if len(dynamic_inputs) == 0:
        raise PackageBuilderError("dynamic_inputs must not be empty")
    if len(dynamic_inputs) != len(set(dynamic_inputs)):
        dupes = sorted({v for v in dynamic_inputs if dynamic_inputs.count(v) > 1})
        raise PackageBuilderError(f"dynamic_inputs contains duplicate name(s): {dupes}")
    return tuple(dynamic_inputs)


def default_local_basin_source_loader(
    forcing_root, qobs_root, area_by_basin: Mapping[str, float], dynamic_inputs: Sequence[str]
) -> Callable[[str], BasinSourceTables]:
    """One thin, concrete ``load_basin_source`` for the established local
    layout: ``<forcing_root>/time_series/<basin_id>.parquet`` (dynamic
    inputs) and ``<qobs_root>/time_series/<basin_id>.nc`` (a ``qobs_m3s``
    variable on a ``time``/``date`` coordinate), matching the frozen
    historical ``scripts/build_stage1_nh_package.py`` source convention.

    ``dynamic_inputs`` is the exact ordered, unique set of forcing columns
    this loader selects from each basin's parquet file -- source products
    are free to carry additional fields beyond the approved model-input
    contract (e.g. extra RTMA variables), and this loader's job is to select
    exactly the approved columns, in order, so that Gate 1's own strict
    forcing-column validation continues to see only the exact contract it
    already enforces. Never fills, interpolates, reindexes, or converts
    values -- any structural defect surfaces through Gate 1's own strict
    validation.
    """
    forcing_root = Path(forcing_root)
    qobs_root = Path(qobs_root)
    validated_dynamic_inputs = _validate_dynamic_inputs(dynamic_inputs)

    def _load(basin_id: str) -> BasinSourceTables:
        forcing_path = forcing_root / "time_series" / f"{basin_id}.parquet"
        if not forcing_path.is_file():
            raise PackageBuilderError(f"forcing file not found for basin {basin_id!r}: {forcing_path}")
        forcing = pd.read_parquet(forcing_path)
        missing_inputs = [c for c in validated_dynamic_inputs if c not in forcing.columns]
        if missing_inputs:
            raise PackageBuilderError(
                f"forcing file for basin {basin_id!r} is missing required dynamic input "
                f"column(s) {missing_inputs}: {forcing_path}"
            )
        forcing = forcing[list(validated_dynamic_inputs)]
        if forcing.index.tz is not None:
            forcing.index = forcing.index.tz_convert("UTC").tz_localize(None)

        qobs_path = qobs_root / "time_series" / f"{basin_id}.nc"
        if not qobs_path.is_file():
            raise PackageBuilderError(f"qobs file not found for basin {basin_id!r}: {qobs_path}")
        with xr.open_dataset(qobs_path) as ds:
            time_coord = "time" if "time" in ds.coords else "date"
            idx = pd.DatetimeIndex(ds.coords[time_coord].values)
            if idx.tz is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            qobs = pd.Series(np.asarray(ds["qobs_m3s"].values, dtype=np.float64), index=idx, name="qobs_m3s")

        if basin_id not in area_by_basin:
            raise PackageBuilderError(f"no DRAIN_SQKM area found for basin {basin_id!r}")
        return BasinSourceTables(forcing=forcing, qobs=qobs, area_km2=area_by_basin[basin_id])

    return _load

"""Stage 1 pure per-basin NetCDF serialization helper
(Compact Scientific Package builder, Gate 2).

Converts one already-assembled basin table (the output contract of
:func:`src.baseline.package_assembly.assemble_basin_package_table`) into an
in-memory ``xarray.Dataset`` and writes that dataset atomically to one
NetCDF file. This is the serialization layer immediately above the Gate 1
assembly core: it never reads forcing/qobs sources, never loops over
basins, never touches static attributes, basin lists, manifests, config
generation, validity masks, NeuralHydrology dataset registration, or any
remote path -- all of that belongs to a later builder/CLI layer
(docs/stage1_baseline_package_implementation_plan.md sec 11/15). The
production builder is responsible for eventually calling
``assemble_basin_package_table(..., expected_index=<the 45,720-hour Stage 1
grid>)`` upstream of this module; this module does not enforce that period
itself and accepts whatever validated hourly table it is given.

Two independently testable capabilities:
- :func:`build_basin_dataset` -- pure, no filesystem access.
- :func:`write_basin_dataset_netcdf` -- atomic disk write of an
  already-built dataset (temp file in the destination directory, reopened
  and validated before an atomic replace; cleaned up on any failure).

:func:`validate_basin_netcdf_file` is a small local read-back integrity
check (reads the file back from disk); it does not replace the future
independent whole-package auditor.

The expected 13-column input contract (exact order) is derived from the
committed Gate 1 constants rather than re-declared by hand:
:data:`src.baseline.package_assembly.DYNAMIC_INPUTS` (8),
:data:`src.baseline.package_assembly.RAW_TARGET_VARIABLE` (1), and the four
default lead-target names from :mod:`src.baseline.lead_targets`.

Dtype/encoding contract (matches the binding Stage 1 policy pins
``target.package_dtype: float32`` / ``audit.package_float32_rtol: 1e-5``,
see src/baseline/policy.py):
- continuous scientific variables (forcing + raw qobs + lead targets):
  float32 on disk, NaN preserved via an explicit float32 NaN fill value.
- gap flags: int8 on disk, no fill value at all (they are validated
  complete/binary before writing, so no fill value could ever be
  ambiguous with a real 0/1).
- time: int64 whole-hour offsets from the table's own first timestamp
  (exact by construction -- the index is already validated strictly
  hourly), never a floating-point offset.

NetCDF engine is pinned explicitly to ``"netcdf4"`` (see
:data:`NETCDF_ENGINE`) -- the only backend installed/exercised in this
environment (h5netcdf/scipy are not installed) -- so behavior never
depends on whichever default xarray happens to pick locally.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .lead_targets import DEFAULT_LEADS_HOURS, DEFAULT_VARIABLE_NAME_TEMPLATE, variable_name_for_lead
from .package_assembly import DYNAMIC_INPUTS, RAW_TARGET_VARIABLE
from .staid import normalize_staid

__all__ = [
    "PackageNetCDFError",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "NETCDF_ENGINE",
    "EXPECTED_VARIABLES",
    "build_basin_dataset",
    "write_basin_dataset_netcdf",
    "validate_basin_netcdf_file",
]

_HOUR = pd.Timedelta(hours=1)

#: Package/schema identity recorded in every file's dataset-level attrs.
SCHEMA_NAME = "stage1_compact_scientific_package_v001"
SCHEMA_VERSION = 1

#: The only NetCDF backend installed/tested in this environment; pinned
#: explicitly for both writing and read-back validation.
NETCDF_ENGINE = "netcdf4"

#: Four default lead-target variable names, in declared lead order.
_LEAD_VARIABLES: tuple[str, ...] = tuple(
    variable_name_for_lead(lead, DEFAULT_VARIABLE_NAME_TEMPLATE) for lead in DEFAULT_LEADS_HOURS
)
_LEAD_HOURS_BY_VARIABLE: dict[str, int] = dict(zip(_LEAD_VARIABLES, DEFAULT_LEADS_HOURS))

#: Exact assemble_basin_package_table() output contract (13 columns,
#: binding order): dynamic inputs, then the raw target, then the lead
#: targets -- derived from the committed Gate 1 constants, never
#: re-declared independently.
EXPECTED_VARIABLES: tuple[str, ...] = (
    tuple(DYNAMIC_INPUTS) + (RAW_TARGET_VARIABLE,) + _LEAD_VARIABLES
)

_GAP_FLAG_VARIABLES = frozenset(name for name in DYNAMIC_INPUTS if name.endswith("_gap"))

_UNITS: dict[str, str] = {
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
for _name in _LEAD_VARIABLES:
    _UNITS[_name] = "mm h-1"
del _name


class PackageNetCDFError(ValueError):
    """Raised for an invalid assembled table, gauge ID, or NetCDF I/O contract."""


# ---------------------------------------------------------------------------
# gauge_id validation (reuses staid.py; never re-implements its rules)
# ---------------------------------------------------------------------------


def _validate_gauge_id(value) -> str:
    """Validate ``value`` via the established staid.py contract and require
    it to already be in exactly the canonical form that contract produces.

    This module never normalizes a gauge ID for the caller -- it preserves
    whatever string is supplied exactly, so any value that would be *changed*
    by normalization (surrounding whitespace, missing zero-padding) is
    rejected rather than silently repaired.
    """
    try:
        normalized = normalize_staid(value)
    except (TypeError, ValueError) as exc:
        raise PackageNetCDFError(f"invalid gauge_id: {exc}") from exc
    if normalized != value:
        raise PackageNetCDFError(
            "gauge_id must already be in canonical station-ID form (no "
            "surrounding whitespace, no missing zero-padding) -- this "
            "serializer preserves the supplied string exactly rather than "
            f"normalizing it; got {value!r}, canonical form would be {normalized!r}"
        )
    return value


# ---------------------------------------------------------------------------
# Table structural validation (self-contained, mirrors the style of
# src/baseline/package_assembly.py)
# ---------------------------------------------------------------------------


def _validate_hourly_index(index, label: str) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise PackageNetCDFError(
            f"{label} index must be a pandas.DatetimeIndex, got {type(index)!r}"
        )
    if len(index) == 0:
        raise PackageNetCDFError(f"{label} index must not be empty")
    if index.tz is not None:
        raise PackageNetCDFError(
            f"{label} index must be timezone-naive; got tz={index.tz!r}. This "
            "serializer never silently converts timezone-aware timestamps."
        )
    if index.has_duplicates:
        raise PackageNetCDFError(f"{label} index must not contain duplicate timestamps")
    if not index.is_monotonic_increasing:
        raise PackageNetCDFError(
            f"{label} index must be strictly increasing (ascending); found a "
            "non-increasing or descending step"
        )
    if len(index) >= 2:
        deltas = index[1:] - index[:-1]
        irregular = np.asarray(deltas != _HOUR)
        if irregular.any():
            first_bad = int(np.flatnonzero(irregular)[0])
            raise PackageNetCDFError(
                f"{label} index must be strictly hourly with no gaps; found "
                f"{int(irregular.sum())} irregular step(s), first at position "
                f"{first_bad} ({index[first_bad]} -> {index[first_bad + 1]}, "
                f"delta {deltas[first_bad]})"
            )


def _validate_gap_flag_series(series: pd.Series, name: str) -> None:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return
    if not pd.api.types.is_numeric_dtype(dtype):
        raise PackageNetCDFError(
            f"{name} is a gap-flag column and must be boolean or numeric "
            f"0/1, got dtype {dtype}"
        )
    values = series.to_numpy(dtype=np.float64)
    n_nonfinite = int(np.count_nonzero(~np.isfinite(values)))
    if n_nonfinite:
        raise PackageNetCDFError(
            f"{name} is a gap-flag column and must contain only finite 0/1 "
            f"values; found {n_nonfinite} NaN/inf value(s)"
        )
    invalid = (values != 0.0) & (values != 1.0)
    n_invalid = int(np.count_nonzero(invalid))
    if n_invalid:
        raise PackageNetCDFError(
            f"{name} is a gap-flag column and must contain only 0/1 values; "
            f"found {n_invalid} value(s) outside {{0, 1}}"
        )


def _validate_scientific_series(series: pd.Series, name: str) -> None:
    dtype = series.dtype
    if not (pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype)):
        raise PackageNetCDFError(
            f"{name} must have a numeric dtype suitable for NetCDF encoding "
            f"(not object/string), got {dtype}"
        )
    values = series.to_numpy(dtype=np.float64)
    n_inf = int(np.count_nonzero(np.isinf(values)))
    if n_inf:
        raise PackageNetCDFError(
            f"{name} contains {n_inf} infinite value(s); NaN is permitted "
            "(preserved as missing) but infinity is not"
        )


def _validate_table(table, dynamic_inputs=EXPECTED_VARIABLES) -> None:
    if not isinstance(table, pd.DataFrame):
        raise PackageNetCDFError(f"table must be a pandas.DataFrame, got {type(table)!r}")
    _validate_hourly_index(table.index, "table")

    actual_columns = list(table.columns)
    expected_columns = list(EXPECTED_VARIABLES)
    if actual_columns != expected_columns:
        actual_set = set(actual_columns)
        expected_set = set(expected_columns)
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        if missing:
            raise PackageNetCDFError(f"table is missing required variable(s): {missing}")
        if extra:
            raise PackageNetCDFError(f"table contains unapproved variable(s): {extra}")
        raise PackageNetCDFError(
            "table columns do not match the required package order exactly; "
            f"expected {expected_columns}, got {actual_columns}"
        )

    for name in EXPECTED_VARIABLES:
        if name in _GAP_FLAG_VARIABLES:
            _validate_gap_flag_series(table[name], name)
        else:
            _validate_scientific_series(table[name], name)


# ---------------------------------------------------------------------------
# Public API: pure DataFrame -> xarray.Dataset construction
# ---------------------------------------------------------------------------


def build_basin_dataset(table: pd.DataFrame, gauge_id: str) -> xr.Dataset:
    """Convert one validated assembled basin table into an in-memory
    ``xarray.Dataset``.

    table: the exact 13-column output of
        :func:`src.baseline.package_assembly.assemble_basin_package_table`
        (see :data:`EXPECTED_VARIABLES` for the binding name/order
        contract), indexed by a strictly increasing, duplicate-free,
        gap-free, timezone-naive hourly ``pandas.DatetimeIndex``. Not
        mutated by this function.
    gauge_id: the basin's station ID string, validated via
        :func:`src.baseline.staid.normalize_staid` and required to already
        be in that function's canonical form -- it is never itself
        normalized, re-padded, or cast to integer, and is preserved exactly
        in the returned dataset's attributes.

    Performs no filesystem access. Does not fill, interpolate, aggregate,
    shift, or convert any scientific value -- this is a serialization layer,
    not an assembly layer.
    """
    gauge_id = _validate_gauge_id(gauge_id)
    _validate_table(table)

    index = table.index
    epoch = index[0]
    time_units = f"hours since {epoch.isoformat()}"

    data_vars: dict[str, tuple] = {}
    for name in EXPECTED_VARIABLES:
        values = table[name].to_numpy()
        if name in _GAP_FLAG_VARIABLES:
            arr = values.astype(np.int8)
        else:
            arr = values.astype(np.float32)
        data_vars[name] = (("time",), arr)

    ds = xr.Dataset(data_vars, coords={"time": index.to_numpy(copy=True)})

    ds.attrs = {
        "gauge_id": gauge_id,
        "package_schema_name": SCHEMA_NAME,
        "package_schema_version": SCHEMA_VERSION,
        "description": (
            "Stage 1 Compact Scientific Package per-basin time-series file "
            "(dynamic forcing inputs, raw discharge, lead targets)."
        ),
        "created_by": "src.baseline.package_netcdf.build_basin_dataset",
    }
    ds["time"].attrs = {
        "description": "UTC hourly timestamps (timezone-naive; no offset stored)",
    }
    ds["time"].encoding = {
        "units": time_units,
        "calendar": "proleptic_gregorian",
        "dtype": "int64",
    }

    for name in EXPECTED_VARIABLES:
        attrs: dict[str, object] = {"units": _UNITS[name]}
        if name in _GAP_FLAG_VARIABLES:
            attrs["long_name"] = f"{name} binary archive-gap flag"
            attrs["flag_values"] = "0, 1"
            attrs["flag_meanings"] = "no_gap gap"
            ds[name].encoding = {"dtype": "int8", "_FillValue": None}
        elif name == RAW_TARGET_VARIABLE:
            attrs["long_name"] = "raw observed discharge"
            attrs["role"] = "audit_provenance_not_training_target"
            ds[name].encoding = {"dtype": "float32", "_FillValue": np.float32(np.nan)}
        elif name in _LEAD_HOURS_BY_VARIABLE:
            lead_hours = _LEAD_HOURS_BY_VARIABLE[name]
            attrs["long_name"] = f"future runoff depth target, lead {lead_hours}h"
            attrs["lead_hours"] = lead_hours
            attrs["role"] = "training_target"
            ds[name].encoding = {"dtype": "float32", "_FillValue": np.float32(np.nan)}
        else:
            attrs["long_name"] = name
            ds[name].encoding = {"dtype": "float32", "_FillValue": np.float32(np.nan)}
        ds[name].attrs = attrs

    return ds


# ---------------------------------------------------------------------------
# Dataset-level contract validation (independent of any source DataFrame).
#
# This is the integrity gate: it is what stands between a malformed or
# mutated in-memory xarray.Dataset and an atomically-promoted file on disk.
# It never compares against a source table -- exact source-value and
# NaN-mask comparison remains the job of validate_basin_netcdf_file().
# ---------------------------------------------------------------------------


def _validate_basin_dataset_contract(dataset: xr.Dataset) -> None:
    if not isinstance(dataset, xr.Dataset):
        raise PackageNetCDFError(f"dataset must be an xarray.Dataset, got {type(dataset)!r}")

    actual_schema_name = dataset.attrs.get("package_schema_name")
    if actual_schema_name != SCHEMA_NAME:
        raise PackageNetCDFError(
            f"dataset package_schema_name mismatch: got {actual_schema_name!r}, "
            f"expected {SCHEMA_NAME!r}"
        )
    actual_schema_version = dataset.attrs.get("package_schema_version")
    if actual_schema_version != SCHEMA_VERSION:
        raise PackageNetCDFError(
            f"dataset package_schema_version mismatch: got {actual_schema_version!r}, "
            f"expected {SCHEMA_VERSION!r}"
        )

    gauge_id = dataset.attrs.get("gauge_id")
    if gauge_id is None:
        raise PackageNetCDFError("dataset is missing required attrs['gauge_id']")
    _validate_gauge_id(gauge_id)

    if "time" not in dataset.dims:
        raise PackageNetCDFError(
            f"dataset must have a 'time' dimension, got dims {tuple(dataset.dims)}"
        )
    if len(dataset.dims) != 1:
        raise PackageNetCDFError(
            f"dataset must have exactly one dimension named 'time', got dims "
            f"{tuple(dataset.dims)}"
        )
    if "time" not in dataset.coords:
        raise PackageNetCDFError("dataset must have a 'time' coordinate")
    if dataset.sizes["time"] == 0:
        raise PackageNetCDFError("dataset 'time' coordinate must not be empty")

    try:
        time_index = pd.DatetimeIndex(dataset["time"].values)
    except (TypeError, ValueError) as exc:
        raise PackageNetCDFError(
            f"dataset time coordinate could not be decoded as datetimes: {exc}"
        ) from exc
    _validate_hourly_index(time_index, "dataset time coordinate")

    actual_vars = set(dataset.data_vars)
    expected_vars = set(EXPECTED_VARIABLES)
    if actual_vars != expected_vars:
        missing = sorted(expected_vars - actual_vars)
        extra = sorted(actual_vars - expected_vars)
        raise PackageNetCDFError(
            f"dataset variable set mismatch: missing {missing}, unapproved extra {extra}"
        )

    n_time = dataset.sizes["time"]
    for name in EXPECTED_VARIABLES:
        var = dataset[name]
        if tuple(var.dims) != ("time",):
            raise PackageNetCDFError(
                f"{name} has unexpected dimensions {var.dims}, expected ('time',)"
            )
        if var.sizes["time"] != n_time:
            raise PackageNetCDFError(
                f"{name} length {var.sizes['time']} does not match time length {n_time}"
            )

        expected_units = _UNITS[name]
        actual_units = var.attrs.get("units")
        if actual_units != expected_units:
            raise PackageNetCDFError(
                f"{name} units mismatch: got {actual_units!r}, expected {expected_units!r}"
            )

        if name in _GAP_FLAG_VARIABLES:
            values_f = np.asarray(var.values, dtype=np.float64)
            if np.count_nonzero(~np.isfinite(values_f)):
                raise PackageNetCDFError(
                    f"{name} is a gap-flag variable and must be complete (no NaN/inf values)"
                )
            invalid = (values_f != 0.0) & (values_f != 1.0)
            if np.count_nonzero(invalid):
                raise PackageNetCDFError(
                    f"{name} is a gap-flag variable and must contain only 0/1 values"
                )
            if (
                var.attrs.get("flag_values") != "0, 1"
                or var.attrs.get("flag_meanings") != "no_gap gap"
            ):
                raise PackageNetCDFError(f"{name} is missing required gap-flag metadata")
            continue

        if name == RAW_TARGET_VARIABLE:
            if var.attrs.get("role") != "audit_provenance_not_training_target":
                raise PackageNetCDFError(
                    f"{name} must have role='audit_provenance_not_training_target', "
                    f"got {var.attrs.get('role')!r}"
                )
        elif name in _LEAD_HOURS_BY_VARIABLE:
            expected_lead = _LEAD_HOURS_BY_VARIABLE[name]
            if var.attrs.get("role") != "training_target":
                raise PackageNetCDFError(
                    f"{name} must have role='training_target', got {var.attrs.get('role')!r}"
                )
            if var.attrs.get("lead_hours") != expected_lead:
                raise PackageNetCDFError(
                    f"{name} lead_hours mismatch: got {var.attrs.get('lead_hours')!r}, "
                    f"expected {expected_lead!r}"
                )

        values_f = np.asarray(var.values, dtype=np.float64)
        n_inf = int(np.count_nonzero(np.isinf(values_f)))
        if n_inf:
            raise PackageNetCDFError(
                f"{name} contains {n_inf} infinite value(s); NaN is permitted "
                "(preserved as missing) but infinity is not"
            )


# ---------------------------------------------------------------------------
# Public API: atomic disk write of an already-built dataset
# ---------------------------------------------------------------------------


def write_basin_dataset_netcdf(
    dataset: xr.Dataset,
    path,
    *,
    overwrite: bool = False,
    create_parent: bool = False,
) -> Path:
    """Atomically write ``dataset`` to one NetCDF file at ``path``.

    Validates ``dataset`` against the Stage 1 dataset-level schema contract
    (:func:`_validate_basin_dataset_contract`) before writing anything, writes
    it to a temporary file in the same destination directory, reopens that
    temporary file with :data:`NETCDF_ENGINE` and re-validates it against the
    same contract, and only then atomically replaces ``path``. This guards
    against a malformed or mutated dataset being promoted to disk -- a file
    that merely reopens successfully is not enough; it must also still
    satisfy the schema. The temporary file is removed on any failure
    (including a post-write contract failure), so a failed write never
    leaves a partial or corrupt file at the destination, and an existing
    destination is left unchanged when ``overwrite=True``. Refuses to
    overwrite an existing destination unless ``overwrite=True``. The parent
    directory is created only if ``create_parent=True`` is explicitly passed.
    """
    _validate_basin_dataset_contract(dataset)

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise PackageNetCDFError(
            f"destination already exists and overwrite=False: {destination}"
        )

    parent = destination.parent
    if not parent.is_dir():
        if create_parent:
            parent.mkdir(parents=True, exist_ok=True)
        else:
            raise PackageNetCDFError(
                f"destination parent directory does not exist: {parent} "
                "(pass create_parent=True to allow creating it)"
            )

    fd, tmp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        dataset.to_netcdf(str(tmp_path), engine=NETCDF_ENGINE)
        with xr.open_dataset(tmp_path, engine=NETCDF_ENGINE) as reopened:
            reopened.load()
            _validate_basin_dataset_contract(reopened)
        os.replace(str(tmp_path), str(destination))
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return destination


# ---------------------------------------------------------------------------
# Public API: local read-back serialization integrity check
# ---------------------------------------------------------------------------


def validate_basin_netcdf_file(
    path,
    table: pd.DataFrame,
    gauge_id: str,
    *,
    rtol: float = 1.0e-5,
) -> None:
    """Read one basin NetCDF file back from disk and verify it matches
    ``table``/``gauge_id`` exactly (local serialization integrity check --
    it does not replace the future independent whole-package auditor).

    Checks: exact gauge ID; exact time coordinate and row count; exact
    expected variable names; expected dimensions; unit metadata; gap flags
    remain binary and complete; NaN locations match ``table`` for every
    scientific variable; all finite numeric values match within ``rtol``.
    """
    gauge_id = _validate_gauge_id(gauge_id)
    _validate_table(table)

    with xr.open_dataset(Path(path), engine=NETCDF_ENGINE) as ds:
        ds.load()
        _validate_basin_dataset_contract(ds)

        actual_gauge_id = ds.attrs.get("gauge_id")
        if actual_gauge_id != gauge_id:
            raise PackageNetCDFError(
                f"gauge_id mismatch: file has {actual_gauge_id!r}, expected {gauge_id!r}"
            )

        if len(ds["time"]) != len(table.index):
            raise PackageNetCDFError(
                f"time row count mismatch: file has {len(ds['time'])}, "
                f"expected {len(table.index)}"
            )
        actual_time = pd.DatetimeIndex(ds["time"].values)
        if not actual_time.equals(table.index):
            raise PackageNetCDFError("time coordinate does not match table.index exactly")

        for name in EXPECTED_VARIABLES:
            var = ds[name]
            disk_values = var.values.astype(np.float64)
            source_values = table[name].to_numpy(dtype=np.float64)

            if name in _GAP_FLAG_VARIABLES:
                if not np.array_equal(disk_values, source_values):
                    raise PackageNetCDFError(f"{name} gap-flag values do not match source exactly")
                continue

            disk_nan = np.isnan(disk_values)
            source_nan = np.isnan(source_values)
            if not np.array_equal(disk_nan, source_nan):
                raise PackageNetCDFError(f"{name} NaN mask does not match source after round trip")
            finite = ~source_nan
            if finite.any() and not np.allclose(
                disk_values[finite], source_values[finite], rtol=rtol, atol=0.0
            ):
                raise PackageNetCDFError(
                    f"{name} finite values do not round-trip within rtol={rtol}"
                )

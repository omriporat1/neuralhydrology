"""Stage 1 pure local per-basin scientific time-series assembly core
(Compact Scientific Package builder, first code increment).

Combines already-loaded, in-memory per-basin inputs -- an hourly forcing
table, a raw discharge series/table, and a scalar basin area -- into one
canonical assembled table on a single validated hourly timeline. This is the
innermost layer of the future package builder: it never touches the
filesystem, never writes NetCDF, never applies validity-mask/gap-timestamp
filtering, and never prepares static attributes. A later builder layer is
responsible for I/O, static-attribute integration, mask artifacts, and the
CLI (docs/stage1_baseline_package_implementation_plan.md sec 11/15).

Reuses the existing approved primitives rather than reimplementing them:
- unit conversion (:func:`src.baseline.units.discharge_m3s_to_runoff_mm_per_h`)
  via :func:`src.baseline.lead_targets.build_lead_targets`;
- lead-target shifting (:mod:`src.baseline.lead_targets`) unchanged.

Canonical timeline contract (deliberately strict -- no silent reindexing):
- the forcing table's ``pandas.DatetimeIndex`` is the canonical timeline for
  the call. It must be strictly increasing, duplicate-free, and exactly
  hourly (no gaps) -- irregular input is rejected, never repaired.
- the qobs series/table's index must equal the forcing index exactly
  (same timestamps, same order). A mismatch is rejected outright; this
  layer never reindexes, resamples, or infers missing rows -- reconciling
  differing source schedules is the caller's job, upstream of this
  function (mirrors the reconciliation boundary documented in
  :mod:`src.baseline.lead_targets`).

Dynamic-input contract: exactly the 8 approved ``v001-core`` variables
(``config/stage1_scientific_baseline_v001.yaml::dynamic_inputs``, see
:data:`DYNAMIC_INPUTS`) -- missing or extra (e.g. historical Smoke-era
``rtma_sp_Pa``/``rtma_tcc_pct``/``rtma_vis_m``/``rtma_gust_ms``/
``rtma_ceil_m``) columns are rejected. Forcing NaNs and gap flags are
preserved exactly as supplied -- this layer never fills, interpolates, or
zeroes them (the historical Smoke 0/1 gap-fill policy does not apply here).

Target contract: raw ``qobs_m3s`` is preserved exactly (NaNs included) as an
audit/provenance column -- it is never the training target. Negative finite
qobs values are rejected (the source contract requires them already
cleaned upstream; this layer does not silently clean them). The four lead
targets (``qobs_mm_per_h_lead01/03/06/12`` by default) are built via
:func:`src.baseline.lead_targets.build_lead_targets` against the scalar
basin area.

Returns a single ``pandas.DataFrame`` on the canonical hourly index with an
explicit, deterministic column order: the dynamic inputs (declared order),
then the raw target, then the lead targets (declared lead order).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .lead_targets import (
    DEFAULT_LEADS_HOURS,
    DEFAULT_VARIABLE_NAME_TEMPLATE,
    build_lead_targets,
    variable_name_for_lead,
)
from .policy import Stage1BaselinePolicyError, validate_stage1_baseline_policy

__all__ = [
    "PackageAssemblyError",
    "DYNAMIC_INPUTS",
    "RAW_TARGET_VARIABLE",
    "assemble_basin_package_table",
]

_HOUR = pd.Timedelta(hours=1)

#: Approved v001-core dynamic inputs
#: (config/stage1_scientific_baseline_v001.yaml::dynamic_inputs). Order is
#: binding -- it defines the leading segment of the output column order.
DYNAMIC_INPUTS: tuple[str, ...] = (
    "mrms_qpe_1h_mm",
    "rtma_2t_K",
    "rtma_2d_K",
    "rtma_2sh_kgkg",
    "rtma_10u_ms",
    "rtma_10v_ms",
    "mrms_qpe_1h_mm_gap",
    "rtma_gap",
)

#: Raw discharge source variable (config/stage1_scientific_baseline_v001.yaml
#: ``target.source_variable``); retained as an audit/provenance column only.
RAW_TARGET_VARIABLE = "qobs_m3s"


class PackageAssemblyError(ValueError):
    """Raised for an invalid forcing/qobs timeline, area, or dynamic-input contract."""


# ---------------------------------------------------------------------------
# Policy-or-default parameter resolution
# ---------------------------------------------------------------------------


def _resolve_parameters(policy: dict | None):
    if policy is None:
        return DYNAMIC_INPUTS, DEFAULT_LEADS_HOURS, DEFAULT_VARIABLE_NAME_TEMPLATE, RAW_TARGET_VARIABLE
    try:
        validate_stage1_baseline_policy(policy)
    except Stage1BaselinePolicyError as exc:
        raise PackageAssemblyError(
            f"supplied policy failed Stage 1 baseline policy validation: {exc}"
        ) from exc
    dynamic_inputs = tuple(policy["dynamic_inputs"])
    leads_hours = tuple(policy["target"]["leads_hours"])
    variable_name_template = policy["target"]["variable_name_template"]
    source_variable = policy["target"]["source_variable"]
    return dynamic_inputs, leads_hours, variable_name_template, source_variable


# ---------------------------------------------------------------------------
# Canonical hourly index validation (self-contained, mirrors the style of
# src/baseline/lead_targets.py and src/baseline/validity_mask.py)
# ---------------------------------------------------------------------------


def _validate_canonical_hourly_index(index, label: str) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise PackageAssemblyError(
            f"{label} index must be a pandas.DatetimeIndex, got {type(index)!r}"
        )
    if len(index) == 0:
        raise PackageAssemblyError(f"{label} index must not be empty")
    if index.tz is not None:
        raise PackageAssemblyError(
            f"{label} index must be timezone-naive (the package's established "
            f"tz-naive UTC-semantics convention); got tz={index.tz!r}. This "
            f"layer never silently converts timezone-aware timestamps."
        )
    if index.has_duplicates:
        raise PackageAssemblyError(f"{label} index must not contain duplicate timestamps")
    if not index.is_monotonic_increasing:
        raise PackageAssemblyError(
            f"{label} index must be strictly increasing (ascending); found a "
            f"non-increasing or descending step"
        )
    if len(index) >= 2:
        deltas = index[1:] - index[:-1]
        irregular = np.asarray(deltas != _HOUR)
        if irregular.any():
            first_bad = int(np.flatnonzero(irregular)[0])
            raise PackageAssemblyError(
                f"{label} index must be strictly hourly with no gaps; found "
                f"{int(irregular.sum())} irregular step(s), first at position "
                f"{first_bad} ({index[first_bad]} -> {index[first_bad + 1]}, "
                f"delta {deltas[first_bad]})"
            )


def _validate_expected_index(forcing_index: pd.DatetimeIndex, expected_index: pd.DatetimeIndex) -> None:
    """Optionally enforce the exact production canonical timeline.

    ``expected_index`` is validated with the same structural rules as the
    forcing index (hourly, tz-naive, duplicate-free, monotonic) and then the
    forcing index must equal it exactly -- no reindexing or repair.
    """
    _validate_canonical_hourly_index(expected_index, "expected_index")
    if not forcing_index.equals(expected_index):
        raise PackageAssemblyError(
            "forcing index does not match the supplied expected_index exactly "
            f"(expected start={expected_index[0]}, end={expected_index[-1]}, "
            f"length={len(expected_index)}; got start={forcing_index[0]}, "
            f"end={forcing_index[-1]}, length={len(forcing_index)})"
        )


# ---------------------------------------------------------------------------
# Forcing table validation
# ---------------------------------------------------------------------------


def _validate_dynamic_column_dtype(series: pd.Series, name: str) -> None:
    dtype = series.dtype
    if not (pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype)):
        raise PackageAssemblyError(
            f"{name} must have a numeric or boolean dtype suitable for later "
            f"NetCDF encoding, got {dtype}"
        )


def _validate_gap_flag_column(series: pd.Series, name: str) -> None:
    """Strict binary gap-flag contract: boolean dtype, or numeric containing
    only finite 0/1 values (no NaN, no fractional, no out-of-range, no inf).
    """
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return
    if not pd.api.types.is_numeric_dtype(dtype):
        raise PackageAssemblyError(
            f"{name} is a gap-flag column and must be boolean or numeric "
            f"0/1, got dtype {dtype}"
        )
    values = series.to_numpy(dtype=np.float64)
    n_nonfinite = int(np.count_nonzero(~np.isfinite(values)))
    if n_nonfinite:
        raise PackageAssemblyError(
            f"{name} is a gap-flag column and must contain only finite 0/1 "
            f"values; found {n_nonfinite} NaN/inf value(s)"
        )
    invalid = (values != 0.0) & (values != 1.0)
    n_invalid = int(np.count_nonzero(invalid))
    if n_invalid:
        raise PackageAssemblyError(
            f"{name} is a gap-flag column and must contain only 0/1 values; "
            f"found {n_invalid} value(s) outside {{0, 1}}"
        )


def _is_gap_flag_name(name: str) -> bool:
    return name.endswith("_gap")


def _validate_forcing_table(forcing, dynamic_inputs) -> None:
    if not isinstance(forcing, pd.DataFrame):
        raise PackageAssemblyError(
            f"forcing must be a pandas.DataFrame, got {type(forcing)!r}"
        )
    _validate_canonical_hourly_index(forcing.index, "forcing")

    actual = list(forcing.columns)
    if len(actual) != len(set(actual)):
        raise PackageAssemblyError("forcing table must not contain duplicate column names")

    actual_set = set(actual)
    expected_set = set(dynamic_inputs)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing:
        raise PackageAssemblyError(
            f"forcing table is missing required dynamic input column(s): {missing}"
        )
    if extra:
        raise PackageAssemblyError(
            f"forcing table contains unapproved dynamic input column(s) -- only "
            f"the {len(dynamic_inputs)} approved v001-core variables are "
            f"allowed (no historical Smoke-era extras): {extra}"
        )
    for name in dynamic_inputs:
        if _is_gap_flag_name(name):
            _validate_gap_flag_column(forcing[name], name)
        else:
            _validate_dynamic_column_dtype(forcing[name], name)


# ---------------------------------------------------------------------------
# qobs extraction + validation
# ---------------------------------------------------------------------------


def _extract_qobs_series(qobs, source_variable: str) -> pd.Series:
    if isinstance(qobs, pd.DataFrame):
        if source_variable not in qobs.columns:
            raise PackageAssemblyError(
                f"qobs table is missing required column {source_variable!r}"
            )
        series = qobs[source_variable]
    elif isinstance(qobs, pd.Series):
        series = qobs
    else:
        raise PackageAssemblyError(
            f"qobs must be a pandas.Series or pandas.DataFrame, got {type(qobs)!r}"
        )
    return series.rename(source_variable)


def _validate_qobs_dtype(series: pd.Series, name: str) -> None:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype):
        raise PackageAssemblyError(
            f"{name} must have a numeric dtype (not boolean or non-numeric "
            f"such as object/string), got {dtype}"
        )


def _reject_negative_finite(series: pd.Series, name: str) -> None:
    values = series.to_numpy(dtype=np.float64)
    bad = np.isfinite(values) & (values < 0.0)
    n_bad = int(np.count_nonzero(bad))
    if n_bad:
        raise PackageAssemblyError(
            f"{name} contains {n_bad} negative finite value(s); the source "
            f"contract requires qobs already cleaned upstream -- this layer "
            f"does not silently clean them"
        )


# ---------------------------------------------------------------------------
# Area validation (scalar only -- no inference from another static column)
# ---------------------------------------------------------------------------


def _validate_scalar_area(area_km2) -> float:
    if isinstance(area_km2, bool) or not isinstance(area_km2, (int, float, np.integer, np.floating)):
        raise PackageAssemblyError(
            f"area_km2 must be a scalar real number (not array-like), got "
            f"{type(area_km2).__name__}"
        )
    value = float(area_km2)
    if not np.isfinite(value):
        raise PackageAssemblyError(f"area_km2 must be finite, got {area_km2!r}")
    if value <= 0.0:
        raise PackageAssemblyError(f"area_km2 must be strictly positive, got {area_km2!r}")
    return value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assemble_basin_package_table(
    forcing: pd.DataFrame,
    qobs,
    area_km2,
    *,
    policy: dict | None = None,
    expected_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Assemble one basin's canonical scientific package table.

    forcing: hourly pandas.DataFrame indexed by a strictly increasing,
        duplicate-free, gap-free, timezone-naive hourly pandas.DatetimeIndex,
        containing exactly the approved dynamic-input columns
        (:data:`DYNAMIC_INPUTS` by default, or ``policy["dynamic_inputs"]``
        if ``policy`` is given). NaNs are preserved unchanged for the
        continuous meteorological variables; the two gap-flag columns must
        be boolean or numeric containing only finite 0/1 values.
    qobs: pandas.Series of raw discharge [m^3/s], or a pandas.DataFrame
        containing that series under the source-variable column name
        (``qobs_m3s`` by default). Must have a numeric (non-boolean) dtype.
        Its index must equal ``forcing``'s index exactly -- no reindexing is
        performed.
    area_km2: scalar basin area [km^2]; must be a real Python/numpy scalar
        (not array-like), finite, and strictly positive.
    policy: optional already-loaded Stage 1 baseline policy dict (see
        src/baseline/policy.py). It is re-validated here with
        :func:`src.baseline.policy.validate_stage1_baseline_policy` before
        use, so a caller cannot smuggle an unapproved dynamic-input list,
        lead set, source variable, or naming template past this function.
        When given, those four items are taken from it instead of the
        module defaults.
    expected_index: optional exact canonical timeline the forcing index must
        equal (e.g. the full Stage 1 45,720-hour research period). When
        omitted, only the general hourly-grid structural checks apply, so
        short synthetic fixtures remain supported. No reindexing, repair,
        interpolation, aggregation, or filling is ever performed either way.

    Returns a new pandas.DataFrame indexed by the canonical hourly index,
    with explicit column order: dynamic inputs (declared order), the raw
    target column, then the lead targets (declared lead order). Purely
    local: no filesystem access, no global state.
    """
    dynamic_inputs, leads_hours, variable_name_template, source_variable = _resolve_parameters(policy)

    _validate_forcing_table(forcing, dynamic_inputs)
    if expected_index is not None:
        _validate_expected_index(forcing.index, expected_index)

    qobs_series = _extract_qobs_series(qobs, source_variable)
    _validate_qobs_dtype(qobs_series, source_variable)
    if not qobs_series.index.equals(forcing.index):
        raise PackageAssemblyError(
            "qobs index must match the forcing table's canonical hourly index "
            "exactly (same timestamps, same order); this layer does not "
            "reindex, resample, or infer missing rows -- reconcile the source "
            "schedules before calling"
        )
    _reject_negative_finite(qobs_series, source_variable)

    area_value = _validate_scalar_area(area_km2)

    try:
        lead_results = build_lead_targets(
            qobs_series.astype(np.float64),
            area_value,
            leads_hours=leads_hours,
            source_variable=source_variable,
            variable_name_template=variable_name_template,
        )
    except ValueError as exc:
        raise PackageAssemblyError(f"failed to build lead targets: {exc}") from exc
    lead_variable_names = [
        variable_name_for_lead(lead_hours, variable_name_template)
        for lead_hours in leads_hours
    ]

    data = {name: forcing[name] for name in dynamic_inputs}
    data[source_variable] = qobs_series
    for name in lead_variable_names:
        series, _metadata = lead_results[name]
        data[name] = series

    column_order = list(dynamic_inputs) + [source_variable] + lead_variable_names
    table = pd.DataFrame(data, index=forcing.index)
    return table[column_order]

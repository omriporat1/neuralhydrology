"""Local hydrograph rendering machinery (Part L.3, docs/stage1_validation_optimization_foundation.md).

Turns an existing NeuralHydrology validation-results pickle into (a) a
deterministic compact ~6-8-basin observed-vs-predicted comparison panel and
(b) a full rendering of the existing Part C 24-basin hydrograph atlas.

L.3d (docs/stage1_validation_optimization_foundation.md) additionally closes
the standing gaps in the panel/compact renderer for the 50k-serious-triage
comparison standard: basin area in every title
(:func:`format_basin_area_title`); MRMS QPE precipitation bars on an
inverted secondary axis (:func:`load_mrms_series`); observed/predicted
series plotted at their physical target-valid time rather than the NH
result-pickle's issuance time (:func:`compute_target_valid_dates`, applied
exactly once inside :func:`load_basin_series`); an externally-derivable
shared axis scale for cross-candidate comparability
(:func:`derive_comparison_scale`); a per-window compact metrics table
(:func:`compute_compact_event_metrics`); and a structured, non-conclusory
interpretation template (:func:`render_interpretation_template`). These are
assembled for a single candidate by the additive
:func:`render_stage1_compact_comparison_package` entrypoint, which does not
alter :func:`render_stage1_hydrographs`'s existing behavior or output
contract.

This module deliberately reuses, and never reimplements:

- :func:`src.baseline.nh_seed_evaluation.period_results_path` /
  :func:`~src.baseline.nh_seed_evaluation.load_period_results` /
  :func:`~src.baseline.nh_seed_evaluation.basin_netcdf_path` for run-dir ->
  result-pickle / basin-NetCDF path resolution and I/O.
- :mod:`src.baseline.nh_raw_space_evaluation` for basin-area self-derivation,
  mm/h -> m^3/s conversion, and every skill metric (NSE/KGE/RMSE/MAE/
  pearson_r/bias/pbias). This module is the ONLY discharge-conversion and
  metric engine used here.
- :func:`src.baseline.hydrograph_atlas_events.select_atlas_events` for event-
  window selection. That function's signature accepts only an observed
  series (see ``tests/test_hydrograph_atlas_events.py``'s structural proof),
  so predicted discharge cannot influence event selection here either.

Safety boundary: only the ``validation`` period is permitted
(:data:`ALLOWED_PERIODS`) -- the safest choice until a separately-approved
mode for other periods exists. Missing basins, missing target variables, and
malformed result-pickle structures all raise :class:`HydrographRenderingError`
rather than being silently skipped.
"""
from __future__ import annotations

import json
import pickle
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from .hydrograph_atlas_events import EventSelectionError, EventWindow, select_atlas_events
from .nh_raw_space_evaluation import (
    RawSpaceEvaluationError,
    convert_period_to_raw_space,
    derive_basin_area_km2_from_netcdf,
    raw_space_metrics,
)
from .nh_seed_evaluation import (
    NHSeedEvaluationError,
    basin_netcdf_path,
    load_period_results,
    period_results_path,
)
from .splits import sha256_of

__all__ = [
    "HydrographRenderingError",
    "ALLOWED_PERIODS",
    "TRACKED_OUTPUT_FORBIDDEN_SUBDIRS",
    "COMPACT_SELECTION_DIMENSIONS",
    "COMPACT_SELECTION_ALGORITHM_ID",
    "COMPACT_SELECTION_ALGORITHM_VERSION",
    "COMPACT_SELECTION_OMITTED_DIMENSION_NOTE",
    "DEFAULT_COMPACT_TARGET_COUNT",
    "BasinSeries",
    "sha256_of",
    "load_atlas_selection_csv",
    "select_compact_basins",
    "extract_basin_xr",
    "load_basin_series",
    "observed_series_for_events",
    "compute_target_valid_dates",
    "format_basin_area_title",
    "MRMS_QPE_VARIABLE",
    "load_mrms_series",
    "ScaleSpec",
    "derive_comparison_scale",
    "DisplayWindow",
    "derive_display_window",
    "compute_compact_event_metrics",
    "render_interpretation_template",
    "render_basin_panel",
    "render_compact_panel",
    "render_stage1_hydrographs",
    "render_stage1_compact_comparison_package",
]


class HydrographRenderingError(Exception):
    """Raised for input-contract violations: missing basin, missing target
    variable, malformed result-pickle structure, a disallowed period, or an
    inconsistent self-derived basin area. Never raised for an ordinary
    poor-skill outcome."""


ALLOWED_PERIODS = ("validation",)

# Output directories must never land inside these tracked top-level
# directories -- generated figures/pickles/CSVs/manifests are untracked by
# policy (docs/repo_policy.md) and must not become stageable by accident.
TRACKED_OUTPUT_FORBIDDEN_SUBDIRS = ("src", "scripts", "config", "docs")

REQUIRED_ATLAS_COLUMNS = ("gauge_id", "skill_stratum", "area_class", "geo_side", "nse")

COMPACT_SELECTION_DIMENSIONS = ("skill_stratum", "area_class", "geo_side")
COMPACT_SELECTION_ALGORITHM_ID = "stage1_compact_hydrograph_panel_greedy_coverage_v1"
COMPACT_SELECTION_ALGORITHM_VERSION = 1
DEFAULT_COMPACT_TARGET_COUNT = 8

COMPACT_SELECTION_OMITTED_DIMENSION_NOTE = (
    "flashy-vs-smoother hydrologic behavior is NOT one of the compact-panel "
    "stratification dimensions: the hydrograph-atlas selection CSV's only "
    "superficially similar field, hydro_class, is an aridity (wet/dry) "
    "tercile of ari_ix_uav (src/baseline/splits.py assign_tercile_class), "
    "not a hydrograph-shape classification. No substitute classification "
    "was invented; the compact rule stratifies on the three dimensions the "
    "existing metadata actually supports: skill_stratum, area_class, "
    "geo_side."
)


# ---------------------------------------------------------------------------
# Atlas-selection CSV loading + deterministic compact-basin selection
# ---------------------------------------------------------------------------

def load_atlas_selection_csv(path) -> pd.DataFrame:
    """Load a hydrograph-atlas basin-selection CSV (the output of
    :func:`src.baseline.hydrograph_atlas_selection.write_selection_artifacts`,
    or an equivalent frame with the same required columns)."""
    path = Path(path)
    if not path.is_file():
        raise HydrographRenderingError(f"atlas selection CSV not found: {path}")
    df = pd.read_csv(path, dtype={"gauge_id": str})
    missing = [c for c in REQUIRED_ATLAS_COLUMNS if c not in df.columns]
    if missing:
        raise HydrographRenderingError(
            f"{path}: missing required column(s) {missing}; have {list(df.columns)}"
        )
    if df["gauge_id"].duplicated().any():
        dupes = sorted(df.loc[df["gauge_id"].duplicated(), "gauge_id"].unique().tolist())
        raise HydrographRenderingError(f"{path}: duplicate gauge_id value(s) {dupes}")
    return df


def select_compact_basins(
    atlas_df: pd.DataFrame, *, target_count: int = DEFAULT_COMPACT_TARGET_COUNT
) -> "tuple[pd.DataFrame, dict]":
    """Deterministically derive a small (~6-8 basin) subset of ``atlas_df``
    that maximizes coverage across the three supported diversity dimensions
    (:data:`COMPACT_SELECTION_DIMENSIONS`).

    Algorithm (:data:`COMPACT_SELECTION_ALGORITHM_ID`): sort candidates by
    ``gauge_id`` ascending (fixes a canonical order so the result does not
    depend on the caller's row order); then greedily pick, ``target_count``
    times, whichever remaining basin adds the most not-yet-covered
    (dimension, value) pairs across skill_stratum/area_class/geo_side, tie-
    breaking by ascending ``gauge_id``. Once every distinct value in every
    dimension is covered, further picks (if ``target_count`` exceeds the
    number of distinct values) continue in ascending ``gauge_id`` order. This
    is fully deterministic and requires no random seed.

    The fourth requested diversity dimension (flashy vs. smoother
    hydrologic behavior) is not derivable from current atlas metadata --
    see :data:`COMPACT_SELECTION_OMITTED_DIMENSION_NOTE`, also recorded in
    the returned manifest piece under ``"dimension_omitted"``.
    """
    missing = [c for c in REQUIRED_ATLAS_COLUMNS if c not in atlas_df.columns]
    if missing:
        raise HydrographRenderingError(f"atlas_df missing required column(s) {missing}")
    if atlas_df["gauge_id"].duplicated().any():
        raise HydrographRenderingError("atlas_df contains duplicate gauge_id values")
    n = len(atlas_df)
    if not (1 <= target_count <= n):
        raise HydrographRenderingError(
            f"target_count={target_count} must be between 1 and the atlas input size ({n})"
        )

    indexed = atlas_df.set_index("gauge_id", drop=False).sort_index()
    remaining = list(indexed.index)

    covered = {dim: set() for dim in COMPACT_SELECTION_DIMENSIONS}
    picks: list = []
    selection_order: list = []

    while len(picks) < target_count:
        best_gid = None
        best_gain = -1
        for gid in remaining:
            row = indexed.loc[gid]
            gain = sum(1 for dim in COMPACT_SELECTION_DIMENSIONS if row[dim] not in covered[dim])
            if gain > best_gain:
                best_gain = gain
                best_gid = gid
        picks.append(best_gid)
        row = indexed.loc[best_gid]
        for dim in COMPACT_SELECTION_DIMENSIONS:
            covered[dim].add(row[dim])
        selection_order.append({
            "gauge_id": best_gid,
            "step": len(picks),
            "marginal_coverage_gain": int(best_gain),
        })
        remaining.remove(best_gid)

    compact_df = indexed.loc[picks].reset_index(drop=True)

    dimension_coverage = {}
    for dim in COMPACT_SELECTION_DIMENSIONS:
        distinct_in_atlas = set(atlas_df[dim].tolist())
        dimension_coverage[dim] = {
            "distinct_values_in_atlas": sorted(str(v) for v in distinct_in_atlas),
            "distinct_values_covered": sorted(str(v) for v in covered[dim]),
            "fully_covered": covered[dim] >= distinct_in_atlas,
        }
    full_coverage_achieved = all(v["fully_covered"] for v in dimension_coverage.values())

    manifest_piece = {
        "algorithm_id": COMPACT_SELECTION_ALGORITHM_ID,
        "algorithm_version": COMPACT_SELECTION_ALGORITHM_VERSION,
        "target_count": target_count,
        "n_atlas_input_basins": n,
        "dimensions_used": list(COMPACT_SELECTION_DIMENSIONS),
        "dimension_omitted": {
            "name": "flashy_vs_smoother",
            "reason": COMPACT_SELECTION_OMITTED_DIMENSION_NOTE,
        },
        "selection_order": selection_order,
        "compact_gauge_ids": list(picks),
        "dimension_coverage": dimension_coverage,
        "full_coverage_achieved": full_coverage_achieved,
    }
    return compact_df, manifest_piece


# ---------------------------------------------------------------------------
# Result-pickle extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BasinSeries:
    """One basin/period's raw-space (m^3/s) observed/predicted series plus
    the reused raw-space evaluator's metrics for it.

    ``dates`` is the *physical target-valid time* of every sample --
    ``issue_dates + lead_hours`` (:func:`compute_target_valid_dates`), i.e.
    the timestamp of the discharge value that ``obs_m3s``/``sim_m3s`` at
    that index actually describe. This is what every consumer (event-window
    selection, plotting, peak-timing metrics) should use. ``issue_dates`` is
    the raw NH result-pickle 'date' coordinate (issuance time; see
    ``src/baseline/nh_dataset.py``) and is kept only for provenance -- no
    code in this module should plot against it."""

    basin_id: str
    dates: pd.DatetimeIndex
    issue_dates: pd.DatetimeIndex
    obs_m3s: np.ndarray
    sim_m3s: np.ndarray
    admitted_mask: np.ndarray
    area_km2: float
    n_admitted: int
    n_total: int
    metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Time alignment, basin area, MRMS precipitation, and comparison-scale
# helpers (L.3d)
# ---------------------------------------------------------------------------

def compute_target_valid_dates(issue_dates, lead_hours: int) -> pd.DatetimeIndex:
    """Convert an NH result pickle's issuance-time 'date' coordinate into
    the physical valid time of the lead-shifted target it predicts, i.e.
    ``issue_time + lead_hours``. Applied exactly once, inside
    :func:`load_basin_series` -- nothing downstream should call this again
    on an already-converted ``BasinSeries.dates``."""
    if lead_hours < 0:
        raise HydrographRenderingError(f"lead_hours={lead_hours} must be >= 0")
    return pd.DatetimeIndex(pd.to_datetime(issue_dates)) + pd.Timedelta(hours=lead_hours)


def format_basin_area_title(basin_id: str, area_km2) -> str:
    """``"STAID -- Area: 1,234 km^2"`` panel-title fragment. Raises
    :class:`HydrographRenderingError` rather than silently omitting the
    area when it is missing, non-finite, or non-positive."""
    if area_km2 is None or not np.isfinite(area_km2) or area_km2 <= 0:
        raise HydrographRenderingError(
            f"basin {basin_id!r}: area_km2={area_km2!r} is missing/non-finite/non-positive; "
            "refusing to render a panel title with an ambiguous or absent basin area"
        )
    return f"{basin_id} — Area: {area_km2:,.0f} km²"


MRMS_QPE_VARIABLE = "mrms_qpe_1h_mm"


def load_mrms_series(package_root, basin_id: str) -> pd.Series:
    """Basin-average hourly MRMS QPE (mm, i.e. mm/h for an hourly
    accumulation) from the same certified package NetCDF used for basin-area
    self-derivation (:func:`~src.baseline.nh_seed_evaluation.basin_netcdf_path`).
    Indexed by its own physical, unshifted valid-time 'date' coordinate --
    never shifted to visually align with a lead-shifted prediction. Raises
    :class:`HydrographRenderingError` if the file or variable is absent."""
    nc_path = basin_netcdf_path(package_root, basin_id)
    if not Path(nc_path).is_file():
        raise HydrographRenderingError(f"basin {basin_id!r}: package NetCDF not found: {nc_path}")
    with xr.open_dataset(nc_path) as ds:
        if MRMS_QPE_VARIABLE not in ds.data_vars:
            raise HydrographRenderingError(
                f"basin {basin_id!r}: package NetCDF {nc_path} has no {MRMS_QPE_VARIABLE!r} "
                f"variable (have {sorted(ds.data_vars)}); MRMS precipitation cannot be rendered"
            )
        dates = pd.to_datetime(np.asarray(ds["date"].values))
        values = np.asarray(ds[MRMS_QPE_VARIABLE].values, dtype=np.float64).squeeze()
    return pd.Series(values, index=pd.DatetimeIndex(dates), name=MRMS_QPE_VARIABLE)


@dataclass(frozen=True)
class ScaleSpec:
    """A basin/window-scoped set of fixed plot limits so that multiple
    candidates' hydrograph panels for the same basin/window are visually
    comparable. Derive once, via :func:`derive_comparison_scale`, from the
    union of every candidate's series, then pass the *same* ``ScaleSpec*``
    unchanged into each candidate's :func:`render_basin_panel` /
    :func:`render_compact_panel` call -- never re-derive it per candidate."""

    x_min: pd.Timestamp
    x_max: pd.Timestamp
    discharge_min: float
    discharge_max: float
    precip_max: Optional[float] = None


def derive_comparison_scale(
    basin_series_list: Sequence[BasinSeries],
    *,
    window: Optional[EventWindow] = None,
    precip_series_list: Sequence[pd.Series] = (),
    discharge_margin: float = 0.05,
    precip_margin: float = 0.05,
) -> ScaleSpec:
    """Derive one shared :class:`ScaleSpec` from the union of one or more
    candidates' admitted observed/predicted discharge (and, optionally,
    MRMS precipitation) for a single basin -- restricted to ``window`` if
    given, otherwise the full period covered by ``basin_series_list``. Not
    coupled to any fixed number of candidates. Never clips: the returned
    discharge limits are padded beyond the actual min/max of the supplied
    data."""
    if not basin_series_list:
        raise HydrographRenderingError("derive_comparison_scale requires at least one BasinSeries")

    if window is not None:
        x_min, x_max = pd.Timestamp(window.window_start), pd.Timestamp(window.window_end)
    else:
        x_min = min(bs.dates.min() for bs in basin_series_list)
        x_max = max(bs.dates.max() for bs in basin_series_list)

    discharge_chunks = []
    for bs in basin_series_list:
        mask = np.asarray(bs.admitted_mask, dtype=bool)
        if window is not None:
            mask = mask & (bs.dates >= x_min) & (bs.dates <= x_max)
        discharge_chunks.append(np.asarray(bs.obs_m3s)[mask])
        discharge_chunks.append(np.asarray(bs.sim_m3s)[mask])
    all_discharge = np.concatenate(discharge_chunks) if discharge_chunks else np.array([])
    all_discharge = all_discharge[np.isfinite(all_discharge)]
    if all_discharge.size == 0:
        raise HydrographRenderingError(
            "derive_comparison_scale: no finite admitted discharge samples in the requested range"
        )
    d_min, d_max = float(all_discharge.min()), float(all_discharge.max())
    d_span = max(d_max - d_min, 1e-9)
    discharge_min = d_min - discharge_margin * d_span
    discharge_max = d_max + discharge_margin * d_span

    precip_max = None
    if precip_series_list:
        precip_chunks = []
        for series in precip_series_list:
            s = series
            if window is not None:
                s = series[(series.index >= x_min) & (series.index <= x_max)]
            precip_chunks.append(np.asarray(s.values, dtype=np.float64))
        all_precip = np.concatenate(precip_chunks) if precip_chunks else np.array([])
        all_precip = all_precip[np.isfinite(all_precip)]
        if all_precip.size:
            p_max = float(all_precip.max())
            precip_max = p_max * (1.0 + precip_margin) if p_max > 0 else 1.0

    return ScaleSpec(
        x_min=x_min, x_max=x_max,
        discharge_min=discharge_min, discharge_max=discharge_max,
        precip_max=precip_max,
    )


@dataclass(frozen=True)
class DisplayWindow:
    """A basin/event's *display* window for plotting -- derived from an
    already-selected, frozen :class:`~src.baseline.hydrograph_atlas_events.EventWindow`
    by pulling ``window_start`` back to expose more antecedent context,
    without altering the frozen event's own ``peak_time``/``peak_value``/
    ``window_start``/``window_end`` (that object is never mutated). Never
    used for event *selection* -- :func:`derive_display_window` is a purely
    presentational widening applied after the event/peak identity is
    already fixed. Exposes ``window_start``/``window_end`` (rather than
    e.g. ``display_start``/``display_end``) so a :class:`DisplayWindow` is a
    drop-in substitute anywhere an :class:`EventWindow`-like ``window_start``/
    ``window_end`` pair is consumed (e.g. :func:`derive_comparison_scale`)."""

    magnitude_class: str
    peak_time: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    requested_pre_hours: int
    actual_pre_hours: float
    clipped_to_series_start: bool


def derive_display_window(
    window: EventWindow, *, min_pre_hours: int, series_start: Optional[pd.Timestamp] = None,
) -> DisplayWindow:
    """Derive a wider *display* window from an already-selected
    :class:`EventWindow`, pulling ``window_start`` back so at least
    ``min_pre_hours`` precede ``window.peak_time`` -- ``window`` itself
    (peak identity, its own ``window_start``/``window_end``) is read-only
    and never modified or re-derived; this never re-runs event selection.
    ``window_end`` is carried over unchanged (the post-event span is not
    affected). If the widened start would precede ``series_start``, it is
    clipped there instead (``clipped_to_series_start=True``) rather than
    extending past the available data.

    Note on time-alignment: ``window.peak_time`` is the observed event's
    *physical target-valid time* (see :class:`BasinSeries`). A model with
    sequence length ``L`` and lead ``lead_hours`` predicting this exact
    sample used raw input history ending at ``peak_time - lead_hours`` and
    spanning back ``L`` hours from there -- i.e.
    ``[peak_time - lead_hours - L + 1h, peak_time - lead_hours]``, not
    ``[peak_time - L, peak_time]``. Callers labeling ``-L h`` markers
    relative to ``peak_time`` on a display axis should treat them as
    nominal antecedent-context reference lines, not as the model's exact
    raw input boundary."""
    if min_pre_hours < 0:
        raise HydrographRenderingError(f"min_pre_hours={min_pre_hours} must be >= 0")
    requested_start = window.peak_time - pd.Timedelta(hours=min_pre_hours)
    display_start = min(window.window_start, requested_start)
    clipped = False
    if series_start is not None and display_start < series_start:
        display_start = pd.Timestamp(series_start)
        clipped = True
    actual_pre_hours = (window.peak_time - display_start).total_seconds() / 3600.0
    return DisplayWindow(
        magnitude_class=window.magnitude_class,
        peak_time=window.peak_time,
        window_start=display_start,
        window_end=window.window_end,
        requested_pre_hours=min_pre_hours,
        actual_pre_hours=actual_pre_hours,
        clipped_to_series_start=clipped,
    )


def extract_basin_xr(results: Mapping, basin_id: str, target_variable: str, *, freq: Optional[str] = None):
    """Extract and validate one basin's ``xarray.Dataset`` from a loaded NH
    evaluation-results pickle, mirroring the structural checks
    ``nh_evaluation_check.py`` already performs. Raises
    :class:`HydrographRenderingError` for any missing/malformed structure."""
    if not isinstance(results, Mapping):
        raise HydrographRenderingError(
            f"malformed result pickle: expected a dict of basin -> freq -> results, got {type(results)}"
        )
    if basin_id not in results:
        available = sorted(results.keys())
        preview = available[:10]
        raise HydrographRenderingError(
            f"basin {basin_id!r} not found in result pickle "
            f"({len(available)} basin(s) available, e.g. {preview})"
        )
    basin_entry = results[basin_id]
    if not isinstance(basin_entry, Mapping) or not basin_entry:
        raise HydrographRenderingError(
            f"malformed result pickle: basin {basin_id!r} entry is not a non-empty dict of freq -> results"
        )

    if freq is None:
        if len(basin_entry) != 1:
            raise HydrographRenderingError(
                f"basin {basin_id!r} has {len(basin_entry)} frequencies "
                f"({sorted(basin_entry.keys())}); pass freq= explicitly"
            )
        freq = next(iter(basin_entry))
    if freq not in basin_entry:
        raise HydrographRenderingError(
            f"basin {basin_id!r}: frequency {freq!r} not found (available: {sorted(basin_entry.keys())})"
        )
    freq_results = basin_entry[freq]
    if not isinstance(freq_results, Mapping) or "xr" not in freq_results:
        raise HydrographRenderingError(
            f"malformed result pickle: basin {basin_id!r} freq {freq!r} entry is missing the 'xr' key"
        )
    xr_ds = freq_results["xr"]
    if not hasattr(xr_ds, "data_vars") or not hasattr(xr_ds, "coords"):
        raise HydrographRenderingError(
            f"malformed result pickle: basin {basin_id!r} freq {freq!r}: 'xr' value is not an xarray.Dataset"
        )

    obs_key = f"{target_variable}_obs"
    sim_key = f"{target_variable}_sim"
    if obs_key not in xr_ds.data_vars or sim_key not in xr_ds.data_vars:
        raise HydrographRenderingError(
            f"basin {basin_id!r}: target variable {target_variable!r} not present "
            f"(expected data vars {obs_key!r}/{sim_key!r}, have {sorted(xr_ds.data_vars)})"
        )
    if "date" not in xr_ds.coords:
        raise HydrographRenderingError(f"basin {basin_id!r}: 'xr' Dataset has no 'date' coordinate")
    return xr_ds


def load_basin_series(
    *,
    results: Mapping,
    basin_id: str,
    target_variable: str,
    package_root,
    lead_hours: int,
    freq: Optional[str] = None,
    min_area_samples: int = 100,
    max_relative_mad: float = 1e-4,
) -> BasinSeries:
    """Extract one basin's observed/predicted series from an already-loaded
    NH result pickle, self-derive its area, and convert to raw m^3/s with
    metrics -- entirely via reused :mod:`src.baseline.nh_raw_space_evaluation`
    functions (no second conversion/metric implementation)."""
    xr_ds = extract_basin_xr(results, basin_id, target_variable, freq=freq)

    dates = pd.to_datetime(np.asarray(xr_ds["date"].values))
    if len(dates) < 2 or not dates.is_monotonic_increasing:
        raise HydrographRenderingError(
            f"basin {basin_id!r}: 'date' coordinate is not strictly increasing "
            "(refusing to silently reorder observations/predictions)"
        )
    issue_dates = pd.DatetimeIndex(dates)
    target_valid_dates = compute_target_valid_dates(issue_dates, lead_hours)

    obs_mm_per_h = np.asarray(xr_ds[f"{target_variable}_obs"].values, dtype=np.float64).squeeze()
    sim_mm_per_h = np.asarray(xr_ds[f"{target_variable}_sim"].values, dtype=np.float64).squeeze()
    if obs_mm_per_h.shape != (len(dates),) or sim_mm_per_h.shape != (len(dates),):
        raise HydrographRenderingError(
            f"basin {basin_id!r}: obs shape {obs_mm_per_h.shape} / sim shape "
            f"{sim_mm_per_h.shape} do not match the {len(dates)}-length date coordinate"
        )

    nc_path = basin_netcdf_path(package_root, basin_id)
    try:
        area_result = derive_basin_area_km2_from_netcdf(
            nc_path,
            basin_id=basin_id,
            target_variable=target_variable,
            lead_hours=lead_hours,
            min_samples=min_area_samples,
            max_relative_mad=max_relative_mad,
        )
    except RawSpaceEvaluationError as exc:
        raise HydrographRenderingError(str(exc)) from exc
    if not area_result.consistent:
        raise HydrographRenderingError(
            f"basin {basin_id!r}: self-derived area inconsistent "
            f"(relative_mad={area_result.relative_mad:.3g} exceeds {max_relative_mad:.3g})"
        )

    try:
        conversion = convert_period_to_raw_space(obs_mm_per_h, sim_mm_per_h, area_result.area_km2)
        metrics = raw_space_metrics(
            conversion.obs_m3s[conversion.admitted_mask],
            conversion.sim_m3s[conversion.admitted_mask],
        )
    except RawSpaceEvaluationError as exc:
        raise HydrographRenderingError(str(exc)) from exc

    return BasinSeries(
        basin_id=basin_id,
        dates=target_valid_dates,
        issue_dates=issue_dates,
        obs_m3s=conversion.obs_m3s,
        sim_m3s=conversion.sim_m3s,
        admitted_mask=conversion.admitted_mask,
        area_km2=area_result.area_km2,
        n_admitted=conversion.n_admitted,
        n_total=conversion.n_total,
        metrics=metrics,
    )


def observed_series_for_events(basin_series: BasinSeries) -> pd.Series:
    """Observed-discharge-only raw m^3/s series, indexed by date, for
    :func:`src.baseline.hydrograph_atlas_events.select_atlas_events`. Takes
    no predicted/simulated argument -- event selection cannot see
    predictions by construction."""
    return pd.Series(basin_series.obs_m3s, index=basin_series.dates, name=basin_series.basin_id)


def _events_for_basin(basin_series: BasinSeries, *, min_separation_hours=72, pre_hours=24, post_hours=48):
    try:
        return select_atlas_events(
            observed_series_for_events(basin_series),
            min_separation_hours=min_separation_hours,
            pre_hours=pre_hours,
            post_hours=post_hours,
        )
    except EventSelectionError as exc:
        raise HydrographRenderingError(
            f"basin {basin_series.basin_id!r}: event-window selection failed: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_DISPLAY_METRIC_KEYS = ("nse", "kge", "rmse", "pbias")


def _metric_subtitle(metrics: Mapping) -> str:
    parts = []
    for key in _DISPLAY_METRIC_KEYS:
        value = metrics.get(key, float("nan"))
        parts.append(f"{key.upper()}={value:.3f}" if np.isfinite(value) else f"{key.upper()}=n/a")
    return "  ".join(parts)


def _add_precip_axis(ax, precip_series: Optional[pd.Series], scale: Optional[ScaleSpec]) -> None:
    """Add MRMS QPE hourly bars on a secondary right-hand y-axis, inverted
    so zero is at the top and precipitation increases downward -- never a
    six-hour (or any other) shift relative to the primary axis's dates."""
    if precip_series is None or len(precip_series) == 0:
        return
    ax2 = ax.twinx()
    ax2.bar(
        precip_series.index, precip_series.values,
        width=pd.Timedelta(hours=1), color="tab:blue", alpha=0.5, label="MRMS QPE",
    )
    if scale is not None and scale.precip_max is not None:
        precip_top = scale.precip_max
    else:
        finite_vals = np.asarray(precip_series.values, dtype=np.float64)
        finite_vals = finite_vals[np.isfinite(finite_vals)]
        precip_top = float(finite_vals.max()) * 1.05 if finite_vals.size and finite_vals.max() > 0 else 1.0
    ax2.set_ylim(precip_top, 0)  # zero at the top, increasing downward
    ax2.set_ylabel("MRMS QPE (mm h^-1)")


def render_basin_panel(
    basin_series: BasinSeries,
    events: Mapping[str, EventWindow],
    *,
    epoch: int,
    out_path,
    precip_series: Optional[pd.Series] = None,
    scale: Optional[ScaleSpec] = None,
    event_scale_by_class: Optional[Mapping[str, ScaleSpec]] = None,
) -> Path:
    """Render one basin's full-validation-period observed-vs-predicted
    hydrograph plus up to four deterministic event-window zooms.

    ``basin_series.dates`` is already the physical target-valid time (see
    :class:`BasinSeries`), so observed and predicted series are plotted at
    the correct valid time with no further shift here. ``precip_series``
    (if supplied, e.g. from :func:`load_mrms_series`) is plotted at its own
    unshifted valid time on an inverted secondary axis. ``scale``/
    ``event_scale_by_class`` (if supplied, e.g. from
    :func:`derive_comparison_scale`) fix the axis limits so multiple
    candidates' panels for this basin are directly comparable."""
    out_path = Path(out_path)
    area_title = format_basin_area_title(basin_series.basin_id, basin_series.area_km2)
    n_event_cols = max(len(events), 1)
    fig = plt.figure(figsize=(max(10, 3 * n_event_cols), 6))
    grid = fig.add_gridspec(2, n_event_cols, height_ratios=[2, 1])

    ax_full = fig.add_subplot(grid[0, :])
    ax_full.plot(basin_series.dates, basin_series.obs_m3s, label="observed", color="black", linewidth=0.8)
    ax_full.plot(basin_series.dates, basin_series.sim_m3s, label="predicted", color="tab:orange", linewidth=0.8)
    ax_full.set_ylabel("discharge [m^3/s]")
    if scale is not None:
        ax_full.set_xlim(scale.x_min, scale.x_max)
        ax_full.set_ylim(scale.discharge_min, scale.discharge_max)
    _add_precip_axis(ax_full, precip_series, scale)
    ax_full.legend(loc="upper right", fontsize=8)
    ax_full.set_title(
        f"{area_title}  epoch={epoch}  {_metric_subtitle(basin_series.metrics)}",
        fontsize=10,
    )

    for i, (magnitude_class, window) in enumerate(sorted(events.items())):
        ax = fig.add_subplot(grid[1, i])
        mask = (basin_series.dates >= window.window_start) & (basin_series.dates <= window.window_end)
        ax.plot(basin_series.dates[mask], basin_series.obs_m3s[mask], color="black", linewidth=0.9)
        ax.plot(basin_series.dates[mask], basin_series.sim_m3s[mask], color="tab:orange", linewidth=0.9)
        event_scale = (event_scale_by_class or {}).get(magnitude_class)
        if event_scale is not None:
            ax.set_xlim(event_scale.x_min, event_scale.x_max)
            ax.set_ylim(event_scale.discharge_min, event_scale.discharge_max)
        if precip_series is not None:
            event_precip = precip_series[
                (precip_series.index >= window.window_start) & (precip_series.index <= window.window_end)
            ]
            _add_precip_axis(ax, event_precip, event_scale)
        ax.set_title(magnitude_class, fontsize=8)
        ax.tick_params(axis="x", labelrotation=45, labelsize=6)
        ax.tick_params(axis="y", labelsize=6)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def render_compact_panel(
    basin_series_list: Sequence[BasinSeries],
    events_by_basin: Mapping[str, Mapping[str, EventWindow]],
    *,
    epoch: int,
    out_path,
    precip_series_by_basin: Optional[Mapping[str, pd.Series]] = None,
    scale_by_basin: Optional[Mapping[str, ScaleSpec]] = None,
    event_scale_by_basin: Optional[Mapping[str, ScaleSpec]] = None,
) -> Path:
    """Render the compact multi-basin panel: one row per basin, full-period
    context plus its single highest-magnitude available event window.

    ``precip_series_by_basin``/``scale_by_basin``/``event_scale_by_basin``
    (all optional, keyed by ``basin_id``) add the MRMS QPE secondary axis
    and fixed comparison scales for the full-period and event-zoom columns
    respectively -- see :func:`render_basin_panel` for the same per-basin
    conventions applied here per row."""
    out_path = Path(out_path)
    n = len(basin_series_list)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.5 * n), squeeze=False)

    for row, bs in enumerate(basin_series_list):
        ax_full, ax_event = axes[row]
        area_title = format_basin_area_title(bs.basin_id, bs.area_km2)
        ax_full.plot(bs.dates, bs.obs_m3s, label="observed", color="black", linewidth=0.7)
        ax_full.plot(bs.dates, bs.sim_m3s, label="predicted", color="tab:orange", linewidth=0.7)
        ax_full.set_ylabel(f"{bs.basin_id}\n[m^3/s]", fontsize=8)
        ax_full.set_title(f"{area_title}  {_metric_subtitle(bs.metrics)}", fontsize=8)

        scale = (scale_by_basin or {}).get(bs.basin_id)
        if scale is not None:
            ax_full.set_xlim(scale.x_min, scale.x_max)
            ax_full.set_ylim(scale.discharge_min, scale.discharge_max)
        precip_series = (precip_series_by_basin or {}).get(bs.basin_id)
        _add_precip_axis(ax_full, precip_series, scale)

        if row == 0:
            ax_full.legend(loc="upper right", fontsize=7)

        events = events_by_basin.get(bs.basin_id, {})
        if events:
            _, window = sorted(events.items(), key=lambda kv: kv[1].peak_value, reverse=True)[0]
            mask = (bs.dates >= window.window_start) & (bs.dates <= window.window_end)
            ax_event.plot(bs.dates[mask], bs.obs_m3s[mask], color="black", linewidth=0.9)
            ax_event.plot(bs.dates[mask], bs.sim_m3s[mask], color="tab:orange", linewidth=0.9)
            ax_event.set_title("largest available event", fontsize=7)

            event_scale = (event_scale_by_basin or {}).get(bs.basin_id)
            if event_scale is not None:
                ax_event.set_xlim(event_scale.x_min, event_scale.x_max)
                ax_event.set_ylim(event_scale.discharge_min, event_scale.discharge_max)
            if precip_series is not None:
                event_precip = precip_series[
                    (precip_series.index >= window.window_start) & (precip_series.index <= window.window_end)
                ]
                _add_precip_axis(ax_event, event_precip, event_scale)
        else:
            ax_event.set_axis_off()
        ax_event.tick_params(axis="x", labelrotation=45, labelsize=6)
        ax_full.tick_params(axis="x", labelrotation=45, labelsize=6)

    fig.suptitle(f"Stage 1 compact hydrograph panel -- epoch {epoch}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Compact metrics table + interpretation template (L.3d items 6-7)
# ---------------------------------------------------------------------------

_COMPACT_METRICS_COLUMNS = (
    "candidate_id", "basin_id", "area_km2", "window_id", "window_start", "window_end", "n_admitted",
    "nse", "kge", "rmse", "mae", "bias", "pbias",
    "obs_peak_value", "obs_peak_time", "sim_peak_value", "sim_peak_time",
    "peak_magnitude_error", "peak_timing_error_hours",
)


def compute_compact_event_metrics(
    basin_series_by_id: Mapping[str, BasinSeries],
    events_by_basin: Mapping[str, Mapping[str, EventWindow]],
    *,
    candidate_id: Optional[str] = None,
) -> pd.DataFrame:
    """Per basin+window compact metrics table: basin ID, area km^2, window
    ID, admitted sample count, NSE/KGE/RMSE/MAE/bias/PBIAS (all reused from
    :func:`~src.baseline.nh_raw_space_evaluation.raw_space_metrics`, never
    reimplemented), observed/predicted peak magnitude+time, and peak
    magnitude/timing error. Deterministic row ordering: ascending
    ``basin_id`` then ascending ``window_id`` (magnitude_class)."""
    rows = []
    for basin_id in sorted(basin_series_by_id):
        bs = basin_series_by_id[basin_id]
        events = events_by_basin.get(basin_id, {})
        for magnitude_class, window in sorted(events.items()):
            mask = (
                np.asarray(bs.admitted_mask, dtype=bool)
                & (bs.dates >= window.window_start) & (bs.dates <= window.window_end)
            )
            n_admitted = int(mask.sum())
            row = {
                "candidate_id": candidate_id,
                "basin_id": basin_id,
                "area_km2": bs.area_km2,
                "window_id": magnitude_class,
                "window_start": window.window_start,
                "window_end": window.window_end,
                "n_admitted": n_admitted,
            }
            if n_admitted == 0:
                row.update({k: float("nan") for k in ("nse", "kge", "rmse", "mae", "bias", "pbias")})
                row.update({
                    "obs_peak_value": float("nan"), "obs_peak_time": pd.NaT,
                    "sim_peak_value": float("nan"), "sim_peak_time": pd.NaT,
                    "peak_magnitude_error": float("nan"), "peak_timing_error_hours": float("nan"),
                })
            else:
                obs_win = np.asarray(bs.obs_m3s)[mask]
                sim_win = np.asarray(bs.sim_m3s)[mask]
                dates_win = bs.dates[mask]
                window_metrics = raw_space_metrics(obs_win, sim_win)
                obs_peak_idx = int(np.nanargmax(obs_win))
                sim_peak_idx = int(np.nanargmax(sim_win))
                obs_peak_value, obs_peak_time = float(obs_win[obs_peak_idx]), dates_win[obs_peak_idx]
                sim_peak_value, sim_peak_time = float(sim_win[sim_peak_idx]), dates_win[sim_peak_idx]
                row.update({
                    "nse": window_metrics.get("nse"), "kge": window_metrics.get("kge"),
                    "rmse": window_metrics.get("rmse"), "mae": window_metrics.get("mae"),
                    "bias": window_metrics.get("bias"), "pbias": window_metrics.get("pbias"),
                    "obs_peak_value": obs_peak_value, "obs_peak_time": obs_peak_time,
                    "sim_peak_value": sim_peak_value, "sim_peak_time": sim_peak_time,
                    "peak_magnitude_error": sim_peak_value - obs_peak_value,
                    "peak_timing_error_hours": (sim_peak_time - obs_peak_time).total_seconds() / 3600.0,
                })
            rows.append(row)
    return pd.DataFrame(rows, columns=list(_COMPACT_METRICS_COLUMNS))


def render_interpretation_template(
    basin_ids: Sequence[str], *, out_path, candidate_id: Optional[str] = None,
) -> Path:
    """Write a structured, non-conclusory Markdown interpretation template:
    one section per basin with fill-in-the-blank prompts for a human
    reviewer covering peak magnitude, peak timing, false peaks, recession,
    baseflow, basin-specific bias, and rainfall-runoff timing. No
    hydrologic conclusion is auto-generated here."""
    out_path = Path(out_path)
    lines = [
        f"# Stage 1 hydrograph interpretation -- {candidate_id or '(candidate id not supplied)'}",
        "",
        "Structured human-review template. Every bullet below is a prompt for",
        "a reviewer to fill in from the rendered compact panel, per-basin",
        "figures, and `compact_event_metrics.csv` -- nothing in this file is",
        "an auto-generated hydrologic conclusion. Observations (black) and",
        "predictions (orange) must be explicitly distinguished; MRMS",
        "precipitation's own physical valid time must be discussed",
        "separately from the lead-6 prediction's target valid time.",
        "",
    ]
    for basin_id in basin_ids:
        lines.extend([
            f"## Basin {basin_id}",
            "",
            "- **Peak magnitude** (observed vs. predicted, from the compact metrics table): ",
            "- **Peak timing** (observed vs. predicted, hours): ",
            "- **False peaks** (predicted peaks not present in the observed record): ",
            "- **Recession behavior**: ",
            "- **Baseflow behavior**: ",
            "- **Basin-specific over-/under-prediction bias**: ",
            "- **Rainfall-runoff timing** (MRMS QPE valid-time bars vs. observed/predicted response): ",
            "- **Rainfall/discharge mismatches** (rainfall with no discharge response, or vice versa): ",
            "",
        ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _current_git_commit(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=10
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _package_manifest_sha256(package_root) -> Optional[str]:
    manifest_path = Path(package_root) / "manifests" / "package_manifest.json"
    return sha256_of(manifest_path) if manifest_path.is_file() else None


def _assert_out_dir_is_safe(out_dir: Path, repo_root: Path, *, force: bool) -> None:
    """Reject an ``out_dir`` that resolves inside a tracked top-level
    directory (``src``/``scripts``/``config``/``docs``), and reject a
    pre-existing non-empty ``out_dir`` unless ``force=True`` -- mirroring
    :func:`src.baseline.hydrograph_atlas_selection.write_selection_artifacts`'s
    overwrite convention, so repeated runs never silently mix outputs from a
    stale previous run into a manifest that claims a fresh, complete one."""
    resolved_out_dir = out_dir.resolve()
    resolved_repo_root = repo_root.resolve()
    for forbidden in TRACKED_OUTPUT_FORBIDDEN_SUBDIRS:
        forbidden_path = resolved_repo_root / forbidden
        if resolved_out_dir == forbidden_path or forbidden_path in resolved_out_dir.parents:
            raise HydrographRenderingError(
                f"--out-dir {out_dir} resolves inside the tracked directory {forbidden_path}; "
                "generated hydrograph-rendering outputs must never be written under "
                "src/, scripts/, config/, or docs/"
            )
    if resolved_out_dir.exists() and any(resolved_out_dir.iterdir()) and not force:
        raise HydrographRenderingError(
            f"output directory already exists and is non-empty: {out_dir} "
            "(pass force=True / --force to overwrite, or use a fresh directory)"
        )


def _event_window_rows(basin_id: str, events: Mapping[str, EventWindow]) -> list:
    rows = []
    for magnitude_class, window in sorted(events.items()):
        rows.append({
            "basin_id": basin_id,
            "magnitude_class": magnitude_class,
            "peak_time": window.peak_time,
            "peak_value": window.peak_value,
            "window_start": window.window_start,
            "window_end": window.window_end,
            "window_clipped": window.window_clipped,
            "n_missing_in_window": window.n_missing_in_window,
        })
    return rows


def render_stage1_hydrographs(
    *,
    run_dir=None,
    result_pickle=None,
    period: str,
    epoch: int,
    package_root,
    target_variable: str,
    lead_hours: int,
    atlas_csv,
    out_dir,
    mode: str = "both",
    compact_target_count: int = DEFAULT_COMPACT_TARGET_COUNT,
    freq: Optional[str] = None,
    min_area_samples: int = 100,
    max_relative_mad: float = 1e-4,
    write_outputs: bool = True,
    force: bool = False,
    repo_root: Optional[Path] = None,
) -> dict:
    """Render the compact panel and/or the full atlas for one already-
    completed NH run's evaluation results. Reads only; never trains or
    evaluates. Only ``period="validation"`` is permitted.

    If ``run_dir`` is supplied (rather than an explicit ``result_pickle``),
    the result-pickle path is resolved via
    :func:`src.baseline.nh_seed_evaluation.period_results_path` -- never
    reconstructed independently.
    """
    if period not in ALLOWED_PERIODS:
        raise HydrographRenderingError(
            f"period={period!r} is not permitted for Stage 1 hydrograph rendering "
            f"(only {ALLOWED_PERIODS} allowed)"
        )
    if mode not in ("compact", "full", "both"):
        raise HydrographRenderingError(f"mode={mode!r} must be one of 'compact'/'full'/'both'")
    if result_pickle is None and run_dir is None:
        raise HydrographRenderingError("either run_dir or result_pickle must be supplied")
    if result_pickle is not None and run_dir is not None:
        raise HydrographRenderingError(
            "result_pickle and run_dir were both supplied; pass exactly one so the result-pickle "
            "path is never ambiguous between an explicit path and a run_dir/period/epoch resolution"
        )

    resolved_repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    if write_outputs:
        _assert_out_dir_is_safe(Path(out_dir), resolved_repo_root, force=force)

    if result_pickle is not None:
        result_pickle_path = Path(result_pickle)
        if not result_pickle_path.is_file():
            raise HydrographRenderingError(f"result pickle not found: {result_pickle_path}")
        with open(result_pickle_path, "rb") as fh:
            results = pickle.load(fh)
    else:
        result_pickle_path = period_results_path(run_dir, period, epoch)
        try:
            results = load_period_results(run_dir, period, epoch)
        except NHSeedEvaluationError as exc:
            raise HydrographRenderingError(str(exc)) from exc

    atlas_df = load_atlas_selection_csv(atlas_csv)

    compact_df = None
    compact_manifest_piece = None
    if mode in ("compact", "both"):
        compact_df, compact_manifest_piece = select_compact_basins(atlas_df, target_count=compact_target_count)

    if mode == "compact":
        atlas_basin_ids = list(compact_df["gauge_id"])
    else:
        atlas_basin_ids = list(atlas_df["gauge_id"])
    working_basin_ids = sorted(set(atlas_basin_ids) | (set(compact_df["gauge_id"]) if compact_df is not None else set()))

    basin_series_by_id: dict = {}
    events_by_basin: dict = {}
    for basin_id in working_basin_ids:
        bs = load_basin_series(
            results=results,
            basin_id=basin_id,
            target_variable=target_variable,
            package_root=package_root,
            lead_hours=lead_hours,
            freq=freq,
            min_area_samples=min_area_samples,
            max_relative_mad=max_relative_mad,
        )
        basin_series_by_id[basin_id] = bs
        events_by_basin[basin_id] = _events_for_basin(bs)

    out_dir = Path(out_dir)
    output_files: dict = {}

    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)

        if mode in ("compact", "both"):
            compact_series = [basin_series_by_id[gid] for gid in compact_df["gauge_id"]]
            compact_events = {gid: events_by_basin[gid] for gid in compact_df["gauge_id"]}
            compact_panel_path = render_compact_panel(
                compact_series, compact_events, epoch=epoch, out_path=out_dir / "compact_panel.png"
            )
            output_files["compact_panel.png"] = compact_panel_path

            membership_path = out_dir / "compact_basin_membership.json"
            membership_path.write_text(
                json.dumps(compact_manifest_piece, indent=2, default=str), encoding="utf-8"
            )
            output_files["compact_basin_membership.json"] = membership_path

        if mode in ("full", "both"):
            atlas_dir = out_dir / "atlas"
            for basin_id in atlas_df["gauge_id"]:
                bs = basin_series_by_id[basin_id]
                fig_path = render_basin_panel(
                    bs, events_by_basin[basin_id], epoch=epoch, out_path=atlas_dir / f"{basin_id}.png"
                )
                output_files[f"atlas/{basin_id}.png"] = fig_path

        metrics_rows = []
        for basin_id in working_basin_ids:
            bs = basin_series_by_id[basin_id]
            metrics_rows.append({
                "basin_id": basin_id,
                "area_km2": bs.area_km2,
                "n_admitted": bs.n_admitted,
                "n_total": bs.n_total,
                **bs.metrics,
            })
        metrics_csv_path = out_dir / "per_basin_metrics.csv"
        pd.DataFrame(metrics_rows).to_csv(metrics_csv_path, index=False)
        output_files["per_basin_metrics.csv"] = metrics_csv_path

        event_rows = []
        for basin_id in working_basin_ids:
            event_rows.extend(_event_window_rows(basin_id, events_by_basin[basin_id]))
        event_table_path = out_dir / "event_window_table.csv"
        pd.DataFrame(event_rows).to_csv(event_table_path, index=False)
        output_files["event_window_table.csv"] = event_table_path

        output_sha256 = {name: sha256_of(path) for name, path in output_files.items()}

        manifest = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "period": period,
            "epoch": epoch,
            "target_variable": target_variable,
            "lead_hours": lead_hours,
            "result_pickle_path": str(result_pickle_path),
            "result_pickle_sha256": sha256_of(result_pickle_path),
            "package_root": str(Path(package_root)),
            "package_manifest_sha256": _package_manifest_sha256(package_root),
            "atlas_csv_path": str(Path(atlas_csv)),
            "atlas_csv_sha256": sha256_of(atlas_csv),
            "compact_selection": compact_manifest_piece,
            "rendered_basin_ids": working_basin_ids,
            "event_selection_basis": "observed_discharge_only",
            "raw_space_conversion_source": "src.baseline.nh_raw_space_evaluation (reused, not reimplemented)",
            "plotting_implementation_git_commit": _current_git_commit(resolved_repo_root),
            "output_files": {name: {"path": str(path), "sha256": output_sha256[name]} for name, path in output_files.items()},
        }
        manifest_path = out_dir / "rendering_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        output_files["rendering_manifest.json"] = manifest_path

        summary = {
            "mode": mode,
            "period": period,
            "epoch": epoch,
            "n_basins_rendered": len(working_basin_ids),
            "compact_basin_ids": list(compact_df["gauge_id"]) if compact_df is not None else [],
            "atlas_basin_ids": list(atlas_df["gauge_id"]) if mode in ("full", "both") else [],
            "output_dir": str(out_dir),
            "manifest_path": str(manifest_path),
        }
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        output_files["summary.json"] = summary_path

        return {**summary, "manifest": manifest, "output_files": {n: str(p) for n, p in output_files.items()}}

    return {
        "mode": mode,
        "period": period,
        "epoch": epoch,
        "n_basins_rendered": len(working_basin_ids),
        "compact_basin_ids": list(compact_df["gauge_id"]) if compact_df is not None else [],
        "atlas_basin_ids": list(atlas_df["gauge_id"]) if mode in ("full", "both") else [],
        "dry_run": True,
    }


# ---------------------------------------------------------------------------
# Standard 50k-serious-triage compact comparison package (L.3d)
# ---------------------------------------------------------------------------

COMPACT_COMPARISON_TIME_ALIGNMENT_CONVENTION = {
    "issue_time_coordinate": (
        "the NH result-pickle 'date' coordinate is the issuance time (see "
        "src/baseline/nh_dataset.py); it is NOT the physical valid time of "
        "the lead-shifted target"
    ),
    "observation_valid_time": "issue_time + lead_hours (compute_target_valid_dates)",
    "prediction_valid_time": (
        "issue_time + lead_hours (compute_target_valid_dates); same basis as the "
        "observation -- no separate/additional shift"
    ),
    "double_shift_guard": "the shift is applied exactly once, inside load_basin_series",
    "precipitation_valid_time": (
        "MRMS QPE is plotted at its own unshifted physical valid time from the "
        "package NetCDF 'date' coordinate; never shifted to visually align with "
        "the lead-6 prediction"
    ),
}


def render_stage1_compact_comparison_package(
    *,
    result_pickle,
    epoch: int,
    package_root,
    target_variable: str,
    lead_hours: int,
    atlas_csv,
    out_dir,
    candidate_id: str,
    compact_target_count: int = DEFAULT_COMPACT_TARGET_COUNT,
    freq: Optional[str] = None,
    min_area_samples: int = 100,
    max_relative_mad: float = 1e-4,
    scale_by_basin: Optional[Mapping[str, ScaleSpec]] = None,
    event_scale_by_basin: Optional[Mapping[str, ScaleSpec]] = None,
    render_individual_basin_panels: bool = True,
    force: bool = False,
    repo_root: Optional[Path] = None,
) -> dict:
    """Render the standard 50k-serious-triage fixed eight-basin compact
    comparison package (docs/stage1_validation_optimization_foundation.md,
    L.3d): area-titled panels (:func:`format_basin_area_title`), MRMS QPE
    precipitation bars (:func:`load_mrms_series`), observed/predicted series
    plotted at physical target-valid time (:class:`BasinSeries`), an
    optional externally-supplied shared ``scale_by_basin``/
    ``event_scale_by_basin`` for cross-candidate visual comparability
    (:func:`derive_comparison_scale`), a per-window compact metrics table
    (:func:`compute_compact_event_metrics`), and a structured interpretation
    template (:func:`render_interpretation_template`).

    This is additive: it reuses the same low-level building blocks as
    :func:`render_stage1_hydrographs` (basin-series loading, event-window
    selection, the compact-basin selection) but writes its own, separate
    output set and does not alter :func:`render_stage1_hydrographs`'s
    existing behavior, signature, or output contract.

    Comparability across candidates is achieved by calling this function
    once per candidate with the *same* externally-derived
    ``scale_by_basin``/``event_scale_by_basin`` -- this function itself
    reads and renders exactly one candidate's already-completed
    ``result_pickle``. Only the fixed ``validation`` period's already-
    computed result pickle is ever read: no period/scope argument is
    exposed here, and no run_dir/period resolution, training, or evaluation
    is performed."""
    resolved_repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    out_dir = Path(out_dir)
    _assert_out_dir_is_safe(out_dir, resolved_repo_root, force=force)

    result_pickle_path = Path(result_pickle)
    if not result_pickle_path.is_file():
        raise HydrographRenderingError(f"result pickle not found: {result_pickle_path}")
    with open(result_pickle_path, "rb") as fh:
        results = pickle.load(fh)

    atlas_df = load_atlas_selection_csv(atlas_csv)
    compact_df, compact_manifest_piece = select_compact_basins(atlas_df, target_count=compact_target_count)
    compact_basin_ids = list(compact_df["gauge_id"])

    basin_series_by_id: dict = {}
    events_by_basin: dict = {}
    precip_by_basin: dict = {}
    for basin_id in compact_basin_ids:
        bs = load_basin_series(
            results=results, basin_id=basin_id, target_variable=target_variable,
            package_root=package_root, lead_hours=lead_hours, freq=freq,
            min_area_samples=min_area_samples, max_relative_mad=max_relative_mad,
        )
        basin_series_by_id[basin_id] = bs
        events_by_basin[basin_id] = _events_for_basin(bs)
        precip_by_basin[basin_id] = load_mrms_series(package_root, basin_id)

    out_dir.mkdir(parents=True, exist_ok=True)
    output_files: dict = {}

    compact_series = [basin_series_by_id[gid] for gid in compact_basin_ids]
    compact_panel_path = render_compact_panel(
        compact_series, events_by_basin, epoch=epoch, out_path=out_dir / "compact_panel.png",
        precip_series_by_basin=precip_by_basin, scale_by_basin=scale_by_basin,
        event_scale_by_basin=event_scale_by_basin,
    )
    output_files["compact_panel.png"] = compact_panel_path

    if render_individual_basin_panels:
        basin_dir = out_dir / "basin_panels"
        for basin_id in compact_basin_ids:
            bs = basin_series_by_id[basin_id]
            fig_path = render_basin_panel(
                bs, events_by_basin[basin_id], epoch=epoch, out_path=basin_dir / f"{basin_id}.png",
                precip_series=precip_by_basin[basin_id], scale=(scale_by_basin or {}).get(basin_id),
            )
            output_files[f"basin_panels/{basin_id}.png"] = fig_path

    membership_path = out_dir / "compact_basin_membership.json"
    membership_path.write_text(json.dumps(compact_manifest_piece, indent=2, default=str), encoding="utf-8")
    output_files["compact_basin_membership.json"] = membership_path

    metrics_table = compute_compact_event_metrics(basin_series_by_id, events_by_basin, candidate_id=candidate_id)
    metrics_csv_path = out_dir / "compact_event_metrics.csv"
    metrics_table.to_csv(metrics_csv_path, index=False)
    output_files["compact_event_metrics.csv"] = metrics_csv_path

    event_rows = []
    for basin_id in compact_basin_ids:
        event_rows.extend(_event_window_rows(basin_id, events_by_basin[basin_id]))
    event_table_path = out_dir / "event_window_table.csv"
    pd.DataFrame(event_rows).to_csv(event_table_path, index=False)
    output_files["event_window_table.csv"] = event_table_path

    interpretation_path = render_interpretation_template(
        compact_basin_ids, out_path=out_dir / "interpretation_template.md", candidate_id=candidate_id,
    )
    output_files["interpretation_template.md"] = interpretation_path

    output_sha256 = {name: sha256_of(path) for name, path in output_files.items()}

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_standard": (
            "stage1_compact_hydrograph_comparison_package_v1 "
            "(docs/stage1_validation_optimization_foundation.md L.3d)"
        ),
        "candidate_id": candidate_id,
        "period": "validation",
        "epoch": epoch,
        "target_variable": target_variable,
        "lead_hours": lead_hours,
        "result_pickle_path": str(result_pickle_path),
        "result_pickle_sha256": sha256_of(result_pickle_path),
        "package_root": str(Path(package_root)),
        "package_manifest_sha256": _package_manifest_sha256(package_root),
        "atlas_csv_path": str(Path(atlas_csv)),
        "atlas_csv_sha256": sha256_of(atlas_csv),
        "compact_selection": compact_manifest_piece,
        "rendered_basin_ids": compact_basin_ids,
        "mrms_qpe_variable": MRMS_QPE_VARIABLE,
        "time_alignment_convention": COMPACT_COMPARISON_TIME_ALIGNMENT_CONVENTION,
        "shared_scale_supplied": scale_by_basin is not None,
        "event_selection_basis": "observed_discharge_only",
        "raw_space_conversion_source": "src.baseline.nh_raw_space_evaluation (reused, not reimplemented)",
        "plotting_implementation_git_commit": _current_git_commit(resolved_repo_root),
        "output_files": {name: {"path": str(path), "sha256": output_sha256[name]} for name, path in output_files.items()},
    }
    manifest_path = out_dir / "rendering_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    output_files["rendering_manifest.json"] = manifest_path

    summary = {
        "candidate_id": candidate_id,
        "epoch": epoch,
        "compact_basin_ids": compact_basin_ids,
        "output_dir": str(out_dir),
        "manifest_path": str(manifest_path),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    output_files["summary.json"] = summary_path

    return {**summary, "manifest": manifest, "output_files": {n: str(p) for n, p in output_files.items()}}

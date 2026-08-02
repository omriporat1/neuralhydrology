"""Local hydrograph rendering machinery (Part L.3, docs/stage1_validation_optimization_foundation.md).

Turns an existing NeuralHydrology validation-results pickle into (a) a
deterministic compact ~6-8-basin observed-vs-predicted comparison panel and
(b) a full rendering of the existing Part C 24-basin hydrograph atlas.

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
    "render_basin_panel",
    "render_compact_panel",
    "render_stage1_hydrographs",
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
    the reused raw-space evaluator's metrics for it."""

    basin_id: str
    dates: pd.DatetimeIndex
    obs_m3s: np.ndarray
    sim_m3s: np.ndarray
    admitted_mask: np.ndarray
    area_km2: float
    n_admitted: int
    n_total: int
    metrics: dict = field(default_factory=dict)


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
        dates=pd.DatetimeIndex(dates),
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


def render_basin_panel(
    basin_series: BasinSeries,
    events: Mapping[str, EventWindow],
    *,
    epoch: int,
    out_path,
) -> Path:
    """Render one basin's full-validation-period observed-vs-predicted
    hydrograph plus up to four deterministic event-window zooms."""
    out_path = Path(out_path)
    n_event_cols = max(len(events), 1)
    fig = plt.figure(figsize=(max(10, 3 * n_event_cols), 6))
    grid = fig.add_gridspec(2, n_event_cols, height_ratios=[2, 1])

    ax_full = fig.add_subplot(grid[0, :])
    ax_full.plot(basin_series.dates, basin_series.obs_m3s, label="observed", color="black", linewidth=0.8)
    ax_full.plot(basin_series.dates, basin_series.sim_m3s, label="predicted", color="tab:orange", linewidth=0.8)
    ax_full.set_ylabel("discharge [m^3/s]")
    ax_full.legend(loc="upper right", fontsize=8)
    ax_full.set_title(
        f"{basin_series.basin_id}  epoch={epoch}  {_metric_subtitle(basin_series.metrics)}",
        fontsize=10,
    )

    for i, (magnitude_class, window) in enumerate(sorted(events.items())):
        ax = fig.add_subplot(grid[1, i])
        mask = (basin_series.dates >= window.window_start) & (basin_series.dates <= window.window_end)
        ax.plot(basin_series.dates[mask], basin_series.obs_m3s[mask], color="black", linewidth=0.9)
        ax.plot(basin_series.dates[mask], basin_series.sim_m3s[mask], color="tab:orange", linewidth=0.9)
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
) -> Path:
    """Render the compact multi-basin panel: one row per basin, full-period
    context plus its single highest-magnitude available event window."""
    out_path = Path(out_path)
    n = len(basin_series_list)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.5 * n), squeeze=False)

    for row, bs in enumerate(basin_series_list):
        ax_full, ax_event = axes[row]
        ax_full.plot(bs.dates, bs.obs_m3s, label="observed", color="black", linewidth=0.7)
        ax_full.plot(bs.dates, bs.sim_m3s, label="predicted", color="tab:orange", linewidth=0.7)
        ax_full.set_ylabel(f"{bs.basin_id}\n[m^3/s]", fontsize=8)
        ax_full.set_title(_metric_subtitle(bs.metrics), fontsize=8)
        if row == 0:
            ax_full.legend(loc="upper right", fontsize=7)

        events = events_by_basin.get(bs.basin_id, {})
        if events:
            _, window = sorted(events.items(), key=lambda kv: kv[1].peak_value, reverse=True)[0]
            mask = (bs.dates >= window.window_start) & (bs.dates <= window.window_end)
            ax_event.plot(bs.dates[mask], bs.obs_m3s[mask], color="black", linewidth=0.9)
            ax_event.plot(bs.dates[mask], bs.sim_m3s[mask], color="tab:orange", linewidth=0.9)
            ax_event.set_title("largest available event", fontsize=7)
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

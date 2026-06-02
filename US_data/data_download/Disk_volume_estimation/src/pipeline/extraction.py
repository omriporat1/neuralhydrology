"""One-hour basin-statistic extraction for MRMS QPE and RTMA analysis grids.

Grid indexing conventions (validated from Milestone 2A grid-definition discovery):
  MRMS regular_ll  (3500 rows × 7000 cols):
    jScansPositively=0  →  arr[0] = northernmost row
    row_idx in weight table maps directly to arr[row_idx, col_idx]
  RTMA Lambert Conformal (1597 rows × 2345 cols):
    jScansPositively=1  →  arr[0] = southernmost row
    row_idx in weight table maps directly to arr[row_idx, col_idx]

MRMS units note:
  GRIB_units and GRIB_name are both 'unknown' — MRMS QPE GRIB2 encoding quirk.
  The physical variable is hourly precipitation accumulation in millimetres (mm).
  See _MRMS_UNITS_CAVEAT for the full note stored in extraction outputs.
"""

from __future__ import annotations

import gzip
import logging
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

_MRMS_PRODUCT = "mrms_qpe_1h_pass1"
_MRMS_SOURCE  = "noaa-mrms-pds"
_MRMS_STANDARD_NAME = "precipitation_amount_1h"
_MRMS_UNITS_CAVEAT = (
    "GRIB_units='unknown': MRMS QPE GRIB2 files omit units metadata; "
    "product is documented as millimetres (mm) 1-hour accumulation."
)

_RTMA_PRODUCT = "rtma_conus_aws_2p5km"
_RTMA_SOURCE  = "noaa-rtma-pds"

# Variables excluded from the default dynamic-forcing output.
# 10wdir: circular variable; linear averaging is invalid (359° and 1° should
#         average near 0°, not 180°). Wind direction can be derived from u/v.
# orog:   static terrain field; not a dynamic meteorological variable.
#         Belongs with static attributes, not the hourly forcing time series.
_RTMA_EXCLUDED_DEFAULT: frozenset[str] = frozenset({"10wdir", "orog"})

# Variable metadata: role, circularity flag, recommendation status.
# Role values:
#   core_dynamic_candidate     – primary forcing for ML model training
#   optional_dynamic_candidate – secondary forcing; useful for some architectures
#   diagnostic_only            – extracted for QC, not recommended for training
#   excluded_by_default        – decoded but excluded from default output
_VAR_METADATA: dict[str, dict] = {
    # MRMS
    "unknown":  {"role": "core_dynamic_candidate",     "circular": False, "recommended": True},
    # RTMA core
    "2t":       {"role": "core_dynamic_candidate",     "circular": False, "recommended": True},
    "t2m":      {"role": "core_dynamic_candidate",     "circular": False, "recommended": True},
    "2d":       {"role": "core_dynamic_candidate",     "circular": False, "recommended": True},
    "d2m":      {"role": "core_dynamic_candidate",     "circular": False, "recommended": True},
    "2sh":      {"role": "core_dynamic_candidate",     "circular": False, "recommended": True},
    "sh2":      {"role": "core_dynamic_candidate",     "circular": False, "recommended": True},
    "q":        {"role": "core_dynamic_candidate",     "circular": False, "recommended": True},
    "10u":      {"role": "core_dynamic_candidate",     "circular": False, "recommended": True},
    "u10":      {"role": "core_dynamic_candidate",     "circular": False, "recommended": True},
    "ugrd":     {"role": "core_dynamic_candidate",     "circular": False, "recommended": True},
    "10v":      {"role": "core_dynamic_candidate",     "circular": False, "recommended": True},
    "v10":      {"role": "core_dynamic_candidate",     "circular": False, "recommended": True},
    "vgrd":     {"role": "core_dynamic_candidate",     "circular": False, "recommended": True},
    # RTMA optional
    "sp":       {"role": "optional_dynamic_candidate", "circular": False, "recommended": False},
    "pres":     {"role": "optional_dynamic_candidate", "circular": False, "recommended": False},
    "pressfc":  {"role": "optional_dynamic_candidate", "circular": False, "recommended": False},
    "10si":     {"role": "optional_dynamic_candidate", "circular": False, "recommended": False},
    "i10fg":    {"role": "optional_dynamic_candidate", "circular": False, "recommended": False},
    "tcc":      {"role": "optional_dynamic_candidate", "circular": False, "recommended": False},
    "tcdc":     {"role": "optional_dynamic_candidate", "circular": False, "recommended": False},
    # RTMA diagnostic only
    "ceil":     {"role": "diagnostic_only",            "circular": False, "recommended": False},
    "vis":      {"role": "diagnostic_only",            "circular": False, "recommended": False},
    # RTMA excluded by default
    "10wdir":   {"role": "excluded_by_default",        "circular": True,  "recommended": False},
    "orog":     {"role": "excluded_by_default",        "circular": False, "recommended": False},
}

# GRIB shortName → CF-like standard name (best-effort; not exhaustive)
_RTMA_VAR_MAP: dict[str, str] = {
    "2t":      "air_temperature_2m",
    "t2m":     "air_temperature_2m",
    "2d":      "dewpoint_temperature_2m",
    "d2m":     "dewpoint_temperature_2m",
    "2sh":     "specific_humidity_2m",
    "sh2":     "specific_humidity_2m",
    "q":       "specific_humidity_2m",
    "10u":     "wind_u_component_10m",
    "u10":     "wind_u_component_10m",
    "ugrd":    "wind_u_component_10m",
    "10v":     "wind_v_component_10m",
    "v10":     "wind_v_component_10m",
    "vgrd":    "wind_v_component_10m",
    "10si":    "wind_speed_10m",
    "i10fg":   "wind_gust_10m",
    "10wdir":  "wind_direction_10m",
    "sp":      "surface_pressure",
    "pres":    "surface_pressure",
    "pressfc": "surface_pressure",
    "tcc":     "total_cloud_cover",
    "tcdc":    "total_cloud_cover",
    "vis":     "visibility",
    "ceil":    "cloud_ceiling",
    "orog":    "orography",
}

# Canonical column order for all output DataFrames
STAT_COLUMNS: list[str] = [
    "STAID", "product", "source", "variable", "variable_standard_name",
    "variable_role", "circular_variable_flag", "recommended_for_initial_model",
    "valid_time_utc", "issue_time_utc", "lead_time_hours", "units",
    "weighted_mean",
    "unweighted_min", "unweighted_max", "unweighted_std",
    "unweighted_q10", "unweighted_q25", "unweighted_q50",
    "unweighted_q75", "unweighted_q90", "unweighted_q95", "unweighted_q99",
    "valid_cell_count", "total_weight", "valid_weight_fraction", "missing_value_fraction",
    "weight_table_path", "source_file_path",
]


@dataclass
class VariableGrid:
    """One decoded 2-D grid variable ready for basin extraction."""
    short_name: str      # GRIB shortName (e.g. "2t", "unknown")
    standard_name: str   # CF-like standard name
    grib_name: str       # GRIB_name metadata value
    units: str           # GRIB_units metadata value
    values: np.ndarray   # 2-D float64 array, shape (nrows, ncols)
    valid_time_utc: str  # ISO 8601 string (UTC)
    product: str         # product identifier
    source: str          # data source label
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _suppress_cfgrib_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"In a future version of xarray the default value for compat.*",
        category=FutureWarning,
        module=r"cfgrib\.xarray_store",
    )


def _format_valid_time(ts: Any) -> str:
    """Normalise a cfgrib/numpy datetime64 value to a clean ISO 8601 UTC string."""
    s = str(ts).split(".")[0]           # strip sub-second precision
    if "T" in s and not s.endswith("Z"):
        s += "Z"
    return s


def _pick_valid_time(da) -> str:
    """Extract valid_time from a cfgrib DataArray's coordinates."""
    for name in ("valid_time", "time"):
        coord = da.coords.get(name)
        if coord is not None:
            raw = np.asarray(coord.values).reshape(-1)[0]
            return _format_valid_time(raw)
    return "unknown"


# ---------------------------------------------------------------------------
# Grid decoders
# ---------------------------------------------------------------------------

def decode_mrms_grid(path: Path) -> VariableGrid:
    """Decode one MRMS GRIB2 or GRIB2.gz file and return a VariableGrid.

    Handles gzip decompression transparently via a TemporaryDirectory.
    """
    import xarray as xr

    path = Path(path)
    work_path = path
    _tmp: Optional[tempfile.TemporaryDirectory] = None
    var_warnings: list[str] = []

    if path.suffix.lower() == ".gz":
        _tmp = tempfile.TemporaryDirectory()
        work_path = Path(_tmp.name) / path.stem
        with gzip.open(path, "rb") as src, work_path.open("wb") as dst:
            dst.write(src.read())
        LOGGER.debug("MRMS: decompressed %s → %s", path.name, work_path.name)

    try:
        _suppress_cfgrib_warnings()
        ds = xr.open_dataset(work_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        try:
            var_name  = next(iter(ds.data_vars))
            da        = ds[var_name]
            arr       = np.squeeze(np.asarray(da.values, dtype=np.float64))
            units     = str(da.attrs.get("GRIB_units",     "unknown"))
            grib_name = str(da.attrs.get("GRIB_name",      "unknown"))
            short_name= str(da.attrs.get("GRIB_shortName", "unknown"))
            valid_time = _pick_valid_time(da)

            if arr.ndim != 2:
                var_warnings.append(f"unexpected ndim={arr.ndim}; expected 2")
            if units == "unknown":
                var_warnings.append(_MRMS_UNITS_CAVEAT)
        finally:
            ds.close()
    finally:
        if _tmp is not None:
            _tmp.cleanup()

    return VariableGrid(
        short_name=short_name,
        standard_name=_MRMS_STANDARD_NAME,
        grib_name=grib_name,
        units=units,
        values=arr,
        valid_time_utc=valid_time,
        product=_MRMS_PRODUCT,
        source=_MRMS_SOURCE,
        warnings=var_warnings,
    )


def decode_rtma_grids(path: Path) -> list[VariableGrid]:
    """Decode all 2-D variables from an RTMA GRIB2 file.

    Uses cfgrib.open_datasets() to handle multi-message GRIB2 files.
    Variables that fail to decode are logged and skipped.
    Returns one VariableGrid per successfully decoded 2-D variable.
    Deduplicates by GRIB shortName (first occurrence kept).
    """
    import cfgrib

    path = Path(path)
    _suppress_cfgrib_warnings()
    grids: list[VariableGrid] = []
    seen: set[str] = set()

    datasets = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
    try:
        for ds in datasets:
            for var_name in list(ds.data_vars):
                try:
                    da         = ds[var_name]
                    arr        = np.squeeze(np.asarray(da.values, dtype=np.float64))
                    if arr.ndim != 2:
                        LOGGER.debug("RTMA var %s: ndim=%d, skipping", var_name, arr.ndim)
                        continue

                    short_name = str(da.attrs.get("GRIB_shortName", var_name)).lower()
                    if short_name in seen:
                        LOGGER.debug("RTMA: duplicate short_name=%s, skipping", short_name)
                        continue
                    seen.add(short_name)

                    grib_name  = str(da.attrs.get("GRIB_name",  var_name))
                    units      = str(da.attrs.get("GRIB_units", "unknown"))
                    std_name   = _RTMA_VAR_MAP.get(short_name, short_name)
                    valid_time = _pick_valid_time(da)

                    grids.append(VariableGrid(
                        short_name=short_name,
                        standard_name=std_name,
                        grib_name=grib_name,
                        units=units,
                        values=arr,
                        valid_time_utc=valid_time,
                        product=_RTMA_PRODUCT,
                        source=_RTMA_SOURCE,
                    ))
                    LOGGER.debug(
                        "RTMA decoded: %s (%s) shape=%s units=%s",
                        short_name, grib_name, arr.shape, units,
                    )
                except Exception as exc:
                    LOGGER.warning("RTMA: failed to decode var %s: %s", var_name, exc)
    finally:
        for ds in datasets:
            try:
                ds.close()
            except Exception:
                pass

    return grids


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def _compute_basin_stats(
    cell_values: np.ndarray,
    norm_weights: np.ndarray,
) -> dict[str, Any]:
    """Compute all required statistics for one basin's set of grid cells.

    Parameters
    ----------
    cell_values:  1-D float64 array of raw grid values for the basin's cells.
    norm_weights: 1-D float64 array of normalised weights (sum should be ~1.0).

    NaN/inf values in cell_values are treated as missing and excluded from
    the weighted mean and percentiles. Their weight contribution is tracked via
    valid_weight_fraction.
    """
    n_total     = int(len(cell_values))
    total_weight = float(np.nansum(norm_weights))
    valid_mask  = np.isfinite(cell_values)
    n_valid     = int(np.sum(valid_mask))

    if n_valid == 0:
        return {
            "weighted_mean":        float("nan"),
            "unweighted_min":       float("nan"),
            "unweighted_max":       float("nan"),
            "unweighted_std":       float("nan"),
            "unweighted_q10":       float("nan"),
            "unweighted_q25":       float("nan"),
            "unweighted_q50":       float("nan"),
            "unweighted_q75":       float("nan"),
            "unweighted_q90":       float("nan"),
            "unweighted_q95":       float("nan"),
            "unweighted_q99":       float("nan"),
            "valid_cell_count":     0,
            "total_weight":         total_weight,
            "valid_weight_fraction": 0.0,
            "missing_value_fraction": 1.0,
        }

    valid_vals   = cell_values[valid_mask]
    valid_w      = norm_weights[valid_mask]
    valid_w_sum  = float(np.nansum(valid_w))

    weighted_mean = (
        float(np.sum(valid_w * valid_vals) / valid_w_sum)
        if valid_w_sum > 0
        else float(np.mean(valid_vals))
    )

    return {
        "weighted_mean":        weighted_mean,
        "unweighted_min":       float(np.min(valid_vals)),
        "unweighted_max":       float(np.max(valid_vals)),
        "unweighted_std":       float(np.std(valid_vals)),
        "unweighted_q10":       float(np.percentile(valid_vals, 10)),
        "unweighted_q25":       float(np.percentile(valid_vals, 25)),
        "unweighted_q50":       float(np.percentile(valid_vals, 50)),
        "unweighted_q75":       float(np.percentile(valid_vals, 75)),
        "unweighted_q90":       float(np.percentile(valid_vals, 90)),
        "unweighted_q95":       float(np.percentile(valid_vals, 95)),
        "unweighted_q99":       float(np.percentile(valid_vals, 99)),
        "valid_cell_count":     n_valid,
        "total_weight":         total_weight,
        "valid_weight_fraction": valid_w_sum,
        "missing_value_fraction": (
            float(n_total - n_valid) / n_total if n_total > 0 else float("nan")
        ),
    }


# ---------------------------------------------------------------------------
# Basin extraction
# ---------------------------------------------------------------------------

def extract_basin_statistics(
    var_grid: VariableGrid,
    weights_df: pd.DataFrame,
    pilot_staids: list[str],
    *,
    weight_table_path: str,
    source_file_path: str,
) -> pd.DataFrame:
    """Extract basin-level statistics for all pilot basins for one VariableGrid.

    pilot_staids must use the same zero-padded 8-digit format as weights_df['STAID'].
    Basins absent from weights_df are logged and omitted from the output.

    Returns a DataFrame with columns defined in STAT_COLUMNS.
    """
    grid         = var_grid.values
    nrows, ncols = grid.shape
    records: list[dict] = []
    missing: list[str]  = []

    for staid in pilot_staids:
        basin_wts = weights_df.loc[weights_df["STAID"] == staid]
        if basin_wts.empty:
            missing.append(staid)
            LOGGER.warning(
                "No weight rows for STAID=%s product=%s — omitting basin",
                staid, var_grid.product,
            )
            continue

        rows   = basin_wts["row_idx"].values
        cols   = basin_wts["col_idx"].values
        norm_w = basin_wts["normalized_weight"].values.astype(np.float64)

        # Safety: drop cells outside grid bounds
        in_bounds = (rows >= 0) & (rows < nrows) & (cols >= 0) & (cols < ncols)
        if not np.all(in_bounds):
            n_oob = int(np.sum(~in_bounds))
            LOGGER.warning(
                "STAID=%s product=%s: %d/%d cells outside grid bounds — dropping",
                staid, var_grid.product, n_oob, len(rows),
            )
            rows   = rows[in_bounds]
            cols   = cols[in_bounds]
            norm_w = norm_w[in_bounds]

        if len(rows) == 0:
            LOGGER.warning(
                "STAID=%s product=%s: no cells after bounds check — omitting",
                staid, var_grid.product,
            )
            continue

        cell_values = grid[rows, cols]
        stats       = _compute_basin_stats(cell_values, norm_w)

        meta = _VAR_METADATA.get(
            var_grid.short_name,
            {"role": "unknown", "circular": False, "recommended": False},
        )
        records.append({
            "STAID":                         staid,
            "product":                       var_grid.product,
            "source":                        var_grid.source,
            "variable":                      var_grid.short_name,
            "variable_standard_name":        var_grid.standard_name,
            "variable_role":                 meta["role"],
            "circular_variable_flag":        meta["circular"],
            "recommended_for_initial_model": meta["recommended"],
            "valid_time_utc":                var_grid.valid_time_utc,
            "issue_time_utc":                None,
            "lead_time_hours":               None,
            "units":                         var_grid.units,
            "weight_table_path":             weight_table_path,
            "source_file_path":              source_file_path,
            **stats,
        })

    if missing:
        LOGGER.warning(
            "extract_basin_statistics: %d/%d basins had no weight rows (product=%s): %s",
            len(missing), len(pilot_staids), var_grid.product, missing,
        )

    if not records:
        return pd.DataFrame(columns=STAT_COLUMNS)

    df = pd.DataFrame(records)
    # Enforce canonical column order
    ordered = [c for c in STAT_COLUMNS if c in df.columns]
    extra   = [c for c in df.columns if c not in STAT_COLUMNS]
    return df[ordered + extra]


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_one_hour_extraction(
    mrms_path: Path,
    rtma_path: Path,
    mrms_weights_path: Path,
    rtma_weights_path: Path,
    pilot_staids: list[str],
    *,
    products: Optional[list[str]] = None,
    include_excluded_vars: bool = False,
) -> dict[str, Any]:
    """Decode one hour of MRMS and/or RTMA data and extract basin statistics.

    Parameters
    ----------
    mrms_path, rtma_path:
        Paths to sample GRIB2 files (MRMS may be .gz).
    mrms_weights_path, rtma_weights_path:
        Paths to the Parquet weight tables produced by Milestone 2B.
    pilot_staids:
        List of 8-digit zero-padded USGS STAIDs matching the weight tables.
    products:
        Products to extract; default = both MRMS and RTMA.
    include_excluded_vars:
        If True, include variables in _RTMA_EXCLUDED_DEFAULT (10wdir, orog)
        in the RTMA output. Default False — these are excluded because 10wdir
        is a circular variable (linear averaging is invalid) and orog is a
        static terrain field (not a dynamic meteorological forcing).

    Returns
    -------
    dict with keys:
        mrms_df           – pd.DataFrame with MRMS basin statistics
        rtma_df           – pd.DataFrame with RTMA basin statistics
        rtma_grids_all    – list[VariableGrid] for all decoded RTMA variables
        rtma_grids        – list[VariableGrid] for included (non-excluded) variables
        rtma_excluded     – list[VariableGrid] for excluded variables
        combined_df       – pd.DataFrame with both products concatenated
        mrms_grid         – VariableGrid (or None if MRMS not requested)
        warnings          – list[str] of non-fatal warnings
    """
    if products is None:
        products = [_MRMS_PRODUCT, _RTMA_PRODUCT]

    run_warnings: list[str] = []
    mrms_grid: Optional[VariableGrid]  = None
    rtma_grids_all: list[VariableGrid] = []
    rtma_grids: list[VariableGrid]     = []
    rtma_excluded: list[VariableGrid]  = []
    mrms_df = pd.DataFrame(columns=STAT_COLUMNS)
    rtma_df = pd.DataFrame(columns=STAT_COLUMNS)

    # ---- MRMS ----------------------------------------------------------------
    if _MRMS_PRODUCT in products:
        LOGGER.info("Loading MRMS weights from %s", mrms_weights_path)
        mrms_weights = pd.read_parquet(mrms_weights_path)

        LOGGER.info("Decoding MRMS grid from %s", mrms_path)
        mrms_grid = decode_mrms_grid(mrms_path)
        run_warnings.extend(mrms_grid.warnings)

        LOGGER.info(
            "MRMS decoded: shape=%s  variable=%s  units=%s  valid_time=%s",
            mrms_grid.values.shape, mrms_grid.short_name,
            mrms_grid.units, mrms_grid.valid_time_utc,
        )

        mrms_df = extract_basin_statistics(
            mrms_grid, mrms_weights, pilot_staids,
            weight_table_path=str(mrms_weights_path),
            source_file_path=str(mrms_path),
        )
        LOGGER.info("MRMS extraction complete: %d rows", len(mrms_df))

    # ---- RTMA ----------------------------------------------------------------
    if _RTMA_PRODUCT in products:
        LOGGER.info("Loading RTMA weights from %s", rtma_weights_path)
        rtma_weights = pd.read_parquet(rtma_weights_path)

        LOGGER.info("Decoding RTMA grids from %s", rtma_path)
        rtma_grids_all = decode_rtma_grids(rtma_path)

        if not rtma_grids_all:
            run_warnings.append("RTMA: no variables decoded from file")
            LOGGER.warning("RTMA: no variables decoded from %s", rtma_path)
        else:
            LOGGER.info(
                "RTMA decoded %d variables: %s",
                len(rtma_grids_all), [g.short_name for g in rtma_grids_all],
            )

        # Apply exclusion filter
        for vg in rtma_grids_all:
            if not include_excluded_vars and vg.short_name in _RTMA_EXCLUDED_DEFAULT:
                rtma_excluded.append(vg)
            else:
                rtma_grids.append(vg)

        if rtma_excluded:
            LOGGER.info(
                "RTMA excluded from default output (%d): %s",
                len(rtma_excluded), [g.short_name for g in rtma_excluded],
            )

        rtma_frames: list[pd.DataFrame] = []
        for vg in rtma_grids:
            df_v = extract_basin_statistics(
                vg, rtma_weights, pilot_staids,
                weight_table_path=str(rtma_weights_path),
                source_file_path=str(rtma_path),
            )
            rtma_frames.append(df_v)
            LOGGER.info(
                "  RTMA variable %-10s: %d rows", vg.short_name, len(df_v)
            )

        rtma_df = (
            pd.concat(rtma_frames, ignore_index=True)
            if rtma_frames
            else pd.DataFrame(columns=STAT_COLUMNS)
        )
        LOGGER.info(
            "RTMA extraction complete: %d rows (%d basins × %d variables, %d excluded)",
            len(rtma_df), len(pilot_staids), len(rtma_grids), len(rtma_excluded),
        )

    combined_df = pd.concat([mrms_df, rtma_df], ignore_index=True)

    return {
        "mrms_df":        mrms_df,
        "rtma_df":        rtma_df,
        "combined_df":    combined_df,
        "mrms_grid":      mrms_grid,
        "rtma_grids_all": rtma_grids_all,
        "rtma_grids":     rtma_grids,
        "rtma_excluded":  rtma_excluded,
        "warnings":       run_warnings,
    }

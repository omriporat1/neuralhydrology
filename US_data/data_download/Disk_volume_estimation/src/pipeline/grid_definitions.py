"""Grid-definition discovery for MRMS and RTMA source products.

Responsible for:
- Locating or downloading one sample file per product (reuse cached copy if present)
- Decoding the GRIB2 / compressed GRIB2 file with cfgrib/xarray
- Extracting coordinate, projection, and grid metadata into a JSON-serialisable dict
- Writing lightweight JSON summaries
- Generating small PNG preview plots with geo-referenced axes where possible

Ground-truth grid properties (confirmed from real files):

  MRMS QPE 1h Pass1:
    gridType  = regular_ll (Equidistant Cylindrical)
    shape     = (3500, 7000)   [rows=lat, cols=lon]
    lat       = 1D, 54.995 -> 20.005  (north to south, lat_descending=True)
    lon       = 1D, -129.995 -> -60.005  (west to east)
    dx = dy   = 0.01 deg  (~1 km)
    GRIB units/name = 'unknown' (MRMS-specific encoding quirk)
    valid_time = period-end timestamp convention

  RTMA 2.5km NDFD analysis:
    gridType  = lambert (Lambert Conformal Conic)
    shape     = (1597, 2345)   [rows=y, cols=x]
    lat       = 2D (1597,2345), 19.23 -> 57.09  (south to north, lat_descending=False)
    lon       = 2D (1597,2345), 221.6 -> 301.0  (needs -360 normalisation)
    dx = dy   = 2539.703 m  (~2.5 km)
    LaDInDeg  = 25.0, LoVInDeg = 265.0 (central meridian)
    jScansPositively = 1  (row 0 = southern edge)

Prerequisites for next milestone:
  These grid definitions are required before computing basin-grid overlap weights.
  Weight computation needs the precise coordinate arrays, not just bounding-box
  approximations, so the weight scripts should re-decode the same sample files
  or use the grid metadata stored here.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)

SAMPLE_TIME_DEFAULT = "2023-01-01T00:00:00"

_CONUS_LON_MIN, _CONUS_LAT_MIN, _CONUS_LON_MAX, _CONUS_LAT_MAX = -126.0, 24.0, -66.0, 50.0

# GRIB attribute keys worth extracting for grid documentation
_GRIB_GRID_KEYS = [
    "GRIB_gridType",
    "GRIB_gridDefinitionDescription",
    "GRIB_Ni", "GRIB_Nj",
    "GRIB_iDirectionIncrementInDegrees",
    "GRIB_jDirectionIncrementInDegrees",
    "GRIB_DxInMetres", "GRIB_DyInMetres",
    "GRIB_latitudeOfFirstGridPointInDegrees",
    "GRIB_longitudeOfFirstGridPointInDegrees",
    "GRIB_latitudeOfLastGridPointInDegrees",
    "GRIB_longitudeOfLastGridPointInDegrees",
    "GRIB_jScansPositively",
    "GRIB_iScansPositively",
    "GRIB_scanningMode",
    "GRIB_LaDInDegrees",
    "GRIB_LoVInDegrees",
    "GRIB_Latin1InDegrees",
    "GRIB_Latin2InDegrees",
    "GRIB_name",
    "GRIB_shortName",
    "GRIB_units",
    "GRIB_typeOfLevel",
]


# ---------------------------------------------------------------------------
# Small private utilities
# ---------------------------------------------------------------------------

def _suppress_cfgrib_futurewarning() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"In a future version of xarray the default value for compat.*",
        category=FutureWarning,
        module=r"cfgrib\.xarray_store",
    )


def _qc_stats(arr: np.ndarray) -> dict[str, float]:
    arrf = arr.astype("float64", copy=False)
    nan_pct = float(np.isnan(arrf).sum() / arrf.size * 100.0) if arrf.size else 100.0
    return {
        "min": float(np.nanmin(arrf)),
        "max": float(np.nanmax(arrf)),
        "mean": float(np.nanmean(arrf)),
        "nan_pct": nan_pct,
    }


def _conus_overlap(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> bool:
    return not (
        lon_max < _CONUS_LON_MIN or lon_min > _CONUS_LON_MAX
        or lat_max < _CONUS_LAT_MIN or lat_min > _CONUS_LAT_MAX
    )


def _pull_grib_attrs(attrs: dict) -> dict:
    """Extract and JSON-serialisable-ify relevant GRIB attributes."""
    out: dict = {}
    for k in _GRIB_GRID_KEYS:
        v = attrs.get(k)
        if v is None:
            continue
        # Normalise longitudes stored as east-of-greenwich (0-360) to [-180,180]
        if "longitudeOf" in k and isinstance(v, (int, float)) and v > 180:
            v = float(v) - 360.0
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        out[k] = v
    return out


def _extract_coord_info(da) -> tuple[Optional[dict[str, float]], Optional[bool]]:
    """Return (bbox_dict, lat_descending) from a cfgrib DataArray's coordinates.

    Handles both 1-D coords (MRMS regular_ll) and 2-D coords (RTMA lambert).
    Returns (None, None) if no lat/lon coordinates are found.
    """
    coords = da.coords
    lat_name = next((n for n in coords if n.lower() in ("latitude", "lat")), None)
    lon_name = next((n for n in coords if n.lower() in ("longitude", "lon")), None)
    if lat_name is None or lon_name is None:
        return None, None

    lat_vals = np.asarray(coords[lat_name].values)
    lon_vals = np.asarray(coords[lon_name].values)

    if lat_vals.size == 0 or lon_vals.size == 0:
        return None, None

    lon_norm = np.where(lon_vals > 180.0, lon_vals - 360.0, lon_vals)

    bbox: dict[str, float] = {
        "lon_min": float(np.nanmin(lon_norm)),
        "lon_max": float(np.nanmax(lon_norm)),
        "lat_min": float(np.nanmin(lat_vals)),
        "lat_max": float(np.nanmax(lat_vals)),
    }

    # Determine scan direction: does latitude decrease from row 0 to row N?
    if lat_vals.ndim == 2:
        # Lambert and other projected grids: compare first vs last row at mid column
        mid = lat_vals.shape[1] // 2
        lat_descending: Optional[bool] = bool(lat_vals[0, mid] > lat_vals[-1, mid])
    elif lat_vals.ndim == 1:
        lat_descending = bool(lat_vals[0] > lat_vals[-1])
    else:
        lat_descending = None

    return bbox, lat_descending


# ---------------------------------------------------------------------------
# Grid discovery: MRMS
# ---------------------------------------------------------------------------

def discover_mrms_grid(
    sample_file: Path,
    *,
    source_name: str = "mrms_qpe_1h_pass1",
) -> tuple[dict[str, Any], Optional[np.ndarray]]:
    """Decode one MRMS GRIB2 (.gz) file and extract grid metadata + preview array.

    Returns:
        (meta_dict, preview_array_or_None)
        meta_dict is JSON-serialisable.
    """
    import xarray as xr  # optional dep; import inside function

    work_path = sample_file
    _tmp: Optional[tempfile.TemporaryDirectory] = None

    if sample_file.suffix.lower() == ".gz":
        _tmp = tempfile.TemporaryDirectory()
        work_path = Path(_tmp.name) / sample_file.stem
        with gzip.open(sample_file, "rb") as src, work_path.open("wb") as dst:
            dst.write(src.read())

    try:
        _suppress_cfgrib_futurewarning()
        ds = xr.open_dataset(
            work_path, engine="cfgrib", backend_kwargs={"indexpath": ""}
        )
        try:
            var_name = next(iter(ds.data_vars))
            da = ds[var_name]
            arr = np.squeeze(np.asarray(da.values))

            bbox, lat_descending = _extract_coord_info(da)
            grib_attrs = _pull_grib_attrs(da.attrs)

            # Fallback bbox from GRIB first/last point
            if bbox is None:
                lf = grib_attrs.get("GRIB_latitudeOfFirstGridPointInDegrees")
                nf = grib_attrs.get("GRIB_longitudeOfFirstGridPointInDegrees")
                ll = grib_attrs.get("GRIB_latitudeOfLastGridPointInDegrees")
                nl = grib_attrs.get("GRIB_longitudeOfLastGridPointInDegrees")
                if all(v is not None for v in [lf, nf, ll, nl]):
                    bbox = {
                        "lon_min": min(nf, nl), "lon_max": max(nf, nl),
                        "lat_min": min(lf, ll), "lat_max": max(lf, ll),
                    }
                    lat_descending = lf > ll

            # Valid time
            time_str = "unknown"
            for tname in ("valid_time", "time"):
                tc = da.coords.get(tname)
                if tc is not None:
                    time_str = str(np.asarray(tc.values).reshape(-1)[0])
                    break

            coord_names = list(da.coords)
            coord_shapes = {n: list(da.coords[n].shape) for n in coord_names}

            dx = grib_attrs.get("GRIB_iDirectionIncrementInDegrees")
            dy = grib_attrs.get("GRIB_jDirectionIncrementInDegrees")
            approx_res = dx or dy

            conus_ok: Optional[bool] = None
            if bbox:
                conus_ok = _conus_overlap(
                    bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"]
                )

            notes: list[str] = []
            grid_type = grib_attrs.get("GRIB_gridType", "unknown")
            if grid_type == "regular_ll":
                notes.append("regular_ll grid (Equidistant Cylindrical / Plate Carree, ~1km CONUS)")
            if lat_descending is True:
                notes.append("lat top-to-bottom (north-up array); imshow origin='upper'")
            elif lat_descending is False:
                notes.append("lat bottom-to-top (south-up array); imshow origin='lower'")
            if grib_attrs.get("GRIB_name") == "unknown":
                notes.append("GRIB_name='unknown': normal for MRMS QPE GRIB2 encoding")
            if grib_attrs.get("GRIB_jScansPositively") == 0:
                notes.append("jScansPositively=0: row 0 is northernmost row (standard for MRMS)")
            if conus_ok:
                notes.append("bounding box overlaps CONUS")
            elif conus_ok is False:
                notes.append("WARNING: bounding box does NOT overlap CONUS")

            warnings_list: list[str] = []
            if arr.ndim != 2:
                warnings_list.append(f"unexpected array ndim={arr.ndim}; expected 2")
            if arr.size == 0:
                warnings_list.append("decoded array is empty")

            meta: dict[str, Any] = {
                "product": source_name,
                "source_file": str(sample_file),
                "source_file_size_bytes": sample_file.stat().st_size,
                "sample_valid_time": time_str,
                "variable_names_decoded": [var_name],
                "preview_variable": var_name,
                "grib_name": grib_attrs.get("GRIB_name", var_name),
                "units": grib_attrs.get("GRIB_units", "unknown"),
                "grid_type": grid_type,
                "grid_shape_rows_cols": list(arr.shape) if arr.ndim == 2 else None,
                "coord_names": coord_names,
                "coord_shapes": coord_shapes,
                "bbox_lon_min": bbox["lon_min"] if bbox else None,
                "bbox_lat_min": bbox["lat_min"] if bbox else None,
                "bbox_lon_max": bbox["lon_max"] if bbox else None,
                "bbox_lat_max": bbox["lat_max"] if bbox else None,
                "lat_descending": lat_descending,
                "approx_resolution_deg": approx_res,
                "approx_resolution_m": None,
                "conus_overlap": conus_ok,
                "grib_attrs": grib_attrs,
                "qc_stats": _qc_stats(arr) if arr.ndim == 2 and arr.size > 0 else None,
                "notes": notes,
                "warnings": warnings_list,
            }

            preview_arr = arr if arr.ndim == 2 and arr.size > 0 else None
            return meta, preview_arr

        finally:
            ds.close()

    except Exception as exc:
        LOGGER.error("MRMS grid discovery failed for %s: %s", sample_file, exc)
        meta = {
            "product": source_name,
            "source_file": str(sample_file),
            "error": str(exc),
            "notes": [],
            "warnings": [f"discovery failed: {exc}"],
        }
        return meta, None

    finally:
        if _tmp is not None:
            _tmp.cleanup()


# ---------------------------------------------------------------------------
# Grid discovery: RTMA
# ---------------------------------------------------------------------------

def discover_rtma_grid(
    sample_file: Path,
    *,
    source_name: str = "rtma_conus_aws_2p5km",
) -> tuple[dict[str, Any], Optional[np.ndarray]]:
    """Decode one RTMA GRIB2 file and extract grid metadata + preview array.

    RTMA files contain multiple GRIB messages → cfgrib.open_datasets() is required.
    Returns:
        (meta_dict, preview_array_or_None)
    """
    import cfgrib  # optional dep; import inside function

    _suppress_cfgrib_futurewarning()

    grid_shape: Optional[list[int]] = None
    coord_names: list[str] = []
    coord_shapes: dict[str, list[int]] = {}
    bbox: Optional[dict[str, float]] = None
    lat_descending: Optional[bool] = None
    vars_decoded: list[str] = []
    timestamp_str = "unknown"
    grib_attrs_sample: dict = {}
    preview_arr: Optional[np.ndarray] = None
    preview_var: Optional[str] = None

    datasets = cfgrib.open_datasets(str(sample_file), backend_kwargs={"indexpath": ""})
    try:
        for ds in datasets:
            for var_name in ds.data_vars:
                da = ds[var_name]
                arr = np.squeeze(np.asarray(da.values))
                if arr.ndim != 2:
                    continue

                short_name = str(da.attrs.get("GRIB_shortName", "")).lower()
                vars_decoded.append(short_name or var_name)

                if grid_shape is None:
                    grid_shape = list(arr.shape)
                    coord_names = list(da.coords)
                    coord_shapes = {n: list(da.coords[n].shape) for n in coord_names}
                    grib_attrs_sample = _pull_grib_attrs(da.attrs)
                    bbox, lat_descending = _extract_coord_info(da)
                    if bbox is None:
                        lf = grib_attrs_sample.get("GRIB_latitudeOfFirstGridPointInDegrees")
                        nf = grib_attrs_sample.get("GRIB_longitudeOfFirstGridPointInDegrees")
                        ll = grib_attrs_sample.get("GRIB_latitudeOfLastGridPointInDegrees")
                        nl = grib_attrs_sample.get("GRIB_longitudeOfLastGridPointInDegrees")
                        if all(v is not None for v in [lf, nf, ll, nl]):
                            bbox = {
                                "lon_min": min(nf, nl), "lon_max": max(nf, nl),
                                "lat_min": min(lf, ll), "lat_max": max(lf, ll),
                            }
                            lat_descending = lf > ll

                # Prefer TMP (2t) for preview
                if short_name in {"2t", "t2m"} and preview_arr is None:
                    preview_arr = arr
                    preview_var = "TMP (GRIB:2t)"
                elif preview_arr is None:
                    preview_arr = arr
                    preview_var = short_name or var_name

                if timestamp_str == "unknown":
                    for tname in ("valid_time", "time"):
                        tc = da.coords.get(tname)
                        if tc is not None:
                            timestamp_str = str(np.asarray(tc.values).reshape(-1)[0])
                            break
    finally:
        for ds in datasets:
            ds.close()

    dx_m = grib_attrs_sample.get("GRIB_DxInMetres")
    dy_m = grib_attrs_sample.get("GRIB_DyInMetres")
    dx_deg = grib_attrs_sample.get("GRIB_iDirectionIncrementInDegrees")
    approx_res_deg = dx_deg
    approx_res_m = dx_m or dy_m

    conus_ok: Optional[bool] = None
    if bbox:
        conus_ok = _conus_overlap(
            bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"]
        )

    notes: list[str] = []
    grid_type = grib_attrs_sample.get("GRIB_gridType", "unknown")
    if "lambert" in str(grid_type).lower():
        lad = grib_attrs_sample.get("GRIB_LaDInDegrees", "?")
        lov = grib_attrs_sample.get("GRIB_LoVInDegrees", "?")
        notes.append(f"Lambert Conformal Conic (NDFD 2.5km); LaD={lad}, LoV={lov}")
        notes.append("2D lat/lon coord arrays provided by cfgrib for this projection")
    j_scans = grib_attrs_sample.get("GRIB_jScansPositively")
    if j_scans == 1:
        notes.append("jScansPositively=1: row 0 is southernmost row; lat_descending=False")
    if lat_descending is False:
        notes.append("lat bottom-to-top (south-up array); imshow origin='lower'")
    elif lat_descending is True:
        notes.append("lat top-to-bottom (north-up array); imshow origin='upper'")
    if conus_ok:
        notes.append("bounding box overlaps CONUS")
    elif conus_ok is False:
        notes.append("WARNING: bounding box does NOT overlap CONUS")

    warnings_list: list[str] = []
    if grid_shape is None:
        warnings_list.append("no 2D arrays decoded from file")

    meta: dict[str, Any] = {
        "product": source_name,
        "source_file": str(sample_file),
        "source_file_size_bytes": sample_file.stat().st_size,
        "sample_valid_time": timestamp_str,
        "variable_names_decoded": list(dict.fromkeys(vars_decoded)),
        "preview_variable": preview_var,
        "grid_type": grid_type,
        "grid_shape_rows_cols": grid_shape,
        "coord_names": coord_names,
        "coord_shapes": coord_shapes,
        "bbox_lon_min": bbox["lon_min"] if bbox else None,
        "bbox_lat_min": bbox["lat_min"] if bbox else None,
        "bbox_lon_max": bbox["lon_max"] if bbox else None,
        "bbox_lat_max": bbox["lat_max"] if bbox else None,
        "lat_descending": lat_descending,
        "approx_resolution_deg": approx_res_deg,
        "approx_resolution_m": float(approx_res_m) if approx_res_m is not None else None,
        "conus_overlap": conus_ok,
        "grib_attrs": grib_attrs_sample,
        "qc_stats": _qc_stats(preview_arr) if preview_arr is not None else None,
        "notes": notes,
        "warnings": warnings_list,
    }

    return meta, preview_arr


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_grid_definition_json(meta: dict[str, Any], out_path: Path) -> None:
    """Write a grid-definition dict as an indented JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)


def plot_grid_preview(
    arr: np.ndarray,
    out_path: Path,
    *,
    title: str,
    var_label: str,
    bbox: Optional[dict[str, float]] = None,
    lat_descending: Optional[bool] = None,
    cmap: str = "Blues",
    dpi: int = 120,
) -> bool:
    """Save a lightweight PNG preview of a 2D grid field.

    Uses geo-referenced axes (extent in degrees) when bbox is available.
    Falls back to grid-index axes otherwise.

    Returns True on success, False if matplotlib is unavailable or the plot fails.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as exc:
        LOGGER.warning("plot_grid_preview: matplotlib not available: %s", exc)
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        if bbox is not None and lat_descending is not None:
            lon_min = bbox["lon_min"]
            lon_max = bbox["lon_max"]
            lat_min = bbox["lat_min"]
            lat_max = bbox["lat_max"]
            origin = "upper" if lat_descending else "lower"
            im = ax.imshow(
                arr,
                origin=origin,
                extent=[lon_min, lon_max, lat_min, lat_max],
                aspect="auto",
                cmap=cmap,
            )
            # Draw CONUS reference bounding box
            ax.add_patch(Rectangle(
                (-126, 24), 60, 26,
                linewidth=1.5, edgecolor="red", facecolor="none",
                linestyle="--", label="CONUS bbox",
            ))
            ax.legend(loc="lower right", fontsize=8)
            ax.set_xlabel("Longitude (deg)")
            ax.set_ylabel("Latitude (deg)")
            bounds_label = (
                f"lon [{lon_min:.2f}, {lon_max:.2f}] "
                f"lat [{lat_min:.2f}, {lat_max:.2f}]"
            )
            ax.text(
                0.02, 0.98, bounds_label,
                transform=ax.transAxes, fontsize=7,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
            )
        else:
            im = ax.imshow(arr, origin="upper", cmap=cmap, aspect="auto")
            ax.set_xlabel("grid column index")
            ax.set_ylabel("grid row index")
            ax.text(
                0.02, 0.98, "no geo-referenced coordinates available",
                transform=ax.transAxes, fontsize=7,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
            )

        fig.colorbar(im, ax=ax, label=var_label, fraction=0.03)
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        return True

    except Exception as exc:
        LOGGER.warning("plot_grid_preview failed for %s: %s", out_path.name, exc)
        try:
            plt.close("all")
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# Sample file management
# ---------------------------------------------------------------------------

def _mrms_timestamp_from_path(p: Path) -> Optional[datetime]:
    """Parse MRMS timestamp from a filename like MRMS_..._YYYYMMDD-HHMMSS.grib2.gz"""
    m = re.search(r"(\d{8})-(\d{6})", p.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _rtma_timestamp_from_path(p: Path) -> Optional[datetime]:
    """Parse RTMA timestamp from rtma2p5.tHHz.2dvaranl_ndfd.grb2_wexp in rtma2p5.YYYYMMDD parent."""
    hour_m = re.search(r"\.t(\d{2})z\.2dvaranl_ndfd\.grb2_wexp$", p.name)
    day_m = re.search(r"rtma2p5\.(\d{8})", p.parent.name)
    if not hour_m or not day_m:
        return None
    try:
        return datetime.strptime(f"{day_m.group(1)}{hour_m.group(1)}", "%Y%m%d%H")
    except ValueError:
        return None


def find_cached_mrms_sample(raw_dir: Path, sample_dt: datetime) -> Optional[Path]:
    """Return path to a cached MRMS .grib2.gz for sample_dt, or None."""
    for p in raw_dir.rglob("*.grib2.gz"):
        dt = _mrms_timestamp_from_path(p)
        if dt is not None and dt == sample_dt:
            return p
    return None


def find_cached_rtma_sample(raw_dir: Path, sample_dt: datetime) -> Optional[Path]:
    """Return path to a cached RTMA .grb2_wexp for sample_dt, or None."""
    for p in raw_dir.rglob("*.grb2_wexp"):
        dt = _rtma_timestamp_from_path(p)
        if dt is not None and dt == sample_dt:
            return p
    return None


def download_mrms_sample(raw_dir: Path, sample_dt: datetime) -> Optional[Path]:
    """Download exactly one MRMS sample file for sample_dt using the validated S3 gateway.

    Saves under raw_dir/<S3-key> (preserves the nested S3 key structure).
    Returns the local path, or None on failure.
    """
    from src.datasources.mrms import MrmsAwsQpe1hPass1
    from src.datasources.base import CONUS_BBOX

    source = MrmsAwsQpe1hPass1(download_concurrency=1)
    objects = source.list_sample_objects(
        start=sample_dt,
        end=sample_dt,
        region=CONUS_BBOX,
        variables=["precip"],
    )
    if not objects:
        LOGGER.warning("MRMS: no objects found for %s", sample_dt.isoformat())
        return None

    # Download only the first matching file
    files = source.download_sample(raw_dir, objects[:1])
    return files[0] if files else None


def download_rtma_sample(raw_dir: Path, sample_dt: datetime) -> Optional[Path]:
    """Download exactly one RTMA sample file for sample_dt using the validated S3 gateway.

    Saves under raw_dir/<S3-key>.
    Returns the local path, or None on failure.
    """
    from src.datasources.rtma import RtmaAwsConusDataSource
    from src.datasources.base import CONUS_BBOX

    source = RtmaAwsConusDataSource(download_concurrency=1)
    objects = source.list_sample_objects(
        start=sample_dt,
        end=sample_dt,
        region=CONUS_BBOX,
        variables=["TMP", "UGRD", "VGRD", "PRES"],
    )
    if not objects:
        LOGGER.warning("RTMA: no objects found for %s", sample_dt.isoformat())
        return None

    files = source.download_sample(raw_dir, objects[:1])
    return files[0] if files else None

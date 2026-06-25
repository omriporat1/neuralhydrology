"""
generate_fullperiod_spatial_mrms_qc.py

Targeted spatial MRMS QPE QC for Stage 1 full-period visual QC.
Produces static PNG snapshots of the MRMS 1h QPE raster at the hour of
maximum basin-mean precipitation, overlaid with the basin polygon and
gauge location. Optionally produces a 6h and 24h accumulated QPE snapshot.

*** SCOPE ***
This is a targeted smoke test, NOT broad all-case rendering.
Default cases: VQC-009 (SW monsoon, AZ) and VQC-012 (small flashy basin, TX).
VQC-007 (winter CO) is optionally supported.

*** RUN LOCATION ***
Designed to run on h2o. Use --dry-run for local path validation.
Raw MRMS GRIB2 files must exist under --raw-mrms-root.

Required h2o data inputs:
  --raw-mrms-root    /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/raw/mrms/
                       CONUS/MultiSensor_QPE_01H_Pass1_00.00
    Layout: {YYYYMMDD}/MRMS_MultiSensor_QPE_01H_Pass1_00.00_{YYYYMMDD}-{HH}0000.grib2.gz

  --chunks-root      /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/chunks
    Layout: {YYYY-MM}/combined_{YYYY-MM}.parquet

  --shapefile        /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/
                       02_basin_geometries/camelsh/shapefiles/CAMELSH_shapefile.shp

  --case-selection-csv  visual_qc_case_selection.csv

Usage (dry-run, local):
  python scripts/generate_fullperiod_spatial_mrms_qc.py \\
      --case-selection-csv tmp/.../visual_qc_case_selection.csv \\
      --mrms-raw-root /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/raw/mrms/CONUS/MultiSensor_QPE_01H_Pass1_00.00 \\
      --forcing-root  /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod \\
      --basin-shapefile /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/02_basin_geometries/camelsh/shapefiles/CAMELSH_shapefile.shp \\
      --out-dir       tmp/stage1_spatial_mrms_qc_dryrun \\
      --dry-run

Usage (h2o real run):
  bash scripts/run_fullperiod_spatial_mrms_qc_h2o.sh
"""

import argparse
import csv
import gzip
import os
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# Optional spatial dependencies — checked at runtime
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

try:
    import cfgrib
    HAS_CFGRIB = True
except ImportError:
    HAS_CFGRIB = False

try:
    import geopandas as gpd
    HAS_GPD = True
except ImportError:
    HAS_GPD = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CASE_IDS   = ["VQC-009", "VQC-012"]
OPTIONAL_CASE_IDS  = ["VQC-007"]           # winter CO; allowed but not default
MAX_CASES_DEFAULT  = 3                     # safety cap

MRMS_PRODUCT       = "mrms_qpe_1h_pass1"
PARQUET_COLS       = ["STAID", "product", "valid_time_utc", "weighted_mean"]
MRMS_FILL_VALUE    = -999.0                # MRMS no-data sentinel
MAP_BUFFER_DEG     = 0.75                  # degrees of padding around basin bbox
DPI_DEFAULT        = 120

ACC_HOURS_SHORT    = 6
ACC_HOURS_LONG     = 24

# CAMELSH shapefile: candidate column name substrings for the STAID/gauge ID field.
# Matched case-insensitively against actual shapefile columns.
SHAPEFILE_STAID_KEYWORDS = ["gage_id", "gauge_id", "hru_id", "station_id", "staid", "site_no"]

MANIFEST_FIELDS = [
    "case_id", "STAID", "category", "basin_name",
    "render_window_start_utc", "render_window_end_utc",
    "max_mrms_hour_utc", "max_mrms_basin_mean_mm",
    "grib_path_max_hour", "grib_exists_max_hour",
    "snapshot_path", "acc6h_path", "acc24h_path",
    "status", "error", "runtime_s",
]

SNAPSHOT_SUFFIX    = "_mrms_max_hour.png"
ACC6H_SUFFIX       = "_mrms_acc6h.png"
ACC24H_SUFFIX      = "_mrms_acc24h.png"
MANIFEST_CSV       = "spatial_mrms_qc_manifest.csv"
SUMMARY_MD         = "spatial_mrms_qc_summary.md"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Targeted spatial MRMS QPE QC snapshots for Stage 1 full-period VQC. "
            "Default: VQC-009 and VQC-012. Run on h2o; use --dry-run locally."
        )
    )
    p.add_argument("--case-selection-csv", required=True, metavar="FILE",
                   help="visual_qc_case_selection.csv from generate_visual_qc_case_selection.py")
    p.add_argument("--case-ids", nargs="+", default=DEFAULT_CASE_IDS,
                   help=f"Case IDs to process (default: {' '.join(DEFAULT_CASE_IDS)})")
    p.add_argument("--mrms-raw-root", required=True, metavar="DIR",
                   help=("Root of raw MRMS GRIB2 tree. "
                         "Expected layout: {YYYYMMDD}/MRMS_..._YYYYMMDD-HH0000.grib2.gz"))
    p.add_argument("--forcing-root", required=True, metavar="DIR",
                   help=("Root of full-period forcing outputs. "
                         "Chunks expected at {forcing_root}/chunks/{YYYY-MM}/combined_{YYYY-MM}.parquet"))
    p.add_argument("--basin-shapefile", required=True, metavar="FILE",
                   help="CAMELSH basin polygons shapefile (.shp)")
    p.add_argument("--out-dir", required=True, metavar="DIR",
                   help="Output directory for PNGs and manifest (should be under tmp/)")
    p.add_argument("--max-cases", type=int, default=MAX_CASES_DEFAULT,
                   help=f"Safety cap on number of cases (default {MAX_CASES_DEFAULT})")
    p.add_argument("--dry-run", action="store_true",
                   help=("Validate case IDs, print expected GRIB2 paths, verify "
                         "shapefile and data paths — no PNGs rendered."))
    p.add_argument("--dpi", type=int, default=DPI_DEFAULT)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-acc", action="store_true",
                   help="Skip 6h and 24h accumulated snapshots (max-hour only)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Case-selection CSV
# ---------------------------------------------------------------------------
def load_case_selection(csv_path: str) -> dict:
    if not os.path.isfile(csv_path):
        print(f"ERROR: case-selection CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["case_id"]: r for r in rows}


def validate_cases(all_cases: dict, requested_ids: list, max_cases: int) -> list:
    allowed = set(DEFAULT_CASE_IDS) | set(OPTIONAL_CASE_IDS)
    bad = [cid for cid in requested_ids if cid not in allowed]
    if bad:
        print(f"ERROR: case IDs not in allowed spatial QC set {sorted(allowed)}: {bad}",
              file=sys.stderr)
        print("  Pass --case-ids VQC-009 VQC-012 (and optionally VQC-007).",
              file=sys.stderr)
        sys.exit(1)
    missing = [cid for cid in requested_ids if cid not in all_cases]
    if missing:
        print(f"ERROR: case IDs not found in CSV: {missing}", file=sys.stderr)
        sys.exit(1)
    if len(requested_ids) > max_cases:
        print(f"ERROR: {len(requested_ids)} cases > --max-cases={max_cases}",
              file=sys.stderr)
        sys.exit(1)
    selected = []
    for cid in requested_ids:
        c = all_cases[cid]
        # STAID must be a string; never apply zero-padding
        c["STAID"] = str(c["STAID"])
        rws = c.get("render_window_start_utc", "")
        rwe = c.get("render_window_end_utc", "")
        if not rws or not rwe:
            print(f"ERROR: {cid} missing render_window_start_utc or render_window_end_utc",
                  file=sys.stderr)
            sys.exit(1)
        selected.append(c)
    return selected


# ---------------------------------------------------------------------------
# MRMS path helpers
# ---------------------------------------------------------------------------
def mrms_grib_path(raw_mrms_root: Path, ts: pd.Timestamp) -> Path:
    """Build GRIB2 path for a given UTC hour. STAID not involved."""
    date_str = ts.strftime("%Y%m%d")
    hour_str = ts.strftime("%H")
    fname = f"MRMS_MultiSensor_QPE_01H_Pass1_00.00_{date_str}-{hour_str}0000.grib2.gz"
    return raw_mrms_root / date_str / fname


def months_spanning(t0: pd.Timestamp, t1: pd.Timestamp) -> list:
    months = []
    cur = t0.to_period("M")
    end = t1.to_period("M")
    while cur <= end:
        months.append(str(cur))
        cur = cur + 1
    return months


# ---------------------------------------------------------------------------
# Parquet loader (basin-mean MRMS series)
# ---------------------------------------------------------------------------
def load_mrms_series(chunks_root: Path, staid: str,
                     render_start: pd.Timestamp, render_end: pd.Timestamp
                     ) -> pd.Series:
    """Load hourly basin-mean MRMS QPE for one basin from combined Parquets."""
    months = months_spanning(render_start, render_end)
    frames = []
    for ym in months:
        pq = chunks_root / ym / f"combined_{ym}.parquet"
        if not pq.exists():
            print(f"  WARNING: Parquet not found: {pq}")
            continue
        df = pd.read_parquet(
            pq,
            columns=PARQUET_COLS,
            filters=[("STAID", "==", staid), ("product", "==", MRMS_PRODUCT)],
        )
        df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"], utc=True)
        df = df[(df["valid_time_utc"] >= render_start) & (df["valid_time_utc"] <= render_end)]
        frames.append(df[["valid_time_utc", "weighted_mean"]])

    full_idx = pd.date_range(render_start, render_end, freq="h", tz="UTC")
    if not frames:
        return pd.Series(np.nan, index=full_idx, name="mrms_mm")
    combined = pd.concat(frames).sort_values("valid_time_utc").drop_duplicates("valid_time_utc")
    return combined.set_index("valid_time_utc")["weighted_mean"].reindex(full_idx)


def find_max_mrms_hour(mrms_s: pd.Series) -> pd.Timestamp:
    """Return the timestamp of maximum basin-mean MRMS; NaN hours excluded."""
    valid = mrms_s.dropna()
    if valid.empty:
        return None
    return valid.idxmax()


# ---------------------------------------------------------------------------
# GRIB2 reader
# ---------------------------------------------------------------------------
def read_mrms_grib_gz(grib_path: Path) -> tuple:
    """
    Decompress and read a .grib2.gz MRMS file.
    Returns (data_2d, lats_1d_or_2d, lons_1d_or_2d).
    Values < 0 (including MRMS fill -999.0) are set to NaN.
    Requires cfgrib.
    """
    if not HAS_CFGRIB:
        raise ImportError("cfgrib is required to read GRIB2 files: pip install cfgrib")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".grib2")
    os.close(tmp_fd)
    try:
        with gzip.open(str(grib_path), "rb") as gz_in, open(tmp_path, "wb") as out:
            shutil.copyfileobj(gz_in, out)

        # cfgrib via xarray.
        # indexpath="" prevents cfgrib from writing a .idx file alongside the temp file.
        import xarray as xr
        ds = xr.open_dataset(
            tmp_path,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
        )
        # MRMS QPE variable is typically named "unknown" (non-WMO product).
        var_candidates = [v for v in ds.data_vars]
        if not var_candidates:
            raise ValueError(f"No data variables found in {grib_path.name}")
        data_var = "unknown" if "unknown" in var_candidates else var_candidates[0]
        data = ds[data_var].values.squeeze().astype(float)
        lats = ds["latitude"].values
        lons = ds["longitude"].values
        ds.close()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Mask missing values (MRMS fill = -999.0; also mask any other negatives)
    data = np.where(data < 0.0, np.nan, data)
    return data, lats, lons


# ---------------------------------------------------------------------------
# Accumulated MRMS raster
# ---------------------------------------------------------------------------
def load_acc_mrms_raster(raw_mrms_root: Path, center_ts: pd.Timestamp,
                          acc_hours: int) -> tuple:
    """
    Sum acc_hours GRIB2 files ending at (and including) center_ts.
    Returns (acc_data, lats, lons) or None if any file is missing.
    """
    times = [center_ts - pd.Timedelta(hours=h) for h in range(acc_hours - 1, -1, -1)]
    acc = None
    lats = lons = None
    for ts in times:
        p = mrms_grib_path(raw_mrms_root, ts)
        if not p.exists():
            print(f"  WARNING: missing for accumulation: {p.name}")
            return None
        data, lats, lons = read_mrms_grib_gz(p)
        acc = data if acc is None else acc + np.where(np.isnan(data), 0.0, data)
    # Where all hours were NaN, set to NaN
    return acc, lats, lons


# ---------------------------------------------------------------------------
# Basin polygon loader
# ---------------------------------------------------------------------------
def load_shapefile(shapefile: str) -> "gpd.GeoDataFrame":
    if not HAS_GPD:
        raise ImportError("geopandas is required: pip install geopandas")
    gdf = gpd.read_file(shapefile)
    return gdf


def find_staid_col(gdf: "gpd.GeoDataFrame") -> str:
    """Case-insensitive match against known gauge ID column name keywords."""
    cols_lower = {c.lower(): c for c in gdf.columns}
    for keyword in SHAPEFILE_STAID_KEYWORDS:
        if keyword.lower() in cols_lower:
            return cols_lower[keyword.lower()]
    raise ValueError(
        f"Cannot find STAID column in shapefile. Available: {list(gdf.columns)}. "
        f"Expected a column matching one of (case-insensitive): {SHAPEFILE_STAID_KEYWORDS}"
    )


def get_basin_polygon(gdf: "gpd.GeoDataFrame", staid_col: str, staid: str):
    """
    Return the basin GeoDataFrame row for staid.
    STAID preserved as string — no zero-padding applied.
    The shapefile may store the ID as int or zero-padded string; try both.
    """
    # Direct string match first
    mask = gdf[staid_col].astype(str) == staid
    subset = gdf[mask]
    if not subset.empty:
        return subset.iloc[0:1]
    # Try matching as integer (in case shapefile stores as int without leading zeros)
    try:
        staid_int = int(staid)
        mask_int = gdf[staid_col].astype(str) == str(staid_int)
        subset_int = gdf[mask_int]
        if not subset_int.empty:
            return subset_int.iloc[0:1]
    except (ValueError, TypeError):
        pass
    return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
_MRMS_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "mrms_qpe",
    ["#ffffff", "#90d090", "#00c800", "#00a000", "#005000",
     "#ffff00", "#ffa000", "#ff5000", "#ff0000", "#c00000"],
)
_MRMS_NORM = mcolors.BoundaryNorm(
    [0, 0.25, 0.5, 1, 2, 5, 10, 15, 25, 50, 100], ncolors=256,
)


def _map_extent_from_basin(basin_row, buffer_deg: float = MAP_BUFFER_DEG):
    """Return [lon_min, lon_max, lat_min, lat_max] with buffer around basin bbox."""
    bbox = basin_row.geometry.values[0].bounds  # (minx, miny, maxx, maxy)
    return [
        bbox[0] - buffer_deg, bbox[2] + buffer_deg,
        bbox[1] - buffer_deg, bbox[3] + buffer_deg,
    ]


def plot_snapshot(data: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                  basin_row, gauge_lat: float, gauge_lon: float,
                  case: dict, title_suffix: str, out_path: Path, dpi: int):
    """
    Render one MRMS QPE raster snapshot with basin polygon and gauge point.
    Uses cartopy if available, falls back to plain matplotlib.
    """
    cid   = case["case_id"]
    staid = case["STAID"]
    name  = case.get("basin_name", "")[:60]
    cat   = case["selection_category"]
    area  = case.get("drain_sqkm", "?")

    title = (f"{cid}  STAID {staid}  |  {cat}\n"
             f"{name}  |  {area} km²  |  {title_suffix}")

    extent = None
    if basin_row is not None:
        try:
            extent = _map_extent_from_basin(basin_row)
        except Exception:
            pass

    if HAS_CARTOPY and HAS_GPD:
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(
            figsize=(10, 7), dpi=dpi,
            subplot_kw={"projection": proj}
        )
        fig.suptitle(title, fontsize=8.5, fontweight="bold")

        # MRMS raster
        if lats.ndim == 1 and lons.ndim == 1:
            # Regular grid: use pcolormesh with meshgrid
            LON, LAT = np.meshgrid(lons, lats)
        else:
            LON, LAT = lons, lats
        ax.pcolormesh(LON, LAT, data, transform=proj,
                      cmap=_MRMS_CMAP, norm=_MRMS_NORM,
                      shading="auto", zorder=1)
        sm = plt.cm.ScalarMappable(cmap=_MRMS_CMAP, norm=_MRMS_NORM)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, label="QPE (mm)")

        # Map features
        ax.add_feature(cfeature.STATES.with_scale("10m"),
                       edgecolor="gray", linewidth=0.7, zorder=2)
        ax.add_feature(cfeature.RIVERS.with_scale("10m"),
                       edgecolor="steelblue", linewidth=0.5, alpha=0.6, zorder=2)

        # Basin polygon
        if basin_row is not None:
            try:
                basin_row.to_crs("EPSG:4326").plot(
                    ax=ax, transform=proj,
                    facecolor="none", edgecolor="black", linewidth=2.0, zorder=4,
                    label=f"STAID {staid}"
                )
            except Exception as e:
                print(f"  WARNING: basin polygon plot failed: {e}")

        # Gauge point
        if gauge_lat and gauge_lon:
            ax.plot(gauge_lon, gauge_lat, transform=proj,
                    marker="*", color="gold", markersize=12, markeredgecolor="black",
                    markeredgewidth=0.7, zorder=5, label="Gauge")
            ax.legend(fontsize=6.5, loc="lower right")

        if extent:
            ax.set_extent(extent, crs=proj)
        ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                     alpha=0.5, linestyle="--", zorder=0)

    else:
        # Fallback: plain imshow without basemap
        fig, ax = plt.subplots(figsize=(10, 7), dpi=dpi)
        fig.suptitle(title + "\n(cartopy/geopandas not available — no basemap)", fontsize=8.5)
        if lats.ndim == 1 and lons.ndim == 1:
            ax.imshow(data, origin="upper",
                      extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                      cmap=_MRMS_CMAP, norm=_MRMS_NORM, aspect="auto", zorder=1)
        else:
            ax.imshow(data, cmap=_MRMS_CMAP, norm=_MRMS_NORM,
                      origin="upper", aspect="auto", zorder=1)
        if extent:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        sm = plt.cm.ScalarMappable(cmap=_MRMS_CMAP, norm=_MRMS_NORM)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.6, label="QPE (mm)")

    plt.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight", dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dry-run report
# ---------------------------------------------------------------------------
def dry_run_report(cases: list, raw_mrms_root: Path, chunks_root: Path,
                   shapefile: str, out_dir: Path):
    OK   = "OK"
    MISS = "MISSING"

    print()
    print("=" * 70)
    print("DRY-RUN REPORT — no PNGs generated")
    print("=" * 70)
    print(f"  Cases:         {[c['case_id'] for c in cases]}")
    print(f"  Raw MRMS root: {raw_mrms_root}")
    print(f"  Chunks root:   {chunks_root}")
    print(f"  Shapefile:     {shapefile}")
    print(f"  Output dir:    {out_dir}")
    print()

    # Shapefile check
    shp_status = OK if os.path.isfile(shapefile) else MISS
    print(f"  Shapefile [{shp_status}]: {shapefile}")
    if shp_status == MISS:
        print("  *** Shapefile missing — spatial plots will fail ***")
    print()

    all_ok = shp_status == OK

    for c in cases:
        cid   = c["case_id"]
        staid = c["STAID"]
        rws   = c["render_window_start_utc"]
        rwe   = c["render_window_end_utc"]
        name  = c.get("basin_name", "")
        cat   = c["selection_category"]

        rstart = pd.Timestamp(rws)
        rend   = pd.Timestamp(rwe)

        print(f"  {'-'*66}")
        print(f"  Case:    {cid}  |  STAID: {staid}  |  {cat}")
        print(f"  Basin:   {name}")
        print(f"  Window:  {rws} → {rwe}")
        print(f"  Gauge:   lat={c.get('lat_gage', '?')} lon={c.get('lng_gage', '?')}")

        # Parquet check
        months = months_spanning(rstart, rend)
        for ym in months:
            pq = chunks_root / ym / f"combined_{ym}.parquet"
            st = OK if pq.exists() else MISS
            if st == MISS:
                all_ok = False
            print(f"  Parquet [{st}]: {pq}")

        # MRMS GRIB2: check a sample of hours in the render window
        # Without loading parquet, we don't know the max hour — check first/last/mid
        sample_times = [rstart, rstart + (rend - rstart) / 2, rend]
        print(f"  MRMS GRIB2 sample paths (first/mid/last of render window):")
        for ts in sample_times:
            gp = mrms_grib_path(raw_mrms_root, ts)
            st = OK if gp.exists() else MISS
            if st == MISS:
                all_ok = False
            print(f"    [{st}]: {gp.name}  @ {gp.parent}")

        # Accumulation GRIB2: check files for 6h and 24h prior to render_end
        print(f"  MRMS GRIB2 for {ACC_HOURS_SHORT}h accumulation (ending at render_end):")
        for h in range(ACC_HOURS_SHORT):
            ts = rend - pd.Timedelta(hours=h)
            gp = mrms_grib_path(raw_mrms_root, ts)
            st = OK if gp.exists() else MISS
            if st == MISS:
                all_ok = False
            print(f"    [{st}]: {gp.name}")

        # Expected outputs
        case_dir = out_dir / cid
        print(f"  Expected outputs:")
        print(f"    {case_dir / (cid + SNAPSHOT_SUFFIX)}")
        print(f"    {case_dir / (cid + ACC6H_SUFFIX)}")
        print(f"    {case_dir / (cid + ACC24H_SUFFIX)}")
        print()

    print(f"  Manifest: {out_dir / MANIFEST_CSV}")
    print(f"  Summary:  {out_dir / SUMMARY_MD}")
    print()
    if all_ok:
        print("DRY-RUN: all checked paths OK")
    else:
        print("DRY-RUN: some paths MISSING — see above")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Per-case processor
# ---------------------------------------------------------------------------
def process_case(case: dict, raw_mrms_root: Path, chunks_root: Path,
                 basin_gdf, staid_col: str,
                 out_dir: Path, dpi: int, overwrite: bool,
                 no_acc: bool) -> dict:
    cid   = case["case_id"]
    staid = case["STAID"]   # string; no zero-padding applied
    rws   = case["render_window_start_utc"]
    rwe   = case["render_window_end_utc"]
    name  = case.get("basin_name", "")
    cat   = case["selection_category"]

    t0 = time.time()
    result = {
        "case_id": cid, "STAID": staid, "category": cat, "basin_name": name,
        "render_window_start_utc": rws, "render_window_end_utc": rwe,
        "max_mrms_hour_utc": "", "max_mrms_basin_mean_mm": "",
        "grib_path_max_hour": "", "grib_exists_max_hour": "",
        "snapshot_path": "", "acc6h_path": "", "acc24h_path": "",
        "status": "PENDING", "error": "", "runtime_s": 0.0,
    }

    case_dir = out_dir / cid
    snap_path  = case_dir / (cid + SNAPSHOT_SUFFIX)
    acc6_path  = case_dir / (cid + ACC6H_SUFFIX)
    acc24_path = case_dir / (cid + ACC24H_SUFFIX)

    if snap_path.exists() and not overwrite:
        print(f"  {cid}: already exists, skipping (--overwrite to force)")
        result["status"] = "SKIPPED"
        result["snapshot_path"] = str(snap_path)
        return result

    case_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(f"  {cid}  STAID={staid}  {rws[:10]} → {rwe[:10]}")
    print(f"{'=' * 64}")

    try:
        render_start = pd.Timestamp(rws)
        render_end   = pd.Timestamp(rwe)

        # Gauge coordinates from case CSV (may be empty strings)
        try:
            gauge_lat = float(case.get("lat_gage") or "nan")
            gauge_lon = float(case.get("lng_gage") or "nan")
        except (ValueError, TypeError):
            gauge_lat = gauge_lon = float("nan")
        if not np.isfinite(gauge_lat) or not np.isfinite(gauge_lon):
            gauge_lat = gauge_lon = None

        # Basin polygon
        basin_row = None
        if basin_gdf is not None and staid_col:
            basin_row = get_basin_polygon(basin_gdf, staid_col, staid)
            if basin_row is None:
                print(f"  WARNING: STAID {staid} not found in shapefile column '{staid_col}'")

        # Basin-mean MRMS series from Parquet
        print(f"  Loading MRMS series from Parquet …")
        mrms_s = load_mrms_series(chunks_root, staid, render_start, render_end)

        max_ts = find_max_mrms_hour(mrms_s)
        if max_ts is None:
            raise ValueError("All MRMS hours are NaN in render window — cannot find max hour")

        max_val = float(mrms_s[max_ts])
        result["max_mrms_hour_utc"]     = str(max_ts)
        result["max_mrms_basin_mean_mm"] = round(max_val, 3)
        print(f"  Max basin-mean MRMS: {max_val:.2f} mm at {max_ts}")

        # Load GRIB2 at max hour
        grib_p = mrms_grib_path(raw_mrms_root, max_ts)
        result["grib_path_max_hour"]    = str(grib_p)
        result["grib_exists_max_hour"]  = str(grib_p.exists())

        print(f"  Loading GRIB2: {grib_p.name} …")
        data, lats, lons = read_mrms_grib_gz(grib_p)
        print(f"  GRIB2 grid: {data.shape}  lat [{np.nanmin(lats):.2f}, {np.nanmax(lats):.2f}]"
              f"  lon [{np.nanmin(lons):.2f}, {np.nanmax(lons):.2f}]")
        print(f"  MRMS raster: max={np.nanmax(data):.2f} mm  "
              f"NaN fraction={np.isnan(data).mean():.3f}")

        # Max-hour snapshot
        title = (f"MRMS 1h QPE max hour: {max_ts.strftime('%Y-%m-%d %H:00 UTC')}  "
                 f"Basin mean: {max_val:.2f} mm")
        print(f"  Rendering max-hour snapshot …")
        plot_snapshot(data, lats, lons, basin_row, gauge_lat, gauge_lon,
                      case, title, snap_path, dpi)
        result["snapshot_path"] = str(snap_path)
        print(f"  Saved: {snap_path}")

        # 6h accumulation
        if not no_acc:
            print(f"  Loading {ACC_HOURS_SHORT}h accumulated MRMS …")
            acc6_result = load_acc_mrms_raster(raw_mrms_root, max_ts, ACC_HOURS_SHORT)
            if acc6_result is not None:
                acc6_data, acc6_lats, acc6_lons = acc6_result
                t6_start = max_ts - pd.Timedelta(hours=ACC_HOURS_SHORT - 1)
                title6 = (f"MRMS {ACC_HOURS_SHORT}h accumulated QPE ending "
                          f"{max_ts.strftime('%Y-%m-%d %H:00 UTC')}  "
                          f"({t6_start.strftime('%H:00')} – {max_ts.strftime('%H:00 UTC')})")
                plot_snapshot(acc6_data, acc6_lats, acc6_lons, basin_row,
                              gauge_lat, gauge_lon, case, title6, acc6_path, dpi)
                result["acc6h_path"] = str(acc6_path)
                print(f"  Saved: {acc6_path}")
            else:
                print(f"  WARNING: skipping {ACC_HOURS_SHORT}h accumulation (missing files)")

            # 24h accumulation
            print(f"  Loading {ACC_HOURS_LONG}h accumulated MRMS …")
            acc24_result = load_acc_mrms_raster(raw_mrms_root, max_ts, ACC_HOURS_LONG)
            if acc24_result is not None:
                acc24_data, acc24_lats, acc24_lons = acc24_result
                t24_start = max_ts - pd.Timedelta(hours=ACC_HOURS_LONG - 1)
                title24 = (f"MRMS {ACC_HOURS_LONG}h accumulated QPE ending "
                           f"{max_ts.strftime('%Y-%m-%d %H:00 UTC')}  "
                           f"({t24_start.strftime('%m-%d %H:00')} – {max_ts.strftime('%m-%d %H:00 UTC')})")
                plot_snapshot(acc24_data, acc24_lats, acc24_lons, basin_row,
                              gauge_lat, gauge_lon, case, title24, acc24_path, dpi)
                result["acc24h_path"] = str(acc24_path)
                print(f"  Saved: {acc24_path}")
            else:
                print(f"  WARNING: skipping {ACC_HOURS_LONG}h accumulation (missing files)")

        result["status"] = "OK"

    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        result["status"] = "FAIL"
        result["error"]  = str(e)
        try:
            plt.close("all")
        except Exception:
            pass

    result["runtime_s"] = round(time.time() - t0, 1)
    return result


# ---------------------------------------------------------------------------
# Manifest and summary
# ---------------------------------------------------------------------------
def write_manifest(out_dir: Path, results: list):
    path = out_dir / MANIFEST_CSV
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS,
                           extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(results)
    return path


def write_summary(out_dir: Path, results: list, elapsed_s: float):
    n_ok   = sum(1 for r in results if r["status"] == "OK")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    lines = [
        "# Stage 1 Forcing — Spatial MRMS QC Snapshots",
        "",
        f"**Cases:** {len(results)}  **OK:** {n_ok}  **FAIL:** {n_fail}  "
        f"**Runtime:** {elapsed_s:.0f}s  ",
        "",
        "| Case | STAID | Max hour | Max mm | Snapshot | 6h acc | 24h acc | Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['case_id']} | {r['STAID']} | {r['max_mrms_hour_utc']} | "
            f"{r['max_mrms_basin_mean_mm']} | "
            f"{'Y' if r['snapshot_path'] else ''} | "
            f"{'Y' if r['acc6h_path'] else ''} | "
            f"{'Y' if r['acc24h_path'] else ''} | "
            f"{r['status']} |"
        )
    lines += [
        "",
        "## Review Instructions",
        "",
        "For each snapshot, assess:",
        "1. **Max-hour raster:** Is the precipitation spatially placed over or near the basin?",
        "2. **Basin polygon overlay:** Does the boundary correctly enclose the watershed?",
        "3. **Gauge point:** Is the gauge marker at the correct outlet location?",
        "4. **6h / 24h accumulation:** Does cumulative QPE pattern match event expectations?",
        "5. **VQC-009 (SW monsoon):** Convective cell visible over or near Sabino Canyon?",
        "6. **VQC-012 (small flashy):** Event captures the small urban watershed (4.6 km²)?",
        "",
        "*Outputs not committed to git.*",
    ]
    path = out_dir / SUMMARY_MD
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    raw_mrms_root = Path(args.mrms_raw_root)
    chunks_root   = Path(args.forcing_root) / "chunks"
    shapefile     = args.basin_shapefile
    out_dir       = Path(args.out_dir)

    all_cases = load_case_selection(args.case_selection_csv)
    cases     = validate_cases(all_cases, args.case_ids, args.max_cases)

    print(f"\nFlash-NH Stage 1 — Spatial MRMS QC")
    print(f"  Mode:         {'DRY-RUN' if args.dry_run else 'RENDER'}")
    print(f"  Cases:        {[c['case_id'] for c in cases]}")
    print(f"  MRMS raw:     {raw_mrms_root}")
    print(f"  Forcing root: {args.forcing_root}")
    print(f"  Chunks root:  {chunks_root}")
    print(f"  Shapefile:    {shapefile}")
    print(f"  Output:       {out_dir}")

    if args.dry_run:
        dry_run_report(cases, raw_mrms_root, chunks_root, shapefile, out_dir)
        return 0

    # Check required libraries
    missing_libs = []
    if not HAS_CFGRIB:
        missing_libs.append("cfgrib")
    if not HAS_GPD:
        missing_libs.append("geopandas")
    if not HAS_CARTOPY:
        print("  WARNING: cartopy not available — basemap features disabled")
    if missing_libs:
        print(f"ERROR: required libraries not installed: {missing_libs}", file=sys.stderr)
        print(f"  pip install {' '.join(missing_libs)}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load shapefile once
    print(f"\n  Loading shapefile …")
    try:
        basin_gdf = load_shapefile(shapefile)
        staid_col = find_staid_col(basin_gdf)
        print(f"  Shapefile: {len(basin_gdf)} basins, STAID col = '{staid_col}'")
    except Exception as e:
        print(f"  WARNING: shapefile load failed: {e} — polygon overlays disabled")
        basin_gdf = None
        staid_col = None

    t0_total = time.time()
    results = []
    for case in cases:
        res = process_case(
            case, raw_mrms_root, chunks_root,
            basin_gdf, staid_col,
            out_dir, dpi=args.dpi, overwrite=args.overwrite,
            no_acc=args.no_acc,
        )
        results.append(res)

    elapsed = time.time() - t0_total
    mcsv = write_manifest(out_dir, results)
    smd  = write_summary(out_dir, results, elapsed)

    print(f"\n{'=' * 64}")
    n_ok = sum(1 for r in results if r["status"] == "OK")
    for r in results:
        print(f"  {r['case_id']}  {r['status']:8s}  "
              f"max={r['max_mrms_basin_mean_mm']} mm @ {r['max_mrms_hour_utc']}  "
              f"{r['runtime_s']:.0f}s")
    print(f"\n  {n_ok}/{len(results)} OK   elapsed: {elapsed:.0f}s")
    print(f"  Manifest: {mcsv}")
    print(f"  Summary:  {smd}")
    print(f"{'=' * 64}")

    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
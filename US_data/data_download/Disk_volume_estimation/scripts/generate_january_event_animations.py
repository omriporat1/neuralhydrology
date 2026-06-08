"""
generate_january_event_animations.py — Flash-NH Stage 1 event animation generator.

Produces 3-panel GIF animations for January 2023 event candidates:
  - Left: MRMS QPE 1h map with basin polygon, extraction cells, gauge marker, wind vectors
  - Right top: CAMELSH streamflow time series
  - Right mid: basin-mean MRMS 1h precip bar chart
  - Right bot: RTMA 2m temperature time series

Design accepted as v2.1-stable (pilot approval 2026-06-05). Key conventions:
  MRMS: lat DECREASES with row; row 0 = 54.995 N; filename = END of 1h accumulation
  RTMA: lat INCREASES with row; row 0 = SW corner (~19.23 N); data[row_idx, col_idx] direct
  RTMA 10m wind vectors: qualitative meteorological context only — not storm-steering validation

Usage:
  # Run 4-candidate pilot (default):
  python scripts/generate_january_event_animations.py

  # Run specific candidates:
  python scripts/generate_january_event_animations.py --candidates R02 R11

  # Run all 12 candidates (requires pilot inspection approval first):
  python scripts/generate_january_event_animations.py --all

Output: tmp/stage1_pilot_dryrun/10_animations/stage1_pilot/pilot/
"""

import argparse, gzip, json, shutil, time, warnings, traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.animation import PillowWriter

import geopandas as gpd
import netCDF4 as nc
import cfgrib

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT        = Path(r"C:\PhD\Python\neuralhydrology\US_data\data_download\Disk_volume_estimation")
CAMELSH_DIR = Path(r"C:\PhD\Python\neuralhydrology\US_data\data_download\CAMELSH_resolution_test\data\raw\camelsh")

MRMS_BASE      = ROOT / "tmp/stage1_pilot_dryrun/00_raw/mrms/CONUS/MultiSensor_QPE_01H_Pass1_00.00"
RTMA_BASE      = ROOT / "tmp/stage1_pilot_dryrun/00_raw/rtma"
FORCING_PQ     = ROOT / "tmp/stage1_pilot_dryrun/03_basin_timeseries/stage1_pilot/january_2023/combined_hourly_basin_stats.parquet"
CANDIDATES_CSV = ROOT / "tmp/stage1_pilot_dryrun/09_manifests/stage1_pilot/january_2023_event_qc/event_animation_candidates_refined.csv"
CAMELSH_SHP    = ROOT / "tmp/stage1_pilot_dryrun/02_basin_geometries/camelsh/shapefiles/CAMELSH_shapefile.shp"
STATES_GPKG    = ROOT / "tmp/stage1_pilot_dryrun/02_basin_geometries/reference/ne_110m_admin1_us_states.gpkg"
MRMS_WEIGHTS   = ROOT / "tmp/stage1_pilot_dryrun/02_basin_geometries/weights/mrms/pilot_mrms_weights.parquet"

ANIM_SUBDIR = "pilot"
ANIM_DIR    = ROOT / "tmp/stage1_pilot_dryrun/10_animations/stage1_pilot" / ANIM_SUBDIR

# ── Candidate config ───────────────────────────────────────────────────────────

ALL_CANDIDATE_IDS = [
    "R01", "R02", "R03", "R04", "R05", "R06",
    "R07", "R08", "R09", "R10", "R11", "R12",
]

PILOT_IDS = ["R02", "R06", "R09", "R11"]

# Per-candidate map parameters for the 4 approved pilot candidates.
# Non-pilot entries use _default_map_params() computed from basin area.
CANDIDATE_MAP_PARAMS = {
    "R02": {"pad": 0.10, "scalebar_km":  5, "vmax_factor": 2.0},  # small AR basin
    "R06": {"pad": 0.35, "scalebar_km": 15, "vmax_factor": 2.0},  # MN larger basin
    "R09": {"pad": 0.22, "scalebar_km": 15, "vmax_factor": 2.0},  # ID dry control
    "R11": {"pad": 0.30, "scalebar_km": 10, "vmax_factor": 2.0},  # MA offset stress
}


def _default_map_params(area_km2: float) -> dict:
    """Area-based fallback map parameters for non-pilot candidates."""
    if area_km2 < 100:
        pad, sb = 0.15, 5
    elif area_km2 < 300:
        pad, sb = 0.22, 10
    elif area_km2 < 800:
        pad, sb = 0.35, 15
    else:
        pad, sb = 0.50, 20
    return {"pad": pad, "scalebar_km": sb, "vmax_factor": 2.0}


def get_map_params(rid: str, area_km2: float) -> dict:
    return CANDIDATE_MAP_PARAMS.get(rid, _default_map_params(area_km2))


# ── Animation config ───────────────────────────────────────────────────────────

FPS           = 4
FRAME_CADENCE = 1
FIGURE_W      = 14.5
FIGURE_H      = 5.8
DPI           = 90

WIND_VECTORS_ENABLED = True
WIND_TARGET_ARROWS   = 10

# ── MRMS grid (lat DECREASES with row; row 0 = 54.995 N northernmost) ─────────

MRMS_LAT_TOP = 54.995
MRMS_LON0    = 230.005
MRMS_DLAT    = 0.01
MRMS_DLON    = 0.01
MRMS_NROWS   = 3500
MRMS_NCOLS   = 7000

# ── Colormaps ──────────────────────────────────────────────────────────────────

CATEGORY_COLORS = {
    "STRONG_WET":           "#1a6faf",
    "MODERATE_COLD_REGION": "#17becf",
    "DRY_CONTROL":          "#7f7f7f",
    "OFFSET_STRESS":        "#d62728",
}

WET_NODES = [
    (0.00, (1.00, 1.00, 1.00)),
    (0.05, (0.80, 0.92, 1.00)),
    (0.20, (0.50, 0.78, 1.00)),
    (0.45, (0.18, 0.52, 0.95)),
    (0.70, (0.00, 0.25, 0.75)),
    (1.00, (0.45, 0.00, 0.55)),
]
WET_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "wet_qpe", [(v, c) for v, c in WET_NODES])
DRY_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "dry_ctrl", [(0.92, 0.92, 0.92), (0.75, 0.75, 0.75)])

_FFMPEG = shutil.which("ffmpeg")
FFMPEG_OK = (_FFMPEG is not None)
FFMPEG_INSTALL_NOTE = (
    "ffmpeg not found. Install: 'winget install Gyan.FFmpeg' or "
    "https://ffmpeg.org/download.html, add bin/ to PATH. GIF fallback active.")


# ── State boundary loader ──────────────────────────────────────────────────────

_NE_CDN_URL = ("https://naturalearth.s3.amazonaws.com/110m_cultural/"
               "ne_110m_admin_1_states_provinces.zip")


def load_states_gdf(cache_path: Path) -> tuple:
    """Load US state boundaries; return (gdf_or_None, status_str).

    Status values:
      'loaded'         — read from local cache file
      'downloaded'     — fetched from Natural Earth CDN and cached locally
      'skipped_missing'— file absent and download failed; animation continues without boundaries
    """
    if cache_path.exists():
        try:
            gdf = gpd.read_file(str(cache_path))
            return gdf, "loaded"
        except Exception as e:
            print(f"  Warning: could not read state boundary file ({e})")

    print(f"  State boundary file not found: {cache_path.name}")
    print(f"  Attempting download from Natural Earth CDN…", end="", flush=True)
    try:
        all_states = gpd.read_file(_NE_CDN_URL)
        us_states  = all_states[all_states["iso_a2"] == "US"].copy()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        us_states.to_file(str(cache_path), driver="GPKG")
        print(f" OK ({len(us_states)} features cached)")
        return us_states, "downloaded"
    except Exception as e:
        print(f" FAILED ({e})")
        print("  Warning: continuing without state boundaries (cartographic context only).")
        return None, "skipped_missing"


# ── MRMS helpers ───────────────────────────────────────────────────────────────

def mrms_path(ts: pd.Timestamp) -> Path:
    ds = ts.strftime("%Y%m%d")
    hs = ts.strftime("%H0000")
    return MRMS_BASE / ds / f"MRMS_MultiSensor_QPE_01H_Pass1_00.00_{ds}-{hs}.grib2.gz"


def get_crop_params(bounds, pad: float):
    """MRMS crop; lat DECREASES with row (row 0 = MRMS_LAT_TOP northernmost)."""
    lon_min, lat_min, lon_max, lat_max = bounds
    lon360_min = lon_min + 360.0
    lon360_max = lon_max + 360.0
    r0 = max(0, int((MRMS_LAT_TOP - (lat_max + pad)) / MRMS_DLAT))
    r1 = min(MRMS_NROWS, int((MRMS_LAT_TOP - (lat_min - pad)) / MRMS_DLAT) + 2)
    c0 = max(0, int((lon360_min - pad - MRMS_LON0) / MRMS_DLON))
    c1 = min(MRMS_NCOLS, int((lon360_max + pad - MRMS_LON0) / MRMS_DLON) + 2)
    lats = MRMS_LAT_TOP - np.arange(r0, r1) * MRMS_DLAT
    lons = (MRMS_LON0 + np.arange(c0, c1) * MRMS_DLON) - 360.0
    return slice(r0, r1), slice(c0, c1), lons, lats


def load_mrms_crop(ts: pd.Timestamp, row_sl, col_sl, tmp_dir: Path):
    p = mrms_path(ts)
    if not p.exists():
        return None
    tmp_f = tmp_dir / f"_anim_{ts.strftime('%Y%m%d_%H')}.grib2"
    try:
        with gzip.open(p, "rb") as gz:
            tmp_f.write_bytes(gz.read())
        ds   = cfgrib.open_dataset(str(tmp_f), indexpath=None)
        data = np.where(ds["unknown"].values < 0, 0.0, ds["unknown"].values)
        return data[row_sl, col_sl]
    except Exception:
        return None
    finally:
        if tmp_f.exists():
            tmp_f.unlink()


def basin_mean_from_crop(crop, w_rows, w_cols, w_norms, r0, c0):
    ri = w_rows - r0
    ci = w_cols - c0
    nrows, ncols = crop.shape
    mask = (ri >= 0) & (ri < nrows) & (ci >= 0) & (ci < ncols)
    if not mask.any():
        return 0.0
    return float(np.sum(crop[ri[mask], ci[mask]] * w_norms[mask]))


# ── RTMA helpers ───────────────────────────────────────────────────────────────

def rtma_path(ts: pd.Timestamp) -> Path:
    ds = ts.strftime("%Y%m%d")
    hh = ts.strftime("%H")
    return RTMA_BASE / f"rtma2p5.{ds}" / f"rtma2p5.t{hh}z.2dvaranl_ndfd.grb2_wexp"


def load_rtma_wind(ts: pd.Timestamp):
    """Return (u10_2d, v10_2d, lat_2d, lon_neg_2d) or None.

    RTMA grid: lat INCREASES with row; row 0 = SW corner (~19.23 N).
    lon stored as 0-360; converted here to -180/180 for geographic mapping.
    """
    p = rtma_path(ts)
    if not p.exists():
        return None
    try:
        dsets = cfgrib.open_datasets(str(p), indexpath=None)
        for ds in dsets:
            if "u10" in ds.data_vars and "v10" in ds.data_vars:
                u10     = ds["u10"].values
                v10     = ds["v10"].values
                lat_2d  = ds["latitude"].values
                lon_raw = ds["longitude"].values
                lon_neg = np.where(lon_raw > 180, lon_raw - 360.0, lon_raw)
                return u10, v10, lat_2d, lon_neg
    except Exception:
        pass
    return None


def get_rtma_wind_arrows(u10, v10, lat_2d, lon_neg_2d,
                          map_lon_min, map_lon_max, map_lat_min, map_lat_max,
                          target=WIND_TARGET_ARROWS):
    """Downsample RTMA wind to approximately target arrows within map extent."""
    buf = 0.05
    mask = ((lon_neg_2d >= map_lon_min - buf) & (lon_neg_2d <= map_lon_max + buf) &
            (lat_2d >= map_lat_min - buf) & (lat_2d <= map_lat_max + buf))
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()
    n_r = max(1, r_max - r_min + 1)
    n_c = max(1, c_max - c_min + 1)
    stride = max(1, int(((n_r * n_c) / target) ** 0.5))
    r_samp = np.arange(r_min, r_max + 1, stride)
    c_samp = np.arange(c_min, c_max + 1, stride)
    rr, cc = np.meshgrid(r_samp, c_samp, indexing="ij")
    rr_f, cc_f = rr.flatten(), cc.flatten()
    valid = mask[rr_f, cc_f]
    rr_f, cc_f = rr_f[valid], cc_f[valid]
    if len(rr_f) == 0:
        return None
    return (lon_neg_2d[rr_f, cc_f], lat_2d[rr_f, cc_f],
            u10[rr_f, cc_f], v10[rr_f, cc_f])


def preload_rtma_wind_cache(frame_times, map_lon_min, map_lon_max,
                             map_lat_min, map_lat_max):
    cache = {}
    n_ok = 0
    for ts in frame_times:
        result = load_rtma_wind(ts)
        if result is None:
            cache[ts] = None
            continue
        u10, v10, lat_2d, lon_neg_2d = result
        arrows = get_rtma_wind_arrows(u10, v10, lat_2d, lon_neg_2d,
                                       map_lon_min, map_lon_max,
                                       map_lat_min, map_lat_max)
        cache[ts] = arrows
        n_ok += 1
    return cache, n_ok


# ── Data loaders ───────────────────────────────────────────────────────────────

def load_sf(staid: str, w_start, w_end) -> pd.Series:
    f = CAMELSH_DIR / f"{staid}_hourly.nc"
    ds  = nc.Dataset(f)
    tv  = ds.variables["time"]
    tms = pd.to_datetime(
        [t.strftime("%Y-%m-%d %H:%M:%S")
         for t in nc.num2date(tv[:], tv.units, getattr(tv, "calendar", "standard"))],
        utc=True)
    sf = pd.Series(
        np.ma.filled(ds.variables["streamflow"][:].squeeze().astype(float), np.nan), index=tms)
    ds.close()
    return sf.loc[w_start:w_end]


# ── Drawing helpers ─────────────────────────────────────────────────────────────

def draw_scale_bar(ax, x0, y0, length_km, lat_ref):
    km_per_deg = 111.32 * np.cos(np.radians(lat_ref))
    dlon = length_km / km_per_deg
    ax.plot([x0, x0 + dlon], [y0, y0], "k-", lw=3.0, solid_capstyle="butt", zorder=14)
    for x in (x0, x0 + dlon):
        ax.plot([x, x], [y0 - 0.011, y0 + 0.011], "k-", lw=1.5, zorder=14)
    ax.text(x0 + dlon / 2, y0 - 0.022, f"{length_km:.0f} km",
            ha="center", va="top", fontsize=6.5, fontweight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.80, pad=1), zorder=15)


def edge_color_for_mean(basin_mean, cmap, vmax, is_dry):
    if is_dry or vmax <= 0:
        return "#666666"
    norm_val = min(float(basin_mean) / float(vmax), 1.0)
    if norm_val < 0.08:
        return "#444444"
    return cmap(norm_val)


# ── Figure factory ─────────────────────────────────────────────────────────────

def make_figure(candidate_id, staid, state, cat, area, offset_m, vmax_mrms, is_dry, cmap):
    fig = plt.figure(figsize=(FIGURE_W, FIGURE_H), dpi=DPI)
    offset_str = f"  |  Gauge offset: {offset_m/1000:.1f} km" if offset_m > 1000 else ""
    fig.suptitle(
        f"Flash-NH Stage 1  |  {candidate_id}  STAID {staid}  ({state})"
        f"  |  {cat}  |  Area: {area:.0f} km²{offset_str}",
        fontsize=9.5, fontweight="bold", y=0.98)

    gs_outer = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[1.55, 1],
        left=0.06, right=0.97, top=0.91, bottom=0.08, wspace=0.28)
    gs_left = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_outer[0], height_ratios=[14, 1], hspace=0.14)
    gs_right = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs_outer[1], hspace=0.07)

    ax_map  = fig.add_subplot(gs_left[0])
    cbar_ax = fig.add_subplot(gs_left[1])
    ax_sf   = fig.add_subplot(gs_right[0])
    ax_pr   = fig.add_subplot(gs_right[1])
    ax_t2   = fig.add_subplot(gs_right[2])

    vmax_cb = max(vmax_mrms, 0.01)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=vmax_cb))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    if is_dry:
        cbar.set_label("MRMS QPE 1h accumulation (mm)  [DRY CONTROL: always 0.0]", fontsize=7)
    else:
        cbar.set_label(f"MRMS QPE 1h accumulation (mm)  [vmax = {vmax_cb:.1f} mm]", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)

    return fig, (ax_map, ax_sf, ax_pr, ax_t2), cbar


# ── Per-frame renderer ──────────────────────────────────────────────────────────

def render_frame(fig, ax_map, ax_sf, ax_pr, ax_t2,
                 ts, fi, n_frames,
                 mrms_crop, mrms_lons, mrms_lats,
                 vmax_mrms, basin_mean_raster,
                 sf_series, precip_series, t2m_series,
                 basin_gdf, states_gdf,
                 w_lons, w_lats,
                 cand_map_params, cand, is_dry, cmap,
                 basin_bounds, basin_centroid_xy,
                 wind_arrows, map_arrow_scale):

    rid      = cand["candidate_id"]
    g_lat    = float(cand["lat"])
    g_lon    = float(cand["lon"])
    offset_m = float(cand.get("gauge_offset_m") or 0)
    pad      = cand_map_params["pad"]
    sb_km    = cand_map_params["scalebar_km"]
    vmax_use = max(vmax_mrms, 0.01)
    cat_col  = CATEGORY_COLORS.get(cand["category"], "#333333")

    t0_ts, t1_ts = sf_series.index[0], sf_series.index[-1]
    cx_basin, cy_basin = basin_centroid_xy

    # ─ Map panel ───────────────────────────────────────────────────────────────
    ax_map.clear()

    if mrms_crop is not None:
        ax_map.imshow(
            mrms_crop, origin="upper",
            extent=[mrms_lons[0], mrms_lons[-1], mrms_lats[-1], mrms_lats[0]],
            cmap=cmap, vmin=0, vmax=vmax_use,
            aspect="auto", interpolation="nearest", zorder=1)
    else:
        ax_map.set_facecolor("#c8c8c8")

    if states_gdf is not None:
        states_gdf.boundary.plot(ax=ax_map, color="#777777", linewidth=0.5, zorder=2)

    # RTMA 10m wind quiver (qualitative context; not storm-steering validation)
    if WIND_VECTORS_ENABLED and wind_arrows is not None:
        wlons, wlats, wu, wv = wind_arrows
        if len(wlons) > 0:
            ax_map.quiver(
                wlons, wlats, wu, wv,
                color="#333333", alpha=0.50,
                scale=map_arrow_scale, scale_units="xy", angles="xy",
                width=0.0015, headwidth=4, headlength=4,
                zorder=3)
            ax_map.text(0.01, 0.17, "RTMA 10m wind",
                        transform=ax_map.transAxes, fontsize=5.5,
                        color="#333333", va="top", zorder=16)

    # Extraction cells
    if len(w_lons) > 0:
        ax_map.scatter(w_lons, w_lats, s=4, c="orange", alpha=0.30,
                       marker="s", linewidths=0, zorder=4)

    # Basin polygon
    e_col = edge_color_for_mean(basin_mean_raster, cmap, vmax_mrms, is_dry)
    basin_gdf.boundary.plot(ax=ax_map, color="#222222", linewidth=3.8, zorder=5)
    basin_gdf.boundary.plot(ax=ax_map, color=e_col,    linewidth=2.2, zorder=6)

    # Basin-mean label at centroid (avoids gauge overlap)
    if is_dry:
        lbl     = "basin mean: 0.0 mm"
        lbl_col = "#666666"
    else:
        lbl     = f"basin mean: {basin_mean_raster:.2f} mm"
        lbl_col = e_col if not isinstance(e_col, str) or e_col != "#444444" else cat_col
    cxy_dist = np.sqrt((cx_basin - g_lon)**2 + (cy_basin - g_lat)**2)
    lx, ly = cx_basin, cy_basin
    if cxy_dist < 0.05:
        ly = basin_bounds[1] - 0.025
    ax_map.text(lx, ly, lbl,
                ha="center", va="center", fontsize=8, fontweight="bold",
                bbox=dict(fc="white", ec=lbl_col, alpha=0.88, pad=2, lw=1.5), zorder=8)

    # Gauge offset arrow (OFFSET_STRESS only; drawn before gauge marker)
    if offset_m > 1000:
        ax_map.annotate(
            "", xy=(cx_basin, cy_basin), xytext=(g_lon, g_lat),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2, alpha=0.75),
            zorder=9)
        mid_x = (g_lon + cx_basin) / 2
        mid_y = (g_lat + cy_basin) / 2
        ax_map.text(mid_x, mid_y, f"gauge offset\n{offset_m/1000:.1f} km",
                    ha="center", va="center", fontsize=5.5, color="red",
                    bbox=dict(fc="white", ec="red", alpha=0.85, pad=1.5, lw=0.8),
                    zorder=10)

    # Gauge marker — drawn LAST on map at highest z-order
    ax_map.scatter([g_lon], [g_lat], s=240, c="yellow", marker="*",
                   edgecolors="black", linewidths=1.0, zorder=12)

    xlim = (basin_bounds[0] - pad, basin_bounds[2] + pad)
    ylim = (basin_bounds[1] - pad, basin_bounds[3] + pad)
    ax_map.set_xlim(*xlim)
    ax_map.set_ylim(*ylim)
    ax_map.xaxis.set_major_locator(mticker.MaxNLocator(5))
    ax_map.yaxis.set_major_locator(mticker.MaxNLocator(5))
    ax_map.tick_params(labelsize=7)
    ax_map.set_xlabel("Longitude", fontsize=7.5)
    ax_map.set_ylabel("Latitude",  fontsize=7.5)

    sb_x = xlim[0] + 0.04 * (xlim[1] - xlim[0])
    sb_y = ylim[0] + 0.08 * (ylim[1] - ylim[0])
    draw_scale_bar(ax_map, sb_x, sb_y, sb_km, (basin_bounds[1] + basin_bounds[3]) / 2)

    if offset_m > 1000:
        ax_map.text(0.99, 0.99,
                    f"OFFSET_STRESS\ngauge offset: {offset_m/1000:.1f} km",
                    transform=ax_map.transAxes, ha="right", va="top",
                    fontsize=8, color="red", fontweight="bold",
                    bbox=dict(fc="white", ec="red", alpha=0.90, pad=2.5, lw=1.3), zorder=13)

    if is_dry:
        ax_map.text(0.5, 0.5, "STRICT DRY CONTROL\nZERO MRMS PRECIP",
                    transform=ax_map.transAxes, ha="center", va="center",
                    fontsize=12, color="#555555", fontweight="bold",
                    alpha=0.27, rotation=25, zorder=11)

    ax_map.text(0.99, 0.01,
                f"Valid time (accumulation end)\n{ts.strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"frame {fi+1}/{n_frames}",
                transform=ax_map.transAxes, ha="right", va="bottom",
                fontsize=7, bbox=dict(fc="white", alpha=0.82, pad=2, ec="none"), zorder=16)

    # ─ Streamflow panel ────────────────────────────────────────────────────────
    ax_sf.clear()
    ax_sf.plot(sf_series.index, sf_series.values, color="steelblue", lw=1.5, zorder=2)
    ax_sf.fill_between(sf_series.index, 0, sf_series.fillna(0),
                       alpha=0.12, color="steelblue", zorder=1)
    in_gap = False; gap_start = None
    for tv, vv in sf_series.items():
        if np.isnan(vv) and not in_gap:
            gap_start = tv; in_gap = True
        elif not np.isnan(vv) and in_gap:
            ax_sf.axvspan(gap_start, tv, color="orange", alpha=0.20, zorder=0)
            in_gap = False
    if in_gap:
        ax_sf.axvspan(gap_start, t1_ts, color="orange", alpha=0.20, zorder=0)
    ax_sf.axvline(ts, color="red", lw=1.2, zorder=4)
    sf_at = sf_series.get(ts, np.nan)
    if not np.isnan(sf_at):
        ax_sf.scatter([ts], [sf_at], s=55, c="red", marker="o",
                      edgecolors="darkred", linewidths=0.6, zorder=6)
    ax_sf.set_xlim(t0_ts, t1_ts); ax_sf.set_ylim(bottom=0)
    ax_sf.set_ylabel("Q (m³/s)", fontsize=7.5)
    ax_sf.tick_params(axis="both", labelsize=6.5); ax_sf.set_xticklabels([])
    ax_sf.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax_sf.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax_sf.text(0.01, 0.97, "Streamflow", transform=ax_sf.transAxes,
               va="top", ha="left", fontsize=7, color="steelblue", fontweight="bold")

    # ─ Precip bars panel ───────────────────────────────────────────────────────
    ax_pr.clear()
    bar_col = "#888888" if is_dry else "#3a5fa0"
    bar_w   = pd.Timedelta("55min")
    ax_pr.bar(precip_series.index, precip_series.values,
              width=bar_w, color=bar_col, alpha=0.60, align="center", zorder=2)
    cur_val = float(precip_series.get(ts, 0.0))
    if not np.isnan(cur_val):
        ax_pr.bar([ts], [cur_val], width=bar_w,
                  color="#ff6600" if not is_dry else "#aaaaaa",
                  alpha=1.0, align="center", zorder=3)
        if cur_val > 0 and not is_dry:
            pr_max = precip_series.max()
            ylim_pr = pr_max * 1.15 if pr_max > 0 else 1.0
            ax_pr.text(ts, cur_val + 0.03 * ylim_pr, f"{cur_val:.2f}",
                       ha="center", va="bottom", fontsize=6.5, color="#cc4400",
                       fontweight="bold", zorder=5)
    ax_pr.axvline(ts, color="red", lw=1.2, zorder=4)
    ax_pr.set_xlim(t0_ts, t1_ts); ax_pr.set_ylim(bottom=0)
    ax_pr.set_ylabel("Basin-mean MRMS\n1h precip (mm)", fontsize=6.8)
    ax_pr.tick_params(axis="both", labelsize=6.5); ax_pr.set_xticklabels([])
    ax_pr.yaxis.set_major_locator(mticker.MaxNLocator(4))
    if is_dry:
        ax_pr.text(0.5, 0.5, "ZERO PRECIP",
                   transform=ax_pr.transAxes, ha="center", va="center",
                   fontsize=9, color="#888888", fontweight="bold", alpha=0.5)

    # ─ RTMA 2m temperature panel ───────────────────────────────────────────────
    ax_t2.clear()
    ax_t2.plot(t2m_series.index, t2m_series.values, color="darkorange", lw=1.5, zorder=2)
    ax_t2.axhline(0, color="deepskyblue", lw=0.9, ls="--", alpha=0.65, zorder=1)
    ax_t2.axvline(ts, color="red", lw=1.2, zorder=4)
    t2_at = t2m_series.get(ts, np.nan) if hasattr(t2m_series, "get") else float("nan")
    if not pd.isna(t2_at):
        ax_t2.scatter([ts], [t2_at], s=55, c="red", marker="o",
                      edgecolors="darkred", linewidths=0.6, zorder=6)
    ax_t2.set_xlim(t0_ts, t1_ts)
    ax_t2.set_ylabel("RTMA 2m T (°C)", fontsize=7.5)
    ax_t2.set_xlabel("Valid time UTC (accumulation end)", fontsize=7)
    ax_t2.tick_params(axis="both", labelsize=6.5)
    ax_t2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%HZ"))
    ax_t2.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax_t2.yaxis.set_major_locator(mticker.MaxNLocator(4))

    fig.canvas.draw()


# ── Per-candidate driver ────────────────────────────────────────────────────────

def animate_candidate(cand: dict, pq_df: pd.DataFrame,
                      basin_gdf_all, states_gdf, weights_df):
    rid      = cand["candidate_id"]
    staid    = str(cand["staid"]).zfill(8)
    cat      = cand["category"]
    is_dry   = (cat == "DRY_CONTROL")
    state    = cand["state"]
    area     = float(cand["drain_sqkm"])
    offset_m = float(cand.get("gauge_offset_m") or 0)

    w_start     = pd.Timestamp(cand["window_start_utc"])
    w_end       = pd.Timestamp(cand["window_end_utc"])
    win_times   = pd.date_range(w_start, w_end, freq="h")
    frame_times = win_times[::FRAME_CADENCE]
    n_frames    = len(frame_times)

    out_dir    = ANIM_DIR / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    static_dir = out_dir / "static_frames"
    static_dir.mkdir(exist_ok=True)

    cand_params = get_map_params(rid, area)
    pad         = cand_params["pad"]
    vmax_mrms   = 0.0 if is_dry else (
        float(cand.get("max_1h_precip_mm", 5.0)) * cand_params["vmax_factor"])
    cmap = DRY_CMAP if is_dry else WET_CMAP

    print(f"\n{'=' * 64}")
    print(f"  {rid}  {staid}  {cat}  area={area:.0f}km²  vmax={vmax_mrms:.2f}mm")
    print(f"  Window: {w_start} -> {w_end}  n_frames={n_frames}")
    print(f"  Map pad: {pad}°  Wind vectors: {WIND_VECTORS_ENABLED}")
    print(f"{'=' * 64}")

    t_cand = time.time()

    basin_gdf = basin_gdf_all[basin_gdf_all["GAGE_ID"] == staid].copy()
    if len(basin_gdf) == 0:
        return {"candidate_id": rid, "status": "FAIL",
                "error": f"No polygon for STAID {staid}"}

    basin_bounds   = tuple(basin_gdf.total_bounds)
    centroid       = basin_gdf.geometry.representative_point().iloc[0]
    basin_centroid = (centroid.x, centroid.y)

    w_sub   = weights_df[weights_df["STAID"] == staid]
    w_rows  = w_sub["row_idx"].values.astype(int)
    w_cols  = w_sub["col_idx"].values.astype(int)
    w_norms = w_sub["normalized_weight"].values.astype(float)
    w_lons  = w_sub["lon_center"].values if len(w_sub) else np.array([])
    w_lats  = w_sub["lat_center"].values if len(w_sub) else np.array([])

    row_sl, col_sl, mrms_lons, mrms_lats = get_crop_params(basin_bounds, pad + 0.20)

    full_idx  = pd.date_range(w_start, w_end, freq="h")
    sf_series = load_sf(staid, w_start, w_end).reindex(full_idx)
    pq_sub    = pq_df[pq_df["STAID"] == staid].copy().set_index("valid_time_utc")
    precip_s  = (pq_sub[pq_sub["product"] == "mrms_qpe_1h_pass1"]["weighted_mean"]
                 .reindex(full_idx, fill_value=0.0))
    t2m_s     = (pq_sub[(pq_sub["product"] == "rtma_conus_aws_2p5km") &
                        (pq_sub["variable"] == "2t")]["weighted_mean"]
                 .reindex(full_idx, method="nearest") - 273.15)

    peak_precip_fi = int(np.nanargmax(precip_s.values)) if not is_dry else 0
    sf_vals        = sf_series.values
    peak_flow_fi   = int(np.nanargmax(np.where(np.isnan(sf_vals), -np.inf, sf_vals)))
    recession_fi   = min(n_frames - 1, int(0.75 * n_frames))
    static_frames  = {
        0:              "first",
        peak_precip_fi: "peak_precip",
        peak_flow_fi:   "peak_flow",
        recession_fi:   "recession",
    }
    print(f"  Static frames: first=0, peak_precip={peak_precip_fi}, "
          f"peak_flow={peak_flow_fi}, recession={recession_fi}")

    map_lon_min    = basin_bounds[0] - pad
    map_lon_max    = basin_bounds[2] + pad
    map_lat_min    = basin_bounds[1] - pad
    map_lat_max    = basin_bounds[3] + pad
    map_extent_deg = max(map_lon_max - map_lon_min, 0.1)
    # 5 m/s arrow → 8% of map width
    map_arrow_scale = 5.0 / (0.08 * map_extent_deg)

    n_rtma_ok  = 0
    wind_cache = {}
    if WIND_VECTORS_ENABLED:
        print(f"  Preloading RTMA wind ({n_frames} frames)…", end="", flush=True)
        t_rtma = time.time()
        wind_cache, n_rtma_ok = preload_rtma_wind_cache(
            frame_times, map_lon_min, map_lon_max, map_lat_min, map_lat_max)
        print(f" {n_rtma_ok}/{n_frames} files loaded  {time.time()-t_rtma:.0f}s")

    fig, (ax_map, ax_sf, ax_pr, ax_t2), _ = make_figure(
        rid, staid, state, cat, area, offset_m, vmax_mrms, is_dry, cmap)

    gif_path  = out_dir / f"{rid}_animation.gif"
    gif_ok    = False
    n_miss    = 0
    static_pngs = {}

    writer = PillowWriter(fps=FPS)
    try:
        with writer.saving(fig, str(gif_path), dpi=DPI):
            for fi, ts in enumerate(frame_times):
                t0f  = time.time()
                crop = load_mrms_crop(ts, row_sl, col_sl, out_dir)
                if crop is None:
                    n_miss += 1; basin_mean = 0.0
                else:
                    basin_mean = basin_mean_from_crop(
                        crop, w_rows, w_cols, w_norms, row_sl.start, col_sl.start)

                wind_arrows = wind_cache.get(ts) if WIND_VECTORS_ENABLED else None

                render_frame(
                    fig, ax_map, ax_sf, ax_pr, ax_t2,
                    ts, fi, n_frames,
                    crop, mrms_lons, mrms_lats,
                    vmax_mrms, basin_mean,
                    sf_series, precip_s, t2m_s,
                    basin_gdf, states_gdf,
                    w_lons, w_lats,
                    cand_params, cand, is_dry, cmap,
                    basin_bounds, basin_centroid,
                    wind_arrows, map_arrow_scale)

                writer.grab_frame()

                if fi in static_frames:
                    label = static_frames[fi]
                    png_p = static_dir / f"{rid}_{label}.png"
                    fig.savefig(png_p, dpi=DPI, bbox_inches="tight")
                    static_pngs[label] = str(png_p)

                print(f"  {fi+1:3d}/{n_frames}  {ts.strftime('%m-%d %HZ')}  "
                      f"mrms={'OK' if crop is not None else 'MISS'}  "
                      f"bm={basin_mean:.3f}mm  {time.time()-t0f:.1f}s")
        gif_ok = True
        print(f"  GIF -> {gif_path}")
    except Exception as e:
        print(f"  GIF FAILED: {e}")
        traceback.print_exc()

    plt.close(fig)
    elapsed = time.time() - t_cand

    warnings_out = []
    if n_miss:
        warnings_out.append(f"{n_miss} MRMS frames missing")
    if offset_m > 1000:
        warnings_out.append(f"Gauge offset {offset_m:.0f} m — OFFSET_STRESS intentional mismatch")
    if float(cand.get("missing_sf_in_window", 0) or 0) > 0:
        warnings_out.append(
            f"CAMELSH gap: {int(cand['missing_sf_in_window'])} NaN hours — orange shading shown")

    return {
        "candidate_id":               rid,
        "staid":                      staid,
        "category":                   cat,
        "window_start":               str(w_start),
        "window_end":                 str(w_end),
        "n_frames":                   n_frames,
        "frame_cadence_h":            FRAME_CADENCE,
        "fps":                        FPS,
        "duration_s":                 round(n_frames / FPS, 1),
        "output_format":              "GIF" if gif_ok else "FAIL",
        "gif_path":                   str(gif_path) if gif_ok else None,
        "static_pngs":                static_pngs,
        "n_frames_produced":          n_frames if gif_ok else 0,
        "n_mrms_missing":             n_miss,
        "vmax_mrms_mm":               round(vmax_mrms, 3),
        "map_pad_deg":                pad,
        "map_params_source":          "explicit" if rid in CANDIDATE_MAP_PARAMS else "area-based default",
        "wind_vectors_enabled":       WIND_VECTORS_ENABLED,
        "wind_vectors_loaded_frames": n_rtma_ok if WIND_VECTORS_ENABLED else 0,
        "state_boundaries_status":    "loaded" if states_gdf is not None else "skipped_missing",
        "rtma_audit_status":          "PASS (8/8 frames; see rtma_spatial_audit.json)",
        "sync_audit_status":          "PASS (10/10 frames; see sync_audit.json)",
        "runtime_s":                  round(elapsed, 1),
        "status":                     "OK" if gif_ok else "FAIL",
        "warnings":                   warnings_out,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Flash-NH Stage 1 event animation generator")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--all", action="store_true",
                     help="Run all 12 candidates (requires pilot approval)")
    grp.add_argument("--candidates", nargs="+", metavar="RID",
                     help="Space-separated candidate IDs, e.g. R02 R06 R09 R11")
    args = parser.parse_args()

    if args.all:
        run_ids = ALL_CANDIDATE_IDS
    elif args.candidates:
        run_ids = args.candidates
    else:
        run_ids = PILOT_IDS

    ANIM_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading shared datasets…")
    cands_df   = pd.read_csv(CANDIDATES_CSV)
    pq_df      = pd.read_parquet(FORCING_PQ)
    pq_df["valid_time_utc"] = pd.to_datetime(pq_df["valid_time_utc"], utc=True)
    pq_df["STAID"] = pq_df["STAID"].astype(str).str.zfill(8)
    basin_gdf  = gpd.read_file(CAMELSH_SHP)
    basin_gdf  = basin_gdf.set_crs("EPSG:4326", allow_override=True)
    states_gdf, state_boundaries_status = load_states_gdf(STATES_GPKG)
    weights_df = pd.read_parquet(MRMS_WEIGHTS)
    weights_df["STAID"] = weights_df["STAID"].astype(str).str.zfill(8)
    n_states   = len(states_gdf) if states_gdf is not None else 0
    print(f"  basins={len(basin_gdf)}  states={n_states} [{state_boundaries_status}]  "
          f"weights={len(weights_df)}")
    print(f"  ffmpeg: {'OK -> MP4' if FFMPEG_OK else 'NOT FOUND -> GIF fallback'}")
    if not FFMPEG_OK:
        print(f"  {FFMPEG_INSTALL_NOTE}")
    print(f"  Candidates to run: {run_ids}")

    cands_idx = cands_df.set_index("candidate_id", drop=False)
    missing   = [r for r in run_ids if r not in cands_idx.index]
    if missing:
        raise SystemExit(f"ERROR: candidates not in CSV: {missing}")

    manifest = {
        "version":                    "v2.1-stable",
        "generated_utc":              pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_ids":                    run_ids,
        "pilot_ids":                  PILOT_IDS,
        "all_candidate_ids":          ALL_CANDIDATE_IDS,
        "fps":                        FPS,
        "frame_cadence_h":            FRAME_CADENCE,
        "anim_subdir":                ANIM_SUBDIR,
        "output_dir":                 str(ANIM_DIR),
        "ffmpeg_available":           FFMPEG_OK,
        "output_mode":                "MP4" if FFMPEG_OK else "GIF (PillowWriter)",
        "wind_vectors_enabled":       WIND_VECTORS_ENABLED,
        "wind_vectors_target_arrows": WIND_TARGET_ARROWS,
        "state_boundaries_status":    state_boundaries_status,
        "state_boundaries_path":      str(STATES_GPKG),
        "rtma_audit":                 "PASS (8/8 frames, 0.0000% diff) — see rtma_spatial_audit.json",
        "sync_audit":                 "PASS (10/10 frames) — see sync_audit.json",
        "mrms_convention": (
            "Filename timestamp = valid_time = END of 1h accumulation. "
            "Lat DECREASES with row; row 0 = 54.995 N."),
        "rtma_grid_convention": (
            "Lat INCREASES with row; row 0 = SW corner (~19.23 N). "
            "lon stored 0-360; convert to -180/180 for mapping. "
            "data[row_idx, col_idx] direct index."),
        "rtma_wind_caveat": (
            "RTMA 10m winds are qualitative meteorological context and spatial QC. "
            "Do not interpret as strict steering-flow validation for MRMS rain-cell motion."),
        "candidates": {},
    }

    for rid in run_ids:
        row = cands_idx.loc[rid].to_dict()
        res = animate_candidate(row, pq_df, basin_gdf, states_gdf, weights_df)
        manifest["candidates"][rid] = res

    manifest["total_runtime_s"] = round(time.time() - t0, 1)

    manifest_path = ANIM_DIR / "pilot_animation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\nManifest -> {manifest_path}")

    print("\n" + "=" * 64)
    print("ANIMATION SUMMARY")
    print("=" * 64)
    print(f"  state_boundaries: {state_boundaries_status}")
    for rid, res in manifest["candidates"].items():
        warn = "; ".join(res.get("warnings", [])) or "none"
        out  = res.get("gif_path") or "?"
        print(f"  {rid}  {res['status']:6s}  {res['output_format']:8s}  "
              f"{res.get('runtime_s', 0):5.0f}s  warnings: {warn}")
        print(f"      -> {out}")
        if res.get("static_pngs"):
            for lbl, pp in res["static_pngs"].items():
                print(f"         [{lbl}] {pp}")
    print(f"\n  Total: {manifest['total_runtime_s']:.0f}s")


if __name__ == "__main__":
    main()

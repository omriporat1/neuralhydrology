#!/usr/bin/env python3
# MRMS coverage vs. filtered basins
# Deps: pandas geopandas shapely fiona s3fs fsspec matplotlib (optional: xarray)
#   conda install -c conda-forge geopandas s3fs fsspec xarray matplotlib

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
import json

import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Polygon, MultiPoint
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import time
from functools import lru_cache

# Optional AWS access
try:
    import s3fs  # type: ignore
except Exception:
    s3fs = None

# -------------------- CONFIG (edit as needed) --------------------
FILTERED_CSV = r"C:\PhD\Python\neuralhydrology\US_data\basin_attribute_values_filtered.csv"
CAMELS_SHP_DIR = r"S:\hydrolab\ShareData\Caravan\shapefiles\camelsh\shapefiles"  # folder containing the CAMELS basin shapefile(s)

OUT_CSV = r"C:\PhD\Python\neuralhydrology\US_data\mrms_coverage_by_basin.csv"
OUT_FIG = r"C:\PhD\Python\neuralhydrology\US_data\mrms_coverage_map.png"

# If True, try to derive MRMS polygon by inspecting AWS directory (best-effort).
# If False (default), use a conservative MRMS CONUS bbox: lon [-130, -60], lat [20, 55].
USE_AWS_FOR_COVERAGE = True

# MRMS product to probe when USE_AWS_FOR_COVERAGE=True (best-effort only)
MRMS_PRODUCT = "RadarOnly_QPE_01H"

UNMATCHED_IDS_CSV = r"C:\PhD\Python\neuralhydrology\US_data\mrms_unmatched_site_ids.csv"  # NEW
ADD_BASEMAP = True   # set True if you install contextily
MRMS_EDGE_COLOR = "#d95f02"
MRMS_FILL_COLOR = "#fdae6b"
BASIN_COVER_COLOR = "#1b9e77"
BASIN_NOCOV_COLOR = "#e41a1c"
EXACT_MRMS_DOMAIN = True  # set True to derive polygon from one MRMS GRIB2 (slow first run)

MAP_DPI = 240  # increased resolution (was 160)

MAX_S3_LIST_DAYS = 7          # look back this many day folders for a sample file
S3_RETRY = 3                  # retries for listing / file
S3_SLEEP = 1.5                # backoff seconds
FALLBACK_TILE_PROVIDERS = [
    "CartoDB.Positron",
    "OpenStreetMap.Mapnik",
    "Stamen.TerrainBackground"
]

EXACT_DOMAIN_CACHE = r"C:\PhD\Python\neuralhydrology\US_data\mrms_exact_domain.geojson"
MAX_LOOKBACK_DAYS = 10   # how many past days to attempt when searching for a sample GRIB2 (if user enables exact)
# ----------------------------------------------------------------

# Candidate ID columns that might appear in the CAMELS shapefiles
ID_CANDIDATES = [
    "site_id", "STAID", "staid", "GAGE_ID", "gage_id", "usgs_id", "site_no",
    "site_no_txt", "gageid", "station_id", "hru_id", "gageid_", "usgs"
]

CRS_WGS84 = "EPSG:4326"
CRS_ALBERS_CONUS = "EPSG:5070"  # equal-area for area computations


def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_filtered_site_ids(csv_path: str | Path) -> List[str]:
    df = pd.read_csv(csv_path)
    # Accept any reasonable ID column name
    candidates = [c for c in df.columns if c.lower() in {"site_id", "staid", "gage_id", "site_no", "usgs_id"}]
    if not candidates:
        # fallback to a column literally named site_id
        candidates = ["site_id"] if "site_id" in df.columns else []
    if not candidates:
        raise SystemExit(f"No site ID column found in {csv_path}. Expected one of: site_id, staid, gage_id, site_no, usgs_id")
    col = candidates[0]
    ids = (
        df[col].astype(str)
        .str.extract(r"(\d{7,8})")[0]      # accept 7 or 8 digits
        .dropna()
        .str.zfill(8)                      # pad to 8
        .tolist()
    )
    return sorted(set(ids))


def find_camels_shapefile(camels_dir: str | Path) -> Path:
    # Heuristic: prefer files with 'basin'/'camels' in the name
    shp_files = sorted(Path(camels_dir).rglob("*.shp"))
    if not shp_files:
        raise SystemExit(f"No shapefiles found under {camels_dir}")
    ranked = sorted(shp_files, key=lambda p: (0 if ("basin" in p.stem.lower() or "camels" in p.stem.lower()) else 1, len(p.stem)))
    return ranked[0]


def detect_id_column(gdf: gpd.GeoDataFrame, desired_ids: List[str]) -> Optional[str]:
    cols = list(gdf.columns)
    # exact candidates
    for c in ID_CANDIDATES:
        if c in cols:
            return c
    # case-insensitive match
    lower_map = {c.lower(): c for c in cols}
    for c in ID_CANDIDATES:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    # try to find a column where many values look like 8-digit IDs
    best_col, best_hits = None, -1
    for c in cols:
        try:
            vals = gdf[c].astype(str).str.extract(r"(\d{8})")[0]
            hits = vals.notna().sum()
            if hits > best_hits:
                best_col, best_hits = c, hits
        except Exception:
            continue
    return best_col


def find_all_shapefiles(camels_dir: str | Path) -> List[Path]:
    shp_files = sorted(Path(camels_dir).rglob("*.shp"))
    if not shp_files:
        raise SystemExit(f"No shapefiles found under {camels_dir}")
    return shp_files


def load_camels_basins(camels_dir: str | Path, site_ids: List[str]) -> gpd.GeoDataFrame:
    shp_files = find_all_shapefiles(camels_dir)
    parts = []
    for shp in shp_files:
        try:
            gdf = gpd.read_file(shp)
        except Exception as e:
            print(f"[warn] could not read {shp.name}: {e}")
            continue
        if gdf.crs is None:
            gdf = gdf.set_crs(CRS_WGS84)
        id_col = detect_id_column(gdf, site_ids)
        if not id_col:
            continue
        gdf["_site_id"] = (
            gdf[id_col].astype(str)
            .str.extract(r"(\d{7,8})")[0]
            .dropna()
            .str.zfill(8)
        )
        gsub = gdf.dropna(subset=["_site_id"])[["_site_id", "geometry"]].rename(columns={"_site_id": "site_id"})
        parts.append(gsub)
    if not parts:
        raise SystemExit("No shapefile produced usable site IDs.")
    all_basins = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["site_id"])
    all_basins = gpd.GeoDataFrame(all_basins, geometry="geometry", crs=CRS_WGS84)
    sel = all_basins[all_basins["site_id"].isin(site_ids)].copy()
    if sel.empty:
        raise SystemExit("No basins matched the filtered site_ids.")
    sel["geometry"] = sel["geometry"].buffer(0)
    print(f"[info] Loaded {len(sel)} basin polygons matching filtered IDs from {len(parts)} shapefile(s).")
    return sel


def mrms_polygon_from_bbox() -> gpd.GeoDataFrame:
    # Conservative CONUS coverage (approximate MRMS mosaic domain)
    poly = box(-130.0, 20.0, -60.0, 55.0)
    return gpd.GeoDataFrame({"name": ["MRMS_CONUS_bbox"]}, geometry=[poly], crs=CRS_WGS84)


def _locate_product_base(product: str) -> Optional[str]:
    """Return s3 prefix to product directory (noaa-mrms-pds/CONUS/... or root)."""
    if s3fs is None:
        return None
    fs = s3fs.S3FileSystem(anon=True)
    candidates = [
        f"noaa-mrms-pds/CONUS/{product}",
        f"noaa-mrms-pds/CONUS/{product}_00.00",
        f"noaa-mrms-pds/{product}",
        f"noaa-mrms-pds/{product}_00.00",
    ]
    for c in candidates:
        try:
            if fs.exists(c):
                print(f"[info] MRMS product base: s3://{c}")
                return c
        except Exception as e:
            print(f"[warn] exists check failed for {c}: {e}")
    return None


def _list_mrms_day_dirs(base: str) -> List[str]:
    fs = s3fs.S3FileSystem(anon=True)
    try:
        # children are YYYYMMDD directories (and maybe other artifacts)
        dirs = [p for p in fs.ls(base) if p.split("/")[-1].isdigit()]
        return sorted(dirs, reverse=True)
    except Exception as e:
        print(f"[warn] listing days under {base} failed: {e}")
        return []


def _list_files(fs, prefix: str) -> List[str]:
    try:
        return fs.ls(prefix)
    except Exception:
        return []


def _find_sample_grib2_hier(base: str, lookback_days: int) -> Optional[str]:
    """Search recent days for a GRIB2 file traversing multiple directory layouts."""
    if s3fs is None:
        return None
    fs = s3fs.S3FileSystem(anon=True)
    today_utc = datetime.utcnow()
    for delta in range(lookback_days):
        day = today_utc - timedelta(days=delta)
        for date_path in _possible_date_paths(base, day):
            # Two possibilities: files directly in date_path or under hour subfolders
            if not fs.exists(date_path):
                continue
            direct_files = _list_files(fs, date_path)
            # gather GRIB2 directly
            gribs_direct = [f for f in direct_files if f.lower().endswith(".grib2")]
            if gribs_direct:
                gribs_direct.sort()
                print(f"[info] Using sample GRIB2 (direct): s3://{gribs_direct[0]}")
                return gribs_direct[0]
            # hour subfolders (00..23)
            hour_dirs = [f for f in direct_files if f.split("/")[-1].isdigit() and len(f.split("/")[-1]) in (2,)]
            for hdir in hour_dirs:
                hour_files = _list_files(fs, hdir)
                gribs_hour = [f for f in hour_files if f.lower().endswith(".grib2")]
                if gribs_hour:
                    gribs_hour.sort()
                    print(f"[info] Using sample GRIB2 (hour): s3://{gribs_hour[0]}")
                    return gribs_hour[0]
    return None


def mrms_polygon_exact(product: str, force_refresh: bool = False) -> Optional[gpd.GeoDataFrame]:
    """Derive exact MRMS convex hull polygon from one recent GRIB2 file; cache result."""
    if not force_refresh and Path(EXACT_DOMAIN_CACHE).is_file():
        try:
            gdf = gpd.read_file(EXACT_DOMAIN_CACHE)
            if not gdf.empty:
                print(f"[info] Loaded cached MRMS exact domain: {EXACT_DOMAIN_CACHE}")
                return gdf.to_crs(CRS_WGS84)
        except Exception:
            pass

    if s3fs is None:
        print("[warn] s3fs not installed for exact MRMS domain.")
        return None
    try:
        import xarray as xr  # type: ignore
    except Exception:
        print("[warn] xarray/cfgrib not installed for exact MRMS domain.")
        return None
    base = _locate_product_base(product)
    if base is None:
        print("[warn] product base not found for exact domain.")
        return None

    key = _find_sample_grib2_hier(base, MAX_LOOKBACK_DAYS)
    if key is None:
        print("[warn] No GRIB2 found in last", MAX_LOOKBACK_DAYS, "days for exact domain.")
        return None

    fs = s3fs.S3FileSystem(anon=True)
    succeeded = False
    for attempt in range(1, S3_RETRY + 1):
        try:
            with fs.open(key, "rb") as f:
                ds = xr.open_dataset(f, engine="cfgrib")
            succeeded = True
            break
        except Exception as e:
            if attempt == S3_RETRY:
                print(f"[warn] final attempt opening GRIB2 failed: {e}")
                return None
            time.sleep(S3_SLEEP * attempt)
    if not succeeded:
        return None

    # Identify lat/lon (support common names)
    lat_candidates = [n for n in ds.variables if "lat" == n.lower() or "latitude" in n.lower()]
    lon_candidates = [n for n in ds.variables if "lon" == n.lower() or "longitude" in n.lower()]
    if not lat_candidates or not lon_candidates:
        print("[warn] Could not find lat/lon variables.")
        return None
    lat = ds[lat_candidates[0]].values
    lon = ds[lon_candidates[0]].values
    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    valid = ~np.isnan(lat) & ~np.isnan(lon)
    lonv, latv = lon[valid], lat[valid]
    if lonv.size == 0:
        print("[warn] No valid coordinates.")
        return None
    # Subsample if huge
    if lonv.size > 500000:
        step = int(lonv.size / 500000) + 1
        lonv = lonv[::step]; latv = latv[::step]
    hull = MultiPoint([tuple(p) for p in np.column_stack([lonv, latv])]).convex_hull
    gdf = gpd.GeoDataFrame({"name": ["MRMS_exact"]}, geometry=[hull], crs=CRS_WGS84)
    try:
        gdf.to_file(EXACT_DOMAIN_CACHE, driver="GeoJSON")
        print(f"[info] Cached exact MRMS polygon -> {EXACT_DOMAIN_CACHE}")
    except Exception as e:
        print(f"[warn] could not cache polygon: {e}")
    print("[info] Derived exact MRMS convex hull polygon.")
    return gdf


def mrms_polygon_from_aws(product: str = MRMS_PRODUCT, exact: bool = False, force_refresh: bool = False) -> Optional[gpd.GeoDataFrame]:
    if exact:
        exact_poly = mrms_polygon_exact(product, force_refresh=force_refresh)
        if exact_poly is not None:
            return exact_poly
    if s3fs is None:
        print("[warn] s3fs is not installed. Falling back to bbox.")
        return None
    base = _locate_product_base(product)
    if base is None:
        print(f"[warn] Could not find product directory for {product}.")
        return None
    print(f"[info] Using product base: s3://{base} (bbox fallback).")
    return mrms_polygon_from_bbox()


def compute_coverage(basins_wgs84: gpd.GeoDataFrame, mrms_poly_wgs84: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compute intersection area and coverage flags."""
    # Reproject to equal-area for area calcs
    basins_eq = basins_wgs84.to_crs(CRS_ALBERS_CONUS)
    mrms_eq = mrms_poly_wgs84.to_crs(CRS_ALBERS_CONUS)

    # Merge MRMS polygons to one geometry
    mrms_union = unary_union(mrms_eq.geometry.values)
    areas_basin = basins_eq.geometry.area  # m^2
    inter_geom = basins_eq.geometry.intersection(mrms_union)
    areas_inter = inter_geom.area

    df = pd.DataFrame({
        "site_id": basins_wgs84["site_id"].values,
        "basin_area_km2": (areas_basin.values / 1e6),
        "covered_area_km2": (areas_inter.values / 1e6),
    })
    df["coverage_frac"] = (df["covered_area_km2"] / df["basin_area_km2"]).where(df["basin_area_km2"] > 0.0, 0.0)
    df["covered"] = df["coverage_frac"] > 0.0
    return df


def get_us_states() -> Optional[gpd.GeoDataFrame]:
    """Download (cached by fiona) Natural Earth admin-1 states/provinces and filter USA."""
    try:
        url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip"
        gdf = gpd.read_file(url)
        gdf = gdf[(gdf["admin"] == "United States of America") & (gdf["postal"] != "PR")]
        return gdf.to_crs(CRS_WGS84)
    except Exception as e:
        print(f"[warn] could not load states layer: {e}")
        return None


def report_unmatched(site_ids: List[str], basins: gpd.GeoDataFrame, out_csv: str):
    matched = set(basins["site_id"])
    missing = sorted(set(site_ids) - matched)
    if not missing:
        print("[info] All filtered site_ids matched a basin polygon.")
        return
    pd.DataFrame({"site_id": missing}).to_csv(out_csv, index=False)
    print(f"[warn] {len(missing)} site_ids had no polygon match. Sample: {missing[:15]}")
    print(f"[warn] Saved unmatched list -> {out_csv}")


def _add_basemap_with_fallback(ax, crs):
    try:
        import contextily as ctx  # type: ignore
    except Exception:
        print("[info] contextily not installed; skipping basemap.")
        return
    for prov in FALLBACK_TILE_PROVIDERS:
        try:
            provider = eval(f"ctx.providers.{prov}") if "." in prov else getattr(ctx.providers, prov)
            ctx.add_basemap(ax, crs=crs, source=provider, attribution=False)
            print(f"[info] basemap OK: {prov}")
            return
        except Exception as e:
            print(f"[warn] basemap provider {prov} failed: {e}")
    print("[warn] all basemap providers failed; continuing without tiles.")


def make_map(basins: gpd.GeoDataFrame,
             mrms_poly: gpd.GeoDataFrame,
             coverage_df: pd.DataFrame,
             out_png: str) -> None:
    gdf = basins.merge(coverage_df[["site_id", "covered"]], on="site_id", how="left")

    states = get_us_states()
    use_basemap = False
    if ADD_BASEMAP:
        try:
            import contextily as ctx  # type: ignore
            use_basemap = True
        except Exception:
            use_basemap = False
            print("[info] contextily not installed; skipping basemap.")

    # Compute extent from basins (fallback to MRMS bbox)
    if not gdf.empty:
        minx, miny, maxx, maxy = gdf.total_bounds
        # pad
        dx, dy = maxx - minx, maxy - miny
        if dx == 0 or dy == 0:
            dx = dy = 1
        padx, pady = dx * 0.1, dy * 0.1
        minx -= padx; maxx += padx; miny -= pady; maxy += pady
        # constrain within CONUS default
        minx = max(minx, -130); maxx = min(maxx, -60)
        miny = max(miny, 20);   maxy = min(maxy, 55)
    else:
        minx, miny, maxx, maxy = -130, 20, -60, 55

    if use_basemap:
        # Reproject to Web Mercator
        gdf_merc = gdf.to_crs(3857)
        mrms_merc = mrms_poly.to_crs(3857)
        states_merc = states.to_crs(3857) if states is not None else None
        # transform extent
        tmp = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs=CRS_WGS84).to_crs(3857).total_bounds
        minx_m, miny_m, maxx_m, maxy_m = tmp
        fig, ax = plt.subplots(figsize=(11, 8), dpi=MAP_DPI)
        if states_merc is not None:
            states_merc.boundary.plot(ax=ax, linewidth=0.4, color="#555555", zorder=1)
        mrms_merc.boundary.plot(ax=ax, color=MRMS_EDGE_COLOR, linewidth=1.2, zorder=2)
        mrms_merc.plot(ax=ax, facecolor=MRMS_FILL_COLOR, alpha=0.08, zorder=1.5)
        covered_gdf = gdf_merc[gdf_merc["covered"] == True]
        not_cov_gdf = gdf_merc[gdf_merc["covered"] != True]
        if not covered_gdf.empty:
            covered_gdf.plot(ax=ax, facecolor=BASIN_COVER_COLOR, edgecolor="#084c3a",
                             linewidth=0.25, alpha=0.6, zorder=3)
        if not not_cov_gdf.empty:
            not_cov_gdf.plot(ax=ax, facecolor=BASIN_NOCOV_COLOR, edgecolor="#67000d",
                             linewidth=0.25, alpha=0.55, zorder=3)
        try:
            ctx.add_basemap(ax, crs=covered_gdf.crs if not covered_gdf.empty else mrms_merc.crs,
                            source=ctx.providers.Stamen.TonerLite, attribution=False)
        except Exception as e:
            print(f"[warn] basemap failed: {e}")
        ax.set_xlim(minx_m, maxx_m)
        ax.set_ylim(miny_m, maxy_m)
        ax.set_axis_off()
    else:
        fig, ax = plt.subplots(figsize=(11, 8), dpi=MAP_DPI)
        if states is not None and not states.empty:
            states.boundary.plot(ax=ax, linewidth=0.4, color="#777777", zorder=1)
        mrms_poly.boundary.plot(ax=ax, color=MRMS_EDGE_COLOR, linewidth=1.2, zorder=2)
        mrms_poly.plot(ax=ax, facecolor=MRMS_FILL_COLOR, alpha=0.07, zorder=1.5)
        if not gdf.empty:
            covered_gdf = gdf[gdf["covered"] == True]
            not_cov_gdf = gdf[gdf["covered"] != True]
            if not covered_gdf.empty:
                covered_gdf.plot(ax=ax, facecolor=BASIN_COVER_COLOR, edgecolor="#084c3a",
                                 linewidth=0.25, alpha=0.55, zorder=3)
            if not not_cov_gdf.empty:
                not_cov_gdf.plot(ax=ax, facecolor=BASIN_NOCOV_COLOR, edgecolor="#67000d",
                                 linewidth=0.25, alpha=0.55, zorder=3)
        ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.grid(ls=":", alpha=0.25)

    # Custom legend
    legend_handles = [
        Patch(facecolor=MRMS_FILL_COLOR, edgecolor=MRMS_EDGE_COLOR, alpha=0.3, label="MRMS domain"),
        Patch(facecolor=BASIN_COVER_COLOR, edgecolor="#084c3a", alpha=0.55, label="Covered basin"),
    ]
    if not gdf.empty and (gdf["covered"] != True).any():
        legend_handles.append(Patch(facecolor=BASIN_NOCOV_COLOR, edgecolor="#67000d", alpha=0.55, label="Not covered"))
    ax.legend(handles=legend_handles, loc="lower left", frameon=True)

    ax.set_title("MRMS coverage vs. filtered CAMELS basins", pad=10)
    plt.tight_layout()
    _ensure_parent(out_png)
    plt.savefig(out_png)
    print(f"[saved] map -> {out_png}")
    plt.close(fig)


def _parse_cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default=MRMS_PRODUCT, help="MRMS product name (e.g., RadarOnly_QPE_01H)")
    ap.add_argument("--exact-domain", action="store_true", help="Derive exact MRMS domain from a recent GRIB2")
    ap.add_argument("--refresh-exact", action="store_true", help="Force re-download even if cached polygon exists")
    ap.add_argument("--no-basemap", action="store_true", help="Disable basemap even if contextily installed")
    ap.add_argument("--dpi", type=int, default=MAP_DPI, help="Figure DPI")
    ap.add_argument("--lookback-days", type=int, default=MAX_LOOKBACK_DAYS, help="Look back this many days for sample GRIB2")
    return ap.parse_args()

def main():
    global ADD_BASEMAP, MAP_DPI, MRMS_PRODUCT, MAX_LOOKBACK_DAYS
    args = _parse_cli()
    MRMS_PRODUCT = args.product
    ADD_BASEMAP = not args.no_basemap
    MAP_DPI = args.dpi
    MAX_LOOKBACK_DAYS = args.lookback_days

    ids = read_filtered_site_ids(FILTERED_CSV)
    print(f"[info] Filtered site_id count: {len(ids)} (from {FILTERED_CSV})")
    basins = load_camels_basins(CAMELS_SHP_DIR, ids)
    report_unmatched(ids, basins, UNMATCHED_IDS_CSV)
    mrms_poly = mrms_polygon_from_aws(MRMS_PRODUCT, exact=args.exact_domain, force_refresh=args.refresh_exact)
    if mrms_poly is None:
        mrms_poly = mrms_polygon_from_bbox()
    df_cov = compute_coverage(basins, mrms_poly)
    _ensure_parent(OUT_CSV)
    df_cov.sort_values(["covered", "site_id"], ascending=[False, True]).to_csv(OUT_CSV, index=False)
    print(f"[saved] coverage CSV -> {OUT_CSV}")
    print(df_cov["covered"].value_counts())
    make_map(basins, mrms_poly, df_cov, OUT_FIG)


if __name__ == "__main__":
    main()
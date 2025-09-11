#!/usr/bin/env python3
# MRMS coverage vs. filtered basins
# Deps: pandas geopandas shapely fiona s3fs fsspec matplotlib (optional: xarray)
#   conda install -c conda-forge geopandas s3fs fsspec xarray matplotlib

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt

# Optional AWS access
try:
    import s3fs  # type: ignore
except Exception:
    s3fs = None

# -------------------- CONFIG (edit as needed) --------------------
FILTERED_CSV = r"C:\PhD\Python\neuralhydrology\US_data\basin_attribute_values_filtered.csv"
CAMELS_SHP_DIR = r"S:\hydrolab\ShareData\Caravan\shapefiles\camels"  # folder containing the CAMELS basin shapefile(s)

OUT_CSV = r"C:\PhD\Python\neuralhydrology\US_data\mrms_coverage_by_basin.csv"
OUT_FIG = r"C:\PhD\Python\neuralhydrology\US_data\mrms_coverage_map.png"

# If True, try to derive MRMS polygon by inspecting AWS directory (best-effort).
# If False (default), use a conservative MRMS CONUS bbox: lon [-130, -60], lat [20, 55].
USE_AWS_FOR_COVERAGE = False

# MRMS product to probe when USE_AWS_FOR_COVERAGE=True (best-effort only)
MRMS_PRODUCT = "RadarOnly_QPE_01H"
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
    ids = df[col].astype(str).str.extract(r"(\d{8})")[0].dropna().tolist()
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


def load_camels_basins(camels_dir: str | Path, site_ids: List[str]) -> gpd.GeoDataFrame:
    shp_path = find_camels_shapefile(camels_dir)
    print(f"[info] Using shapefile: {shp_path}")
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        print("[warn] shapefile has no CRS; assuming WGS84 (EPSG:4326)")
        gdf = gdf.set_crs(CRS_WGS84)
    id_col = detect_id_column(gdf, site_ids)
    if not id_col:
        raise SystemExit(f"Could not detect an ID column in {shp_path}. Checked: {ID_CANDIDATES}")
    print(f"[info] Detected ID column: {id_col}")

    # Normalize IDs to 8-digit strings for join/filter
    gdf["_site_id"] = gdf[id_col].astype(str).str.extract(r"(\d{8})")[0]
    gdf = gdf.dropna(subset=["_site_id"])
    gdf["_site_id"] = gdf["_site_id"].str.zfill(8)

    gdf_sel = gdf[gdf["_site_id"].isin(site_ids)].copy()
    if gdf_sel.empty:
        raise SystemExit("No basins matched the site IDs from the filtered CSV. Check the shapefile and ID column.")
    gdf_sel = gdf_sel.to_crs(CRS_WGS84)
    gdf_sel["geometry"] = gdf_sel["geometry"].buffer(0)  # fix minor topology
    gdf_sel = gdf_sel[["_site_id", "geometry"]].rename(columns={"_site_id": "site_id"})
    print(f"[info] Loaded {len(gdf_sel)} basins matching filtered site IDs")
    return gdf_sel


def mrms_polygon_from_bbox() -> gpd.GeoDataFrame:
    # Conservative CONUS coverage (approximate MRMS mosaic domain)
    poly = box(-130.0, 20.0, -60.0, 55.0)
    return gpd.GeoDataFrame({"name": ["MRMS_CONUS_bbox"]}, geometry=[poly], crs=CRS_WGS84)


def mrms_polygon_from_aws(product: str = MRMS_PRODUCT) -> Optional[gpd.GeoDataFrame]:
    """Best-effort attempt to derive MRMS coverage from AWS. Falls back to None if not possible."""
    if s3fs is None:
        print("[warn] s3fs is not installed. Falling back to bbox.")
        return None
    try:
        fs = s3fs.S3FileSystem(anon=True)
        # List top-level keys to hint structure (does not download data)
        # NOTE: The PDS layout can change; we only probe a few known prefixes and pick the first that exists.
        prefixes = [
            f"noaa-mrms-pds/{product}",
            f"noaa-mrms-pds/CONUS/{product}",
            f"noaa-mrms-pds/{product}_00.00",  # some products include sampling resolution in dir name
            f"noaa-mrms-pds/CONUS/{product}_00.00",
        ]
        chosen = None
        for p in prefixes:
            if fs.exists(p):
                chosen = p
                break
        if chosen is None:
            print(f"[warn] Could not locate product directory for {product} in noaa-mrms-pds. Falling back to bbox.")
            return None

        # We won’t download full data. Instead, return a very rough, known domain polygon (same result as bbox).
        # If you want exact outline from a grid’s valid-mask, we can add a netCDF/GRIB read here.
        print(f"[info] Found AWS path: s3://{chosen}. Using standard CONUS bbox as coverage for now.")
        return mrms_polygon_from_bbox()
    except Exception as e:
        print(f"[warn] AWS probe failed: {e}. Falling back to bbox.")
        return None


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


def make_map(basins: gpd.GeoDataFrame, mrms_poly: gpd.GeoDataFrame, coverage_df: pd.DataFrame, out_png: str) -> None:
    # Join coverage back to basins
    gdf = basins.merge(coverage_df[["site_id", "covered"]], on="site_id", how="left")

    # USA outline (Natural Earth)
    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        usa = world[world["name"] == "United States of America"].to_crs(CRS_WGS84)
    except Exception:
        usa = None

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

    if usa is not None and not usa.empty:
        usa.boundary.plot(ax=ax, linewidth=0.6, color="gray", zorder=1)

    # MRMS domain
    mrms_poly.boundary.plot(ax=ax, color="#d95f02", linewidth=1.2, label="MRMS domain", zorder=2)
    mrms_poly.plot(ax=ax, color="#d95f02", alpha=0.08, zorder=1.5)

    # Basins
    if not gdf.empty:
        gdf[gdf["covered"] == True].plot(ax=ax, color="#1b9e77", alpha=0.35, linewidth=0.2, edgecolor="#1b9e77", label="Covered basins", zorder=3)
        gdf[gdf["covered"] != True].plot(ax=ax, color="#e41a1c", alpha=0.35, linewidth=0.2, edgecolor="#e41a1c", label="Not covered", zorder=3)

    ax.set_title("MRMS coverage vs. filtered CAMELS basins")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="lower left", frameon=True)
    ax.set_xlim(-130, -60)
    ax.set_ylim(20, 55)
    ax.grid(True, alpha=0.2)

    _ensure_parent(out_png)
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"[saved] map -> {out_png}")
    try:
        plt.close(fig)
    except Exception:
        pass


def main():
    # 1) Read site IDs
    ids = read_filtered_site_ids(FILTERED_CSV)
    print(f"[info] Filtered site_id count: {len(ids)} (from {FILTERED_CSV})")

    # 2) Load basins subset by site_id
    basins = load_camels_basins(CAMELS_SHP_DIR, ids)  # WGS84

    # 3) Build MRMS coverage polygon
    mrms_poly = None
    if USE_AWS_FOR_COVERAGE:
        mrms_poly = mrms_polygon_from_aws(MRMS_PRODUCT)
    if mrms_poly is None:
        mrms_poly = mrms_polygon_from_bbox()

    # 4) Compute coverage metrics
    df_cov = compute_coverage(basins, mrms_poly)
    _ensure_parent(OUT_CSV)
    df_cov.sort_values(["covered", "site_id"], ascending=[False, True]).to_csv(OUT_CSV, index=False)
    print(f"[saved] coverage CSV -> {OUT_CSV}")
    print(df_cov["covered"].value_counts())

    # 5) Map
    make_map(basins, mrms_poly, df_cov, OUT_FIG)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
r"""
Plot filtered basins on a CONUS background, optionally colored by an attribute.

- Reads basin IDs from a filtered CSV (default: basin_attribute_values_filtered.csv).
- Loads basin polygons from a directory of shapefiles (recursively).
- Matches polygons to IDs and produces a map PNG.
- Optional basemap (WebMercator) if contextily is available.
- Optional coloring by an attribute column from the filtered CSV.
- Produces multiple maps colored by interesting attributes (climate, topography, etc).

Run (PowerShell examples):
    py C:\PhD\Python\neuralhydrology\US_data\plot_filtered_basins_map.py

    py C:\PhD\Python\neuralhydrology\US_data\plot_filtered_basins_map.py \
        --color-col p-mean --dpi 200 --out C:\PhD\Python\neuralhydrology\US_data\filtered_basins_map.png
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# Settings (edit here)
# -----------------------
FILTERED_CSV_DEFAULT = r"C:\PhD\Python\neuralhydrology\US_data\basin_attribute_values_filtered.csv"
SHP_DIR_DEFAULT      = r"S:\hydrolab\ShareData\CAMELSH\shapefiles"
OUT_PNG_DEFAULT      = r"C:\PhD\Python\neuralhydrology\US_data\filtered_basins_map.png"
UNMATCHED_CSV_DEFAULT= r"C:\PhD\Python\neuralhydrology\US_data\filtered_basins_unmatched.csv"

# ID column candidates in shapefiles
ID_CANDIDATES = [
    "site_id", "STAID", "staid", "gage_id", "GAGE_ID", "gauge_id",
    "usgs_id", "USGS_ID", "site_no", "site_no_txt", "site_no_txt_1",
    "gageid", "station_id", "Station_ID", "COMID", "HYBAS_ID"
]

CONUS_BOUNDS = (-125.0, 24.0, -66.5, 49.0)  # lon_min, lat_min, lon_max, lat_max

# Interesting attributes to map (plot_name, column_name, cmap, units)
INTERESTING_ATTRIBUTES = [
    ("Mean Annual Precipitation", "p-mean", "Blues", "mm/year"),
    ("Potential Evapotranspiration", "pet-mean", "Oranges", "mm/year"),
    ("Aridity Index", "aridity-index", "RdYlGn_r", "dimensionless"),
    ("Mean Elevation (m)", "elev_mean_m_basin", "terrain", "meters"),
    ("Basin Slope (%)", "slope_pct", "copper", "%"),
    ("Drainage Area (km²)", "area_km2", "viridis", "km²"),
    ("Fraction Snow Precipitation", "frac-snow", "Purples", "fraction"),
]


def normalize_id(val: object) -> Optional[str]:
    """Normalize a site/basin ID to an 8-digit zero-padded numeric string if possible.
    Falls back to a stripped string if not purely numeric.
    """
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    # keep only digits; if no digits, return stripped string
    digits = re.sub(r"\D", "", s)
    if digits:
        return digits.zfill(8)
    return s


def read_filtered_ids(csv_path: str, id_col: str = "site_id") -> pd.DataFrame:
    """Read filtered CSV and add normalized site_id_norm column.
    Returns the full DataFrame with all columns plus site_id_norm.
    """
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns]
    if id_col not in cols:
        for cand in ID_CANDIDATES:
            if cand in cols:
                id_col = cand
                break
    if id_col not in df.columns:
        raise SystemExit(f"Could not find an ID column in {csv_path}. Tried: {['site_id']+ID_CANDIDATES}")
    # Add normalized ID column to the full dataframe (like US_basins_statistics.py)
    df["site_id_norm"] = df[id_col].apply(normalize_id)
    df = df.dropna(subset=["site_id_norm"])
    return df


def find_shapefiles(shp_dir: str) -> List[Path]:
    root = Path(shp_dir)
    if not root.exists():
        raise SystemExit(f"Shapefile directory does not exist: {shp_dir}")
    return list(root.rglob("*.shp"))


def load_polygons(shp_paths: List[Path]) -> gpd.GeoDataFrame:
    gdfs = []
    for shp in shp_paths:
        try:
            gdf = gpd.read_file(shp)
            if gdf.empty or "geometry" not in gdf.columns:
                continue
            # detect ID column
            print(f"[debug] {shp.name} columns: {list(gdf.columns)}")
            found = None
            for cand in ID_CANDIDATES:
                if cand in gdf.columns:
                    found = cand
                    break
            if found is None:
                # if none found, skip
                print(f"[warn] {shp.name}: no ID column found in {ID_CANDIDATES}")
                continue
            gdf = gdf[[found, "geometry"]].copy()
            gdf["site_id_norm"] = gdf[found].apply(normalize_id)
            gdf = gdf.dropna(subset=["site_id_norm"]).drop_duplicates("site_id_norm")
            # ensure valid geometries
            gdf = gdf[gdf["geometry"].notna()].copy()
            gdf["geometry"] = gdf["geometry"].buffer(0)
            gdfs.append(gdf[["site_id_norm", "geometry"]])
        except Exception:
            # ignore shapefiles that fail to read
            continue
    if not gdfs:
        raise SystemExit("No polygons loaded from shapefiles. Check directory or ID columns.")
    merged = pd.concat(gdfs, ignore_index=True)
    # deduplicate polygons per ID (keep first)
    merged = merged.drop_duplicates(subset=["site_id_norm"])  # type: ignore
    return gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")


def conus_polygon() -> gpd.GeoDataFrame:
    xmin, ymin, xmax, ymax = CONUS_BOUNDS
    poly = box(xmin, ymin, xmax, ymax)
    return gpd.GeoDataFrame(pd.DataFrame({"name": ["CONUS"]}), geometry=[poly], crs="EPSG:4326")


def load_states_boundary() -> Optional[gpd.GeoDataFrame]:
    """Load US state boundaries from Natural Earth 10m data."""
    try:
        # Load Natural Earth 10m states boundaries
        url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip"
        states = gpd.read_file(url)
        # Filter to USA only
        states = states[states['admin'] == 'United States of America'].copy()
        states = states[states.geometry.is_valid]
        if not states.empty:
            print(f"[info] Loaded {len(states)} US state boundaries from Natural Earth")
            return states
    except Exception as e:
        print(f"[warn] Could not load from Natural Earth online ({e}); trying local method")
    
    # Fallback: try to use locally cached naturalearth data
    try:
        states = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        states = states[states['iso_a3'] == 'USA'].copy()
        # Dissolve into a single US polygon
        states = gpd.GeoDataFrame(
            [{'geometry': states.dissolve().geometry.values[0], 'name': 'USA'}],
            crs=states.crs
        )
        print(f"[info] Loaded US outline from local naturalearth data")
        return states
    except Exception as e:
        print(f"[warn] Could not load states boundary ({e})")
        return None


def build_plot_gdf(
    gdf_poly: gpd.GeoDataFrame,
    df_ids: pd.DataFrame,
    color_col: Optional[str],
) -> gpd.GeoDataFrame:
    """Merge polygons with attribute data from the filtered CSV.
    Preserves all columns from df_ids.
    """
    # Prepare the attribute dataframe with all columns
    df_attrs = df_ids[["site_id_norm"] + [c for c in df_ids.columns if c != "site_id_norm"]].copy()
    
    # Merge polygons with attributes
    g = gdf_poly.merge(df_attrs, on="site_id_norm", how="inner")
    return g


def plot_map(
    gdf_basins: gpd.GeoDataFrame,
    gdf_conus: gpd.GeoDataFrame,
    gdf_states: Optional[gpd.GeoDataFrame],
    out_png: str,
    basemap: bool = False,
    dpi: int = 150,
    color_col: Optional[str] = None,
    title: Optional[str] = None,
):
    use_basemap = basemap
    try:
        import contextily as cx  # type: ignore
    except Exception:
        print("[info] contextily not available; plotting without basemap")
        use_basemap = False

    if use_basemap:
        print("[info] Using WebMercator + contextily basemap")
        # Project to WebMercator
        gdf_conus_3857 = gdf_conus.to_crs(3857)
        gdf_basins_3857 = gdf_basins.to_crs(3857)
        gdf_states_3857 = gdf_states.to_crs(3857) if gdf_states is not None else None
        fig, ax = plt.subplots(figsize=(14, 9))
        gdf_conus_3857.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#999999", linewidth=0.8)
        if gdf_states_3857 is not None:
            gdf_states_3857.plot(ax=ax, facecolor="none", edgecolor="#333333", linewidth=0.5)
        if color_col and color_col in gdf_basins_3857.columns:
            gdf_basins_3857.plot(ax=ax, column=color_col, cmap="viridis", legend=True, 
                                markersize=2, edgecolor="none", alpha=0.7)
        else:
            gdf_basins_3857.plot(ax=ax, color="#1f77b4", markersize=2, edgecolor="none", alpha=0.7)
        cx.add_basemap(ax, source=cx.providers.Stamen.TerrainBackground)
        ax.set_title(title or "Filtered Basins over CONUS", fontsize=14, fontweight="bold")
        ax.set_axis_off()
    else:
        print("[info] Plotting in WGS84 (no basemap)")
        # Plot in WGS84
        fig, ax = plt.subplots(figsize=(14, 9))
        gdf_conus.plot(ax=ax, facecolor="#f7f7f7", edgecolor="#888", linewidth=0.8)
        if gdf_states is not None:
            gdf_states.plot(ax=ax, facecolor="none", edgecolor="#333333", linewidth=0.5)
        if color_col and color_col in gdf_basins.columns:
            gdf_basins.plot(ax=ax, column=color_col, cmap="viridis", legend=True, 
                           markersize=3, edgecolor="none", alpha=0.7)
        else:
            gdf_basins.plot(ax=ax, color="#1f77b4", markersize=3, edgecolor="none", alpha=0.7)
        ax.set_title(title or "Filtered Basins over CONUS", fontsize=14, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        xmin, ymin, xmax, ymax = CONUS_BOUNDS
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.2)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    print(f"[info] Saving map to {out_png} (dpi={dpi})")
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {out_png}")
    plt.close()  # Close figure to free memory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites-csv", default=FILTERED_CSV_DEFAULT,
                    help="Filtered basins CSV with site IDs and optional attributes")
    ap.add_argument("--shp-dir", default=SHP_DIR_DEFAULT,
                    help="Directory containing basin shapefiles (searched recursively)")
    ap.add_argument("--out", default=OUT_PNG_DEFAULT, help="Output PNG path (used as base for multiple maps)")
    ap.add_argument("--unmatched", default=UNMATCHED_CSV_DEFAULT,
                    help="CSV to write unmatched site IDs")
    ap.add_argument("--color-col", default=None, help="Single column to map; skip auto-generation")
    ap.add_argument("--basemap", action="store_true", help="Add web basemap (requires contextily)")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    args = ap.parse_args()

    df_ids = read_filtered_ids(args.sites_csv)
    print(f"[info] Read {len(df_ids)} rows from {args.sites_csv}")
    shp_paths = find_shapefiles(args.shp_dir)
    print(f"[info] Found {len(shp_paths)} shapefile(s) under {args.shp_dir}")
    gdf_poly = load_polygons(shp_paths)
    print(f"[info] Loaded {len(gdf_poly)} polygons from shapefiles")

    # Build plot dataframe
    gdf_basins = build_plot_gdf(gdf_poly, df_ids, None)  # Don't filter by color_col yet
    total_ids = int(pd.Series(df_ids["site_id_norm"]).dropna().nunique())
    matched = len(gdf_basins)
    print(f"[info] Matched polygons for {matched} of {total_ids} IDs")
    print(f"[info] Columns in matched GeoDataFrame: {list(gdf_basins.columns)}")
    if matched == 0:
        print("[warn] No matched basins; saving CONUS-only map")

    # Report unmatched IDs
    matched_ids = set(gdf_basins["site_id_norm"].tolist())
    all_ids = set(df_ids["site_id_norm"].dropna().tolist())
    unmatched = sorted(all_ids - matched_ids)
    if unmatched:
        Path(args.unmatched).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"site_id_norm": unmatched}).to_csv(args.unmatched, index=False)
        print(f"[info] Unmatched IDs: {len(unmatched)} -> {args.unmatched}")

    gdf_conus = conus_polygon()
    gdf_states = load_states_boundary()

    # If user specified a single color column, just make that map
    if args.color_col:
        gdf_basins_colored = build_plot_gdf(gdf_poly, df_ids, args.color_col)
        plot_map(gdf_basins_colored, gdf_conus, gdf_states, args.out, 
                basemap=args.basemap, dpi=args.dpi, color_col=args.color_col,
                title=f"Filtered Basins: {args.color_col}")
    else:
        # Generate multiple maps with interesting attributes
        out_base = Path(args.out)
        out_stem = out_base.stem
        out_dir = out_base.parent
        
        for attr_label, attr_col, cmap, units in INTERESTING_ATTRIBUTES:
            # Check if column exists
            if attr_col not in gdf_basins.columns:
                print(f"[skip] {attr_col} not in data; skipping")
                continue
            
            # Filter out NaN values for this attribute
            gdf_sub = gdf_basins.dropna(subset=[attr_col]).copy()
            if len(gdf_sub) == 0:
                print(f"[skip] {attr_col} has no valid data; skipping")
                continue
            
            # Convert precipitation from mm/day to mm/year if needed
            if attr_col in ["p-mean", "pet-mean"]:
                gdf_sub[attr_col] = gdf_sub[attr_col] * 365
            
            # Create output path
            safe_name = attr_col.replace("-", "_").replace("/", "_")
            out_path = out_dir / f"{out_stem}_{safe_name}.png"
            
            print(f"\n[map] Generating: {attr_label}")
            
            # Create a figure
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # Plot CONUS background
            gdf_conus.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#cccccc", linewidth=0.5, alpha=0.5)
            
            # Plot basin polygons colored by the attribute
            vmin, vmax = gdf_sub[attr_col].min(), gdf_sub[attr_col].max()
            gdf_sub.plot(ax=ax, column=attr_col, cmap="viridis", edgecolor="#333333", 
                        linewidth=0.3, alpha=0.8, vmin=vmin, vmax=vmax, legend=False)
            
            # Add colorbar with units (only set label, not in colorbar call)
            sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.02)
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label(f"{attr_label} ({units})", fontsize=13, fontweight="bold")
            
            # Plot US outline on top
            if gdf_states is not None:
                gdf_states.plot(ax=ax, facecolor="none", edgecolor="#000000", linewidth=0.8, zorder=10)
            
            ax.set_title(f"{attr_label}", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Longitude", fontsize=13, fontweight="bold")
            ax.set_ylabel("Latitude", fontsize=13, fontweight="bold")
            ax.tick_params(labelsize=12)
            
            # Expand map northward
            xmin, ymin, xmax, ymax = CONUS_BOUNDS
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin - 2, ymax + 3)  # Expand northward and southward slightly
            ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
            
            plt.tight_layout()
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
            print(f"[saved] {out_path}")
            plt.close()
        
        # Generate combined horizontal figure with 3 subplots
        print("\n[combined] Generating 3-panel horizontal figure")
        combined_attrs = [
            ("Drainage Area (km²)", "area_km2", "viridis", "km²"),
            ("Mean Annual Precipitation", "p-mean", "Blues", "mm/year"),
            ("Basin Slope (%)", "slope_pct", "copper", "%"),
        ]
        
        # A4 width is ~8.27 inches, third of height is ~3.9 inches
        # Use wider aspect for better readability: 11 x 3.5 inches
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
        
        for idx, (attr_label, attr_col, cmap, units) in enumerate(combined_attrs):
            ax = axes[idx]
            
            # Check if column exists and has data
            if attr_col not in gdf_basins.columns:
                print(f"[skip] {attr_col} not in data; skipping subplot")
                continue
            
            gdf_sub = gdf_basins.dropna(subset=[attr_col]).copy()
            if len(gdf_sub) == 0:
                print(f"[skip] {attr_col} has no valid data; skipping subplot")
                continue
            
            # Convert precipitation from mm/day to mm/year
            if attr_col in ["p-mean"]:
                gdf_sub[attr_col] = gdf_sub[attr_col] * 365
            
            # Plot CONUS background
            gdf_conus.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#cccccc", linewidth=0.3, alpha=0.4)
            
            # Adjust color scales for better visualization
            if attr_col == "area_km2":
                # Use log scale for drainage area (wide range)
                vmin, vmax = np.log10(gdf_sub[attr_col].quantile(0.01)), np.log10(gdf_sub[attr_col].quantile(0.99))
                gdf_sub_plot = gdf_sub.copy()
                gdf_sub_plot["area_log"] = np.log10(gdf_sub[attr_col])
                gdf_sub_plot.plot(ax=ax, column="area_log", cmap=cmap, edgecolor="none", 
                                linewidth=0, alpha=0.9, vmin=vmin, vmax=vmax, legend=False)
            elif attr_col == "p-mean":
                # Clip extremes for precipitation (saturate top 1%)
                vmin, vmax = gdf_sub[attr_col].quantile(0.01), gdf_sub[attr_col].quantile(0.99)
                gdf_sub.plot(ax=ax, column=attr_col, cmap=cmap, edgecolor="none", 
                           linewidth=0, alpha=0.9, vmin=vmin, vmax=vmax, legend=False)
            elif attr_col == "slope_pct":
                # Clip extremes for slope (saturate top 2%)
                vmin, vmax = gdf_sub[attr_col].quantile(0.01), gdf_sub[attr_col].quantile(0.98)
                gdf_sub.plot(ax=ax, column=attr_col, cmap=cmap, edgecolor="none", 
                           linewidth=0, alpha=0.9, vmin=vmin, vmax=vmax, legend=False)
            else:
                vmin, vmax = gdf_sub[attr_col].min(), gdf_sub[attr_col].max()
                gdf_sub.plot(ax=ax, column=attr_col, cmap=cmap, edgecolor="none", 
                           linewidth=0, alpha=0.9, vmin=vmin, vmax=vmax, legend=False)
            
            # Add colorbar
            if attr_col == "area_km2":
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
                # Convert log ticks back to original scale
                tick_locs = cbar.get_ticks()
                cbar.set_ticklabels([f"{10**t:.0f}" for t in tick_locs])
                cbar.ax.tick_params(labelsize=8)
                cbar.set_label(units, fontsize=9, fontweight="bold")
            else:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
                cbar.ax.tick_params(labelsize=8)
                cbar.set_label(units, fontsize=9, fontweight="bold")
            
            # Plot external CONUS border on top
            if gdf_states is not None:
                # Dissolve to get only external boundary
                conus_boundary = gdf_states.dissolve()
                conus_boundary.plot(ax=ax, facecolor="none", edgecolor="#000000", linewidth=0.3, zorder=10)
            
            # Add subplot label (a), (b), (c)
            subplot_labels = ['(a)', '(b)', '(c)']
            ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes, 
                   fontsize=11, fontweight='bold', va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))
            
            ax.set_title(attr_label, fontsize=10, fontweight="bold", pad=8)
            ax.set_xlabel("Longitude", fontsize=8)
            ax.set_ylabel("Latitude", fontsize=8)
            ax.tick_params(labelsize=7)
            
            # Set extent
            xmin, ymin, xmax, ymax = CONUS_BOUNDS
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin - 2, ymax + 3)
            ax.grid(True, alpha=0.15, linestyle="--", linewidth=0.3)
        
        plt.tight_layout()
        combined_path = out_dir / f"{out_stem}_combined_3panel.png"
        plt.savefig(combined_path, dpi=300, bbox_inches="tight")
        print(f"[saved] {combined_path}")
        plt.close()


if __name__ == "__main__":
    main()

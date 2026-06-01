"""Basin polygon loading and fallback geometry utilities for Flash-NH.

PRODUCTION PATH — CAMELSH shapefiles required
----------------------------------------------
This module expects real CAMELSH basin polygons. The preferred source is:
  CAMELSH_shapefile.shp  (GAGES-II derived, contains GAGE_ID column)

Polygon provenance:
  Current CAMELSH reference: DOI 10.5281/zenodo.16763144 (v7, 2025-08-14)
  Shapefiles available at:   DOI 10.5281/zenodo.15066778
  File in archive:           shapefiles.7z → CAMELSH_shapefile.shp
  Download size:             ~506 MB (shapefiles.7z only; NOT the 21 GB time series)

Standard placement in the Flash-NH data root:
  {data_root}/02_basin_geometries/camelsh/CAMELSH_shapefile.shp

The script auto-discovers this path. Alternatively, set camelsh.basin_polygons
in configs/pilot_stage1.yaml.

TEST-ONLY FALLBACK — circular buffers
--------------------------------------
When real polygons are unavailable, a circular-buffer approximation is available
as a SMOKE TEST ONLY.  It is DISABLED by default and requires the explicit CLI
flag --allow-fallback-circles on the calling script.

Circular buffers are NOT valid for:
  - Production basin-average extraction
  - Training data generation
  - Scientific QC or evaluation
  - Any result that will be used downstream

The only legitimate use of circular buffers is mechanics testing of the
weight-computation and extraction code before real polygons are available.

All geometry returned from this module is in EPSG:5070 (NAD83 / Conus Albers
Equal Area) to ensure consistent area calculations throughout the weight pipeline.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

CRS_WGS84 = 4326
CRS_ALBERS_CONUS = 5070  # NAD83 / Conus Albers Equal Area

# CAMELSH provenance constants
CAMELSH_PREFERRED_FILE = "CAMELSH_shapefile.shp"
CAMELSH_SECONDARY_FILE = "CAMELSH_shapefile_hydroATLAS.shp"
CAMELSH_ID_COL = "GAGE_ID"
CAMELSH_CURRENT_DOI = "10.5281/zenodo.16763144"
CAMELSH_POLYGON_SOURCE_DOI = "10.5281/zenodo.15066778"
# Standard auto-discovery locations within the data root (in priority order).
# The shapefiles.7z from Zenodo extracts into a 'shapefiles/' subdirectory,
# so both the flat and nested paths are checked.
CAMELSH_STANDARD_SUBPATHS = [
    "02_basin_geometries/camelsh/shapefiles/CAMELSH_shapefile.shp",  # extracted from Zenodo 7z
    "02_basin_geometries/camelsh/CAMELSH_shapefile.shp",             # flat layout fallback
]

_NO_POLYGON_MESSAGE = textwrap.dedent("""\
    No CAMELSH basin polygons configured or found.

    Real basin polygons are REQUIRED for production weights.

    To obtain the shapefile:
      1. Download shapefiles.7z (~506 MB) from DOI {polygon_doi}:
           https://zenodo.org/records/15066778/files/shapefiles.7z
      2. Extract to {{data_root}}/02_basin_geometries/camelsh/
         (7-Zip creates a 'shapefiles/' subdirectory inside that folder.)
         The script auto-discovers the path:
           {{data_root}}/02_basin_geometries/camelsh/shapefiles/CAMELSH_shapefile.shp
      3. Alternatively, set camelsh.basin_polygons in configs/pilot_stage1.yaml.

    CAMELSH reference: DOI {reference_doi} (v7, 2025-08-14)
    Shapefile columns: GAGE_ID (8-digit zero-padded USGS gauge ID), AREA, geometry
    CRS note: CAMELSH_shapefile.shp has no .prj file; EPSG:4326 is assumed.

    For SMOKE TESTING ONLY, re-run with --allow-fallback-circles.
    Circular-buffer weights are NOT valid for training or scientific extraction.
""").format(
    polygon_doi=CAMELSH_POLYGON_SOURCE_DOI,
    reference_doi=CAMELSH_CURRENT_DOI,
)


# ---------------------------------------------------------------------------
# Path resolution helper (exported so callers can record the actual path)
# ---------------------------------------------------------------------------

def resolve_polygon_path(
    explicit_path: Optional[Path] = None,
    data_root: Optional[Path] = None,
) -> Optional[Path]:
    """Return the first existing CAMELSH polygon path using the standard priority.

    Priority:
      1. explicit_path (if given and exists)
      2. Auto-discovery at each entry of CAMELSH_STANDARD_SUBPATHS under data_root

    Returns None if nothing is found (caller decides whether to fall back or abort).
    This mirrors the discovery logic in load_pilot_geometries() so callers can obtain
    the resolved path for provenance recording without parsing it out of the return tuple.
    """
    if explicit_path is not None and Path(explicit_path).exists():
        return Path(explicit_path)
    if data_root is not None:
        for subpath in CAMELSH_STANDARD_SUBPATHS:
            auto = Path(data_root) / subpath
            if auto.exists():
                return auto
    return None


# ---------------------------------------------------------------------------
# STAID normalisation
# ---------------------------------------------------------------------------

def normalise_staid(raw) -> str:
    """Return an 8-digit zero-padded USGS STAID string."""
    return str(raw).strip().lstrip("0").zfill(8)


# ---------------------------------------------------------------------------
# CAMELSH polygon loading
# ---------------------------------------------------------------------------

def load_camelsh_polygons(gpkg_or_shp_path: Path, id_col: Optional[str] = None):
    """Load basin polygons from a file (GeoPackage or Shapefile); return GeoDataFrame in EPSG:4326.

    id_col: explicit basin-ID column name; auto-detected if None.
    Adds a 'STAID_NORM' column containing normalised 8-digit STAIDs.
    """
    import geopandas as gpd

    gdf = gpd.read_file(gpkg_or_shp_path)
    if gdf.crs is None:
        # CAMELSH_shapefile.shp has no .prj file; CRS is WGS84 (EPSG:4326) by convention.
        LOGGER.warning(
            "%s has no CRS metadata (.prj absent) — assigning EPSG:4326 (WGS84). "
            "Verify this is correct if area values look wrong.",
            Path(gpkg_or_shp_path).name,
        )
        gdf = gdf.set_crs(CRS_WGS84)
    else:
        gdf = gdf.to_crs(CRS_WGS84)

    if id_col is None:
        for candidate in (CAMELSH_ID_COL, "hru_id", "STAID", "staid",
                          "gauge_id", "basin_id", "site_no", "gageid"):
            if candidate in gdf.columns:
                id_col = candidate
                break

    if id_col is None:
        raise ValueError(
            f"Cannot detect basin-ID column in {Path(gpkg_or_shp_path).name}. "
            f"Available columns: {list(gdf.columns)}. Set basin_polygon_id_column in config."
        )

    gdf["STAID_NORM"] = gdf[id_col].apply(normalise_staid)
    LOGGER.info(
        "Loaded %d polygons from %s (id_col='%s', CRS=%s)",
        len(gdf), Path(gpkg_or_shp_path).name, id_col, gdf.crs,
    )
    return gdf, id_col


def match_pilot_staids(gdf, pilot_staids: list[str]):
    """Match pilot STAIDs against a GeoDataFrame that has 'STAID_NORM'.

    Returns (matched_gdf, missing_staids_list).
    """
    norm_map = {normalise_staid(s): s for s in pilot_staids}
    matched = gdf[gdf["STAID_NORM"].isin(norm_map)].copy()
    matched_norm = set(matched["STAID_NORM"].tolist())
    missing = [norm_map[n] for n in norm_map if n not in matched_norm]
    return matched, missing


# ---------------------------------------------------------------------------
# Circular-buffer fallback (TEST ONLY)
# ---------------------------------------------------------------------------

def _circular_buffer_from_metadata(lat: float, lon: float, area_km2: float):
    """Create a circular basin approximation in EPSG:5070 (TEST ONLY).

    Radius = sqrt(area_km2 / pi) km. NOT valid for production use.
    """
    from shapely.geometry import Point
    from pyproj import Transformer, CRS

    radius_m = float(np.sqrt(max(area_km2, 0.01) * 1e6 / np.pi))
    t = Transformer.from_crs(CRS.from_epsg(CRS_WGS84), CRS.from_epsg(CRS_ALBERS_CONUS), always_xy=True)
    x, y = t.transform(float(lon), float(lat))
    return Point(x, y).buffer(radius_m, resolution=32)


# ---------------------------------------------------------------------------
# Unified geometry loader
# ---------------------------------------------------------------------------

def load_pilot_geometries(
    pilot_csv_path: Path,
    polygon_path: Optional[Path] = None,
    *,
    allow_fallback: bool = False,
    id_col: Optional[str] = None,
    data_root: Optional[Path] = None,
) -> tuple[Any, str, list[str]]:
    """Load or build geometries for pilot basins.

    Priority:
      1. Real CAMELSH polygons: from polygon_path (explicit) or auto-discovered
         at {data_root}/02_basin_geometries/camelsh/CAMELSH_shapefile.shp.
      2. Circular-buffer fallback ONLY if allow_fallback=True (smoke-test mode).

    Returns:
        basin_gdf    : GeoDataFrame (EPSG:5070) with STAID, DRAIN_SQKM, pilot_role,
                       LAT_GAGE, LNG_GAGE, geometry
        method       : "camelsh_shapefile" | "circular_buffer_test_only"
        missing      : STAIDs for which no geometry was found / built
    """
    import geopandas as gpd

    pilot_df = pd.read_csv(pilot_csv_path, dtype={"STAID": str})
    pilot_df["STAID"] = pilot_df["STAID"].apply(normalise_staid)
    pilot_staids = pilot_df["STAID"].tolist()

    # --- Locate polygon file ---
    resolved_path: Optional[Path] = None

    if polygon_path and Path(polygon_path).exists():
        resolved_path = Path(polygon_path)

    if resolved_path is None and data_root is not None:
        for subpath in CAMELSH_STANDARD_SUBPATHS:
            auto = Path(data_root) / subpath
            if auto.exists():
                resolved_path = auto
                LOGGER.info("Auto-discovered CAMELSH polygons: %s", resolved_path)
                break

    # --- Attempt real polygon load ---
    if resolved_path is not None:
        try:
            poly_gdf, detected_id_col = load_camelsh_polygons(resolved_path, id_col=id_col)
            matched, missing_poly = match_pilot_staids(poly_gdf, pilot_staids)

            if matched.empty:
                LOGGER.warning(
                    "CAMELSH file loaded (%d polygons) but zero pilot STAIDs matched — "
                    "check id_col (detected='%s') and STAID format.",
                    len(poly_gdf), detected_id_col,
                )
            else:
                matched = matched.rename(columns={"STAID_NORM": "STAID"})
                meta_cols = [c for c in ("DRAIN_SQKM", "pilot_role", "LAT_GAGE", "LNG_GAGE")
                             if c in pilot_df.columns]
                merged = matched.merge(pilot_df[["STAID"] + meta_cols], on="STAID", how="inner")
                merged = merged.to_crs(CRS_ALBERS_CONUS)

                # Sanity check: compare polygon area to DRAIN_SQKM
                if "DRAIN_SQKM" in merged.columns:
                    poly_area_km2 = merged.geometry.area / 1e6
                    drain = merged["DRAIN_SQKM"].astype(float)
                    ratio = poly_area_km2 / (drain + 1e-9)
                    n_suspicious = int(((ratio < 0.1) | (ratio > 10)).sum())
                    if n_suspicious > 0:
                        LOGGER.warning(
                            "%d basins have polygon area deviating >10× from DRAIN_SQKM — "
                            "verify id_col and CRS.", n_suspicious,
                        )

                LOGGER.info(
                    "Loaded %d/%d real polygons from %s (id_col='%s', %d missing).",
                    len(merged), len(pilot_staids),
                    resolved_path.name, detected_id_col, len(missing_poly),
                )
                return merged, "camelsh_shapefile", missing_poly

        except Exception as exc:
            LOGGER.warning("Failed to load CAMELSH polygons from %s: %s", resolved_path, exc)
            resolved_path = None

    # --- No real polygons available ---
    if not allow_fallback:
        raise RuntimeError(_NO_POLYGON_MESSAGE)

    # --- Circular-buffer fallback (TEST ONLY) ---
    LOGGER.warning(
        "CIRCULAR-BUFFER FALLBACK ACTIVE (--allow-fallback-circles). "
        "Weights produced are TEST-ONLY and MUST NOT be used for production."
    )
    required = {"LAT_GAGE", "LNG_GAGE", "DRAIN_SQKM"}
    missing_cols = required - set(pilot_df.columns)
    if missing_cols:
        raise RuntimeError(
            f"Circular-buffer fallback requires {required} columns but {missing_cols} are missing."
        )

    missing: list[str] = []
    records = []
    for _, row in pilot_df.iterrows():
        staid = row["STAID"]
        try:
            lat, lon, area = float(row["LAT_GAGE"]), float(row["LNG_GAGE"]), float(row["DRAIN_SQKM"])
            if any(np.isnan(v) for v in [lat, lon, area]) or area <= 0:
                raise ValueError("NaN or non-positive metadata")
            records.append({
                "STAID": staid,
                "DRAIN_SQKM": area,
                "pilot_role": row.get("pilot_role", "UNKNOWN"),
                "LAT_GAGE": lat,
                "LNG_GAGE": lon,
                "geometry": _circular_buffer_from_metadata(lat, lon, area),
            })
        except Exception as exc:
            LOGGER.warning("Cannot build buffer for STAID %s: %s", staid, exc)
            missing.append(staid)

    basin_gdf = gpd.GeoDataFrame(records, crs=CRS_ALBERS_CONUS)
    LOGGER.info(
        "Built %d/%d circular-buffer geometries (TEST ONLY, %d missing).",
        len(basin_gdf), len(pilot_staids), len(missing),
    )
    return basin_gdf, "circular_buffer_test_only", missing

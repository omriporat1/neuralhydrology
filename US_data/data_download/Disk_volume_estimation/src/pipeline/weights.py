"""Basin-grid overlap weight computation for MRMS and RTMA.

Algorithm overview
------------------
For each basin polygon (in EPSG:5070, Albers Equal Area):

  MRMS (regular_ll, 0.01 deg):
    1. Find candidate cells via bounding-box index lookup (+2-cell border).
    2. Build cell-corner coordinates (4 points per cell) in EPSG:4326.
    3. Batch-transform corners to EPSG:5070 via pyproj (vectorised).
    4. Create cell polygons with shapely.polygons() (vectorised).
    5. Compute basin–cell intersection areas with shapely.intersection() (vectorised).
    6. Keep cells with overlap_area > 0; record row/col indices and areas.

  RTMA (Lambert Conformal Conic, ~2540 m):
    1. Reconstruct the RTMA Lambert CRS from the stored GRIB parameters.
    2. Compute the SW-corner projected position (x0, y0) of the grid.
    3. Find candidate cells via bounding-box in projected RTMA coordinates.
    4. Build cell-corner coordinates (4 points per cell) in RTMA LCC space.
    5. Batch-transform corners to EPSG:5070 via pyproj (vectorised).
    6. Create cell polygons (vectorised) and compute intersections (vectorised).

Weights are the raw intersection areas (m² in EPSG:5070).
Normalised weights are raw_weight / sum(raw_weight) per STAID+product
so that they sum exactly to 1.0 per basin per product (within floating-point).

RTMA Lambert projection parameters (NWS NDFD CONUS grid):
  proj=lcc  lat_0=25  lon_0=-95  lat_1=25  lat_2=25  R=6371200  units=m
  (lon_0 = LoVInDegrees - 360 = 265 - 360 = -95)

Grid layouts:
  MRMS: row 0 = northern edge (lat_first=54.995), row increases southward.
  RTMA: row 0 = southern edge (jScansPositively=1), row increases northward.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

CRS_WGS84 = 4326
CRS_ALBERS = 5070

# RTMA NWS NDFD grid CRS (sphere, R=6371200 m)
_PROJ4_RTMA = "+proj=lcc +lat_0=25 +lon_0=-95 +lat_1=25 +lat_2=25 +R=6371200 +units=m +no_defs"


# ---------------------------------------------------------------------------
# Grid specification dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MrmsGridSpec:
    """Parameters of the MRMS regular_ll CONUS grid."""
    lat_first: float   # latitude of row 0  (northernmost, ~54.995)
    lon_first: float   # longitude of col 0 (westernmost,  ~-129.995)
    dy: float          # abs(latitudinal step), positive value
    dx: float          # longitudinal step,    positive value
    nrows: int         # number of rows (latitude direction)
    ncols: int         # number of columns (longitude direction)

    @property
    def lat_last(self) -> float:
        return self.lat_first - (self.nrows - 1) * self.dy

    @property
    def lon_last(self) -> float:
        return self.lon_first + (self.ncols - 1) * self.dx


@dataclass
class RtmaGridSpec:
    """Parameters of the RTMA Lambert Conformal CONUS grid."""
    lat_first_deg: float   # lat of first grid point (SW corner)
    lon_first_deg: float   # lon of first grid point (SW corner), [-180,180]
    dx_m: float            # cell width in metres
    dy_m: float            # cell height in metres
    ni: int                # number of columns
    nj: int                # number of rows
    lov_deg: float = 265.0  # LoVInDegrees (central meridian, 0-360 convention)
    lad_deg: float = 25.0   # LaDInDegrees (standard parallel)
    latin1: float = 25.0
    latin2: float = 25.0
    # projected SW-corner position; computed once on first use
    _x0: Optional[float] = field(default=None, repr=False)
    _y0: Optional[float] = field(default=None, repr=False)
    _crs: Any = field(default=None, repr=False)

    @property
    def crs(self):
        if self._crs is None:
            from pyproj import CRS
            try:
                self._crs = CRS.from_proj4(_PROJ4_RTMA)
            except Exception:
                self._crs = CRS.from_proj4(
                    f"+proj=lcc +lat_0={self.lad_deg} "
                    f"+lon_0={self.lov_deg - 360.0:.4f} "
                    f"+lat_1={self.latin1} +lat_2={self.latin2} "
                    "+ellps=WGS84 +units=m +no_defs"
                )
        return self._crs

    @property
    def x0(self) -> float:
        if self._x0 is None:
            self._compute_origin()
        return self._x0

    @property
    def y0(self) -> float:
        if self._y0 is None:
            self._compute_origin()
        return self._y0

    def _compute_origin(self):
        from pyproj import Transformer, CRS
        t = Transformer.from_crs(CRS.from_epsg(CRS_WGS84), self.crs, always_xy=True)
        x, y = t.transform(self.lon_first_deg, self.lat_first_deg)
        self._x0 = float(x)
        self._y0 = float(y)
        LOGGER.debug("RTMA grid origin: x0=%.1f m, y0=%.1f m", self._x0, self._y0)


# ---------------------------------------------------------------------------
# Grid spec loaders
# ---------------------------------------------------------------------------

def mrms_spec_from_json(json_path: Path) -> MrmsGridSpec:
    """Build MrmsGridSpec from a grid-definition JSON written by build_stage1_grid_definitions."""
    d = json.loads(Path(json_path).read_text(encoding="utf-8"))
    shape = d["grid_shape_rows_cols"]  # [nrows, ncols]
    dx = float(d["grib_attrs"].get("GRIB_iDirectionIncrementInDegrees", 0.01))
    dy = float(d["grib_attrs"].get("GRIB_jDirectionIncrementInDegrees", 0.01))
    lat_first = float(d["bbox_lat_max"])  # north edge (lat_descending=True → row 0 = max lat)
    lon_first = float(d["bbox_lon_min"])  # west edge
    return MrmsGridSpec(
        lat_first=lat_first,
        lon_first=lon_first,
        dy=dy,
        dx=dx,
        nrows=shape[0],
        ncols=shape[1],
    )


def rtma_spec_from_json(json_path: Path) -> RtmaGridSpec:
    """Build RtmaGridSpec from a grid-definition JSON written by build_stage1_grid_definitions."""
    d = json.loads(Path(json_path).read_text(encoding="utf-8"))
    g = d["grib_attrs"]
    shape = d["grid_shape_rows_cols"]  # [nrows, ncols]
    return RtmaGridSpec(
        lat_first_deg=float(g["GRIB_latitudeOfFirstGridPointInDegrees"]),
        lon_first_deg=float(g["GRIB_longitudeOfFirstGridPointInDegrees"]),  # already normalised to [-180,180]
        dx_m=float(g["GRIB_DxInMetres"]),
        dy_m=float(g["GRIB_DyInMetres"]),
        ni=shape[1],
        nj=shape[0],
        lov_deg=float(g.get("GRIB_LoVInDegrees", 265.0)),
        lad_deg=float(g.get("GRIB_LaDInDegrees", 25.0)),
        latin1=float(g.get("GRIB_Latin1InDegrees", 25.0)),
        latin2=float(g.get("GRIB_Latin2InDegrees", 25.0)),
    )


# ---------------------------------------------------------------------------
# Per-basin weight helpers
# ---------------------------------------------------------------------------

def _corners_to_polygons_5070(
    lon_lo: np.ndarray, lat_lo: np.ndarray,
    lon_hi: np.ndarray, lat_hi: np.ndarray,
    src_crs,
) -> np.ndarray:
    """Batch-project cell corners (src_crs → EPSG:5070) and return shapely polygon array.

    Accepts lon_lo/lat_lo/lon_hi/lat_hi as flat 1-D arrays of cell extents.
    Returns a 1-D object array of shapely Polygons (all in EPSG:5070).
    """
    import shapely
    from pyproj import Transformer, CRS

    t = Transformer.from_crs(src_crs, CRS.from_epsg(CRS_ALBERS), always_xy=True)

    # Project 4 corners per cell
    x_ll, y_ll = t.transform(lon_lo, lat_lo)
    x_lr, y_lr = t.transform(lon_hi, lat_lo)
    x_ur, y_ur = t.transform(lon_hi, lat_hi)
    x_ul, y_ul = t.transform(lon_lo, lat_hi)

    # Build ring coordinate arrays: shape (n, 5, 2) — 5 points to close ring
    n = len(lon_lo)
    rings = np.zeros((n, 5, 2))
    rings[:, 0, :] = np.column_stack([x_ll, y_ll])
    rings[:, 1, :] = np.column_stack([x_lr, y_lr])
    rings[:, 2, :] = np.column_stack([x_ur, y_ur])
    rings[:, 3, :] = np.column_stack([x_ul, y_ul])
    rings[:, 4, :] = rings[:, 0, :]  # close ring

    return shapely.polygons(rings)


def _intersect_cells(basin_geom_5070, cells_5070: np.ndarray) -> np.ndarray:
    """Vectorised intersection: return area array (m²) for each cell against basin."""
    import shapely

    basin_arr = np.full(len(cells_5070), basin_geom_5070)
    intersections = shapely.intersection(basin_arr, cells_5070)
    return shapely.area(intersections)


def _compute_mrms_basin_weights(
    staid: str,
    basin_geom_5070,
    mrms_spec: MrmsGridSpec,
    geom_method: str,
) -> list[dict]:
    """Compute MRMS weight records for one basin."""
    from pyproj import Transformer, CRS
    import shapely

    # Project basin to EPSG:4326 to get bbox for grid-cell selection
    t_5070_to_4326 = Transformer.from_crs(
        CRS.from_epsg(CRS_ALBERS), CRS.from_epsg(CRS_WGS84), always_xy=True
    )
    # Get basin bounds in EPSG:5070, then sample extreme points in 4326
    xmin, ymin, xmax, ymax = basin_geom_5070.bounds
    lons_corners, lats_corners = t_5070_to_4326.transform(
        [xmin, xmax, xmin, xmax], [ymin, ymin, ymax, ymax]
    )
    lon_min_4326 = float(np.min(lons_corners))
    lon_max_4326 = float(np.max(lons_corners))
    lat_min_4326 = float(np.min(lats_corners))
    lat_max_4326 = float(np.max(lats_corners))

    # Add 2-cell buffer
    buf = 2 * mrms_spec.dx
    lon_min_4326 -= buf; lon_max_4326 += buf
    lat_min_4326 -= buf; lat_max_4326 += buf

    # Clip to MRMS extent
    lon_min_4326 = max(lon_min_4326, mrms_spec.lon_first)
    lon_max_4326 = min(lon_max_4326, mrms_spec.lon_last)
    lat_min_4326 = max(lat_min_4326, mrms_spec.lat_last)
    lat_max_4326 = min(lat_max_4326, mrms_spec.lat_first)

    if lon_min_4326 >= lon_max_4326 or lat_min_4326 >= lat_max_4326:
        return []  # basin outside grid extent

    # MRMS row: i = round((lat_first - lat) / dy); row 0 = northernmost
    row_min = max(0, int(np.floor((mrms_spec.lat_first - lat_max_4326) / mrms_spec.dy)) - 1)
    row_max = min(mrms_spec.nrows - 1, int(np.ceil((mrms_spec.lat_first - lat_min_4326) / mrms_spec.dy)) + 1)
    col_min = max(0, int(np.floor((lon_min_4326 - mrms_spec.lon_first) / mrms_spec.dx)) - 1)
    col_max = min(mrms_spec.ncols - 1, int(np.ceil((lon_max_4326 - mrms_spec.lon_first) / mrms_spec.dx)) + 1)

    if row_min > row_max or col_min > col_max:
        return []

    # Build candidate cell arrays
    rows = np.arange(row_min, row_max + 1)
    cols = np.arange(col_min, col_max + 1)
    RR, CC = np.meshgrid(rows, cols, indexing="ij")
    row_flat = RR.ravel()
    col_flat = CC.ravel()

    lat_c = mrms_spec.lat_first - row_flat * mrms_spec.dy
    lon_c = mrms_spec.lon_first + col_flat * mrms_spec.dx
    half_dy = mrms_spec.dy / 2
    half_dx = mrms_spec.dx / 2

    # Build cell polygons in EPSG:5070 via batch corner projection
    cells_5070 = _corners_to_polygons_5070(
        lon_c - half_dx, lat_c - half_dy,
        lon_c + half_dx, lat_c + half_dy,
        src_crs=CRS.from_epsg(CRS_WGS84),
    )
    cell_areas = shapely.area(cells_5070)

    # Vectorised intersection
    overlap_areas = _intersect_cells(basin_geom_5070, cells_5070)
    mask = overlap_areas > 0

    if not mask.any():
        return []

    return [
        {
            "STAID": staid,
            "product": "mrms_qpe_1h_pass1",
            "row_idx": int(row_flat[i]),
            "col_idx": int(col_flat[i]),
            "grid_cell_id": f"{int(row_flat[i]):04d}_{int(col_flat[i]):04d}",
            "lon_center": float(lon_c[i]),
            "lat_center": float(lat_c[i]),
            "x_center_m": None,
            "y_center_m": None,
            "overlap_area_m2": float(overlap_areas[i]),
            "cell_area_m2": float(cell_areas[i]),
            "geometry_method": geom_method,
        }
        for i in np.where(mask)[0]
    ]


def _compute_rtma_basin_weights(
    staid: str,
    basin_geom_5070,
    rtma_spec: RtmaGridSpec,
    geom_method: str,
) -> list[dict]:
    """Compute RTMA weight records for one basin."""
    import shapely
    from pyproj import Transformer, CRS

    # Project basin to RTMA CRS for bbox selection
    t_5070_to_rtma = Transformer.from_crs(
        CRS.from_epsg(CRS_ALBERS), rtma_spec.crs, always_xy=True
    )
    xmin_5070, ymin_5070, xmax_5070, ymax_5070 = basin_geom_5070.bounds
    # Transform bbox corners
    xs, ys = t_5070_to_rtma.transform(
        [xmin_5070, xmax_5070, xmin_5070, xmax_5070],
        [ymin_5070, ymin_5070, ymax_5070, ymax_5070],
    )
    x_min_rtma, x_max_rtma = float(np.min(xs)), float(np.max(xs))
    y_min_rtma, y_max_rtma = float(np.min(ys)), float(np.max(ys))

    # Find candidate cell range (with 2-cell buffer)
    half_dx = rtma_spec.dx_m / 2
    half_dy = rtma_spec.dy_m / 2
    x0, y0 = rtma_spec.x0, rtma_spec.y0

    col_min = max(0, int(np.floor((x_min_rtma - x0) / rtma_spec.dx_m)) - 2)
    col_max = min(rtma_spec.ni - 1, int(np.ceil((x_max_rtma - x0) / rtma_spec.dx_m)) + 2)
    row_min = max(0, int(np.floor((y_min_rtma - y0) / rtma_spec.dy_m)) - 2)
    row_max = min(rtma_spec.nj - 1, int(np.ceil((y_max_rtma - y0) / rtma_spec.dy_m)) + 2)

    if col_min > col_max or row_min > row_max:
        return []

    rows = np.arange(row_min, row_max + 1)
    cols = np.arange(col_min, col_max + 1)
    RR, CC = np.meshgrid(rows, cols, indexing="ij")
    row_flat = RR.ravel()
    col_flat = CC.ravel()

    x_c = x0 + col_flat * rtma_spec.dx_m
    y_c = y0 + row_flat * rtma_spec.dy_m

    # Project cell centres to EPSG:4326 for lat/lon reporting
    t_rtma_to_4326 = Transformer.from_crs(
        rtma_spec.crs, CRS.from_epsg(CRS_WGS84), always_xy=True
    )
    lon_c, lat_c = t_rtma_to_4326.transform(x_c, y_c)

    # Build cell polygons in EPSG:5070 via batch corner projection
    cells_5070 = _corners_to_polygons_5070(
        x_c - half_dx, y_c - half_dy,
        x_c + half_dx, y_c + half_dy,
        src_crs=rtma_spec.crs,
    )
    cell_areas = shapely.area(cells_5070)

    # Vectorised intersection
    overlap_areas = _intersect_cells(basin_geom_5070, cells_5070)
    mask = overlap_areas > 0

    if not mask.any():
        return []

    return [
        {
            "STAID": staid,
            "product": "rtma_conus_aws_2p5km",
            "row_idx": int(row_flat[i]),
            "col_idx": int(col_flat[i]),
            "grid_cell_id": f"{int(row_flat[i]):04d}_{int(col_flat[i]):04d}",
            "lon_center": float(lon_c[i]),
            "lat_center": float(lat_c[i]),
            "x_center_m": float(x_c[i]),
            "y_center_m": float(y_c[i]),
            "overlap_area_m2": float(overlap_areas[i]),
            "cell_area_m2": float(cell_areas[i]),
            "geometry_method": geom_method,
        }
        for i in np.where(mask)[0]
    ]


# ---------------------------------------------------------------------------
# Public compute functions
# ---------------------------------------------------------------------------

def compute_mrms_weights(
    basin_gdf,
    mrms_spec: MrmsGridSpec,
    geom_method: str = "unknown",
) -> pd.DataFrame:
    """Compute MRMS cell weights for all basins in basin_gdf.

    basin_gdf must be in EPSG:5070 with columns STAID, geometry.
    Returns a DataFrame with raw_weight and normalized_weight columns.
    """
    all_records: list[dict] = []
    n_total = len(basin_gdf)
    for idx, row in enumerate(basin_gdf.itertuples()):
        staid = row.STAID
        records = _compute_mrms_basin_weights(staid, row.geometry, mrms_spec, geom_method)
        all_records.extend(records)
        if (idx + 1) % 10 == 0 or (idx + 1) == n_total:
            LOGGER.info("MRMS weights: %d/%d basins processed (%d cells so far)",
                        idx + 1, n_total, len(all_records))

    if not all_records:
        return pd.DataFrame(columns=[
            "STAID", "product", "row_idx", "col_idx", "grid_cell_id",
            "lon_center", "lat_center", "x_center_m", "y_center_m",
            "overlap_area_m2", "cell_area_m2", "raw_weight", "normalized_weight",
            "geometry_method",
        ])

    df = pd.DataFrame(all_records)
    df["raw_weight"] = df["overlap_area_m2"]
    return _normalise_weights(df)


def compute_rtma_weights(
    basin_gdf,
    rtma_spec: RtmaGridSpec,
    geom_method: str = "unknown",
) -> pd.DataFrame:
    """Compute RTMA cell weights for all basins in basin_gdf.

    basin_gdf must be in EPSG:5070 with columns STAID, geometry.
    Returns a DataFrame with raw_weight and normalized_weight columns.
    """
    all_records: list[dict] = []
    n_total = len(basin_gdf)
    for idx, row in enumerate(basin_gdf.itertuples()):
        staid = row.STAID
        records = _compute_rtma_basin_weights(staid, row.geometry, rtma_spec, geom_method)
        all_records.extend(records)
        if (idx + 1) % 10 == 0 or (idx + 1) == n_total:
            LOGGER.info("RTMA weights: %d/%d basins processed (%d cells so far)",
                        idx + 1, n_total, len(all_records))

    if not all_records:
        return pd.DataFrame(columns=[
            "STAID", "product", "row_idx", "col_idx", "grid_cell_id",
            "lon_center", "lat_center", "x_center_m", "y_center_m",
            "overlap_area_m2", "cell_area_m2", "raw_weight", "normalized_weight",
            "geometry_method",
        ])

    df = pd.DataFrame(all_records)
    df["raw_weight"] = df["overlap_area_m2"]
    return _normalise_weights(df)


def _normalise_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized_weight = raw_weight / sum(raw_weight) per (STAID, product)."""
    total = df.groupby(["STAID", "product"])["raw_weight"].transform("sum")
    df["normalized_weight"] = df["raw_weight"] / total
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_weight_table(
    weights_df: pd.DataFrame,
    pilot_staids: list[str],
    product: str,
    tol: float = 0.02,
) -> dict[str, Any]:
    """Validate a weight table and return a summary dict.

    tol: acceptable deviation of weight sum from 1.0 (default ±0.02).
    """
    from src.pipeline.geometries import normalise_staid

    norm_pilot = [normalise_staid(s) for s in pilot_staids]
    norm_in_table = set(weights_df["STAID"].unique())

    missing_weights = [s for s in norm_pilot if s not in norm_in_table]
    n_with_weights = len(norm_in_table)

    # Weight sum per basin
    sums = weights_df.groupby("STAID")["normalized_weight"].sum()
    sum_vals = sums.values

    bad_sum = sums[np.abs(sums - 1.0) > tol]
    negative = weights_df[weights_df["normalized_weight"] < 0]

    cells_per_basin = weights_df.groupby("STAID").size()
    cell_vals = cells_per_basin.values

    suspicious: list[dict] = []
    for staid, s in sums.items():
        nc = int(cells_per_basin.get(staid, 0))
        flags = []
        if abs(s - 1.0) > tol:
            flags.append(f"weight_sum={s:.4f}")
        if nc <= 1:
            flags.append(f"n_cells={nc}")
        if nc > 5000:
            flags.append(f"n_cells={nc} (unusually high)")
        if flags:
            suspicious.append({"STAID": staid, "flags": flags})

    return {
        "product": product,
        "n_pilot_staids": len(norm_pilot),
        "n_with_weights": n_with_weights,
        "n_missing_weights": len(missing_weights),
        "missing_weight_staids": missing_weights,
        "total_weight_records": len(weights_df),
        "n_negative_weights": len(negative),
        "weight_sum_min": float(np.min(sum_vals)) if len(sum_vals) else None,
        "weight_sum_median": float(np.median(sum_vals)) if len(sum_vals) else None,
        "weight_sum_max": float(np.max(sum_vals)) if len(sum_vals) else None,
        "n_basins_bad_sum": len(bad_sum),
        "cells_per_basin_min": int(np.min(cell_vals)) if len(cell_vals) else None,
        "cells_per_basin_median": float(np.median(cell_vals)) if len(cell_vals) else None,
        "cells_per_basin_max": int(np.max(cell_vals)) if len(cell_vals) else None,
        "suspicious_basins": suspicious,
        "validation_tolerance": tol,
        "checks": {
            "table_nonempty": len(weights_df) > 0,
            "no_negative_weights": len(negative) == 0,
            "all_pilots_have_weights": len(missing_weights) == 0,
            "weight_sums_within_tolerance": len(bad_sum) == 0,
        },
    }

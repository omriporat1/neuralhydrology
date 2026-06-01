# Flash-NH Stage 1 — Basin-Grid Overlap Weights

**Milestone:** 2B (corrected)
**Status:** Production weights implemented with real CAMELSH basin polygons.
**Prerequisite:** Milestone 2A (grid-definition discovery, commit ae2a200)
**Required before:** Milestone 2C — one-hour basin-statistic extraction

---

## 1. Purpose

Basin-grid overlap weights answer: "for this basin and this grid product,
which grid cells contribute, and in what proportion?"

Precomputed per-basin, per-product weight tables enable efficient extraction:
- For each timestep: `basin_average = sum(cell_value[i] × normalized_weight[i])`
- No repeated polygon-grid intersection: weights are computed once and reused.

---

## 2. CAMELSH basin polygon source (production)

**Preferred polygon file:** `CAMELSH_shapefile.shp` (GAGES-II derived)

| Provenance field | Value |
|---|---|
| CAMELSH reference | DOI `10.5281/zenodo.16763144` (v7, published 2025-08-14) |
| Polygon source package | DOI `10.5281/zenodo.15066778` |
| File in package | `shapefiles.7z` → `shapefiles/CAMELSH_shapefile.shp` |
| Download size | ~506 MB (shapefile archive only; NOT the 21 GB time-series archive) |
| Polygon count | 9,008 basins (full CAMELSH coverage) |
| ID column | `GAGE_ID` (8-digit zero-padded USGS gauge ID) |
| CRS | No `.prj` file; EPSG:4326 (WGS84) assumed and assigned at load time |
| Pilot match | **50/50** pilot STAIDs matched |

**Secondary polygon file (comparison only):** `CAMELSH_shapefile_hydroATLAS.shp`
(HydroATLAS-derived; same Zenodo package; not used for Flash-NH production weights)

### Obtaining the shapefile

```
1. Download from DOI 10.5281/zenodo.15066778:
   https://zenodo.org/records/15066778/files/shapefiles.7z

2. Extract to {data_root}/02_basin_geometries/camelsh/
   7-Zip creates a shapefiles/ subdirectory:
   {data_root}/02_basin_geometries/camelsh/shapefiles/CAMELSH_shapefile.shp

3. The script auto-discovers this path — no config change required.
   Or set camelsh.basin_polygons in configs/pilot_stage1.yaml explicitly.
```

---

## 3. Default behavior and fallback guard

**Default: real polygons required.**
Running without polygons configured or discoverable causes an immediate error:

```
ERROR: No CAMELSH basin polygons configured or found.
Real basin polygons are REQUIRED for production weights.
To obtain the shapefile: ...
For SMOKE TESTING ONLY, re-run with --allow-fallback-circles.
```

**Circular-buffer fallback (smoke test only):**
```bash
python scripts/build_stage1_basin_weights.py \
    --config configs/pilot_stage1.yaml \
    --allow-fallback-circles
```
When `--allow-fallback-circles` is used:
- `geometry_method` in all outputs is `circular_buffer_test_only`
- `summary.md` displays a large `WARNING — TEST-ONLY WEIGHTS` banner
- Weights produced are **NOT valid** for training, extraction, or any scientific use

### Why circular buffers must not be used for production

Real drainage basins are irregular polygons shaped by topography and hydrology.
A circular buffer centred on the gauge gives:
- Correct centroid location ✓
- Correct total area ✓
- Incorrect shape ✗ — biased towards cells directly around the gauge

The resulting basin averages would be spatially biased: cells upstream or in
the contributing area but not near the gauge point would receive zero or low
weight. For precipitation events, this systematically misrepresents forcing.

---

## 4. Weighting method

All area calculations are performed in **EPSG:5070** (NAD83 / Conus Albers
Equal Area), which preserves area for CONUS locations.

### MRMS (regular lat/lon, 0.01°)

1. For each basin: find candidate cells via bounding-box lookup (basin bbox in EPSG:4326, ±2 cells).
2. Build cell corner coordinates in EPSG:4326 (four corners per cell).
3. Batch-transform corners to EPSG:5070 using pyproj (vectorised).
4. Create cell polygons in EPSG:5070 with shapely 2.x vectorised `shapely.polygons`.
5. Compute `overlap_area_m2` = area(basin ∩ cell) via shapely vectorised intersection.
6. `normalized_weight = overlap_area_m2 / sum(overlap_area_m2)` per basin.

### RTMA (Lambert Conformal Conic, ~2540 m)

1. Reconstruct RTMA LCC CRS: `+proj=lcc +lat_0=25 +lon_0=-95 +lat_1=25 +lat_2=25 +R=6371200 +units=m`
2. Compute grid origin `(x0, y0)` by projecting the SW corner (lat=19.229°, lon=−126.277°).
3. Cell (row j, col i): projected centre `(x0 + i*dx, y0 + j*dy)` where `dx=dy=2539.703m`.
4. Find candidate cells via basin bbox in RTMA projected coordinates.
5. Build cell corner coordinates in RTMA LCC space (exact squares of 2539.703m × 2539.703m).
6. Batch-transform corners to EPSG:5070 and compute intersections.

### Why product-native grids are kept separate

MRMS and RTMA are NOT resampled to a common grid. Each uses its native
coordinate system throughout weight computation. This follows the
`database_design.md` policy: *"Standardization does NOT mean forcing all
sources onto one shared spatial resolution."*

---

## 5. Weight table schema

Both `pilot_mrms_weights.parquet` and `pilot_rtma_weights.parquet` contain:

| Column | Type | Description |
|---|---|---|
| STAID | str | 8-digit zero-padded USGS gauge ID |
| product | str | `mrms_qpe_1h_pass1` or `rtma_conus_aws_2p5km` |
| row_idx | int | Grid row (MRMS: 0=north; RTMA: 0=south) |
| col_idx | int | Grid col (0=west) |
| grid_cell_id | str | `"{row:04d}_{col:04d}"` deterministic cell ID |
| lon_center | float | Cell centre longitude (degrees) |
| lat_center | float | Cell centre latitude (degrees) |
| x_center_m | float | Projected x in RTMA LCC (m); null for MRMS |
| y_center_m | float | Projected y in RTMA LCC (m); null for MRMS |
| overlap_area_m2 | float | Intersection area in EPSG:5070 (m²) |
| cell_area_m2 | float | Grid cell area in EPSG:5070 (m²) |
| raw_weight | float | = overlap_area_m2 |
| normalized_weight | float | raw_weight / sum(raw_weight) per STAID+product |
| geometry_method | str | `camelsh_shapefile` (production) or `circular_buffer_test_only` (smoke test) |

---

## 6. Production weight summary (real CAMELSH polygons)

From the run with `CAMELSH_shapefile.shp`:

| Metric | MRMS | RTMA |
|---|---|---|
| Total weight records | 17,910 | 3,522 |
| Basins with weights | 50 / 50 | 50 / 50 |
| Cells/basin (min/med/max) | 15 / 219 / 1235 | 6 / 48 / 228 |
| Weight sum (all basins) | exactly 1.000000 | exactly 1.000000 |
| Negative weights | 0 | 0 |
| `geometry_method` | `camelsh_shapefile` | `camelsh_shapefile` |

---

## 7. How to run

```powershell
# Production (real polygons required — auto-discovered or set in config)
python scripts/build_stage1_basin_weights.py --config configs/pilot_stage1.yaml

# With explicit data root
python scripts/build_stage1_basin_weights.py --config configs/pilot_stage1.yaml --data-root D:/Flash-NH_data

# Smoke test only (MUST NOT be used for production)
python scripts/build_stage1_basin_weights.py --config configs/pilot_stage1.yaml --allow-fallback-circles
```

Prerequisites:
```
{data_root}/09_manifests/stage1_pilot/pilot_basin_manifest.csv
{data_root}/09_manifests/stage1_pilot/grid_definitions/mrms_grid_definition.json
{data_root}/09_manifests/stage1_pilot/grid_definitions/rtma_grid_definition.json
{data_root}/02_basin_geometries/camelsh/shapefiles/CAMELSH_shapefile.shp  (auto-discovered)
```

---

## 8. Known limitations

1. **CAMELSH shapefile CRS**: No `.prj` file distributed with `CAMELSH_shapefile.shp`.
   EPSG:4326 (WGS84) is assumed. No polygon-area errors were observed during validation.
2. **RTMA sphere model**: NWS NDFD uses R=6371200m sphere, not the WGS84 ellipsoid.
   Position error is < 20km at grid edges; negligible for normalised weights.
3. **MRMS cell area variation**: cells at 55°N are ~40% smaller than cells at 20°N.
   EPSG:5070 projection accounts for this correctly.
4. **Full-scale computation** for 2,843 basins uses the same code with no architectural changes.

---

## 9. Next milestone: Milestone 2C — one-hour basin extraction

1. Select one MRMS sample file and one RTMA sample file (cached under `00_raw/`).
2. Decode each file with cfgrib (validated in Milestone 2A).
3. For each pilot basin, look up the precomputed weight rows.
4. Compute `basin_average = sum(grid_value[row,col] × normalized_weight)`.
5. Record: mean, std, min, max, p10–p99, valid_fraction, n_valid_pixels.
6. Write one Parquet row per basin × variable × timestep to `03_basin_timeseries/stage1_pilot/`.
7. Validate: non-NaN values, physically plausible ranges, weight sums ≈ 1.0.

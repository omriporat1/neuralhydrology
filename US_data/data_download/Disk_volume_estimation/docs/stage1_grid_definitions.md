# Flash-NH Stage 1 — Grid Definition Discovery

**Milestone:** 2A  
**Status:** Implemented  
**Prerequisite:** Stage 1 pilot skeleton (Milestone 1, commit 6190c81)  
**Required before:** Milestone 2B — basin-grid overlap weight computation

---

## 1. Purpose

Before computing basin-grid overlap weights, the pipeline needs to know each product's exact grid coordinate system. This step:

1. Downloads (or reuses cached) one sample GRIB2 file per product.
2. Decodes it with cfgrib/xarray to extract coordinate arrays and GRIB metadata.
3. Validates grid coverage and orientation.
4. Writes a JSON grid-definition file per product.
5. Generates a small geo-referenced PNG preview.

The JSON files produced here are the authoritative grid description used by downstream weight-computation and extraction code. Do not hard-code grid parameters elsewhere.

---

## 2. Confirmed grid properties

### MRMS QPE 1h Pass1

| Property | Value |
|---|---|
| Grid type | `regular_ll` (Equidistant Cylindrical / Plate Carree) |
| Shape | (3500, 7000) rows × cols |
| Latitude | 1D, 54.995°N → 20.005°N (north-to-south, **lat_descending=True**) |
| Longitude | 1D, -129.995° → -60.005° (west-to-east) |
| Resolution | 0.01° ≈ 1 km |
| GRIB `jScansPositively` | 0 (row 0 = northernmost row) |
| Bounding box | lon [-130, -60], lat [20, 55] |
| CONUS coverage | Yes (full CONUS domain) |
| Timestamp convention | Period-end / valid time of 1-hour accumulation |
| cfgrib coordinate names | `latitude` (1D), `longitude` (1D) |

Preview orientation: `imshow(arr, origin='upper')` → north-up.

### RTMA 2.5km NDFD Analysis

| Property | Value |
|---|---|
| Grid type | `lambert` (Lambert Conformal Conic) |
| Shape | (1597, 2345) rows × cols |
| Latitude | **2D** (1597, 2345), 19.23°N → 57.09°N (south-to-north, **lat_descending=False**) |
| Longitude | **2D** (1597, 2345), ~221.6°E → ~301.0°E (normalise: subtract 360) |
| Resolution | 2539.703 m ≈ 2.5 km |
| Projection | LCC: LaD=25°N, LoV=265°E (central meridian), Latin1=Latin2=25° |
| GRIB `jScansPositively` | 1 (row 0 = southernmost row) |
| Bounding box | lon [~-138, ~-59], lat [~19, ~57] |
| CONUS coverage | Yes (full CONUS NDFD domain) |
| Timestamp convention | Analysis valid time (instantaneous) |
| cfgrib coordinate names | `latitude` (2D), `longitude` (2D) |

Preview orientation: `imshow(arr, origin='lower')` → north-up (south at bottom since row 0 = south edge).

**Important for weight computation:** RTMA coordinates are 2D arrays, not 1D. Each grid cell has its own lat/lon value. The weight computation must use the full 2D arrays, not just the bounding box.

---

## 3. How to run

```powershell
# With default data root (tmp/stage1_pilot_dryrun/, gitignored)
python scripts/build_stage1_grid_definitions.py --config configs/pilot_stage1.yaml

# With external data root
python scripts/build_stage1_grid_definitions.py --config configs/pilot_stage1.yaml --data-root D:/Flash-NH_data

# Dry-run: decode cached files only, do not download
python scripts/build_stage1_grid_definitions.py --dry-run
```

If the sample files are not cached, the script downloads them via the existing validated datasource classes (`MrmsAwsQpe1hPass1`, `RtmaAwsConusDataSource`). No credentials are required; both use anonymous S3 access.

---

## 4. Outputs

```
{data_root}/
  00_raw/
    mrms/                            <- MRMS sample .grib2.gz (S3 key structure)
    rtma/                            <- RTMA sample .grb2_wexp (S3 key structure)

  09_manifests/stage1_pilot/grid_definitions/
    mrms_grid_definition.json        <- full grid metadata for MRMS
    rtma_grid_definition.json        <- full grid metadata for RTMA
    grid_definition_summary.json     <- cross-product summary + validation
    grid_definition_summary.md       <- human-readable summary
    manifest.json                    <- run provenance (command, config, git hash)
    summary.json                     <- compact pass/fail
    summary.md                       <- readable run summary
    run_command.txt
    git_commit.txt
    config_snapshot.yaml

  06_qc_reports/stage1_pilot/grid_definitions/
    mrms_grid_preview.png            <- geo-referenced precipitation field
    rtma_grid_preview.png            <- geo-referenced temperature field
```

None of these outputs are tracked by git (they go to the external data root or `tmp/`).

---

## 5. Why this precedes weight computation

Basin-grid weight computation requires:
1. **Basin polygon geometries** (CAMELSH GeoPackage, to be configured in `configs/pilot_stage1.yaml`)
2. **Exact grid coordinates** for each product (provided by this step)

For MRMS (regular_ll): the 1D lat/lon arrays define a uniform grid — weights can be computed by projecting basin polygons onto this regular grid.

For RTMA (lambert): the 2D lat/lon arrays are the actual geographic positions of each grid cell — weights must use these 2D arrays directly, not assume uniform spacing in degrees.

Both products use fractional area overlap (rasterisation or polygon intersection), not simple nearest-neighbour. The grid coordinate arrays from this step are the input to that computation.

---

## 6. Known limitations

- Only one sample hour is downloaded/decoded (2023-01-01T00:00:00 UTC by default). Grid geometry is stable over time for both products, so this is sufficient.
- MRMS GRIB2 metadata reports `GRIB_name = 'unknown'` and `GRIB_units = 'unknown'` — a known quirk of MRMS encoding. The physical variable is QPE precipitation in mm.
- RTMA preview uses the bounding-box approximation for the `imshow` extent, which slightly distorts the Lambert Conformal grid in geographic space. This is acceptable for QC purposes; the actual 2D coordinates are used for weight computation.
- The weight computation is not yet implemented (Milestone 2B).

---

## 7. Next milestone: Milestone 2B — Basin-grid overlap weights

Steps:
1. Load CAMELSH basin polygons (`camelsh.basin_polygons` in config).
2. Load MRMS 1D lat/lon coordinates (from `mrms_grid_definition.json` or re-decoded from sample).
3. Load RTMA 2D lat/lon coordinates (from `rtma_grid_definition.json` or re-decoded from sample).
4. Compute fractional overlap area between each pilot basin polygon and each grid cell.
5. Write Parquet weight tables to `{data_root}/02_basin_geometries/weights/mrms/` and `.../rtma/`.
6. Validate: weight sums ≈ 1.0 per basin, no negatives, plausible cell counts.

Required packages: `geopandas`, `rasterio` or `exactextract` (not yet in venv).

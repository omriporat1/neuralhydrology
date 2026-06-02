# Stage 1 Milestone 2C — One-Hour Basin-Statistic Extraction

## Purpose

This milestone validates the full extraction pipeline for one sample hour before
running full January 2023 extraction. It decodes real MRMS QPE and RTMA analysis
GRIB2 files, applies pre-computed CAMELSH polygon basin-grid weights, and writes
basin-level statistics for 50 pilot basins.

**Variable policy note**: not all decoded RTMA variables are equally suitable for
ML model training. The extraction deliberately over-decodes (capture all available
fields) while the output schema carries `variable_role` and
`recommended_for_initial_model` columns to guide feature selection downstream.
Two variables are excluded from the default dynamic output — see
[Variable Policy](#variable-policy) below.

---

## Prerequisites

- Milestone 2A complete: grid-definition JSONs under
  `{data_root}/09_manifests/stage1_pilot/grid_definitions/`
- Milestone 2B complete: weight Parquet files under
  `{data_root}/02_basin_geometries/weights/{mrms,rtma}/`
- Pilot basin manifest at
  `{data_root}/09_manifests/stage1_pilot/pilot_basin_manifest.csv`
- CAMELSH shapefile at
  `{data_root}/02_basin_geometries/camelsh/shapefiles/CAMELSH_shapefile.shp`
  (for focused QC maps; extraction proceeds without it)
- One MRMS sample file for 2023-01-01T00:00Z (auto-located or downloaded)
- One RTMA sample file for 2023-01-01T00:00Z (auto-located or downloaded)

---

## Command

```bash
python scripts/extract_stage1_one_hour.py \
    --config configs/pilot_stage1.yaml \
    --data-root tmp/stage1_pilot_dryrun
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/pilot_stage1.yaml` | YAML config path |
| `--data-root` | from config / env | Override data root directory |
| `--sample-time` | `2023-01-01T00:00:00` | Sample hour (ISO 8601 UTC) |
| `--products` | `mrms,rtma` | Comma-separated: `mrms`, `rtma`, or both |
| `--include-excluded-vars` | off | Also extract `10wdir` and `orog` (diagnostics only) |

The script reuses cached sample files on re-run; no re-download occurs.

---

## Input Files

| File | Description |
|------|-------------|
| `00_raw/mrms/.../MRMS_..._20230101-000000.grib2.gz` | MRMS QPE 1h Pass1 (gzip GRIB2) |
| `00_raw/rtma/rtma2p5.20230101/rtma2p5.t00z.2dvaranl_ndfd.grb2_wexp` | RTMA 2.5km analysis (GRIB2) |
| `02_basin_geometries/weights/mrms/pilot_mrms_weights.parquet` | MRMS basin-grid weights (CAMELSH polygons) |
| `02_basin_geometries/weights/rtma/pilot_rtma_weights.parquet` | RTMA basin-grid weights (CAMELSH polygons) |
| `09_manifests/stage1_pilot/pilot_basin_manifest.csv` | 50 pilot basin metadata |

---

## Output Schema

All output Parquet/CSV files share the same column schema:

| Column | Type | Description |
|--------|------|-------------|
| `STAID` | str | 8-digit zero-padded USGS station ID |
| `product` | str | `mrms_qpe_1h_pass1` or `rtma_conus_aws_2p5km` |
| `source` | str | Data source: `noaa-mrms-pds` or `noaa-rtma-pds` |
| `variable` | str | GRIB shortName (e.g. `unknown`, `2t`, `10u`) |
| `variable_standard_name` | str | CF-like standard name |
| `variable_role` | str | Role classification (see Variable Policy) |
| `circular_variable_flag` | bool | True if circular statistics are required |
| `recommended_for_initial_model` | bool | True for core dynamic candidates |
| `valid_time_utc` | str | ISO 8601 UTC timestamp |
| `issue_time_utc` | null | Not applicable for analyses |
| `lead_time_hours` | null | Not applicable for analyses |
| `units` | str | GRIB units string (see caveats below) |
| `weighted_mean` | float | Area-weighted basin mean (primary forcing value) |
| `unweighted_min` | float | Minimum cell value in basin |
| `unweighted_max` | float | Maximum cell value in basin |
| `unweighted_std` | float | Standard deviation of cell values |
| `unweighted_q10` … `unweighted_q99` | float | Percentiles (10, 25, 50, 75, 90, 95, 99) |
| `valid_cell_count` | int | Number of non-NaN grid cells for this basin |
| `total_weight` | float | Sum of normalised weights (~1.0) |
| `valid_weight_fraction` | float | Weight covered by non-NaN cells |
| `missing_value_fraction` | float | Fraction of cells with missing/NaN values |
| `weight_table_path` | str | Absolute path to weight Parquet used |
| `source_file_path` | str | Absolute path to source GRIB2 file used |

### Statistics Definitions

- **`weighted_mean`**: `Σ(w_i × v_i) / Σ(w_i)` over non-NaN cells (area-weighted).
  Primary basin forcing value for ML training.
- **`unweighted_*`**: Equal-weight statistics over all non-NaN cells.
  Retained for diagnostics and potential future model inputs.
- **`valid_weight_fraction`**: Sum of normalised weights for non-NaN cells.
  Should be ~1.0 with no missing data.
- **`total_weight`**: Sum of all normalised weights (~1.0 before missing masking).

---

## Variable Policy

### Philosophy

The pipeline **over-extracts** to support diagnostics and future architectural
experiments. Not all decoded variables are recommended for ML training. The
`variable_role` column documents each variable's intended use:

| Role | Meaning |
|------|---------|
| `core_dynamic_candidate` | Primary atmospheric forcing; recommended for initial model |
| `optional_dynamic_candidate` | Secondary forcing; useful for some architectures |
| `diagnostic_only` | QC use only; not recommended for training |
| `excluded_by_default` | Decoded but not written to default output |

### Default variable set

| Variable | Standard name | Role | Recommended |
|----------|--------------|------|------------|
| MRMS `unknown` | precipitation_amount_1h | core_dynamic | True |
| RTMA `2t` | air_temperature_2m | core_dynamic | True |
| RTMA `2d` | dewpoint_temperature_2m | core_dynamic | True |
| RTMA `2sh` | specific_humidity_2m | core_dynamic | True |
| RTMA `10u` | wind_u_component_10m | core_dynamic | True |
| RTMA `10v` | wind_v_component_10m | core_dynamic | True |
| RTMA `sp` | surface_pressure | optional_dynamic | False |
| RTMA `10si` | wind_speed_10m | optional_dynamic | False |
| RTMA `i10fg` | wind_gust_10m | optional_dynamic | False |
| RTMA `tcc` | total_cloud_cover | optional_dynamic | False |
| RTMA `ceil` | cloud_ceiling | diagnostic_only | False |
| RTMA `vis` | visibility | diagnostic_only | False |

### Excluded from default dynamic output

| Variable | Reason |
|----------|--------|
| RTMA `10wdir` | **Circular variable** — linear averaging is invalid. 359° and 1° should average near 0°, not 180°. Wind direction is redundant given `10u` and `10v`; if needed, derive it from basin-mean u/v or implement circular statistics explicitly. |
| RTMA `orog` | **Static terrain field** — orography is the terrain height embedded in the RTMA grid, not a dynamic meteorological variable. It does not change hour to hour. Including it in the hourly dynamic forcing table is misleading and wasteful. Basin elevation belongs in the static-attributes pipeline (e.g. from CAMELSH attributes). |

To include them for diagnostics, use `--include-excluded-vars`.

### Radiation note

Radiation variables (shortwave, longwave, net radiation) are not present in RTMA
and are **not extracted here**. Radiation is hydrologically important — particularly
for evapotranspiration, snowmelt, and antecedent soil drying — but must come from
products that provide it explicitly (ERA5-Land, GDAS, GFS, IFS, or NLDAS-2).

Stage 1 MRMS + RTMA extraction covers precipitation and near-surface atmospheric
state only. Full energy-balance forcing will be addressed in Stage 2 (ERA5-Land,
GDAS, or another radiation-providing product).

---

## Product-Specific Caveats

### MRMS QPE

- **Variable**: GRIB `shortName='unknown'`, `name='unknown'` — MRMS QPE GRIB2 encoding quirk.
- **Units**: `GRIB_units='unknown'` — units not encoded in MRMS GRIB2 files.
  Product is documented as **mm** (1-hour accumulation).
- **Grid**: `regular_ll`, 3500 × 7000, ~1 km, row 0 = northernmost.
- **Expected output**: 50 rows (one per basin).

### RTMA

- **Variables**: Up to 13 decoded; 2 excluded by default (10wdir, orog); 11 written.
- **Units**: From GRIB metadata (K for temperatures, Pa for pressure, etc.).
  No unit conversion is performed.
- **Grid**: Lambert Conformal, 1597 × 2345, ~2.5 km, row 0 = southernmost.
- **Expected output**: 50 rows × 11 variables = 550 rows (default).

---

## Output Files

```
{data_root}/
  03_basin_timeseries/stage1_pilot/one_hour/
    mrms_one_hour_basin_stats.parquet          # 50 rows
    rtma_one_hour_basin_stats.parquet          # 50 × 11 rows
    combined_one_hour_basin_stats.parquet      # 600 rows
    combined_one_hour_basin_stats_preview.csv  # human-readable

  06_qc_reports/stage1_pilot/one_hour_extraction/
    mrms_grid_preview_with_basins.png          # broad MRMS map, shared scale
    rtma_temperature_preview_with_basins.png   # broad RTMA 2t map, shared scale
    basin_value_histograms.png                 # distribution across basins
    weighted_mean_vs_q50_scatter.png           # weighted mean vs Q50 check
    focused/
      wet_mrms_<STAID>.png
      dry_mrms_<STAID>.png
      small_basin_mrms_<STAID>.png
      large_basin_mrms_<STAID>.png
      rtma_temperature_2t_<STAID>.png
      rtma_wind_u_10u_<STAID>.png

  09_manifests/stage1_pilot/one_hour_extraction/
    manifest.json  /  summary.json  /  summary.md  /  …
```

---

## Validation Checks

| Check | Description |
|-------|-------------|
| `mrms_source_file_exists_nonempty` | MRMS file exists and size > 0 |
| `rtma_source_file_exists_nonempty` | RTMA file exists and size > 0 |
| `mrms/rtma_weight_table_nonempty` | Weight Parquets non-empty |
| `mrms_grid_decoded` | MRMS grid decoded successfully |
| `rtma_grids_decoded` | At least one RTMA variable decoded |
| `mrms_50_basins_in_output` | Exactly 50 basins in MRMS output |
| `rtma_10wdir_excluded` | 10wdir absent from default RTMA output |
| `rtma_orog_excluded` | orog absent from default RTMA output |
| `rtma_50_basins_per_variable` | 50 × N_vars rows in RTMA output |
| `mrms/rtma_no_all_null_weighted_mean` | weighted_mean not all NaN |
| `mrms/rtma_valid_weight_fraction_reasonable` | valid_weight_fraction > 0.5 for all |
| `mrms_total_weight_close_to_1` | `|total_weight − 1| < 0.05` for all |
| `mrms/rtma_variable_role_column_present` | `variable_role` column present |
| `combined_parquet_written` | combined Parquet exists on disk |

---

## QC Plots

### MRMS units in plots

MRMS QPE GRIB2 files do not encode units metadata (`GRIB_units='unknown'`).
The output table preserves this raw GRIB field unchanged.
**All QC plots label MRMS precipitation as `mm`** (documented product units for
1-hour QPE accumulation). The discrepancy between schema (`unknown`) and plot
labels (`mm`) is intentional and documented here.

### Broad maps

**Purpose**: overview / smoke test — confirms that extraction values are
geographically plausible at continental scale.

Grid and basin-marker color scales use the **same `vmin`/`vmax`** for each map,
allowing direct visual comparison between grid-cell values and basin averages.
A single colorbar covers both.

Geographic context (country borders, US state boundaries) from Natural Earth 110m
is overlaid when available (cached locally after first download). Lat/lon axes are
retained for large-domain grid placement debugging.

### Focused maps

**Purpose**: primary geometry and extraction QC — confirms that basin polygons,
grid cells, and extracted weighted means are spatially consistent at basin scale.

Each focused map shows one basin at hydrologic scale:

- **Opaque raster** (background): source gridded field at full color — primary visual.
- **CAMELSH polygon** (crimson boundary): actual basin polygon used for weight computation.
- **Extraction cells** (hollow rings): grid cells contributing to the basin weighted mean;
  hollow (no colormap fill) to avoid duplicating the raster.
- **Basin weighted_mean** (◆ diamond): extracted value, same colormap and `vmin`/`vmax`
  as the raster — the key visual comparison between extraction result and source field.
- **Gauge location** (★ yellow star): USGS gauge position; distinct from the diamond.
- **Annotation box**: STAID, basin mean value, cell count — no legend.
- **Km scale bar**: approximate, using `1° lon ≈ 111.32 × cos(lat_center) km`.
- **State/coastline context**: where available.

Representative basins: wettest MRMS, driest MRMS, smallest by cell count, largest
by cell count, mid-latitude RTMA temperature, mid-latitude RTMA wind vector (10u + 10v).

The RTMA wind focused map is a vector plot: wind speed raster + quiver arrows +
crimson basin-mean arrow. RTMA u/v are grid-relative (LCC); rotation error at
QC scale is negligible.

### Histograms (`basin_value_histograms.png`)

**Purpose**: range check — confirms that extracted weighted means for the 50 pilot
basins fall within physically plausible ranges for each variable. Not a correctness
proof; outliers here warrant investigation but may be legitimate.

### Scatter (`weighted_mean_vs_q50_scatter.png`)

**Purpose**: rough sanity check — if area-weighted mean deviates substantially from
the unweighted median (Q50), it suggests the basin has a skewed distribution of
grid values (e.g. strong gradient, edge effect, or missing data). Points near the
1:1 line are expected for spatially smooth fields. Not a correctness proof.

---

## Target-Consistent Basin Geometry

### Why the polygon source matters for streamflow modeling

The Flash-NH **streamflow target** is USGS gauge discharge. A physically meaningful
forcing extraction requires that the basin polygon represents the drainage area that
actually contributes to the USGS station. Polygon accuracy is measured relative to
USGS-documented drainage area (`DRAIN_SQKM` in the CAMELSH static attributes).

### Active polygon source

| Property | Value |
|----------|-------|
| Active polygon file | `CAMELSH_shapefile.shp` |
| Source family | CAMELSH GAGES-II (GAGES-II polygon boundaries, USGS-derived) |
| ID field | `GAGE_ID` (8-digit zero-padded USGS station ID) |
| CRS | None embedded (EPSG:4326 assumed; WGS84 lat/lon coordinates) |
| Feature count | 9,008 |
| Pilot match | 50/50 |
| Median area error vs DRAIN_SQKM | **0.18%** |
| Max area error vs DRAIN_SQKM | 1.35% |

This is confirmed via code-path audit: `src/pipeline/geometries.py` defines
`resolve_polygon_path()` which searches only for `CAMELSH_shapefile.shp` in its
standard auto-discovery list. The HydroATLAS polygon (`CAMELSH_shapefile_hydroATLAS.shp`)
is defined as a named constant but is **never included in the resolution search path**.
It cannot be silently selected.

### Why HydroATLAS polygons are not suitable as primary forcing polygons

A comparison of both available CAMELSH polygon variants against DRAIN_SQKM was run
for all 50 pilot basins:

| Polygon source | Median area error vs DRAIN_SQKM | Max area error |
|---------------|--------------------------------|----------------|
| `CAMELSH_shapefile.shp` (GAGES-II) | **0.18%** | 1.35% |
| `CAMELSH_shapefile_hydroATLAS.shp` | 7.82% | **3,157%** |

The HydroATLAS polygons use a different delineation algorithm (HydroBASINS) that does
not reproduce USGS-reported drainage areas. For example, STAID 01199050 has
`DRAIN_SQKM = 76` km² but the HydroATLAS polygon gives 1,685 km² — a 2,114% error.
These are incorrect watershed boundaries for USGS-gauge-based streamflow modeling.

The median IoU between the two polygon sources is 0.82, meaning the polygons overlap
substantially on average, but individual basin mismatches can be severe.

**Recommendation: keep CAMELSH GAGES-II polygons.** HydroATLAS polygons are useful
for attribute comparison but are not a valid substitute for USGS-gauge forcing extraction.

### Coordinate source audit

The only gauge coordinate source available locally is `LAT_GAGE`/`LNG_GAGE` from
the CAMELSH pilot manifest. These coordinates originate from GAGES-II and are the
same source family as the active polygon. No USGS/NWIS native coordinate file was
found in the local data.

Coordinate-to-polygon status distribution for the 50 pilot basins
(using active GAGESII polygon):

| Status | Count | Interpretation |
|--------|-------|---------------|
| `INSIDE_NEAR_BOUNDARY_LE_250M` | 17 | Gauge near outlet — expected |
| `NEAR_OR_ON_BOUNDARY_LE_250M` | 6 | Gauge at boundary — expected |
| `OUTSIDE_250M_TO_1KM` | 10 | Minor offset — typical metadata mismatch |
| `OUTSIDE_1_TO_5KM` | 8 | Needs inspection (see WARN in audit) |
| `INSIDE_MODERATE_250M_TO_1KM` | 7 | Gauge moderately inside polygon |
| `INSIDE_DEEP_GT_1KM` or `RELATIVE_GT_0.10` | 2 | Gauge deep inside — unusual |

Gauges with `OUTSIDE_1_TO_5KM` (8 basins: 01100627, 01390450, 01490000, 02077670,
01115630, 01662800, 01585200, 03106300) should be inspected. The offsets likely
reflect GAGES-II coordinate/delineation differences rather than extraction errors,
as the forcing polygon areas match DRAIN_SQKM to within 1.4% for all pilot basins.

### Static attribute alignment

The pilot manifest static attributes (BFI_AVE, HYDRO_DISTURB_INDX, WATERNLCD06, etc.)
are from the CAMELSH dataset, which derives them from GAGES-II basin delineations.
They are consistent with the active forcing polygon. No HydroATLAS-derived attributes
were identified in the manifest. If HydroATLAS attributes are added in future (e.g.
from CAMELSH `camelsh_attributes_v2.0`), provenance columns should document which
attributes correspond to which basin delineation.

### Audit command

The spatial lineage audit is reproducible via:

```bash
python scripts/audit_stage1_spatial_lineage.py \
    --config configs/pilot_stage1.yaml \
    --data-root tmp/stage1_pilot_dryrun
```

The script uses `src.pipeline.geometries.resolve_polygon_path` to locate the active
polygon (same resolver used by weight computation), so it is always consistent with
the pipeline.

### Audit outputs

Written to `09_manifests/stage1_pilot/one_hour_extraction/`:

| File | Contents |
|------|----------|
| `spatial_source_inventory.csv/.json` | All locally available spatial sources |
| `coordinate_source_vs_active_polygon.csv/.json` | Gauge coordinate distances to active polygon |
| `polygon_candidate_target_consistency.csv/.json` | Per-basin metrics for both polygon candidates |
| `polygon_source_target_consistency_ranking.csv` | Summary ranking of polygon sources |
| `polygon_area_vs_static_area_audit.csv` | Geometry area vs DRAIN_SQKM per basin and polygon source |
| `spatial_lineage_audit_summary.json/.md` | Human-readable and machine-readable summary |

### Code-path summary

| Aspect | Finding |
|--------|---------|
| Polygon source resolution | `geometries.resolve_polygon_path()` searches only for `CAMELSH_shapefile.shp` |
| HydroATLAS silently used? | **No** — HydroATLAS path not in auto-discovery list |
| Gauge coordinate source | `LAT_GAGE`/`LNG_GAGE` from `pilot_basin_manifest.csv` |
| ID normalization | `normalise_staid()` = `lstrip("0").zfill(8)` applied consistently |
| Polygon source ambiguity | None — GAGES-II polygon always selected |

---

## Gauge Location vs. Basin Polygon

The gauge coordinates (`LAT_GAGE`, `LNG_GAGE`) in the pilot manifest are USGS
stream gauge locations. They are used **only as a visual reference** in QC maps.

**Extraction uses CAMELSH polygon weights — not the gauge point.**

Small gauge-polygon offsets are expected and harmless:

- USGS gauge coordinates and CAMELSH polygon delineations come from different sources
  and may use slightly different datums or reference frames.
- Stream gauges are often located at a hydraulic structure (bridge, weir) that may
  be at the edge of, or slightly outside, the delineated basin polygon.
- Offsets up to a few hundred metres are typical metadata variability.

A gauge-to-polygon distance audit is run automatically each time the script
executes. Results are saved to:

```
06_qc_reports/stage1_pilot/one_hour_extraction/gauge_polygon_distance_audit.csv
09_manifests/stage1_pilot/one_hour_extraction/gauge_polygon_distance_audit.json
```

**Warning thresholds** (informational only; do not fail extraction):

| Category | Distance | Action |
|----------|----------|--------|
| `INSIDE_OR_ON_BOUNDARY` | 0 m | Expected |
| `NEAR_POLYGON_LE_250M` | ≤ 250 m | Expected; minor metadata mismatch |
| `OFFSET_250M_TO_1KM` | 250 m – 1 km | Note; investigate if time permits |
| `OFFSET_GT_1KM` | > 1 km | **WARN** — manual inspection recommended |
| `OFFSET_GT_5KM` | > 5 km | **WARN_STRONG** — likely data issue |

Focused maps annotate basins where the gauge is > 1 km outside the polygon.

---

## Why This Precedes Full January Extraction

1. **Validates grid indexing** — confirms row/col mapping before 744 hourly grids.
2. **Validates weight alignment** — confirms all 50 basins match and weights sum correctly.
3. **Confirms unit handling** — flags unknown units before large-scale output.
4. **Reference values** — 2023-01-01T00Z statistics serve as regression test baselines.
5. **Fast feedback** — runs in ~20 seconds, not minutes.

---

## Next Step

**Milestone 2D — Full January 2023 pilot extraction**:

```bash
python scripts/extract_stage1_january.py \
    --config configs/pilot_stage1.yaml \
    --data-root <your-data-root>
```

Processes all 744 hourly MRMS + RTMA files for January 2023, producing full
time-series Parquet files for the 50 pilot basins.

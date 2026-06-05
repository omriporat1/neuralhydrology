# Stage 1 January 2023 Pilot Extraction

## Purpose

Milestone 2D scales the validated one-hour extraction (Milestone 2C) to the full
configured pilot time range: **2023-01-01T00:00:00 – 2023-01-31T23:00:00 UTC, hourly**.

Two equally important goals:

1. **Functional extraction** — Produce a complete January 2023 hourly basin forcing
   dataset for all 50 pilot basins (MRMS QPE + 11 RTMA variables).

2. **Resource/scaling evaluation** — Measure download volume, raw-cache volume,
   processed-output volume, and timing breakdowns (download / decode / extract / write)
   per product/hour. Use those measurements to project full Flash-NH Stage 1 storage
   and compute requirements (2,843 basins, 2020–2025).

## Command to Run

```bash
# Smoke test (first 3 hours — always run this first):
python scripts/extract_stage1_january.py \
    --config configs/pilot_stage1.yaml \
    --data-root tmp/stage1_pilot_dryrun \
    --max-hours 3

# Full January extraction (default: selected-message RTMA mode):
python scripts/extract_stage1_january.py \
    --config configs/pilot_stage1.yaml \
    --data-root tmp/stage1_pilot_dryrun

# Resume an interrupted run (skips hours that already have staging parquets):
python scripts/extract_stage1_january.py \
    --config configs/pilot_stage1.yaml \
    --data-root tmp/stage1_pilot_dryrun \
    --resume

# Explicit full-file RTMA mode (debug / fallback):
python scripts/extract_stage1_january.py \
    --config configs/pilot_stage1.yaml \
    --data-root tmp/stage1_pilot_dryrun \
    --rtma-mode full_file

# Override parallel download workers (default 4; use 1 for serial, 8 for HPC):
python scripts/extract_stage1_january.py \
    --config configs/pilot_stage1.yaml \
    --data-root tmp/stage1_pilot_dryrun \
    --download-workers 8

# Single-product run (MRMS only):
python scripts/extract_stage1_january.py \
    --config configs/pilot_stage1.yaml \
    --data-root tmp/stage1_pilot_dryrun \
    --products mrms

# Override time range:
python scripts/extract_stage1_january.py \
    --config configs/pilot_stage1.yaml \
    --data-root tmp/stage1_pilot_dryrun \
    --start 2023-01-15T00:00:00 \
    --end   2023-01-15T23:00:00
```

## Inputs

| Input | Path (relative to data_root) |
|-------|------------------------------|
| Pilot basin manifest | `09_manifests/stage1_pilot/pilot_basin_manifest.csv` |
| MRMS weight table | `02_basin_geometries/weights/mrms/pilot_mrms_weights.parquet` |
| RTMA weight table | `02_basin_geometries/weights/rtma/pilot_rtma_weights.parquet` |
| MRMS raw files | `00_raw/mrms/**/*.grib2.gz` (downloaded or cached) |
| RTMA raw files | `00_raw/rtma/**/*.grb2_wexp` (downloaded or cached) |

Weight tables are produced by Milestone 2B (`build_stage1_basin_weights.py`).
Raw files are downloaded on demand from AWS S3 (`noaa-mrms-pds`, `noaa-rtma-pds`).

## Output Schema and Partitioning

### Basin time-series Parquet files

All three output Parquet files share the same schema as the one-hour extraction
(see `src/pipeline/extraction.py`, `STAT_COLUMNS`):

| Column | Type | Description |
|--------|------|-------------|
| STAID | str | 8-digit zero-padded USGS gauge ID |
| product | str | `mrms_qpe_1h_pass1` or `rtma_conus_aws_2p5km` |
| source | str | S3 bucket name |
| variable | str | GRIB shortName |
| variable_standard_name | str | CF-like standard name |
| variable_role | str | role classification |
| circular_variable_flag | bool | True for wind direction |
| recommended_for_initial_model | bool | recommended flag |
| valid_time_utc | str | ISO 8601 UTC valid time |
| issue_time_utc | str | None for analysis products |
| lead_time_hours | float | None for analysis products |
| units | str | GRIB units string |
| weighted_mean | float | area-weighted basin mean |
| unweighted_min/max/std | float | unweighted statistics |
| unweighted_q10/q25/q50/q75/q90/q95/q99 | float | unweighted percentiles |
| valid_cell_count | int | grid cells with finite values |
| total_weight | float | sum of normalised weights |
| valid_weight_fraction | float | fraction of weight from valid cells |
| missing_value_fraction | float | fraction of cells with NaN/Inf |
| weight_table_path | str | path to the Parquet weight table used |
| source_file_path | str | path to the raw GRIB2 file used |

#### File layout (not partitioned — single files per product):

```
03_basin_timeseries/stage1_pilot/january_2023/
    mrms_hourly_basin_stats.parquet       # 744 hours × 50 basins = 37,200 rows
    rtma_hourly_basin_stats.parquet       # 744 hours × 50 basins × 11 vars = 409,200 rows
    combined_hourly_basin_stats.parquet   # 446,400 rows total
    preview_mrms.csv                      # first 5 rows (not full export)
    preview_rtma.csv                      # first 5 rows
```

Intermediate staging (for resume support):
```
tmp/january_2023_staging/mrms/<YYYYMMDDHH>.parquet   # 50 rows per hour
tmp/january_2023_staging/rtma/<YYYYMMDDHH>.parquet   # 550 rows per hour
```

## Expected Row Counts (full January 2023)

| Product | Formula | Expected rows |
|---------|---------|--------------|
| MRMS | 744 hours × 50 basins | 37,200 |
| RTMA | 744 hours × 50 basins × 11 variables | 409,200 |
| Combined | — | 446,400 |

If missing source files reduce the hour count, the script reports exact
missing hours in `missing_files.csv` and adjusts expected counts accordingly.

## RTMA Variable Policy

**11 variables included** (default):

| GRIB shortName | Standard name | Role |
|----------------|---------------|------|
| 2t | air_temperature_2m | core_dynamic_candidate |
| 2d or 2sh | dewpoint / specific_humidity_2m | core_dynamic_candidate |
| 10u | wind_u_component_10m | core_dynamic_candidate |
| 10v | wind_v_component_10m | core_dynamic_candidate |
| sp | surface_pressure | optional_dynamic_candidate |
| tcc | total_cloud_cover | optional_dynamic_candidate |
| vis | visibility | diagnostic_only |
| ceil | cloud_ceiling | diagnostic_only |
| 10si | wind_speed_10m | optional_dynamic_candidate |
| i10fg | wind_gust_10m | optional_dynamic_candidate |

*Note: Exact variable count may vary by hour depending on RTMA file contents. The
policy requires exactly 10wdir and orog to be absent.*

**Excluded by default:**

- `10wdir` — circular variable; linear averaging is invalid (359° + 1° ≠ 180°).
  Wind direction can be derived from the u/v components.
- `orog` — static terrain field; belongs with static attributes, not dynamic forcings.

## RTMA Acquisition Mode

The `--rtma-mode` flag controls how RTMA files are downloaded when not cached locally.

### Mode comparison (benchmarked 2026-06-05, 2023-01-01T00:00Z)

The RTMA `2dvaranl_ndfd.grb2_wexp` file contains **13 GRIB2 messages** in the `.idx`:

```
hgt, pres, tmp, dpt, ugrd, vgrd, spfh, wdir, wind, gust, vis, ceil, tcdc
```

Stage 1 selects **11 messages** (excludes `hgt`/orography and `wdir`/wind-direction).

| Metric | full_file (conservative) | selected_messages (default) |
|--------|--------------------------|------------------------------|
| File/download bytes | 84.3 MB | 71.2 MB (-16%) |
| Download time (local, ~15 Mbps) | ~43 s | ~29 s (-33%) |
| .idx overhead | none | 0.7 KB in 0.8 s |
| Decode time (cfgrib) | ~3.5 s | ~1.3 s (-63%) |
| Variables decoded | 11 | 11 (identical) |
| Output rows | 550 | 550 (identical) |
| Equivalence | baseline | **EQUIVALENT** (max \|diff\| = 0.0) |
| S3 Range requests | 1 GetObject (no Range) | 2 merged GetObject + Range |

**Full-period estimates (52,608 hours, 2020–2025):**

| | full_file | selected_messages |
|---|---|---|
| Raw download cache | 4.43 TB | 3.74 TB (-0.7 TB) |
| Serial wall-clock | ~690 h | ~449 h |
| 32 HPC tasks | ~22 h | ~14 h |
| 128 HPC tasks | ~5.4 h | ~3.5 h |

> **Caveat**: estimates assume ~15 Mbps local network. On HPC with 10 Gbps, absolute
> times will be much lower and the relative advantage of selected_messages will shrink.
> The **storage saving** (0.7 TB) is bandwidth-independent and is the primary HPC benefit.

### Implementation

- **`selected_messages`** (default): fetches `.idx` from S3, identifies the 11 Stage 1
  GRIB message byte ranges, downloads only those ranges using HTTP `Range` headers.
  Uses merged ranges to minimise S3 API calls (typically 2 requests).
  Falls back to `full_file` automatically if `.idx` is unavailable.

- **`full_file`**: downloads the entire `grb2_wexp` file with a single S3 `GetObject`
  (no `Range` header). Simpler; useful as a debug fallback.

- **Cache reuse**: cached files are always reused regardless of which mode created them.
  A file downloaded in `full_file` mode can be reused by `selected_messages` and
  vice versa (the subset decodes identically for Stage 1 purposes).

### Why _selected_targets() in rtma.py needed extending

The `measure_selected_variable_bytes()` method (used for size estimation) had
`_selected_targets()` covering only: TMP, SPFH/DPT, UGRD, VGRD, PRES, TCDC.
This missed 4 Stage 1 variables (10si, i10fg, vis, ceil), which appear in the .idx
as `wind`, `gust`, `vis`, `ceil` respectively.

`_selected_targets()` was extended (2026-06-05) to include WIND, GUST, VIS, CEIL
families. The size-estimation method is now more accurate for Stage 1.

**Important**: the DPT/SPFH mutual exclusion in `_is_selected_short_name()` applies
only to `measure_selected_variable_bytes()` (size estimation). The production download
method `download_selected_messages()` does **not** apply mutual exclusion — it
includes **both** `dpt` (→ 2d) and `spfh` (→ 2sh) messages because Stage 1 requires
both variables.

## Caching and Download Behaviour

The script minimises redundant S3 API calls:

1. **S3 listing**: `list_sample_objects()` is called **once per product** for the
   full range at startup, returning all available S3 objects as a datetime→object map.

2. **Local cache index**: All existing raw files are indexed once at startup (O(n_files)),
   enabling O(1) per-hour cache lookups.

3. **Per-hour download**: Only files absent from the local cache are downloaded.
   Downloaded files are added to the in-memory cache index for reuse within the same run.

4. **Resume support** (`--resume`): If a per-hour staging parquet exists and was
   previously marked successful in `hourly_runtime_and_volume.csv`, extraction is
   skipped and the staging file is loaded directly.

5. **Missing S3 objects**: Hours where the source file does not exist in S3 are
   recorded in `missing_files.csv` with `reason=not_in_s3`. Processing continues.

## Validation Checks

The manifest validation table reports pass/fail for each check. Key checks:

| Check | Description |
|-------|-------------|
| `mrms_extracted_hours_gt_zero` | At least one successful MRMS hour |
| `mrms_50_basins_per_ok_hour` | Exactly 50 rows per successful MRMS hour |
| `rtma_11_variables` | Exactly 11 RTMA variables present in output |
| `rtma_10wdir_absent` | 10wdir not present in RTMA output |
| `rtma_orog_absent` | orog not present in RTMA output |
| `rtma_50_basins_x_11_vars_per_ok_hour` | Expected RTMA row count matches |
| `mrms_no_all_null_weighted_mean` | MRMS weighted_mean is not all NaN |
| `rtma_no_all_null_weighted_mean` | RTMA weighted_mean is not all NaN |
| `mrms_parquet_written` | MRMS output Parquet exists on disk |
| `rtma_parquet_written` | RTMA output Parquet exists on disk |
| `combined_parquet_written` | Combined output Parquet exists |
| `hourly_runtime_volume_csv_written` | Timing CSV exists |
| `scaling_estimates_json_written` | Scaling estimates JSON exists |

## QC Plots

Plots are written to `06_qc_reports/stage1_pilot/january_2023_extraction/`:

| File | Description |
|------|-------------|
| `hourly_availability.png` | Per-product hourly success/missing timeline |
| `runtime_by_hour.png` | Total processing time per hour, median line |
| `raw_file_size_by_hour.png` | Raw file size (MB) per hour, median line |
| `cumulative_volume.png` | Cumulative raw cache + processed Parquet (GB) |
| `representative_timeseries.png` | MRMS precip + RTMA 2m T for 3 pilot basins |
| `variable_completeness_rtma.png` | RTMA variable completeness (%) |
| `basin_completeness_distribution.png` | Histogram of per-basin hour completeness |
| `full_dataset_storage_estimate.png` | Estimated full-dataset raw + Parquet storage (GB) |

## Resource Measurement Methodology

The following metrics are recorded per (product, hour) in
`09_manifests/stage1_pilot/january_2023_extraction/hourly_runtime_and_volume.csv`:

| Column | Description |
|--------|-------------|
| `product` | Product identifier |
| `valid_time_utc` | Hour timestamp (ISO 8601 UTC) |
| `raw_file_path` | Local path to the raw GRIB2 file |
| `raw_file_size_bytes` | On-disk size of the raw file |
| `file_reused` | True if file was already in local cache |
| `download_time_s` | Time to download from S3 (0 if reused) |
| `decode_time_s` | Time to decode GRIB2 to VariableGrid array(s) |
| `extraction_time_s` | Time to compute all basin statistics |
| `write_time_s` | Time to write per-hour staging Parquet |
| `total_processing_time_s` | Sum of all phases including download |
| `n_output_rows` | Rows written for this hour/product |
| `output_parquet_bytes` | Size of the per-hour staging Parquet file |
| `status` | `success`, `missing_s3`, `download_failed`, `decode_extract_error`, `empty` |
| `warning_message` | Error message for non-success rows |

**Note on MRMS decompressed size**: MRMS files are stored compressed (`.grib2.gz`).
The `raw_file_size_bytes` reflects the compressed file size. Uncompressed size is
approximately 3500 × 7000 × 4 bytes ≈ 98 MB per file; the compressed files are
typically 5–15 MB (6–20× compression).

## Raw Download Volume vs. Processed Output Volume

These scale differently with basin count and must not be conflated:

| Volume type | Scales with | Why |
|-------------|-------------|-----|
| Raw MRMS download | Hours | One CONUS-wide file per hour; same file regardless of basin count |
| Raw RTMA download | Hours | One CONUS-wide file per hour per analysis cycle |
| Processed MRMS Parquet | Hours × basins | One row per basin per hour |
| Processed RTMA Parquet | Hours × basins × variables | One row per basin × variable per hour |

To go from 50 pilot basins to 2,843 basins:
- Raw download volume is **unchanged** (same CONUS files)
- Processed Parquet grows by 56.86× (2843/50)

## Scaling Estimate Assumptions

From the January pilot measurements:

- `raw_bytes_per_hour` = total raw bytes / successful hours (per product)
- `parquet_bytes_per_basin_hour` = total Parquet bytes / (successful hours × 50 basins)
- `median_processing_s_per_hour` = median of `total_processing_time_s` values

Full-dataset extrapolations:

| Parameter | Pilot | Full |
|-----------|-------|------|
| Hours | 744 | 52,608 |
| Basins | 50 | 2,843 |
| Period | Jan 2023 | 2020–2025 |

Serial wall-clock time = `median_processing_s_per_hour` × 52,608.
HPC estimates assume ideal linear parallel scaling at file level
(one process per file/hour); scheduler overhead and I/O contention not modelled.

## Local vs. HPC Implications

**Local machine**: Serial processing is feasible for 744 hours (January pilot)
but impractical for 52,608 hours (full 6-year dataset). Key bottlenecks:
- S3 download speed (typically 10–50 MB/s)
- RTMA GRIB2 decode time (multi-message, involves cfgrib)

**HPC / SLURM**: Recommended approach for the full dataset:
- Array job with one task per day (365/366 tasks per year × 6 years)
- Each task processes 24 hours for all basins for one product
- Or one task per product per day (2 products × ~2190 days = 4,380 tasks)
- Shared NFS/Lustre for raw file cache avoids duplicate downloads

The weight tables (pilot_mrms_weights.parquet, pilot_rtma_weights.parquet) are
read-only inputs; multiple tasks can read them concurrently without locking.

## Known Caveats

- MRMS GRIB2 files encode `GRIB_units='unknown'`; the physical unit is millimetres
  per hour accumulation (documented product convention, not in file metadata).
- RTMA files occasionally lack cloud cover (tcc) or ceiling (ceil) variables;
  these hours will show fewer than expected rows for those variables.
- RTMA humidity variable may appear as `2sh` (specific humidity) or `2d` (dewpoint)
  depending on the analysis hour. Both are decoded; output rows reflect whichever
  is present.
- S3 listing latency for the full January range (31 days × 2 products) is typically
  5–30 seconds at the start of the run.
- First-run download times dominate for uncached files. Subsequent runs (with
  `--resume`) will be substantially faster.

## RTMA Full-CONUS Download-Speed Optimization

This section documents the download-speed audit conducted on 2026-06-05.

### Why the storage saving from selected_messages mode was modest

The `selected_messages` mode (default) downloads 11 of 13 GRIB2 messages,
excluding only `hgt` (orography, static) and `wdir` (wind direction, circular).
Since 11/13 messages are selected, the savings are inherently bounded:
**71 MB instead of 84 MB per file (-16%)**.

This is the correct trade-off: keeping the full CONUS grid and all 11 Stage 1
variables is the project requirement. Spatial subsetting by HUC or basin tile
is intentionally **not used** as the primary strategy — the full CONUS RTMA grid
is retained for future projects.

### Why single-stream S3 is slow (~2–3 MB/s)

Each S3 `GetObject` call opens one TCP connection. For `noaa-rtma-pds` (public
unsigned bucket in `us-east-1`), this yields ~2–3 MB/s per connection — consistent
with a residential broadband connection (~15–25 Mbps per stream). The bottleneck is
**S3 per-connection rate limiting**, not total client bandwidth.

Evidence: with `download_file()` + `TransferConfig(max_concurrency=10)` (10 internal
Range GET threads for the same file), throughput jumps to **7.4 MB/s**. With 4 parallel
files (each on one stream), aggregate throughput reaches **8.3 MB/s**.

### Single-file download mechanism benchmark (t03z, 83.8 MB)

Benchmarked 2026-06-05 on a ~15–25 Mbps residential connection:

| Mechanism | Bytes | DL time | MB/s | Vars | Notes |
|-----------|-------|---------|------|------|-------|
| A: boto3 GetObject (current default) | 83.8 MB | 31.6 s | 2.65 | 13 | single TCP stream |
| B: selected-message Range GETs | 70.7 MB | 31.8 s | 2.23 | 11 | 2 merged ranges + .idx fetch |
| **C: boto3 Transfer Manager** | **83.8 MB** | **11.3 s** | **7.40** | **13** | **10 concurrent Range GETs** |
| D: AWS CLI | — | — | — | — | not installed |
| E: httpx HTTPS | 83.8 MB | 30.5 s | 2.75 | 13 | HTTP/1.1 single stream |
| F: requests HTTPS | 83.8 MB | 57.9 s | 1.45 | 13 | slower than boto3 |
| G: Herbie | — | — | — | — | not installed |

**Key finding**: `boto3` S3 Transfer Manager with `max_concurrency=10` achieves
**7.4 MB/s** (2.8× faster) by splitting the file into ~10 MB chunks and downloading
them over 10 concurrent `Range GET` connections. This is a drop-in replacement using
only the already-installed `boto3` library.

Selected-message mode (B) does NOT benefit from the Transfer Manager approach because
the 2 merged ranges are already as few connections as possible. The per-connection
rate limit (~2.3 MB/s) is the ceiling for each Range GET.

### Multi-file concurrency benchmark (8 × 83.8 MB = 670 MB, Mode A)

| Workers | Total time | Agg MB/s | Speedup vs w=1 |
|---------|-----------|----------|----------------|
| 1 | 305.6 s | 2.19 | 1.00× |
| 2 | 140.1 s | 4.79 | 2.18× |
| 4 | 81.1 s | 8.27 | 3.77× |
| **8** | **42.0 s** | **15.95** | **7.27×** |

Scaling is near-linear up to w=4, then saturates. For a January 2023 run (744 hours),
w=4 reduces RTMA download time from ~9.1 h to ~2.4 h.

### File-level parallel acquisition (--download-workers)

The extraction script pre-fetches all RTMA files in parallel before the serial
decode/extract loop. This separates the I/O-bound download phase from the
CPU-bound decode/extract phase:

```
Phase 1 (parallel): N workers download all uncached RTMA files concurrently
Phase 2 (serial):   decode → extract → write staging Parquet for each hour
```

Only the RTMA download is parallelised; MRMS stays serial (files are ~1.2 MB,
negligible download time). The decode/extract phase is kept serial to avoid
cfgrib threading issues and to maintain deterministic output ordering.

**CLI flag**: `--download-workers N` (default: **4**)

- `--download-workers 1`: fully serial, equivalent to the original design
- `--download-workers 4`: recommended local default
- `--download-workers 8`: recommended HPC per-job setting

**Cache behaviour**: Pre-fetch only downloads files not already in the local cache.
On `--resume`, files with existing staging Parquets are skipped entirely.

### Recommended approach

**Local (debugging / smoke tests)**:
- Use `--rtma-mode selected_messages --download-workers 4` (defaults).
- Files hit local cache after first run; repeat runs are fast (no re-download).
- Use `--rtma-mode full_file` as a debug fallback if .idx is unavailable.

**HPC production (recommended)**:
1. Use SLURM array jobs: 1 task per day (or 1 per product-day).
2. Each task uses `--download-workers 8` (8 parallel RTMA downloads per task).
3. At 10 Gbps HPC, download time drops to <0.1 s/file; decode/extract dominate.
4. Shared NFS/Lustre raw cache avoids duplicate downloads across tasks.
5. Do NOT use spatial subsetting — keep full CONUS grid.

### Transfer Manager implementation note

To upgrade `RtmaAwsConusDataSource._download_s3_object()` to use the Transfer Manager:

```python
from boto3.s3.transfer import TransferConfig
_TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=8 * 1024 * 1024,
    max_concurrency=10,
    multipart_chunksize=8 * 1024 * 1024,
    use_threads=True,
)
# In download:
s3_client = boto3.client("s3", config=BotoConfig(
    signature_version=UNSIGNED,
    max_pool_connections=25,  # >= max_concurrency + buffer
))
s3_client.download_file(bucket, key, str(out_path), Config=_TRANSFER_CONFIG)
```

This change gives 2.8× speedup on a residential connection and likely larger gains
on HPC (where per-stream throughput is already high, so Transfer Manager overhead is minimal).

This change is not yet merged to `RtmaAwsConusDataSource` (requires broader validation
across 2020–2025 file range). Tracked for Stage 1 full-run implementation.

### Why NOT spatial subsetting / HUC tiling

Spatial subsetting (cropping RTMA to a bounding box or HUC region) would require
wgrib2 or cfgrib-based reprojection and is GRIB2-format specific. It would:
- Reduce per-file size by a basin-count-dependent factor (50 basins ≈ not much)
- Complicate the pipeline (additional preprocessing step)
- Lose the full CONUS grid potentially needed for future work
- Not be available via S3 byte-range without GRIB2 spatial indexing

Message-level selection (current `selected_messages` mode) achieves the same "only
download what you need" philosophy but at the GRIB message (variable) level, which
aligns naturally with the S3 `.idx` structure.

## Next Milestone: 2E Event/Window Animations with Hydrographs

Milestone 2E will:
- Identify 2–3 significant precipitation events in January 2023 from the extracted data
- Generate animated MRMS/RTMA maps for event windows
- Overlay streamflow hydrographs (CAMELSH hourly Q) for selected basins
- Compare basin-averaged MRMS precipitation to gauge-observed runoff

# Flash-NH Stage 1 — Full-Period USGS IV Streamflow Target-Source Design

**Milestone:** 2I-A (design/audit-planning only)  
**Date:** 2026-06-11  
**Status:** Design document — no bulk download, no model training, no generated data

---

## 1. Scope and Non-Goals

### In scope

- Design of the canonical full-period streamflow target source for Flash-NH Stage 1.
- Specification of acquisition strategy, output schema, gap audit design, and
  HPC/SLURM structure for future implementation milestones.
- Planning estimates for storage and runtime.
- Acceptance criteria for the next pilot milestone (2I-B).

### Not in scope / guardrails for this milestone

- No bulk USGS IV download.
- No full-period dataset build (HPC or local).
- No model training.
- No modification of existing 2G or 2H generated packages.
- No large generated output files.
- No push to remote.

### Relationship to prior work

The January 2023 dry-run package (Milestone 2H-D, commit `0595384`) intentionally
mixed local CAMELSH hourly data and recovered USGS IV for a pilot/smoke rebuild.
That mixing was acceptable for a one-month smoke test but is **not the target policy
for full-period HPC packaging**.

CAMELSH remains valuable for:
- Static attribute files (basin geometry, physiographic indices, GAGES-II fields).
- Polygon/shapefile alignment.
- Discovery and catalog cross-referencing.
- Comparison diagnostics against USGS IV.

For the full-period research targets (`qobs_m3s`) entering NeuralHydrology training,
the canonical source is **USGS NWIS Instantaneous Values (IV) only** — as specified below.

---

## 2. Target-Source Policy

| Field | Value |
|---|---|
| **Canonical source** | USGS NWIS Instantaneous Values (IV) |
| **API endpoint** | `https://waterservices.usgs.gov/nwis/iv/` |
| **Parameter code** | `00060` (Discharge, cubic feet per second) |
| **Target variable name** | `streamflow` in CAMELSH-like NC; `qobs_m3s` in NeuralHydrology package NC |
| **Units** | m³ s⁻¹ |
| **Conversion** | ft³/s × 0.028316846592 |
| **Time zone** | UTC throughout; no local-time conversion |
| **Target grid** | Uniform hourly; one observation per integer UTC hour |
| **Missing values** | `NaN` — no fill, no interpolation, no sentinel |
| **Encoding sentinel** | `_FillValue = -9999.0` in NeuralHydrology package NC only; canonical files use native NaN |
| **No interpolation** | Strict; never interpolate across a gap |
| **No sentinel in canonical NC** | `NaN` only in the streamflow-only canonical file |
| **Provenance** | Preserved in per-basin NC attributes and per-basin acquisition log |

### Timestamp snapping policy (provisional)

For each target UTC hour `T`:

1. If an observation exists at exactly `T` (to the second) → use it. Method: `exact`.
2. Else if an observation exists within `[T − Δ, T + Δ]` → use the nearest. Method: `nearest_within_tolerance`.
3. Else → `NaN`. Method: `missing`.

Current tolerance: `Δ = 15 minutes` (inherited from Milestones 2H-C/2H-D).

**Open question (Section 13, Q1):** Is ±15 min the correct tolerance for all cadences?
At 15-min cadence, ±15 min always finds exactly one candidate. At 5-min cadence,
up to three candidates exist and the nearest is chosen. The policy is defensible but
should be documented explicitly in the acceptance criteria for 2I-B.

**Systematic-offset flag:** If a station's median raw IV cadence is ≥ 60 minutes
and its observations are systematically offset by more than ±15 min from integer
UTC hours (e.g., all observations fall near HH:30 rather than HH:00), the station
must be flagged as `SYSTEMATIC_TIME_OFFSET_REVIEW` in the per-basin coverage audit.
The tolerance must **not** be silently widened to accommodate such stations; manual
review is required to decide whether to accept, re-snap, or exclude the affected hours.

When multiple candidates are equidistant, use the earlier observation (ties broken earlier).

### Rationale for USGS IV as canonical source

A prior diagnostic comparing high-frequency USGS IV to hourly nearest/mean
representations for 100 basins < 1,000 km² (2013–2016) found that hourly resolution
preserves event peak magnitudes well, including high quantiles. Sub-hourly data
primarily affects rapid-onset timing and native flashiness metrics. Hourly targets
are therefore defensible for Stage 1 regional modeling. Native sub-hourly data
remains available for diagnostic comparison.

Using USGS IV directly (rather than CAMELSH) provides:
- A single reproducible API source for all 50 pilot basins and any future expansion.
- Consistent provenance and request metadata.
- Access to USGS qualifier and status fields (provisional/approved/ice-affected etc.).
- The ability to re-acquire data after the CAMELSH subset window without re-downloading
  a large catalog zip.

---

## 3. Basin Universe

Three nested scopes require separate handling:

### 3a. 50-basin Stage 1 pilot

From `tmp/stage1_pilot_dryrun/09_manifests/stage1_pilot/pilot_basin_manifest.csv`.
Roles from manifest:

| Role | Count | Notes |
|---|---|---|
| TRAIN | 40 | All eligible for training targets |
| HOLDOUT_QC | 5 | Held for evaluation; targets needed but not training lists |
| EXCLUDE_QC | 5 | All excluded from training and evaluation; `human_decision=EXCLUDE` |

For EXCLUDE_QC basins:
- Their USGS IV data **may** be fetched for QC lineage purposes if desired.
- They must **never** appear in training or evaluation STAID lists.
- 10336700 has no CAMELSH data and no recovery was attempted; the same exclusion
  applies for full-period USGS IV.

### 3b. 2,843 initial training basin set

Not yet materialized in Stage 1 scripts. Full TRAIN/HOLDOUT/EXCLUDE breakdown
for this set has not been decided. The same role-based guardrails apply:
EXCLUDE_QC basins must never enter training lists regardless of data availability.

### 3c. Future expanded basin sets

Out of scope for Stage 1. Policy should be carried forward.

---

## 4. Full-Period Window

| Parameter | Value |
|---|---|
| **Research period start** | `2020-10-14T00:00:00Z` |
| **Research period end** | `2025-12-31T23:00:00Z` |
| **Total duration** | 1,905 days = 45,720 hourly steps |
| **Water years covered** | WY2021 (Oct 2020–Sep 2021) through WY2025 (Oct 2024–Sep 2025) + partial WY2026 (Oct–Dec 2025) |
| **Approximate months** | 63 calendar months |

### Water-year accounting

| Water Year | Start | End | Hours |
|---|---|---|---|
| WY2021 | 2020-10-14 | 2021-09-30 | 8,448 (partial start, 352 days) |
| WY2022 | 2021-10-01 | 2022-09-30 | 8,760 |
| WY2023 | 2022-10-01 | 2023-09-30 | 8,760 |
| WY2024 | 2023-10-01 | 2024-09-30 | 8,784 (leap year) |
| WY2025 | 2024-10-01 | 2025-09-30 | 8,760 |
| WY2026 partial | 2025-10-01 | 2025-12-31 | 2,208 |
| **Total** | | | **45,720** |

**Implementation note:** Scripts must generate the target time index with
`pd.date_range("2020-10-14", "2025-12-31 23:00", freq="h")` or equivalent and use
the computed index length. Never hard-code the hour count directly.

### Known caveat: 03298135 late-2025 gap

Milestone 2H-B discovered that STAID `03298135` (Chenoweth Run, KY) had its last
observed value at `2025-11-24T15:55Z` — more than 14 days before 2025-12-31. This
may indicate a station outage or delayed data upload. January 2023 data for this
station is valid and was included in 2H-C/2H-D.

**Required action for full-period build:** Full per-water-year gap accounting for
`03298135` before including it in HPC training runs. The gap coverage audit (Section 8)
must flag per-WY coverage for this station explicitly.

---

## 5. Proposed Output Layout

```
{OUT_ROOT}/                              # e.g., stage1_usgs_iv_full_period/
  raw_cache/                             # optional; raw IV observations
    {STAID}/
      {STAID}_{YYYY}{MM}.parquet         # station-month or station-WY chunk
    .request_log.jsonl                   # one line per API request: URL, time, status, n_rows
  canonical/                             # processed hourly targets
    {STAID}_hourly.nc                    # single-file, full period, float32/NaN
  audit/
    per_basin_coverage.csv               # total valid/NaN hours, first/last obs, cadence
    per_water_year_coverage.csv          # per-basin per-WY valid-hour count and fraction
    per_month_coverage.csv               # per-basin per-calendar-month coverage
    gap_audit.csv                        # per-basin: n_gaps, longest_gap_h, gap_starts
    quality_audit.csv                    # per-basin: negatives, flatlines, qualifiers, duplicates
    failed_requests.jsonl                # requests that failed or returned no data
    excluded_basins.txt                  # EXCLUDE_QC basins; not acquired or marked explicitly
  manifests/
    acquisition_manifest.json            # what was fetched: STAID, chunk, URL, rows, UTC
    output_manifest.json                 # NC files written: STAID, n_valid, n_nan, checksum
    basin_roles.csv                      # STAID, pilot_role, human_decision
  logs/
    run_{RUN_TS}.log                     # full run log
    {STAID}_acquire.log                  # per-basin acquisition log (verbose)
  provenance/
    run_provenance.json                  # script, git hash, args, environment
```

Design notes:
- All outputs under `{OUT_ROOT}` (under `tmp/` during development; configurable for HPC).
- `raw_cache/` is optional; may be omitted to save disk if only canonical NCs are needed.
  If retained, store as Parquet (compressed Snappy or zstd) rather than raw JSON.
- `failed_requests.jsonl` enables targeted retry without re-fetching successful requests.
- The canonical `{STAID}_hourly.nc` is a single file per basin covering the full period.
  Station-month chunks from the API are merged at write time.
- `output_manifest.json` includes a checksum (SHA-256 of the file) for integrity verification.
- All paths are relative to `{OUT_ROOT}` in manifests so the tree is relocatable.

---

## 6. Canonical NetCDF Schema

### Filename convention

```
{STAID}_hourly.nc
```

`{STAID}` is always an 8-character zero-padded string (e.g., `01585200`).

### Dimensions and coordinates

| Name | Type | Length | Encoding |
|---|---|---|---|
| `time` | `datetime64[ns]` | 45,720 (full period) | `float64`, `units='hours since 2020-10-14 00:00:00'`, `calendar='proleptic_gregorian'` |

### Variables

| Name | Type | Units | Notes |
|---|---|---|---|
| `streamflow` | `float32` | `m3 s-1` | Canonical CAMELSH-like name; `NaN` for missing |

When packaging into NeuralHydrology format, `streamflow` is renamed to `qobs_m3s`
with units `m3 s**-1` and encoding `_FillValue=-9999.0`, coordinate renamed to `date`.
This conversion is performed in the package-builder script, not the acquisition script.

### `_FillValue=-9999.0` encoding policy

- The canonical USGS IV hourly files use `NaN` (IEEE 754 float32/float64 not-a-number) to
  represent missing streamflow. No sentinel value is stored in these files.
- `_FillValue=-9999.0` is used **only** as a NetCDF encoding hint in NeuralHydrology package
  files. When xarray or the NeuralHydrology loader reads such a file, it decodes the stored
  `-9999.0` values back to `NaN` automatically via the `_FillValue` attribute.
- **NeuralHydrology training and evaluation code must therefore see decoded `NaN`, never
  actual `-9999` values in the `qobs_m3s` array.**
- No CSV, audit table, or model-facing array should contain actual `-9999` streamflow values.
- Auditors verifying package files must read back each NC through xarray (with default
  `mask_and_scale=True`) and confirm that `(ds["qobs_m3s"].values == -9999).any()` is
  `False` — i.e., all encoded sentinels decode to `NaN` before any downstream use.

### Required variable attributes

```python
{
    "units":                  "m3 s-1",
    "long_name":              "Discharge (USGS NWIS IV, hourly nearest-snap)",
    "source_product":         "USGS NWIS Instantaneous Values",
    "source_url":             "https://waterservices.usgs.gov/nwis/iv/",
    "parameter_code":         "00060",
    "parameter_description":  "Discharge, cubic feet per second",
    "conversion_factor":      "0.028316846592",
    "conversion_formula":     "m3/s = ft3/s * 0.028316846592",
    "timestamp_policy":       "provisional: exact HH:00 UTC; nearest +-15 min; else NaN; no interpolation",
    "snap_tolerance_minutes": "15",
    "time_zone":              "UTC (naive datetime64; no tz offset stored)",
    "period_start_utc":       "2020-10-14T00:00:00Z",
    "period_end_utc":         "2025-12-31T23:00:00Z",
    "generated_utc":          "{ISO timestamp at generation time}",
    "git_commit":             "{commit hash of acquisition script}",
    "acquisition_script":     "scripts/recover_usgs_iv_full_period_hourly.py",
    "request_chunks":         "{JSON array of {start, end, n_rows, request_utc}}",
    "data_access_date":       "{ISO date when USGS data was fetched}",
    "usgs_provisional_note":  "Data may include provisional USGS values; qualifiers in sidecar audit",
}
```

### Required global dataset attributes

```python
{
    "staid":        "{STAID}",
    "milestone":    "Flash-NH Stage 1 full-period USGS IV target",
    "conventions":  "CF-1.8",
    "history":      "{generated_utc}: created by acquire script {git_commit}",
}
```

### Handling excluded and all-NaN basins

- EXCLUDE_QC basins (`02324400`, `03106300`, `07263580`, `10336700`, `13112000`):
  either skip entirely (not fetched) or write an all-NaN file with an explicit
  `excluded_reason` global attribute. Either approach is acceptable; the choice
  should be documented in the manifest.
- For any basin where the full-period USGS IV returns no data or only provisional
  data flagged for removal: write an all-NaN file with `data_status=no_data_available`
  in the global attributes.

---

## 7. Acquisition Strategy

### API request structure

USGS IV base endpoint:
```
https://waterservices.usgs.gov/nwis/iv/?sites={STAID}&parameterCd=00060&startDT={start}&endDT={end}&format=json
```

### Recommended chunking: station × water-year

| Strategy | Chunk size | Requests (50 basins) | Requests (2,843 basins) | Notes |
|---|---|---|---|---|
| Station-month | ~1 month | 3,150 | 179,109 | Fine granularity; safe resume; many requests |
| **Station-water-year** | ~12 months | **300** | **17,058** | Recommended; balances size vs count |
| Station-full-period | ~63 months | 50 | 2,843 | Risk of timeout for 5-min stations |

**Recommendation:** station-water-year as the primary chunk. If a water-year request
fails or times out, automatically retry with station-month sub-chunks. Implement
as a two-level fallback: try WY → on failure try month-by-month → log failed months
to `failed_requests.jsonl`.

Water-year chunks for the research period:
- WY2021: `2020-10-14T00:00:00Z` to `2021-09-30T23:00:00Z` (partial start)
- WY2022–WY2025: standard full water years
- Partial WY2026: `2025-10-01T00:00:00Z` to `2025-12-31T23:00:00Z`

### Rate limiting

- USGS waterservices does not publish a hard rate limit for read requests.
- Safe practice observed in the field: **1–5 requests/second per IP**.
- Recommended: `sleep(0.5)` between requests on a single worker; use per-worker
  sleep rather than a global lock when parallelising.
- On HTTP 429 or 503: exponential back-off starting at 5s, capped at 120s, maximum 5 retries.

### Retry and resume behavior

- Before fetching any chunk, check whether a Parquet file for that `(STAID, WY)` pair
  already exists in `raw_cache/`. If it does and `--force` is not set, skip.
- This makes the acquisition **idempotent**: partial runs can be resumed without
  re-fetching completed chunks.
- Log each successful fetch to `acquisition_manifest.json` atomically (append-mode JSONL
  during the run; converted to a single JSON at the end).
- On failure, write the failed request to `failed_requests.jsonl` with the error,
  timestamp, and full URL.
- Provide a `--retry-failed` flag that reads `failed_requests.jsonl` and retries only
  those chunks.

### Output integrity

- Write each canonical NC file atomically: write to a `.tmp` path, then rename.
- Include a SHA-256 checksum in `output_manifest.json` per basin file.
- Do not overwrite an existing canonical NC without `--force`.

---

## 8. Gap and Quality Audit Design

The audit scripts must produce the following outputs for each basin and in aggregate.

### Per-basin summary audit (`per_basin_coverage.csv`)

| Column | Description |
|---|---|
| `STAID` | 8-char STAID |
| `pilot_role` | TRAIN / HOLDOUT_QC / EXCLUDE_QC |
| `period_start_utc` | First target hour |
| `period_end_utc` | Last target hour |
| `n_hours_total` | Total hourly steps in period |
| `n_valid` | Non-NaN streamflow hours |
| `n_nan` | NaN hours |
| `coverage_fraction` | `n_valid / n_hours_total` |
| `first_obs_utc` | First non-NaN hour |
| `last_obs_utc` | Last non-NaN hour |
| `median_cadence_minutes` | Median raw IV cadence (from raw cache or snapping stats) |
| `n_exact_snaps` | Hours filled by exact match |
| `n_nearest_snaps` | Hours filled by nearest-within-tolerance |
| `n_missing` | Hours with NaN (no snap) |
| `n_negative_values` | Count of `streamflow < 0` |
| `n_duplicate_raw_timestamps` | Duplicate timestamps in raw IV before snapping |

### Per-water-year coverage (`per_water_year_coverage.csv`)

Rows: `(STAID, water_year)`. Columns: `n_valid`, `n_nan`, `coverage_fraction`.
Flag basins where any water year falls below a coverage threshold (e.g., < 90%).

### Per-month coverage (`per_month_coverage.csv`)

Rows: `(STAID, YYYY, MM)`. Columns: `n_valid`, `n_nan`, `coverage_fraction`.
Useful for identifying systematic monthly outages or download gaps.

### Gap audit (`gap_audit.csv`)

| Column | Description |
|---|---|
| `STAID` | |
| `n_gaps` | Count of contiguous NaN runs |
| `longest_gap_hours` | Longest contiguous NaN run |
| `total_gap_hours` | Sum of all NaN hours |
| `gap_start_utc_list` | JSON array of gap start timestamps (for large gaps, e.g., > 24h) |
| `late_2025_gap_flag` | `True` if last obs before 2025-12-31 is > 14 days earlier |

### Quality audit (`quality_audit.csv`)

| Column | Description |
|---|---|
| `STAID` | |
| `n_negative_values` | Values where `streamflow < 0` after unit conversion |
| `n_flatline_runs` | Count of constant-value runs ≥ 24h |
| `longest_flatline_hours` | |
| `n_provisional_values` | If USGS qualifiers are retained: provisional count |
| `n_ice_affected_values` | Qualifier `e` or similar USGS ice codes |
| `n_estimated_values` | Qualifier `e` (estimated) |
| `unusual_status_flags` | Any USGS status codes beyond `A` (approved) or `P` (provisional) |
| `suspicious_value_flag` | `True` if any value > 99th percentile of all-period values by > 5× |
| `units_observed_in_raw` | Should always be `ft3/s` (00060); flag deviations |

### Station-specific warnings

Certain basins have documented issues to flag automatically:
- `03298135`: check late-2025 gap; flag if last obs > 14 days before `2025-12-31T23:00Z`.
- Any station where the raw IV cadence changes mid-period (5-min → 15-min transition).

---

## 9. HPC / SLURM Design

### Job-array structure

**Option A: One task per (basin, water-year)**

```
#SBATCH --array=0-{N_TASKS-1}
# N_TASKS = n_basins × n_water_years
# For 50 basins:   50 × 6 = 300 tasks
# For 2,843 basins: 2,843 × 6 = 17,058 tasks
```

Pros: fine-grained resume; failed tasks are at WY granularity.
Cons: many tasks; scheduler overhead if each task is very short.

**Option B: One task per basin (recommended for < 5,000 basins)**

```
#SBATCH --array=0-{N_BASINS-1}
# Each task fetches all WYs for its basin sequentially
# For 50 basins:   50 tasks, each ~6 requests
# For 2,843 basins: 2,843 tasks, each ~6 requests
```

Pros: simpler; fewer tasks; each task fetches one basin fully.
Cons: a failed basin must be fully re-fetched (mitigated by per-WY Parquet cache).

**Recommendation:** Option B for ≤ 5,000 basins. Use per-WY caching so partial
completions within a task are not lost on re-run.

### Expected task counts

| Basin set | Option A (basin × WY) | Option B (basin) |
|---|---|---|
| 50-basin pilot | 300 | 50 |
| 2,843 basins | 17,058 | 2,843 |
| Future (10,000) | 60,000 | 10,000 |

### Retry and resume

- A `--retry-failed` script reads `failed_requests.jsonl`, extracts `(STAID, WY)` pairs,
  and submits a targeted re-run array for those only.
- A post-run merge/audit step (submitted as a dependency after the array completes)
  validates all canonical NCs and writes the aggregate audit tables.

### Recommended structure

```
scripts/
  recover_usgs_iv_full_period_hourly.py   # worker: one basin per call
  audit_usgs_iv_recovered_targets.py      # post-run audit: all basins
slurm/
  submit_acquire.sh                       # submits array
  submit_audit.sh                         # submits post-run audit as dependency
  submit_retry.sh                         # submits retry array from failed_requests.jsonl
```

### Environment assumptions

- Python ≥ 3.10; `requests`, `numpy`, `pandas`, `xarray`, `netCDF4` (or `scipy.io.netcdf`) in the environment.
- Network access to `waterservices.usgs.gov` from compute nodes (check HPC firewall policy before submission).
- Output directory on a shared filesystem accessible by all nodes.
- Logging: each task writes `logs/{STAID}_acquire.log`; combine with a post-run `cat` or a log-merge script.

---

## 10. Storage and Runtime Estimates

All estimates are **planning-only**. Figures marked `[pilot-calibrated]` are derived
from Milestone 2H-C/2H-D pilot outputs. Figures marked `[formula-derived]` are
computed from the pilot numbers scaled to the full period. HPC wall-clock must be
calibrated by a future small full-period pilot (see Section 12) — do not treat the
runtime estimates below as benchmarks.

### Calibration from 2H-C pilot

| Metric | Value | Source |
|---|---|---|
| Streamflow-only NC file size (744h, uncompressed) | 23 KB / basin-month | [pilot-calibrated] |
| Raw IV observations, 15-min cadence | 2,973 obs / basin-month | [pilot-calibrated] |
| Raw IV observations, 5-min cadence | 8,917 obs / basin-month | [pilot-calibrated] |
| 5-min cadence stations in pilot (4/21) | 01585200, 02146381, 03298135, 07103700 | [pilot] |

### Full-period scaling

Full period = 63 calendar months = 45,720 hourly steps.

**Canonical hourly NC (streamflow only, per basin):**

| Estimate | Method | Result |
|---|---|---|
| Lower (with zlib compression, typical compressibility) | 23 KB/month × 63 months × 0.4 | ~580 KB / basin |
| Nominal (no compression, linear scale) | 23 KB/month × 63 months | ~1.4 MB / basin |
| Upper (larger headers, metadata, debug attrs) | 23 KB/month × 63 months × 1.5 | ~2.2 MB / basin |

Aggregate canonical NC:
- 50-basin pilot: 29 MB – 110 MB [formula-derived]
- 2,843 basins: 1.7 GB – 6.3 GB [formula-derived]

**Raw IV cache (optional Parquet, per basin):**

Assumed 20 bytes/row compressed (Parquet snappy; includes STAID, datetime, value, qualifier fields).

| Cadence | Obs/month | Full period obs | Raw Parquet size / basin |
|---|---|---|---|
| 15-min | 2,973 | 187,299 | ~3.7 MB |
| 5-min | 8,917 | 561,771 | ~11.2 MB |
| Mixed (assume 80% 15-min, 20% 5-min across 2,843 basins) | — | — | ~4.9 MB avg |

Aggregate raw IV cache:
- 50-basin pilot (4 five-min, 46 fifteen-min): ~200 MB [formula-derived; uncertainty ×0.5–×3]
- 2,843 basins at 4.9 MB avg: ~14 GB [formula-derived; uncertainty ×0.5–×3]

**Uncertainty note:** Raw IV cache size is particularly uncertain because:
(a) actual Parquet compression ratio depends on value distributions and cadence consistency;
(b) USGS returns verbose JSON which, even after parsing to Parquet, retains metadata;
(c) some stations may have denser observations than the median for the pilot.
Treat ×0.5–×3 as a reasonable planning range.

### API request count

Using station-water-year chunks (6 per basin):

| Basin set | Requests | At 1 req/s sequential | At 5 req/s sequential |
|---|---|---|---|
| 50-basin pilot | 300 | ~5 min | ~1 min |
| 2,843 basins | 17,058 | ~4.7 h | ~57 min |

**Important:** These are sequential single-worker estimates. With N workers on HPC,
wall-clock divides roughly by N (network/API being the bottleneck, not CPU).
Actual HPC wall-clock must be measured — see Section 12.

Likely bottleneck: network latency and USGS API throughput, not local I/O or compute.
The USGS IV API has been observed to return larger time-ranges (e.g., full WY for
a 5-min station: ~8,760 obs) in < 2 seconds under normal conditions, but can spike
to 10–30 seconds under load. Budget 3 seconds/request as a planning conservative.

At 3 s/request:
- 50 basins: 300 × 3s = 15 min sequential; ~1 min with 20 workers
- 2,843 basins: 17,058 × 3s = 14.2 h sequential; ~1.7 h with 8 workers

**Conclusion:** Full-period acquisition for 2,843 basins is likely feasible in 2–4 hours
of HPC wall-time with 8–32 simultaneous worker processes, but this must be validated
with a small benchmark before committing to the full run.

---

## 11. Proposed Scripts for Future Milestones

These scripts are **not implemented in this milestone.** Intended roles:

### `scripts/audit_usgs_iv_full_period_availability.py`

- Input: list of STAIDs (from manifest or text file).
- Action: for each STAID, query USGS IV for a lightweight metadata check
  (e.g., first/last date, station info) without downloading full data.
- Output: availability table (`staid`, `available`, `first_obs_utc`, `last_obs_utc`,
  `late_2025_gap_flag`, `cadence_minutes`).
- Purpose: a pre-flight audit before the full acquisition; catches stations that
  have been decommissioned, renumbered, or have significant coverage gaps before
  any HPC job array is submitted.

### `scripts/recover_usgs_iv_full_period_hourly.py`

- Input: `--staid`, `--out-dir`, `--start`, `--end`, optional `--force`, `--dry-run`.
- Action: fetch USGS IV for the full research period in WY chunks; snap to hourly UTC grid;
  write canonical `{STAID}_hourly.nc`; write per-basin acquisition log and Parquet cache.
- Resume behavior: skip any WY chunk whose Parquet file already exists (unless `--force`).
- Designed to be called once per basin from a SLURM array task.
- Output: `canonical/{STAID}_hourly.nc`, `raw_cache/{STAID}/`, `logs/{STAID}_acquire.log`.

### `scripts/audit_usgs_iv_recovered_targets.py`

- Input: `--canonical-dir`, `--out-dir`, `--manifest`.
- Action: reads all canonical NC files; computes per-basin and aggregate audit tables;
  writes `per_basin_coverage.csv`, `per_water_year_coverage.csv`, `gap_audit.csv`,
  `quality_audit.csv`; flags stations with known caveats (03298135 etc.).
- Designed to run as a post-run dependency job after the acquisition array completes.

### Optional: package-integration script

- Reads canonical `{STAID}_hourly.nc` files plus 2G forcing NCs.
- Writes NeuralHydrology-compatible package NCs with `qobs_m3s` variable and `date` coordinate.
- Handles TRAIN/HOLDOUT/EXCLUDE role filtering and basin list generation.
- Inherits the design from `scripts/build_stage1_neuralhydrology_january_with_recovery.py`
  but generalises to the full period and a single USGS IV source.

---

## 12. Acceptance Criteria for Next Pilot Milestone (2I-B)

Before any HPC-scale full-period acquisition is submitted, a small full-period pilot
must pass the following criteria.

### 2I-B pilot scope

- **Basin count:** 5–10 basins.
- **Required coverage of basin types:**

  | Type | Example STAID | Reason |
  |---|---|---|
  | 5-min cadence | 01585200 | Tests snapping density, raw cache size |
  | 15-min cadence | 02073000 | Baseline case |
  | Gap-prone / WY-incomplete | 10164500 | Tests gap handling (19h NaN in Jan 2023) |
  | Flashy / high-variability | 02077670 | Tests high-magnitude preservation |
  | Snow/western headwaters | 10164500 or another WY basin | Tests seasonal gap patterns |
  | HOLDOUT_QC | 02266500 | Confirms recovery ≠ training approval |
  | 03298135 (late-gap caveat) | 03298135 | Confirms gap audit flags correctly |

- **Period:** full `2020-10-14` to `2025-12-31`.
- **Local run only** (no HPC submission) for this pilot.

### 2I-B required outputs

1. `canonical/{STAID}_hourly.nc` for each pilot basin (7 files).
2. `per_basin_coverage.csv` and `per_water_year_coverage.csv`.
3. `gap_audit.csv` — must flag `03298135` late-2025 gap correctly.
4. `quality_audit.csv`.
5. Runtime report: total wall-clock, requests made, requests failed, per-basin timing.
6. Storage report: raw cache size, canonical NC sizes, totals.
7. Comparison plot: Jan 2023 slice of new canonical NC vs 2H-C recovered NC for the same basins.

### 2I-B pass criteria

- All 7 canonical NCs present and valid (744h per month, no sentinels, no interpolation).
- `per_water_year_coverage.csv` has no missing WY rows.
- `03298135` late-2025 gap correctly flagged in `gap_audit.csv`.
- No EXCLUDE_QC basin in any training list.
- Jan 2023 slice matches 2H-C recovered values to float32 precision.
- No unhandled exceptions in the run log.
- Storage and runtime report produced (no minimum threshold, but must be plausible).

Only after 2I-B passes should the HPC job array for all 50 pilot basins (or 2,843 basins)
be submitted.

---

## 13. Open Questions

The following questions require explicit decisions before implementation and should
be resolved — ideally in a brief design review — before 2I-B begins.

**Q1 — Snap tolerance**  
Is ±15 min the correct tolerance for all cadences?

- At 15-min cadence: ±15 min always finds exactly one candidate (unless at a gap boundary).
- At 5-min cadence: ±15 min allows up to 3 candidates; nearest is chosen.
- Alternative: ±7.5 min (half-interval) for each cadence independently.
- Alternative: ±0 min (exact only) for 5-min stations (more conservative; more NaN at first/last hour).

Current default (from 2H-C/2H-D): ±15 min for all. **Decision pending.**

Edge case addressed (see Section 2): if median cadence is ≥ 60 min and observations are
systematically offset by more than ±15 min from HH:00, the station is flagged as
`SYSTEMATIC_TIME_OFFSET_REVIEW`. Tolerance is never silently widened for such stations.

**Q2 — Raw IV cache retention policy**  
Should the raw/native IV Parquet cache be retained permanently or only during the run
(deleted after canonical NC is validated)?

- Permanent: enables re-snapping with different tolerance or re-auditing without re-downloading.
  ~14 GB for 2,843 basins. Worth retaining on HPC scratch for the duration of the project.
- Delete after validation: saves storage; but forces re-download if policy changes.

**Recommendation:** retain on HPC scratch, delete from local development machines.
**Decision pending.**

**Q3 — USGS qualifier/status fields in canonical output** *(DECIDED)*  
USGS `qualifiers` (e.g., `P` provisional, `A` approved, `e` estimated, `Ice`) are stored
in **sidecar audit/debug tables only**, not in the canonical NC.

- Canonical NC schema: time + streamflow + metadata attributes only. No qualifier variable.
- Summary qualifier counts (total provisional, estimated, ice-affected hours) appear in
  `quality_audit.csv` for every basin.
- Detailed per-hour qualifier data, if needed for manual inspection, goes in an optional
  `{STAID}_qualifier_debug.csv` sidecar file, written on request.
- This keeps the canonical NC format clean and compatible with CAMELSH-like conventions.

**Q4 — Provisional vs approved data** *(DECIDED)*  
Provisional (`P`) values are **included by default**.

- USGS provisional data is routinely used for operational forecasting; it is generally
  trustworthy for training and approved data is not available for recent months.
- The `usgs_provisional_note` attribute is retained on all canonical NC files.
- Per-basin, per-month, and per-water-year provisional fractions are reported in
  `quality_audit.csv`.
- `Ice`-qualified and `e` (estimated) values are **retained initially but flagged strongly**
  in the quality audit. They are not automatically set to NaN. Any decision to exclude
  specific qualified values requires an explicit later filtering step and must be documented
  as a new provenance entry — it is not performed silently here.

**Q5 — TRAIN/HOLDOUT/EXCLUDE policy for full-period package**  
The 50-basin pilot manifest (TRAIN=40, HOLDOUT_QC=5, EXCLUDE_QC=5) is **fixed**.

For the 2,843-basin set, the TRAIN/HOLDOUT/EXCLUDE split has not yet been decided.
That split must be locked before any full-period package build that assigns basins to
training or evaluation lists.

**Acquisition may proceed independently of the split decision:** canonical streamflow
NC files can be produced for all candidate basins without yet knowing their role.
Package-building (which assigns `qobs_m3s` to training vs holdout vs excluded lists)
is a downstream step that applies the split. This decoupling means the 2I-B and
subsequent HPC acquisition milestones are not blocked by the 2,843-basin split decision.

**Q6 — USGS IV vs CAMELSH validation** *(Required step)*  
Milestone 2I-B **must** include a comparison of January 2023 hourly values from the
new full-period canonical USGS IV files against the existing 2H-C recovered files.
This is already a required 2I-B pass criterion (Section 12): values must match to
float32 precision.

A broader USGS IV vs CAMELSH comparison (covering the full CAMELSH overlap window)
is deferred to a potential Milestone 2I-C. If the 2I-B pilot comparison reveals
systematic offsets or suspicious periods for any basin, 2I-C becomes mandatory before
the HPC full-period build proceeds. If the 2I-B comparison is clean, 2I-C may be
treated as an optional diagnostic.

---

## Summary Table

| Section | Key decision | Status |
|---|---|---|
| Canonical source | USGS IV `00060` | **Decided** |
| Snap tolerance | ±15 min (provisional); systematic-offset stations flagged, not silently widened | Open — Q1 |
| Raw cache retention | Retain on HPC scratch, delete from local dev | Open — Q2 |
| Qualifier handling | Sidecar audit only; summary counts in `quality_audit.csv` | **Decided** — Q3 |
| Provisional data | Include; flag by basin/month/WY; ice/estimated retained but flagged | **Decided** — Q4 |
| `_FillValue=-9999.0` | NetCDF encoding only; decoded arrays must be NaN; no -9999 in model input | **Decided** |
| 2,843-basin split | Acquisition decoupled from split; split must be locked before package-building | Clarified — Q5 |
| USGS IV vs CAMELSH validation | 2I-B Jan-2023 slice comparison required; broader 2I-C if issues found | Required — Q6 |
| Chunk strategy | Station × water-year | **Decided** |
| Resume behavior | Parquet per-WY cache; `--force` | **Decided** |
| EXCLUDE_QC handling | Never in training lists; optionally fetched for QC lineage | **Decided** |
| Basin set for next milestone | 2I-B: 5–10 basins, full period, local run | **Decided** |
| HPC before 2I-B passes | No | **Decided** |

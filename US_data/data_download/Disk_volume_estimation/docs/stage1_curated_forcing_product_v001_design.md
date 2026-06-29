# Stage 1 — Curated Forcing Product v001 Design

**Created:** 2026-06-28  
**Updated:** 2026-06-29 — builder, auditor, and launcher implemented; smoke PASS (Milestone 2K-F-B).  
**Milestone:** 2K-F-A — design frozen; 2K-F-B — scripts implemented and smoke-tested.  
**Status:** DESIGN FROZEN — implementation COMPLETE; smoke PASS (2026-06-29, h2o).  
**Depends on:** `docs/stage1_forcing_fullperiod_audit.md` (PASS_WITH_CAVEATS, 2026-06-24),
`docs/stage1_forcing_fullperiod_postrun_audit_plan.md §9`, pilot visual QC PASS (2026-06-28).

---

## 1. Purpose and Scope

The curated forcing product v001 is the canonical per-basin forcing time-series product
assembled from the Stage 1 full-period MRMS+RTMA monthly extraction chunks. It sits between
the raw monthly chunk Parquets (extractor output) and the final NeuralHydrology training
package. It is the first artifact that has a stable, audited, per-basin layout and can be
transferred to Moriah for NH package assembly.

This document freezes the product contract: format, schema, gap policy, layout, and
acceptance criteria. Implementation of the builder and auditor scripts is deferred to the
next milestone.

**Scope:**
- 2,752 v001 basins (same set as `docs/stage1_target_package_builder.md`)
- Period: 2020-10-14T00:00:00Z – 2025-12-31T23:00:00Z (45,720 hours)
- Source products: MRMS QPE 1h Pass1 + RTMA CONUS 2.5 km (11 variables, excluding `10wdir`
  and `orog`)
- All known gaps preserved as NaN; no imputation; no interpolation in the raw product

---

## 2. Relationship to Other Pipeline Components

```
[Monthly extraction chunks]          [v001 target package]
  /stage1_forcing_fullperiod/          /stage1_target_package_v001/
  chunks/{YYYY-MM}/                    time_series/{STAID}.nc
    combined_{YYYY-MM}.parquet
             │                                   │
             │   build_stage1_curated_forcing_basin_parquets  │   (future NH assembly)
             ▼                                   ▼
[Curated forcing product v001]   +   [Streamflow targets]
  stage1_basin_hourly_forcings_v001/                │
  {STAID}/{STAID}_hourly_forcings.parquet    ──────►│
                                                    ▼
                            [NeuralHydrology package]
                              (per-basin NC, configs, splits)
                                        │
                                        ▼
                            [Moriah cluster — training only]
```

The curated forcing product is derived from the monthly extraction Parquets — it
re-organizes the existing data by basin rather than by month. No new downloads, no
re-extraction. The monthly chunk Parquets remain the authoritative raw source.

It is **not** the NeuralHydrology package. It does not contain streamflow, static
attributes, or train/val/test splits. Those are added in the separate NH assembly step.

---

## 3. Product Name and Output Locations

**Product name:** `stage1_basin_hourly_forcings_v001`

**Full-product output (h2o):**
```
/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/stage1_basin_hourly_forcings_v001/
```

**Smoke-test output (h2o, not committed):**
```
/data42/omrip/Flash-NH/tmp/stage1_basin_hourly_forcings_v001_smoke_<TIMESTAMP>/
```

**Local smoke-test output (not committed):**
```
tmp/stage1_basin_hourly_forcings_v001_smoke_<TIMESTAMP>/
```

Both full-product and smoke directories are **not committed to git**. Only scripts,
docs, and manifests (if small and plaintext) are committed.

---

## 4. Per-Basin File Layout

```
stage1_basin_hourly_forcings_v001/
├── manifest.json                    # Per-basin: STAID, n_hours, n_gap_mrms, n_gap_rtma,
│                                    #   n_valid_hours, coverage_fraction, sha256
├── checksums.sha256                 # SHA-256 for every per-basin forcing Parquet
├── dataset_config.json              # Variable list, units, period, basin count, schema version
├── run_provenance.json              # Builder version, run date, input paths, env name
├── build_summary.md                 # Per-basin summary table; wall time; smoke flag
├── audit_summary.md                 # Condensed audit result (REQUIRED for full build;
│                                    #   NOT yet written by auditor — for smoke, PASS
│                                    #   is captured in smoke.log; see §9)
└── time_series/
    └── {STAID}.parquet
```

All data files are under `time_series/` (flat, no per-basin subdirectory). STAID is
preserved as-is (no zero-padding added or removed; consistent with v001 target package
and monthly Parquet `STAID` column).

> **Implementation note (2K-F-B):** The implementation uses JSON for manifest and config
> (not `.csv`/`.yaml` as originally planned) and a flat `time_series/{STAID}.parquet`
> path (not `{STAID}/{STAID}_hourly_forcings.parquet`). These deviations are binding
> for the full build.

---

## 5. Per-Basin Parquet Schema

### 5.1 Format decision: wide

The per-basin Parquet files use **wide format** (one row per hour, one column per
variable). This decision is binding for v001.

**Rationale:**

| Dimension | Long format | Wide format (chosen) |
|---|---|---|
| File size | Larger — `STAID`, `product`, `variable` repeated every row | Smaller — one row per hour |
| NaN handling | Gap rows are omitted or marked with a sentinel column | NaN in the affected column; no row omitted |
| Gap-flag columns | Requires join on `variable` | One `<var>_gap` boolean column per variable |
| NH loader compatibility | Requires pivot before NH DataLoader | Ready for direct use as DataFrame |
| Inspection ease | Filter by variable name | Readable in any Parquet viewer |
| Schema stability | Adding a variable adds rows | Adding a variable adds a column |

The monthly extraction Parquets are long format and remain so — they are the extraction
output. The per-basin forcing files are a separate derived product; a format change is
appropriate here.

**Gap representation in wide format:** a gap hour produces a full row with `NaN` in
the affected variable column(s) and `True` in the corresponding `_gap` flag column(s).
The row is **never dropped**. The hourly index is always complete (45,720 rows per basin
for the full period; fewer for a smoke run).

### 5.2 Required columns

| Column | Type | Units | Notes |
|---|---|---|---|
| `valid_time_utc` | datetime64[ns, UTC] | — | Hourly timestamp; primary key |
| `mrms_qpe_1h_mm` | float32 | mm | MRMS QPE 1h Pass1 basin-mean; NaN for gap hours |
| `mrms_qpe_1h_mm_gap` | bool | — | True if this hour is a `not_in_s3` gap |
| `rtma_2t_K` | float32 | K | RTMA 2m temperature |
| `rtma_2d_K` | float32 | K | RTMA 2m dewpoint |
| `rtma_2sh_kgkg` | float32 | kg/kg | RTMA 2m specific humidity |
| `rtma_sp_Pa` | float32 | Pa | RTMA surface pressure |
| `rtma_10u_ms` | float32 | m/s | RTMA 10m U-wind |
| `rtma_10v_ms` | float32 | m/s | RTMA 10m V-wind |
| `rtma_tcc_pct` | float32 | % | RTMA total cloud cover |
| `rtma_vis_m` | float32 | m | RTMA visibility |
| `rtma_gust_ms` | float32 | m/s | RTMA wind gust |
| `rtma_weasd_kgm2` | float32 | kg/m² | RTMA water equivalent accumulated snow depth |
| `rtma_ceil_m` | float32 | m | RTMA cloud ceiling |
| `rtma_gap` | bool | — | True if this hour is a gap in any RTMA variable |

**Notes on the RTMA gap flag:** A single `rtma_gap` boolean covers all RTMA variables
simultaneously, because the two known RTMA gaps (2020-11-12T09Z and T10Z) affect all
RTMA variables for the missing hours. If future runs reveal variable-specific gaps,
per-variable `rtma_<var>_gap` flags can be added; the schema version must be bumped.

### 5.3 Column naming convention

```
{product}_{ecmwf_short_name}_{unit_suffix}
```

| Component | Example | Rule |
|---|---|---|
| Product prefix | `mrms` / `rtma` | Lowercase, no version tag |
| ECMWF short name | `2t`, `10u`, `sp` | As in GRIB2 metadata |
| Unit suffix | `_K`, `_mm`, `_ms`, `_Pa`, `_pct`, `_m`, `_kgkg`, `_kgm2` | Abbreviated SI |

Rationale: names are self-documenting in the NH DataLoader; no external lookup table
needed to interpret units. The MRMS variable is prefixed with `mrms_` and named
`qpe_1h_mm` to encode product type and accumulation window explicitly.

---

## 6. RTMA Variable Policy

### 6.1 Included variables (9 dynamic + 2 diagnostic)

Confirmed present in all 63 months of the audit (uniform schema):

| ECMWF short name | Description | Product column |
|---|---|---|
| `2t` | 2m temperature | `rtma_2t_K` |
| `d2m` | 2m dewpoint | `rtma_2d_K` |
| `sh2` (or `2sh`) | 2m specific humidity | `rtma_2sh_kgkg` |
| `sp` | Surface pressure | `rtma_sp_Pa` |
| `10u` | 10m U-wind | `rtma_10u_ms` |
| `10v` | 10m V-wind | `rtma_10v_ms` |
| `tcc` | Total cloud cover | `rtma_tcc_pct` |
| `vis` | Visibility | `rtma_vis_m` |
| `gust` | Wind gust | `rtma_gust_ms` |
| `weasd` | Water equivalent accum. snow depth | `rtma_weasd_kgm2` |
| `ceil` | Cloud ceiling | `rtma_ceil_m` |

`vis` and `ceil` are diagnostic variables confirmed present in the RTMA CONUS 2.5 km
product used for extraction. They are retained in v001 because they are already
extracted at no additional download cost, and ceiling/visibility carry relevant
meteorological context for flooding events.

### 6.2 Excluded variables (binding decision)

| ECMWF short name | Reason for exclusion |
|---|---|
| `10wdir` | Absent from S3 in all 63 months (`rtma_10wdir_absent=True`; confirmed in audit) |
| `orog` | Absent from S3 in all 63 months (`rtma_orog_absent=True`; confirmed in audit) |

These two variables are **not present in the curated product**. No placeholder columns
or NaN columns are added for them. If they become available in future RTMA vintages,
they require a new product version tag (`v002`) and a fresh extraction.

---

## 7. Gap Policy

The gap policy for the curated product is strict and non-negotiable for v001.

### 7.1 Raw product rule: no filling, no interpolation, no row dropping

- All known MRMS and RTMA gaps (`not_in_s3`, confirmed by the extraction audit) are
  represented as NaN in the corresponding column(s) for that hour.
- The hourly index is always contiguous and complete — no rows are dropped for gap hours.
- No values are interpolated, forward-filled, or backward-filled in the raw curated product.
- No silent omission: every gap hour has a corresponding `_gap` flag column set to `True`.

This policy is identical to the raw extraction Parquet policy. The curated product
merely reorganizes the representation; it does not add or remove information.

### 7.2 Known gaps (from the extraction audit)

| Source | Missing hours | Reason | Notes |
|---|---|---|---|
| MRMS | 136 | `not_in_s3` | Permanent upstream S3 archive absences |
| RTMA | 2 | `not_in_s3` | 2020-11-12T09Z and T10Z only |
| Total | 138 | — | Across 20 of 63 months |

No product-synchronized gaps (no hour missing in both MRMS and RTMA simultaneously).

### 7.3 Model-window impact

949 / 45,697 possible 24h MRMS windows (2.08%) contain ≥1 missing hour and must be
excluded from training data assembly. This window-impact fraction is pre-computed in
the extraction audit and does not need to be recomputed by the builder.

### 7.4 Derived layer policy (future, not v001)

Derived versions (e.g., `stage1_basin_hourly_forcings_v001_filled`) may apply:
- Linear interpolation for isolated RTMA 1h gaps (gap length = 1, both neighbors finite),
  variable-by-variable, with a `<var>_interpolated` companion flag.
- No MRMS interpolation (precipitation is not a slowly-varying variable).

Derived layers carry their own version tag and are never back-merged into v001. The
raw curated product (v001) is immutable once built and checksummed.

---

## 8. Manifest, Checksums, Provenance, and Audit

### 8.1 manifest.json

Per-basin record written by the builder. Required columns:

| Column | Type | Notes |
|---|---|---|
| `STAID` | string | USGS site ID |
| `n_hours_expected` | int | 45,720 for full period |
| `n_hours_written` | int | Must equal `n_hours_expected` |
| `n_mrms_gap_hours` | int | Hours where `mrms_qpe_1h_mm_gap = True` |
| `n_rtma_gap_hours` | int | Hours where `rtma_gap = True` |
| `n_valid_combined_hours` | int | Hours where both products are present |
| `coverage_fraction` | float | `n_valid_combined_hours / n_hours_expected` |
| `file_path` | string | Relative path within product root |
| `sha256` | string | SHA-256 of the per-basin Parquet file |
| `builder_version` | string | Git commit of repo at build time |
| `built_at` | ISO 8601 | UTC timestamp of this file's assembly |

### 8.2 checksums.sha256

One line per file: `<sha256>  <relative_path>`. Verifiable with `sha256sum -c`.

### 8.3 dataset_config.json

```json
{
  "product_name": "stage1_basin_hourly_forcings_v001",
  "schema_version": "1.0",
  "month": "YYYY-MM",
  "period_start_utc": "2020-10-14T00:00:00+00:00",
  "period_end_utc": "2025-12-31T23:00:00+00:00",
  "n_hours_expected": 45720,
  "n_basins": 2752,
  "mrms_product": "mrms_qpe_1h_pass1",
  "rtma_product": "rtma_conus_aws_2p5km",
  "variables": [
    "rtma_2t_K", "rtma_2d_K", "rtma_2sh_kgkg", "rtma_sp_Pa",
    "rtma_10u_ms", "rtma_10v_ms", "rtma_tcc_pct", "rtma_vis_m",
    "rtma_gust_ms", "rtma_weasd_kgm2", "rtma_ceil_m", "mrms_qpe_1h_mm"
  ],
  "gap_flag_columns": ["mrms_qpe_1h_mm_gap", "rtma_gap"],
  "excluded_variables": ["10wdir", "orog"],
  "gap_policy": "raw_preserve_nan_no_interpolation",
  "smoke_build": false
}
```

### 8.4 run_provenance.json

Written at builder exit. Must include:
- `builder_script`: `scripts/build_stage1_curated_forcing_basin_parquets.py`
- `repo_commit`: git hash of HEAD at run time
- `conda_env`: conda env prefix and Python version
- `input_chunk_root`: path to monthly extraction chunks
- `input_basin_list`: path to v001 basin list CSV
- `n_basins_attempted`: number of basins builder processed
- `n_basins_success`: number of Parquets written without error
- `n_basins_failed`: any failures (builder must abort and report if > 0)
- `run_start_utc`, `run_end_utc`: wall-clock timestamps
- `host`: machine hostname

### 8.5 Audit requirements for the built product

The builder is accompanied by an auditor script
(`scripts/audit_stage1_curated_forcing_basin_parquets.py`, not yet written) that:

1. Verifies all 2,752 Parquets exist and are non-empty.
2. Verifies the hourly index of each Parquet: must be contiguous, no duplicates,
   must match `[period_start_utc, period_end_utc]` at hourly frequency.
3. Verifies column set matches `dataset_config.yaml` exactly.
4. Verifies `n_hours_written` == 45,720 for every basin.
5. Verifies gap flag columns are boolean dtype and not all-False (i.e., at least
   some True values exist across the full dataset, consistent with audit).
6. Verifies known MRMS gaps: sum of `mrms_qpe_1h_mm_gap` across all basins and all
   hours equals `136 × 2752` (136 gap hours × 2752 basins, because the gap applies
   to all basins uniformly for an `not_in_s3` hour).
7. Verifies known RTMA gaps: sum of `rtma_gap` across all basins equals `2 × 2752`.
8. Verifies SHA-256 of each Parquet matches `manifest.csv`.
9. Reports per-basin coverage fractions; flags any basin deviating from audit expectations.
10. Writes an auditor result JSON with pass/fail status and per-check results.

---

## 9. Builder and Auditor Scripts

| Script | Status |
|---|---|
| `scripts/build_stage1_curated_forcing_basin_parquets.py` | **Implemented — smoke PASS (2026-06-29, h2o, commit `6f4de49`)** |
| `scripts/audit_stage1_curated_forcing_basin_parquets.py` | **Implemented — smoke PASS (2026-06-29, h2o, commit `6f4de49`)** |
| `scripts/run_stage1_curated_forcing_smoke_h2o.sh` | **Implemented — smoke launcher used for 2K-F-B run** |

Script names use the `_curated_forcing_basin_parquets` infix to distinguish this product
from the earlier January-pilot scripts and from any future NH-package NetCDF builder.
The legacy name `build_stage1_forcing_basin_ncs.py` (from prior docs) is retired; all
future references must use the names above (OC-1 resolved — see §12).

**audit_summary.md gap:** The auditor currently writes its pass/fail verdict to stdout,
which is captured in `smoke.log` during the smoke run. It does not write a standalone
`audit_summary.md` to the product directory. For the full 2,752-basin build (Milestone
2K-F-C), the auditor must be extended to write `audit_summary.md` before the product
directory is considered complete. This is a pre-build requirement, not blocking the smoke.

---

## 10. Smoke-Test Plan

Before full-scale assembly, a **5-basin, 1-month smoke test** verifies that the builder
and auditor work correctly on a small subset.

### 10.1 Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Month | **2020-11** | Contains the 2 known RTMA gap hours (T09Z/T10Z on Nov 12) |
| Basins | 5 (see §10.2) | Small enough to run in <5 min; large enough to catch per-basin errors |
| Output | `tmp/stage1_basin_hourly_forcings_v001_smoke_<TIMESTAMP>/` | Not committed |

Choosing 2020-11 ensures the smoke test exercises:
- RTMA gap NaN preservation and `rtma_gap=True` flag
- MRMS gap handling (2020-11 has 0 MRMS gaps — clean baseline for MRMS)
- Normal hours (720 hours in November; 718 valid RTMA hours per basin)
- Manifest row counts match expected values

### 10.2 Smoke basin selection (to be confirmed at implementation time)

Select 5 basins from the v001 basin list covering:
- 1 basin from the GIS gap-fraction median (control)
- 1 SMALL_FLASHY_BASIN (high RBI)
- 1 HIGH_ALTITUDE basin
- 1 LARGE_BASIN
- 1 from the RTMA_GAP_ADJACENT VQC case (i.e., the VQC-004 basin: `01440000`)

Exact STAID selection is confirmed by the implementation. The 5-basin set is not fixed
here to avoid coupling the design doc to a specific run.

### 10.3 Acceptance checks for smoke test

| Check | Criterion |
|---|---|
| Files written | 5 Parquets, `manifest.json`, `checksums.sha256`, `dataset_config.json` |
| Rows per basin | 720 (Nov has 720 hours) |
| Column set | Exactly the 14 columns from §5.2 (12 variables + 2 gap flags) |
| `valid_time_utc` | Contiguous hourly index, 2020-11-01T00Z to 2020-11-30T23Z, UTC-aware |
| RTMA gap flags | `rtma_gap=True` for T09Z and T10Z on 2020-11-12, for all 5 basins |
| MRMS gap flags | `mrms_qpe_1h_mm_gap=False` for all rows (no MRMS gaps in Nov 2020) |
| NaN consistency | NaN in `rtma_2t_K` (and all RTMA cols) for the 2 gap rows |
| SHA-256 | `manifest.json` `sha256` matches computed hash of each Parquet |
| Auditor result | `audit_pass=True`; 0 errors; expected gap counts confirmed |

### 10.4 Smoke test result (2026-06-29) — PASS

Smoke run on h2o: `bash scripts/run_stage1_curated_forcing_smoke_h2o.sh` at commit `6f4de49`.

| Metric | Value |
|---|---|
| Month | 2020-11 |
| Basins | 5 (`01440000`, `03021350`, `08155541`, `09484000`, `01019000`) |
| Hours per basin | 720 |
| MRMS gaps (total) | 0 |
| RTMA gaps (total) | 10 (2/basin at 2020-11-12T09Z and T10Z) |
| Coverage fraction | 0.9972 (718 valid combined hours / 720) |
| Auditor exit | 0 (PASS) |
| Wall time | 0.1 s |
| h2o output | `/data42/omrip/Flash-NH/tmp/stage1_curated_forcing_smoke_20260629T132757Z` |
| Commit at run | `6f4de498f1326e5e6fcd3de8157ba410ad28a6a9` |

All 9 acceptance checks from §10.3 passed. RTMA gaps at both known timestamps confirmed
`True`; all 11 RTMA columns NaN at those hours; MRMS not falsely flagged at RTMA-only
gap hours.

**Prior failed explicit-basin run:** `02231000` was passed via `--staids` but is absent
from the 2020-11 monthly source chunk. Builder correctly halted with 0 basins built
rather than silently skipping. Not a smoke failure; the builder's per-basin abort-on-miss
behavior is correct. Basin replaced by `01019000` for the passing run.

---

## 11. Acceptance Criteria for the Future Implementation Milestone (2K-F-B)

The implementation milestone (2K-F-B, not this milestone) is considered PASS when:

1. **Smoke test PASS** — all 9 checks in §10.3 pass for 5 basins, 2020-11.
2. **Full-build PASS** — 2,752/2,752 Parquets written; 0 builder errors.
3. **Auditor PASS** — `audit_pass=True`; known MRMS gap count = 136 × 2752;
   known RTMA gap count = 2 × 2752; all SHA-256 checksums match.
4. **Manifest completeness** — `manifest.csv` has 2,752 rows; all required columns present.
5. **Provenance recorded** — `run_provenance.json` written; git commit hash recorded.
6. **No data committed** — only scripts and docs are committed; all Parquets and
   manifests remain under `tmp/` or the designated h2o path.
7. **Auditor is independent** — the auditor script is invoked separately from the builder;
   it must be able to run on a completed product directory without re-running the builder.

---

## 12. Resolved Open Choices

All five open choices from the initial design draft were resolved on 2026-06-29 before
the 2K-F-A commit. They are recorded here for traceability.

| # | Question | Decision |
|---|---|---|
| OC-1 | Builder/auditor script naming | **Resolved:** use `build_stage1_curated_forcing_basin_parquets.py` and `audit_stage1_curated_forcing_basin_parquets.py`. Legacy `_ncs` name retired. |
| OC-2 | Full-period build location | **Resolved:** first build stays under `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/stage1_basin_hourly_forcings_v001/`. Promotion to `/data42/hydrolab/Data/Flash-NH_data/` is a separate explicit gate after full audit and evidence-bundle review. |
| OC-3 | RTMA gap flag granularity | **Resolved:** one shared `rtma_gap` boolean column for v001. Known RTMA gaps are whole-product-hour absences, not variable-specific decode failures. The auditor must still check per-variable completeness and fail if variable-specific missingness appears outside known product-hour gaps. |
| OC-4 | `vis` and `ceil` inclusion | **Resolved:** include all 11 extracted RTMA variables, including `vis` and `ceil`. Curated product preservation and first-model-input variable selection are separate decisions; the first NH smoke config may use a narrower subset. |
| OC-5 | Remaining 15 VQC cases | **Resolved:** remaining 15/21 cases are **not required** before the 2K-F-B smoke test or the full 2,752-basin build. Gate for full build: 2K-F-B smoke PASS + no schema/gap/provenance failures. Render 2–3 additional targeted VQC cases only if smoke or design review reveals a new concern. |

No open choices remain for this design document.

---

## 13. Non-Goals

The following are explicitly out of scope for v001 and this milestone:

- **No full 2,752-basin build in this milestone.** The full build is Milestone 2K-F-B.
- **No NeuralHydrology package assembly.** NH assembly is a downstream step after the
  curated product is built and audited.
- **No Moriah transfer.** Transfer layout and checksum verification are planned but not
  started.
- **No NeuralHydrology training.** Training is designated for Moriah and does not begin
  until the NH package passes audit on Moriah.
- **No rendering of remaining 15/21 VQC cases.** Per the animation plan, these are on
  hold unless the reviewer requests them.
- **No gap imputation or interpolation.** The raw curated product preserves NaN for all
  known gaps. Derived filled versions are a future v001_filled artifact.
- **No new downloads or re-extraction.** The builder reads existing monthly chunk Parquets.
  It does not touch S3 or GRIB2 files.

---

*Design frozen in Milestone 2K-F-A (2026-06-28). Builder and auditor implemented and smoke-tested in*
*Milestone 2K-F-B (2026-06-29). Full 2,752-basin build is Milestone 2K-F-C (not yet authorized).*
*All generated outputs remain under `tmp/` and must not be committed.*
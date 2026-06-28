# Decision Log

Project: Flash-NH — near-real-time and forecast-aware hydrological modeling pipeline.

## 2026-05-06

- Confirmed GFS `.idx` byte-range extraction end-to-end; the acquisition path is validated and frozen except for plotting.
- Confirmed IFS 00/12 UTC MARS retrievals succeed, while 06/18 UTC remain unresolved and should stay open as an access/request issue.
- Confirmed IMERG NC4 download is valid; crop logic still needs repair and final nonzero selected-CONUS confirmation.
- Retained RTMA despite size/time cost because it is scientifically important for the Stage 1 pipeline.

## 2026-05-06 IFS Stream Investigation

- Tested `oper/fc` requests for 06 UTC and 18 UTC on 2023-01-01 with `area` subset and without `area`; all returned `MARS_EXPECTED_FIELDS` with 0 retrieved fields.
- Tested `oper/fc` with the current variable set at `step=0` and `step=0/to/24/by/1`; both area-subset and full-domain forms failed for 06 UTC and 18 UTC.
- Tested `scda` alternatives with a minimal `2T` request at `step=0`.
- `scda/type=fc` succeeded for both 06 UTC and 18 UTC.
- `scda/type=an` succeeded for both 06 UTC and 18 UTC.
- `scda/type=cf` failed with `MARS_EXPECTED_FIELDS` and 0 retrieved fields.
- The area subset `50/-126/24/-66` did not explain the oper failures.
- **Stream fix implemented**: 00/12 UTC use `oper/fc`, while 06/18 UTC use `scda/fc`.
- Provisional recommendation: use the deterministic `scda` path for historical 06/18 UTC retrievals; keep `oper` as the working path for 00/12 UTC and do not assume only 2 historical IFS cycles exist.

## 2026-05-06 IFS Resolution Comparison

- Tested 2023-01-01 historical retrievals at two grid resolutions:
  - **0.25/0.25**: current configuration
  - **0.1/0.1**: proposed higher resolution
- For cycles 00 UTC (oper/fc) and 06 UTC (scda/fc), tested both minimal requests (1 variable, 1 step) and full requests (7 variables, 25 steps).
- **Results**: Both resolutions retrieved successfully with 100% pass rate.
  - 0.25/0.25: ~17.9 MB total for both cycles (8.9 MB per cycle average)
  - 0.1/0.1: ~110.5 MB total for both cycles (55.3 MB per cycle average)
  - Ratio: 0.1/0.1 is ~6.2× larger
- **Timing**: Higher resolution added ~5–15 seconds per request but remained well within operational tolerance.
- **Recommendation**: Adopt **0.1/0.1 resolution** to align with IFS's scientific value (higher resolution than GFS).
  - Justification: Estimated annual burden ~80 GB (acceptable for 2-year window); retrieval time penalty negligible; area subset remains supported.
  - Contingency: Revert to 0.25/0.25 requires single config change if burden becomes untenable.
- **Decision**: Update `IfsMarsConfig.grid` to `0.1/0.1` and document stream logic in code comments.

## 2026-05-07 IFS 0.1-Degree Estimate Verification

- Verified estimate inputs without changing IFS retrieval logic:
  - 00/12 UTC: `oper/fc`
  - 06/18 UTC: `scda/fc`
  - `grid=0.1/0.1`, `area=50/-126/24/-66`, 7 variables, `step=0..24`, 4 cycles/day.
- Recomputed period: 2020-10-14T00:00:00 to 2025-12-31T23:59:59 (inclusive, 1,905 days; 7,620 cycles).
- Empirical sample bytes per cycle (full request, from resolution test): **54,920,250 bytes**.
- Bytes per day (4 cycles): **219,681,000 bytes** (~219.681 MB/day, ~209.505 MiB/day).
- Full-period raw download estimate: **418,492,305,000 bytes** (~418.492 GB, ~389.751 GiB).
- Full-period retained raw estimate: **418,492,305,000 bytes** (same as download estimate under current workflow assumptions).
- Derived basin-average estimate (9,000 basins; hourly; 7 vars; float32 parquet): **11,521,440,000 bytes** (~11.521 GB, ~10.730 GiB).
- Estimated acquisition time (using measured full-request mean cycle time: (54.40s + 42.36s)/2 = 48.38s):
  - ~193.52s/day (~3.23 min/day)
  - ~368,655.6s total (~102.40 h, ~4.27 days) if executed sequentially.
- Validation of prior wording: **"~80 GB/year" is approximately correct in decimal units**.
  - Recomputed value: **80.184 GB/year** (or **74.677 GiB/year** binary).

## 2026-05-07 IMERG Crop And Preview Plot Repair

- Repaired IMERG CONUS crop handling for dynamic coordinate layouts and dimension order, including `time,lat,lon`, `time,lon,lat`, `lat,lon`, and `lon,lat` forms.
- Added robust crop logging for IMERG:
  - original dims and coordinate names
  - original lon/lat bounds
  - cropped lon/lat bounds
  - cropped shape and min/max/mean/nan_pct
- Added hard failure when IMERG crop result is empty and when `selected_conus_bytes` is zero.
- Verified targeted IMERG validation on 2023-01-01 (`3B-DAY-L.MS.MRG.3IMERG.20230101-S000000-E235959.V07B.nc4`):
  - `selected_conus_bytes=624000` (nonzero)
  - crop bounds `lon=[-125.950, -66.050]`, `lat=[24.050, 49.950]`
  - crop shape `(260, 600)`
- Repaired preview plotting axes/orientation to use true lon/lat extent and north-up orientation logic.
- Added preview bounds validation logging (`preview_bounds_validation=PASS/FAIL`) and summary payloads.
- Verified preview bounds validation passed for:
  - IMERG: PASS
  - GFS: PASS
  - IFS: PASS
- Generated run artifacts under `reports/audit_2026_04_29/run_07_imerg_plot_repair/` with lightweight review bundle (no raw NC4/GRIB files).

## 2026-05-09 Final All-Source Acquisition Audit

- Executed unified 24-hour acquisition audit (2023-01-01) in dry-run mode to validate all 7 implemented datasources without large downloads.
- Orchestration script: [scripts/run_final_all_source_audit.py](../scripts/run_final_all_source_audit.py).
- Audit outputs under `reports/final_all_source_audit_2026_05/`:
  - **24-hour summaries**: `final_audit_summary.{json,csv,md,html}` + `datasource_matrix.{csv,md}`
  - **Decision-support plots**: `storage_breakdown_by_source.png`, `reduction_waterfall_by_source.png`, `download_time_vs_size.png`, `availability_timeline.png`, `crop_validation_overview.png`
  - **Previews & request specs**: organized by source in `previews/` and `request_specs/` directories
  - **Lightweight review bundle**: `review_bundle/` with representative artifacts, logs (truncated), manifests, and docs
- Validation status: All sources validated with current request logic; no architecture changes applied.
- 7-day stability audit attempted for subset (MRMS, RTMA, GFS, IFS, IMERG) but dry-run unable to cache full 7-day sample; logic implemented and ready for real-run.
- Final recommendation sections added to summary markdown with operational stack priorities:
  - **Stage 1**: MRMS + RTMA (high-priority, smallest acquisition burden)
  - **Stage 2**: ERA5-Land + GDAS + IMERG Late Daily (medium-priority, moderate burden)
  - **Stage 3**: GFS + IFS (low-priority, highest burden; IFS uses 0.1/0.1 grid with stream split logic)
- Caveats: External data provider availability, credential lifecycle, and throughput variance remain operational (not logic) issues.

## 2026-06-20 Stage 1 Forcing Throughput Optimization (Milestone 2K-D)

**D1 — Serial extraction optimization (commit `3ff4965`):**
- Pre-grouped the weight DataFrame into a `{STAID: (row_idx, col_idx, norm_w)}` dict at startup,
  eliminating 90,816 O(N) scans per RTMA-hour and shifting per-basin-hour lookup to O(1).
- Replaced 7 sequential `np.percentile` calls with one batched call (635,712 redundant sort passes
  eliminated per RTMA-hour).
- Measured result: `extraction_median_s` 91.976 s → 2.17 s/hr (**24.7× speedup**).
- Bottleneck fully shifted to S3 download. D2 process-workers judged unnecessary and **deferred**.

**Download-worker sensitivity benchmark (48h RTMA-only, 2,752 basins):**
- dw2 → dw16 scanned: individual download time increases (31 → 45 s/file) due to S3 bandwidth
  sharing, but wall-clock decreases via prefetch concurrency.
- dw16 = 570.5 s wall → 6.29 days projected (GREEN vs 14-day target, but not compelling alone).

**Outer-parallelism x2 (2 chunks × dw8, commit `cf8db74`):**
- Parent wall 736 s → 4.057 days projected — **YELLOW** (partial scaling; insufficient alone).
- Decision: do not proceed with x2.

**Outer-parallelism x3 (3 chunks × dw6, commit `a275296`):**
- Parent wall 826 s → 3.035 days projected — **USEFUL GREEN** (within acceptable range).
- All 3 chunks: `all_pass=True`, 48/48 hours, 1,453,056 rows each.
- Decision: **stop optimization here**.

**Final decisions (all binding):**
1. Full-period launch configuration: **3 concurrent chunk processes × 6 download workers each**
   (18 total S3 connections). Splits 63 months into 3 groups (~21 months each).
2. D2 process-workers: **deferred indefinitely.** Extraction is 2.17 s/hr; download dominates.
3. x4 outer-parallelism: **not recommended.** S3 contention risk; marginal gain; x3 is sufficient.
4. `run_stage1_forcing_fullperiod_h2o.sh` needs outer-parallel group support before Phase 2 launch.
5. All h2o paths remain under `/data42/omrip/Flash-NH/` (system `/tmp` prohibited).

## 2026-06-20 Stage 1 Forcing — 2K-E Pre-Launch Patch

**Goal:** Enable 3-way outer parallelism without a new launcher script.

**Changes applied (pre-launch patch, not yet run):**

- `GROUP_ID=A|B|C` env var added to `run_stage1_forcing_fullperiod_h2o.sh`; filters the 63-month
  `MONTH_LIST` to the group's sub-range before the loop. Empty `GROUP_ID` preserves original
  sequential all-63-month behaviour.
  - Group A: 2020-10 → 2022-06 (21 months)
  - Group B: 2022-07 → 2024-01 (19 months)
  - Group C: 2024-02 → 2025-12 (23 months)
- `DRY_RUN=1` mode prints the selected month list and extractor command template, then exits.
  Used to confirm group month counts before committing to a multi-day run.
- Per-group run logs: `manifests/group_{a,b,c}_run_log.txt` (independent; no write conflicts).
- Path safety guard: launcher fails immediately if `FORCING_ROOT` does not begin with
  `/data42/omrip/Flash-NH/`.
- `TMPDIR` redirected to `/data42/omrip/Flash-NH/tmp/tmpdir_flashnh`; never writes to system `/tmp`.
- `${FORCING_ROOT}/logs/` created at startup for screen `tee` targets.
- `report_stage1_forcing_progress_h2o.sh` Section 1 updated to scan all three group logs.

**Decision:** Do not launch extraction until this commit is on h2o and dry-run is confirmed PASS.

## 2026-06-24 Stage 1 Forcing — Full-Period Extraction Audit Acceptance

**Decision:** Accept the full-period MRMS+RTMA forcing extraction as **PASS_WITH_CAVEATS**.
No rerun required.

**Basis:**

- 63/63 monthly chunks `all_pass=True`; 0 failures across Groups A/B/C (PASS=21/19/23).
- 1,509,422,464 combined rows; 0 row-count formula mismatches (11 RTMA vars x n_basins x successful_hours).
- Schema: `rtma_10wdir_absent=True` and `rtma_orog_absent=True` confirmed for all 63 months.
- 138 missing hour-products across 20 months; all `not_in_s3` (permanent S3 archive absences).
  MRMS: 136 hours; RTMA: 2 hours (2020-11-12T09Z and T10Z - newly discovered in audit).
- 0 product-synchronized gaps; 0 unexpected warnings.
- MRMS 24h window impact: 949 / 45,697 possible windows (2.08%).
- Evidence: `tmp/stage1_forcing_fullperiod_evidence_20260624T060504Z.tar.gz` (local, not committed).
- Audit tables: `tmp/stage1_forcing_fullperiod_postrun_audit_20260624T060504Z/` (local, not committed).

**Caveats recorded:**

1. **Two-commit provenance:** 2020-10 used extractor commit `194a489` (Phase 1 run);
   62 other months used `7e43760` (D1-optimized full-period extractor). Both pass all 12
   validation checks - documentation caveat only, no functional inconsistency.
2. **MRMS not_in_s3 gaps:** 136 missing MRMS hours are permanent S3 absences. Gap policy:
   preserve as NaN in raw curated product; isolated 1h gaps may be interpolated in derived
   package layers only (per `docs/stage1_forcing_fullperiod_postrun_audit_plan.md section 6`).
3. **RTMA gap discovery:** 2 RTMA hours missing in 2020-11. Not anticipated prior to audit.
   Month remains `all_pass=True`; no corrective action warranted.

**This acceptance does not authorize** curated forcing product v001 assembly (requires
visual QC gate) or NeuralHydrology package assembly or model training.

**Full result:** `docs/stage1_forcing_fullperiod_audit.md`

## 2026-06-25/28 Stage 1 Forcing — Pilot Visual QC PASS

**Decision:** Accept the pilot visual QC evidence as **PASS** for the 6-case basin-timeseries
pilot and the 2-case spatial MRMS smoke. This is a technical/rendering PASS and a scientific
QC evidence improvement. It is **not a final full forcing certification**.

**Basin-timeseries pilot (6/6 OK, 2026-06-25):**

- Cases: VQC-001, VQC-004, VQC-007, VQC-009, VQC-012, VQC-020.
- Time-series rendering, MRMS gap labeling (gray bars), RTMA gap labeling (orange shading),
  qobs hydrograph alignment, VQC-001 period-boundary clip annotation — all pass.
- h2o output: `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod_visual_qc_pilot_20260625T123337Z`
- Generated GIF/PNG/CSV/manifest outputs are not committed.

**Spatial MRMS smoke (VQC-009 + VQC-012, 2026-06-25/28):**

- Script: `scripts/generate_fullperiod_spatial_mrms_qc.py`
- Both cases: `basin=Y`, `gauge=Y` (basin polygon and gauge marker rendered).
- Cartopy unavailable on h2o — plain lon/lat axes with pcolormesh raster used. No basemap.
  This is sufficient for spatial placement QC; not a rendering blocker.
- MRMS lon normalization (0–360 → −180–180) applied in script; CAMELSH CRS auto-assigned
  EPSG:4326 (shapefile has no `.prj`; bounds confirmed geographic).

| Case | Observation | Interpretation |
|---|---|---|
| VQC-012 (08155541, small flashy TX) | Strong near-basin rainfall at max-hour | Consistent with sharp qobs response; no alignment failure |
| VQC-009 (09484000, SW monsoon AZ) | Patchy convective rainfall near/partly over Sabino Creek | Weak qobs response plausible (partial spatial overlap); not an extraction failure |

**h2o output directories:**
- VQC-012: `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod_spatial_mrms_qc_smoke_20260625T142012Z`
- VQC-009: `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod_spatial_mrms_qc_smoke_20260625T142332Z/VQC-009/`

**Scope of this acceptance:**
- Authorizes proceeding to curated forcing product v001 design.
- Does **not** authorize skipping the remaining 15 cases if reviewer finds the pilot evidence
  insufficient for full certification.
- Does **not** authorize NeuralHydrology package assembly or model training.
- Generated PNG/GIF/CSV/summary outputs remain under `tmp/` and must not be committed.

**Full evidence:** `docs/stage1_forcing_fullperiod_visual_qc_animation_plan.md`

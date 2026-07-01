# Decision Log

Project: Flash-NH — near-real-time and forecast-aware hydrological modeling pipeline.

## 2026-07-01 Moriah env install PASS + pilot package transfer PASS

**Decision:** Accept Moriah env install (Slurm job `45365952`) and pilot package transfer
as PASS. Both are confirmed done. Smoke 0 is the next step (not done yet; 2K-G-C not yet
complete). Generated evidence (logs) remain on Moriah, not committed locally.

**Env install (job 45365952):**
- Script: `scripts/setup_flashnh_moriah_env.sbatch` (after manual module fixes, see below)
- Env prefix: `/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah`
- `nh-run` confirmed at `envs/flashnh-moriah/bin/nh-run`; `nh-run --help` lists valid modes
  (`train`, `continue_training`, `finetune`, `evaluate`)
- `neuralhydrology` imports OK; no `__version__` attribute (expected)
- Log ended with `=== done ===`. Matplotlib font cache message in stderr is harmless.

**Module fixes (binding for all future Moriah Slurm scripts):**
Initial sbatch run failed because `module` is not in PATH in non-interactive Slurm shells,
and `miniconda3` requires `spack/all` to be loaded first. Three corrections applied:
1. Source a module-system init file at job start if `module` is not already in scope.
2. `module load spack/all` before any other module.
3. Use exact module name `miniconda3/24.3.0-gcc-iqeknet` (not `miniconda3/24.3.0`).
Both `scripts/setup_flashnh_moriah_env.sbatch` and `scripts/run_stage1_smoke0_moriah.sbatch`
updated in this commit.

**Pilot package transfer (h2o → Moriah):**
- Source: `/data42/omrip/Flash-NH/tmp/stage1_nh_pilot_v001/`
- Destination: `/sci/labs/efratmorin/omripo/Flash-NH/data/stage1_pilot_v001`
- Verified: 5 NC files under `time_series/`, `run_provenance.json` present,
  `configs/stage1_smoke0_nh.yml` present, `attributes.csv` present, size 19 MB.

## 2026-07-01 Milestone 2K-F-C: Corrected full-period curated forcing v001 PASS

**Decision:** Accept the corrected full-period curated forcing product v001 rebuild as PASS.
This closes the 2K-F-C-B schema correction loop and unblocks full 2,752-basin NH package
generation (pending Smoke 0 PASS and attribute-source cleanup — see separate gates).

**Build facts (from evidence bundle, locally at `tmp/stage1_curated_forcing_v001_corrected_fullperiod_evidence/`):**
- Product: `stage1_basin_hourly_forcings_v001`
- h2o location: `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/stage1_basin_hourly_forcings_v001`
- Period: 2020-10-14T00:00:00Z – 2025-12-31T23:00:00Z
- Months processed: 63 / 63; basins built: 2,752 / 2,752; 0 failed
- Rows per basin: 45,720 (full period)
- Total MRMS gap-hours: 374,272 (= 136 × 2,752 — exact match)
- Total RTMA gap-hours: 5,504 (= 2 × 2,752 — exact match)
- Wall time: 51,932.6 s (14.43 h)
- Run start: 2026-06-30T10:08:53Z; run end: 2026-07-01T00:34:26Z
- Repo commit at run: `5f07d4b`

**Audit result (full-period mode): PASS**
- 2,752 / 2,752 basins checked; 45,720 rows per basin
- MRMS gap-hours/basin = 136 ✓; RTMA gap-hours/basin = 2 ✓
- Known RTMA gap timestamps: 2020-11-12T09Z and T10Z ✓

**Sample20 diagnostic: ALL PASS**
- 20 Parquets spot-checked; each 45,720 rows
- `mrms_qpe_1h_mm` non-null = 45,584 (= 45,720 − 136 MRMS gaps) ✓
- All RTMA variables non-null = 45,718 (= 45,720 − 2 RTMA gaps) ✓
- `rtma_2d_K` populated ✓ (confirms 2K-F-C-B dewpoint mapping fix)
- `rtma_weasd_kgm2` absent ✓ (confirms 2K-F-C-B schema removal)

**Generated evidence (not committed):**
- `tmp/stage1_curated_forcing_v001_corrected_fullperiod_evidence/` (local archive)
- Includes `build_summary.md`, `audit_summary.md`, `run_provenance.json`, `manifest.json`,
  `checksums.sha256`, `dataset_config.json`, `build.log`

**What this unblocks:** Full 2,752-basin NH package generation now has a valid corrected
forcing library input. Remaining gates before full NH package: (1) Smoke 0 PASS;
(2) attribute-source cleanup (staged `all_basins_merged.parquet` not committed).

## 2026-06-30 Milestone 2K-G-C-A: Moriah GPU/Conda/Slurm preflight facts recorded

**Decision:** Record real Moriah/HURCS facts gathered via interactive `ssh`/`srun`
reconnaissance, and prepare (but not run) two Slurm templates. This is **preflight
documentation and script preparation only** — no job was run on Moriah, the env is not
installed, the pilot package was not transferred, and Smoke 0 was not attempted. 2K-G-C
is not complete; only 2K-G-C-A.

**Facts recorded (see `docs/stage1_neuralhydrology_preflight.md` §10.6 for full detail):**
1. Login node `moriah-gw-01`; lab storage `/sci/labs/efratmorin/omripo/Flash-NH` with
   subdirs `repos, envs, data, runs, logs, slurm, evidence`. Do not rely on
   `/sci/home/omripo` inside Slurm jobs.
2. Slurm partitions confirmed via `sinfo`: `catfish` (L4, 7-day limit, chosen for
   Smoke 0), `salmon` (L40S), `goldfish` (H200), `dogfish` (A100, drained at check time),
   `glacier` (CPU default).
3. Working interactive allocation:
   `srun --partition=catfish --gres=gpu:l4:1 --cpus-per-task=4 --mem=16G --time=00:10:00 --pty bash`.
4. On allocated node `catfish-05`: L4 GPU, 23034 MiB; `nvidia-smi` requires
   `module load nvidia/580.95.05` (reports driver 580.95.05 / CUDA 13.0); toolkit
   `module load cuda/12.8.1` confirmed (`nvcc` → 12.8, V12.8.93).
5. Conda is module-gated; compute allocations auto-load `miniconda3/24.3.0-gcc-iqeknet`.
   Moriah env must be a **prefix env** (`conda create -p ...`) under the Flash-NH project
   root, not a named env and not under `/sci/home`.

**Scripts prepared (not run):**
- `scripts/setup_flashnh_moriah_env.sbatch` — env install on `catfish`. Leaves the
  PyTorch CUDA wheel choice as an explicit TODO rather than guessing a wheel tag, since
  the driver-reported CUDA (13.0) and loaded toolkit (12.8.1) differ and the actual
  compatible wheel was not verified.
- `scripts/run_stage1_smoke0_moriah.sbatch` — Smoke 0 on `catfish`. Chooses
  `nh-run train --config-file ...` (upstream NH console-script entry point) as the first
  invocation to try, with `python -m neuralhydrology.nh_run train` documented as the
  fallback. Explicitly does **not** use `python -m neuralhydrology.training` — the
  invocation baked into `scripts/build_stage1_nh_package.py`'s `_write_slurm` helper —
  which is flagged as likely incorrect and unverified. Reconciling that generator
  function is a follow-up, not done in this milestone.

**Transfer procedure documented (not executed):** `scp` from h2o
`/data42/omrip/Flash-NH/tmp/stage1_nh_pilot_v001/` to Moriah
`/sci/labs/efratmorin/omripo/Flash-NH/data/stage1_pilot_v001/`, verified by NC file count
(expect 5), `run_provenance.json` presence, and package size (~25 MB). No checksum
manifest exists for this package, so file-count/manifest-presence is the practical check.

**Not done:** h2o jobs, Moriah jobs, env install, package transfer, Smoke 0, 2K-G-C
completion.

## 2026-06-30 Milestone 2K-G-B h2o validation: NH pilot package PASS

Audit result: PASS — 0 errors, 5 warnings, 217 OK checks.
Package: `/data42/omrip/Flash-NH/tmp/stage1_nh_pilot_v001/`
Build time: 4.2 s; wall timestamp: 2026-06-30T12:38:35Z.

**Findings and decisions:**

**1. 5-basin corrected pilot is sufficient for package-builder validation.**
All 5 basins pass all structural, gap-count, and dewpoint-fix checks. The builder and
auditor are validated. Full 2,752-basin package generation is a separate, authorized step.

**2. Gap-fill report matches expected counts exactly.**
MRMS: 136 NaN → 0.0 mm per basin (all 5). RTMA: 2 NaN → linear interp per variable per
basin (all 10 RTMA variables, all 5 basins). No unexpected gaps introduced by reindex.

**3. qobs NaN preserved exactly; 5 warnings are expected.**
Warnings are informational: qobs NaN counts of 515, 6,751, 12,088, 3,035, 6 for the five
basins. These are the missing-discharge gaps from the target package v001. NH loss-masks
these at training time. No action needed.

**4. rtma_2d_K non-null == 45,720 confirmed per NC.**
The 2K-F-C-B dewpoint mapping fix (`d2m` → `2d` in the forcing builder) is confirmed to
have propagated correctly into the NH package. This check is retained in the auditor
permanently for any future package rebuild.

**5. Static attribute file not committed to git — cleanup gate established.**
`reports/flashnh_basin_screening_v001/all_basins_merged.parquet` is not tracked in git
(confirmed with `git ls-files` on h2o). Builder used a manually staged copy at
`/data42/omrip/Flash-NH/tmp/all_basins_merged.parquet`. The pilot PASS is valid.
Resolution options before full-scale packaging:
  (a) Commit the parquet to the repo (adds ~1–2 MB to git history; clean path).
  (b) Document it as a canonical h2o-resident file at a fixed path, with explicit
      provenance note in `--attributes-csv` docs and run_provenance.json.
This is a cleanup gate, not a scientific blocker.

**6. Next milestone is 2K-G-C: Moriah transfer + NH environment preflight + Smoke 0.**
Package is ready. Transfer via `scp`. No scientific training; Smoke 0 is plumbing/ingestion
verification (seq_length=24, 2 epochs, 5 basins, rain-only). Smoke 0 PASS = NH environment
and package format confirmed compatible.

## 2026-06-30 Milestone 2K-G-B: NH package builder and auditor design decisions

Scripts implemented: `scripts/build_stage1_nh_package.py`, `scripts/audit_stage1_nh_package.py`.
Status: IMPLEMENTED (local syntax PASS); h2o run pending.

**1. Attribute source: `reports/flashnh_basin_screening_v001/all_basins_merged.parquet`.**
This file is committed to the repo and all 5 pilot basins confirmed present with required
columns (`DRAIN_SQKM`, `LAT_GAGE`, `LNG_GAGE`, `BFI_AVE`). `--attributes-csv` accepts
`.parquet` or `.csv`; STAID column (int64) normalized to 8-char zero-padded string.
All available columns (not just the 4 required) are written to `attributes.csv` — NH
reads only the columns listed in `static_attributes` config at runtime.

**2. All 14 NC variables: 11 forcing data + 2 gap flags + qobs_m3s.**
`rtma_sp_Pa` is included in the NC (not excluded) so it is available for Smoke 2 without
rebuilding the package. It is excluded from the `dynamic_inputs` list in `stage1_smoke1_nh.yml`.

**3. Gap flags stored as float32 (0.0/1.0), not bool.**
NH GenericDataset expects numeric input arrays. Bool xarray variables may cause issues with
NH normalization. Explicitly converting gap flags to float32 in the builder.

**4. Atomic NC writes (tmp + rename).**
Same pattern as target builder: write to `{STAID}.nc.tmp`, then rename. Avoids partial-write
files if the builder is interrupted.

**5. Auditor checks mrms_qpe_1h_mm_gap sum == 136 and rtma_gap sum == 2 per basin.**
These are the expected gap counts from the corrected v001 forcing library. A mismatch would
indicate a gap-fill bug or a wrong forcing source directory.

**6. Auditor checks rtma_2d_K non-null == 45720 explicitly.**
This check directly confirms that the 2K-F-C-B dewpoint mapping fix (`d2m` → `2d`) was
correctly carried through to the NH package NC files. If the old mapping bug recurs in a
future rebuild, this check will catch it.

## 2026-06-30 Milestone 2K-G-A corrections: Smoke design and gap-fill policy revision

Corrections to the 2K-G-A preflight design (commit `fa6754b`) before 2K-G-B implementation.
No code changes; docs-only patch.

**1. Smoke 0 `seq_length`: 336 h → 24 h; add `predict_last_n: 1`.**
Smoke 0 is a pure plumbing/ingestion test, not a scientific baseline. 14-day lookback is
unnecessary overhead for verifying that NH loads the package and produces finite loss.
24 h minimises runtime and memory before package-loading is proven. `seq_length: 336` is
reserved for later hyperparameter testing.

**2. Smoke 1 `seq_length`: not locked to 336 h.**
First meteorology smoke uses `seq_length: 72` or `168` h as the next step. 336 h is a
later candidate, not the first default.

**3. MRMS gap-fill policy: Smoke 0/1 pilot policy only — not final scientific training policy.**
Precipitation is the primary forcing driver; silently treating archive gaps as true no-rain
must not carry into scientific baseline training. For final training, evaluate:
- window/sample exclusion (exclude training windows that intersect MRMS gap hours —
  do NOT remove rows from the NC file; the 45,720-h `date` coordinate stays aligned);
- or a deliberately tested NH `nan_handling_method`.
The 0.0 mm fill + gap flag strategy is accepted for Smoke 0/1 only.

**4. RTMA gap-fill: linear interpolation (2 hours) is accepted as pilot/package policy;
review before final scientific training.**

**5. 2K-G-B unblocked from full rebuild wait.**
The 5-basin NH pilot package builder can be implemented and tested now, using the
already-passing corrected 5-basin forcing pilot. Full-scale package generation (2,752
basins) waits for the full rebuild PASS, but the 5-basin builder and audit do not.

## 2026-06-30 Milestone 2K-G-A: NeuralHydrology Pilot Package Preflight Design

Design frozen in `docs/stage1_neuralhydrology_preflight.md` (Part I). Key decisions:

**1. Package format: GenericDataset single NC per basin.**
One NC per basin with all dynamic vars (forcings + gap flags) + `qobs_m3s` target on
a shared `date` coordinate. Matches Milestone 2G format proven with NH, avoids a
custom dataset class. Float32 values, `_FillValue=-9999.0`, no tz offset in coordinate.

**2. Gap-fill policy for NH package (binding for v001).**
- MRMS gaps (136 h / basin, 0.30%): fill with 0.0 mm (conservative no-rain assumption)
- RTMA gaps (2 h / basin, 0.004%): fill with linear interpolation (2 hours; both neighbors always available)
- Gap flags (`mrms_qpe_1h_mm_gap`, `rtma_gap`) retained as explicit dynamic inputs
- Do NOT rely on NH `nan_handling_method` as the primary strategy; pre-fill in the package builder
- Rationale: transparency — NaN in dynamic inputs is dangerous in LSTM by default; pre-fill is
  auditable in the package file; gap flags preserve the information signal

**3. Smoke levels.**
- Smoke 0 (rain-only technical): mrms_qpe_1h_mm + gap flag; 5 basins; 2 epochs; purpose is
  NH load/train verification only, not a scientific model
- Smoke 1 (minimal meteorology): + rtma_{2t,2d,2sh,10u,10v} + rtma_gap
- rtma_sp_Pa: include in NC file (for future use), exclude from Smoke 1 dynamic_inputs
  (large magnitude ~70k–101k Pa; defer normalization review to Smoke 2)

**4. Train/val/test split.**
Train: 2020-10-14 – 2022-12-31 | Val: 2023 | Test: 2024-2025
Rationale: 2024–2025 is the quasi-operational period; hold out entirely. Val is 2023
for generalization monitoring; contains varied seasonality.

**5. NH setup: clean upstream clone, no fork until specific limitation demonstrated.**
Old Flash-NH fork is abandoned. All custom logic lives in: NH YAML configs (in this repo),
package builder script, and future `src/flashnh/` custom classes. Fork only when a
config-layer workaround is exhausted.

**6. Moriah layout.**
`/sci/labs/efratmorin/omripo/Flash-NH/{repos,envs,data,runs,logs,slurm,evidence}`
Blocking unknown: GPU partition name, CUDA version — must check Moriah wiki and `sinfo`.

## 2026-06-30 Milestone 2K-F-C-B: Curated Forcing v001 Schema/Mapping Correction

Full-period build (2,752 basins × 45,720 h) structurally passed on h2o (2026-06-30,
commit `addfdd2`, 14.49 h wall). Post-build non-null check found two all-NaN variables,
triggering a schema correction before certification.

**Schema findings:**

| Variable | Non-null (5 sampled basins) | Decision |
|---|---|---|
| `rtma_2d_K` | 0 / 45,720 | **Retain** — mapping bug fixed (`d2m`→`2d`) |
| `rtma_weasd_kgm2` | 0 / 45,720 | **Remove** — `weasd` absent from all 63 source months |
| `rtma_2t_K` | 45,718 / 45,720 | Retain (normal) |
| `rtma_sp_Pa` | 45,718 / 45,720 | Retain (normal) |

**Decisions (all binding for v001):**

1. **Dewpoint retained, mapping corrected.** Source variable is `2d` (`dewpoint_temperature_2m`),
   not `d2m`. Confirmed present with `recommended_for_initial_model=True` in all 5 sampled months.
   Both builders updated: `"2d" → "rtma_2d_K"`.

2. **`rtma_weasd_kgm2` removed from v001 schema.** `weasd` is absent from all 63 monthly
   source chunks. RTMA precipitation (`ACPC01`) is not present in the RTMA CONUS source.
   Precipitation is supplied by MRMS QPE; no RTMA precip column is added. `rtma_weasd_kgm2`
   is now in `_FORBIDDEN_COLS` in the auditor — its presence in output is a FAIL.

3. **Full-period structural build is schema-superseded, not failed.** Gap counts (136 MRMS,
   2 RTMA per basin), row counts (45,720), and checksums were correct. The product correctly
   reflects the source data; the errors were a missing dewpoint (now fixed) and a spurious
   NaN column (now removed). A corrected 5-basin full-period pilot on h2o is required before
   the full 2,752-basin rebuild is authorized.

4. **Auditor non-null coverage checks added.** Full-period mode now verifies exact non-null
   counts: `mrms_qpe_1h_mm` → 45,584; each RTMA var → 45,718. Single-month mode: not-all-NaN
   guard for all data variables.

5. **`build.log` caveat.** An accidental second launch was stopped early after the first PASS.
   Post-interruption full-period audit PASS confirmed the product was not corrupted. `build.log`
   may contain aborted-rerun lines after the first complete PASS block.

**Corrected v001 schema:** 1 MRMS + 10 RTMA + 2 gap flags = 13 columns total (was 14).

**Evidence:** `tmp/stage1_curated_forcing_v001_schema_issue_evidence/` (not committed).

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

## 2026-06-29 Stage 1 Forcing — Milestone 2K-F-A: Curated Product v001 Design

**Decision:** Freeze the curated forcing product v001 contract. No data is built in this
milestone. Builder and auditor implementation are deferred to Milestone 2K-F-B.

**Design decisions (all binding for v001):**

1. **Format: wide Parquet per basin.** One row per hour; one column per variable. The monthly
   extraction Parquets (long format) remain unchanged. The per-basin product is a separate
   derived format chosen for NH DataLoader compatibility.
2. **Schema: 12 data columns + 2 gap-flag columns.** 1 MRMS variable (`mrms_qpe_1h_mm`) +
   11 RTMA variables (9 dynamic + `vis` + `ceil`). Gap flags: `mrms_qpe_1h_mm_gap` (bool)
   and `rtma_gap` (bool). `10wdir` and `orog` excluded (absent from S3 in all 63 months).
3. **Gap policy: NaN preserved, no imputation, no row dropping.** Known gaps (136 MRMS hours,
   2 RTMA hours) are NaN in value columns and `True` in gap-flag columns. Every gap hour
   has a complete row in the hourly index.
4. **Smoke test month: 2020-11.** Chosen because it contains the 2 known RTMA gap hours
   (2020-11-12T09Z/T10Z) and 0 MRMS gaps — best stress test of RTMA gap handling.
5. **Product name and path confirmed:** `stage1_basin_hourly_forcings_v001` under
   `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/stage1_basin_hourly_forcings_v001/`
   (full build); smoke under `tmp/stage1_basin_hourly_forcings_v001_smoke_<TIMESTAMP>/`.

**All five open choices resolved (2026-06-29 follow-up patch).**

**Full design:** `docs/stage1_curated_forcing_product_v001_design.md`

## 2026-06-29 Stage 1 Forcing — Milestone 2K-F-A: Open Choices Resolved

**OC-1 — Script naming:** Builder: `scripts/build_stage1_curated_forcing_basin_parquets.py`;
auditor: `scripts/audit_stage1_curated_forcing_basin_parquets.py`. Legacy name
`build_stage1_forcing_basin_ncs.py` is retired. Rationale: product format is wide Parquet;
the future NH-package builder (separate milestone) will create NetCDFs.

**OC-2 — Full-build output location:** First build stays under
`/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/stage1_basin_hourly_forcings_v001/`.
Promotion to `/data42/hydrolab/Data/Flash-NH_data/` is a separate explicit gate after
full audit, checksums, and evidence-bundle review.

**OC-3 — RTMA gap flag granularity:** One shared `rtma_gap` boolean column for v001.
Known RTMA gaps are whole-product-hour absences (2020-11-12T09Z/T10Z), not variable-specific
decode failures. The auditor must check per-variable completeness and fail if variable-specific
missingness appears outside known product-hour gaps.

**OC-4 — `vis` and `ceil` inclusion:** Include all 11 extracted RTMA variables in the
curated product. Curated product preservation and first-model-input variable selection are
separate decisions; the first NH smoke config may use a narrower subset without changing v001.

**OC-5 — Remaining 15 VQC cases:** Not required before the 2K-F-B smoke test or the full
2,752-basin build. Gate for full build: 2K-F-B smoke PASS + no schema/gap/provenance failures.
Render 2–3 additional targeted VQC cases only if the smoke or design review reveals a concern.

## 2026-06-29 Stage 1 Forcing — Milestone 2K-F-B: Curated Forcing v001 Builder + Smoke PASS

**Decision:** Milestone 2K-F-B is COMPLETE. Builder, auditor, and h2o launcher implemented
and smoke-tested. Commit `6f4de498f1326e5e6fcd3de8157ba410ad28a6a9`.

**Smoke test result (h2o, 2026-06-29T13:27:57Z):**

| Metric | Value |
|---|---|
| Month | 2020-11 |
| Basins | 5 (`01440000`, `03021350`, `08155541`, `09484000`, `01019000`) |
| Hours per basin | 720 |
| MRMS gap-hours | 0 (correct — 2020-11 has no MRMS S3 gaps) |
| RTMA gap-hours | 10 total (2/basin at `2020-11-12T09:00:00Z` and `T10:00:00Z`) |
| Coverage fraction | 0.9972 (718 valid combined hours / 720) |
| Auditor | PASS — exit 0; all metadata, checksum, schema, and gap-flag checks passed |
| Wall time | 0.1 s |
| h2o output | `/data42/omrip/Flash-NH/tmp/stage1_curated_forcing_smoke_20260629T132757Z` |

**Gap verification:**
- `rtma_gap=True` at both known gap timestamps for all 5 basins — confirmed
- All 11 RTMA data columns NaN at gap hours — confirmed
- `mrms_qpe_1h_mm_gap=False` at RTMA-only gap hours (no false flagging) — confirmed
- SHA-256 verified for all 5 Parquets

**Prior failed explicit-basin run (same session):**
`02231000` was passed via `--staids` but is absent from the 2020-11 monthly source chunk.
Builder correctly halted with 0 basins built rather than silently skipping. Not a smoke
failure. Basin replaced by `01019000` for the passing 5-basin run.

**audit_summary.md gap:**
The auditor writes its verdict to stdout (captured in `smoke.log`). It does not write a
standalone `audit_summary.md`. For the full build (Milestone 2K-F-C), the auditor must
write `audit_summary.md` to the product directory before the build is closed.
This is a pre-build implementation requirement, not a blocker for closing 2K-F-B.

**Implementation decisions binding for 2K-F-C (full build):**
1. Metadata in JSON: `manifest.json`, `dataset_config.json`, `run_provenance.json` (not `.csv`/`.yaml`).
2. Per-basin files: flat `time_series/{STAID}.parquet` (not `{STAID}/{STAID}_hourly_forcings.parquet`).
3. Gap detection by row absence from source Parquet — consistent with `not_in_s3` semantics.
4. RTMA variable aliases: `sh2`/`2sh` → `rtma_2sh_kgkg`; `gust`/`i10fg` → `rtma_gust_ms`.
5. Path safety guard in launcher: `OUT_DIR` must begin with `/data42/omrip/Flash-NH/`.

**Authorization:** Full 2,752-basin build (Milestone 2K-F-C) requires explicit authorization.
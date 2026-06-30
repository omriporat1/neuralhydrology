# Flash-NH Current State

Last updated: 2026-06-30 (2K-G-C-A — Moriah preflight facts recorded)

## Current milestone

**Milestone 2K-G-C-A PREFLIGHT FACTS RECORDED (2026-06-30): Moriah GPU/Conda/Slurm
reconnaissance + two new Slurm templates.**

This is documentation and script preparation only. **No job has been run on Moriah, the
flashnh-moriah env is not installed, the pilot package has not been transferred, and
Smoke 0 has not been attempted.** 2K-G-C is not complete — only this preflight sub-step.

**Facts confirmed via interactive `ssh`/`srun` reconnaissance:**
- Login node `moriah-gw-01`; project root `/sci/labs/efratmorin/omripo/Flash-NH`
  (`repos, envs, data, runs, logs, slurm, evidence`). Do not rely on `/sci/home/omripo`
  inside Slurm jobs.
- Slurm partitions (`sinfo`): `catfish` (L4, `gpu:l4:8`, 7-day limit — **chosen for
  Smoke 0**), `salmon` (L40S), `goldfish` (H200), `dogfish` (A100, drained), `glacier`
  (CPU default).
- Interactive allocation confirmed working:
  `srun --partition=catfish --gres=gpu:l4:1 --cpus-per-task=4 --mem=16G --time=00:10:00 --pty bash`.
- GPU node `catfish-05`: L4, 23034 MiB; `nvidia-smi` needs `module load nvidia/580.95.05`
  (driver 580.95.05, CUDA 13.0); `module load cuda/12.8.1` confirmed (`nvcc` → 12.8.93).
- Conda is module-gated (`miniconda3/24.3.0-gcc-iqeknet`); Moriah env must be a **prefix
  env** under the Flash-NH project root, not a named env, not under `/sci/home`.

**New templates prepared (not run):**
- `scripts/setup_flashnh_moriah_env.sbatch` — env install; PyTorch CUDA wheel left as an
  explicit TODO (driver CUDA 13.0 vs. toolkit 12.8.1 — exact compatible wheel unverified).
- `scripts/run_stage1_smoke0_moriah.sbatch` — Smoke 0; chooses `nh-run train
  --config-file ...` as the first invocation to try (upstream NH console-script entry
  point), with `python -m neuralhydrology.nh_run train` as documented fallback.
  Explicitly avoids the unverified `python -m neuralhydrology.training` invocation still
  present in `scripts/build_stage1_nh_package.py`'s `_write_slurm` helper.

**Transfer procedure documented (not executed):** `scp` h2o
`/data42/omrip/Flash-NH/tmp/stage1_nh_pilot_v001/` → Moriah
`/sci/labs/efratmorin/omripo/Flash-NH/data/stage1_pilot_v001/`; verify via NC file count
(5), `run_provenance.json` presence, package size (~25 MB).

Full detail: `docs/stage1_neuralhydrology_preflight.md` §10.6.
Decisions: `docs/decision_log.md` (2026-06-30 Milestone 2K-G-C-A entry).

---

**Milestone 2K-G-B COMPLETE (2026-06-30): NeuralHydrology pilot package built and audited on h2o.**

h2o audit result: **PASS** — 0 errors, 5 warnings, 217 OK checks.
Build time: 4.2 s. Audit timestamp: 2026-06-30T12:38:40Z.
Package: `/data42/omrip/Flash-NH/tmp/stage1_nh_pilot_v001/`
Evidence: `tmp/stage1_nh_pilot_v001_evidence/` (not committed)

**5-basin audit summary (all pass):**

| Basin | Rows | MRMS gap | RTMA gap | qobs NaN | qobs coverage |
|---|---|---|---|---|---|
| 01019000 | 45,720 | 136 | 2 | 515 | 98.87% |
| 01022500 | 45,720 | 136 | 2 | 6,751 | 85.23% |
| 01033000 | 45,720 | 136 | 2 | 12,088 | 73.56% |
| 01038000 | 45,720 | 136 | 2 | 3,035 | 93.36% |
| 01049500 | 45,720 | 136 | 2 | 6 | 99.99% |

**5 warnings (all expected):** one per basin — qobs NaN counts logged (normal; NH loss-masks missing targets).
No forcing NaN warnings. All forcing variables non-null after gap-fill.

**Key checks confirmed:**
- All 14 variables present per NC (11 forcing + 2 gap flags + qobs_m3s)
- `rtma_weasd_kgm2` absent (forbidden — confirmed)
- `rtma_2d_K` non-null == 45,720 (confirms 2K-F-C-B dewpoint mapping fix carried through)
- `mrms_qpe_1h_mm_gap sum == 136` per basin; `rtma_gap sum == 2` per basin
- Gap fill: MRMS 136 NaN → 0.0 mm/basin; RTMA 2 NaN → linear interp per variable/basin

**Static attribute caveat (cleanup required before full-scale package):**
`reports/flashnh_basin_screening_v001/all_basins_merged.parquet` is **not tracked in git**
(verified with `git ls-files` on h2o). The h2o builder used a manually staged copy at
`/data42/omrip/Flash-NH/tmp/all_basins_merged.parquet`.
The 5-basin pilot PASS is valid. Before full 2,752-basin NH package generation, this file
must be made canonical: committed to the repo or documented as a stable h2o-resident input
with explicit provenance. This is a cleanup gate, not a blocker for Moriah transfer.

**Package structure (on h2o):**
```
/data42/omrip/Flash-NH/tmp/stage1_nh_pilot_v001/
  time_series/{STAID}.nc     # 5 NCs; 14 vars; 45,720 rows; float32; _FillValue=-9999.0
  attributes.csv             # 5 basins × 47 cols
  basins/smoke{0,1}_{train,val,test}.txt
  configs/stage1_smoke{0,1}_nh.yml
  slurm/smoke{0,1}.sh        # Moriah Slurm job templates
  manifests/                 # dataset_manifest.json + variable_schema.csv + gap_fill_report.csv + per_basin_summary.csv
  run_provenance.json + README.md + audit_summary.md
```

**Next: 2K-G-C — Moriah transfer + environment preflight + Smoke 0.**
Transfer pilot package (`scp`), confirm NH conda env on GPU node, run Smoke 0 (seq_length=24, 2 epochs).
No NH training has run yet. Full 2,752-basin NH package generation waits for:
(1) corrected full forcing rebuild PASS on h2o; (2) attribute-source cleanup.

---

**Milestone 2K-G-A COMPLETE (2026-06-30): NeuralHydrology pilot package preflight design + corrections.**

Design frozen in `docs/stage1_neuralhydrology_preflight.md` (Part I), with corrections applied
after initial commit `fa6754b`:
- NH package format: GenericDataset single-NC-per-basin, `date` coord, float32, `_FillValue=-9999.0`
- Smoke 0: rain-only (mrms_qpe_1h_mm + gap flag, 5 basins, 1–2 epochs); `seq_length: 24`, `predict_last_n: 1`
- Smoke 1: minimal meteorology (6 forcings: mrms + rtma_{2t,2d,2sh,10u,10v}); `seq_length: 72` or `168`
- Gap-fill policy (Smoke 0/1 pilot policy only): MRMS gaps → 0.0 mm; RTMA gaps → linear interp; gap flags retained
- Final training gap policy: window-exclusion preferred over silent fill; to be decided after Smoke 1
- Moriah layout: `/sci/labs/efratmorin/omripo/Flash-NH/{repos,envs,data,runs,logs,slurm,evidence}`
- NH setup: clean upstream `neuralhydrology` clone; no fork until specific limitation demonstrated

---

**Milestone 2K-F-C-B COMPLETE (2026-06-30): Curated forcing schema/mapping correction.**

Full-period build structurally PASS on h2o (2026-06-30, 2,752 basins, 45,720 h, 14.49 h wall),
but post-build non-null check found two all-NaN variables. Build is **schema-superseded**;
corrected rebuild required before final certification.

**Schema issues found and corrected in code:**
- `rtma_2d_K` (dewpoint): all-NaN because builder mapped source `d2m` → `rtma_2d_K`, but
  actual source variable is `2d`. Fixed: `"2d" → "rtma_2d_K"` in both builders.
- `rtma_weasd_kgm2`: all-NaN because `weasd` is absent from all 63 monthly source chunks.
  RTMA precipitation (`ACPC01`) is not present. Removed from schema entirely.
- `rtma_2d_K` is **retained** (source `2d` confirmed present in all sampled months with
  `variable_standard_name=dewpoint_temperature_2m`).

**Corrected v001 schema:** 1 MRMS variable + 10 RTMA variables + 2 gap flags = 13 columns.

**Full-period structural build evidence (schema-superseded, not committed):**
- Period: 2020-10-14T00Z – 2025-12-31T23Z
- 63/63 months, 2,752 basins, 45,720 rows/basin, 374,272 MRMS gap-hrs, 5,504 RTMA gap-hrs
- Full-period audit: PASS (structural); wall time 14.49 h; commit at run `addfdd2`
- Note: accidental second launch was stopped early; post-interruption audit PASS confirmed
  product not corrupted. `build.log` may contain aborted-rerun lines after first PASS.
- Evidence under `tmp/stage1_curated_forcing_v001_schema_issue_evidence/` (not committed)

**Next h2o action:** corrected 5-basin full-period pilot, then full rebuild authorization.
**Design doc:** `docs/stage1_curated_forcing_product_v001_design.md`

---

**Milestone 2K-F-B COMPLETE (2026-06-29): Curated forcing product v001 builder + smoke test — PASS.**

Builder (`build_stage1_curated_forcing_basin_parquets.py`), auditor
(`audit_stage1_curated_forcing_basin_parquets.py`), and h2o launcher implemented and committed
(`6f4de49`). 5-basin / 2020-11 smoke test run on h2o: all 9 acceptance criteria PASS.
- 5/5 basins (`01440000`, `03021350`, `08155541`, `09484000`, `01019000`); 720 h each
- 0 MRMS gap-hours; 10 RTMA gap-hours (2/basin) at 2020-11-12T09Z/T10Z
- Coverage 0.9972; `rtma_gap=True` confirmed at both known timestamps; MRMS not falsely flagged
- Auditor exit 0; SHA-256 checksums verified; commit at run `6f4de498`
- Note: `02231000` attempted but absent from 2020-11 source; builder correctly halted; not a failure
- h2o output: `/data42/omrip/Flash-NH/tmp/stage1_curated_forcing_smoke_20260629T132757Z`

---

**Milestone 2K-E COMPLETE (2026-06-24): Full-period forcing extraction audit — PASS_WITH_CAVEATS.**

Full-period MRMS+RTMA basin-average forcing extraction (63 months, 2020-10 → 2025-12,
2,752 basins) is complete on h2o. Post-run audit finished locally.

**Audit result summary:**
- 63/63 months `all_pass=True`, 0 failures
- 1,509,422,464 combined rows (125,447,168 MRMS + 1,383,975,296 RTMA); 0 row-count mismatches
- 11 RTMA variables, uniform; `rtma_10wdir_absent` and `rtma_orog_absent` confirmed all months
- 138 missing hour-products across 20 months (136 MRMS + 2 RTMA), all `not_in_s3`
- MRMS 24h window impact: 949 / 45,697 windows (2.08%); RTMA: 25 / 45,697 (0.05%)
- 0 basin×product pairs incomplete across all months
- 0 unexpected warnings
- Caveat: two-commit provenance (2020-10 → `194a489`; 2020-11 → 2025-12 → `7e43760`); documentation only
- **No rerun required**

**Full audit result:** `docs/stage1_forcing_fullperiod_audit.md`  
**Audit plan:** `docs/stage1_forcing_fullperiod_postrun_audit_plan.md`  
**Generated audit tables (not committed):** `tmp/stage1_forcing_fullperiod_postrun_audit_20260624T060504Z/`

**Next step:** Milestone 2K-F-B — curated forcing product v001 builder + 5-basin smoke test on h2o.
Design frozen in `docs/stage1_curated_forcing_product_v001_design.md`. Not model training yet.

**Pilot visual QC PASS (2026-06-25/28):**
- Basin-timeseries pilot: 6/6 cases OK (VQC-001, -004, -007, -009, -012, -020).
  Time-series rendering, gap labeling, VQC-001 boundary clip, and qobs alignment all pass.
  h2o output: `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod_visual_qc_pilot_20260625T123337Z`
- Spatial MRMS smoke (VQC-012, VQC-009): basin=Y, gauge=Y. Raster placement consistent
  with observed qobs responses. No extraction or alignment failures detected.
  h2o output (VQC-012): `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod_spatial_mrms_qc_smoke_20260625T142012Z`
- This is a technical/rendering PASS and scientific QC evidence improvement.
  It is **not a final full forcing certification** — 15 of 21 cases not yet animated.
- Generated outputs (PNG/GIF/CSV/summary) remain under `tmp/` and are not committed.
- See `docs/stage1_forcing_fullperiod_visual_qc_animation_plan.md` for full evidence.

---

Stage 1 full 2,843-basin USGS IV target acquisition structurally complete (2026-06-13).
Target policy configured (`config/stage1_target_policy.yaml`, 2026-06-15).
h2o preprocessing environment installed and smoke-tested (`flashnh-stage1`, 2026-06-15).
Target package builder + auditor implemented, smoke-tested, and h2o policy-smoke PASS (2026-06-15).
**v001 target package (2,752 basins) built and audited on h2o (2026-06-16): PASS — 0 errors, 0 warnings.**
**Milestone 2K-A COMPLETE (2026-06-18): v001 basin-weight tables built on h2o — 2,752/2,752 basins, PASS.**
**Milestone 2K-B COMPLETE (2026-06-18): forcing extraction smoke test — PASS. RTMA 48/48 h; MRMS 27/48 h (21 `not_in_s3`, expected early archive gap).**
**Milestone 2K-C COMPLETE (2026-06-18): October 2020 one-month run — PASS.
432h, 2,752 basins, 396/432 MRMS, 432/432 RTMA, 14,167,296 rows, 15h 05m wall.
Full-period extraction PAUSED — 66.5-day projected wall time requires 2K-D optimization.**
**Milestone 2K-D COMPLETE (2026-06-20): D1 serial optimization → 24.7× speedup
(91.9 s → 2.17 s/hr, commit `3ff4965`). Outer-parallelism x3×dw6 → 3.04 days projected
(commit `a275296`). D2 deferred. x4 not recommended.
Decision: full-period launch — 3 concurrent chunks × 6 download workers.**
**Milestone 2K-E pre-launch patch COMPLETE (2026-06-20): `GROUP_ID=A/B/C` and `DRY_RUN=1`
added to fullperiod launcher; path safety guard and per-group logs; reporter updated.
Dry-run validation pending on h2o. Full-period extraction NOT yet launched.**

See `docs/stage1_hpc_transition_preflight.md` for the full audit summary and
`docs/stage1_target_policy.md` for target-policy rationale.

### Quick summary

- 2,843 canonical hourly NetCDF files on h2o at `/data42/omrip/Flash-NH/tmp/stage1_full_2843/`
- Coverage 0.9652 overall; 2,754 basins with `historical_training_utility_flag=True`
- 89 basins with late-period gaps (`TARGET_OPERATIONAL_REVIEW`) — hold out of first package
- 18 basins with negative qobs in the acquisition audit — set to NaN during package build
  (2 heavily-negative special-review basins excluded from v001; 16 basins cleaned, 235 values neg→NaN)
- `TARGET_QUALITY_REVIEW` (1,375 basins): eligible for training; spike flag is advisory only
- No systematic offset issues (0 basins)

### h2o / Moriah operating plan (as of 2026-06-15)

Key policy clarifications from PI:
- h2o is **storage, downloads, preprocessing, and assembly** — not training
- h2o has **no usable GPU** (`nvidia-smi` not found; PI confirms)
- No scheduler by design; `screen` is the agreed background job manager
- CPU compute allowed with etiquette: ≤50–60% CPU; start 16–32 workers; notify before long jobs
- `/data42/omrip` is not auto-deleted; `/data42` is not backed up
- `/data42/hydrolab/Data/Flash-NH_data/` subfolders allowed with reproducibility provenance
- **NeuralHydrology training → Moriah cluster** (`/sci/labs/efratmorin/omripo/PhD`)

See `docs/stage1_h2o_operations_preflight.md` for full gate status.

### h2o environment status (as of 2026-06-18)

- **Prefix:** `/data42/omrip/Flash-NH/envs/flashnh-stage1`
- **Python:** `3.11.15` | **Size:** `7.0 G`
- **Smoke test:** ALL PASS — core, geospatial, dask, cfgrib/eccodes, NetCDF, Parquet, neuralhydrology
- **Log:** `/data42/omrip/Flash-NH/tmp/env_smoke_20260615T120918Z/env_smoke.log`
- **Activation on h2o:** `source /opt/conda/etc/profile.d/conda.sh && conda activate /data42/omrip/Flash-NH/envs/flashnh-stage1`
- **Activation caveat:** The shell prompt may show `(flashnh-stage1)` while `which python` still
  points to `/opt/conda/envs/iacpy3_2025/bin/python`. Always run the explicit `source` + `conda activate`
  sequence and verify with `which python` before running any job. Observed during 2K-A (2026-06-18);
  clean reactivation resolved it.
- **py7zr added (2026-06-18):** Installed `py7zr` into `flashnh-stage1` using the standard h2o workaround:
  `CONDA_PKGS_DIRS=/home/omrip/.conda/pkgs conda install --solver classic py7zr`.
- **Caveat:** `neuralhydrology` pip-pulled CUDA torch (2.12.0+cu130); env is 7.0 G vs lean CPU intent.
  `cuda_available=False` on h2o — functionally harmless. Future spec revision to use `--no-deps` or CPU torch.
- **h2o is not for NeuralHydrology training.** Training remains designated for Moriah cluster.

See `docs/stage1_environment.md` for full install notes, workaround, and CUDA caveat details.

### Target package builder status (as of 2026-06-16)

Milestone 2J-B: **COMPLETE** — scripts implemented, smoke-tested locally and on h2o, full v001 build PASS.

- **Builder:** `scripts/build_stage1_target_package.py`
- **Auditor:** `scripts/audit_stage1_target_package.py`
- **Launcher:** `scripts/run_stage1_target_package_v001_h2o.sh` (commit `3ac51ff`)
- **Doc:** `docs/stage1_target_package_builder.md`
- **Local smoke result (2026-06-15):** 5/5 PASS — 0 errors, 0 warnings
- **h2o policy smoke (2026-06-15):** PASS — 4 basins, 01135300 excluded (hist_util=False),
  08010000 cleaned 95 neg→NaN; audit 0 errors/0 warnings; 02299472 halt confirmed (EXIT 1)
  - `canonical_merged` confirmed: 2,843 flat NCs, 2,843 unique STAIDs, 0 recursive duplicates
- **Full h2o v001 build (2026-06-16): PASS — 0 errors, 0 warnings**
  - Input: 2,843 NCs from `canonical_merged`
  - Excluded: 2 (`--exclude-staids`) + 89 (policy: `hist_util=False`) = 91 total excluded
  - Built: **2,752 basins**, 0 failed
  - Cleaned: 235 neg→NaN across 16 basins; NaN 3,880,507 → 3,880,742; valid hours 121,940,698
  - Audit: 2,752/2,752 checksums OK; 89 held-out absent; SR basins absent; 1,373 TQR advisory
  - Audit runtime: 18.8 s
  - policy_sha256: `449165686d033b9cdbd395ad70e64a3bfa82d01757021e62059f254a2a30d691`
  - Evidence bundle: `tmp/stage1_target_package_v001_evidence/` (not committed)
  - Full result: `docs/stage1_target_package_v001_result.md`
- **Special-review 02299472/04073468:** excluded from v001; disposition open for future v002

See `docs/stage1_target_package_builder.md` for full commands and acceptance criteria.

### Stage 1 forcing — Milestone 2K-A (completed 2026-06-18)

Input preflight and v001 basin-weight table build on h2o. **PASS — 2,752/2,752 basins.**

**Input preflight (`verify_stage1_forcing_inputs_h2o.sh`):** 10/10 PASS, 0 WARN, 0 FAIL.

**Key input locations on h2o:**

| Item | Path | Notes |
|---|---|---|
| v001 basin list CSV | `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/v001_basin_list.csv` | 2,752 rows excl. header |
| CAMELSH shapefile | `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/02_basin_geometries/camelsh/shapefiles/CAMELSH_shapefile.shp` | 2,752 polygons; no `.prj`, EPSG:4326 |
| MRMS grid def | `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/grid_definitions/mrms_grid_definition.json` | v001 flat layout (not pilot path) |
| RTMA grid def | `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/grid_definitions/rtma_grid_definition.json` | same |

**Weight Parquets (output):**

| File | Size | Basins |
|---|---|---|
| `02_basin_geometries/weights/mrms/v001_2752_mrms_weights.parquet` | 37 MB | 2,752/2,752 |
| `02_basin_geometries/weights/rtma/v001_2752_rtma_weights.parquet` | 12 MB | 2,752/2,752 |

All paths relative to `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/`.

**Clean build command:**

```bash
python scripts/build_stage1_basin_weights.py \
    --config configs/stage1_forcing_fullperiod.yaml \
    --data-root /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod \
    --basin-list /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/v001_basin_list.csv \
    --out-tag v001_2752 \
    --grid-def-dir /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/grid_definitions \
    --skip-qc-plots
```

Fatal validation: all PASS. `--skip-qc-plots` used because the h2o CAMELSH shapefile lacks
`LNG_GAGE`, `LAT_GAGE`, `DRAIN_SQKM` columns (schema: `LAYER, MAP_NAME, AREA, PERIMETER, GAGE_ID, geometry`).
QC plotting is advisory; the fix is in commit `026c363`.

**Operational lessons from 2K-A:**

- **Activation caveat:** Shell prompt can show `(flashnh-stage1)` while `which python` points to
  the wrong env. Always verify with `which python` after activation.
- **py7zr:** Added to `flashnh-stage1` on h2o using `CONDA_PKGS_DIRS` + `--solver classic` workaround.
- **PS1 helper broken:** `scripts/prepare_stage1_forcing_inputs_h2o.ps1` fails to parse on
  Windows PowerShell 5.1 (8 AST parse errors). It is not needed for 2K-B (grid JSONs and
  CAMELSH shapefile are already in place). Fix in a separate commit before relying on it again.
- **Stale verifier message:** `verify_stage1_forcing_inputs_h2o.sh` still prints
  "Ready to proceed to Milestone 2K-A" even after weights are built. Minor stale message;
  not a blocker. Clean up in a later small commit.
- **Grid-def path:** `build_stage1_basin_weights.py` now supports `--grid-def-dir` with 3-level
  auto-discovery (explicit → v001 flat → pilot legacy). Pass it explicitly to avoid ambiguity.

### Stage 1 forcing — Milestone 2K-B (completed 2026-06-18)

Forcing extraction smoke test on h2o. **PASS — all 12 validation checks passed.**

**Evidence:** Compact evidence bundle inspected locally from `tmp/stage1_forcing_smoke_evidence/`
(not committed). Evidence files: `smoke_manifest.json`, `smoke_summary.md`,
`smoke_live_run.log`, `smoke_hourly_runtime_and_volume.csv`, `smoke_missing_files.csv`.

**Original smoke was run via direct extractor invocation.** The launcher (`scripts/run_stage1_forcing_smoke_h2o.sh`)
raised `CondaError: Run 'conda init' before 'conda activate'` when invoked as `bash script.sh`
in a non-interactive shell, even after the PATH-prepend patch in `43af035`. The launcher
activation block was subsequently patched (commit `ccb2631`) to source `conda.sh` unconditionally
and make `conda activate` non-fatal.

**Launcher activation fix verified on h2o (2026-06-18):** After pulling `ccb2631`,
`bash scripts/run_stage1_forcing_smoke_h2o.sh` completed end-to-end via the launcher
wrapper. Python resolved correctly to `/data42/omrip/Flash-NH/envs/flashnh-stage1/bin/python (Python 3.11.15)`.
This was a cached/resume rerun (0.0 B downloaded, ~1m 12s elapsed); output row counts and
PASS status matched the original uncached run. **The launcher activation bug is resolved.**
Download and runtime estimates for 2K-C should be taken from the original uncached run (10m 13s, ~3.2 GB), not this verification rerun.

**Smoke results:**

| Metric | Value |
|---|---|
| Period | 2020-10-14T00:00:00Z – 2020-10-15T23:00:00Z |
| Basins | 10 |
| MRMS hours extracted | 27/48 |
| MRMS missing | 21 (`not_in_s3`, 2020-10-14T00Z–20Z — see note below) |
| RTMA hours extracted | 48/48 |
| RTMA missing | 0 |
| `mrms_smoke.parquet` rows | 270 (27 h × 10 basins) |
| `rtma_smoke.parquet` rows | 5,280 (48 h × 10 basins × 11 vars) |
| `combined_smoke.parquet` rows | 5,550 |
| Wall clock | 10m 13s |
| Downloaded | ~3.2 GB (RTMA `selected_messages`, 4 workers) |
| `all_pass` (manifest) | `true` |
| Git commit at run time | `43af035d` |

**MRMS 21-hour early archive gap (expected):** `noaa-mrms-pds` QPE 1h Pass1 coverage for
2020-10-14 begins at 21:00Z, not midnight. The first 21 hours (00Z–20Z) are genuinely
absent from S3 — this is a permanent upstream archive gap, not a pipeline error.
The full-period first chunk (`2020-10`) will carry the same 21-hour gap in its
`missing_files.csv`. All subsequent months have complete MRMS coverage.

**Validation checks (all PASS):**
`mrms_extracted_hours_gt_zero` · `mrms_N_basins_per_ok_hour` · `mrms_no_all_null_weighted_mean`
· `mrms_valid_weight_fraction_ok` · `mrms_parquet_written` · `rtma_extracted_hours_gt_zero`
· `rtma_10wdir_absent` · `rtma_orog_absent` · `rtma_at_least_8_variables`
· `rtma_no_all_null_weighted_mean` · `rtma_parquet_written` · `combined_parquet_written`

**Performance notes:**
- RTMA `selected_messages` download: median ~42 s/file at 4 workers → ~33–40 h total at 16 workers.
- MRMS download: ~0.3–1.3 s/file (cfgrib cold start on first file only). Negligible vs RTMA.
- Estimated full-period RTMA raw: ~3.2 TB (`selected_messages`); MRMS raw: ~0.5 TB.

### Stage 1 forcing — Milestone 2K-C (completed 2026-06-18)

October 2020 one-month forcing extraction on h2o. **PASS — all 12 extractor validation checks passed.**

**Evidence:** Compact bundle in `tmp/stage1_evidence_exports/2020-10/` (not committed).

| Metric | Value |
|---|---|
| Period | 2020-10-14T00Z – 2020-10-31T23Z |
| Scheduled hours | 432 |
| Basins | 2,752 |
| MRMS extracted | 396/432 |
| MRMS not_in_s3 | 36 (3 clusters — see below) |
| RTMA extracted | 432/432 |
| RTMA variables | 11 (incl. diagnostic `ceil`, `vis`; `10wdir`/`orog` absent, confirmed) |
| Combined rows | 14,167,296 (1,089,792 MRMS + 13,077,504 RTMA) |
| MRMS raw | 207 MB |
| RTMA raw | 30.7 GB |
| Wall clock | 15h 04m 57s (`download_workers=8`) |
| `all_pass` | `true` |
| Git commit at run | `194a489` |

**MRMS 36-hour gap (permanent S3 gaps — not pipeline errors):**

| Cluster | Hours | Timestamps |
|---|---|---|
| Archive-start | 21 h | 2020-10-14T00Z–20Z |
| Oct 25–26 outage | 14 h | 2020-10-25T23Z; 2020-10-26T00Z–11Z, 15Z |
| Oct 29 spot | 1 h | 2020-10-29T23Z |

**Throughput and full-period projection:**

- Actual throughput: 125.7 s/hr (serial, extraction-dominated)
- Full-period projection at current serial code: **66.5 days** (45,720 h × 125.7 s / 86400)
- Primary bottleneck: `extract_basin_statistics` in `src/pipeline/extraction.py:396`
  — `weights_df.loc[weights_df["STAID"] == staid]` O(N) scan, 30,272 calls per RTMA hour
- The 20.2-day figure from `scaling_estimates.json` was computed from RTMA download time only
  (download is pipelined/prefetched and is NOT on the serial critical path)

**Full-period extraction was PAUSED at 2K-C completion.** 2K-D is now COMPLETE — see section below.

### Stage 1 forcing — Milestone 2K-D (completed 2026-06-20)

Extraction optimization and outer-parallelism throughput benchmark.
**COMPLETE — effective full-period projection 3.04 days (3 concurrent chunks × 6 download workers).**

#### D1: Serial extraction optimization (commit `3ff4965`)

Two targeted changes to `src/pipeline/extraction.py` and `scripts/extract_stage1_forcing_chunk.py`:

1. **Pre-grouped weight lookup** — `_build_basin_cells()` pre-groups the weight DataFrame by
   STAID into a `{STAID: (row_idx, col_idx, norm_w)}` dict at startup. Each per-basin-hour call
   becomes an O(1) dict lookup instead of an O(N) boolean scan over the 2,752-row weight table
   (90,816 scans/RTMA-hour eliminated).
2. **Batched percentile computation** — 7 sequential `np.percentile` calls replaced with one
   batched call, eliminating 635,712 redundant sort passes per RTMA-hour.

**Measured result:** `extraction_median_s` 91.976 s → 2.17 s/hr (**24.7× speedup**).
Bottleneck fully shifted from extraction CPU to S3 download. D2 process-workers not needed.

#### Download-worker sensitivity benchmark (48h RTMA-only, 2,752 basins)

Commit `3ff4965`; RTMA `selected_messages`; Oct 2020 period; all runs `all_pass=True`.

| Workers | Wall (s) | Proj. days | dl_median (s) | ext_median (s) |
|---|---|---|---|---|
| 2  | 1157.7 | 12.76 | 31.3 | 2.21 |
| 4  | 804.8  | 8.87  | 31.3 | 2.19 |
| 8  | 642.9  | 7.09  | 35.9 | 2.18 |
| 16 | 570.5  | **6.29** | 44.9 | 2.17 |

Individual download time increases with worker count (S3 bandwidth sharing) but wall-clock improves
via prefetch concurrency. dw16 projects 6.29 days. D2 process-workers deferred; outer parallelism
is the lever for sub-4-day throughput.

#### Outer-parallelism benchmarks (RTMA-only, 48h per chunk, 2,752 basins)

All chunks `all_pass=True`, `successful_hours=48/48`, `actual_rows=1,453,056`.

**x2 — 2 chunks × dw8 (16 total S3 connections):**
Commits `cf8db74`; evidence `tmp/stage1_2kd_evidence/outer_parallel_rtma_48h_dw8_x2/`.

| Chunk | Chunk wall (s) | dl_median (s) | ext_median (s) |
|---|---|---|---|
| outer-x2-a | 735.4 | 47.2 | 2.195 |
| outer-x2-b | 720.0 | 43.1 | 2.291 |
| **Parent wall** | **736 s** | | |

Projection: 45720 × 736 / (2 × 48) / 86400 = **4.057 days — YELLOW (partial scaling).**

**x3 — 3 chunks × dw6 (18 total S3 connections):**
Commit `a275296`; evidence `tmp/stage1_2kd_evidence/outer_parallel_rtma_48h_dw6_x3/`.

| Chunk | Chunk wall (s) | dl_median (s) | ext_median (s) |
|---|---|---|---|
| outer-x3-a | 825.9 | 45.9 | 2.233 |
| outer-x3-b | 801.1 | 43.9 | 2.206 |
| outer-x3-c | 801.2 | 42.5 | 2.204 |
| **Parent wall** | **826 s** | | |

Projection: 45720 × 826 / (3 × 48) / 86400 = **3.035 days — USEFUL GREEN.**

#### Decisions (all binding)

- **Stop performance optimization.** 3.04 days projected is within the acceptable range.
- **D2 process-workers: deferred indefinitely.** Extraction is 2.17 s/hr; download (43–46 s/file)
  dominates. Process parallelism within a single chunk would not improve end-to-end throughput.
- **x4 outer-parallelism: not recommended.** x3 achieves 3.04 days; x4 would push total S3
  concurrency to 24 workers, increasing contention and operational risk for marginal gain.
  RTMA-only benchmark may understate MRMS+RTMA mixed-product overhead.
- **Full-period launch recommendation:** 3 concurrent chunk processes × 6 download workers each.
  All outputs under `/data42/omrip/Flash-NH/`. Mechanism: 3 independent screen sessions covering
  non-overlapping month groups (~21 months each), or a new parallel launcher.
  See updated `docs/stage1_forcing_fullperiod_launch_plan.md` for Phase 2 outer-parallel details.

### Immediate next steps

The v001 target package is **streamflow-only**. Full NeuralHydrology training requires
forcing data and package assembly on h2o before any Moriah transfer.

1. ~~**Push 2K-E pre-launch patch and pull on h2o**~~ — **COMPLETE (2026-06-20).**
2. ~~**Stage 1 forcing acquisition plan + weight build (2K-A)**~~ — **COMPLETE (2026-06-18).**
3. ~~**Milestone 2K-B — forcing extraction smoke test**~~ — **COMPLETE (2026-06-18): PASS.**
4. ~~**Milestone 2K-C — October 2020 one-month run**~~ — **COMPLETE (2026-06-18): PASS.**
4b. ~~**Milestone 2K-D — extraction optimization + h2o CPU-parallel benchmark**~~ — **COMPLETE (2026-06-20): PASS.**
4c. ~~**Milestone 2K-E — full-period forcing extraction**~~ — **COMPLETE and AUDITED (2026-06-24): PASS_WITH_CAVEATS.**
    63/63 months, 1.51B rows, 0 failures. See `docs/stage1_forcing_fullperiod_audit.md`.
5. ~~**Visual / event QC case selection + pilot animation + spatial MRMS QC**~~ — **PILOT VISUAL QC PASS (2026-06-25/28).**
   21 cases generated (seed=42). Basin-timeseries pilot 6/6 OK. Spatial MRMS smoke VQC-009/VQC-012 PASS (basin=Y, gauge=Y).
   Case selection: `docs/stage1_forcing_fullperiod_visual_qc_selection.md`.
   Animation plan and evidence: `docs/stage1_forcing_fullperiod_visual_qc_animation_plan.md`.
   Outputs under `tmp/` (not committed). 15 remaining cases not yet animated — not a final certification.
6. ~~**Curated forcing product v001 design (Milestone 2K-F-A)**~~ — **COMPLETE (2026-06-29).**
   Product contract frozen: wide-format per-basin Parquet, gap-flag columns, manifest, provenance.
   Design doc: `docs/stage1_curated_forcing_product_v001_design.md`.
7. ~~**Curated forcing product v001 — builder + smoke test (Milestone 2K-F-B)**~~ — **COMPLETE (2026-06-29): PASS.**
   5/5 basins, 720 h, 0 MRMS gaps, 10 RTMA gap-hours (coverage 0.9972). Scripts: commit `6f4de49`.
   h2o output: `/data42/omrip/Flash-NH/tmp/stage1_curated_forcing_smoke_20260629T132757Z/`.
8. **Curated forcing product v001 — corrected schema build (Milestone 2K-F-C)** — schema
   corrected in 2K-F-C-B (2026-06-30): dewpoint mapping fixed, `rtma_weasd_kgm2` removed.
   Next: corrected 5-basin full-period pilot on h2o (`--max-basins 5 --overwrite`), then
   full 2,752-basin rebuild authorization. Full rebuild NOT yet authorized.
9. **Milestone 2K-G-B — NH pilot package builder** — implement `scripts/build_stage1_nh_package.py`
   on h2o: merge corrected forcing Parquets + target NCs into 5-basin GenericDataset NCs,
   apply gap-fill policy (MRMS→0.0, RTMA→interp), write `attributes.csv` and basin lists.
   Transfer pilot package (~25 MB) to Moriah.
9a. **Milestone 2K-G-C — Moriah environment + Smoke 0** — install `flashnh-moriah` conda env
    (PyTorch+CUDA, NH), run Smoke 0 Slurm job (5 basins, 2 epochs, mrms_qpe_1h_mm only),
    confirm finite loss and checkpoint.
9b. **Milestone 2K-G-D — Smoke 1** — add RTMA meteorology, confirm `rtma_2d_K` non-null.
    Preflight design: `docs/stage1_neuralhydrology_preflight.md`.
10. **Moriah transfer layout and checksum-verified transfer** — define directory structure
    and `rsync`/`scp` transfer procedure; verify checksums on arrival before training.
11. **Moriah training environment and config** — only after the assembled package passes
    audit on Moriah. NeuralHydrology training remains designated for Moriah cluster.

#### 2K-C pre-launch checklist and caution

Before any 2K-C run, confirm all of the following:

**Launcher verification (new requirement):**
- Pull latest commits on h2o: `git pull --ff-only`
- Run a dry activation test: `bash scripts/run_stage1_forcing_smoke_h2o.sh --help` or check that
  the launcher reaches the Python version line without error.
- The launcher activation bug (CondaError in non-interactive shells) is patched in the current commit.
  **Verify the fix is working on h2o before launching 2K-C.**

**One-month dry run before full 63-month launch:**
- Run 2020-10 alone first (`screen -S flashnh-2020-10 bash scripts/run_stage1_forcing_fullperiod_h2o.sh`
  with the month list reduced to a single entry, or via direct extractor for 2020-10-14T21Z – 2020-10-31T23Z).
- Confirm the 2020-10 chunk manifest is written, `missing_files.csv` contains exactly 21 MRMS
  `not_in_s3` entries for 2020-10-14T00Z–20Z, and Parquet row counts are consistent.
- Pull the 2020-10 evidence bundle locally before enabling the full loop.

**Expected 2020-10 MRMS 21-hour gap:**
- 2020-10-14T00Z–20Z will appear as `not_in_s3` in `missing_files.csv` for the first chunk.
- This is a documented upstream archive gap, not a pipeline error. Do not treat as a blocker.
- All hours from 2020-10-14T21Z onward and all subsequent months have complete MRMS coverage.

**PI notification:**
- Notify PI/machine owner before starting the full 63-month extraction loop.
- Check `uptime` before launch; hold if 1-min load > 0.7 × nproc.
- Target ≤ 50–60% CPU; start with 16 workers; increase only after monitoring a full chunk.

**Storage and raw GRIB2 deletion policy:**
- Raw MRMS + RTMA GRIB2 cache accumulates to ~3.7 TB over the full period.
- After each quarter's monthly chunk Parquets are written and checksummed, delete the
  corresponding raw GRIB2 cache to free space. Do not delete until Parquets are verified.
- Monthly chunk Parquets + per-basin forcing NCs are the curated products; raw GRIB2 is reproducible.
- Do not exceed ~20 TB total across all Flash-NH data on `/data42`.

**Evidence-bundle pull policy:**
- After every quarter (roughly every 3 months of chunks), transfer compact evidence bundles
  locally: chunk manifests (`*_manifest.json`) and missing-file CSVs (`*_missing_files.csv`).
- Do not transfer raw GRIB2, staging Parquets, or combined chunk Parquets unless needed for debugging.
- Document each quarterly bundle in `docs/FLASHNH_CURRENT_STATE.md` before proceeding.

**Progress monitoring:**
- Attach to the screen session with `screen -r flashnh-fullperiod` to check live log output.
- Each monthly chunk writes a progress log to `{FORCING_ROOT}/manifests/{chunk_label}_live_run.log`.
- Check `uptime` and `df -h /data42` periodically (once per few hours).
- A per-month completion summary will be logged; each month's manifest is the checkpoint.

**Stop and resume procedure:**
- To stop cleanly: `Ctrl-C` inside the screen session; the current hour's staging Parquet may be incomplete.
- To resume: re-run the launcher with `--resume`; already-written staging Parquets for completed hours
  are skipped automatically.
- Each completed monthly chunk is independent; re-running a month re-uses cached raw files and
  skips already-extracted hours.

**Special-review disposition (02299472/04073468)** — open for future v002, not a blocker
for steps 3–8 above. 02299472: 2,605 neg; 04073468: 2,054 neg.

The following require additional confirmation before proceeding:

- Promotion of curated data to shared lab storage — gate G4 CONDITIONALLY UNBLOCKED
  (confirm write access to `/data42/hydrolab/Data/Flash-NH_data/` before first promotion).
- NeuralHydrology training — gate G3 NOT PLANNED ON h2o; blocked on Moriah scheduler
  confirmation and env setup.

**Do not run TB-scale spatial downloads without smoke-test sign-off under etiquette rules.**

---

## Milestone 2G — NeuralHydrology NetCDF builder + preflight auditor (completed 2026-06-09)

NeuralHydrology-compatible January 2023 pilot package built and audited.
Full documentation: `docs/stage1_neuralhydrology_preflight.md`

**Scripts:**
- `scripts/build_stage1_neuralhydrology_january_pilot.py` — builder (~8s)
- `scripts/audit_stage1_neuralhydrology_january_pilot.py` — auditor (~20s)

**Package:** `tmp/stage1_pilot_dryrun/12_neuralhydrology_january_pilot_dataset/package/` (gitignored)

**Audit result:** PASS — 0 errors, 1 warning

**Package summary:**
- 50 per-basin NetCDF files; `date` coordinate; 744 hourly UTC steps; January 2023
- 11 variables per basin (10 dynamic forcings + `qobs_m3s` target)
- Smoke dynamic inputs: `mrms_qpe_1h_mm`, `rtma_2t_K`, `rtma_2d_K`, `rtma_2sh_kgkg`, `rtma_10u_ms`, `rtma_10v_ms`
- `attributes_full.csv`: 50 rows × 238 cols (237 attribute cols + `gauge_id`)
  - Manifest records 237, counting only attribute cols; both are correct
- Full HydroATLAS integration: 50/50 pilot match; 193 new columns
- Streamflow: 20 full, 8 partial, 22 all-NaN (CAMELSH files missing locally)
- Audit warning (expected, S2): nulls in `max_abs_hourly_jump_over_Q50` (1), `q95_q50_ratio` (1), `wet_cl_smj` (14) — NaN preserved, no imputation

**No model training run. No generated files committed.**

---

## Milestone 2F — NeuralHydrology package design (completed 2026-06-08)

Design and decision documentation for the NeuralHydrology package.
Full documentation: `tmp/stage1_pilot_dryrun/12_neuralhydrology_january_pilot_dataset/design/`

Key decisions: V1 (both rtma_2d_K and rtma_2sh_kgkg in smoke), V2 (rtma_sp_Pa in wide only),
V3 (rtma_tcc_pct in wide only), S1 (22 missing CAMELSH → all-NaN qobs_m3s, 2H blocker),
S2 (preserve NaN, no imputation), S3 (full HydroATLAS 50/50), S4 (seed=42, streamflow-only split).

---

## Milestone 2E — Event animation pipeline (completed 2026-06-07)

Pilot animations (R02, R06, R09, R11) generated and approved in v2.1-stable design.
Pipeline cleanup completed.

**Stable scripts:**
- `scripts/generate_january_event_animations.py` — main animation generator
- `scripts/audit_rtma_spatial_alignment.py` — RTMA spatial audit gate
- `scripts/audit_january_event_animation_sync.py` — MRMS sync audit gate

**Audits confirmed:**
- RTMA spatial audit: 8/8 PASS, 0.0000% diff (2t, 10u, 10v)
- MRMS sync audit: 10/10 PASS, 0.0000% diff

**Key v2.1 design notes:**
- MRMS lat DECREASES with row (row 0 = 54.995 N)
- RTMA lat INCREASES with row (row 0 = 19.229 N)
- RTMA 10m wind quiver is qualitative context only — not storm-steering validation

**All-12 command** (not yet executed; run after explicit approval):
```bash
python scripts/audit_rtma_spatial_alignment.py
python scripts/audit_january_event_animation_sync.py
python scripts/generate_january_event_animations.py --all
```
Output: `tmp/stage1_pilot_dryrun/10_animations/stage1_pilot/pilot/`
Estimated runtime: ~27 min (local, GIF mode).

---

## RTMA/URMA-family precipitation diagnostic (completed 2026-06-08)

Diagnostic-only follow-up to Milestone 2E. Confirmed RTMA/URMA grid, weight,
and timestamp consistency against MRMS. **Did not modify Stage 1 model inputs.**

Full documentation: `docs/stage1_rtma_urma_mrms_diagnostic.md`

**Key findings:**

- Regular RTMA Stage 1 files have no precipitation field.
- URMA QPE `pcp_01h.wexp.grb2` contains `tp` (Total Precipitation, kg m**-2 = mm).
- URMA and RTMA share the same 1597 x 2345 LCC 2.5 km CONUS grid exactly.
- Existing `pilot_rtma_weights.parquet` reused without modification. No new weights.
- Timestamp convention A confirmed (filename HH = end of accumulation):
  r = 0.961 on R02; shifted alternatives much worse; peak at Jan 29 08Z for both.

**Pilot metrics (Convention A):**

| Candidate | r | RMSE (mm) | Note |
|---|---|---|---|
| R02 (AR, STRONG_WET) | 0.963 | 1.12 | URMA smooths peak vs MRMS |
| R06 (MN, MOD_COLD) | 0.913 | 0.70 | URMA higher; snow/mixed-precip context |
| R11 (MA, OFFSET) | 0.944 | 0.39 | Strong agreement |

**Scripts (committed):**
- `scripts/discover_rtma_urma_precip_january2023.py`
- `scripts/urma_mrms_timestamp_and_pilot.py`

**Diagnostic outputs (untracked):**
`tmp/stage1_pilot_dryrun/11_rtma_urma_mrms_diagnostics/`

---

## Completed extraction state

January 2023 pilot extraction for 50 basins:
- MRMS: 744/744 hours, 37,200 rows
- RTMA: 744/744 hours, 409,200 rows
- Combined: 446,400 rows
- valid_weight_fraction = 1.0

Streamflow: CAMELSH hourly NetCDF, 28/50 pilot basins have January 2023 data.

Refined event candidates: R01–R12 (R03 usable-with-gap).
Pilot animations: R02, R06, R09, R11 — reviewed and approved.

---

## Standing cautions

- Do not generate all 12 animations until explicitly instructed.
- Do not start model training yet.
- Do not commit generated MP4/GIF/PNG/Parquet/GRIB/NetCDF/log outputs.
- Keep local-to-HPC transition in mind.
- RTMA 10m wind vectors are qualitative context only — not storm-steering validation.
- URMA precipitation is diagnostic-only — do not add to Stage 1 model inputs.

---

## Historical note: Milestone 2H — Streamflow recovery for 22 missing CAMELSH basins

> This section is superseded for full-period target-package construction, which is now
> complete (v001, 2026-06-16). The recovery work below applied to the January 2023 pilot
> package (Milestone 2G) and is retained for reference. The current top-level next step
> is Moriah transfer layout design (see Immediate next steps above).

Recovery was needed because the January 2023 pilot package built from CAMELSH files
had 22 basins with all-NaN `qobs_m3s`. Those basins were recovered from USGS IV
(Milestones 2H–2H-D) and are fully represented in the full-period v001 package.

Recovery plan (historical): `tmp/stage1_pilot_dryrun/12_neuralhydrology_january_pilot_dataset/design/streamflow_recovery_plan.md`

**Original pending tasks (now completed or superseded):**

1. Milestone 2H: CAMELSH streamflow recovery for 22 missing basins.
2. Decide on all-12 animation run (2E follow-up).
3. Event QC conclusions: finalize which of R01–R12 are included in Stage 1 training.
4. HPC transfer planning.
5. Stage 1 model configuration and first training run.

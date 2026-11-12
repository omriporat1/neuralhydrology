# WY2026 data plan — download and preprocessing to basin-average hourly series

Status: **PLAN, decided, not launched.** Written 2026-11-12. Nothing in this plan has been run on h2o.

Omri's decisions (2026-11-12, in conversation; the `docs/decision_log.md` entry is still to be drafted and added by the session that owns that file):
- **(a)** refetch the whole of WY2026 from USGS as one vintage.
- **(b)** structural and availability QC only, as proposed in §8b; counts of USGS qualifier codes are allowed.
- **(c)** h2o may be used now and over the weekend (other users consulted). Launch once commit, push, pull and review are done.

Scope: water year 2026 (2025-10-01T00Z .. 2026-09-30T23Z, 8,760 h). This is the protected test year under the rotation adopted 2026-11-11 (see `docs/decision_log.md`). The plan stops at basin-average hourly series:

- MRMS QPE and RTMA basin-average forcing;
- USGS hourly discharge NetCDFs.

Building the NH package and changing the split/policy belong to the rotation-split work, not to this plan.

The plan reproduces the scheme that produced the 2020-10-14..2025-12-31 data. It keeps the same products, the same v001_2752 area weights, the same extractor and fetch code, and the same audit and export pattern. It adds only new arguments and wrapper scripts. No pinned default changes, so the v001 products stay reproducible.

## 0. Protected-year rules (apply to every step)

- Do not open, summarise or plot any WY2026 discharge values. The QC is structural and availability only (see §8b).
- WY2026 availability must not change:
  - the 2,307-basin development population;
  - the 250-basin holdout list;
  - any admission or exclusion list.

  No WY2026 audit output feeds a `target_status`, exclusion or admission file.
- WY2026 target outputs go in their own new directory, which no existing package builder or loader reads (§5).
- Forcing (MRMS/RTMA) is not a target. It is still extracted only for the fixed v001 basin list, and its QC is limited to the extractor's existing structural checks.

## 1. What already exists for WY2026

| Product | Oct–Dec 2025 | Jan–Sep 2026 |
|---|---|---|
| MRMS Pass1 01H forcing chunks | **done**: months 2025-10/11/12, accepted at 7e43760; part of curated v001 | not downloaded |
| RTMA 2dvaranl_ndfd forcing chunks | **done**: same | not downloaded |
| USGS IV hourly discharge | in the 45,720-h per-basin NetCDFs of the June 2026 fetch (`stage1_full_2843`, vintage 2026-06-13) | not downloaded |

Upstream availability (object listing only, no values; checked 2026-11-12; evidence in `.scratch_local/wy2026_data_plan_20261112/upstream_listing_wy2026.csv`):

| Months | MRMS Pass1 01H | RTMA 2dvaranl_ndfd |
|---|---|---|
| 2025-10..12 | 2,208/2,208 h | 2,208/2,208 h |
| 2026-01..09 | 6,552/6,552 h (3.7 GB) | 6,551/6,552 h. Missing: **2026-09-23 19Z** (expected to appear as 1 RTMA gap hour, NaN + gap flag, as in 2020-11-12 09/10Z) |

## 2. Step-by-step mapping: old chain → WY2026

### 2a. Forcing (MRMS + RTMA)

| # | Old step (2020-10-14..2025-12-31) | WY2026 equivalent | Change |
|---|---|---|---|
| F1 | `scripts/prepare_stage1_forcing_inputs_h2o.ps1`: grid definitions and shapefiles to h2o | **reuse**: already in `stage1_forcing_fullperiod/` | none |
| F2 | `verify_stage1_forcing_inputs_h2o.sh` (10/10 PASS) | **reuse** (the inputs have not changed) | none |
| F3 | `scripts/export_v001_basin_list.py` → `v001_basin_list.csv` (2,752) | **reuse the same file**. The forcing basin list stays v001, so WY2026 availability cannot change it | none |
| F4 | `scripts/build_stage1_basin_weights.py --out-tag v001_2752` | **reuse** `v001_2752_{mrms,rtma}_weights.parquet` unchanged | none |
| F5 | Smoke run `run_stage1_forcing_smoke_h2o.sh` (2020-10-14..15, 10 basins) | skipped: same code and same grids. Optional: `MAX_MONTHS=1` on one group is the real first month | none |
| F6 | One-month benchmark `run_stage1_forcing_onemonth_h2o.sh` | not needed: per-month wall time is known (§6) | — |
| F7 | `run_stage1_forcing_fullperiod_h2o.sh`: 63 months, 3 screen groups A/B/C, `DOWNLOAD_WORKERS=6`, `selected_messages`, `--resume --no-plots` | **new wrapper `scripts/run_wy2026_forcing_extraction_h2o.sh`**: identical per-month extractor call, month list 2026-01..2026-09, groups D/E/F (3 months each). Refuses any month before 2026-01 | new wrapper only |
| F8 | Extractor `scripts/extract_stage1_forcing_chunk.py` (12 validation checks; exit 0 only if `all_pass`) | **same file, unchanged** (identical at the h2o checkout 98b7d42 and at master) | none |
| F9 | `report_stage1_forcing_progress_h2o.sh` (TOTAL_MONTHS=63) | monitor with the wrapper's `wy2026_group_*_run_log.txt` plus `manifests/2026-MM_live_progress.json`. The old reporter's month count is cosmetic | none |
| F10 | `generate_fullperiod_audit_tables.py` on the pulled manifests (PASS_WITH_CAVEATS) | same script on the 9 pulled 2026 manifests plus the 3 accepted 2025-10..12 manifests (evidence-root = a new local folder) | none expected; check at run time |
| F11 | Curated build `run_stage1_curated_forcing_fullperiod_h2o.sh` → `build_stage1_curated_forcing_fullperiod.py` → `stage1_basin_hourly_forcings_v001` | **new product `stage1_basin_hourly_forcings_wy2026_v001`** (2025-10-01..2026-09-30, 8,760 h) from the accepted 2025-10..12 chunks plus the new 2026 chunks | **additive CLI needed** (§3, item C1) |
| F12 | Curated audit `audit_stage1_curated_forcing_basin_parquets.py --full-period` | same auditor with WY2026 period and expected-gap arguments | **additive CLI needed** (C2) |
| F13 | Evidence tarball (manifests, summaries, `missing_files.csv`) | same pattern, under `.scratch_local/wy2026_forcing_evidence_<date>/` locally | none |
| F14 | `build_stage1_baseline_nh_package.py --forcing-root …` (v002) | **not part of this plan**: rotation-split session | — |

Consistency check, available at no extra cost: the WY2026 curated product's Oct–Dec 2025 rows must be bit-identical to the same rows of curated v001. Both are built from the same accepted chunks.

### 2b. USGS discharge

| # | Old step | WY2026 equivalent | Change |
|---|---|---|---|
| U1 | Basin manifest `config/stage1_initial_training_basin_manifest.csv` (2,843 STAIDs) | **same manifest**. It is the fixed fetch list, not an admission list | none |
| U2 | `scripts/launch_usgs_iv_recovery_shards.py --out-root …/stage1_full_2843 --n-shards 4` (screen session `flashnh_2843`; 2026-06-13, 10,780 s) | same launcher, **new out-root** `/data42/omrip/Flash-NH/tmp/stage1_usgs_iv_wy2026_v001`, 4 shards, plus `--wy-chunks derived` | **additive flag, implemented** (U-change below) |
| U3 | `scripts/recover_usgs_iv_full_period_hourly.py` (NWIS IV 00060; ±15-min nearest snap to UTC hours; ft³/s → m³/s; raw cache per WY chunk; `write_canonical_nc` refuses overwrite) | **same script**, `--start/--end` for WY2026 plus `--wy-chunks derived` | **additive flag, implemented** |
| U4 | `canonical_merged/`: manual flat copy of `shard_*/canonical/*.nc` (no recorded command) | same flat copy, recorded as a command (§7) | none |
| U5 | `scripts/audit_usgs_iv_recovered_targets.py` (coverage, gaps, late-gap rule, validate; **plus** value statistics, flatline/spike, QC PNGs, Jan-2023 comparison) | **structural-only subset** (§8b), period WY2026 | **additive CLI needed** (C3) |
| U6 | `scripts/run_stage1_target_package_v001_h2o.sh` → `build_stage1_target_package.py` (drops OPERATIONAL_REVIEW/utility-false basins; negatives → NaN) → `stage1_target_package_v001` | **not run as is.** Its status-driven basin drop would let WY2026 availability remove basins. If a cleaned WY2026 target set is wanted, run the builder with explicit new paths and the drop disabled. Needs a decision (§8b) | deferred |
| U7 | Lead-6 shift and area normalisation (package build) | **not part of this plan**: rotation-split session | — |

**U-change, implemented here (uncommitted):** `recover_usgs_iv_full_period_hourly.py` hard-codes `WY_CHUNKS` ending 2025-12-31. Before this change, `--end 2026-09-30` would silently request nothing after 2025-12-31 and write NaN for all of Jan–Sep 2026. The change adds three things:

- `--wy-chunks {legacy,derived}` (default `legacy`, so the old product is unchanged), passed through by the shard launcher.
- A hard error if the requested period runs past the chunk table, replacing the silent NaN.
- `derive_wy_chunks()`, one chunk per water year. The outermost bounds keep the ±15-min snap buffer, so a standalone WY2026 fetch still requests the 2025-09-30T23:45–23:59 readings that snap to 2025-10-01T00Z. The multi-year June run got those readings from its WY2025 chunk.

- After the independent review (2026-11-12, no blockers), two more additions:
  - A derived-mode **window lock** (`<shard out-dir>/wy_chunk_windows.json`). The raw cache is keyed by STAID + label (`WY2026`), not by request window. The lock therefore refuses two things:
    - a derived run whose windows differ from the recorded ones, which would otherwise silently reuse cached rows from a different window;
    - any derived run into an out-dir that already holds NCs or a cache but no lock, so `--force` can never overwrite the legacy `stage1_full_2843` products.

    A dry run checks the lock but does not write it; a resume with the same window passes.
  - Provenance: the `wy_chunk_mode` NC attribute and `wy_chunks` in the acquisition manifest, written only in derived mode, so legacy outputs stay byte-identical.

Tests: `tests/test_usgs_iv_wy_chunks.py`, 13 passed and network-free. Inside the legacy period the derived and legacy windows are identical except for that head buffer.

## 3. Additive code changes still needed (not yet written)

They are needed only **after** the downloads finish, so they do not block the launch.

- **C1. Curated builder** (`build_stage1_curated_forcing_fullperiod.py`). Add `--period-start/--period-end` (default 2020-10-14/2025-12-31), `--expected-hours`, `--expected-mrms-gaps` and `--expected-rtma-gaps`, all with defaults equal to the current pins. The month list should derive from the period instead of `_ALL_MONTHS`. The "full build" detection should derive from the expected month count instead of `== 63`. Also add a launcher wrapper `run_wy2026_curated_forcing_h2o.sh` with `PRODUCT_DIR=…/stage1_basin_hourly_forcings_wy2026_v001`. Never use `--overwrite` on v001.
- **C2. Curated auditor** (`audit_stage1_curated_forcing_basin_parquets.py`). Add the same period arguments plus `--known-rtma-gap-timestamps`; for WY2026 the expected value is 2026-09-23T19Z. Defaults keep v001 behaviour.
- **C3. USGS auditor** (`audit_usgs_iv_recovered_targets.py`). Add period arguments (WY boundaries derived from the period) and `--structural-only`. That mode skips `compute_quality_audit` (negatives/zeros/flatline/spike/min/max/p50), all PNG plots and the Jan-2023 comparison. It writes only: coverage per basin, per month and per WY; gap count; longest gap; last valid timestamp; NaN count; snap counts; file/attribute validation; counts of USGS qualifier codes (P/A/ice/equipment) per basin, approved 2026-11-12, counts only. The fixed summary period string must come from the arguments. In structural-only mode, no output may be named or shaped like `target_status.csv`.

Each change gets focused synthetic tests. Under `docs/agent_handoff_rules.md` §3.2 (risky integration / sealed-scope protection), C3 gets an independent review before it runs on WY2026.

## 4. Date pins that change, and how

| Pin | Location | WY2026 handling |
|---|---|---|
| 63-month `MONTH_LIST`, groups A/B/C | `run_stage1_forcing_fullperiod_h2o.sh` L183-261 | untouched. The new wrapper carries its own 9-month list and groups D/E/F |
| One-month defaults `START/END/MONTH_LABEL` | `run_stage1_forcing_onemonth_h2o.sh` L64-83 | not used |
| `TOTAL_MONTHS=63` | progress reporter | untouched (cosmetic) |
| period / 45,720 h | `configs/stage1_forcing_fullperiod.yaml` L70-72 | untouched. The WY2026 period is recorded in the wrapper and manifests |
| `_PERIOD_START/_END`, `_ALL_MONTHS`, `_EXPECTED_*` (45,720 h, 136 MRMS, 2 RTMA), `== 63` | curated builder L83-94, L348-569 | C1: new arguments, defaults unchanged |
| `_PERIOD_*`, `_KNOWN_RTMA_GAP_TIMESTAMPS` | curated auditor L52-70 | C2 |
| `DEFAULT_START/END` | recover script L81-82, launcher L59-60 | overridden with `--start/--end` (existing arguments) |
| **`WY_CHUNKS` (ends 2025-12-31)** | recover script L89-96 | **`--wy-chunks derived` (implemented)** |
| `PERIOD_START/END`, `WY_BOUNDARIES`, summary period string | USGS auditor L56-67, L643 | C3 |
| Fixed paths `canonical_merged`, `audit/target_status.csv`, `stage1_target_package_v001` | `run_stage1_target_package_v001_h2o.sh` | not run (U6) |
| Package / policy pins | `policy.py` L205-207, `build_stage1_nh_package.py` L65-68 | **not touched**: rotation-split session |

## 5. Output locations on h2o

All outputs go under `/data42/omrip/Flash-NH/tmp/`.

| Output | Path | Read by existing loaders? |
|---|---|---|
| Raw GRIB cache, per-hour staging, monthly chunks `2026-01..09`, manifests `2026-MM_*` | `stage1_forcing_fullperiod/{raw,staging,chunks,manifests}`: the same root as the accepted months, with new labels and keys and no collisions | the v001 curated builder pins its 63 months and ignores 2026 chunks |
| Wrapper run logs | `stage1_forcing_fullperiod/manifests/wy2026_group_{d,e,f}_run_log.txt`, `…/logs/wy2026_group_{d,e,f}.log` | no |
| Curated WY2026 forcing | `stage1_forcing_fullperiod/stage1_basin_hourly_forcings_wy2026_v001/` | no |
| USGS WY2026 fetch (shards, raw cache, canonical NCs, manifests, launcher summary) | `stage1_usgs_iv_wy2026_v001/` (new root) | **no**: the v001 driver reads `stage1_full_2843/canonical_merged`; the package builder reads `stage1_target_package_v001/time_series` |
| USGS flat copy | `stage1_usgs_iv_wy2026_v001/canonical_merged/` | no |
| USGS structural audit | `stage1_usgs_iv_wy2026_v001/audit_structural/` | no |

Code runs from the h2o checkout `/data42/omrip/Flash-NH/repos/flash-nh`, currently at `98b7d42` with a clean status. It must be brought to a commit that contains the new wrapper and the `--wy-chunks` flag (§7, step L0) so that the NetCDF/manifest `git_commit` attribute names the code that produced them.

## 6. Wall time and storage (from the earlier run records)

**Forcing.**
- Full-period manifests show 8,100–11,750 s per month per stream at `DOWNLOAD_WORKERS=6` with 3 streams in parallel (2026-06-21..24). That is ~2.3–3.3 h per month.
- Dec 2025 alone took 8,355 s.
- Estimate for 3 streams × 3 months: **≈ 8–10 h wall**. A single stream takes ≈ 25 h.
- In the old run, 2020-10 took 54,297 s because it included first-time setup, which is not repeated here.
- Storage, based on Dec 2025 (RTMA raw ~50 GB/month with byte-range selection, MRMS ~0.4 GB/month; chunk dir 2.4 GB/month):

| Item | Jan–Sep 2026 |
|---|---|
| RTMA raw GRIB (selected messages) | ~450–550 GB |
| MRMS raw | ~4 GB |
| Per-hour staging parquets | ~15–20 GB |
| Monthly chunks | ~22 GB |
| Curated WY2026 product (v001 = 7.5 GB for 45,720 h) | ~1.5 GB |
| **Total** | **≈ 0.5–0.6 TB** |

**USGS.**
- The June fetch took 10,780 s for 2,843 basins × ~6 WY requests (~17k requests).
- One WY chunk per basin is ~2,843 requests. Estimate **≈ 0.5–1 h** with 4 shards.
- After 2026-11-16 the 2 s/request delay adds ~25 min across 4 shards.
- Storage ≈ 1–2 GB (the 45,720-h run was 6.6 GB).

**Curated build plus audits** (after the code changes in §3): the v001 build took 14.4 h for 63 months, so WY2026 should take **≈ 2.5–3 h**. The structural USGS audit takes minutes.

**h2o headroom** (2026-11-12): `/data42` has **54 TB free** of 201 TB, against ≲ 0.6 TB needed. CPU: load average ≈ 100 of 128 cores, from other users' long jobs (swan.exe ×4 at ~23 cores each; caltra/python). There are no omrip processes. The extractor is mostly I/O- and network-bound with 6 download workers per stream, so 3 streams add ~18 S3 connections plus modest CPU for regridding.

## 7. Transfer to Moriah

As in the old scheme, the forcing and targets are **not** shipped to Moriah on their own. Only the NH package built from them goes (v002 sits at `/sci/labs/efratmorin/omripo/Flash-NH/data/stage1_scientific_package_v002`). Packaging the WY2026 series is the rotation-split session's step. When it happens, the transfer follows `docs/remote_operations.md`: `scp -O` (or a tarball plus `scp -O`) from h2o to Moriah. No transfer of the old package was recorded, so record this one's command and checksums.

The only transfer this plan needs is compact evidence pulled to the local repo, into `.scratch_local/wy2026_data_plan_20261112/evidence/`:
- 2026-MM manifests, summaries and `missing_files.csv`;
- the wrapper run logs;
- the USGS `launcher_summary.*` and shard manifests;
- the structural audit CSVs.

No raw GRIB, staging, chunks or discharge values are transferred.

## 8. Decisions for Omri

**(a) USGS vintage: refetch all of WY2026, or keep the June fetch for Oct–Dec 2025 and add Jan–Sep 2026?**

- *Recommendation: refetch all of WY2026 as one vintage* (`--start 2025-10-01T00:00:00Z --end 2026-09-30T23:00:00Z`).
  - It reproduces the old scheme exactly: one fetch gives one per-basin NetCDF over the whole target period.
  - No stitching code, which the old scheme never had.
  - One provenance stamp.
  - Oct–Dec 2025 gets the more-approved current values.
- *Strongest argument against:* Oct–Dec 2025 values would then differ from the June 2026 vintage, which is the one the existing 45,720-h products and the 89-basin late-gap (`TARGET_OPERATIONAL_REVIEW`) classification were built from. Two vintages of the same three months would then exist on h2o. (The 89-basin classification concerns admission, which WY2026 must not change in any case, so it is not re-derived from the new fetch.)
- *Append option:* `--start 2026-01-01T00:00:00Z`, plus a new stitch step that concatenates the June Oct–Dec slice. This is new code and a second vintage boundary at 2026-01-01.
- Either way, most of Jan–Sep 2026 will be *provisional* data, the same policy as the earlier data.

**(b) QC allowed on the protected year.**

- *Recommendation: structural and availability only, the same rule for targets and forcing.*
- USGS (C3 `--structural-only`):
  - file/attribute validation;
  - hourly grid length (8,760);
  - coverage per basin, per month and per WY;
  - gap count; longest gap; last valid timestamp;
  - NaN count; snap counts;
  - count of negatives turned into NaN (a mechanical cleaning rule; only the count is reported).
- No min/max/quantiles, flatline or spike tests, plots, hydrographs, metrics, or Jan-2023 comparison.
- Forcing: the extractor's 12 structural checks, the gap inventory, row counts, and the Oct–Dec bit-identity check against v001. No visual QC or animations.
- Open sub-question: are USGS **qualifier counts** (provisional vs approved, ice, equipment) allowed? They describe data status, not values. The recommendation is yes, as counts only.
- *Strongest argument against:* a structural audit cannot catch value-level faults (unit errors, a stuck gauge) before the test score is computed. Those would surface only at final evaluation, when nothing can be changed without contaminating the test.
- Corollary that needs your OK: a cleaned target set (U6) must not drop basins by WY2026 status. The WY2026 test population is fixed by the existing lists.

**(c) Launch window on h2o.**

- *Recommendation: launch the USGS fetch as soon as approved, ideally before **Mon 2026-11-16**.* USGS adds a 2 s per-request delay to WaterServices that day; 10 s from 2027-01-11; 30–60 s from late Jan/Feb; outages on 2027-01-27 and 2027-02-15..17; decommissioning on **2027-02-22**. Sources:
  - [USGS Water Data blog, "Intentional degradations to WaterServices…"](https://waterdata.usgs.gov/blog/wdfn-waterservices-degradation/) (last updated 2026-10-27);
  - [USGS GovDelivery bulletin](https://content.govdelivery.com/accounts/USDOIGS/bulletins/42b03d7).

  The fetch is network-bound and light on CPU (4 processes), so the current load of ≈ 100 on h2o is not a blocker. After 2027-02-22 the "same scheme" fetch cannot be repeated: any refetch, for example to pick up more approved data, would need a port to the new Water Data OGC API, which is a scheme change.
- *Forcing:* a weekend window, e.g. **Sat 2026-11-14 morning UTC**, 3 streams, ≈ 8–10 h.
- *Strongest argument against launching now:* h2o is shared and loaded by other users, and the full-period script asks for load ≤ 40 and notifying the PI before long jobs. If that norm should hold, the forcing waits for a quieter window. The USGS fetch is network-only.

## 9. Launch checklist (in order)

**Steps that need Omri, before anything runs:**
- [ ] (a), (b), (c) decided.
- [ ] Code review: `git diff` of `scripts/recover_usgs_iv_full_period_hourly.py`, `scripts/launch_usgs_iv_recovery_shards.py`, plus the new `scripts/run_wy2026_forcing_extraction_h2o.sh` and `tests/test_usgs_iv_wy_chunks.py`. Small and additive; a fresh-session review per §3.2 item 4 (HPC plumbing) if wanted.
- [ ] **L0: commit and push authorised.** Commit only these four files plus this plan, without `git add -A`. The rotation-split session's files in the same checkout stay out. Pushing master also pushes its other unpushed commits (3dad779, 92942ca, …), so it is Omri's call. Then `git -C /data42/omrip/Flash-NH/repos/flash-nh pull --ff-only` on h2o. The checkout is clean, so this is a fast-forward from 98b7d42.
- [ ] h2o etiquette: PI/lab notified if required; `uptime` and `df -h /data42` checked just before launch.

**Launch commands (ready, NOT run).**
- Run on h2o. Per memory, enter via `ssh flashnh-h2o` from the PowerShell tool.
- Each command runs in its own screen session from `/data42/omrip/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation`.
- Every block below first loads the conda env:

```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate /data42/omrip/Flash-NH/envs/flashnh-stage1
```

L1. USGS dry run, then fetch. The dry run makes no network calls and only checks plumbing and chunk windows. Variant (a) refetch shown; for (a) append, set `--start 2026-01-01T00:00:00Z`.

```bash
screen -S wy2026-usgs
python scripts/launch_usgs_iv_recovery_shards.py --staids-file config/stage1_initial_training_basin_manifest.csv --out-root /data42/omrip/Flash-NH/tmp/stage1_usgs_iv_wy2026_v001 --n-shards 4 --start 2025-10-01T00:00:00Z --end 2026-09-30T23:00:00Z --wy-chunks derived --dry-run
python scripts/launch_usgs_iv_recovery_shards.py --staids-file config/stage1_initial_training_basin_manifest.csv --out-root /data42/omrip/Flash-NH/tmp/stage1_usgs_iv_wy2026_v001 --n-shards 4 --start 2025-10-01T00:00:00Z --end 2026-09-30T23:00:00Z --wy-chunks derived 2>&1 | tee /data42/omrip/Flash-NH/tmp/stage1_usgs_iv_wy2026_v001_launch.log
```

Check in the dry-run output that each shard log prints `WY chunk table: derived` and `WY2026: 2025-09-30T23:45:00Z to 2026-09-30T23:15:00Z`. Poll within 1–3 minutes of the real launch.

L2. Forcing: plan print, then 3 streams.

```bash
DRY_RUN=1 GROUP_ID=D bash scripts/run_wy2026_forcing_extraction_h2o.sh
```

```bash
screen -S wy2026-forcing-d
GROUP_ID=D bash scripts/run_wy2026_forcing_extraction_h2o.sh 2>&1 | tee -a /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/logs/wy2026_group_d.log
```

```bash
screen -S wy2026-forcing-e
GROUP_ID=E bash scripts/run_wy2026_forcing_extraction_h2o.sh 2>&1 | tee -a /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/logs/wy2026_group_e.log
```

```bash
screen -S wy2026-forcing-f
GROUP_ID=F bash scripts/run_wy2026_forcing_extraction_h2o.sh 2>&1 | tee -a /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/logs/wy2026_group_f.log
```

**Monitoring (read-only).** Run `bash scripts/report_wy2026_status_h2o.sh` on h2o; no conda env is needed. It prints:
- load and `/data42` free space, and the wy2026 screen sessions;
- USGS: per shard, NCs written out of stations, fallback-failure and no-data request counts, and the last log time; then shard return codes once `launcher_summary.json` exists;
- forcing: per month, not started / running (% done, hours left, failed hour-products) / finished (`all_pass`), plus each group's last run-log line.

It never prints a discharge or forcing value.

To pause safely after the current month, use `touch /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/STOP_AFTER_MONTH`. Re-running a group skips months that already passed.

L3. USGS flat copy, after L1 shows `return_code` 0 for all shards in `launcher_summary.json`:

```bash
mkdir -p /data42/omrip/Flash-NH/tmp/stage1_usgs_iv_wy2026_v001/canonical_merged && cp -n /data42/omrip/Flash-NH/tmp/stage1_usgs_iv_wy2026_v001/shard_*/canonical/*_hourly.nc /data42/omrip/Flash-NH/tmp/stage1_usgs_iv_wy2026_v001/canonical_merged/ && ls /data42/omrip/Flash-NH/tmp/stage1_usgs_iv_wy2026_v001/canonical_merged | wc -l
```

The expected count is 2,843 less any basin with no data at all; the launcher summary lists them.

**After the downloads (needs the §3 changes plus review):**
- L4. Structural USGS audit (C3) → `audit_structural/`.
- L5. Curated WY2026 forcing build and audit (C1/C2) → `stage1_basin_hourly_forcings_wy2026_v001/`, plus the Oct–Dec bit-identity check against v001.
- L6. Pull compact evidence (§7). Report to the rotation-split session: the paths of the two WY2026 basin-average products, for its package step.

**Acceptance for this plan's scope:**
- All 9 forcing months `all_pass=true`, with the RTMA gap at 2026-09-23T19Z the only expected gap. MRMS gaps are listed in `missing_files.csv`.
- All 4 USGS shards return 0.
- Canonical NCs have 8,760 hourly steps.
- Structural audit written.
- No WY2026 value was opened, summarised or plotted.

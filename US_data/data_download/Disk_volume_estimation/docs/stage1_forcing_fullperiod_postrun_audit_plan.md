# Stage 1 Forcing — Full-Period Post-Run Audit and Package-Readiness Plan

**Created:** 2026-06-23  
**Status:** Audit complete — see result document below  
**Supersedes:** Nothing — companion document to `stage1_forcing_fullperiod_launch_plan.md`

> **Audit result (2026-06-24):** PASS_WITH_CAVEATS.
> Full result and generated tables: `docs/stage1_forcing_fullperiod_audit.md`.

---

## 1. Purpose and Scope

This document defines the complete post-run audit sequence and package-readiness criteria
for the Stage 1 full-period MRMS QPE 1h Pass1 + RTMA CONUS 2.5km basin-average forcing
extraction (Milestone 2K-C Phase 2).

**This plan does not:**
- Certify the current extraction while it is still running
- Prescribe monitoring actions during the run (see `stage1_forcing_fullperiod_launch_plan.md`)
- Build or certify a NeuralHydrology training package (that is a later, separate step)
- Connect to h2o or inspect live run state

**Applies to:** The extraction under
`/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/` on h2o, covering:
- **Period:** 2020-10-14T00Z – 2025-12-31T23Z (45,720 hours)
- **Monthly chunks:** 63 months
- **Basins:** 2,752 v001 basins (design target from Milestone 2J-B; verify against run manifest before using this number in any formula)
- **Launcher groups:** A (2020-10 → 2022-06, 21 months), B (2022-07 → 2024-01, 19 months),
  C (2024-02 → 2025-12, 23 months)

---

## 2. Final Extraction Completion Gate

Before any audit work begins, confirm that the extraction has reached a terminal state.
All checks are performed on locally pulled audit files — not on pasted terminal output.

### 2.1 Month accounting

| Check | Required value |
|---|---|
| Total months accounted for | **63 / 63** |
| Months with `all_pass: true` in manifest | ≥ 62 (see §8 for fail handling) |
| Months with `all_pass: false` | Recorded; see §8 for triage |
| October 2020 manifest `all_pass` | Must be `true` (Phase 1 baseline) |
| MRMS October 2020 gap | Exactly 36 `not_in_s3` entries — archive + S3 outage gaps |

### 2.2 Group terminal status

Confirm each group's run log shows a clean terminal exit (no orphaned screen session):

| Group | Screen name | Months | Expected terminal line |
|---|---|---|---|
| A | `flashnh-group-a` | 2020-10 → 2022-06 | `[group-a] All months complete` |
| B | `flashnh-group-b` | 2022-07 → 2024-01 | `[group-b] All months complete` |
| C | `flashnh-group-c` | 2024-02 → 2025-12 | `[group-c] All months complete` |

A group that exited on a fatal error shows `[group-X] Aborting` with the month label.
Partial completions are recoverable via `RESUME=1`; see the launch plan.

### 2.3 Summary counts (read from manifests — do not estimate)

> **After terminal state only.** Run or evaluate the source-column commands on h2o only
> after all group screen sessions have exited and the run is confirmed complete. Do not
> collect these values while a group is still running — partial counts will be wrong.

| Metric | Source | Record |
|---|---|---|
| Failed months | Count of `all_pass: false` across all 63 manifests | |
| Parse warnings | Sum of `parse_warnings` field across all manifests | |
| Diagnostic JSON files | `ls manifests/*_diagnostic*.json \| wc -l` | |
| Output Parquet files | `ls chunks/*/*.parquet \| wc -l` | |
| Disk usage (full tree) | `du -sh /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/` | |
| Git commit at run time | `run_provenance/*.json` → `git_commit` field | |
| Run command | `run_provenance/*_run_command.txt` | |
| Environment | `run_provenance/*.json` → `conda_env` and `python_version` fields | |
| Log locations | `logs/group_{a,b,c}.log` + `manifests/*_live_run.log` | |

Blank cells are filled from pulled audit files after the run completes.

---

## 3. Compact Audit Export Bundle

### 3.1 What to include

The evidence bundle is the authoritative record. Do not document the run from
pasted terminal summaries — the bundle must be pulled and inspected locally.

```
Include per-month (for all 63 months):
  <YYYY-MM>_manifest.json
  <YYYY-MM>_summary.md
  <YYYY-MM>_hourly_runtime_and_volume.csv
  <YYYY-MM>_scaling_estimates.json        (if present)
  <YYYY-MM>_validation_checks.csv
  <YYYY-MM>_live_run.log
  <YYYY-MM>_missing_files.csv             (if present)
  <YYYY-MM>_variable_completeness.csv     (if present)
  <YYYY-MM>_basin_completeness.csv        (if present)
  <YYYY-MM>_gap_inventory.csv             (if present)
  run_provenance/<YYYY-MM>_run_provenance.json
  run_provenance/<YYYY-MM>_run_command.txt

Include group-level:
  group_a_run_log.txt
  group_b_run_log.txt
  group_c_run_log.txt

Include cross-month summaries:
  fullperiod_row_counts.csv               (all months, MRMS + RTMA)
  fullperiod_warning_inventory.csv        (all warnings, type + month + count)
  fullperiod_diagnostic_inventory.csv     (all diagnostic JSON entries)
  fullperiod_gap_inventory.csv            (all missing hours, product + YYYY-MM + hour)
  fullperiod_variable_coverage.csv        (per-variable, per-month completeness)
  fullperiod_disk_summary.txt             (du -sh output)
```

### 3.2 What to exclude

```
Exclude:
  raw/mrms/**            (raw GRIB2 files — may already be partially deleted)
  raw/rtma/**            (raw GRIB2 files)
  staging/mrms/**        (hourly staging Parquets)
  staging/rtma/**        (hourly staging Parquets)
  chunks/**/*.parquet    (monthly chunk Parquets — too large; verify checksums separately)
  __pycache__/
  *.tmp
```

### 3.3 Build and pull command (run on h2o after terminal state confirmed)

```bash
MANIFEST_DIR=/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/manifests
EXPORT_DIR=/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/evidence_exports
BUNDLE_NAME="stage1_forcing_fullperiod_v001_audit_export_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"

mkdir -p "${EXPORT_DIR}"

tar czf "${EXPORT_DIR}/${BUNDLE_NAME}" \
  -C "$(dirname "${MANIFEST_DIR}")" \
  manifests/

# Transfer to local (Windows):
scp "flashnh-h2o:${EXPORT_DIR}/${BUNDLE_NAME}" tmp/stage1_forcing_fullperiod_evidence/
```

### 3.4 Documentation requirement

All post-run documentation, acceptance status decisions, and caveat records must be based
on inspected, locally-pulled audit files. Pasting terminal output into a doc is not
sufficient. The bundle is the record of truth.

---

## 4. Product Completeness and Schema Audit

### 4.1 Expected dimensions

| Dimension | Expected value |
|---|---|
| Total months | 63 |
| Total hours | 45,720 (2020-10-14T00Z – 2025-12-31T23Z) |
| Basin count | 2,752 per v001 target package — **confirm from run manifest before computing row counts** |
| MRMS product | `mrms_qpe_1h_pass1` |
| RTMA product | `rtma_conus_aws_2p5km` |
| MRMS variables | 1 (`precip_rate_mm_h` or canonical extraction name) |
| RTMA variables (model-facing) | 11 (see variable policy in §4.4) |

### 4.2 Row-count formulas

These formulas apply to the long-format Parquet output where each row is one
(basin, hour, variable) observation.

Let **n_basins** = actual basin count confirmed from the v001 weight table and run manifest.
The design target is 2,752; use the manifest value, not the design target, if they differ.

**MRMS monthly row count:**
```
mrms_rows(YYYY-MM) = mrms_extracted_hours(YYYY-MM) × n_basins
```
October 2020 baseline: `396 × 2752 = 1,089,792` (36 archive gaps; 396 extracted hours
from a partial month — 2020-10-14T00Z through 2020-10-31T23Z = 432 calendar hours, minus 36 gaps).

For all other months: `n_calendar_hours(YYYY-MM) × n_basins`, minus any transient S3
outage hours recorded in `missing_files.csv`.

**Deriving n_calendar_hours:** Always read from the actual timestamp range in the chunk
manifest, not from the calendar month name. Key cases to handle explicitly:

| Month | n_calendar_hours | Note |
|---|---|---|
| 2020-10 | 432 | Partial month: 2020-10-14T00Z – 2020-10-31T23Z |
| February (common year) | 672 | 28 × 24 |
| February 2024 | 696 | Leap year: 29 × 24 |
| 30-day months | 720 | Apr, Jun, Sep, Nov |
| 31-day months | 744 | Jan, Mar, May, Jul, Aug, Oct, Dec (full months) |

**RTMA monthly row count:**
```
rtma_rows(YYYY-MM) = rtma_extracted_hours(YYYY-MM) × n_basins × n_rtma_variables
```
October 2020 baseline: `432 × 2752 × 11 = 13,077,504` (432 hours = full calendar hours
for the partial-month range; RTMA had no archive gap unlike MRMS).  
Substitute the actual `n_rtma_variables` confirmed from the extraction schema.

**Combined monthly row count:**
```
combined_rows(YYYY-MM) = mrms_rows(YYYY-MM) + rtma_rows(YYYY-MM)
```
October 2020 baseline: `1,089,792 + 13,077,504 = 14,167,296`.

**Full-period combined row count:**
```
total_combined_rows = Σ combined_rows(YYYY-MM) for all 63 months
```
Compute from `fullperiod_row_counts.csv`; do not estimate from the formula alone.

### 4.3 Variable policy

#### MRMS
| Variable | Model-facing | Notes |
|---|---|---|
| `precip_rate_mm_h` (QPE 1h Pass1) | Yes | Basin-weighted mean |

#### RTMA CONUS 2.5km — included in model-facing dynamic forcings
| cfgrib name | Canonical name | Model-facing | Notes |
|---|---|---|---|
| `t2m` | `2t` | Yes | 2m temperature (K) |
| `d2m` | `d2m` | Yes | 2m dewpoint temperature (K) |
| `u10` | `10u` | Yes | 10m U-wind component (m/s) |
| `v10` | `10v` | Yes | 10m V-wind component (m/s) |
| `sp` | `sp` | Yes | Surface pressure (Pa) |
| `tp` or `prate` | `tp` / `prate` | Yes | Total precipitation / precipitation rate |
| `tcc` | `tcc` | Yes | Total cloud cover (0–1) |
| `vis` | `vis` | Yes | Visibility (m) |
| `gust` | `gust` | Yes | Wind gust (m/s) |
| `weasd` | `weasd` | Yes | Water equivalent of snow depth (kg/m2) |
| *(confirm 11th)* | *(confirm)* | Yes | Verify from extraction schema |

#### RTMA variables excluded from model-facing dynamic forcings
| cfgrib name | Canonical name | Model-facing | Reason |
|---|---|---|---|
| `wdir10` | `10wdir` | **No** | Wind direction (degrees) — redundant with 10u/10v vectors; directional variable not suitable for direct NeuralHydrology input |
| `orog` | `orog` | **No** | Orography (static field) — time-invariant; not appropriate as a dynamic forcing; static basin attributes are handled separately |

`10wdir` and `orog` may be present in the raw extracted Parquet but must not appear in
the model-facing forcing product columns. Verify their absence from the forcing package
schema (§9).

### 4.4 All-null variable check

No variable column in any monthly chunk Parquet may be all-null. Flag any variable that
has zero finite values across all basins for a given month as a critical schema error.
Partial null is expected for RTMA hours with gaps; all-null for a whole month is not.

### 4.5 Valid-weight and missing-value summaries

| Check | Pass criterion |
|---|---|
| MRMS weight table: basins with ≥1 finite weight | n_basins / n_basins (design target: 2,752 / 2,752) |
| RTMA weight table: basins with ≥1 finite weight | n_basins / n_basins (design target: 2,752 / 2,752) |
| MRMS rows with NaN weighted mean | Recorded; expected only for gap hours |
| RTMA rows with NaN weighted mean | Recorded; expected only for gap hours |
| Basins with zero MRMS coverage (all 45,720 h missing) | Expected: 0 |
| Basins with zero RTMA coverage (all 45,720 h missing) | Expected: 0 |

---

## 5. Forcing Gap Audit

Gap auditing is a required step before any product is declared complete or usable.
It is not a diagnostic-only exercise; its findings drive the gap policy application in §6.

### 5.1 Gap definitions

**MRMS gap:** Any calendar hour in 2020-10-14T00Z – 2025-12-31T23Z that is absent from
the MRMS chunk Parquet for a given basin. Subdivided by cause recorded in
`missing_files.csv`:
- `not_in_s3`: S3 archive has no file for that hour — permanent and pipeline-independent
- `download_fail`: S3 had the file but the download failed — potentially recoverable
- `decode_fail`: File downloaded but cfgrib/wgrib2 decode failed

**RTMA gap:** Same definition applied to RTMA hours.

**Isolated gap:** A gap of exactly 1 consecutive hour, with finite values on both
the preceding hour and the following hour (both within the 45,720h period).

**Multi-hour gap:** A gap of ≥2 consecutive hours.

### 5.2 Gap inventory to produce

Compute from `fullperiod_gap_inventory.csv` and monthly `missing_files.csv` files:

| Metric | Granularity |
|---|---|
| Total missing hours | By product (MRMS / RTMA), by YYYY-MM |
| Missing hours by cause | `not_in_s3` / `download_fail` / `decode_fail` |
| Isolated 1h gaps | By product, by basin, by month |
| Multi-hour gaps (≥2h) | By product, by basin; record gap start, length, cause |
| Product-synchronized gaps | Hours missing in both MRMS and RTMA for the same hour — indicates S3 outage vs. product-specific issue |
| Repeated/systematic gaps | Any hour that is missing across ≥10% of basins — likely S3 archive absence rather than per-basin issue |

### 5.3 Model-window impact (24h input sequence)

A standard NeuralHydrology sequence of 24 consecutive hours is invalid if any of its
24 hours has a missing forcing value for any required variable.

**Per missing-hour cost:**
```
windows_lost_per_isolated_gap_hour = min(24, remaining_hours_in_period)
```
For a consecutive gap of length L hours:
```
windows_lost = L + 23   (each additional gap hour costs 1 additional window)
```

**Full-period window impact (per basin):**
```
n_extracted_hours = total unique hours in run (from fullperiod_row_counts.csv)
                    design target: 45,720 (2020-10-14T00Z – 2025-12-31T23Z)
max_possible_windows = n_extracted_hours - 24 + 1
n_invalid_24h_windows(basin) = |{w : any hour in window w is missing}|
fraction_training_windows_lost(basin) = n_invalid_24h_windows(basin) / max_possible_windows
```
Read `n_extracted_hours` from the manifest; do not assume 45,720 if any months were
skipped or failed.

**Report:**

| Metric | Value |
|---|---|
| n_invalid_24h_windows: mean across n_basins basins (per manifest) | |
| n_invalid_24h_windows: worst basin (STAID, count) | |
| fraction_training_windows_lost: mean | |
| fraction_training_windows_lost: worst basin | |
| Worst month by windows lost (YYYY-MM, product) | |
| Basins where fraction > 5% | Count and STAID list |

These cells are filled from pulled audit files after run completes.

---

## 6. Proposed Forcing Gap Policy

The gap policy governs how forcing gaps are handled in derived products and which
model input windows are considered valid for training. The raw curated forcing product
(see §9) always preserves original missingness; any filling is a derived layer.

### 6.1 RTMA continuous variables

**Scope: derived package layers only.** The raw curated forcing product (§9) is never
modified. Any interpolation described here applies exclusively when building a derived
package layer from the raw product as input. See §6.4 for the raw-vs-derived boundary.

**Isolated 1h gaps (gap length = 1, both neighbors finite) — derived layer only:**
- May be filled by linear interpolation at the basin time-series level.
- Filling is performed variable-by-variable, basin-by-basin.
- A companion boolean flag column (`<var>_interpolated`) must be set to `True` for
  any filled hour; `False` (or absent) otherwise.
- This is an optional step and produces a versioned derived artifact separate from
  the raw curated product; the raw product always retains NaN for the gap hour.
- Applies to: `2t`, `d2m`, `sp`, `10u`, `10v`, `tcc`, `vis`, `gust`, `weasd`,
  and any other continuous RTMA variable confirmed in the schema.
- Does **not** apply to RTMA precipitation variables (see §6.3).

**Gaps ≥2h (multi-hour):**
- Preserve as missing in all products.
- Any 24h model input window that includes ≥1 missing hour is flagged as invalid
  and skipped during training data assembly.
- Do not extrapolate or forward-fill.

### 6.2 MRMS precipitation

**Default: no interpolation.**

Precipitation is not a continuous slowly-varying variable; interpolating across a gap
fabricates storm timing and intensity. Even a 1h gap in MRMS should be preserved as
missing unless a separate evidence review demonstrates a specific recoverable case.

**Isolated 1h MRMS gap near-zero-neighbor case (future optional policy — not default):**
If both the preceding and following MRMS hours show zero or near-zero precipitation
(< 0.1 mm/h basin mean), treating the gap hour as zero may be physically defensible
in some contexts. This is listed as a candidate future policy for review; it is not
activated by default in v001.

**Gaps ≥2h:**
- Preserve as missing.
- Skip affected model windows.

### 6.3 RTMA precipitation variables

Any RTMA precipitation-rate variable (`tp`, `prate`, or equivalent) follows the same
no-interpolation rule as MRMS. These are precipitation estimates and subject to the
same physical constraints as §6.2.

### 6.4 Raw curated product vs. derived package layers

```
Raw curated product (stage1_basin_hourly_forcings_v001):
  - Preserves original missingness exactly as extracted.
  - No interpolation applied.
  - NaN for any gap hour across all variables for that hour.
  - This is the archival record.

Derived package layer (for training use):
  - Applies gap policy (§6.1) to continuous RTMA variables.
  - Includes companion fill-flag columns.
  - Tags model windows as valid/invalid based on gap presence.
  - Built from the raw curated product as input; never modifies the curated product.
  - Version-tagged separately from the raw curated product.
```

The NeuralHydrology dataset config (§9) references the raw curated product; any
gap-filled variant is a separate artifact with its own version and provenance.

---

## 7. Targeted Visual/Event QC Layer

Visual QC animations are a **formal post-run QC gate**, not optional presentation
material. No acceptance status above PASS_WITH_CAVEATS may be declared without
completing this gate.

### 7.1 Scope and sampling strategy

Do not generate all possible animations. Generate a **curated sample of 12–24 cases**
selected to stress-test the extraction across dimensions that cannot be captured by
row counts and gap tables alone.

**Stratification targets:**

| Category | Target count | Selection criterion |
|---|---|---|
| Strong precipitation events | 3–4 | High basin-mean MRMS; strong streamflow rise; warm season or frontal |
| Cold / snow-dominated cases | 2 | Low RTMA 2t (< −5 C); nonzero MRMS; near-zero streamflow response |
| Dry control cases | 2 | Low or zero MRMS over a 72h window; confirms no spurious precipitation |
| High-offset cases | 2 | Gauge > 1 km from polygon centroid; tests spatial coherence |
| Small flashy basins | 2 | DRAIN_SQKM < 100 km²; rapid Q rise ratio |
| Large basins | 1–2 | DRAIN_SQKM > 1,000 km²; tests spatial aggregation at scale |
| Warm-season convective cases | 2–3 | Summer months; isolated convective cells; high spatial variability |
| Winter stratiform cases | 1–2 | Winter months; broad frontal precipitation; cold RTMA |
| Cases near forcing gaps | 2 | Select a month with confirmed MRMS or RTMA gaps; animate the gap window |
| Random ordinary cases | 2–3 | Random basin × month draws; moderate forcing; no special features |

### 7.2 What the human reviewer must inspect

For each animation, the reviewer checks:

| Element | What to look for |
|---|---|
| Basin polygon | Polygon rendered over correct geographic region; no coordinate shift |
| Gauge marker | Gauge at plausible location relative to polygon; offset arrow shown if > 1 km |
| Extraction cells | MRMS cell overlay covers the polygon; no off-by-one or resolution mismatch |
| Raster alignment | MRMS precipitation raster aligns with extraction cells and basin geography |
| Basin mean vs. raster | Basin-mean label tracks the visible raster intensity; no sign inversion or scale error |
| Time synchronization | Frame timestamp matches raster timestamp; red cursor in panels moves with the frame |
| Precipitation–streamflow timing | Lag between MRMS peak and Q peak is physically plausible for basin size and type |
| RTMA plausibility | RTMA 2m temperature and wind direction are seasonally and regionally plausible |
| Gap visibility | If the animation window includes a gap hour: NaN frame is rendered as blank/grey, not as zero or interpolated |

### 7.3 Output and git status

Animation outputs are written to the local `tmp/` tree. They are gitignored and must
not be committed. The manifest records which cases were generated and reviewed; the
manifest itself may be committed as part of the audit summary.

```
Generated outputs (gitignored):
  tmp/stage1_forcing_fullperiod_qc/animations/<STAID>_<YYYY-MM>_<category>.gif
  tmp/stage1_forcing_fullperiod_qc/animations/<STAID>_<YYYY-MM>_<category>.mp4

Committable (not gitignored):
  tmp/stage1_forcing_fullperiod_qc/visual_qc_case_selection.csv   (reproducible case record; schema below)
  tmp/stage1_forcing_fullperiod_qc/animation_review_log.md        (reviewer findings)
```

**visual_qc_case_selection.csv — column schema (one row per animation case):**

| Column | Type | Description |
|---|---|---|
| `case_id` | str | Sequential label (e.g., `VQC01`) |
| `staid` | str | USGS station ID |
| `yyyy_mm` | str | Forcing chunk month (e.g., `2021-07`) |
| `window_start_utc` | str | Animation window start, ISO-8601 (e.g., `2021-07-14T06Z`) |
| `window_end_utc` | str | Animation window end, ISO-8601 |
| `category` | str | Stratification label from §7.1 (e.g., `STRONG_PRECIP`, `COLD_SNOW`) |
| `selection_rationale` | str | One sentence explaining why this case was chosen |
| `basin_area_km2` | float | DRAIN_SQKM from v001 basin metadata |
| `gauge_offset_m` | float | Gauge-to-polygon-centroid distance (m) |
| `gap_present` | bool | True if a confirmed forcing gap falls within the animation window |
| `reviewer` | str | Initials or name of human reviewer |
| `review_outcome` | str | One of: `PASS` / `PASS_WITH_NOTES` / `FAIL` |
| `notes` | str | Free-text reviewer observations |

This file is the reproducibility record for animation QC. It must be committed (not
gitignored) so that case selection can be audited and animations regenerated if needed.
`reviewer` and `review_outcome` are filled after human inspection; the other columns
are filled at case-selection time before animation generation begins.

### 7.4 Animation design reference

The v2.1-stable layout from `docs/stage1_january_event_qc.md` (§5) is the reference
for map + streamflow + precipitation + temperature panel design. Adapt for the
full-period context: basin count is n_basins per manifest (design target 2,752),
period is 2020–2025, no restriction to January 2023 or to the 50-basin pilot set.

---

## 8. Acceptance Status Taxonomy

Each monthly chunk and the overall extraction receive one of the following statuses:

| Status | Meaning |
|---|---|
| **PASS** | All row counts match expected, zero unexpected FAIL statuses in validation checks, missing hours accounted for (all `not_in_s3`), visual QC gate complete |
| **PASS_WITH_CAVEATS** | Row counts and schema checks pass; one or more documented and explained anomalies (e.g., confirmed S3 outage gaps, parse warnings with known cause); visual QC gate complete |
| **NEEDS_TARGETED_REPAIR** | Row count or schema issue on a small number of basins or hours; repairable without re-running the full chunk; targeted intervention plan defined |
| **NEEDS_RERUN_FOR_SELECTED_MONTHS** | ≥1 monthly chunk has unexplained FAIL statuses, unexpected gaps, or schema errors not repairable in place; those months need a clean re-run |
| **FAIL** | Extraction-wide or structural failure; fundamental re-run required; not declared on an individual month |

**Overall extraction status** is the most severe status across all 63 months. If any
month is `NEEDS_RERUN_FOR_SELECTED_MONTHS`, the overall status is at least that level.

**Recording:** The overall acceptance status and per-month status are recorded in the
audit summary document after the audit is complete (not in this design plan).

---

## 9. Curated Forcing Product v001 Design

The curated forcing product is a separate deliverable from the extraction run output
and from the NeuralHydrology package. It is assembled after the extraction audit passes.

### 9.1 Proposed name and location

```
Product name:  stage1_basin_hourly_forcings_v001
Location:      /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/
               stage1_basin_hourly_forcings_v001/
```

### 9.2 Required contents

```
stage1_basin_hourly_forcings_v001/
├── manifest.csv                     # Per-basin: STAID, n_mrms_hours, n_rtma_hours,
│                                    #   n_gap_hours_mrms, n_gap_hours_rtma, sha256
├── checksums.sha256                 # SHA-256 for every per-basin forcing file
├── dataset_config.yaml              # Variable list, product names, period, basin count
├── source_git_commit.txt            # Git hash of repo at extraction time
├── run_provenance.json              # Extractor version, run date, group logs, env
├── audit_summary.md                 # Post-run audit findings and acceptance status
├── gap_policy.md                    # This document's §6, versioned and dated
├── variable_policy.md               # Included / excluded variables with reasons
└── <STAID>/
    └── <STAID>_hourly_forcings.parquet   # Per-basin forcing time series
                                           # Columns: timestamp, product, variable, value
                                           # NaN preserved for gap hours
                                           # 10wdir and orog absent from columns
```

### 9.3 What this product is not

- It is not a NeuralHydrology training package. The NH package is assembled later,
  after forcing audit approval, using this product as input.
- It does not contain gap-filled values. Any interpolated version is a derived layer
  with its own version tag (e.g., `stage1_basin_hourly_forcings_v001_filled`).
- It does not contain streamflow data. Streamflow is assembled from the USGS IV target
  package (built under Milestone 2J-B).

---

## 10. Forcing-to-NeuralHydrology Package Readiness

### 10.1 Gate: forcing audit must pass first

The NeuralHydrology package assembly does not begin until:
1. The extraction receives an overall status of at least PASS_WITH_CAVEATS.
2. The visual/event QC gate (§7) is complete.
3. The gap policy (§6) is confirmed and documented.
4. The curated forcing product v001 (§9) is assembled and checksummed.

Package design decisions (variable selection, normalization, sequence length, split
strategy) are made in a separate design document after the forcing audit is approved.

### 10.2 Smoke package before full assembly

Before assembling the full NH package (expected n_basins per forcing manifest — design
target 2,752), run a 10-basin smoke package:
- Select basins spanning the range of gap fractions (low, median, high).
- Include ≥1 cold-season basin and ≥1 flashy basin.
- Confirm the NH dataset loader reads the forcing correctly (shape, dtype, no NaN
  in valid windows, correct variable order).

### 10.3 Training is out of scope for h2o

NeuralHydrology model training is designated for the Moriah cluster
(`/sci/labs/efratmorin/omripo/PhD`). No training runs on h2o. The NH package
assembly (local file structuring, config writing) may occur on h2o, but model fitting
does not.

---

## Appendix: Checklist Summary

| Step | Status | Notes |
|---|---|---|
| Extraction terminal state confirmed (63/63 months) | — | After run completes |
| Group A/B/C terminal logs confirm clean exit | — | |
| Audit export bundle pulled locally | — | |
| Month accounting complete (fail count, warning count) | — | |
| Row-count audit complete (MRMS + RTMA formulas) | — | |
| Variable schema audit complete (10wdir / orog absent) | — | |
| All-null variable check: 0 variables all-null | — | |
| Gap inventory complete (isolated, multi-hour, synchronized) | — | |
| Model-window impact computed (n_invalid_24h_windows) | — | |
| Gap policy document written and dated | — | |
| Visual/event QC: case sample selected (12–24) | — | |
| Visual/event QC: all cases reviewed and logged | — | |
| Acceptance status assigned (overall + per month) | — | |
| Curated forcing product v001 assembled | — | |
| Checksums generated | — | |
| Smoke NH package test: PASS | — | |
| Full NH package assembly authorized | — | |

---

*This is a design document. All status cells above are filled after the extraction
run on h2o reaches a terminal state and audit files are pulled locally.*
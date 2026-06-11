# Flash-NH Stage 1 — HPC Transition Preflight Notes

Date: 2026-06-11  
HPC host: `h2o.es.huji.ac.il`

---

## Environment

| Item | Value |
|---|---|
| HPC host | `h2o.es.huji.ac.il` |
| Personal work root | `/data42/omrip/Flash-NH` |
| Generated outputs | `/data42/omrip/Flash-NH/tmp/` |
| Repo clone | `/data42/omrip/Flash-NH/repos/flash-nh` |
| Stage 1 working dir | `/data42/omrip/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation` |
| Job scheduler | None detected at preflight; SLURM not yet configured for this project |
| Python env | Shared system env used for smoke; project-local venv preferred for production |

---

## Preflight Checklist

| Check | Result |
|---|---|
| `git clone` of `master` succeeded | PASS |
| `git pull` brings latest `edda406` (HPC preflight patch) | PASS |
| USGS NWIS IV network access from h2o | PASS — requests.get() to waterservices.usgs.gov succeeded |
| `recover_usgs_iv_full_period_hourly.py --dry-run` | PASS |
| 1-month smoke for `02073000` (Jan 2023, post-`edda406`) | PASS — 743/744 valid (1 NaN at T=0, acceptable) |
| Full-period smoke for `02073000` (post-`edda406`) | PASS — 45,720/45,720 valid |
| PROJ/pyproj warning at import | NON-BLOCKING — warning observed but irrelevant for streamflow scripts |
| Pilot manifest on HPC | NOW VERSIONED — `config/stage1_pilot_basin_manifest.csv` committed; no longer requires generated tmp path |
| 2H-C comparison files on HPC | ABSENT — use `--skip-jan2023-comparison`; comparison will be SKIPPED |

---

## Issues Found During Smoke and Fixes Applied

### Issue 1 — Recovery script fetched all 6 WY chunks for a 1-month smoke

**Symptom:** Running with `--start 2023-01-01T00:00:00Z --end 2023-01-31T23:00:00Z` still
issued 6 water-year API requests (WY2021–WY2026), most returning empty responses.

**Root cause:** `load_or_fetch_all_chunks` iterated over the global `WY_CHUNKS` list
regardless of the requested period.

**Fix (post-`ba4b577`):** Added `compute_active_chunks(start_ts, end_ts)` which filters
`WY_CHUNKS` to chunks overlapping the requested interval and clips request windows to that
interval (plus a 15-minute tail buffer for snap-tolerance coverage). `main()` now prints
the planned active chunks before any fetch, making dry-run output unambiguous.

For a Jan 2023 smoke, only **WY2023** is fetched, clipped to
`2023-01-01T00:00:00Z` → `2023-01-31T23:15:00Z`.

Full-period default runs are unaffected (all 6 chunks pass the filter; WY2026 tail
changes from `23:59:59` to `23:15:00`, which is functionally identical for the ±15-min snap).

---

### Issue 2 — Audit falsely flagged a late-period gap for 1-month smoke files

**Symptom:** Running the audit against the Jan 2023 smoke output produced a
`late_2025_gap_flag = True` for every basin because it compared the last valid
observation (`2023-01-31T23:00:00`) against the full-period constant
`PERIOD_END = 2025-12-31T23:00:00`, yielding ~1,065 days "gap".

**Root cause:** `compute_per_basin_coverage` and `compute_gap_audit` hardcoded `PERIOD_END`.

**Fix (post-`ba4b577`):** Both functions now accept a `period_end` keyword argument
(defaulting to `PERIOD_END` for backward compatibility). `main()` resolves the effective
audit period in priority order:

1. `--expected-start` / `--expected-end` CLI args (explicit override)
2. `period_start_utc` / `period_end_utc` attrs embedded in the first canonical NC file
   (written by `recover_usgs_iv_full_period_hourly.py` since Milestone 2I-B)
3. Module-level defaults (`PERIOD_START` / `PERIOD_END` = full Stage 1 period)

For a Jan 2023 smoke file with `period_end_utc = 2023-01-31T23:00:00Z` in its NC attrs,
the late-gap check compares against `2023-01-31T23:00:00`, yielding 0 days → not flagged.
For full-period files, `03298135` is still correctly flagged at 37.3 days.

---

### Issue 3 — Missing pilot manifest and 2H-C files

**Symptom:** Audit printed warnings about missing `pilot_basin_manifest.csv` and
`15_streamflow_recovery_january_eligible/` files on HPC; advisory statuses showed `UNKNOWN`.

**Behavior (already clean before this patch):** `compare_jan2023` returned `NOT_APPLICABLE`
for each basin when the 2H-C file didn't exist. `load_pilot_manifest` returned `{}` with a
warning; advisory statuses showed `UNKNOWN` pilot_role.

**Additional controls added:**
- `--pilot-manifest PATH` — override the manifest path (e.g. point to a copy on HPC, or
  pass a nonexistent path to suppress the role lookup without error)
- `--skip-jan2023-comparison` — skip the comparison function entirely; all basins get
  `comparison_status = SKIPPED`; the QC plot is not generated

---

## Recommended Smoke Commands on h2o

After `git pull` in the working directory:

```bash
cd /data42/omrip/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation
git pull

# 1. Dry-run — verify planned chunk windows (no network, no writes)
python scripts/recover_usgs_iv_full_period_hourly.py \
    --staids 02073000 \
    --start 2023-01-01T00:00:00Z \
    --end   2023-01-31T23:00:00Z \
    --out-dir /data42/omrip/Flash-NH/tmp/smoke_jan2023_02073000 \
    --dry-run

# 2. Real 1-month smoke fetch (single basin, Jan 2023 only)
python scripts/recover_usgs_iv_full_period_hourly.py \
    --staids 02073000 \
    --start 2023-01-01T00:00:00Z \
    --end   2023-01-31T23:00:00Z \
    --out-dir /data42/omrip/Flash-NH/tmp/smoke_jan2023_02073000 \
    --force

# 3. Audit of smoke output (period inferred from NC attrs; skip comparisons unavailable on HPC)
python scripts/audit_usgs_iv_recovered_targets.py \
    --canonical-dir /data42/omrip/Flash-NH/tmp/smoke_jan2023_02073000/canonical \
    --out-dir       /data42/omrip/Flash-NH/tmp/smoke_jan2023_02073000 \
    --skip-jan2023-comparison
```

Expected output:
- Recovery: `Active WY chunks (1/6 total): WY2023: 2023-01-01T00:00:00Z to 2023-01-31T23:15:00Z`
- Recovery: `PASS valid=744 nan=0 neg=0`
- Audit: `Audit period (NC attrs): 2023-01-01 00:00:00 to 2023-01-31 23:00:00`
- Audit: no late-period gap flags
- Audit: `comparison_status = SKIPPED` for all basins

---

## h2o Smoke Results After edda406

Both smokes ran after pulling commits `7f17f17` (period-aware recovery/audit) and
`edda406` (request head buffer).

### Smoke A — 1-month Jan 2023 (`02073000`)

| Item | Value |
|---|---|
| Basin | `02073000` |
| Period | `2023-01-01T00:00:00Z` to `2023-01-31T23:00:00Z` |
| Target grid | 744 hourly steps |
| Active chunks | 1/6 — WY2023 only |
| API request window | `2022-12-31T23:45:00Z` to `2023-01-31T23:15:00Z` (±15 min buffered) |
| Raw rows | 2,975 |
| Result | **PASS** |
| valid / NaN | 743 / 1 |
| Missing timestamp | `2023-01-01 00:00:00` |
| Audit period | Inferred from NC attrs: Jan 2023 |
| Late-gap flag | False |
| Jan 2023 comparison | SKIPPED (`--skip-jan2023-comparison`) |
| Output footprint | 104 KB under `/data42/omrip/Flash-NH/tmp/smoke_jan2023_02073000` |

**NaN interpretation:** The single NaN at `2023-01-01T00:00:00Z` is acceptable. USGS
returned no raw observation within ±15 min of that target hour even with the head buffer
applied. This is not a snapping failure — it is a genuine data gap at the very first
timestamp of the requested window for this station.

---

### Smoke B — Full-period (`02073000`, 2020-10-14 through 2025-12-31)

| Item | Value |
|---|---|
| Basin | `02073000` |
| Period | `2020-10-14T00:00:00Z` to `2025-12-31T23:00:00Z` |
| Target grid | 45,720 hourly steps |
| Active chunks | 6/6 |
| Raw rows | 181,894 |
| Snap: exact | 45,389 |
| Snap: nearest (within ±15 min) | 331 |
| Snap: missing | 0 |
| Result | **PASS** |
| valid / NaN / negative | 45,720 / 0 / 0 |
| Station wall time | 14.2 s |
| Total wall time | 14.7 s |
| Raw cache | ~1.5 MB |
| Canonical NC | ~549 KB |
| Output footprint | 2.2 MB under `/data42/omrip/Flash-NH/tmp/smoke_fullperiod_02073000` |
| Audit period | Inferred from NC attrs: full Stage 1 period |
| Late-gap flag | False |
| Jan 2023 comparison | SKIPPED (`--skip-jan2023-comparison`) |

100% coverage, 0 NaN. Full-period pipeline validated end-to-end on h2o.

---

## Parallelism and Scale-Up

The current script is sequential within a station and across stations when invoked
directly. For the 50-basin Stage 1 full-period build this is fine on a single node;
at ~15 s/basin the full 50-basin run takes roughly 12–15 minutes.

For scale-up beyond 50 basins, prefer **external station-level parallelism** rather
than adding concurrency inside the script (which would complicate the Parquet cache
and rate-limiting):

| Approach | When to use |
|---|---|
| SLURM job array (`--array=0-49`) | If scheduler access on h2o is confirmed |
| `GNU parallel` / `xargs -P` with `--delay` | If h2o is a shared server without SLURM |
| Small Python launcher (`concurrent.futures`, `max_workers=4–8`) | If neither of the above is available |

**Rate-limit guidance:** Do not use aggressive concurrency against USGS NWIS IV.
Start with 4 parallel stations. After validating that all 4 succeed cleanly, cautiously
raise to 8. The script already adds a polite inter-station delay; parallel launchers
should add an additional inter-request delay of at least 0.5–1 s.

---

## Versioned Pilot Basin Manifest

`config/stage1_pilot_basin_manifest.csv` is now committed to the repo.  It contains
the frozen Stage 1 pilot role assignments for all 50 basins (STAID + pilot_role only),
derived from the generated tmp manifest produced by `select_pilot_basins.py` (seed=42).

| pilot_role | Count |
|---|---|
| TRAIN | 40 |
| HOLDOUT_QC | 5 |
| EXCLUDE_QC | 5 |

The audit script resolves the pilot manifest in this priority order:

1. `--pilot-manifest PATH` (explicit CLI override)
2. `config/stage1_pilot_basin_manifest.csv` (versioned, works on HPC)
3. `tmp/stage1_pilot_dryrun/09_manifests/stage1_pilot/pilot_basin_manifest.csv` (legacy local fallback)

After `git pull` on h2o, the audit will find the versioned manifest and advisory
`pilot_role` values will no longer show `UNKNOWN` for the 50 known pilot basins.

---

## Scheduler Status

No SLURM configuration has been set up yet for this project on h2o. The Stage 1
50-basin full-period acquisition will be the first scheduled job array. Script design
is ready; submission scripts to be written in the next milestone (2I-C or equivalent).

---

## PROJ Warning

At import on h2o, the following warning appeared:

```
UserWarning: pyproj unable to set database path ...
```

This is non-blocking for all Flash-NH Stage 1 streamflow scripts (no geographic projection
needed). It will be resolved when the project-local Python venv is set up with matching
PROJ data files.

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
| **50-basin full-period smoke (post-`8c58416`)** | **PASS — 50/50 stations, 2,224,973 valid hours, coverage 0.9733** |
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

### Smoke C — 50-basin full-period (post-`8c58416`)

Run location: `/data42/omrip/Flash-NH/tmp/smoke_fullperiod_50basins`

#### Recovery

| Item | Value |
|---|---|
| Stations processed | 50 |
| PASS / FAIL / ERROR | 50 / 0 / 0 |
| Period per basin | `2020-10-14T00:00:00Z` to `2025-12-31T23:00:00Z` |
| Target grid | 45,720 hourly steps/basin |
| Total runtime | 748.1 s (~14.96 s/basin) |
| Raw cache | 95.8 MB |
| Canonical NCs | 26.8 MB |
| Output footprint (post-audit) | 125 MB |
| Negative values | 0 |
| Failed/zero-observation chunks | 3 (`07263580` WY2026 Oct/Nov/Dec) |

The 3 failed chunks are expected: `07263580` is `EXCLUDE_QC` with a late-period outage (last obs 2025-05-01). This is not a pipeline failure.

#### Audit

| Item | Value |
|---|---|
| Basins audited | 50 |
| Total valid hours | 2,224,973 |
| Total NaN hours | 61,027 |
| Overall coverage | 0.9733 |
| Basins ≥ 90% coverage | 48 / 50 |
| Late-2025 gap flags | 3 |
| Suspicious spike flags | 24 |
| Jan 2023 comparison | SKIPPED (`--skip-jan2023-comparison`) |
| Pilot roles (versioned manifest) | TRAIN 40, HOLDOUT_QC 5, EXCLUDE_QC 5 |

#### Target-status distribution

| target_status | Count |
|---|---|
| TARGET_QUALITY_REVIEW | 22 |
| TARGET_USABLE_WITH_GAPS | 12 |
| TARGET_READY_CONTINUOUS | 9 |
| TARGET_ROLE_EXCLUDED | 5 |
| TARGET_OPERATIONAL_REVIEW | 2 |

| Advisory flag | True | False |
|---|---|---|
| `operational_readiness_flag` | 21 | 29 |
| `historical_training_utility_flag` | 43 | 7 |

#### Late-period gap basins

| STAID | Last obs | Days before period end | pilot_role |
|---|---|---|---|
| `03298135` | 2025-11-24 16:00:00 | 37.3 | TRAIN |
| `05372995` | 2025-12-01 12:00:00 | 30.5 | TRAIN |
| `07263580` | 2025-05-01 04:00:00 | 244.8 | EXCLUDE_QC |

#### High-review basins

| STAID | pilot_role | target_status | coverage | longest_gap_h | late_gap |
|---|---|---|---|---|---|
| `07263580` | EXCLUDE_QC | TARGET_ROLE_EXCLUDED | 0.8698 | 5,875 | True |
| `01390450` | TRAIN | TARGET_QUALITY_REVIEW | 0.6748 | 14,854 | False |
| `03298135` | TRAIN | TARGET_OPERATIONAL_REVIEW | 0.9750 | 895 | True |
| `05372995` | TRAIN | TARGET_OPERATIONAL_REVIEW | 0.9385 | 731 | True |
| `10348850` | TRAIN | TARGET_USABLE_WITH_GAPS | 0.9425 | 905 | False |

#### Interpretation

**Result: PASS.** The 50-basin full-period acquisition pipeline is validated end-to-end on h2o.

- Late-period gaps (`03298135`, `05372995`) are operational-readiness issues; they do not automatically exclude basins from historical training.
- `01390450` (coverage 0.6748, longest gap 14,854 h) requires scientific review before use in training.
- Suspicious-spike flags (24 basins) are advisory and require manual interpretation; they do not disqualify basins.
- `07263580` (EXCLUDE_QC) is excluded from training by policy regardless of coverage.
- The `TARGET_QUALITY_REVIEW` majority reflects conservative spike thresholds (5× p99), not a systemic pipeline problem.

#### Prerequisites before scaling to full 2,843-basin run

1. Exact 2,843 STAID list confirmed.
2. Job persistence method on h2o decided (`tmux`, `screen`, or `nohup`).
3. Decision on sequential vs. conservative external station-level parallelism (see **Parallelism and Scale-Up** section).
4. Confirmation that h2o is acceptable for multi-hour unattended sequential runs.

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

## Versioned Basin Manifests

### 50-basin pilot roles

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

### 2,843-basin initial training list

`config/stage1_initial_training_basin_manifest.csv` is now committed to the repo.
This is **curated selection metadata** — not a generated run artifact. It captures the
finalized Stage 1 initial training basin set as of the pre-training selection step,
independent of any specific run output. The broader generated final-selection reports
(`reports/flashnh_final_basin_selection_v001/`) remain untracked.

| Column | Description |
|---|---|
| `STAID` | Zero-padded to 8 chars (longer IDs preserved as-is) |
| `final_training_status` | `TRAIN_CORE` or `TRAIN_SOFT_KEEP` |

| final_training_status | Count |
|---|---|
| TRAIN_CORE | 2,216 |
| TRAIN_SOFT_KEEP | 627 |
| **Total** | **2,843** |

Sort order: ascending by STAID (deterministic lexicographic; equivalent to ascending numeric for standard 8-char IDs).

Pass this manifest directly to the recovery script via `--staids-file`:

```bash
python scripts/recover_usgs_iv_full_period_hourly.py \
    --staids-file config/stage1_initial_training_basin_manifest.csv \
    --out-dir /data42/omrip/Flash-NH/tmp/stage1_full_2843 \
    --force
```

`--staids-file` accepts any CSV with a `STAID` column (case-insensitive header).
It is **mutually exclusive** with `--staids` — provide one or the other, not both.

---

## Full 2,843-Basin Recovery — Recommended Approach

Use `scripts/launch_usgs_iv_recovery_shards.py` to run the full 2,843-basin recovery
with controlled station-level parallelism. h2o has 128 cores and no SLURM, so the
launcher runs multiple `recover_usgs_iv_full_period_hourly.py` subprocesses in parallel,
each writing to its own shard output directory.

### Why 4 shards for the first run

- 4 concurrent USGS requests is polite and well within NWIS IV rate limits.
- At ~15 s/basin sequential, 4 shards of ~711 basins each run in ~177 min instead of ~12 h.
- Failures are isolated per shard: if one shard fails, the others continue and the
  failed shard can be re-run individually with its own `manifests/shard_XX.csv`.
- After one clean 4-shard run, raising to 8 shards is safe if h2o has no complaints.

### Shard assignment

STAIDs are split into contiguous sequential blocks by row order
(shard_00 = rows 0–710, shard_01 = rows 711–1421, …). Block sizes differ by at most 1.
Round-robin was not used; sequential blocks keep geographically/numerically adjacent
basins together and simplify partial re-runs.

### h2o commands — full 2,843-basin run with screen

Run inside a `screen` session so the launcher survives SSH disconnects:

```bash
cd /data42/omrip/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation
git pull

# Start a named screen session
screen -S flashnh_2843

# Inside screen: first do a dry-run to confirm shard manifests and commands
python scripts/launch_usgs_iv_recovery_shards.py \
    --out-root /data42/omrip/Flash-NH/tmp/stage1_full_2843 \
    --n-shards 4 \
    --dry-run

# If dry-run PASS, launch the real run
python scripts/launch_usgs_iv_recovery_shards.py \
    --out-root /data42/omrip/Flash-NH/tmp/stage1_full_2843 \
    --n-shards 4 \
    --force

# Detach from screen (keep running): Ctrl-A  D
# Reattach later:
screen -r flashnh_2843
```

Expected shard layout under `/data42/omrip/Flash-NH/tmp/stage1_full_2843/`:

```
manifests/
  shard_00.csv  (~711 STAIDs)
  shard_01.csv  (~711 STAIDs)
  shard_02.csv  (~711 STAIDs)
  shard_03.csv  (~710 STAIDs)
shard_00/
  canonical/    (711 × *_hourly.nc)
  raw_cache/
  logs/
    recovery.log       (aggregate stdout from the shard subprocess)
    <staid>_acquire.log  (per-station logs written by recovery script)
shard_01/ ... shard_03/   (same layout)
launcher_summary.json
launcher_summary.md
```

After all 4 shards complete (~3 h at 4× parallel), check the launcher summary:

```bash
cat /data42/omrip/Flash-NH/tmp/stage1_full_2843/launcher_summary.md
```

Expected: `n_pass=4  n_fail=0`.  If any shard failed, re-run it individually:

```bash
python scripts/recover_usgs_iv_full_period_hourly.py \
    --staids-file /data42/omrip/Flash-NH/tmp/stage1_full_2843/manifests/shard_02.csv \
    --out-dir     /data42/omrip/Flash-NH/tmp/stage1_full_2843/shard_02 \
    --force
```

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

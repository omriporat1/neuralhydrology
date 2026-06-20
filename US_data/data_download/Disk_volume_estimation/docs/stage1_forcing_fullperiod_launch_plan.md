# Stage 1 Forcing — Full-Period Launch Plan (Milestone 2K-C)

## Overview

Milestone 2K-C is the full-period extraction of MRMS QPE 1h Pass1 + RTMA CONUS 2.5km forcing for all 2,752 v001 basins over 2020-10-14T00Z – 2025-12-31T23Z (45,720 hours, 63 monthly chunks).

**Completed prerequisites:**
- 2K-A: v001 weight tables built (MRMS 37 MB, RTMA 12 MB)
- 2K-B: 48h × 10-basin smoke PASS (commit 8cf5974); launcher activation verified on h2o

**What is still prohibited until explicitly approved:**
- TB-scale extraction — requires Phase 1 (October 2020) evidence review first
- Changing worker count above 16 without Phase 1 evidence
- Model training — h2o is for download/preprocess/assembly only

---

## Two-Phase Launch Strategy

**Phase 1 — October 2020 (one month):**
Run `run_stage1_forcing_onemonth_h2o.sh` (default configuration).
- 432 scheduled hours × 2,752 basins
- Validates MRMS archive gap handling, RTMA throughput at 16 workers, disk usage rate
- Produces scaling estimate for the remaining 62 months
- Evidence review gates Phase 2

**Phase 2 — Full 63-month run:**
Run `run_stage1_forcing_fullperiod_h2o.sh` only after Phase 1 evidence is reviewed and approved.
- October 2020 is auto-skipped (already complete; `all_pass=true`)
- 62 remaining months run sequentially under one screen session

---

## Pre-Launch Checklist

Before running Phase 1, confirm all of the following on h2o:

```bash
# 1. Weights present and non-empty
ls -lh /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/02_basin_geometries/weights/mrms/v001_2752_mrms_weights.parquet
ls -lh /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/02_basin_geometries/weights/rtma/v001_2752_rtma_weights.parquet

# 2. Basin list present
wc -l /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/v001_basin_list.csv

# 3. Disk headroom (need ≥5 TB free; 5 TB total run estimate)
df -h /data42

# 4. System load is acceptable
uptime    # 1-min load should be <50% of available CPUs

# 5. Conda env is usable
/data42/omrip/Flash-NH/envs/flashnh-stage1/bin/python --version

# 6. Git repo is current
cd /data42/omrip/Flash-NH/repo
git log --oneline -3
git status --short
```

---

## Phase 1 — October 2020 One-Month Run

### MRMS archive gap (expected — not an error)

October 2020 has **36 confirmed permanent S3 gaps** across three clusters:

| Cluster | Hours | Timestamps |
|---|---|---|
| Archive-start gap | 21 h | 2020-10-14T00Z – 20Z (the `noaa-mrms-pds` QPE 1h Pass1 archive begins at 2020-10-14T21Z) |
| Oct 25–26 S3 outage | 14 h | 2020-10-25T23Z; 2020-10-26T00Z–11Z; 2020-10-26T15Z |
| Oct 29 spot gap | 1 h | 2020-10-29T23Z |

All 36 appear as `not_in_s3` in `missing_files.csv`; zero are pipeline failures.
`2020-10_manifest.json` records `all_pass=true`.
The validation check `mrms_202010_gap=36` must be PASS for October 2020.

### Expected output sizes

Observed October 2020 actuals (from 2026-06-18 run, `download_workers=8`):

| Item | Observed value |
|---|---|
| MRMS raw GRIB2 (396 files) | 207 MB |
| RTMA raw GRIB2 (432 files × ~68 MB) | 30.7 GB |
| MRMS output Parquet | 23 MB |
| RTMA output Parquet | 864 MB |
| Combined rows | 14,167,296 (396×2752 MRMS + 432×2752×11 RTMA) |
| Manifests + CSVs | <10 MB |
| Wall clock | 15h 04m 57s at 8 download workers |

A fresh run (no cache) at 16 download workers would take somewhat less wall time.
Full-period throughput estimate: 125.7 s/hr → **66.5 days** at current serial code.
See go/no-go table below; full-period extraction is paused pending 2K-D optimization.

### Dry run first (verify configuration without downloading)

```bash
# On h2o, from the repo root:
DRY_RUN=1 bash scripts/run_stage1_forcing_onemonth_h2o.sh
```

Review the printed output. Confirm:
- Basin count = 2,752
- Hour count = 432
- MRMS gap note for 2020-10 is shown
- All 3 input files found
- Disk headroom is sufficient

### Launch under screen

```bash
# On h2o:
cd /data42/omrip/Flash-NH/repo

screen -S flashnh-202010 bash scripts/run_stage1_forcing_onemonth_h2o.sh

# Detach with: Ctrl-A, D
# Reattach with: screen -r flashnh-202010
```

### Monitoring while running

```bash
# Attach to screen:
screen -r flashnh-202010

# Or watch the live log from another terminal:
tail -f /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/manifests/2020-10_live_run.log

# Progress summary (filter for key lines):
grep -E "(hour|PASS|FAIL|ERROR|RTMA|MRMS)" \
  /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/manifests/2020-10_live_run.log | tail -20

# Live progress JSON:
python -c "
import json
d = json.load(open('/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/manifests/2020-10_live_progress.json'))
print(json.dumps(d, indent=2))
"

# Disk usage:
df -h /data42
du -sh /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/

# System load:
uptime
```

### Stop cleanly between hours

The launcher exits immediately on extractor failure. To stop voluntarily between months
(relevant for the fullperiod loop, not this one-month script):

```bash
# Create the stop file; the fullperiod loop checks it between months:
touch /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/STOP_AFTER_MONTH

# Check if the fullperiod loop has honored it:
tail /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/manifests/fullperiod_run_log.txt
```

For the one-month launcher there is no stop-file mechanism — kill the screen session
or wait for the month to finish.

### Resume an interrupted one-month run

```bash
RESUME=1 bash scripts/run_stage1_forcing_onemonth_h2o.sh
```

The extractor skips any hour that already has a staging Parquet. Resume is safe to run
at any point; it re-processes only incomplete hours.

### Evidence export (after run completes)

**Do not transfer raw GRIB2 or Parquet files.** Only transfer the compact evidence bundle.

```bash
# On h2o — run after extraction completes:
MANIFEST_DIR=/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/manifests
EXPORT_DIR=/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/evidence_exports
mkdir -p "${EXPORT_DIR}"

cd "${MANIFEST_DIR}" && tar czf \
  "${EXPORT_DIR}/stage1_forcing_202010_v001_audit_export.tar.gz" \
  2020-10_manifest.json \
  2020-10_summary.md \
  2020-10_hourly_runtime_and_volume.csv \
  2020-10_scaling_estimates.json \
  2020-10_validation_checks.csv \
  2020-10_live_run.log \
  $(ls 2020-10_missing_files.csv 2>/dev/null) \
  $(ls 2020-10_variable_completeness.csv 2>/dev/null) \
  $(ls 2020-10_basin_completeness.csv 2>/dev/null) \
  run_provenance/2020-10_run_provenance.json \
  run_provenance/2020-10_run_command.txt

# Transfer to local (Windows):
scp flashnh-h2o:"${EXPORT_DIR}/stage1_forcing_202010_v001_audit_export.tar.gz" \
  tmp/stage1_forcing_202010_evidence/
```

The launcher also prints these exact commands at completion.

---

## Phase 1 — Go/No-Go Criteria

Before launching Phase 2 (full 63 months), all of the following must hold
based on evidence transferred locally:

**October 2020 PASS — all gates met as of 2026-06-18 run.**

| Gate | Criterion | October 2020 result |
|---|---|---|
| `manifest_all_pass` | `all_pass: true` in `2020-10_manifest.json` | **PASS** |
| `mrms_hours_ok` | MRMS extracted hours > 0 (actual 396) | **PASS** |
| `rtma_hours_ok` | RTMA extracted hours = 432 | **PASS** |
| `mrms_202010_gap` | `missing_files.csv` has exactly 36 `not_in_s3` rows (3 clusters) | **PASS** (36=36) |
| No unexpected FAIL statuses | `validation_checks.csv` shows 0 FAIL rows | **PASS** |
| Disk headroom | ≥4 TB free after October 2020 data is on disk | Confirmed |
| Scaling estimate | Full 45,720h ≤ 14 days at optimized code | **PASS — 3.04 days projected (3×dw6, commit `a275296`; 2K-D COMPLETE 2026-06-20)** |

**Phase 2 is unblocked.** All Phase 1 go/no-go gates are met. See 2K-D summary in
`docs/FLASHNH_CURRENT_STATE.md` for benchmark details (D1 optimization 24.7×; x3/dw6 outer-parallel
projection 3.035 days). Launch using the outer-parallel strategy described in Phase 2 below.

---

## Phase 2 — Full 63-Month Run

**Mechanism: 3-way outer parallelism (3 concurrent chunk processes × 6 download workers each).**
Validated by Milestone 2K-D benchmark (commit `a275296`): projected 3.035 days effective wall time.

### Outer-parallel launch strategy

Split the 63 monthly chunks into 3 non-overlapping groups (~21 months each), launch each group under
a dedicated `screen` session, and let all three run concurrently.

**Month groups (63 months, 2020-10 through 2025-12):**

| Group | Screen name | Months | Count |
|---|---|---|---|
| A | `flashnh-group-a` | 2020-10 → 2022-06 | 21 months |
| B | `flashnh-group-b` | 2022-07 → 2024-01 | 19 months |
| C | `flashnh-group-c` | 2024-02 → 2025-12 | 23 months |

> Note: October 2020 (Phase 1) is already complete with `all_pass=true`. The fullperiod launcher
> auto-skips it when the valid manifest exists.

**Pre-launch (once, on h2o):**

```bash
cd /data42/omrip/Flash-NH/repo
git pull --ff-only
head -60 scripts/run_stage1_forcing_fullperiod_h2o.sh   # review MONTHS_A/B/C config
```

**Launch all three groups:**

```bash
# Group A
screen -dmS flashnh-group-a bash -c "
  bash scripts/run_stage1_forcing_fullperiod_h2o.sh --group A \
      --download-workers 6 \
      2>&1 | tee /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/logs/group_a.log"

# Group B
screen -dmS flashnh-group-b bash -c "
  bash scripts/run_stage1_forcing_fullperiod_h2o.sh --group B \
      --download-workers 6 \
      2>&1 | tee /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/logs/group_b.log"

# Group C
screen -dmS flashnh-group-c bash -c "
  bash scripts/run_stage1_forcing_fullperiod_h2o.sh --group C \
      --download-workers 6 \
      2>&1 | tee /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/logs/group_c.log"

# Verify all three are running:
screen -ls
```

> **Note:** `run_stage1_forcing_fullperiod_h2o.sh` does not yet have a `--group` flag. If the
> launcher remains a single sequential script, run it three times with different `MONTHS_*`
> environment overrides, or create separate scripts for each group. The outer-parallel launcher
> design is the recommended next implementation milestone before launching Phase 2.

**Monitor:**

```bash
screen -r flashnh-group-a    # Ctrl-A D to detach
htop                          # confirm ~18 S3 connections, CPU ≤ 50%
df -h /data42                 # watch disk headroom
```

### Skip a month during the full run

```bash
# Skip specific months (e.g., if S3 is temporarily unavailable):
SKIP_MONTHS="2021-03 2021-04" bash scripts/run_stage1_forcing_fullperiod_h2o.sh
```

### Worker count policy for full run

**2K-D benchmark recommendation: 6 workers per chunk, 3 concurrent chunks (18 total S3 connections).**

| Configuration | Projected days | Condition |
|---|---|---|
| 1 chunk × dw16 (serial baseline) | 6.29 days | Single session; safe but slow |
| 2 chunks × dw8 (x2 outer-parallel) | 4.06 days (YELLOW) | Insufficient; do not use |
| **3 chunks × dw6 (x2 outer-parallel)** | **3.04 days** | **Recommended — USEFUL GREEN** |
| 4 chunks × dw6 (x4 outer-parallel) | Not benchmarked | Not recommended — S3 risk |

Do not increase to x4 outer-parallelism without a new controlled benchmark. The x3/dw6 result
showed marginal diminishing returns (per-chunk dl_median 43–46 s vs 35 s at single-chunk dw8),
indicating S3 bandwidth sharing. x4 would add risk for minimal gain.

### Quarterly evidence bundle pull

Pull evidence bundles at Q1 (end of 2020), Q2 (end of 2021-03), and then quarterly.
Transfer manifests and CSVs only — no Parquets or GRIB2.

```bash
# On h2o — quarterly bundle (e.g., after 2021-03 completes):
MANIFEST_DIR=/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/manifests
tar czf /tmp/stage1_q1_2021_evidence.tar.gz \
  -C "${MANIFEST_DIR}" \
  2020-10_manifest.json 2020-11_manifest.json 2020-12_manifest.json \
  2021-01_manifest.json 2021-02_manifest.json 2021-03_manifest.json \
  fullperiod_run_log.txt
scp flashnh-h2o:/tmp/stage1_q1_2021_evidence.tar.gz tmp/
```

---

## Storage and Deletion Policy

| Data type | Location | Size estimate | Retain? |
|---|---|---|---|
| Raw MRMS GRIB2 | `raw/mrms/` | ~0.5 TB | Delete after each quarter's Parquets are verified |
| Raw RTMA GRIB2 | `raw/rtma/` | ~3.2 TB | Delete after each quarter's Parquets are verified |
| Staging Parquets (hourly) | `staging/mrms/`, `staging/rtma/` | ~0.5–1 TB | Delete after monthly chunk Parquet verified |
| Monthly chunk Parquets | `chunks/<YYYY-MM>/` | ~200–400 GB | Retain until per-basin NCs assembled (2K-D) |
| Manifests, CSVs, logs | `manifests/` | <1 GB | Retain permanently; commit summaries to repo |
| Per-basin forcing NCs | `stage1_forcing_package_v001/` | ~130 GB | Retain; this is the final deliverable |

**Deletion procedure (raw GRIB2 after quarterly verification):**

```bash
# Verify chunk Parquet and manifest exist with all_pass=true FIRST, then:
# Example: delete Q1 2021 raw MRMS after verifying manifests are PASS
rm -rf /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/raw/mrms/20201[0-2]*
rm -rf /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/raw/rtma/rtma2p5.20201[0-2]*
```

**Never delete:**
- Monthly chunk Parquets until 2K-D assembly is complete and checksummed
- Manifests, CSVs, or logs (these are the compact evidence record)

---

## Git Hygiene

- All launcher scripts and this plan are committed; generated outputs are NOT.
- The `.gitignore` excludes `tmp/`, `raw/`, `staging/`, `chunks/`, `manifests/` output directories.
- After the one-month run, do not `git add` any file from `tmp/` or `FORCING_ROOT`.
- If adding new scripts or docs, commit message should note the milestone: `Add 2K-C one-month launcher and launch plan`.

Check repo cleanliness before any commit:

```bash
git status --short
# Expected: only tracked script/doc/config modifications should appear.
# If tmp/ files appear, check .gitignore.
```

---

## Known Gaps and Non-Errors

| Gap | Expected behavior |
|---|---|
| MRMS 00Z–20Z on 2020-10-14 | `not_in_s3` in `missing_files.csv`; `all_pass=true` unaffected |
| RTMA occasional missing hour | `not_in_s3` in `missing_files.csv`; hourly row absent from chunk Parquet |
| `2sh` / `2d` alternating by hour | Both decoded; staging Parquet contains whichever is present for that hour |
| `tcc` occasionally absent | Row absent for that variable-hour; not an error |

MRMS coverage from 2020-10-14T21Z onward is nearly complete; gaps after this date (if any) are true S3 outages, not archive limits, and should be investigated if they appear.

---

## Relevant Scripts

| Script | Purpose |
|---|---|
| `scripts/run_stage1_forcing_onemonth_h2o.sh` | Phase 1 one-month launcher (default: Oct 2020) |
| `scripts/run_stage1_forcing_fullperiod_h2o.sh` | Phase 2 full 63-month loop (sequential; extend for --group) |
| `scripts/run_stage1_forcing_smoke_h2o.sh` | 2K-B smoke test (48h × 10 basins; already PASS) |
| `scripts/extract_stage1_forcing_chunk.py` | Core extractor (called by all launchers) |
| `scripts/build_stage1_basin_weights.py` | Weight builder (2K-A; already complete) |
| `scripts/verify_stage1_forcing_inputs_h2o.sh` | Input file preflight checker |
| `scripts/bench_stage1_outer_parallel_h2o.sh` | 2K-D outer-parallel benchmark (x3/dw6; COMPLETE) |
| `scripts/summarize_stage1_outer_parallel.py` | 2K-D benchmark summary + projection (CLI: bench_base path) |

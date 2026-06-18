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

The `noaa-mrms-pds` S3 bucket for `MultiSensor_QPE_01H_Pass1` starts at **2020-10-14T21:00:00Z**.
Hours 00Z–20Z on 2020-10-14 (21 hours) have no S3 object and will appear as `not_in_s3` in `missing_files.csv`.

This is a permanent upstream gap. It will show in `missing_files.csv` and is recorded
in `2020-10_manifest.json` with `all_pass=true` because the pipeline correctly handles missing S3 objects.
The validation check `mrms_202010_gap=21` must be PASS for October 2020.

### Expected output sizes

| File | Expected size |
|---|---|
| Raw MRMS GRIB2 (411 files × ~10 MB) | ~4 GB |
| Raw RTMA GRIB2 (432 files × ~71 MB) | ~31 GB |
| Staging Parquets (hourly, all basins) | ~5–15 GB |
| Chunk Parquet `combined_2020-10.parquet` | ~3–8 GB |
| Manifests + CSVs | <10 MB |
| **Total October 2020** | **~40–55 GB** |

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
  2020-10_summary.json \
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

| Gate | Criterion |
|---|---|
| `manifest_all_pass` | `all_pass: true` in `2020-10_manifest.json` |
| `mrms_hours_ok` | MRMS extracted hours > 0 (expect 411) |
| `rtma_hours_ok` | RTMA extracted hours = 432 |
| `mrms_202010_gap` | `missing_files.csv` has exactly 21 `not_in_s3` rows (hours 00Z–20Z on 2020-10-14) |
| Scaling estimate | RTMA full 45,720h estimate ≤ 48h (2 days) at 16 workers |
| Disk headroom | ≥4 TB free after October 2020 data is on disk |
| No unexpected FAIL statuses | `validation_checks.csv` shows 0 FAIL rows (WARN for gap is OK) |

If scaling estimate > 48h, discuss increasing workers to 32 before Phase 2.

---

## Phase 2 — Full 63-Month Run

After Phase 1 evidence is approved:

```bash
# On h2o, pull latest commits first:
cd /data42/omrip/Flash-NH/repo
git pull

# Dry-run the fullperiod launcher to confirm skip behavior:
# (No DRY_RUN mode in fullperiod launcher — just review the config section)
head -60 scripts/run_stage1_forcing_fullperiod_h2o.sh

# Launch under screen:
screen -S flashnh-forcing bash scripts/run_stage1_forcing_fullperiod_h2o.sh
```

The fullperiod launcher:
- Auto-skips October 2020 if `all_pass=true` manifest exists
- Runs months sequentially; does NOT abort on a single month's failure
- Checks for `STOP_AFTER_MONTH` stop-file between months
- Logs each month outcome to `manifests/fullperiod_run_log.txt`
- Checks disk every 12 months

### Skip a month during the full run

```bash
# Skip specific months (e.g., if S3 is temporarily unavailable):
SKIP_MONTHS="2021-03 2021-04" bash scripts/run_stage1_forcing_fullperiod_h2o.sh
```

### Worker count policy for full run

| Worker count | Condition required |
|---|---|
| 16 (default) | Phase 1 evidence PASS; always safe to start here |
| 32 | Phase 1 evidence shows < 50% CPU load at 16 workers AND scaling estimate > 36h total |
| 64+ | Requires explicit PI approval |

To increase workers for the fullperiod run:

```bash
DOWNLOAD_WORKERS=32 bash scripts/run_stage1_forcing_fullperiod_h2o.sh
```

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
| `scripts/run_stage1_forcing_fullperiod_h2o.sh` | Phase 2 full 63-month loop |
| `scripts/run_stage1_forcing_smoke_h2o.sh` | 2K-B smoke test (48h × 10 basins; already PASS) |
| `scripts/extract_stage1_forcing_chunk.py` | Core extractor (called by all launchers) |
| `scripts/build_stage1_basin_weights.py` | Weight builder (2K-A; already complete) |
| `scripts/verify_stage1_forcing_inputs_h2o.sh` | Input file preflight checker |

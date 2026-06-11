# Flash-NH Stage 1 — Streamflow Recovery for 22 Missing-CAMELSH Basins

## Background

The Flash-NH Stage 1 pilot uses 50 basins selected from the GAGES-II catalog with a seeded random
draw (seed=42, January 2023 pilot window). Of those 50, only 28 have local CAMELSH hourly
streamflow NetCDF files; the remaining 22 are absent from the local CAMELSH hourly download
(5 767 files covering the broader dataset).

This document describes why those 22 basins are missing, how their availability was verified, and
how January 2023 streamflow was recovered for the pilot from USGS NWIS Instantaneous Values.

---

## Why 22 Basins Are Missing from the Local CAMELSH Dataset

The local CAMELSH hourly dataset (`CAMELSH_resolution_test/data/raw/camelsh/`) is a subset of the
full 9 008-basin CAMELSH polygon catalog. The 22 missing pilot basins are not absent because of a
download failure; they simply are not included in the hourly streamflow subset that was downloaded
(CAMELSH hourly v7, DOI 10.5281/zenodo.16763144).

Verified by Milestone 2H (2026-06-10):

| Check | Result |
|---|---|
| All 22 STAIDs in GAGES-II BasinID catalog | PASS (22/22) |
| All 22 STAIDs in CAMELSH polygon shapefile (9 008 basins) | PASS (22/22) |
| All 22 STAIDs in 2G static attributes file | PASS (22/22) |
| Any of 22 STAIDs in local CAMELSH hourly dir | NONE (0/22) |
| Any of 22 STAIDs in alternate CAMELSH paths | NONE (0/22) |
| All 22 STAIDs available via USGS NWIS IV (00060) | PASS (22/22) |

The 22 STAIDs are candidates for direct USGS IV recovery rather than sourcing from CAMELSH.

---

## USGS NWIS Instantaneous Values (IV)

Source: [https://waterservices.usgs.gov/nwis/iv/](https://waterservices.usgs.gov/nwis/iv/)  
Parameter code: `00060` (Discharge, cubic feet per second)

Unit conversion applied in all recovery scripts:

```
m³/s = ft³/s × 0.028316846592
```

Most stations report at 15-minute cadence. Five stations report at 5 minutes:
`01585200`, `02072500`, `02146381`, `03298135`, `07103700`.

Three stations have shorter USGS record starts:
- `02344605`: record starts 2008-06-28
- `03305000`: record starts 2010-09-28
- `10164500`: record starts 2011-06-03

All three still cover the full Flash-NH research period (2020-10-14 onward).

---

## Timestamp Policy (provisional)

The following policy is applied when converting irregular IV observations to a uniform UTC hourly
grid. It is labelled **provisional** in the NetCDF attributes pending formal project acceptance.

```
For each target UTC hour T in the pilot window:
  1. If an observation exists at exactly T  ->  use it.           (method: exact)
  2. Else if an observation exists within T-15min to T+15min
     ->  use the nearest observation.                             (method: nearest_within_15min)
  3. Else  ->  NaN.                                               (method: missing)

NO interpolation. NO hourly mean. NO fill from adjacent hours.
```

Missing values are preserved as `NaN` in the output NetCDF and are clearly visible in the
per-station debug CSV (`{STAID}_hourly_assignment_debug.csv`) under `assignment_method=missing`.

---

## Milestone 2H — Discovery (2026-06-10)

Script: `scripts/audit_stage1_streamflow_recovery_discovery.py`

Produced a 22-row discovery table
(`tmp/stage1_pilot_dryrun/13_streamflow_recovery_discovery/tables/streamflow_recovery_discovery_22.csv`)
confirming:

- All 22 in GAGES-II, polygon catalog, static attributes
- None in local CAMELSH hourly download
- All 22 have USGS IV 00060 available for January 2023
- January 2023 coverage: 19/22 = 744 h full; 02344700 = 743 h; 10164500 = 726 h; 10336700 = 720 h
- Recommended action for all 22: `RECOVER_FROM_USGS_IV_LATER`

The discovery table recorded `iv_end = 2025-09-05` for all stations, sourced from the local
`iv_scan_results.csv`. That date was the date the local scan was run, **not** a USGS data end date.

---

## Milestone 2H-B Part A — Late-2025 Scan Correction (2026-06-10)

Script: `scripts/recover_stage1_usgs_iv_streamflow_pilot.py` (Part A)

Queried USGS IV directly for two windows per station:
- WY2025 tail: 2025-09-06 to 2025-09-30
- Late cal-2025: 2025-10-01 to 2025-12-31

**Result: The 2025-09-05 end date in `iv_scan_results.csv` was a stale-scan artefact.**

| Interpretation | Count | STAIDs |
|---|---|---|
| `SOURCE_DATA_AVAILABLE_LOCAL_SCAN_STALE` | 21 | All except 03298135 |
| `SOURCE_DATA_AVAILABLE_POSSIBLE_LATE_2025_GAP` | 1 | 03298135 |

For 21 stations, late-2025 data was confirmed through 2025-12-31 or later (most to 2026-01-01 UTC).
Correct availability status: **`CONFIRMED_FULL_2020_2025`** for those 21.

### 03298135 caveat

STAID 03298135 (Chenoweth Run at Louisville, KY, TRAIN_SOFT_KEEP) returned late-2025 data, but its
last observation in the Oct–Dec 2025 window is `2025-11-24T15:55:00Z` — more than 14 days before
2025-12-31. This is either a temporary station outage or a data-upload delay. The station is NOT
excluded from recovery, but a full-period gap-accounting step is required before it can be included
in an HPC-scale 2020–2025 training build.

---

## Milestone 2H-B Part B — January 2023 Recovery Pilot (2026-06-10)

Script: `scripts/recover_stage1_usgs_iv_streamflow_pilot.py` (Part B)

Three non-EXCLUDE_QC pilot basins were selected to validate the recovery workflow:

| STAID | Pilot Role | Cadence | January 2023 result |
|---|---|---|---|
| 01585200 | TRAIN_CORE | 5 min | 743/744 h valid; 1 NaN (first hour — USGS starts at 01:00 UTC) |
| 02344700 | TRAIN_SOFT_KEEP | 15 min | 742/744 h valid; 2 NaN (first hour + 2023-01-17 10:00 UTC) |
| 10164500 | TRAIN_SOFT_KEEP | 15 min | 725/744 h valid; 19 NaN (first hour + mid-month + 14 h end-of-Jan real gap) |

EXCLUDE_QC basins are never included in recovery:

```python
EXCLUDE_QC_STAIDS = {"10336700"}
```

### NetCDF output format

Each recovered file follows the CAMELSH-like convention:

| Field | Value |
|---|---|
| Variable | `streamflow` (float64) |
| Units | `m3 s-1` |
| Coordinate | `time` (UTC naive datetime64 in file; timezone noted in attrs) |
| Timestamps | 744 hourly from 2023-01-01T00:00 to 2023-01-31T23:00 UTC |
| Missing values | `NaN` (no sentinels, no fill) |
| CF conventions | CF-1.8 |

### Validation (all 3 files PASS)

- 744 timestamps each, monotonically increasing, uniformly hourly
- Variable `streamflow` present with units `m3 s-1`
- No sentinel values (< -999990)
- No negative values
- No interpolated values

---

## Outputs

All recovery outputs are written under `tmp/` and are **not committed to the repository**.

```
tmp/stage1_pilot_dryrun/
  13_streamflow_recovery_discovery/
    tables/streamflow_recovery_discovery_22.csv     # 22-row 2H discovery table
    summary.md / summary.json
    provenance/run_provenance.json

  14_streamflow_recovery_pilot/
    recovered_camelsh_like/
      01585200_hourly.nc
      02344700_hourly.nc
      10164500_hourly.nc
    tables/
      usgs_late_2025_availability_check.csv         # 22-row Part A availability
      usgs_iv_hourly_recovery_audit.csv             # 3-row Part B audit
      {STAID}_hourly_assignment_debug.csv           # per-hour snap debug (3 files)
    qc/
      recovered_streamflow_pilot_hydrographs.png    # visual QC hydrograph
    summary.md / summary.json
    provenance/run_provenance.json
```

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/audit_stage1_streamflow_recovery_discovery.py` | Milestone 2H discovery/audit |
| `scripts/recover_stage1_usgs_iv_streamflow_pilot.py` | Milestone 2H-B Part A + Part B |

To re-run the full combined 2H-B workflow:

```bash
python scripts/recover_stage1_usgs_iv_streamflow_pilot.py --force
```

---

## Milestone 2H-C — Full January 2023 Recovery (2026-06-10)

Script: `scripts/recover_stage1_usgs_iv_streamflow_january.py`

Extended recovery to all 21 eligible basins (22 missing − 1 EXCLUDE_QC: 10336700).
All 21 passed validation: 744 timestamps each, no sentinels, no negatives.

Output: `tmp/stage1_pilot_dryrun/15_streamflow_recovery_january_eligible/recovered_camelsh_like/`
(21 NC files, 23 KB each).

Coverage gain across 50-basin pilot: **+15,583 valid hours**
(20,576 → 36,159 valid qobs hours out of 37,200 total).

---

## Milestone 2H-D — Package Rebuild with Recovery (2026-06-10)

Script: `scripts/build_stage1_neuralhydrology_january_with_recovery.py` (commit `0595384`)

Built new NeuralHydrology-compatible January 2023 package under
`tmp/stage1_pilot_dryrun/16_neuralhydrology_january_with_recovery/`.
Reads forcing from the 2G package NC files; replaces only `qobs_m3s`.

Streamflow sources in the rebuilt package:

| Source label | Count | Description |
|---|---|---|
| `local_CAMELSH` | 24 | 24 TRAIN/HOLDOUT_QC basins with original local CAMELSH qobs |
| `USGS_IV_recovered` | 21 | 21 basins recovered via 2H-C; includes 4 HOLDOUT_QC |
| `EXCLUDE_QC_local_CAMELSH` | 4 | EXCLUDE_QC basins with local CAMELSH data; QC lineage only |
| `EXCLUDE_QC_missing` | 1 | 10336700; qobs all-NaN; never recovered |

All 5 EXCLUDE_QC basins are excluded from all training and evaluation lists.

---

## Full-Period Target Source Design

For the **full-period research window (2020-10-14 through 2025-12-31)**, USGS NWIS IV
(parameter code 00060) is the canonical and sole streamflow target source for Flash-NH
Stage 1. The CAMELSH-based recovery documented above applies to the January 2023 pilot only.

Detailed specification for the full-period USGS IV acquisition strategy is in:

**[`docs/stage1_usgs_iv_full_period_target_plan.md`](stage1_usgs_iv_full_period_target_plan.md)**

That document covers:
- Target-source policy (API endpoint, parameter, units, timestamp policy)
- Full-period window: 1,905 days = 45,720 hourly steps
- Proposed output layout and canonical NetCDF schema
- Acquisition strategy (station × water-year chunks, rate limiting, resume)
- Gap and quality audit design
- HPC / SLURM job array design
- Storage and runtime estimates anchored to pilot calibration data
- Acceptance criteria for next pilot milestone (2I-B)
- Open questions (snap tolerance, qualifier handling, raw cache retention)

**03298135 late-2025 caveat** applies to the full-period build; that document
contains the specific flag requirement in the gap audit design.

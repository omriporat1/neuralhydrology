# Stage 1 Forcing — Full-Period Visual QC Animation Plan

**Created:** 2026-06-25  
**Status:** Scripts ready; pending h2o run  
**Depends on:** `docs/stage1_forcing_fullperiod_visual_qc_selection.md`

---

## Overview

This document describes the animation generation pipeline for the Stage 1
full-period MRMS+RTMA forcing visual QC gate. Animations are generated on h2o
from the full-period combined Parquet files and v001 target package NetCDF files.
They are **not committed to git** and must stay under `tmp/`.

---

## Why Animations (Not Static Maps)

The January 2023 pilot animations (`scripts/generate_january_event_animations.py`)
used raw GRIB2 spatial maps and local CAMELSH shapefiles. For the full-period run:

- Raw GRIB2 files are **deleted from h2o** after extraction to save disk space.
- Spatial weights survive (per-basin CSV), but there is no per-pixel hourly raster.
- Basin-average time series (combined Parquet) are the only forcing data available
  for arbitrary basins and months.

The full-period animations therefore use a **3-panel time-series figure** with
no spatial map panel. This is sufficient for the QC goals:

| Goal | Covered by |
|---|---|
| NaN / gap rendering at gap hours | MRMS bars: gray placeholder at gap timesteps |
| Period-boundary clip (VQC-001) | Explicit red shading + text annotation |
| Precipitation magnitude plausibility | MRMS QPE bars (basin mean mm/h) |
| Temperature context (winter, monsoon) | RTMA 2m temperature line |
| Streamflow response to event | qobs_m3s hydrograph |
| Animation cursor validation | Red vertical line sweeps across all panels |

---

## Script and Launcher

| File | Purpose |
|---|---|
| `scripts/generate_fullperiod_visual_qc_animations.py` | Main animation script (CLI, dry-run, h2o render) |
| `scripts/run_fullperiod_visual_qc_pilot_h2o.sh` | h2o launcher: env activation, preflight, log |

### Key CLI arguments

```
--case-selection-csv   visual_qc_case_selection.csv (required)
--case-ids             VQC-001 VQC-004 ...   (default: the 6 pilot cases)
--forcing-root         /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod
--target-package-root  /data42/omrip/Flash-NH/tmp/stage1_target_package_v001
--out-dir              tmp/stage1_forcing_fullperiod_visual_qc_pilot_<TIMESTAMP>
--max-cases            6   (safety cap)
--dry-run              Validate paths, print expected files — no data loaded
--check-data           With --dry-run: verify h2o paths exist on disk
--format               gif (default) or mp4 (requires ffmpeg)
--fps                  4 (default)
--dpi                  90 (default)
--overwrite            Re-animate existing output files
```

### Dry-run (local validation, animation script only)

```bash
python scripts/generate_fullperiod_visual_qc_animations.py \
    --case-selection-csv tmp/stage1_forcing_fullperiod_visual_qc_selection_20260625T081859Z/visual_qc_case_selection.csv \
    --case-ids VQC-001 VQC-004 VQC-007 VQC-009 VQC-012 VQC-020 \
    --forcing-root /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod \
    --target-package-root /data42/omrip/Flash-NH/tmp/stage1_target_package_v001 \
    --out-dir tmp/stage1_forcing_fullperiod_visual_qc_pilot_dryrun \
    --dry-run
```

### h2o run — Mode A: default (regenerate VQC CSV on h2o)

The launcher regenerates the case-selection CSV from the audit tables and
metadata that live on h2o, then feeds it to the animation script. No manual
CSV transfer is needed if the two required h2o-local inputs are present:

| Input | Expected h2o path | Tracked by git? |
|---|---|---|
| Forcing audit tables | `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod_postrun_audit_20260624T060504Z/` | No — audit run output |
| Basin metadata CSV | `${REPO_ROOT}/reports/flashnh_usgs_site_metadata_v001/tables/wy2024_metrics_with_site_metadata.csv` | No — untracked in repo |

If either is missing, the launcher stops with exact `rsync` copy instructions.

```bash
# Typical h2o default run
bash scripts/run_fullperiod_visual_qc_pilot_h2o.sh

# Override audit-dir if the timestamp differs
bash scripts/run_fullperiod_visual_qc_pilot_h2o.sh \
    --audit-dir /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod_postrun_audit_YYYYMMDDTHHMMSSZ

# Override meta-csv path
bash scripts/run_fullperiod_visual_qc_pilot_h2o.sh \
    --meta-csv /path/to/wy2024_metrics_with_site_metadata.csv
```

### h2o run — Mode B: override with existing VQC CSV (--vqc-csv)

Use `--vqc-csv` to skip CSV regeneration entirely. This is the right choice
when you have already transferred or generated the CSV on h2o and do not want
to re-run case selection.

```bash
# Transfer the CSV from local first (run from local machine)
rsync -av --progress \
    tmp/stage1_forcing_fullperiod_visual_qc_selection_20260625T081859Z/ \
    h2o:/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod_visual_qc_selection_20260625T081859Z/

# Then run with override (run on h2o)
bash scripts/run_fullperiod_visual_qc_pilot_h2o.sh \
    --vqc-csv /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod_visual_qc_selection_20260625T081859Z/visual_qc_case_selection.csv
```

---

## Data Sources (h2o)

### Forcing Parquets

```
/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/
  chunks/{YYYY-MM}/combined_{YYYY-MM}.parquet   ← all basins × all vars
```

Relevant columns used:

| Column | Role |
|---|---|
| `STAID` | Basin filter (string; no zero-padding) |
| `product` | `mrms_qpe_1h_pass1` or `rtma_conus_aws_2p5km` |
| `variable` | `2t` for RTMA 2m temperature |
| `valid_time_utc` | Hourly timestamp |
| `weighted_mean` | Basin-average value (QPE in mm; temperature in K → converted to °C) |

Loading uses `pd.read_parquet(columns=[...], filters=[("STAID", "==", staid)])` for
efficiency. Predicate pushdown reduces column width; full rows for other basins are
still read at the row-group level but not materialized in Python.

### Target Package NetCDF

```
/data42/omrip/Flash-NH/tmp/stage1_target_package_v001/
  time_series/{STAID}.nc   ← variable: qobs_m3s, coordinate: date (hourly UTC)
```

Streamflow is loaded with `xarray.open_dataset()` and sliced to the render window.

---

## Pilot Cases (6 of 21)

| Case | STAID | Category | Render start | Hours | Clipped |
|---|---|---|---|---|---|
| VQC-001 | 01170100 | MRMS_GAP_ADJACENT | 2020-10-14T00Z | 34 | **YES** |
| VQC-004 | 01440000 | RTMA_GAP_ADJACENT | 2020-11-12T08Z | 72 | no |
| VQC-007 | 10343500 | WINTER_MIXED_PRECIP | 2024-01-21T21Z | 72 | no |
| VQC-009 | 09484000 | WARM_SEASON_CONVECTIVE | 2023-08-27T16Z | 72 | no |
| VQC-012 | 05540275 | SMALL_FLASHY_BASIN | 2022-03-12T17Z | 72 | no |
| VQC-020 | 03021350 | RANDOM_CONTROL | 2025-03-01T12Z | 72 | no |

VQC-001 is a **boundary-stress case**: its render window starts at the forcing
period start (2020-10-14T00:00:00Z, clipped from the nominal 2020-10-12T10:00:00Z).
MRMS T00Z–T20Z (21 of 34 hours) are permanently absent from the S3 archive.

---

## Figure Layout (3-Panel Time Series)

```
┌─────────────────────────────────────────────────────────────────┐
│ TITLE: VQC-xxx | STAID | basin_name | category                 │
│        render window | [CLIPPED note if VQC-001]               │
├─────────────────────────────────────────────────────────────────┤
│ Panel 1 (40%): MRMS 1h QPE bar chart (basin mean, mm)          │
│   Gap hours → gray placeholder bars                             │
│   VQC-001 → red shaded region T00Z–T20Z with label             │
│   VQC-001 → dashed vertical line at period start               │
├─────────────────────────────────────────────────────────────────┤
│ Panel 2 (20%): RTMA 2m temperature (°C), line                 │
│   0°C dashed reference line                                     │
│   Gap hours → orange shaded spans                               │
├─────────────────────────────────────────────────────────────────┤
│ Panel 3 (40%): Streamflow qobs_m3s, line + fill                │
│   NaN hours → orange shaded spans                               │
└─────────────────────────────────────────────────────────────────┘
Per-frame: red vertical cursor sweeps right. Frame label shows
"YYYY-MM-DD HH:MM UTC  frame N/N" in Panel 1 upper right.
```

---

## Output Files Per Case

```
{out_dir}/
  {CASE_ID}/
    {CASE_ID}_animation.gif     ← per-frame animated GIF (or .mp4 if ffmpeg present)
    {CASE_ID}_quicklook.png     ← static overview (no cursor)
  animation_manifest.csv        ← per-case status, paths, gap counts
  animation_summary.md          ← human-readable review guide
  run.log                       ← full launcher log (from bash script)
```

---

## VQC-001 Special Annotations

Because VQC-001 is the only boundary case and the primary test of NaN rendering:

1. MRMS Panel: Red-shaded region from T00Z to T20Z with text label
   "MRMS archive-start gap (T00Z–T20Z, 21h)"
2. All panels: Dashed vertical line at render_window_start_utc (period start)
   labeled "Render start (period start, clipped)"
3. Frame labels clearly show that frames 1–21 are within the NaN span.

---

## Reviewer Instructions

For each case, examine:

1. **MRMS QPE bars**: Is precipitation present where expected for this
   category (MRMS_GAP_ADJACENT = some NaN, WARM_SEASON_CONVECTIVE = rain burst,
   WINTER_MIXED_PRECIP = sustained precip)? Are NaN hours clearly rendered?
2. **RTMA 2m temperature**: Plausible for month and region? Below 0°C for
   winter/high-altitude cases?
3. **Streamflow**: Does the hydrograph respond to the QPE event? Any
   implausible negative values (should be NaN or zero)?
4. **Gap frames**: NaN rendered as gray MRMS bars or orange shading (not
   as phantom precipitation or phantom temperature)?
5. **VQC-001 only**: Archive-start gap correctly shows 21 NaN frames with
   red shading annotation? Cursor moves through the gap smoothly?

Record outcomes in `visual_qc_case_selection.csv`:

| Column | Example values |
|---|---|
| `reviewer` | your initials |
| `review_outcome` | PASS / PASS_WITH_CAVEATS / FAIL / NEEDS_RECHECK |
| `notes` | free text; describe any anomalies |

Return the filled CSV to the project directory. The QC gate closes when
all 21 cases reach PASS or PASS_WITH_CAVEATS.

---

## Limitations

- Basin-average QPE is a scalar per timestep — no spatial pattern visible.
  High-intensity convective cores may be smoothed out for larger basins.
- RTMA temperature only; wind, humidity, and surface pressure are not shown.
  These are available in the Parquet but omitted from the pilot animation.
- If qobs is missing for a case (target NC absent), the streamflow panel
  will show all-NaN orange shading — this is flagged in the manifest.

---

*Animations are not committed to git and must stay under `tmp/`.*  
*Tracked files: `scripts/generate_fullperiod_visual_qc_animations.py`,*  
*`scripts/run_fullperiod_visual_qc_pilot_h2o.sh`, this doc.*
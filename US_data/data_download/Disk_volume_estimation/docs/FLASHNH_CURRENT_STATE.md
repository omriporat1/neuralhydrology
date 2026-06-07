# Flash-NH Current State

Last updated: 2026-06-07

## Current milestone

Stage 1 Milestone 2E — event animation pipeline, cleanup complete.

## What just happened (2026-06-05 to 2026-06-07)

Pilot animations (R02, R06, R09, R11) generated and approved in v2.1-stable design.
Pipeline cleanup completed:

**Scripts written (stable, no version numbers):**
- `scripts/generate_january_event_animations.py` — main animation generator
- `scripts/audit_rtma_spatial_alignment.py` — RTMA spatial audit gate
- `scripts/audit_january_event_animation_sync.py` — MRMS sync audit gate

**Scripts deleted (superseded or experimental):**
- `generate_pilot_animations.py` (v1, MRMS orientation bug)
- `generate_pilot_animations_v2.py` (v2, superseded by v2.1)
- `generate_pilot_animations_v2_1.py` (versioned, replaced by stable name)
- `rtma_spatial_audit_v2_1.py` (versioned, replaced by stable name)
- `sync_audit_v2_1.py` (versioned, replaced by stable name)
- `sync_audit_pilot.py` (superseded)
- `make_january_event_animations.py` (old skeleton)
- `_probe_rtma.py`, `_sanity_mrms.py`, `_verify_r03.py` (exploratory)

**Audits confirmed (stable scripts):**
- RTMA spatial audit: 8/8 PASS, 0.0000% diff (2t, 10u, 10v)
- MRMS sync audit: 10/10 PASS, 0.0000% diff

**Key v2.1 design fixes over v1:**
- MRMS lat DECREASES with row (was wrong in v1; caused 0.0 basin means everywhere)
- Tighter per-candidate map extents (R02 0.10°, R06 0.35°, R09 0.22°, R11 0.30°)
- Gauge marker at z=12, s=240 (always visible over polygon)
- Basin-mean label at polygon centroid, not top edge
- RTMA 10m wind quiver (~10 arrows, qualitative context only)
- Static PNG frames: first/peak_precip/peak_flow/recession per candidate

## Completed state

January 2023 pilot extraction completed for 50 basins:
- MRMS: 744/744 hours, 37,200 rows
- RTMA: 744/744 hours, 409,200 rows
- Combined: 446,400 rows
- valid_weight_fraction = 1.0
- value diagnostics plausible

Streamflow: CAMELSH hourly NetCDF, 28/50 pilot basins have January 2023 data.

Refined event candidates: R01–R12 (R03 usable-with-gap).
Pilot animations: R02, R06, R09, R11 — reviewed and approved.

## Current cautions

- Do not generate all 12 animations until explicitly instructed (pilot approved, awaiting all-12 go-ahead).
- Do not move to model training yet.
- Do not commit generated MP4/GIF/PNG/Parquet/GRIB/NetCDF/log outputs.
- Keep local-to-HPC transition in mind.
- RTMA 10m wind vectors are qualitative context only — not storm-steering validation.

## All-12 command (do not run yet; document only)

```bash
python scripts/audit_rtma_spatial_alignment.py        # must PASS first
python scripts/audit_january_event_animation_sync.py  # must PASS first
python scripts/generate_january_event_animations.py --all
```

Output: `tmp/stage1_pilot_dryrun/10_animations/stage1_pilot/pilot/`
Estimated runtime: ~27 min for all 12 (local, GIF mode, no ffmpeg).
Install ffmpeg for MP4: `winget install Gyan.FFmpeg`

## Next likely steps

1. Approve all-12 animation run (user decision).
2. Generate all 12 animations.
3. Commit stable 2E source/docs (scripts + docs only; no generated outputs).
4. Plan 2F/2G: event QC conclusions and first model-ready data checks.
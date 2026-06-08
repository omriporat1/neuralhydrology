# Flash-NH Current State

Last updated: 2026-06-08

## Current milestone

Stage 1 Milestone 2E complete. RTMA/URMA-family precipitation diagnostic complete.
Next stage: Stage 1 consolidation / model-ready preflight.

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

## Next stage: Stage 1 consolidation / model-ready preflight

Tasks to plan (not yet started; user approval required):

1. Decide on all-12 animation run.
2. Event QC conclusions: finalize which of R01–R12 are included in Stage 1 training.
3. Model-ready data preflight: verify combined parquet schema, feature ranges,
   missing-value fractions, and temporal coverage.
4. HPC transfer planning.
5. Stage 1 model configuration and first training run.

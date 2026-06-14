# Flash-NH Current State

Last updated: 2026-06-14

## Current milestone

Stage 1 full 2,843-basin USGS IV target acquisition structurally complete (2026-06-13).
**Next: Stage 1 target-policy and NeuralHydrology full-period package build.**

See `docs/stage1_hpc_transition_preflight.md` for the full audit summary and
target-policy recommendations.

### Quick summary

- 2,843 canonical hourly NetCDF files on h2o at `/data42/omrip/Flash-NH/tmp/stage1_full_2843/`
- Coverage 0.9652 overall; 2,754 basins with `historical_training_utility_flag=True`
- 89 basins with late-period gaps (`TARGET_OPERATIONAL_REVIEW`) — hold out of first package
- 18 basins with negative qobs — set to NaN during package build
- `TARGET_QUALITY_REVIEW` (1,375 basins): eligible for training; spike flag is advisory only
- No systematic offset issues (0 basins)

### Immediate next steps

The following can proceed **without h2o** and without heavy execution:

1. Target-policy configuration design (basin inclusion rules, negative-qobs handling).
2. Package-builder script design for the full-period NeuralHydrology package.

The following are **blocked** pending h2o operations preflight
(`docs/stage1_h2o_operations_preflight.md`). Partial answers received (2026-06-14):
`/data42` is not backed up; ~20 TB overall volume target; `/data42/hydrolab/Data`
is the likely shared data root. Scheduler policy, CPU limits, GPU, and promotion
policy remain open.

3. Large spatial-data bulk downloads (MRMS, RTMA, URMA, NWM) — gate G1 BLOCKED.
4. Basin-average preprocessing — gate G2 BLOCKED.
5. NeuralHydrology training — gate G3 BLOCKED.
6. Promotion of curated data to shared lab storage — gate G4 BLOCKED.

**Do not initiate heavy h2o workloads until the operations preflight gates are cleared.**

---

## Milestone 2G — NeuralHydrology NetCDF builder + preflight auditor (completed 2026-06-09)

NeuralHydrology-compatible January 2023 pilot package built and audited.
Full documentation: `docs/stage1_neuralhydrology_preflight.md`

**Scripts:**
- `scripts/build_stage1_neuralhydrology_january_pilot.py` — builder (~8s)
- `scripts/audit_stage1_neuralhydrology_january_pilot.py` — auditor (~20s)

**Package:** `tmp/stage1_pilot_dryrun/12_neuralhydrology_january_pilot_dataset/package/` (gitignored)

**Audit result:** PASS — 0 errors, 1 warning

**Package summary:**
- 50 per-basin NetCDF files; `date` coordinate; 744 hourly UTC steps; January 2023
- 11 variables per basin (10 dynamic forcings + `qobs_m3s` target)
- Smoke dynamic inputs: `mrms_qpe_1h_mm`, `rtma_2t_K`, `rtma_2d_K`, `rtma_2sh_kgkg`, `rtma_10u_ms`, `rtma_10v_ms`
- `attributes_full.csv`: 50 rows × 238 cols (237 attribute cols + `gauge_id`)
  - Manifest records 237, counting only attribute cols; both are correct
- Full HydroATLAS integration: 50/50 pilot match; 193 new columns
- Streamflow: 20 full, 8 partial, 22 all-NaN (CAMELSH files missing locally)
- Audit warning (expected, S2): nulls in `max_abs_hourly_jump_over_Q50` (1), `q95_q50_ratio` (1), `wet_cl_smj` (14) — NaN preserved, no imputation

**No model training run. No generated files committed.**

---

## Milestone 2F — NeuralHydrology package design (completed 2026-06-08)

Design and decision documentation for the NeuralHydrology package.
Full documentation: `tmp/stage1_pilot_dryrun/12_neuralhydrology_january_pilot_dataset/design/`

Key decisions: V1 (both rtma_2d_K and rtma_2sh_kgkg in smoke), V2 (rtma_sp_Pa in wide only),
V3 (rtma_tcc_pct in wide only), S1 (22 missing CAMELSH → all-NaN qobs_m3s, 2H blocker),
S2 (preserve NaN, no imputation), S3 (full HydroATLAS 50/50), S4 (seed=42, streamflow-only split).

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

## Next: Milestone 2H — Streamflow recovery for 22 missing CAMELSH basins

The 22 basins with all-NaN `qobs_m3s` must be recovered before:
- Serious NeuralHydrology training runs
- Scientific performance claims
- HPC-scale 2020–2025 packaging

Recovery plan: `tmp/stage1_pilot_dryrun/12_neuralhydrology_january_pilot_dataset/design/streamflow_recovery_plan.md`

**Pending tasks (require user approval before starting):**

1. Milestone 2H: CAMELSH streamflow recovery for 22 missing basins.
2. Decide on all-12 animation run (2E follow-up).
3. Event QC conclusions: finalize which of R01–R12 are included in Stage 1 training.
4. HPC transfer planning.
5. Stage 1 model configuration and first training run.

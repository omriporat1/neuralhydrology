# Stage 1 Forcing — Full-Period Visual QC Case Selection

**Created:** 2026-06-25
**Updated:** 2026-06-28
**Status:** Pilot visual QC PASS (6/6 basin-timeseries + spatial MRMS smoke VQC-009/VQC-012).
Remaining 15/21 cases are **on hold** — not forgotten; not yet animated.
**Audit plan reference:** `docs/stage1_forcing_fullperiod_postrun_audit_plan.md §7`
**Full evidence:** `docs/stage1_forcing_fullperiod_visual_qc_animation_plan.md`

---

## Purpose

This document describes the visual / event QC gate for the Stage 1 full-period
MRMS+RTMA basin-average forcing extraction (Milestone 2K-E, PASS_WITH_CAVEATS).

The gate requires a human reviewer to inspect forcing-data animations for a curated
set of cases spanning gap contexts, event types, basin types, and geographic regions.
The case-selection table is a **reproducibility record** — it does not certify the
forcing product by itself.

**Pilot animations generated and reviewed (2026-06-25/28):** 6/6 basin-timeseries cases OK;
spatial MRMS smoke passed for VQC-009 and VQC-012. See `docs/stage1_forcing_fullperiod_visual_qc_animation_plan.md`.
Remaining 15/21 cases are on hold. This document covers case selection and reproducibility.

---

## Inputs

| Input | Path | Notes |
|---|---|---|
| Audit tables | `tmp/stage1_forcing_fullperiod_postrun_audit_20260624T060504Z/` | `fullperiod_gap_inventory.csv`, `fullperiod_basin_completeness.csv` |
| Basin metadata | `reports/flashnh_usgs_site_metadata_v001/tables/wy2024_metrics_with_site_metadata.csv` | RBI, area, altitude, HUC02, state |
| Generation script | `scripts/generate_visual_qc_case_selection.py` | CLI args; `seed=42`; deterministic |

---

## Outputs (not committed)

Generated outputs live under `tmp/` and are gitignored:

```
tmp/stage1_forcing_fullperiod_visual_qc_selection_20260625T081859Z/
  visual_qc_case_selection.csv   — 21 cases, 29 columns
  summary.md                     — selection logic, boundary-window policy, limitations
```

To reproduce:

```bash
python -X utf8 scripts/generate_visual_qc_case_selection.py \
    --audit-dir  tmp/stage1_forcing_fullperiod_postrun_audit_20260624T060504Z \
    --meta-csv   reports/flashnh_usgs_site_metadata_v001/tables/wy2024_metrics_with_site_metadata.csv \
    --out-dir    tmp/stage1_forcing_fullperiod_visual_qc_selection_YYYYMMDDTHHMMSSZ
```

Re-running on identical inputs produces an identical CSV (`seed=42`).

---

## Selection Categories and Counts

| Category | n | Description |
|---|---|---|
| MRMS_GAP_ADJACENT | 3 | Windows spanning the three largest MRMS gap clusters; test NaN rendering |
| RTMA_GAP_ADJACENT | 1 | The only 2 RTMA missing hours in 45,720h extraction (2020-11-12T09Z/T10Z) |
| WINTER_MIXED_PRECIP | 3 | DJF months; NE, SE, and West mountain basins |
| WARM_SEASON_CONVECTIVE | 3 | JJA months; Midwest, SW monsoon, SE |
| SMALL_FLASHY_BASIN | 2 | High RBI / low area; urban Midwest and TX |
| LARGE_BASIN | 1 | Largest v001 basin (~997 km²; no continental-scale basins in v001) |
| HIGH_ALTITUDE | 1 | Highest-altitude v001 basin (>10,000 ft); RTMA temperature check |
| DRY_CONTROL | 2 | Arid/semi-arid regions; expected near-zero MRMS QPE |
| HURRICANE_TRACK | 1 | Hurricane Milton (Oct 9-10, 2024); SE Florida basin |
| RANDOM_CONTROL | 3 | Seed=42 random selection; clean months; different years and regions |
| COLD_SHOULDER | 1 | March shoulder season; rain/snow phase-boundary check |
| **Total** | **21** | |

---

## CSV Column Schema (29 columns)

| Column | Type | Notes |
|---|---|---|
| `case_id` | string | VQC-001 … VQC-021 |
| `STAID` | string | USGS site ID; preserved as string (may be 8 or 9 digits) |
| `basin_name` | string | NWIS monitoring location name |
| `state` | string | 2-letter state code |
| `huc02` | string | 2-digit Hydrologic Unit Code |
| `lat_gage` | float | Gauge latitude (decimal degrees) |
| `lng_gage` | float | Gauge longitude (decimal degrees) |
| `drain_sqkm` | float | Drainage area (km²) |
| `altitude_m` | float | Gauge altitude (m) |
| `rbi` | float | Richards-Baker Flashiness Index |
| `bfi_ave` | float | Baseflow Index |
| `month` | string | YYYY-MM of the anchor |
| `window_start_utc` | ISO 8601 | Nominal unclipped window start (anchor − 48h); may pre-date period start |
| `window_end_utc` | ISO 8601 | Nominal window end (anchor + 24h); always within period |
| `anchor_time_utc` | ISO 8601 | Event anchor time |
| `window_clipped_by_period` | true/false | **true** only for VQC-001 |
| `render_window_start_utc` | ISO 8601 | **Animation must use this as start bound** |
| `render_window_end_utc` | ISO 8601 | Animation end bound (always == `window_end_utc`) |
| `rendered_window_hours` | int | Render window duration in hours (72 for all except VQC-001 = 34) |
| `selection_category` | string | One of the 11 categories above |
| `selection_reason` | string | Free-text rationale |
| `product_gap_context` | string | e.g. `mrms_gap_21h`, `no_gaps_in_window` |
| `mrms_gap_hours_in_render_window` | int | MRMS gap hours within the render window |
| `rtma_gap_hours_in_render_window` | int | RTMA gap hours within the render window |
| `expected_products_available` | string | `MRMS_COMPLETE RTMA_COMPLETE` or `MRMS_PARTIAL …` |
| `animation_priority` | HIGH/MEDIUM | HIGH = gap-adjacent or high-impact event |
| `reviewer` | string | Filled in by reviewer |
| `review_outcome` | string | Filled in by reviewer |
| `notes` | string | Flags and caveats |

---

## Period-Boundary Window Policy (Policy A — Clip)

**Applies to VQC-001 only.**

VQC-001 intentionally targets the MRMS archive-start gap (2020-10-14T00Z–T20Z).
Its anchor is 2020-10-14T10:00:00Z. The nominal 72 h window start
(anchor − 48 h = 2020-10-12T10:00:00Z) pre-dates the Stage 1 forcing period start
(2020-10-14T00:00:00Z) by 38 hours.

**Policy A** clips the render window to the forcing period start:

| Field | VQC-001 value |
|---|---|
| `window_start_utc` | `2020-10-12T10:00:00Z` — preserved as nominal reference; pre-period |
| `window_clipped_by_period` | `true` |
| `render_window_start_utc` | `2020-10-14T00:00:00Z` — animation must start here |
| `render_window_end_utc` | `2020-10-15T10:00:00Z` |
| `rendered_window_hours` | `34` |
| `mrms_gap_hours_in_render_window` | `21` (61.8% of the render window) |

**Animation instructions for VQC-001:**  
Use `render_window_start_utc` and `render_window_end_utc` as time bounds.  
Do not use `window_start_utc` — it is pre-period and has no forcing data.  
The animation will have 34 frames; frames 1–21 will show MRMS NaN (archive-start gap);
frames 22–34 will show valid MRMS data. RTMA is complete throughout.

Policy A was chosen over Policy B (replace VQC-001) because this case is
the most direct test of NaN rendering at the period boundary — a behavior
that cannot be tested with any other case in the set.

---

## Key Limitations

Seasonal/event category labels are **hypotheses**, not verified observations:

- No per-hour MRMS precipitation values are available locally (on h2o in Parquet files).
  Event magnitude for WINTER_MIXED_PRECIP, WARM_SEASON_CONVECTIVE, and HURRICANE_TRACK
  is inferred from month, season, and known historical records.
- No per-basin hourly streamflow for the full period (on h2o). Flash-flood timing
  and peak flow cannot be verified locally.
- Cold/snow cases are identified by altitude and month, not snowpack observations.

**The reviewer must confirm event significance from animations.**

---

## QC Gate Status

**Pilot QC closed (2026-06-28):** Basin-timeseries pilot PASS (VQC-001, -004, -007, -009, -012, -020).
Spatial MRMS smoke PASS (VQC-009, VQC-012). Authorized proceeding to curated forcing product
v001 design. See `docs/decision_log.md §2026-06-25/28`.

**Remaining 15 cases (VQC-002, -003, -005, -006, -008, -010, -011, -013 through -021 excluding
the 6 pilot cases):** On hold per `docs/stage1_forcing_fullperiod_visual_qc_animation_plan.md`.
Do not animate unless the reviewer explicitly requests full gate closure.

**Next step for this gate:** Curated forcing product v001 smoke test (Milestone 2K-F-B).
Full animation of all 21 cases is not required before the smoke test.

Acceptance taxonomy per `audit_plan.md §7`:
PASS / PASS_WITH_CAVEATS / NEEDS_TARGETED_REPAIR / NEEDS_RERUN_FOR_SELECTED_MONTHS / FAIL

---

*Generated CSV and summary under `tmp/` — not committed to git.*  
*Script: `scripts/generate_visual_qc_case_selection.py`*
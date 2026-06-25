"""
Generate visual QC case selection table for the Stage 1 full-period forcing run.

This script produces a GENERATED REPRODUCIBILITY RECORD for human visual QC review.
It does NOT generate animations, certify the forcing product, or connect to h2o.

Reads from local audit tables and basin metadata only — does not access h2o,
scan large forcing Parquet files, or modify any h2o outputs.

Inputs (local):
  --audit-dir   Audit table output directory from generate_fullperiod_audit_tables.py
                  fullperiod_gap_inventory.csv         (gap inventory)
                  fullperiod_basin_completeness.csv    (v001 STAID list)
  --meta-csv    wy2024_metrics_with_site_metadata.csv  (local basin attributes)

Outputs (under --out-dir, not committed to git):
  visual_qc_case_selection.csv   — 21 cases, 29 columns
  summary.md                     — selection logic, boundary-window policy, limitations

Usage:
  python scripts/generate_visual_qc_case_selection.py \\
      --audit-dir  tmp/stage1_forcing_fullperiod_postrun_audit_20260624T060504Z \\
      --meta-csv   reports/flashnh_usgs_site_metadata_v001/tables/wy2024_metrics_with_site_metadata.csv \\
      --out-dir    tmp/stage1_forcing_fullperiod_visual_qc_selection_YYYYMMDDTHHMMSSZ

Selection logic:
  - Gap-context categories (MRMS_GAP_ADJACENT, RTMA_GAP_ADJACENT) are data-driven
    from fullperiod_gap_inventory.csv.
  - Seasonal/event categories are climatologically-informed (month/season heuristics
    and known historical events). The reviewer must confirm event significance from
    animations — these labels are hypotheses, not verified observations.
  - Random control cases use seed=42 for reproducibility.

Period-boundary policy (Policy A — clip):
  If a nominal 72h window extends before the Stage 1 forcing period start
  (2020-10-14T00:00:00Z), it is clipped to the period start.
  Columns window_clipped_by_period, render_window_start_utc, render_window_end_utc,
  and rendered_window_hours record the clip. window_start_utc preserves the unclipped
  nominal start for reference. Only VQC-001 is affected.
"""

import argparse
import csv
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

# Reproducible seed for random basin/window selections
RANDOM_SEED = 42

# Window geometry: anchor ± these offsets in hours
WINDOW_PRE_H  = 48   # hours before anchor
WINDOW_POST_H = 24   # hours after anchor

# Stage 1 forcing period boundaries (UTC, inclusive)
FORCING_PERIOD_START = datetime(2020, 10, 14, 0, 0, 0, tzinfo=timezone.utc)
FORCING_PERIOD_END   = datetime(2025, 12, 31, 23, 0, 0, tzinfo=timezone.utc)

# Product identifiers (match gap_inventory.csv)
MRMS = "mrms_qpe_1h_pass1"
RTMA = "rtma_conus_aws_2p5km"

# Required audit table files (relative to --audit-dir)
REQUIRED_AUDIT_FILES = [
    "fullperiod_gap_inventory.csv",
    "fullperiod_basin_completeness.csv",
]

# Output CSV column order (29 fields)
FIELDS = [
    "case_id", "STAID", "basin_name", "state", "huc02",
    "lat_gage", "lng_gage", "drain_sqkm", "altitude_m",
    "rbi", "bfi_ave", "month",
    "window_start_utc", "window_end_utc", "anchor_time_utc",
    "window_clipped_by_period",
    "render_window_start_utc", "render_window_end_utc", "rendered_window_hours",
    "selection_category", "selection_reason",
    "product_gap_context", "mrms_gap_hours_in_render_window",
    "rtma_gap_hours_in_render_window", "expected_products_available",
    "animation_priority", "reviewer", "review_outcome", "notes",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Generate visual QC case selection table for the Stage 1 full-period "
            "forcing run. Outputs go under --out-dir (not committed to git). "
            "Does not generate animations or connect to h2o."
        )
    )
    p.add_argument("--audit-dir", required=True, metavar="DIR",
                   help="Audit table directory from generate_fullperiod_audit_tables.py "
                        "(must contain fullperiod_gap_inventory.csv and "
                        "fullperiod_basin_completeness.csv)")
    p.add_argument("--meta-csv", required=True, metavar="FILE",
                   help="wy2024_metrics_with_site_metadata.csv (local basin attributes)")
    p.add_argument("--out-dir", required=True, metavar="DIR",
                   help="Output directory for CSV and summary.md (created if absent; "
                        "should be under tmp/ to stay gitignored)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def utc(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def fmt_utc(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def nominal_window(anchor_dt):
    """Return (window_start, window_end) as UTC strings for a 72h window."""
    return (
        fmt_utc(anchor_dt - timedelta(hours=WINDOW_PRE_H)),
        fmt_utc(anchor_dt + timedelta(hours=WINDOW_POST_H)),
    )


def compute_render_window(ws_str, we_str):
    """
    Apply Policy A: clip the window to FORCING_PERIOD_START if needed.

    Returns dict with 4 clip fields:
      window_clipped_by_period  : "true" / "false"
      render_window_start_utc   : clipped or original window_start
      render_window_end_utc     : always == window_end_utc
      rendered_window_hours     : integer hours in the render window
    """
    ws_dt = utc(ws_str)
    we_dt = utc(we_str)

    if ws_dt < FORCING_PERIOD_START:
        render_start = FORCING_PERIOD_START
        clipped = "true"
    else:
        render_start = ws_dt
        clipped = "false"

    hours = int((we_dt - render_start).total_seconds() / 3600)
    return {
        "window_clipped_by_period": clipped,
        "render_window_start_utc":  fmt_utc(render_start),
        "render_window_end_utc":    we_str,
        "rendered_window_hours":    hours,
    }


# ---------------------------------------------------------------------------
# Load audit data
# ---------------------------------------------------------------------------
def load_audit(audit_dir):
    gaps = load_csv(os.path.join(audit_dir, "fullperiod_gap_inventory.csv"))
    bc   = load_csv(os.path.join(audit_dir, "fullperiod_basin_completeness.csv"))
    return gaps, bc


def build_gap_index(gaps):
    """Build dict: utc_hour_string -> set of products that are missing that hour."""
    gap_hours = defaultdict(set)
    for g in gaps:
        start = utc(g["gap_start_utc"])
        end   = utc(g["gap_end_utc"])
        prod  = g["product"]
        h = start
        while h <= end:
            gap_hours[fmt_utc(h)].add(prod)
            h += timedelta(hours=1)
    return gap_hours


def gap_context_for_render_window(gap_hours, render_start_str, we_str):
    """
    Count gap hours within the render window (period-clipped start to window_end).
    Scans only hours in [render_start, window_end] — pre-period hours are excluded.

    Returns (mrms_gap_h, rtma_gap_h, context_string, products_available_string).
    """
    start = utc(render_start_str)
    end   = utc(we_str)
    mrms_n = rtma_n = 0
    h = start
    while h <= end:
        hs = fmt_utc(h)
        missing = gap_hours.get(hs, set())
        if MRMS in missing:
            mrms_n += 1
        if RTMA in missing:
            rtma_n += 1
        h += timedelta(hours=1)

    if mrms_n == 0 and rtma_n == 0:
        ctx = "no_gaps_in_window"
    elif mrms_n > 0 and rtma_n == 0:
        ctx = f"mrms_gap_{mrms_n}h"
    elif rtma_n > 0 and mrms_n == 0:
        ctx = f"rtma_gap_{rtma_n}h"
    else:
        ctx = f"mrms_gap_{mrms_n}h_rtma_gap_{rtma_n}h"

    mrms_ok = "COMPLETE" if mrms_n == 0 else "PARTIAL"
    rtma_ok = "COMPLETE" if rtma_n == 0 else "PARTIAL"
    return mrms_n, rtma_n, ctx, f"MRMS_{mrms_ok} RTMA_{rtma_ok}"


# ---------------------------------------------------------------------------
# Load and index basin metadata
# ---------------------------------------------------------------------------
def load_meta(meta_csv, v001_staids):
    rows = load_csv(meta_csv)
    meta = {}
    for r in rows:
        sid = r["STAID"]          # preserve as string, never convert
        if sid in v001_staids:
            meta[sid] = r
    return meta


# ---------------------------------------------------------------------------
# Basin selectors
# ---------------------------------------------------------------------------
def pick_basin(meta, huc02=None, min_area=None, max_area=None,
               min_alt_ft=None, exclude=None, rank_by=None, ascending=True):
    """Return the best-matching STAID (string) by criteria, or None."""
    candidates = []
    for s, m in meta.items():
        if exclude and s in exclude:
            continue
        if huc02 and m.get("HUC02") != huc02:
            continue
        try:
            area = float(m.get("DRAIN_SQKM") or 0)
            alt  = float(m.get("altitude_ft") or 0)
        except ValueError:
            continue
        if min_area  is not None and area < min_area:
            continue
        if max_area  is not None and area > max_area:
            continue
        if min_alt_ft is not None and alt < min_alt_ft:
            continue
        key = float(m.get(rank_by) or 0) if rank_by else area
        candidates.append((s, key))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=not ascending)
    return candidates[0][0]


def pick_random_basin(meta, huc02, min_area=50, max_area=2000,
                      exclude=None, rng=None):
    """Pick a reproducible-random basin in a HUC02 band."""
    pool = sorted(
        s for s, m in meta.items()
        if m.get("HUC02") == huc02
        and (exclude is None or s not in exclude)
        and m.get("DRAIN_SQKM")
        and min_area <= float(m["DRAIN_SQKM"]) <= max_area
    )
    if not pool:
        return None
    return (rng or random).choice(pool)


# ---------------------------------------------------------------------------
# Build one case row
# ---------------------------------------------------------------------------
def build_case(case_id, staid, meta, anchor_str, category, reason,
               gap_hours, notes="", priority="MEDIUM"):
    """
    Build a single CSV row dict.  STAIDs are preserved as strings throughout.
    Applies Policy A period-boundary clipping and counts gap hours in the
    render window only (not in pre-period nominal hours).
    """
    m = meta.get(staid, {})
    anchor_dt = utc(anchor_str)
    ws, we = nominal_window(anchor_dt)
    clip   = compute_render_window(ws, we)

    mrms_n, rtma_n, ctx, avail = gap_context_for_render_window(
        gap_hours, clip["render_window_start_utc"], we
    )
    month = anchor_dt.strftime("%Y-%m")

    alt_m = ""
    try:
        alt_ft = float(m.get("altitude_ft") or 0)
        alt_m  = f"{alt_ft * 0.3048:.0f}"
    except (ValueError, TypeError):
        pass

    return {
        "case_id":                        case_id,
        "STAID":                          staid,          # string, never cast
        "basin_name":                     m.get("monitoring_location_name", ""),
        "state":                          m.get("STATE", ""),
        "huc02":                          m.get("HUC02", ""),
        "lat_gage":                       m.get("LAT_GAGE", m.get("latitude", "")),
        "lng_gage":                       m.get("LNG_GAGE", m.get("longitude", "")),
        "drain_sqkm":                     m.get("DRAIN_SQKM", ""),
        "altitude_m":                     alt_m,
        "rbi":                            m.get("RBI", ""),
        "bfi_ave":                        m.get("BFI_AVE", ""),
        "month":                          month,
        "window_start_utc":               ws,
        "window_end_utc":                 we,
        "anchor_time_utc":                anchor_str,
        "window_clipped_by_period":       clip["window_clipped_by_period"],
        "render_window_start_utc":        clip["render_window_start_utc"],
        "render_window_end_utc":          clip["render_window_end_utc"],
        "rendered_window_hours":          clip["rendered_window_hours"],
        "selection_category":             category,
        "selection_reason":               reason,
        "product_gap_context":            ctx,
        "mrms_gap_hours_in_render_window": mrms_n,
        "rtma_gap_hours_in_render_window": rtma_n,
        "expected_products_available":    avail,
        "animation_priority":             priority,
        "reviewer":                       "",
        "review_outcome":                 "",
        "notes":                          notes,
    }


# ---------------------------------------------------------------------------
# Case-selection logic
# ---------------------------------------------------------------------------
def select_cases(meta, gap_hours):
    rng = random.Random(RANDOM_SEED)
    cases = []
    used  = set()   # used STAIDs (each basin appears at most once)

    # ------------------------------------------------------------------
    # 1. MRMS_GAP_ADJACENT — 3 cases covering the three largest MRMS gaps
    # ------------------------------------------------------------------

    # VQC-001: archive-start gap 2020-10-14T00Z–20Z (21h).
    # Nominal window pre-dates the forcing period start → Policy A clip applied.
    s = pick_basin(meta, huc02="01", min_area=100, max_area=800, exclude=used)
    if s:
        cases.append(build_case(
            "VQC-001", s, meta, "2020-10-14T10:00:00Z",
            "MRMS_GAP_ADJACENT",
            ("MRMS S3 archive starts 2020-10-14T21Z; hours T00Z-T20Z permanently absent. "
             "Anchor is T10Z (within the gap). Nominal 72h window pre-dates the forcing "
             "period start (2020-10-14T00Z) — Policy A clip applied: "
             "render_window_start_utc=2020-10-14T00:00:00Z, rendered_window_hours=34. "
             "Tests NaN rendering at period start; 21 of 34 render-window MRMS hours are absent."),
            gap_hours, priority="HIGH",
            notes=("BOUNDARY CASE — Policy A (clip). render_window_start clipped to "
                   "2020-10-14T00:00:00Z; rendered_window_hours=34. "
                   "Phase 1 extractor commit 194a489.")))
        used.add(s)

    # VQC-002: Oct 25-26 outage (~13h, largest mid-period MRMS gap in Oct 2020)
    s = pick_basin(meta, huc02="07", min_area=100, max_area=600, exclude=used)
    if s:
        cases.append(build_case(
            "VQC-002", s, meta, "2020-10-26T00:00:00Z",
            "MRMS_GAP_ADJACENT",
            ("MRMS outage 2020-10-25T23Z – 2020-10-26T11Z (not_in_s3). "
             "Largest continuous mid-period MRMS gap in the Oct 2020 chunk. "
             "Window fully inside the forcing period — no clipping."),
            gap_hours, priority="HIGH",
            notes="Same month as VQC-001 but different gap cluster."))
        used.add(s)

    # VQC-003: 2021-07-19 (two gap runs: 6h T05Z-T10Z + 3h T16Z-T18Z = 9h in window)
    s = pick_basin(meta, huc02="06", min_area=50, max_area=500, exclude=used)
    if s:
        cases.append(build_case(
            "VQC-003", s, meta, "2021-07-19T07:00:00Z",
            "MRMS_GAP_ADJACENT",
            ("MRMS has two gap runs on 2021-07-19: 6h (T05Z-T10Z) + 3h (T16Z-T18Z). "
             "Both fall within the 72h render window → 9 MRMS gap hours total. "
             "Longest warm-season multi-hour MRMS outage in the full-period run."),
            gap_hours, priority="HIGH"))
        used.add(s)

    # ------------------------------------------------------------------
    # 2. RTMA_GAP_ADJACENT — 1 case (only 2 RTMA missing hours in all 45,720h)
    # ------------------------------------------------------------------
    s = pick_basin(meta, huc02="01", min_area=100, max_area=800, exclude=used)
    if s:
        cases.append(build_case(
            "VQC-004", s, meta, "2020-11-12T09:00:00Z",
            "RTMA_GAP_ADJACENT",
            ("RTMA absent at 2020-11-12T09Z and T10Z (not_in_s3). "
             "These 2 hours are the only RTMA missing hours in the entire 45,720h extraction. "
             "Tests RTMA-specific NaN rendering; MRMS complete in this window."),
            gap_hours, priority="HIGH",
            notes="Discovered in post-run audit; 2020-11 all_pass=True with 718/720 RTMA hours."))
        used.add(s)

    # ------------------------------------------------------------------
    # 3. WINTER_MIXED_PRECIP (DJF) — 3 cases across three regions
    # ------------------------------------------------------------------

    # NE/Mid-Atlantic (2021-01 — highest MRMS window-impact month, 20.5%)
    s = pick_basin(meta, huc02="02", min_area=100, max_area=600, exclude=used)
    if s:
        cases.append(build_case(
            "VQC-005", s, meta, "2021-01-15T12:00:00Z",
            "WINTER_MIXED_PRECIP",
            ("January 2021 has the highest MRMS window impact of any month (45 missing h, 20.5%). "
             "Window anchored Jan 15 — before the heavy gap cluster (Jan 22-26) — "
             "to show clean winter forcing. Mid-Atlantic/NE basin; mixed cold precip expected."),
            gap_hours,
            notes="Month has severe S3 gaps; this specific window is gap-free."))
        used.add(s)

    # Southeast (2022-12 — overlaps Winter Storm Elliott Dec 22-26)
    s = pick_basin(meta, huc02="03", min_area=50, max_area=500, exclude=used)
    if s:
        cases.append(build_case(
            "VQC-006", s, meta, "2022-12-22T12:00:00Z",
            "WINTER_MIXED_PRECIP",
            ("December 2022 overlapping Winter Storm Elliott (Dec 22-26, 2022). "
             "SE basin captures potential ice/cold-rain mix and sub-freezing RTMA temperatures. "
             "No MRMS gaps in 2022-12. Event significance must be confirmed from animations."),
            gap_hours,
            notes="Climatologically-inferred event; confirm from MRMS QPE and RTMA 2t in animation."))
        used.add(s)

    # West/Mountain (2024-01 — Upper Colorado snowpack period)
    s = pick_basin(meta, huc02="14", min_area=50, max_area=1000, exclude=used)
    if not s:
        s = pick_basin(meta, huc02="15", min_area=50, max_area=1000, exclude=used)
    if s:
        cases.append(build_case(
            "VQC-007", s, meta, "2024-01-13T12:00:00Z",
            "WINTER_MIXED_PRECIP",
            ("January 2024, Upper Colorado / Great Basin. "
             "High-altitude basin expected to show snow/rain-mixed MRMS QPE and "
             "near/below-freezing RTMA 2t at elevation. No MRMS gaps in 2024-01."),
            gap_hours))
        used.add(s)

    # ------------------------------------------------------------------
    # 4. WARM_SEASON_CONVECTIVE (JJA) — 3 cases across Midwest, SW monsoon, SE
    # ------------------------------------------------------------------

    # Midwest (2022-07, no gaps; highest-RBI basin)
    s = pick_basin(meta, huc02="07", min_area=100, max_area=600, exclude=used,
                   rank_by="RBI", ascending=False)
    if s:
        cases.append(build_case(
            "VQC-008", s, meta, "2022-07-15T20:00:00Z",
            "WARM_SEASON_CONVECTIVE",
            ("July 2022 Midwest; evening anchor (20Z ~ late-afternoon local). "
             "Highest-RBI available Midwest basin chosen for rapid QPE response. "
             "No MRMS gaps in 2022-07. Event significance hypothetical; confirm from animations."),
            gap_hours))
        used.add(s)

    # Southwest monsoon (2023-08, no gaps; HUC15 Arizona)
    s = pick_basin(meta, huc02="15", min_area=100, max_area=1500, exclude=used)
    if not s:
        s = pick_basin(meta, huc02="13", min_area=100, max_area=1500, exclude=used)
    if s:
        cases.append(build_case(
            "VQC-009", s, meta, "2023-08-10T20:00:00Z",
            "WARM_SEASON_CONVECTIVE",
            ("August 2023, North American Monsoon region (HUC15/13). "
             "Afternoon convective anchor. Tests MRMS capture of monsoon convection "
             "and RTMA moisture fields. No gaps in 2023-08."),
            gap_hours))
        used.add(s)

    # Southeast (2023-07; window anchored Jul 26 — clear of the 2h gap at Jul 19T18Z-19Z)
    s = pick_basin(meta, huc02="08", min_area=50, max_area=400, exclude=used)
    if s:
        cases.append(build_case(
            "VQC-010", s, meta, "2023-07-26T20:00:00Z",
            "WARM_SEASON_CONVECTIVE",
            ("July 2023 Southeast; window anchored Jul 26 to avoid the 2h MRMS gap "
             "(2023-07-19T18Z-19Z). Humid subtropical basin; expected heavy convective QPE."),
            gap_hours))
        used.add(s)

    # ------------------------------------------------------------------
    # 5. SMALL_FLASHY_BASIN — 2 cases (high RBI, low area)
    # ------------------------------------------------------------------

    # Urban Midwest (top-RBI, <50 km², HUC07)
    s = pick_basin(meta, huc02="07", min_area=5, max_area=50, exclude=used,
                   rank_by="RBI", ascending=False)
    if s:
        cases.append(build_case(
            "VQC-011", s, meta, "2022-08-06T20:00:00Z",
            "SMALL_FLASHY_BASIN",
            ("Highest-RBI small urban basin in HUC07. Tiny catchment → direct QPE-runoff "
             "response expected. No MRMS gaps in 2022-08."),
            gap_hours, priority="HIGH",
            notes="High RBI likely from urban impervious surface; verify from NLCD metadata."))
        used.add(s)

    # Urban TX/Southern Plains (<50 km², HUC12)
    s = pick_basin(meta, huc02="12", min_area=2, max_area=50, exclude=used,
                   rank_by="RBI", ascending=False)
    if not s:
        s = pick_basin(meta, huc02="11", min_area=2, max_area=50, exclude=used,
                       rank_by="RBI", ascending=False)
    if s:
        cases.append(build_case(
            "VQC-012", s, meta, "2022-03-21T20:00:00Z",
            "SMALL_FLASHY_BASIN",
            ("Small flashy Texas/Southern Plains basin, spring convective period. "
             "Tests MRMS QPE accuracy for intense, short-duration convective cells "
             "over tiny catchments. No gaps in 2022-03."),
            gap_hours))
        used.add(s)

    # ------------------------------------------------------------------
    # 6. LARGE_BASIN — 1 case (v001 max ~997 km²; no continental-scale basins in v001)
    # ------------------------------------------------------------------
    s = pick_basin(meta, min_area=700, exclude=used,
                   rank_by="DRAIN_SQKM", ascending=False)
    if s:
        area_str = f"{float(meta[s].get('DRAIN_SQKM', 0)):.0f}"
        cases.append(build_case(
            "VQC-013", s, meta, "2023-04-15T12:00:00Z",
            "LARGE_BASIN",
            (f"Largest available v001 basin ({area_str} km²; max in v001 ~997 km² — "
             "no continental-scale basins). Spring anchor tests area-averaged QPE "
             "over a large drainage. No gaps in 2023-04."),
            gap_hours))
        used.add(s)

    # ------------------------------------------------------------------
    # 7. HIGH_ALTITUDE — 1 case (>6000 ft; RTMA temperature accuracy at elevation)
    # ------------------------------------------------------------------
    s = pick_basin(meta, min_alt_ft=6000, max_area=200, exclude=used,
                   rank_by="altitude_ft", ascending=False)
    if s:
        alt_ft = float(meta[s].get("altitude_ft", 0))
        cases.append(build_case(
            "VQC-014", s, meta, "2022-04-16T12:00:00Z",
            "HIGH_ALTITUDE",
            (f"Highest-altitude v001 basin ({alt_ft:.0f} ft). April anchor: spring "
             "snowmelt context; QPE may be mixed rain/snow; RTMA 2t expected near 0°C. "
             "2022-04 has MRMS gaps but this window may be clear — check gap context."),
            gap_hours))
        used.add(s)

    # ------------------------------------------------------------------
    # 8. DRY_CONTROL — 2 cases (arid/semi-arid, expected near-zero MRMS QPE)
    # ------------------------------------------------------------------

    # Great Basin (HUC16 / Upper Truckee area, summer)
    s = pick_basin(meta, huc02="16", min_area=100, max_area=2000, exclude=used)
    if not s:
        s = pick_basin(meta, huc02="14", min_area=100, max_area=2000, exclude=used)
    if s:
        cases.append(build_case(
            "VQC-015", s, meta, "2022-07-26T12:00:00Z",
            "DRY_CONTROL",
            ("Great Basin / Upper Colorado arid region. July anchor outside monsoon influence. "
             "Expected near-zero MRMS QPE and high RTMA temperatures — "
             "baseline for instrument-zero check. No gaps in 2022-07."),
            gap_hours,
            notes="Expected low QPE; instrument-zero baseline."))
        used.add(s)

    # Lower Colorado / AZ pre-monsoon (HUC15, May)
    s = pick_basin(meta, huc02="15", min_area=100, max_area=2000, exclude=used)
    if s:
        cases.append(build_case(
            "VQC-016", s, meta, "2021-05-15T12:00:00Z",
            "DRY_CONTROL",
            ("Lower Colorado / Arizona region, May (pre-monsoon dry season). "
             "Expected near-zero MRMS QPE and high RTMA temperatures. "
             "No MRMS gaps in 2021-05."),
            gap_hours))
        used.add(s)

    # ------------------------------------------------------------------
    # 9. HURRICANE_TRACK — 1 case (Hurricane Milton, Oct 9-10 2024)
    # ------------------------------------------------------------------
    s = pick_basin(meta, huc02="03", min_area=50, max_area=500, exclude=used)
    if s:
        cases.append(build_case(
            "VQC-017", s, meta, "2024-10-10T12:00:00Z",
            "HURRICANE_TRACK",
            ("Hurricane Milton (Category 3) made landfall near Siesta Key, FL on "
             "2024-10-09. SE basin anchored at Oct 10 12Z captures landfall + inland QPE. "
             "No MRMS gaps in 2024-10. Event confirmed from historical records; "
             "MRMS QPE magnitude must be verified from animation."),
            gap_hours, priority="HIGH",
            notes="Event inferred from historical records; confirm MRMS QPE from animation."))
        used.add(s)

    # ------------------------------------------------------------------
    # 10. RANDOM_CONTROL — 3 cases (seed=42; clean months, different years/regions)
    # ------------------------------------------------------------------

    # 2022, HUC04 Great Lakes
    s = pick_random_basin(meta, huc02="04", exclude=used, rng=rng)
    if s:
        cases.append(build_case(
            "VQC-018", s, meta, "2022-03-11T12:00:00Z",
            "RANDOM_CONTROL",
            ("Randomly selected Great Lakes basin (seed=42), March 2022. "
             "No MRMS gaps in 2022-03. Ordinary early-spring forcing window."),
            gap_hours))
        used.add(s)

    # 2023, HUC10U Upper Missouri
    s = pick_random_basin(meta, huc02="10U", exclude=used, rng=rng)
    if not s:
        s = pick_random_basin(meta, huc02="10", exclude=used, rng=rng)
    if s:
        cases.append(build_case(
            "VQC-019", s, meta, "2023-06-15T12:00:00Z",
            "RANDOM_CONTROL",
            ("Randomly selected Upper Missouri basin (seed=42), June 2023. "
             "No MRMS gaps in 2023-06. Ordinary summer forcing window."),
            gap_hours))
        used.add(s)

    # 2025, HUC05 Ohio
    s = pick_random_basin(meta, huc02="05", exclude=used, rng=rng)
    if s:
        cases.append(build_case(
            "VQC-020", s, meta, "2025-03-15T12:00:00Z",
            "RANDOM_CONTROL",
            ("Randomly selected Ohio Valley basin (seed=42), March 2025. "
             "No MRMS gaps in 2025-03. Ordinary early-spring forcing window."),
            gap_hours))
        used.add(s)

    # ------------------------------------------------------------------
    # 11. COLD_SHOULDER (MAM/SON shoulder season, not DJF)
    # ------------------------------------------------------------------
    # 2021-03 has a 4h MRMS gap at Mar 03T21Z; anchor Mar 20 is clear
    s = pick_basin(meta, huc02="02", min_area=100, max_area=500, exclude=used)
    if s:
        cases.append(build_case(
            "VQC-021", s, meta, "2021-03-20T12:00:00Z",
            "COLD_SHOULDER",
            ("March 2021 shoulder season; window anchored Mar 20 — clear of the 4h "
             "MRMS gap (2021-03-03T21Z). Mid-Atlantic NE basin; rain/snow mix expected. "
             "Tests RTMA boundary between snow and rain phases."),
            gap_hours,
            notes="Shoulder season mixed precip; confirm phase boundary from RTMA 2t field."))
        used.add(s)

    return cases


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------
def write_summary(path, cases):
    n = len(cases)
    by_cat = defaultdict(list)
    for c in cases:
        by_cat[c["selection_category"]].append(c["case_id"])

    gap_cases     = [c for c in cases if c["product_gap_context"] != "no_gaps_in_window"]
    clean_cases   = [c for c in cases if c["product_gap_context"] == "no_gaps_in_window"]
    clipped_cases = [c for c in cases if c["window_clipped_by_period"] == "true"]

    lines = [
        "# Stage 1 Forcing — Visual QC Case Selection",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}  ",
        f"**Seed:** {RANDOM_SEED}  ",
        f"**Total cases:** {n}  ",
        f"**Cases with gap in render window:** {len(gap_cases)} / {n}  ",
        f"**Cases with clean render window:** {len(clean_cases)} / {n}  ",
        f"**Cases with period-boundary clip (Policy A):** {len(clipped_cases)} / {n}  ",
        "",
        "---",
        "",
        "## Purpose",
        "",
        "This table is a **reproducibility record for human visual QC review**. "
        "It selects the cases that a reviewer will animate and inspect. "
        "It does **not** certify the forcing product, generate animations, or "
        "connect to h2o.",
        "",
        "---",
        "",
        "## Selection Logic",
        "",
        "Cases are selected from two local data sources:",
        "",
        "1. **Audit tables** (`fullperiod_gap_inventory.csv`, `fullperiod_basin_completeness.csv`)",
        "   — gap-context categories are data-driven from these files.",
        "2. **Basin metadata** (`wy2024_metrics_with_site_metadata.csv`)",
        "   — RBI, area, altitude, HUC02, state used for stratified selection.",
        "",
        "Seasonal/event categories (WINTER_MIXED_PRECIP, WARM_SEASON_CONVECTIVE,",
        "HURRICANE_TRACK) are **climatologically inferred from month and known historical",
        "events** — not from observed precipitation or streamflow values.",
        "**These labels are hypotheses until the reviewer confirms them from animations.**",
        "",
        f"Random control cases use `seed={RANDOM_SEED}` for reproducibility. "
        "Re-running this script on identical inputs produces an identical CSV.",
        "",
        "Nominal window geometry: anchor − 48 h → anchor + 24 h (72 h total).",
        "Period-boundary cases are clipped; see Policy A below.",
        "",
        "---",
        "",
        "## Period-Boundary Window Policy (Policy A — Clip)",
        "",
        "**Applies to:** VQC-001 only.",
        "",
        "VQC-001's anchor is 2020-10-14T10:00:00Z. The nominal 72 h window start",
        "(anchor − 48 h = 2020-10-12T10:00:00Z) pre-dates the forcing period start",
        "(2020-10-14T00:00:00Z) by 38 hours.",
        "",
        "**Policy A** clips the render window to the forcing period start:",
        "",
        "| Field | Value for VQC-001 |",
        "|---|---|",
        "| `window_start_utc` | `2020-10-12T10:00:00Z` (nominal, pre-period — reference only) |",
        "| `window_end_utc` | `2020-10-15T10:00:00Z` |",
        "| `window_clipped_by_period` | `true` |",
        "| `render_window_start_utc` | `2020-10-14T00:00:00Z` (clipped to period start) |",
        "| `render_window_end_utc` | `2020-10-15T10:00:00Z` |",
        "| `rendered_window_hours` | `34` |",
        "",
        "The animation script **must** use `render_window_start_utc` and `render_window_end_utc`",
        "as its time bounds, not `window_start_utc`. For all other cases",
        "(`window_clipped_by_period = false`) the three start fields are identical.",
        "",
        "VQC-001 is a **boundary-stress case**: 21 of its 34 render-window MRMS hours",
        "are absent (archive-start gap T00Z–T20Z). It will produce a 34-frame animation",
        "with MRMS NaN for the first 21 frames. This is intentional — it tests NaN",
        "rendering at the period boundary.",
        "",
        "---",
        "",
        "## Category Summary",
        "",
        "| Category | n | Case IDs |",
        "|---|---|---|",
    ]
    for cat, ids in sorted(by_cat.items()):
        lines.append(f"| {cat} | {len(ids)} | {', '.join(ids)} |")

    lines += [
        "",
        "---",
        "",
        "## Limitations",
        "",
        "The following data are **not available locally** and constrain event selection:",
        "",
        "- **Per-hour MRMS precipitation values** — on h2o in monthly Parquet files "
        "(~125 M rows). Without these, 'strongest event' is inferred from season and "
        "climatology, not observed peak QPE.",
        "- **Per-basin hourly streamflow for the full period** — on h2o in the v001 "
        "target package. Flash-flood timing and peak flow cannot be verified locally.",
        "- **Snow cover or SWE** — cold/snow cases are identified by altitude and month, "
        "not confirmed by snowpack observations.",
        "",
        "### Additional local files that would enable data-driven selection",
        "",
        "| Data needed | Format | Purpose |",
        "|---|---|---|",
        "| Monthly MRMS Parquet peak-QPE summary | Parquet (h2o) | Rank months by basin-peak QPE |",
        "| v001 streamflow hourly NetCDF (sample) | NetCDF (h2o) | Confirm flash-flood event timing |",
        "| CAMELSH static attributes with snow fraction | CSV (local?) | Confirm snow-dominated basins |",
        "",
        "---",
        "",
        "## Case List",
        "",
        "Clipped = `window_clipped_by_period`; Rend h = `rendered_window_hours`.",
        "",
        "| ID | STAID | Basin | State | Area km² | Category | Month | Gap ctx | Clipped | Rend h | P |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cases:
        clipped_flag = "YES" if c["window_clipped_by_period"] == "true" else ""
        lines.append(
            f"| {c['case_id']} | {c['STAID']} | "
            f"{c['basin_name'][:38]} | {c['state']} | "
            f"{float(c['drain_sqkm'] or 0):.0f} | "
            f"{c['selection_category']} | {c['month']} | "
            f"{c['product_gap_context']} | {clipped_flag} | "
            f"{c['rendered_window_hours']} | {c['animation_priority']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Note on Animations",
        "",
        "**Animations are not generated yet.** This table is the input to the animation",
        "generation step. The generated CSV and this summary are not committed to git",
        "(outputs remain under `tmp/`).",
        "",
        "---",
        "",
        "*Reproducible: `seed=42`; re-running on identical inputs yields identical output.*  ",
        "*Generated by `scripts/generate_visual_qc_case_selection.py`.*",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    audit_dir = os.path.abspath(args.audit_dir)
    meta_csv  = os.path.abspath(args.meta_csv)
    out_dir   = os.path.abspath(args.out_dir)

    # Pre-flight: required input files
    errors = []
    if not os.path.isdir(audit_dir):
        errors.append(f"ERROR: --audit-dir not found: {audit_dir}")
    else:
        for fname in REQUIRED_AUDIT_FILES:
            fpath = os.path.join(audit_dir, fname)
            if not os.path.isfile(fpath):
                errors.append(f"ERROR: required audit file missing: {fpath}")
    if not os.path.isfile(meta_csv):
        errors.append(f"ERROR: --meta-csv not found: {meta_csv}")
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    # Load
    print("Loading audit tables ...")
    gaps, bc = load_audit(audit_dir)
    gap_hours = build_gap_index(gaps)
    print(f"  {len(gaps)} gap runs → {len(gap_hours)} indexed gap hours")

    print("Loading basin metadata ...")
    v001_staids = set(r["STAID"] for r in bc if r["product"] == MRMS)
    meta = load_meta(meta_csv, v001_staids)
    print(f"  {len(meta)}/{len(v001_staids)} v001 basins have metadata")

    # Select
    print("Selecting cases ...")
    cases = select_cases(meta, gap_hours)
    print(f"  {len(cases)} cases selected")
    cat_counts = Counter(c["selection_category"] for c in cases)
    for cat, n in sorted(cat_counts.items()):
        print(f"    {cat}: {n}")

    # Validate: all STAIDs unique, all render windows inside forcing period
    staids = [c["STAID"] for c in cases]
    if len(staids) != len(set(staids)):
        print("WARNING: duplicate STAIDs in output", file=sys.stderr)
    for c in cases:
        rs = utc(c["render_window_start_utc"])
        re = utc(c["render_window_end_utc"])
        if rs < FORCING_PERIOD_START:
            print(f"WARNING: {c['case_id']} render_window_start before period start",
                  file=sys.stderr)
        if re > FORCING_PERIOD_END:
            print(f"WARNING: {c['case_id']} render_window_end after period end",
                  file=sys.stderr)

    # Write
    csv_path = os.path.join(out_dir, "visual_qc_case_selection.csv")
    write_csv(csv_path, FIELDS, cases)

    md_path = os.path.join(out_dir, "summary.md")
    write_summary(md_path, cases)

    # Console table
    print()
    hdr = f"{'ID':<9} {'STAID':<14} {'Cat':<26} {'Month':<8} {'Gap ctx':<28} {'Clip':<5} {'Rndh':>5} P"
    print(hdr)
    print("-" * len(hdr))
    for c in cases:
        clip_flag = "Y" if c["window_clipped_by_period"] == "true" else ""
        print(f"{c['case_id']:<9} {c['STAID']:<14} "
              f"{c['selection_category']:<26} {c['month']:<8} "
              f"{c['product_gap_context']:<28} {clip_flag:<5} "
              f"{c['rendered_window_hours']:>5} {c['animation_priority']}")

    print(f"\nWritten: {csv_path}")
    print(f"Written: {md_path}")
    print(f"\nAll outputs under: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
generate_fullperiod_visual_qc_animations.py

Generates visual QC animations for selected cases from the Stage 1 full-period
MRMS+RTMA basin-average forcing extraction.

*** RUN LOCATION ***
This script is designed to execute on h2o, where the full-period forcing
Parquets and v001 target package reside. Use --dry-run for local syntax/logic
validation. Do NOT run locally without --dry-run unless the required h2o data
files are explicitly present.

Data sources (h2o paths):
  --forcing-root    /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod
    Layout: chunks/{YYYY-MM}/combined_{YYYY-MM}.parquet   (all basins × vars)
  --target-package-root  /data42/omrip/Flash-NH/tmp/stage1_target_package_v001
    Layout: time_series/{STAID}.nc   (qobs_m3s with 'date' coordinate, hourly UTC)

Animation design (3-panel time series; no spatial map — raw GRIB2 not retained):
  Panel 1 (top):    Basin-mean MRMS 1h QPE (mm), bar chart with gap markers
  Panel 2 (middle): RTMA basin-mean 2m temperature (°C)
  Panel 3 (bottom): Streamflow qobs_m3s

VQC-001 special handling:
  render_window_start_utc is 2020-10-14T00:00:00Z (period start, clipped).
  window_start_utc (2020-10-12T10:00:00Z) is PRE-PERIOD — animations MUST use
  render_window_start_utc, not window_start_utc. MRMS T00Z–T20Z are absent
  (archive-start gap); those frames are explicitly labeled.

Usage (dry-run, local validation):
  python scripts/generate_fullperiod_visual_qc_animations.py \\
      --case-selection-csv  tmp/.../visual_qc_case_selection.csv \\
      --case-ids VQC-001 VQC-004 VQC-007 VQC-009 VQC-012 VQC-020 \\
      --forcing-root  /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod \\
      --target-package-root /data42/omrip/Flash-NH/tmp/stage1_target_package_v001 \\
      --out-dir tmp/stage1_forcing_fullperiod_visual_qc_pilot_TIMESTAMP \\
      --dry-run

Usage (h2o real run, via launcher):
  bash scripts/run_fullperiod_visual_qc_pilot_h2o.sh
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
import traceback
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.animation import PillowWriter

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default pilot case IDs — the 6 cases selected for the first h2o run.
# The launcher uses this list; --case-ids can override.
PILOT_CASE_IDS = [
    "VQC-001",   # MRMS archive-start boundary/gap case (clipped to period start)
    "VQC-004",   # Only RTMA gap case (2020-11-12T09Z/T10Z)
    "VQC-007",   # Winter / high-altitude / mixed-precip
    "VQC-009",   # Warm-season convective / SW monsoon
    "VQC-012",   # Very small flashy urban basin (4.6 km²)
    "VQC-020",   # Random/control
]

# VQC-001 archive-start MRMS gap (inclusive range, UTC)
VQC001_STAID              = "01170100"
VQC001_ARCHIVE_GAP_START  = "2020-10-14T00:00:00Z"
VQC001_ARCHIVE_GAP_END    = "2020-10-14T20:00:00Z"
VQC001_RENDER_START_EXPECTED = "2020-10-14T00:00:00Z"

# Column names in the combined monthly Parquet (from STAT_COLUMNS in extraction.py)
PARQUET_LOAD_COLS  = ["STAID", "product", "variable", "valid_time_utc", "weighted_mean"]
MRMS_PRODUCT       = "mrms_qpe_1h_pass1"
RTMA_PRODUCT       = "rtma_conus_aws_2p5km"
RTMA_VAR_T2M       = "2t"

# Target package NC layout
TARGET_NC_SUBDIR   = "time_series"
TARGET_NC_QOBS_VAR = "qobs_m3s"
TARGET_NC_TIME_DIM = "date"

# Animation defaults
FPS_DEFAULT        = 4
DPI_DEFAULT        = 90
FIGURE_W           = 13.0
FIGURE_H           = 7.5
MAX_CASES_DEFAULT  = 6     # safety cap; never run all 21 without explicit --max-cases 21

# Output file suffixes per case
ANIM_SUFFIX_GIF    = "_animation.gif"
ANIM_SUFFIX_MP4    = "_animation.mp4"
QUICKLOOK_SUFFIX   = "_quicklook.png"
MANIFEST_CSV       = "animation_manifest.csv"
SUMMARY_MD         = "animation_summary.md"

# Manifest CSV columns
MANIFEST_FIELDS = [
    "case_id", "STAID", "basin_name", "category", "month",
    "render_window_start_utc", "render_window_end_utc", "rendered_window_hours",
    "window_clipped_by_period", "product_gap_context",
    "n_frames", "fps", "n_mrms_missing_frames", "n_rtma_missing_frames",
    "n_qobs_nan_hours", "output_format", "output_path",
    "quicklook_path", "status", "error", "runtime_s",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Generate visual QC animations for Stage 1 full-period forcing cases. "
            "Designed to run on h2o. Use --dry-run for local validation."
        )
    )
    p.add_argument("--case-selection-csv", required=True, metavar="FILE",
                   help="visual_qc_case_selection.csv from generate_visual_qc_case_selection.py")
    p.add_argument("--case-ids", nargs="+", metavar="ID",
                   default=PILOT_CASE_IDS,
                   help=f"Case IDs to animate (default: {' '.join(PILOT_CASE_IDS)})")
    p.add_argument("--forcing-root", required=True, metavar="DIR",
                   help=("Root of full-period forcing outputs on h2o. "
                         "Expected: chunks/{YYYY-MM}/combined_{YYYY-MM}.parquet"))
    p.add_argument("--target-package-root", required=True, metavar="DIR",
                   help=("Root of v001 target package on h2o. "
                         "Expected: time_series/{STAID}.nc"))
    p.add_argument("--out-dir", required=True, metavar="DIR",
                   help="Output directory (created if absent; should be under tmp/)")
    p.add_argument("--max-cases", type=int, default=MAX_CASES_DEFAULT,
                   help=f"Safety cap on number of cases to animate (default {MAX_CASES_DEFAULT})")
    p.add_argument("--dry-run", action="store_true",
                   help=("Validate cases, print expected data paths, list expected output "
                         "files — do not read large Parquet/NC files or render animations."))
    p.add_argument("--check-data", action="store_true",
                   help=("In combination with --dry-run, also verify that h2o data paths "
                         "exist on disk. Off by default to allow local dry-run."))
    p.add_argument("--format", choices=["gif", "mp4"], default="gif",
                   dest="anim_format",
                   help="Animation format (default: gif). mp4 requires ffmpeg.")
    p.add_argument("--fps", type=int, default=FPS_DEFAULT,
                   help=f"Frames per second (default {FPS_DEFAULT})")
    p.add_argument("--dpi", type=int, default=DPI_DEFAULT,
                   help=f"DPI for output figures (default {DPI_DEFAULT})")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-animate cases that already have output files")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Case-selection CSV loader
# ---------------------------------------------------------------------------
def load_case_selection(csv_path: str) -> dict:
    """Return {case_id: row_dict} with all fields from visual_qc_case_selection.csv."""
    if not os.path.isfile(csv_path):
        print(f"ERROR: --case-selection-csv not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"ERROR: case-selection CSV is empty: {csv_path}", file=sys.stderr)
        sys.exit(1)
    return {r["case_id"]: r for r in rows}


def validate_and_select_cases(all_cases: dict, requested_ids: list, max_cases: int) -> list:
    """Validate requested case IDs and return ordered list of case dicts."""
    missing = [cid for cid in requested_ids if cid not in all_cases]
    if missing:
        print(f"ERROR: case IDs not found in CSV: {missing}", file=sys.stderr)
        print(f"  Available: {sorted(all_cases.keys())}", file=sys.stderr)
        sys.exit(1)

    selected = [all_cases[cid] for cid in requested_ids]

    if len(selected) > max_cases:
        print(f"ERROR: {len(selected)} cases requested but --max-cases={max_cases}. "
              f"Reduce --case-ids or increase --max-cases.", file=sys.stderr)
        sys.exit(1)

    # Critical: verify render windows are used (not nominal window_start)
    for c in selected:
        cid = c["case_id"]
        rws = c.get("render_window_start_utc", "")
        rwe = c.get("render_window_end_utc", "")
        ws  = c.get("window_start_utc", "")
        if not rws or not rwe:
            print(f"ERROR: {cid} missing render_window_start_utc or render_window_end_utc.",
                  file=sys.stderr)
            sys.exit(1)
        if rws != ws and c.get("window_clipped_by_period") != "true":
            print(f"WARNING: {cid} render_window_start differs from window_start but "
                  f"window_clipped_by_period is not 'true'", file=sys.stderr)
        # VQC-001 specific guard
        if c.get("STAID") == VQC001_STAID or cid == "VQC-001":
            if rws != VQC001_RENDER_START_EXPECTED:
                print(f"WARNING: VQC-001 render_window_start_utc={rws!r}, "
                      f"expected {VQC001_RENDER_START_EXPECTED}", file=sys.stderr)

    return selected


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------
def months_in_render_window(render_start: pd.Timestamp, render_end: pd.Timestamp) -> list:
    """Return list of 'YYYY-MM' strings spanning the render window."""
    months = []
    cur = render_start.to_period("M")
    end_p = render_end.to_period("M")
    while cur <= end_p:
        months.append(str(cur))
        cur = cur + 1
    return months


def chunk_parquet_path(forcing_root: Path, ym: str) -> Path:
    return forcing_root / "chunks" / ym / f"combined_{ym}.parquet"


def target_nc_path(target_root: Path, staid: str) -> Path:
    # STAID is preserved as string; no zero-padding applied
    return target_root / TARGET_NC_SUBDIR / f"{staid}.nc"


# ---------------------------------------------------------------------------
# Dry-run output
# ---------------------------------------------------------------------------
def dry_run_report(cases: list, forcing_root: Path, target_root: Path,
                   out_dir: Path, check_data: bool):
    """Print a complete dry-run validation report without reading any large files."""
    PASS = "OK"
    MISS = "MISSING"

    print()
    print("=" * 70)
    print("DRY-RUN REPORT — no animations generated")
    print("=" * 70)
    print(f"  Cases to animate: {[c['case_id'] for c in cases]}")
    print(f"  Forcing root:     {forcing_root}")
    print(f"  Target root:      {target_root}")
    print(f"  Output directory: {out_dir}")
    print()

    all_ok = True
    for c in cases:
        cid   = c["case_id"]
        staid = c["STAID"]   # string; no zfill
        rws   = c["render_window_start_utc"]
        rwe   = c["render_window_end_utc"]
        rh    = c["rendered_window_hours"]
        clp   = c["window_clipped_by_period"]
        gap   = c["product_gap_context"]
        cat   = c["selection_category"]
        name  = c.get("basin_name", "")

        rstart = pd.Timestamp(rws)
        rend   = pd.Timestamp(rwe)
        months = months_in_render_window(rstart, rend)

        print(f"  {'-'*66}")
        print(f"  Case:     {cid}  |  STAID: {staid}  |  {cat}")
        print(f"  Basin:    {name}")
        print(f"  Window:   render_window_start_utc = {rws}  (nominal ws = {c['window_start_utc']})")
        print(f"            render_window_end_utc   = {rwe}")
        print(f"            rendered_window_hours   = {rh}  |  clipped: {clp}")
        print(f"  Gap ctx:  {gap}")
        if clp == "true":
            print(f"  *** BOUNDARY CASE: render starts at period start; "
                  f"animation uses render_window_start_utc, NOT window_start_utc ***")

        # VQC-001 specific note
        if cid == "VQC-001":
            print(f"  *** VQC-001: MRMS archive-start gap "
                  f"{VQC001_ARCHIVE_GAP_START}–{VQC001_ARCHIVE_GAP_END} "
                  f"(21h NaN) will be labeled in animation ***")

        # Expected Parquet files
        print(f"  Forcing months needed: {months}")
        for ym in months:
            pq = chunk_parquet_path(forcing_root, ym)
            if check_data:
                status = PASS if pq.exists() else MISS
                flag = "" if status == PASS else " <-- MISSING"
                all_ok = all_ok and (status == PASS)
            else:
                status = "not-checked (--check-data not set)"
                flag = ""
            print(f"    Parquet [{status}]{flag}: {pq}")

        # Expected target NC
        nc = target_nc_path(target_root, staid)
        if check_data:
            status = PASS if nc.exists() else MISS
            flag = "" if status == PASS else " <-- MISSING"
            all_ok = all_ok and (status == PASS)
        else:
            status = "not-checked"
            flag = ""
        print(f"  Target NC [{status}]{flag}: {nc}")

        # Expected output files
        case_dir = out_dir / cid
        print(f"  Expected outputs:")
        print(f"    {case_dir / (cid + ANIM_SUFFIX_GIF)}")
        print(f"    {case_dir / (cid + QUICKLOOK_SUFFIX)}")
        print()

    print(f"  Manifest: {out_dir / MANIFEST_CSV}")
    print(f"  Summary:  {out_dir / SUMMARY_MD}")
    print()
    if check_data:
        print(f"DRY-RUN DATA CHECK: {'PASS — all required files found' if all_ok else 'FAIL — see MISSING entries above'}")
    else:
        print("DRY-RUN COMPLETE (pass --check-data to also verify data paths exist).")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_basin_forcing(forcing_root: Path, staid: str,
                       render_start: pd.Timestamp, render_end: pd.Timestamp
                       ) -> tuple:
    """
    Load MRMS QPE and RTMA 2t for one basin from the combined monthly Parquets.
    Returns (mrms_series, t2m_series_celsius) indexed by hourly UTC timestamps.
    Missing hours (gaps) appear as NaN.
    """
    months = months_in_render_window(render_start, render_end)
    mrms_frames = []
    rtma_frames = []

    for ym in months:
        pq_path = chunk_parquet_path(forcing_root, ym)
        if not pq_path.exists():
            print(f"  WARNING: Parquet not found: {pq_path} — hours in {ym} will be NaN")
            continue
        try:
            # Load only needed columns; STAID filter reduces rows read
            df = pd.read_parquet(
                pq_path,
                columns=PARQUET_LOAD_COLS,
                filters=[("STAID", "==", staid)],
            )
        except Exception as e:
            print(f"  WARNING: failed to read {pq_path}: {e}")
            continue

        df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"], utc=True)
        # Clip to render window
        df = df[(df["valid_time_utc"] >= render_start) & (df["valid_time_utc"] <= render_end)]

        mrms_frames.append(
            df[df["product"] == MRMS_PRODUCT][["valid_time_utc", "weighted_mean"]]
        )
        rtma_t2 = df[(df["product"] == RTMA_PRODUCT) & (df["variable"] == RTMA_VAR_T2M)]
        rtma_frames.append(rtma_t2[["valid_time_utc", "weighted_mean"]])

    full_idx = pd.date_range(render_start, render_end, freq="h", tz="UTC")

    def to_series(frames, name):
        if not frames:
            return pd.Series(np.nan, index=full_idx, name=name)
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values("valid_time_utc").drop_duplicates("valid_time_utc")
        return combined.set_index("valid_time_utc")["weighted_mean"].reindex(full_idx)

    mrms_s = to_series(mrms_frames, "mrms_mm")
    t2m_s  = to_series(rtma_frames, "t2m_C")
    t2m_s  = t2m_s - 273.15   # Kelvin → Celsius

    return mrms_s, t2m_s


def load_qobs(target_root: Path, staid: str,
              render_start: pd.Timestamp, render_end: pd.Timestamp) -> pd.Series:
    """
    Load qobs_m3s from the v001 target package NetCDF for one basin.
    Returns a Series indexed by hourly UTC timestamps; NaN where missing.
    STAID preserved as string — no zero-padding applied.
    """
    nc_path = target_nc_path(target_root, staid)
    full_idx = pd.date_range(render_start, render_end, freq="h", tz="UTC")

    if not nc_path.exists():
        print(f"  WARNING: target NC not found: {nc_path} — qobs will be all NaN")
        return pd.Series(np.nan, index=full_idx, name="qobs_m3s")

    try:
        import xarray as xr
        with xr.open_dataset(nc_path, mask_and_scale=True) as ds:
            times = pd.DatetimeIndex(ds[TARGET_NC_TIME_DIM].values, tz="UTC")
            qv    = ds[TARGET_NC_QOBS_VAR].values.squeeze().astype(float)
        s = pd.Series(qv, index=times, name="qobs_m3s")
        return s.loc[render_start:render_end].reindex(full_idx)
    except Exception as e:
        print(f"  WARNING: failed to read qobs from {nc_path}: {e}")
        return pd.Series(np.nan, index=full_idx, name="qobs_m3s")


# ---------------------------------------------------------------------------
# Figure / animation helpers
# ---------------------------------------------------------------------------
def _map_params_from_area(area_km2: float) -> dict:
    """Area-based colormap vmax for MRMS QPE display."""
    if area_km2 < 20:
        vmax = 25.0
    elif area_km2 < 100:
        vmax = 15.0
    elif area_km2 < 500:
        vmax = 10.0
    else:
        vmax = 8.0
    return {"mrms_vmax": vmax}


def _shade_gap_spans(ax, gap_times, t0, t1, color="#cccccc", alpha=0.50, zorder=0):
    """Shade horizontal spans at gap hours (NaN positions in the time index)."""
    if not len(gap_times):
        return
    dt = pd.Timedelta("30min")
    for gt in gap_times:
        ax.axvspan(gt - dt, gt + dt, color=color, alpha=alpha, zorder=zorder)


def _make_figure(case: dict, mrms_s: pd.Series, t2m_s: pd.Series, qobs_s: pd.Series,
                 fps: int, dpi: int) -> tuple:
    """Create and return the base 3-panel figure (axes + static elements)."""
    cid    = case["case_id"]
    staid  = case["STAID"]
    name   = case.get("basin_name", "")[:50]
    cat    = case["selection_category"]
    rws    = case["render_window_start_utc"]
    rwe    = case["render_window_end_utc"]
    rh     = case["rendered_window_hours"]
    clp    = case["window_clipped_by_period"] == "true"
    gap_ctx = case.get("product_gap_context", "")
    area   = float(case.get("drain_sqkm") or 0)
    state  = case.get("state", "")

    clp_note = "  [render window CLIPPED to period start]" if clp else ""
    title = (f"Flash-NH Stage 1 | Full-period Visual QC | {cid}  "
             f"STAID {staid}  ({state})  {cat}\n"
             f"{name}  |  {area:.0f} km²  |  "
             f"Render: {rws[:10]} → {rwe[:10]}  ({rh}h){clp_note}")

    fig = plt.figure(figsize=(FIGURE_W, FIGURE_H), dpi=dpi)
    fig.suptitle(title, fontsize=8.5, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(
        3, 1, figure=fig,
        hspace=0.15, top=0.90, bottom=0.08, left=0.09, right=0.97,
        height_ratios=[2, 1, 1.5])
    ax_mrms = fig.add_subplot(gs[0])
    ax_t2m  = fig.add_subplot(gs[1], sharex=ax_mrms)
    ax_qobs = fig.add_subplot(gs[2], sharex=ax_mrms)

    t0 = mrms_s.index[0]
    t1 = mrms_s.index[-1]

    # -- MRMS panel (bar chart) ---
    mrms_vals = mrms_s.values.copy()
    bar_colors = np.where(np.isnan(mrms_vals), "#dddddd", "#3a5fa0")
    bar_heights = np.where(np.isnan(mrms_vals), 0.0, mrms_vals)
    bar_w = pd.Timedelta("50min")
    ax_mrms.bar(mrms_s.index, bar_heights, width=bar_w,
                color=bar_colors, alpha=0.75, align="center", zorder=2)

    # VQC-001: shade archive-start MRMS gap in red
    if cid == "VQC-001":
        gap_s = pd.Timestamp(VQC001_ARCHIVE_GAP_START, tz="UTC")
        gap_e = pd.Timestamp(VQC001_ARCHIVE_GAP_END, tz="UTC")
        ax_mrms.axvspan(gap_s - pd.Timedelta("30min"),
                        gap_e + pd.Timedelta("30min"),
                        color="#ffcccc", alpha=0.60, zorder=0,
                        label="MRMS archive-start gap (T00Z–T20Z, 21h)")
        ax_mrms.text(gap_s + (gap_e - gap_s) / 2, 0.85,
                     "MRMS archive-start\ngap (T00Z–T20Z)",
                     transform=ax_mrms.get_xaxis_transform(),
                     ha="center", va="top", fontsize=6.5, color="#cc0000",
                     fontweight="bold",
                     bbox=dict(fc="white", ec="#cc0000", alpha=0.85, pad=1.5), zorder=10)
        # Mark period start / clipped render start
        ax_mrms.axvline(t0, color="#888800", lw=1.5, ls="--", zorder=3,
                        label=f"Render start (period start, clipped)")
        ax_mrms.text(t0, 1.02, "Period start\n(render window\nclipped here)",
                     transform=ax_mrms.get_xaxis_transform(),
                     ha="left", va="bottom", fontsize=5.5, color="#888800")
    else:
        # Shade non-VQC-001 MRMS gaps
        mrms_gap_idx = mrms_s.index[np.isnan(mrms_vals)]
        _shade_gap_spans(ax_mrms, mrms_gap_idx, t0, t1,
                         color="#ffcccc", alpha=0.45)

    ax_mrms.set_ylabel("MRMS 1h QPE\n(basin mean, mm)", fontsize=7.5)
    ax_mrms.set_ylim(bottom=0)
    ax_mrms.yaxis.set_major_locator(mticker.MaxNLocator(5))
    ax_mrms.tick_params(axis="both", labelsize=7)
    plt.setp(ax_mrms.get_xticklabels(), visible=False)
    if cid == "VQC-001":
        ax_mrms.legend(fontsize=5.5, loc="upper right")

    # -- RTMA 2t panel ---
    t2m_gap_idx = t2m_s.index[np.isnan(t2m_s.values)]
    _shade_gap_spans(ax_t2m, t2m_gap_idx, t0, t1, color="#ffe0cc", alpha=0.55)
    ax_t2m.plot(t2m_s.index, t2m_s.values, color="darkorange", lw=1.3, zorder=2)
    ax_t2m.axhline(0, color="deepskyblue", lw=0.9, ls="--", alpha=0.65, zorder=1,
                   label="0 °C")
    ax_t2m.set_ylabel("RTMA 2m T (°C)", fontsize=7.5)
    ax_t2m.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax_t2m.tick_params(axis="both", labelsize=7)
    plt.setp(ax_t2m.get_xticklabels(), visible=False)

    # -- Streamflow panel ---
    qobs_gap_idx = qobs_s.index[np.isnan(qobs_s.values)]
    ax_qobs.plot(qobs_s.index, qobs_s.values, color="steelblue", lw=1.3, zorder=2)
    ax_qobs.fill_between(qobs_s.index, 0, qobs_s.fillna(0),
                         alpha=0.12, color="steelblue", zorder=1)
    _shade_gap_spans(ax_qobs, qobs_gap_idx, t0, t1, color="orange", alpha=0.25)
    ax_qobs.set_ylabel("Streamflow\nq$_{obs}$ (m³/s)", fontsize=7.5)
    ax_qobs.set_ylim(bottom=0)
    ax_qobs.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax_qobs.tick_params(axis="both", labelsize=7)
    ax_qobs.set_xlabel("Valid time (UTC)", fontsize=7)
    ax_qobs.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%HZ"))
    ax_qobs.xaxis.set_major_locator(mdates.HourLocator(interval=12))

    for ax in (ax_mrms, ax_t2m, ax_qobs):
        ax.set_xlim(t0, t1)

    fig.canvas.draw()
    return fig, (ax_mrms, ax_t2m, ax_qobs)


def _render_frame(fig, axes, ts: pd.Timestamp, fi: int, n_frames: int,
                  mrms_s: pd.Series, t2m_s: pd.Series, qobs_s: pd.Series,
                  vlines: list, case_id: str):
    """
    Draw the per-frame cursor lines on the three panels.
    Uses add/remove artist approach to avoid full redraw of static content.
    """
    ax_mrms, ax_t2m, ax_qobs = axes
    # Remove previous cursor lines
    for vl in vlines:
        try:
            vl.remove()
        except Exception:
            pass
    vlines.clear()

    for ax in (ax_mrms, ax_t2m, ax_qobs):
        vl = ax.axvline(ts, color="red", lw=1.3, zorder=5)
        vlines.append(vl)

    # Highlight current MRMS bar
    mrms_val = mrms_s.get(ts)
    t2m_val  = t2m_s.get(ts)
    q_val    = qobs_s.get(ts)

    # Frame timestamp annotation on MRMS panel
    # Remove previous text if it exists
    for txt in ax_mrms.texts:
        if getattr(txt, "_is_frame_label", False):
            txt.remove()
    txt = ax_mrms.text(
        0.99, 0.97,
        f"{ts.strftime('%Y-%m-%d %H:%M UTC')}  frame {fi+1}/{n_frames}",
        transform=ax_mrms.transAxes, ha="right", va="top",
        fontsize=7, bbox=dict(fc="white", alpha=0.85, pad=2, ec="none"), zorder=10)
    txt._is_frame_label = True
    fig.canvas.draw()


# ---------------------------------------------------------------------------
# Quicklook PNG (static — shows no cursor)
# ---------------------------------------------------------------------------
def save_quicklook(fig, case_dir: Path, case_id: str, dpi: int):
    path = case_dir / (case_id + QUICKLOOK_SUFFIX)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# ---------------------------------------------------------------------------
# Per-case animation driver
# ---------------------------------------------------------------------------
def animate_case(case: dict, forcing_root: Path, target_root: Path,
                  out_dir: Path, fps: int, dpi: int,
                  anim_format: str, overwrite: bool) -> dict:
    cid   = case["case_id"]
    staid = case["STAID"]       # string, no zfill — preserves 9-digit STAIDs
    rws   = case["render_window_start_utc"]
    rwe   = case["render_window_end_utc"]
    rh    = int(case["rendered_window_hours"])
    clp   = case["window_clipped_by_period"] == "true"

    t0_case = time.time()
    result = {
        "case_id":   cid, "STAID": staid,
        "basin_name": case.get("basin_name", ""),
        "category": case.get("selection_category", ""),
        "month": case.get("month", ""),
        "render_window_start_utc": rws, "render_window_end_utc": rwe,
        "rendered_window_hours": rh,
        "window_clipped_by_period": case.get("window_clipped_by_period", ""),
        "product_gap_context": case.get("product_gap_context", ""),
        "n_frames": 0, "fps": fps,
        "n_mrms_missing_frames": 0, "n_rtma_missing_frames": 0,
        "n_qobs_nan_hours": 0, "output_format": "",
        "output_path": "", "quicklook_path": "",
        "status": "PENDING", "error": "", "runtime_s": 0.0,
    }

    case_dir = out_dir / cid
    anim_suffix = ANIM_SUFFIX_MP4 if (anim_format == "mp4" and shutil.which("ffmpeg")) \
                  else ANIM_SUFFIX_GIF
    anim_path = case_dir / (cid + anim_suffix)

    if anim_path.exists() and not overwrite:
        print(f"  {cid}: already exists, skipping (use --overwrite to force)")
        result["status"] = "SKIPPED"
        result["output_path"] = str(anim_path)
        return result

    case_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 64}")
    print(f"  {cid}  STAID={staid}  render={rws[:10]} → {rwe[:10]}  ({rh}h)")
    print(f"{'=' * 64}")

    try:
        render_start = pd.Timestamp(rws)
        render_end   = pd.Timestamp(rwe)

        # Load data
        print(f"  Loading MRMS+RTMA from Parquet …")
        mrms_s, t2m_s = load_basin_forcing(
            forcing_root, staid, render_start, render_end)

        print(f"  Loading qobs from target NC …")
        qobs_s = load_qobs(target_root, staid, render_start, render_end)

        n_mrms_miss = int(np.isnan(mrms_s.values).sum())
        n_rtma_miss = int(np.isnan(t2m_s.values + 273.15).sum())
        n_qobs_nan  = int(np.isnan(qobs_s.values).sum())
        print(f"  Data: MRMS {rh - n_mrms_miss}/{rh}h, "
              f"RTMA-t2m {rh - n_rtma_miss}/{rh}h, "
              f"qobs {rh - n_qobs_nan}/{rh}h")

        result["n_mrms_missing_frames"] = n_mrms_miss
        result["n_rtma_missing_frames"] = n_rtma_miss
        result["n_qobs_nan_hours"]       = n_qobs_nan

        # Build figure
        print(f"  Building figure …")
        fig, axes = _make_figure(case, mrms_s, t2m_s, qobs_s, fps, dpi)

        # Save quicklook (no cursor)
        ql_path = save_quicklook(fig, case_dir, cid, dpi)
        print(f"  Quicklook -> {ql_path}")
        result["quicklook_path"] = str(ql_path)

        # Animate
        frame_times = pd.date_range(render_start, render_end, freq="h", tz="UTC")
        n_frames = len(frame_times)
        result["n_frames"] = n_frames

        print(f"  Rendering {n_frames} frames → {anim_path.name} …")
        vlines = []
        writer = PillowWriter(fps=fps)
        with writer.saving(fig, str(anim_path), dpi=dpi):
            for fi, ts in enumerate(frame_times):
                _render_frame(fig, axes, ts, fi, n_frames,
                              mrms_s, t2m_s, qobs_s, vlines, cid)
                writer.grab_frame()
                if (fi + 1) % 10 == 0 or fi == n_frames - 1:
                    print(f"    frame {fi+1:3d}/{n_frames}  {ts.strftime('%m-%d %HZ')}")

        plt.close(fig)
        print(f"  Animation -> {anim_path}")
        result["output_format"] = anim_suffix.lstrip("_").upper().rstrip(".")
        result["output_path"]   = str(anim_path)
        result["status"]        = "OK"

    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        result["status"] = "FAIL"
        result["error"]  = str(e)
        try:
            plt.close("all")
        except Exception:
            pass

    result["runtime_s"] = round(time.time() - t0_case, 1)
    return result


# ---------------------------------------------------------------------------
# Manifest and summary writers
# ---------------------------------------------------------------------------
def write_manifest_csv(out_dir: Path, results: list):
    path = out_dir / MANIFEST_CSV
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS,
                           extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(results)
    return path


def write_summary_md(out_dir: Path, results: list,
                     start_utc: str, elapsed_s: float, args):
    n_ok   = sum(1 for r in results if r["status"] == "OK")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_skip = sum(1 for r in results if r["status"] == "SKIPPED")

    lines = [
        "# Stage 1 Forcing — Full-Period Visual QC Animation Pilot",
        "",
        f"**Generated:** {start_utc}  ",
        f"**Cases attempted:** {len(results)}  ",
        f"**OK:** {n_ok}  **FAIL:** {n_fail}  **SKIPPED:** {n_skip}  ",
        f"**Total runtime:** {elapsed_s:.0f}s  ",
        f"**Output directory:** {out_dir}  ",
        "",
        "---",
        "",
        "| Case | STAID | Category | Window | Rend h | Clipped | Gap ctx | "
        "MRMS miss | Status | Runtime |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['case_id']} | {r['STAID']} | {r['category']} | "
            f"{r['render_window_start_utc'][:10]} | {r['rendered_window_hours']} | "
            f"{'YES' if r['window_clipped_by_period'] == 'true' else ''} | "
            f"{r['product_gap_context']} | {r['n_mrms_missing_frames']}h | "
            f"{r['status']} | {r['runtime_s']:.0f}s |"
        )
    lines += [
        "",
        "---",
        "",
        "## VQC-001 Boundary Case Note",
        "",
        "VQC-001's render window is clipped to the Stage 1 forcing period start "
        "(2020-10-14T00:00:00Z). The animation covers 34 hours. MRMS hours "
        "T00Z–T20Z (21h) are permanently absent from S3 (archive-start gap) "
        "and appear as NaN bars with a red-shaded region in the animation.",
        "",
        "## Animation Review Instructions",
        "",
        "For each case, review:",
        "1. MRMS 1h QPE: plausible precipitation pattern for month/category?",
        "2. RTMA 2m T: realistic temperatures? Freezing/snow context for winter cases?",
        "3. Streamflow: does the hydrograph respond to the QPE event?",
        "4. Gap frames: NaN hours rendered as gray bars / shading — no phantom values?",
        "5. VQC-001: archive-start gap correctly shows 21 NaN frames with label?",
        "",
        "Record outcomes in the `reviewer` and `review_outcome` columns of "
        "`visual_qc_case_selection.csv` and return the filled CSV alongside "
        "the animations.",
        "",
        "---",
        "",
        f"*Animations are not committed to git. Script: "
        f"`scripts/generate_fullperiod_visual_qc_animations.py`.*",
    ]
    path = out_dir / SUMMARY_MD
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Resolve paths (no hardcoded user paths)
    forcing_root = Path(args.forcing_root)
    target_root  = Path(args.target_package_root)
    out_dir      = Path(args.out_dir)

    # Load and validate cases
    all_cases = load_case_selection(args.case_selection_csv)
    cases     = validate_and_select_cases(all_cases, args.case_ids, args.max_cases)

    start_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0_total  = time.time()

    print(f"\nFlash-NH Stage 1 — Full-period visual QC animation pilot")
    print(f"  Mode:        {'DRY-RUN' if args.dry_run else 'RENDER'}")
    print(f"  Cases:       {[c['case_id'] for c in cases]}")
    print(f"  Forcing:     {forcing_root}")
    print(f"  Target pkg:  {target_root}")
    print(f"  Output:      {out_dir}")
    print(f"  Format:      {args.anim_format}  fps={args.fps}  dpi={args.dpi}")

    if args.dry_run:
        dry_run_report(cases, forcing_root, target_root, out_dir,
                       check_data=args.check_data)
        return 0

    # Check ffmpeg once
    ffmpeg_ok = (args.anim_format == "mp4" and shutil.which("ffmpeg") is not None)
    if args.anim_format == "mp4" and not ffmpeg_ok:
        print("  ffmpeg not found — falling back to GIF output")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Animate each case
    results = []
    for case in cases:
        res = animate_case(
            case, forcing_root, target_root, out_dir,
            fps=args.fps, dpi=args.dpi,
            anim_format=args.anim_format, overwrite=args.overwrite)
        results.append(res)

    elapsed = time.time() - t0_total

    # Write manifest and summary
    mcsv = write_manifest_csv(out_dir, results)
    smd  = write_summary_md(out_dir, results, start_utc, elapsed, args)

    print(f"\n{'=' * 64}")
    print("ANIMATION SUMMARY")
    print(f"{'=' * 64}")
    n_ok   = sum(1 for r in results if r["status"] == "OK")
    n_fail = sum(1 for r in results if r["status"] != "OK" and r["status"] != "SKIPPED")
    for r in results:
        print(f"  {r['case_id']}  {r['status']:8s}  "
              f"MRMS_miss={r['n_mrms_missing_frames']}h  "
              f"{r['runtime_s']:.0f}s  "
              f"{'-> ' + r['output_path'] if r['output_path'] else ''}")
    print(f"\n  {n_ok}/{len(results)} OK   {n_fail} failed   "
          f"Total: {elapsed:.0f}s")
    print(f"  Manifest: {mcsv}")
    print(f"  Summary:  {smd}")
    print(f"{'=' * 64}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
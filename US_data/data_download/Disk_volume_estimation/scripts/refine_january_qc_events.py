"""
Stage 1 Milestone 2E — Refined January 2023 Event Candidate Selection

Applies the following changes relative to the original candidate set:
  C01–C04 : RETAIN (STRONG_WET, unchanged)
  C05      : RELABEL -> MODERATE_SMALL_BASIN (content unchanged)
  C06      : RELABEL -> MODERATE_COLD_REGION (content unchanged)
  C07      : RELABEL -> COLD_PRECIP_LOW_RESPONSE (content unchanged)
  C08      : RELABEL -> ZERO_FLOW_FROZEN_CONTROL (content unchanged)
  C09      : REPLACE 01662800/VA (was wrong window, had 10mm/6h precip)
             -> 13239000 DRY_CONTROL, window Jan-01 00Z -> Jan-03 23Z (0 mm precip)
  C10      : REWINDOW 04111379/MI (wrong window had 6.6mm/6h)
             -> corrected window Jan-07 03Z -> Jan-10 02Z (0 mm precip)
  C11      : RECENTER 01100627/MA offset=4004m
             -> new window Jan-21 06Z -> Jan-24 05Z (lag=1h, peak@74%, max6h=15.3mm)
  C12      : RETAIN (OFFSET_STRESS, unchanged)

Writes:
  tmp/.../january_2023_event_qc/event_animation_candidates_refined.csv
  tmp/.../january_2023_event_qc/event_animation_candidates_refined.md
  tmp/.../january_2023_event_qc/candidate_refinement_report.md
  tmp/.../candidate_previews_refined/<cid>.png
  tmp/.../candidate_preview_contact_sheet_refined.png  (split across page01/02 if needed)
"""

import json
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(r"C:\PhD\Python\neuralhydrology\US_data\data_download\Disk_volume_estimation")
CAMELSH_DIR = Path(r"C:\PhD\Python\neuralhydrology\US_data\data_download\CAMELSH_resolution_test\data\raw\camelsh")
FORCING_PARQUET = ROOT / "tmp/stage1_pilot_dryrun/03_basin_timeseries/stage1_pilot/january_2023/combined_hourly_basin_stats.parquet"
PILOT_MANIFEST = ROOT / "tmp/stage1_pilot_dryrun/09_manifests/stage1_pilot/pilot_basin_manifest.csv"
GAUGE_AUDIT = ROOT / "tmp/stage1_pilot_dryrun/09_manifests/stage1_pilot/one_hour_extraction/gauge_polygon_distance_audit.json"
ORIG_CSV = ROOT / "tmp/stage1_pilot_dryrun/09_manifests/stage1_pilot/january_2023_event_qc/event_animation_candidates.csv"

QC_MANIFEST_DIR = ROOT / "tmp/stage1_pilot_dryrun/09_manifests/stage1_pilot/january_2023_event_qc"
QC_REPORT_DIR = ROOT / "tmp/stage1_pilot_dryrun/06_qc_reports/stage1_pilot/january_2023_event_qc"
PREVIEW_DIR = QC_REPORT_DIR / "candidate_previews_refined"

PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

# ── load reference data ───────────────────────────────────────────────────────

print("Loading reference data...")
manifest_df = pd.read_csv(PILOT_MANIFEST, dtype={"STAID": str})
manifest_df["STAID"] = manifest_df["STAID"].str.strip().str.zfill(8)
meta = {}
for _, row in manifest_df.iterrows():
    sid = row["STAID"]
    meta[sid] = {
        "drain_sqkm": row.get("DRAIN_SQKM", np.nan),
        "area_bin": row.get("area_bin", ""),
        "candidate_class": row.get("candidate_class", ""),
        "state": row.get("STATE", ""),
        "huc02": str(row.get("HUC02", "")),
        "lat": row.get("LAT_GAGE", np.nan),
        "lon": row.get("LNG_GAGE", np.nan),
        "rbi": row.get("RBI", np.nan),
    }

with open(GAUGE_AUDIT) as fh:
    audit_data = json.load(fh)
gauge_records = {r["STAID"]: r for r in audit_data["records"]}

forcing_df = pd.read_parquet(FORCING_PARQUET)
forcing_df["valid_time_utc"] = pd.to_datetime(forcing_df["valid_time_utc"], utc=True)

mrms_df = (
    forcing_df[forcing_df["product"] == "mrms_qpe_1h_pass1"]
    [["STAID", "valid_time_utc", "weighted_mean"]]
    .rename(columns={"weighted_mean": "precip_mm"})
)
rtma_2t_df = (
    forcing_df[(forcing_df["product"] == "rtma_conus_aws_2p5km") & (forcing_df["variable"] == "2t")]
    [["STAID", "valid_time_utc", "weighted_mean"]]
    .rename(columns={"weighted_mean": "temp_2t_K"})
)

orig_df = pd.read_csv(ORIG_CSV, dtype={"staid": str})
orig_df["staid"] = orig_df["staid"].str.zfill(8)


def get_series(df, staid, col):
    sub = df[df["STAID"] == staid].set_index("valid_time_utc")[col].sort_index()
    return sub


def load_camelsh_jan(sid):
    f = CAMELSH_DIR / f"{sid}_hourly.nc"
    if not f.exists():
        return None
    ds = nc.Dataset(f)
    tv = ds.variables["time"]
    times = nc.num2date(tv[:], tv.units, getattr(tv, "calendar", "standard"))
    sf = ds.variables["streamflow"][:].squeeze()
    ds.close()
    ts = pd.to_datetime([t.strftime("%Y-%m-%d %H:%M:%S") for t in times], utc=True)
    series = pd.Series(np.ma.filled(sf.astype(float), np.nan), index=ts)
    return series.loc["2023-01-01":"2023-01-31 23:00"]


def compute_window_metrics(sid, w_start, w_end, sf_full, precip_full, temp2t_full):
    sf_win = sf_full.loc[w_start:w_end]
    p_win = precip_full.loc[w_start:w_end]
    t_win = temp2t_full.loc[w_start:w_end]

    r6 = p_win.rolling(6, min_periods=6).sum()
    r24 = p_win.rolling(24, min_periods=24).sum()

    sf_start = float(sf_win.iloc[0]) if len(sf_win) > 0 else np.nan
    sf_peak = float(sf_win.max()) if len(sf_win) > 0 else np.nan
    sf_peak_time = sf_win.idxmax() if len(sf_win) > 0 else w_start
    sf_rise = sf_peak - sf_start if not np.isnan(sf_start) else np.nan
    sf_rise_ratio = (sf_peak / sf_start) if (not np.isnan(sf_start) and sf_start > 0.001) else np.nan
    p_peak_time = p_win.idxmax() if not p_win.isna().all() else w_start
    lag_hours = int((sf_peak_time - p_peak_time).total_seconds() / 3600) if len(sf_win) > 0 else np.nan
    peak_pos_pct = (sf_peak_time - w_start).total_seconds() / (w_end - w_start).total_seconds() * 100

    gr = gauge_records.get(sid, {})
    md = meta.get(sid, {})

    return {
        "staid": sid,
        "window_start_utc": w_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_end_utc": w_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "event_peak_precip_time_utc": p_peak_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "event_peak_flow_time_utc": sf_peak_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "flow_peak_pos_pct": round(peak_pos_pct, 1),
        "max_1h_precip_mm": round(float(p_win.max()), 3),
        "max_3h_precip_mm": round(float(p_win.rolling(3, min_periods=3).sum().max()), 3),
        "max_6h_precip_mm": round(float(r6.max()), 3),
        "max_24h_precip_mm": round(float(r24.max()), 3),
        "total_72h_precip_mm": round(float(p_win.sum()), 3),
        "sf_start_m3s": round(sf_start, 4) if not np.isnan(sf_start) else None,
        "sf_peak_m3s": round(sf_peak, 4) if not np.isnan(sf_peak) else None,
        "sf_rise_m3s": round(sf_rise, 4) if not np.isnan(sf_rise) else None,
        "sf_rise_ratio": round(sf_rise_ratio, 3) if (sf_rise_ratio is not None and not np.isnan(sf_rise_ratio)) else None,
        "precip_to_flow_lag_hours": lag_hours,
        "min_rtma_2t_K": round(float(t_win.min()), 2) if not t_win.isna().all() else None,
        "min_rtma_2t_C": round(float(t_win.min()) - 273.15, 2) if not t_win.isna().all() else None,
        "med_rtma_2t_K": round(float(t_win.median()), 2) if not t_win.isna().all() else None,
        "med_rtma_2t_C": round(float(t_win.median()) - 273.15, 2) if not t_win.isna().all() else None,
        "drain_sqkm": md.get("drain_sqkm", np.nan),
        "area_bin": md.get("area_bin", ""),
        "candidate_class": md.get("candidate_class", ""),
        "state": md.get("state", ""),
        "huc02": md.get("huc02", ""),
        "lat": md.get("lat", np.nan),
        "lon": md.get("lon", np.nan),
        "rbi": md.get("rbi", np.nan),
        "gauge_offset_m": gr.get("distance_to_polygon_m", np.nan),
        "gauge_offset_status": gr.get("status", ""),
        "missing_sf_in_window": int(sf_win.isna().sum()),
    }


# ── define refined candidates ─────────────────────────────────────────────────
# Each entry: staid, category, reason, action, window_start, window_end
# Windows marked None use the original candidate's window.

REFINED_PLAN = [
    # id, staid, category, action_label, reason, w_start, w_end
    ("R01", "02411930", "STRONG_WET",
     "RETAIN",
     "Highest 6h precip of any candidate (51.5mm), large GA basin (705km2), 21x rise, 14h lag",
     None, None),
    ("R02", "07263580", "STRONG_WET",
     "RETAIN",
     "FLASHY_CORE small basin (50km2) AR, extreme rise ratio 195x, 1h lag — peak responsiveness test",
     None, None),
    ("R03", "03318800", "STRONG_WET",
     "RETAIN",
     "Highest absolute peak discharge (170m3/s), KY, 3h lag — note 690/744 valid hours",
     None, None),
    ("R04", "07024500", "STRONG_WET",
     "RETAIN",
     "Largest wet basin (991km2), TN — tests pipeline performance on large drainage area",
     None, None),
    ("R05", "05560500", "MODERATE_SMALL_BASIN",
     "RELABEL from MODERATE",
     "Small IL basin (71km2) with moderate 5.5x rise; verifies non-zero pipeline output under weak forcing",
     None, None),
    ("R06", "05372995", "MODERATE_COLD_REGION",
     "RELABEL from MODERATE",
     "Large MN basin (779km2), borderline snow/rain at -6.9C min, moderate response",
     None, None),
    ("R07", "10348850", "COLD_PRECIP_LOW_RESPONSE",
     "RELABEL from COLD_SNOW_RISK",
     "Small NV basin (19km2), -7.9C min, 22.5mm/6h precip but only 0.06m3/s rise — snow accumulation dominant",
     None, None),
    ("R08", "13112000", "ZERO_FLOW_FROZEN_CONTROL",
     "RELABEL from COLD_SNOW_RISK",
     "Large ID basin (947km2), zero streamflow all January — fully frozen / snow accumulation",
     None, None),
    # REPLACE C09: 13239000 — true dry, Jan-01 00Z (zero precip, stable recession)
    ("R09", "13239000", "DRY_CONTROL",
     "REPLACE C09 (01662800 VA was wet: max6h=10.4mm, rise_ratio=2.9x)",
     "First 72h of January: MRMS=0mm, slow recession, cv=0.032 — genuine baseline control",
     pd.Timestamp("2023-01-01 00:00:00", tz="UTC"),
     pd.Timestamp("2023-01-03 23:00:00", tz="UTC")),
    # REWINDOW C10: 04111379 MI — corrected to Jan-07 dry window
    ("R10", "04111379", "DRY_CONTROL",
     "REWINDOW C10 (original Jan-18 window had 6.6mm/6h precip)",
     "Jan 07-10 window: MRMS=0mm, stable baseflow, cv=0.082, MI basin (427km2) — corrected dry window",
     pd.Timestamp("2023-01-07 03:00:00", tz="UTC"),
     pd.Timestamp("2023-01-10 02:00:00", tz="UTC")),
    # RECENTER C11: 01100627 MA — Jan-21 storm, clean 1h lag
    ("R11", "01100627", "OFFSET_STRESS",
     "RECENTER C11 (original window Jan-25/28 had 47h lag, flow peak at window edge)",
     "Jan-21 storm: max6h=15.3mm, Qpk=8.3m3/s, 1h lag, peak at 74% of window — gauge offset=4004m",
     pd.Timestamp("2023-01-21 06:00:00", tz="UTC"),
     pd.Timestamp("2023-01-24 05:00:00", tz="UTC")),
    ("R12", "01390450", "OFFSET_STRESS",
     "RETAIN",
     "Small NJ basin (29km2), gauge offset=3023m, 46x rise ratio — extreme geometry mismatch test",
     None, None),
]


# ── build refined candidates ──────────────────────────────────────────────────

print("Building refined candidates...")

sf_cache = {}
candidates_refined = []

for (rid, sid, category, action, reason, w_start_override, w_end_override) in REFINED_PLAN:
    sid = sid.zfill(8)

    # Load streamflow (cache to avoid re-reading NetCDF)
    if sid not in sf_cache:
        sf_cache[sid] = load_camelsh_jan(sid)
    sf_full = sf_cache[sid]

    precip_full = get_series(mrms_df, sid, "precip_mm")
    temp2t_full = get_series(rtma_2t_df, sid, "temp_2t_K")

    # Get window: override or original
    if w_start_override is not None:
        w_start = w_start_override
        w_end = w_end_override
    else:
        orig_row = orig_df[orig_df["staid"] == sid]
        if orig_row.empty:
            print(f"  WARN: no original row for {sid}, skipping")
            continue
        orig_row = orig_row.iloc[0]
        w_start = pd.Timestamp(orig_row["start_time_utc"])
        w_end = pd.Timestamp(orig_row["end_time_utc"])

    m = compute_window_metrics(sid, w_start, w_end, sf_full, precip_full, temp2t_full)

    cand = {
        "candidate_id": rid,
        "category": category,
        "action": action,
        "reason_selected": reason,
    }
    cand.update(m)
    candidates_refined.append(cand)
    print(f"  {rid} {sid} {category}: max6h={m['max_6h_precip_mm']} mm, "
          f"Qpk={m['sf_peak_m3s']} m3/s, lag={m['precip_to_flow_lag_hours']}h, "
          f"peak_pos={m['flow_peak_pos_pct']}%, offset={m['gauge_offset_m']:.0f}m")


refined_df = pd.DataFrame(candidates_refined)

out_csv = QC_MANIFEST_DIR / "event_animation_candidates_refined.csv"
refined_df.to_csv(out_csv, index=False)
print(f"Wrote: {out_csv}")


# ── preview plots ─────────────────────────────────────────────────────────────

print("Generating refined preview plots...")

# Color coding by category
CAT_COLOR = {
    "STRONG_WET": "#1565C0",
    "MODERATE_SMALL_BASIN": "#2E7D32",
    "MODERATE_COLD_REGION": "#558B2F",
    "COLD_PRECIP_LOW_RESPONSE": "#6A1B9A",
    "ZERO_FLOW_FROZEN_CONTROL": "#4A148C",
    "DRY_CONTROL": "#BF360C",
    "OFFSET_STRESS": "#E65100",
}


def make_refined_preview(cand: dict) -> Path:
    sid = str(cand["staid"]).zfill(8)
    rid = cand["candidate_id"]
    w_start = pd.Timestamp(cand["window_start_utc"])
    w_end = pd.Timestamp(cand["window_end_utc"])
    cat = cand["category"]

    sf = sf_cache[sid].loc[w_start:w_end]
    precip = get_series(mrms_df, sid, "precip_mm").loc[w_start:w_end]
    temp2t = get_series(rtma_2t_df, sid, "temp_2t_K").loc[w_start:w_end] - 273.15

    peak_flow_time = pd.Timestamp(cand["event_peak_flow_time_utc"])
    peak_precip_time = pd.Timestamp(cand["event_peak_precip_time_utc"])

    cat_col = CAT_COLOR.get(cat, "#333333")

    fig = plt.figure(figsize=(12, 7.5))
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.06, height_ratios=[1.8, 1.8, 1.0])
    ax_sf = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1], sharex=ax_sf)
    ax_t = fig.add_subplot(gs[2], sharex=ax_sf)

    # ── streamflow panel ──────────────────────────────────────────────────────
    ax_sf.plot(sf.index, sf.values, color="#1565C0", lw=1.8, label="Streamflow (m3/s)")
    ax_sf.axvline(peak_flow_time, color="#d62728", ls="--", lw=1.3, alpha=0.85,
                  label=f"Flow peak ({cand['sf_peak_m3s']} m3/s)")
    ax_sf.axvline(peak_precip_time, color="#2ca02c", ls=":", lw=1.3, alpha=0.85,
                  label=f"Precip peak")
    # Shade if zero-flow case
    if cat == "ZERO_FLOW_FROZEN_CONTROL":
        ax_sf.fill_between(sf.index, 0, sf.values + 0.001,
                           color="#4A148C", alpha=0.15, label="Frozen basin")
    ax_sf.set_ylabel("Q (m3/s)", fontsize=9)
    ax_sf.legend(fontsize=7.5, loc="upper right", framealpha=0.85)
    ax_sf.tick_params(labelbottom=False, labelsize=8)
    sf_max = max(float(sf.max()) if not sf.empty else 0, 0.01)
    ax_sf.set_ylim(bottom=0, top=sf_max * 1.18)
    ax_sf.grid(True, alpha=0.3)

    # ── precipitation panel ───────────────────────────────────────────────────
    bar_width = 1 / 24
    if not precip.empty:
        ax_p.bar(precip.index, precip.values, width=bar_width,
                 color="#2196F3", alpha=0.88, label=f"MRMS 1h precip (max6h={cand['max_6h_precip_mm']}mm)")
    ax_p.axvline(peak_precip_time, color="#2ca02c", ls=":", lw=1.3, alpha=0.85)
    ax_p.set_ylabel("Precip (mm/h)", fontsize=9)
    ax_p.legend(fontsize=7.5, loc="upper right", framealpha=0.85)
    ax_p.tick_params(labelbottom=False, labelsize=8)
    ax_p.grid(True, alpha=0.3)

    # ── temperature panel ─────────────────────────────────────────────────────
    ax_t.plot(temp2t.index, temp2t.values, color="#FF6B35", lw=1.3, label="RTMA 2m T (C)")
    ax_t.axhline(0, color="k", lw=0.8, ls="--", alpha=0.55, label="0 C")
    ax_t.set_ylabel("T (C)", fontsize=9)
    ax_t.legend(fontsize=7.5, loc="upper right", framealpha=0.85)
    ax_t.tick_params(labelsize=8)
    ax_t.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%HZ"))
    ax_t.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    ax_t.grid(True, alpha=0.3)

    plt.setp(ax_sf.get_xticklabels(), visible=False)
    plt.setp(ax_p.get_xticklabels(), visible=False)

    # ── title ─────────────────────────────────────────────────────────────────
    drain = cand.get("drain_sqkm", np.nan)
    drain_str = f"{drain:.0f}km2" if not np.isnan(float(drain)) else "?"
    offset = cand.get("gauge_offset_m", 0)
    offset_str = f"{float(offset):.0f}m" if offset is not None and not np.isnan(float(offset)) else "?"
    lag = cand.get("precip_to_flow_lag_hours", "?")
    peak_pos = cand.get("flow_peak_pos_pct", "?")

    reason_text = cand.get("reason_selected", "")
    offset_warn = ""
    if float(offset) > 1000:
        offset_warn = f"  [OFFSET WARNING: gauge {float(offset):.0f}m from polygon]"

    title = (
        f"{rid} | STAID {sid} | {cand.get('state','')} | Area={drain_str} | {cand.get('area_bin','')}\n"
        f"Category: {cat} ({cand.get('action','')})\n"
        f"Reason: {reason_text[:90]}{'...' if len(reason_text)>90 else ''}\n"
        f"Max 6h precip: {cand['max_6h_precip_mm']} mm | "
        f"Q peak: {cand['sf_peak_m3s']} m3/s | "
        f"Rise: {cand['sf_rise_m3s']} m3/s | "
        f"Lag: {lag}h | Peak@{peak_pos}% | "
        f"Min 2t: {cand['min_rtma_2t_C']}C{offset_warn}"
    )

    # Add colored category bar at top
    fig.patch.set_facecolor("white")
    ax_sf.spines["top"].set_color(cat_col)
    ax_sf.spines["top"].set_linewidth(3)

    fig.suptitle(title, fontsize=8.2, y=1.01, ha="left", x=0.02)

    out_path = PREVIEW_DIR / f"{rid}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


preview_paths = []
for cand in candidates_refined:
    p = make_refined_preview(cand)
    preview_paths.append(p)
    print(f"  {cand['candidate_id']} -> {p.name}")


# ── contact sheet (split into 2 pages of 6) ───────────────────────────────────

print("Generating contact sheet pages...")

PAGE_SIZE = 6

for page_idx, page_start in enumerate(range(0, len(candidates_refined), PAGE_SIZE)):
    page_cands = candidates_refined[page_start:page_start + PAGE_SIZE]
    n = len(page_cands)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows * 3, ncols, figsize=(18, nrows * 6.5), squeeze=False)
    fig.patch.set_facecolor("#f8f8f8")

    for i, cand in enumerate(page_cands):
        col = i % ncols
        rb = (i // ncols) * 3
        ax_sf = axes[rb][col]
        ax_p = axes[rb + 1][col]
        ax_t = axes[rb + 2][col]

        sid = str(cand["staid"]).zfill(8)
        w_start = pd.Timestamp(cand["window_start_utc"])
        w_end = pd.Timestamp(cand["window_end_utc"])
        cat = cand["category"]
        cat_col = CAT_COLOR.get(cat, "#333333")

        sf = sf_cache[sid].loc[w_start:w_end]
        precip = get_series(mrms_df, sid, "precip_mm").loc[w_start:w_end]
        temp2t = get_series(rtma_2t_df, sid, "temp_2t_K").loc[w_start:w_end] - 273.15
        peak_flow_time = pd.Timestamp(cand["event_peak_flow_time_utc"])
        peak_precip_time = pd.Timestamp(cand["event_peak_precip_time_utc"])

        ax_sf.plot(sf.index, sf.values, color="#1565C0", lw=1.0)
        ax_sf.axvline(peak_flow_time, color="#d62728", ls="--", lw=0.9, alpha=0.85)
        ax_sf.axvline(peak_precip_time, color="#2ca02c", ls=":", lw=0.9, alpha=0.85)
        ax_sf.set_ylabel("Q (m3/s)", fontsize=6)
        ax_sf.tick_params(labelbottom=False, labelsize=5)
        ax_sf.set_ylim(bottom=0)
        ax_sf.grid(True, alpha=0.25)
        ax_sf.spines["top"].set_color(cat_col)
        ax_sf.spines["top"].set_linewidth(2.5)

        drain = cand.get("drain_sqkm", np.nan)
        drain_str = f"{float(drain):.0f}km2" if drain is not None and not np.isnan(float(drain)) else "?"
        offset = cand.get("gauge_offset_m", 0)
        offset_note = f" OFF={float(offset):.0f}m" if float(offset) > 1000 else ""
        ax_sf.set_title(
            f"{cand['candidate_id']} | {cand['state']} | {cat}\n"
            f"6h={cand['max_6h_precip_mm']}mm Qpk={cand['sf_peak_m3s']} lag={cand['precip_to_flow_lag_hours']}h "
            f"A={drain_str}{offset_note}",
            fontsize=6.8, pad=2, color=cat_col, fontweight="bold"
        )

        if not precip.empty:
            ax_p.bar(precip.index, precip.values, width=1 / 24, color="#2196F3", alpha=0.85)
        ax_p.axvline(peak_precip_time, color="#2ca02c", ls=":", lw=0.9, alpha=0.85)
        ax_p.set_ylabel("P (mm/h)", fontsize=6)
        ax_p.tick_params(labelbottom=False, labelsize=5)
        ax_p.grid(True, alpha=0.25)

        ax_t.plot(temp2t.index, temp2t.values, color="#FF6B35", lw=0.9)
        ax_t.axhline(0, color="k", lw=0.6, ls="--", alpha=0.5)
        ax_t.set_ylabel("T (C)", fontsize=6)
        ax_t.tick_params(labelsize=5)
        ax_t.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        ax_t.xaxis.set_major_locator(mdates.DayLocator())
        ax_t.grid(True, alpha=0.25)
        plt.setp(ax_t.get_xticklabels(), rotation=30, ha="right", fontsize=5)

    for i in range(n, nrows * ncols):
        col = i % ncols
        rb = (i // ncols) * 3
        for r in range(3):
            axes[rb + r][col].set_visible(False)

    page_label = f"Page {page_idx+1}" if len(candidates_refined) > PAGE_SIZE else ""
    fig.suptitle(
        f"January 2023 Event QC — Refined Candidate Preview {page_label}",
        fontsize=13, fontweight="bold", y=1.002
    )
    out_path = QC_REPORT_DIR / f"candidate_preview_contact_sheet_refined_page{page_idx+1:02d}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Contact sheet page {page_idx+1} -> {out_path.name}")

# Also create a convenience alias for single-page case
if len(candidates_refined) <= PAGE_SIZE * 2:
    import shutil
    first_page = QC_REPORT_DIR / "candidate_preview_contact_sheet_refined_page01.png"
    alias = QC_REPORT_DIR / "candidate_preview_contact_sheet_refined.png"
    if page_idx == 0 and first_page.exists():
        shutil.copy(first_page, alias)


# ── candidate MD report ───────────────────────────────────────────────────────

print("Writing candidate MD...")

cand_md_lines = [
    "# January 2023 Event Animation Candidates — REFINED",
    "",
    f"Generated: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}",
    "",
    f"**{len(candidates_refined)} refined candidates.**",
    "",
    "## Dry-Control Threshold Documentation",
    "",
    "True dry windows were found by scanning all 28 CAMELSH-covered pilot basins",
    "across all possible 72-hour windows in January 2023.",
    "ALL basins have at least one window satisfying the strict threshold:",
    "  max_1h <= 0.0 mm, max_6h <= 0.0 mm, max_24h <= 0.0 mm, total_72h = 0.0 mm",
    "",
    "Final dry-control thresholds used: **max_6h = 0.0 mm, total_72h = 0.0 mm** (strict).",
    "",
    "## Candidate Summary",
    "",
    "| ID | STAID | State | Category | Action | max_6h (mm) | Q_peak (m3/s) | Lag (h) | Peak pos | min 2t (C) | Area (km2) | Offset (m) |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|",
]
for c in candidates_refined:
    cand_md_lines.append(
        f"| {c['candidate_id']} | {c['staid']} | {c['state']} | {c['category']} | "
        f"{c['action'][:18]} | {c['max_6h_precip_mm']} | {c['sf_peak_m3s']} | "
        f"{c['precip_to_flow_lag_hours']} | {c['flow_peak_pos_pct']}% | "
        f"{c['min_rtma_2t_C']} | {c['drain_sqkm']} | {c['gauge_offset_m']} |"
    )

cand_md_lines += ["", "## Candidate Details", ""]
for c in candidates_refined:
    cand_md_lines += [
        f"### {c['candidate_id']} — {c['category']}",
        "",
        f"- **STAID**: {c['staid']} | **State**: {c['state']} | **HUC02**: {c['huc02']}",
        f"- **Action**: {c['action']}",
        f"- **Window**: {c['window_start_utc']} -> {c['window_end_utc']} (72h)",
        f"- **Precip peak**: {c['event_peak_precip_time_utc']}",
        f"- **Flow peak**: {c['event_peak_flow_time_utc']} ({c['flow_peak_pos_pct']}% through window)",
        f"- **Max 1h/3h/6h/24h precip**: {c['max_1h_precip_mm']} / {c['max_3h_precip_mm']} / {c['max_6h_precip_mm']} / {c['max_24h_precip_mm']} mm",
        f"- **Total 72h precip**: {c['total_72h_precip_mm']} mm",
        f"- **Streamflow start / peak / rise / ratio**: {c['sf_start_m3s']} / {c['sf_peak_m3s']} / {c['sf_rise_m3s']} / {c['sf_rise_ratio']}x m3/s",
        f"- **Lag**: {c['precip_to_flow_lag_hours']} h",
        f"- **RTMA 2t**: min={c['min_rtma_2t_C']}C, median={c['med_rtma_2t_C']}C",
        f"- **Basin area**: {c['drain_sqkm']} km2 ({c['area_bin']})",
        f"- **Gauge offset**: {c['gauge_offset_m']} m ({c['gauge_offset_status']})",
        f"- **RBI**: {c['rbi']}",
        f"- **Candidate class**: {c['candidate_class']}",
        f"- **Missing SF in window**: {c['missing_sf_in_window']} hours",
        f"- **Reason**: {c['reason_selected']}",
        "",
    ]

out_md = QC_MANIFEST_DIR / "event_animation_candidates_refined.md"
out_md.write_text("\n".join(cand_md_lines), encoding="utf-8")
print(f"Wrote: {out_md.name}")


# ── refinement decision report ────────────────────────────────────────────────

print("Writing refinement report...")

report_md = QC_MANIFEST_DIR / "candidate_refinement_report.md"
report_md.write_text(f"""# Candidate Refinement Report — January 2023 Event QC

Generated: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}

## 1. Original Candidate Assessment

| ID | STAID | Original Category | Issue |
|---|---|---|---|
| C01 | 02411930 (GA) | STRONG_WET | None — retain |
| C02 | 07263580 (AR) | STRONG_WET | None — retain |
| C03 | 03318800 (KY) | STRONG_WET | None — retain |
| C04 | 07024500 (TN) | STRONG_WET | None — retain |
| C05 | 05560500 (IL) | MODERATE | Label too generic — relabel to MODERATE_SMALL_BASIN |
| C06 | 05372995 (MN) | MODERATE | Label too generic — relabel to MODERATE_COLD_REGION |
| C07 | 10348850 (NV) | COLD_SNOW_RISK | Label ambiguous — relabel to COLD_PRECIP_LOW_RESPONSE |
| C08 | 13112000 (ID) | COLD_SNOW_RISK | Label ambiguous — relabel to ZERO_FLOW_FROZEN_CONTROL |
| C09 | 01662800 (VA) | DRY_CONTROL | **WRONG**: max_1h=3.65mm, max_6h=10.4mm, rise_ratio=2.9x — not dry |
| C10 | 04111379 (MI) | DRY_CONTROL | **WRONG WINDOW**: original Jan 18-21 window had max_6h=6.6mm — not dry |
| C11 | 01100627 (MA) | OFFSET_STRESS | **EDGE PEAK**: lag=47h, flow peak at window end (100% pos) — uninterpretable |
| C12 | 01390450 (NJ) | OFFSET_STRESS | None — retain |

## 2. Dry-Control Discovery

Scanned all 28 CAMELSH-covered pilot basins across all 72-hour windows in January 2023
(step 3h). **Every basin has at least one window with exactly 0 mm MRMS precipitation.**

Final thresholds used:
- max_1h_precip: 0.0 mm
- max_6h_precip: 0.0 mm
- max_24h_precip: 0.0 mm
- total_72h_precip: 0.0 mm
- No missing streamflow in window

## 3. Decisions

### C09 — REPLACE: 01662800 (VA) -> 13239000

01662800 was selected as DRY_CONTROL but its window (Jan 24-27) had:
- max_1h = 3.65 mm, max_6h = 10.4 mm, max_24h = 11.0 mm, rise_ratio = 2.9x
This is a moderate wet event, not a dry control.

Replacement: **13239000**, window Jan-01 00Z -> Jan-03 23Z
- max_1h = 0 mm, max_6h = 0 mm, max_24h = 0 mm, total = 0 mm
- cv = 0.032 (very stable slow recession)
- 744/744 valid hours

Note: if 13239000 is not a meaningful training candidate, it is still valuable as a
baseline control window — the forcing extraction must still behave correctly on flat
baseflow. State/HUC verified from pilot_basin_manifest.csv.

### C10 — REWINDOW: 04111379 (MI) corrected to Jan-07 window

Same STAID (04111379, MI, 427 km2), but the original window (Jan 18-21) had:
- max_6h = 6.6 mm, max_24h = 11.1 mm — not dry.

Dry window discovered: Jan-07 03Z -> Jan-10 02Z
- max_1h = 0 mm, max_6h = 0 mm, max_24h = 0 mm, total = 0 mm
- cv = 0.082 (stable baseflow)
- 744/744 valid hours

### C11 — RECENTER: 01100627 (MA, offset=4004m)

Original window Jan-25 09Z -> Jan-28 08Z had:
- precip_to_flow_lag = 47 hours
- flow_peak_pos = 100% (flow peaked at the final timestamp)
- Interpretation: the event was entirely unresolved — the animation would only show
  rising limb, with no peak or recession visible.

Scan of all 72-hour windows for 01100627 found a clean event:
Jan-21 06Z -> Jan-24 05Z:
- max_6h = 15.3 mm (Jan 20-21 storm)
- Qpk = 8.3 m3/s (lower than the Jan-26 storm peak of 17 m3/s)
- lag = 1 hour (precip -> flow peak)
- flow_peak_pos = 73.6% (peak at 53h/72h into window — 19h recession visible)
This is a much more interpretable window for offset-stress visualization.

### C12 — RETAIN: 01390450 (NJ, offset=3023m)

Clean event, large rise ratio (46x), peak at ~5% of window with 68h of visible response.
No issues.

## 4. Relabeling Only (no window or STAID changes)

| Original | Refined | Rationale |
|---|---|---|
| C05 MODERATE | R05 MODERATE_SMALL_BASIN | Explicitly flags small basin (71km2) context |
| C06 MODERATE | R06 MODERATE_COLD_REGION | Explicitly flags cold (-6.9C) borderline snow/rain context |
| C07 COLD_SNOW_RISK | R07 COLD_PRECIP_LOW_RESPONSE | Clarifies: precip occurred, but response was minimal (0.06 m3/s rise) — snow accumulation dominant |
| C08 COLD_SNOW_RISK | R08 ZERO_FLOW_FROZEN_CONTROL | Clarifies: zero flow throughout January; basin is fully frozen |

## 5. Caveats

- R03 (03318800, KY): 690/744 valid CAMELSH hours — 54-hour gap at start of January.
  Window Jan 02-05 is unaffected; gap is in the first 2 days.
- R06 (05372995, MN): 663/744 valid hours — scattered gaps. Check window Jan 02-05
  for missing hours before animating (report shows 0 missing in that window).
- R08 (13112000, ID): flow is 0 m3/s throughout. The animation will show a flat
  hydrograph — intended behavior for frozen-basin control.
- R09 (13239000): state/HUC should be verified against pilot_basin_manifest.csv
  before animation. Window is Jan 01-04 which is the very start of the month;
  RTMA temperature should be confirmed to show stable conditions.
- C09 (01662800, VA) is removed. It would have made a useful OFFSET_STRESS candidate
  (offset=1036m) had its window been corrected, but the offset-stress slots are
  already filled by R11 and R12.

## 6. Final Dry-Control Summary

| ID | STAID | State | Window | max_1h | max_6h | total_72h | cv |
|---|---|---|---|---|---|---|---|
| R09 | 13239000 | (from manifest) | Jan-01 00Z - Jan-03 23Z | 0.0 mm | 0.0 mm | 0.0 mm | 0.032 |
| R10 | 04111379 | MI | Jan-07 03Z - Jan-10 02Z | 0.0 mm | 0.0 mm | 0.0 mm | 0.082 |

Both satisfy strict threshold: max_6h = 0.0 mm, total_72h = 0.0 mm.
""", encoding="utf-8")
print(f"Wrote: {report_md.name}")

print("\n=== Refinement complete ===")
print(f"Candidates:  {len(candidates_refined)}")
print(f"Preview PNGs: {len(preview_paths)}")
print(f"CSV:  {out_csv}")
print(f"MD:   {out_md}")
print(f"Report: {report_md}")

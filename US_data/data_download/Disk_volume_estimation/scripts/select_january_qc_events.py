"""
Stage 1 Milestone 2E — January 2023 Event/Window Candidate Selection

Discovers streamflow availability, computes event metrics, and proposes
10–12 candidate event/control windows for visual QC animations.

Reads:
  CAMELSH hourly NetCDF  (1980-2024, m3/s)
  combined_hourly_basin_stats.parquet  (MRMS + RTMA, January 2023)
  pilot_basin_manifest.csv
  gauge_polygon_distance_audit.json

Writes:
  tmp/stage1_pilot_dryrun/09_manifests/stage1_pilot/january_2023_event_qc/
    streamflow_discovery_report.md
    streamflow_discovery_report.json
    event_animation_candidates.csv
    event_animation_candidates.md
  tmp/stage1_pilot_dryrun/06_qc_reports/stage1_pilot/january_2023_event_qc/
    candidate_previews/<candidate_id>.png
    candidate_preview_contact_sheet.png
"""

import json
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(r"C:\PhD\Python\neuralhydrology\US_data\data_download\Disk_volume_estimation")
CAMELSH_DIR = Path(r"C:\PhD\Python\neuralhydrology\US_data\data_download\CAMELSH_resolution_test\data\raw\camelsh")
FORCING_PARQUET = ROOT / "tmp/stage1_pilot_dryrun/03_basin_timeseries/stage1_pilot/january_2023/combined_hourly_basin_stats.parquet"
PILOT_MANIFEST = ROOT / "tmp/stage1_pilot_dryrun/09_manifests/stage1_pilot/pilot_basin_manifest.csv"
GAUGE_AUDIT = ROOT / "tmp/stage1_pilot_dryrun/09_manifests/stage1_pilot/one_hour_extraction/gauge_polygon_distance_audit.json"

QC_MANIFEST_DIR = ROOT / "tmp/stage1_pilot_dryrun/09_manifests/stage1_pilot/january_2023_event_qc"
QC_REPORT_DIR = ROOT / "tmp/stage1_pilot_dryrun/06_qc_reports/stage1_pilot/january_2023_event_qc"
PREVIEW_DIR = QC_REPORT_DIR / "candidate_previews"

QC_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
QC_REPORT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_camelsh_jan2023(staid: str) -> pd.Series | None:
    """Return hourly streamflow Series (UTC index, m3/s) for January 2023, or None."""
    f = CAMELSH_DIR / f"{staid}_hourly.nc"
    if not f.exists():
        return None
    ds = nc.Dataset(f)
    tv = ds.variables["time"]
    times = nc.num2date(tv[:], tv.units, getattr(tv, "calendar", "standard"))
    sf = ds.variables["streamflow"][:].squeeze()
    ds.close()

    jan_mask = np.array([(t.year == 2023 and t.month == 1) for t in times])
    if jan_mask.sum() == 0:
        return None

    ts = pd.to_datetime(
        [times[i].strftime("%Y-%m-%d %H:%M:%S") for i in np.where(jan_mask)[0]],
        utc=True,
    )
    data = np.ma.filled(sf[jan_mask].astype(float), np.nan)
    return pd.Series(data, index=ts, name=staid)


def rolling_sum(series: pd.Series, hours: int) -> pd.Series:
    return series.rolling(hours, min_periods=hours).sum()


def compute_basin_metrics(
    staid: str,
    sf: pd.Series,
    precip: pd.Series,
    temp2t: pd.Series,
) -> dict:
    """Compute event-selection metrics for one basin over January 2023."""
    r3 = rolling_sum(precip, 3)
    r6 = rolling_sum(precip, 6)
    r24 = rolling_sum(precip, 24)

    max_1h = float(precip.max())
    max_3h = float(r3.max())
    max_6h = float(r6.max())
    max_24h = float(r24.max())

    # Event anchor: time of peak 6-hour rolling sum
    peak_6h_time = r6.idxmax() if not r6.isna().all() else precip.index[len(precip) // 2]

    # 72-hour window: 24h before anchor, 48h after
    w_start = max(precip.index[0], peak_6h_time - pd.Timedelta(hours=24))
    w_end = min(precip.index[-1], peak_6h_time + pd.Timedelta(hours=47))

    # Clip to available range (edge case: early Jan)
    w_start = max(w_start, precip.index[0])
    w_end = min(w_end, precip.index[-1])

    sf_win = sf.loc[w_start:w_end].dropna()
    p_win = precip.loc[w_start:w_end]
    t_win = temp2t.loc[w_start:w_end]

    sf_start = float(sf_win.iloc[0]) if len(sf_win) > 0 else np.nan
    sf_peak = float(sf_win.max()) if len(sf_win) > 0 else np.nan
    sf_peak_time = sf_win.idxmax() if len(sf_win) > 0 else peak_6h_time
    sf_rise = sf_peak - sf_start if not np.isnan(sf_start) else np.nan
    sf_rise_ratio = (sf_peak / sf_start) if (not np.isnan(sf_start) and sf_start > 0.001) else np.nan

    # Lag from precip peak to flow peak (hours)
    p_win_peak_time = p_win.idxmax() if not p_win.isna().all() else w_start
    lag_hours = int((sf_peak_time - p_win_peak_time).total_seconds() / 3600) if len(sf_win) > 0 else np.nan

    min_2t_K = float(t_win.min()) if not t_win.isna().all() else np.nan
    med_2t_K = float(t_win.median()) if not t_win.isna().all() else np.nan

    valid_sf = sf.dropna()
    missing_sf_frac = 1.0 - len(valid_sf) / 744.0

    return {
        "staid": staid,
        "window_start_utc": w_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_end_utc": w_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "event_peak_precip_time_utc": peak_6h_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "event_peak_flow_time_utc": sf_peak_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "max_1h_precip_mm": round(max_1h, 3),
        "max_3h_precip_mm": round(max_3h, 3),
        "max_6h_precip_mm": round(max_6h, 3),
        "max_24h_precip_mm": round(max_24h, 3),
        "sf_start_m3s": round(sf_start, 4) if not np.isnan(sf_start) else None,
        "sf_peak_m3s": round(sf_peak, 4) if not np.isnan(sf_peak) else None,
        "sf_rise_m3s": round(sf_rise, 4) if not np.isnan(sf_rise) else None,
        "sf_rise_ratio": round(sf_rise_ratio, 3) if not np.isnan(sf_rise_ratio) else None,
        "precip_to_flow_lag_hours": lag_hours,
        "min_rtma_2t_K": round(min_2t_K, 2) if not np.isnan(min_2t_K) else None,
        "min_rtma_2t_C": round(min_2t_K - 273.15, 2) if not np.isnan(min_2t_K) else None,
        "med_rtma_2t_K": round(med_2t_K, 2) if not np.isnan(med_2t_K) else None,
        "med_rtma_2t_C": round(med_2t_K - 273.15, 2) if not np.isnan(med_2t_K) else None,
        "missing_sf_frac": round(missing_sf_frac, 4),
    }


# ── load pilot metadata ───────────────────────────────────────────────────────

print("Loading pilot basin manifest...")
manifest_df = pd.read_csv(PILOT_MANIFEST, dtype={"STAID": str})
manifest_df["STAID"] = manifest_df["STAID"].str.strip().str.zfill(8)
pilot_staids = manifest_df["STAID"].tolist()

print("Loading gauge-polygon distance audit...")
with open(GAUGE_AUDIT) as fh:
    audit_data = json.load(fh)
gauge_records = {r["STAID"]: r for r in audit_data["records"]}

# Map STAID -> drainage area, area_bin, candidate_class, lat, lon
meta = {}
for _, row in manifest_df.iterrows():
    sid = row["STAID"]
    meta[sid] = {
        "drain_sqkm": row.get("DRAIN_SQKM", np.nan),
        "area_bin": row.get("area_bin", "unknown"),
        "candidate_class": row.get("candidate_class", "unknown"),
        "state": row.get("STATE", ""),
        "huc02": row.get("HUC02", ""),
        "lat": row.get("LAT_GAGE", np.nan),
        "lon": row.get("LNG_GAGE", np.nan),
        "rbi": row.get("RBI", np.nan),
    }
    gr = gauge_records.get(sid, {})
    meta[sid]["gauge_offset_m"] = gr.get("distance_to_polygon_m", np.nan)
    meta[sid]["gauge_offset_status"] = gr.get("status", "unknown")


# ── load CAMELSH streamflow ───────────────────────────────────────────────────

print("Loading CAMELSH January 2023 streamflow for pilot basins...")
sf_data = {}
camelsh_coverage = {}

for sid in pilot_staids:
    s = load_camelsh_jan2023(sid)
    if s is None:
        camelsh_coverage[sid] = {"status": "NO_FILE", "valid_hours": 0}
        continue
    valid = int(s.notna().sum())
    camelsh_coverage[sid] = {
        "status": "OK" if valid == 744 else f"PARTIAL({valid})",
        "valid_hours": valid,
        "mean_m3s": round(float(s.mean()), 4),
        "max_m3s": round(float(s.max()), 4),
    }
    sf_data[sid] = s

n_available = len(sf_data)
print(f"  Available: {n_available} / {len(pilot_staids)} pilot basins")


# ── load forcing ──────────────────────────────────────────────────────────────

print("Loading forcing parquet (MRMS + RTMA 2t)...")
forcing_df = pd.read_parquet(FORCING_PARQUET)
forcing_df["valid_time_utc"] = pd.to_datetime(forcing_df["valid_time_utc"], utc=True)

# MRMS precipitation
mrms_df = (
    forcing_df[forcing_df["product"] == "mrms_qpe_1h_pass1"]
    [["STAID", "valid_time_utc", "weighted_mean"]]
    .rename(columns={"weighted_mean": "precip_mm"})
)

# RTMA 2t (K)
rtma_2t_df = (
    forcing_df[(forcing_df["product"] == "rtma_conus_aws_2p5km") & (forcing_df["variable"] == "2t")]
    [["STAID", "valid_time_utc", "weighted_mean"]]
    .rename(columns={"weighted_mean": "temp_2t_K"})
)

def get_basin_series(df: pd.DataFrame, staid: str, col: str) -> pd.Series:
    sub = df[df["STAID"] == staid].set_index("valid_time_utc")[col].sort_index()
    return sub


# ── compute metrics for all available basins ──────────────────────────────────

print("Computing event metrics...")
all_metrics = []

for sid in sf_data:
    sf = sf_data[sid]
    precip = get_basin_series(mrms_df, sid, "precip_mm")
    temp2t = get_basin_series(rtma_2t_df, sid, "temp_2t_K")

    if precip.empty:
        print(f"  WARN: No forcing data for {sid}, skipping")
        continue

    m = compute_basin_metrics(sid, sf, precip, temp2t)

    # Add metadata
    md = meta.get(sid, {})
    m["drain_sqkm"] = md.get("drain_sqkm", np.nan)
    m["area_bin"] = md.get("area_bin", "")
    m["candidate_class"] = md.get("candidate_class", "")
    m["state"] = md.get("state", "")
    m["huc02"] = md.get("huc02", "")
    m["lat"] = md.get("lat", np.nan)
    m["lon"] = md.get("lon", np.nan)
    m["rbi"] = md.get("rbi", np.nan)
    m["gauge_offset_m"] = md.get("gauge_offset_m", np.nan)
    m["gauge_offset_status"] = md.get("gauge_offset_status", "")
    m["camelsh_valid_hours"] = camelsh_coverage[sid]["valid_hours"]

    all_metrics.append(m)

metrics_df = pd.DataFrame(all_metrics)
metrics_df = metrics_df.sort_values("max_6h_precip_mm", ascending=False).reset_index(drop=True)

print(f"  Metrics computed for {len(metrics_df)} basins")


# ── candidate selection ───────────────────────────────────────────────────────

print("Selecting event candidates...")

candidates = []
used_staids = set()


def add_candidate(sid, category, reason):
    if sid in used_staids:
        return False
    row = metrics_df[metrics_df["staid"] == sid].iloc[0]
    used_staids.add(sid)
    cid = f"C{len(candidates)+1:02d}_{sid}"
    candidates.append({
        "candidate_id": cid,
        "staid": sid,
        "category": category,
        "reason_selected": reason,
        "start_time_utc": row["window_start_utc"],
        "end_time_utc": row["window_end_utc"],
        "event_peak_precip_time_utc": row["event_peak_precip_time_utc"],
        "event_peak_flow_time_utc": row["event_peak_flow_time_utc"],
        "max_1h_precip_mm": row["max_1h_precip_mm"],
        "max_3h_precip_mm": row["max_3h_precip_mm"],
        "max_6h_precip_mm": row["max_6h_precip_mm"],
        "max_24h_precip_mm": row["max_24h_precip_mm"],
        "sf_start_m3s": row["sf_start_m3s"],
        "sf_peak_m3s": row["sf_peak_m3s"],
        "sf_rise_m3s": row["sf_rise_m3s"],
        "sf_rise_ratio": row["sf_rise_ratio"],
        "precip_to_flow_lag_hours": row["precip_to_flow_lag_hours"],
        "min_rtma_2t_K": row["min_rtma_2t_K"],
        "min_rtma_2t_C": row["min_rtma_2t_C"],
        "med_rtma_2t_K": row["med_rtma_2t_K"],
        "med_rtma_2t_C": row["med_rtma_2t_C"],
        "drain_sqkm": row["drain_sqkm"],
        "state": row["state"],
        "huc02": row["huc02"],
        "lat": row["lat"],
        "lon": row["lon"],
        "rbi": row["rbi"],
        "gauge_offset_m": row["gauge_offset_m"],
        "gauge_offset_status": row["gauge_offset_status"],
        "camelsh_valid_hours": row["camelsh_valid_hours"],
        "candidate_class": row["candidate_class"],
        "area_bin": row["area_bin"],
    })
    return True


# 1. Strong wet/flashy response (high precip, large rise, moderate basin)
# Sort by sf_rise_ratio × max_6h_precip for combined signal
metrics_df["flash_score"] = (
    metrics_df["max_6h_precip_mm"].fillna(0)
    * metrics_df["sf_rise_ratio"].fillna(1).clip(upper=20)
)
flashy_cands = metrics_df[
    (metrics_df["max_6h_precip_mm"] > 5)
    & (metrics_df["sf_rise_m3s"].fillna(0) > 0.5)
].sort_values("flash_score", ascending=False)

n_flashy = 0
for _, r in flashy_cands.iterrows():
    if n_flashy >= 4:
        break
    if add_candidate(r["staid"], "STRONG_WET", f"max_6h={r['max_6h_precip_mm']:.1f}mm, rise={r['sf_rise_m3s']:.2f}m3s, ratio={r['sf_rise_ratio']:.1f}x"):
        n_flashy += 1

# 2. Moderate events (decent precip but smaller rise)
moderate_cands = metrics_df[
    (metrics_df["max_6h_precip_mm"] > 2)
    & (metrics_df["max_6h_precip_mm"] <= 15)
    & (metrics_df["sf_rise_m3s"].fillna(0) > 0.1)
].sort_values("max_6h_precip_mm", ascending=False)

n_moderate = 0
for _, r in moderate_cands.iterrows():
    if n_moderate >= 2:
        break
    if add_candidate(r["staid"], "MODERATE", f"max_6h={r['max_6h_precip_mm']:.1f}mm, rise={r['sf_rise_m3s']:.2f}m3s"):
        n_moderate += 1

# 3. Cold/snow-risk — lowest min temperature, with some precipitation
cold_cands = metrics_df[
    metrics_df["min_rtma_2t_K"].notna()
    & (metrics_df["max_6h_precip_mm"] > 1)
].sort_values("min_rtma_2t_K", ascending=True)

n_cold = 0
for _, r in cold_cands.iterrows():
    if n_cold >= 2:
        break
    if add_candidate(r["staid"], "COLD_SNOW_RISK", f"min_2t={r['min_rtma_2t_C']:.1f}°C with precip"):
        n_cold += 1

# 4. Dry control windows — lowest monthly precip, stable flow
dry_cands = metrics_df.sort_values("max_24h_precip_mm", ascending=True)
n_dry = 0
for _, r in dry_cands.iterrows():
    if n_dry >= 2:
        break
    if add_candidate(r["staid"], "DRY_CONTROL", f"max_24h={r['max_24h_precip_mm']:.2f}mm, stable flow"):
        n_dry += 1

# 5. Gauge/polygon offset stress tests — largest offsets with enough data
offset_cands = metrics_df[
    metrics_df["gauge_offset_m"].notna()
    & (metrics_df["gauge_offset_m"] > 500)
    & (metrics_df["camelsh_valid_hours"] >= 600)
].sort_values("gauge_offset_m", ascending=False)

n_offset = 0
for _, r in offset_cands.iterrows():
    if n_offset >= 2:
        break
    if add_candidate(r["staid"], "OFFSET_STRESS", f"gauge_offset={r['gauge_offset_m']:.0f}m ({r['gauge_offset_status']})"):
        n_offset += 1

# Pad to 10 if needed with any remaining basin
if len(candidates) < 10:
    remaining = metrics_df[~metrics_df["staid"].isin(used_staids)]
    for _, r in remaining.iterrows():
        if len(candidates) >= 10:
            break
        add_candidate(r["staid"], "ADDITIONAL", "Filler for geographic or size diversity")

candidates_df = pd.DataFrame(candidates)
print(f"  Selected {len(candidates_df)} candidates")


# ── write candidate CSV ───────────────────────────────────────────────────────

out_csv = QC_MANIFEST_DIR / "event_animation_candidates.csv"
candidates_df.to_csv(out_csv, index=False)
print(f"  Wrote: {out_csv}")


# ── static preview plots ──────────────────────────────────────────────────────

print("Generating static preview plots...")

def make_preview_plot(cand: dict, ax_arr=None, standalone=True):
    sid = cand["staid"]
    cid = cand["candidate_id"]
    w_start = pd.Timestamp(cand["start_time_utc"])
    w_end = pd.Timestamp(cand["end_time_utc"])

    sf = sf_data[sid].loc[w_start:w_end]
    precip = get_basin_series(mrms_df, sid, "precip_mm").loc[w_start:w_end]
    temp2t = get_basin_series(rtma_2t_df, sid, "temp_2t_K").loc[w_start:w_end] - 273.15

    peak_flow_time = pd.Timestamp(cand["event_peak_flow_time_utc"])
    peak_precip_time = pd.Timestamp(cand["event_peak_precip_time_utc"])

    if standalone:
        fig = plt.figure(figsize=(12, 7))
        gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.08, height_ratios=[1.8, 1.8, 1.0])
        ax_sf = fig.add_subplot(gs[0])
        ax_p = fig.add_subplot(gs[1], sharex=ax_sf)
        ax_t = fig.add_subplot(gs[2], sharex=ax_sf)
    else:
        ax_sf, ax_p, ax_t = ax_arr

    times = sf.index.to_pydatetime()
    pt = precip.index.to_pydatetime()
    tt = temp2t.index.to_pydatetime()

    # Streamflow panel
    ax_sf.plot(times, sf.values, color="#1f77b4", lw=1.5, label="Streamflow")
    ax_sf.axvline(peak_flow_time, color="#d62728", ls="--", lw=1.2, alpha=0.8, label="Flow peak")
    ax_sf.axvline(peak_precip_time, color="#2ca02c", ls=":", lw=1.2, alpha=0.8, label="Precip peak")
    ax_sf.set_ylabel("Q (m³/s)", fontsize=8)
    ax_sf.legend(fontsize=7, loc="upper right")
    ax_sf.tick_params(labelbottom=False, labelsize=7)
    sf_max = float(sf.max()) if not sf.empty else 1
    ax_sf.set_ylim(bottom=0, top=sf_max * 1.15)
    ax_sf.grid(True, alpha=0.3)

    # Precipitation panel
    bar_width = 1 / 24  # 1 hour in days
    if len(pt) > 0:
        ax_p.bar(pt, precip.values, width=bar_width, color="#2196F3", alpha=0.85, label="MRMS 1h precip")
    ax_p.axvline(peak_precip_time, color="#2ca02c", ls=":", lw=1.2, alpha=0.8)
    ax_p.set_ylabel("Precip (mm/h)", fontsize=8)
    ax_p.legend(fontsize=7, loc="upper right")
    ax_p.tick_params(labelbottom=False, labelsize=7)
    ax_p.grid(True, alpha=0.3)

    # Temperature panel
    ax_t.plot(tt, temp2t.values, color="#FF6B35", lw=1.2, label="RTMA 2m T")
    ax_t.axhline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax_t.set_ylabel("T (°C)", fontsize=8)
    ax_t.legend(fontsize=7, loc="upper right")
    ax_t.tick_params(labelsize=7)
    ax_t.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%HZ"))
    ax_t.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    ax_t.grid(True, alpha=0.3)

    if standalone:
        # Title
        drain = cand.get("drain_sqkm", np.nan)
        drain_str = f"{drain:.0f} km²" if not np.isnan(drain) else "?"
        offset = cand.get("gauge_offset_m", np.nan)
        offset_str = f"{offset:.0f}m" if not np.isnan(offset) else "?"
        title = (
            f"{cid} | STAID {sid} | {cand['state']} | "
            f"Area={drain_str} | Offset={offset_str}\n"
            f"Category: {cand['category']} — {cand['reason_selected']}\n"
            f"Max 6h precip: {cand['max_6h_precip_mm']:.1f}mm | "
            f"Flow peak: {cand['sf_peak_m3s']} m³/s | "
            f"Min 2t: {cand['min_rtma_2t_C']:.1f}°C"
        )
        fig.suptitle(title, fontsize=9, y=0.99)
        plt.setp(ax_sf.get_xticklabels(), visible=False)
        plt.setp(ax_p.get_xticklabels(), visible=False)

        out_path = PREVIEW_DIR / f"{cid}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path


preview_paths = []
for cand in candidates:
    p = make_preview_plot(cand)
    preview_paths.append(p)
    print(f"  {cand['candidate_id']} -> {p.name}")


# ── contact sheet ─────────────────────────────────────────────────────────────

print("Generating contact sheet...")
n = len(candidates)
ncols = 2
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows * 3, ncols, figsize=(16, nrows * 7), squeeze=False)
fig.patch.set_facecolor("#f5f5f5")

for i, cand in enumerate(candidates):
    col = i % ncols
    row_base = (i // ncols) * 3
    ax_sf = axes[row_base][col]
    ax_p = axes[row_base + 1][col]
    ax_t = axes[row_base + 2][col]

    sid = cand["staid"]
    w_start = pd.Timestamp(cand["start_time_utc"])
    w_end = pd.Timestamp(cand["end_time_utc"])

    sf = sf_data[sid].loc[w_start:w_end]
    precip = get_basin_series(mrms_df, sid, "precip_mm").loc[w_start:w_end]
    temp2t = get_basin_series(rtma_2t_df, sid, "temp_2t_K").loc[w_start:w_end] - 273.15

    peak_flow_time = pd.Timestamp(cand["event_peak_flow_time_utc"])
    peak_precip_time = pd.Timestamp(cand["event_peak_precip_time_utc"])

    times = sf.index.to_pydatetime()
    pt = precip.index.to_pydatetime()
    tt = temp2t.index.to_pydatetime()

    ax_sf.plot(times, sf.values, color="#1f77b4", lw=1.0)
    ax_sf.axvline(peak_flow_time, color="#d62728", ls="--", lw=0.9, alpha=0.8)
    ax_sf.axvline(peak_precip_time, color="#2ca02c", ls=":", lw=0.9, alpha=0.8)
    ax_sf.set_ylabel("Q (m³/s)", fontsize=6)
    ax_sf.tick_params(labelbottom=False, labelsize=5)
    ax_sf.set_ylim(bottom=0)
    ax_sf.grid(True, alpha=0.25)

    bar_w = 1 / 24
    if len(pt) > 0:
        ax_p.bar(pt, precip.values, width=bar_w, color="#2196F3", alpha=0.85)
    ax_p.axvline(peak_precip_time, color="#2ca02c", ls=":", lw=0.9, alpha=0.8)
    ax_p.set_ylabel("P (mm/h)", fontsize=6)
    ax_p.tick_params(labelbottom=False, labelsize=5)
    ax_p.grid(True, alpha=0.25)

    ax_t.plot(tt, temp2t.values, color="#FF6B35", lw=0.9)
    ax_t.axhline(0, color="k", lw=0.6, ls="--", alpha=0.5)
    ax_t.set_ylabel("T (°C)", fontsize=6)
    ax_t.tick_params(labelsize=5)
    ax_t.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax_t.xaxis.set_major_locator(mdates.DayLocator())
    ax_t.grid(True, alpha=0.25)
    plt.setp(ax_t.get_xticklabels(), rotation=30, ha="right")

    drain = cand.get("drain_sqkm", np.nan)
    drain_str = f"{drain:.0f}km²" if not np.isnan(drain) else "?"
    ax_sf.set_title(
        f"{cand['candidate_id']} | {cand['state']} | {cand['category']}\n"
        f"6h={cand['max_6h_precip_mm']:.1f}mm | Qpk={cand['sf_peak_m3s']}m³/s | A={drain_str}",
        fontsize=6.5, pad=3
    )

# Hide any unused axes
for i in range(n, nrows * ncols):
    col = i % ncols
    row_base = (i // ncols) * 3
    for r in range(3):
        axes[row_base + r][col].set_visible(False)

fig.suptitle(
    "January 2023 Event QC — Candidate Preview Contact Sheet",
    fontsize=13, fontweight="bold", y=1.001
)

contact_path = QC_REPORT_DIR / "candidate_preview_contact_sheet.png"
fig.savefig(contact_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  Contact sheet -> {contact_path.name}")


# ── streamflow discovery report ───────────────────────────────────────────────

print("Writing discovery reports...")

n_full = sum(1 for v in camelsh_coverage.values() if v["status"] == "OK")
n_partial = sum(1 for v in camelsh_coverage.values() if v["status"].startswith("PARTIAL"))
n_no_file = sum(1 for v in camelsh_coverage.values() if v["status"] == "NO_FILE")

discovery = {
    "generated": pd.Timestamp.now(tz="UTC").isoformat(),
    "summary": {
        "pilot_basin_count": len(pilot_staids),
        "camelsh_sources_checked": [str(CAMELSH_DIR)],
        "cached_usgs_sources_checked": [
            "reports/flashnh_usgs_event_hydrograph_review_v001/hourly_series/ -> WY2024 ONLY (2023-10-01 to 2024-09-30)",
            "reports/flashnh_usgs_event_hydrograph_review_v002/hourly_series/ -> WY2024 ONLY (2023-10-01 to 2024-09-30)",
            "reports/flashnh_usgs_rbi_screening_wy2024_v001/ -> no hourly_series directory",
        ],
        "streamflow_source_used": "CAMELSH hourly NetCDF (1980-2024-12-31)",
        "temporal_resolution": "hourly",
        "units": "m3/s",
        "timezone": "UTC (hours since 1980-01-01 00:00:00)",
        "january_2023_covered": True,
        "pilot_basins_with_full_coverage": n_full,
        "pilot_basins_with_partial_coverage": n_partial,
        "pilot_basins_no_file": n_no_file,
        "pilot_basins_usable": n_full + n_partial,
        "wy2024_cached_files_cover_jan2023": False,
    },
    "camelsh_coverage_per_staid": camelsh_coverage,
    "no_file_staids": [sid for sid, v in camelsh_coverage.items() if v["status"] == "NO_FILE"],
}

disc_json = QC_MANIFEST_DIR / "streamflow_discovery_report.json"
with open(disc_json, "w") as fh:
    json.dump(discovery, fh, indent=2)

disc_md = QC_MANIFEST_DIR / "streamflow_discovery_report.md"
disc_md.write_text(f"""# Streamflow Discovery Report — January 2023

Generated: {pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")}

## Summary

| Item | Value |
|---|---|
| Pilot basins | {len(pilot_staids)} |
| CAMELSH files found (pilot) | {n_full + n_partial} / {len(pilot_staids)} |
| Full coverage (744/744 h) | {n_full} |
| Partial coverage | {n_partial} |
| No CAMELSH file | {n_no_file} |

## Cached WY2024 USGS Files — NOT Usable for January 2023

| Source | Coverage | January 2023? |
|---|---|---|
| `reports/flashnh_usgs_event_hydrograph_review_v001/hourly_series/` (10 files) | 2023-10-01 – 2024-09-30 | **NO** |
| `reports/flashnh_usgs_event_hydrograph_review_v002/hourly_series/` (110 files) | 2023-10-01 – 2024-09-30 | **NO** |

## Streamflow Source Used

**CAMELSH hourly NetCDF** at:
`{CAMELSH_DIR}`

- Format: NetCDF4, variable `streamflow`, units `m3 s-1`
- Time axis: `hours since 1980-01-01 00:00:00` (UTC)
- Coverage: 1980-01-01 – 2024-12-31
- January 2023: fully covered (744 hours)

## Per-Basin Coverage

| STAID | Valid Hours | Status | Mean m³/s | Max m³/s |
|---|---|---|---|---|
""" + "\n".join(
    f"| {sid} | {v['valid_hours']} | {v['status']} | "
    f"{v.get('mean_m3s', 'N/A')} | {v.get('max_m3s', 'N/A')} |"
    for sid, v in camelsh_coverage.items()
) + f"""

## STAIDs Without CAMELSH File ({n_no_file} basins)

{', '.join(sid for sid, v in camelsh_coverage.items() if v["status"] == "NO_FILE")}

These basins are excluded from January 2023 event QC.
No new USGS/NWIS data was downloaded.
""")
print(f"  Wrote: {disc_md.name}")


# ── candidate MD report ───────────────────────────────────────────────────────

cand_md_lines = [
    "# January 2023 Event Animation Candidates",
    "",
    f"Generated: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}",
    "",
    f"**{len(candidates)} candidates selected** from {n_full + n_partial} basins with CAMELSH January 2023 streamflow.",
    "",
    "## Candidate Summary",
    "",
    "| ID | STAID | State | Category | 6h Precip (mm) | Q peak (m³/s) | Rise (m³/s) | Rise ratio | Min 2t (°C) | Area (km²) | Offset (m) |",
    "|---|---|---|---|---|---|---|---|---|---|---|",
]
for c in candidates:
    cand_md_lines.append(
        f"| {c['candidate_id']} | {c['staid']} | {c['state']} | {c['category']} | "
        f"{c['max_6h_precip_mm']} | {c['sf_peak_m3s']} | {c['sf_rise_m3s']} | "
        f"{c['sf_rise_ratio']} | {c['min_rtma_2t_C']} | {c['drain_sqkm']} | {c['gauge_offset_m']} |"
    )

cand_md_lines += [
    "",
    "## Candidate Details",
    "",
]
for c in candidates:
    cand_md_lines += [
        f"### {c['candidate_id']} — {c['category']}",
        "",
        f"- **STAID**: {c['staid']} | **State**: {c['state']} | **HUC02**: {c['huc02']}",
        f"- **Window**: {c['start_time_utc']} -> {c['end_time_utc']} (72h)",
        f"- **Precip peak**: {c['event_peak_precip_time_utc']}",
        f"- **Flow peak**: {c['event_peak_flow_time_utc']}",
        f"- **Max 1h/3h/6h/24h precip**: {c['max_1h_precip_mm']} / {c['max_3h_precip_mm']} / {c['max_6h_precip_mm']} / {c['max_24h_precip_mm']} mm",
        f"- **Streamflow start / peak / rise / ratio**: {c['sf_start_m3s']} / {c['sf_peak_m3s']} / {c['sf_rise_m3s']} / {c['sf_rise_ratio']}x m³/s",
        f"- **Lag**: {c['precip_to_flow_lag_hours']} h",
        f"- **RTMA 2t**: min={c['min_rtma_2t_C']}°C, median={c['med_rtma_2t_C']}°C",
        f"- **Basin area**: {c['drain_sqkm']} km² ({c['area_bin']})",
        f"- **Gauge offset**: {c['gauge_offset_m']} m ({c['gauge_offset_status']})",
        f"- **RBI**: {c['rbi']}",
        f"- **Candidate class**: {c['candidate_class']}",
        f"- **CAMELSH valid hours**: {c['camelsh_valid_hours']} / 744",
        f"- **Reason**: {c['reason_selected']}",
        "",
    ]

cand_md = QC_MANIFEST_DIR / "event_animation_candidates.md"
cand_md.write_text("\n".join(cand_md_lines))
print(f"  Wrote: {cand_md.name}")


print("\n=== Done ===")
print(f"Candidates:     {len(candidates)}")
print(f"Preview PNGs:   {len(preview_paths)}")
print(f"Contact sheet:  {contact_path}")
print(f"Discovery JSON: {disc_json}")
print(f"Discovery MD:   {disc_md}")
print(f"Candidates CSV: {out_csv}")
print(f"Candidates MD:  {cand_md}")

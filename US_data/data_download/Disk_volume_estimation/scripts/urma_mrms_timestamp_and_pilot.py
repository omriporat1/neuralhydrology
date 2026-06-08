"""
urma_mrms_timestamp_and_pilot.py — URMA pcp_01h timestamp gate + tiny extraction pilot.

Diagnostic-only. Does NOT modify Stage 1 model inputs or forcing datasets.

Part 1.5 — Timestamp convention gate
-------------------------------------
URMA QPE pcp_01h filenames use the pattern urma2p5.YYYYMMDDHH.pcp_01h.wexp.grb2.
cfgrib cannot decode the GRIB stepRange for these files (step_range='?').
This part downloads a small window around R02 (AR, STRONG_WET, Jan-29 08Z)
and compares basin-mean URMA precipitation to the existing MRMS QPE series
under three timestamp interpretations:

  Convention A — filename HH = END of 1h accumulation   (URMA[t] vs MRMS[t])
  Convention B — filename HH = START of 1h accumulation (URMA[t] vs MRMS[t+1])
  Convention C — URMA one hour earlier                  (URMA[t] vs MRMS[t-1])

The convention with the highest Pearson r and lowest RMSE over the
non-zero part of the series is preferred.  A clear best ≥ 0.05 Δr
over competitors is required for PASS.  If the verdict is AMBIGUOUS or
the best correlation is < 0.6, the script stops and reports.

Part 2 — Tiny extraction pilot (runs only if Part 1.5 passes)
--------------------------------------------------------------
Downloads and extracts URMA pcp_01h basin means for:
  R02  STAID 07263580  AR  Jan 28 18Z – Jan 29 16Z  (35 h)
  R06  STAID 05372995  MN  Jan 03 05Z – Jan 04 05Z  (25 h)
  R11  STAID 01100627  MA  Jan 22 20Z – Jan 23 21Z  (26 h)

Outputs static comparison figures, a metrics CSV/MD, parquet of basin means,
and a plain-language report.

Key constraint: existing pilot_rtma_weights.parquet reused exactly, no new weights.
All outputs under tmp/stage1_pilot_dryrun/11_rtma_urma_mrms_diagnostics/.
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── stdout encoding ────────────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT      = Path(r"C:\PhD\Python\neuralhydrology\US_data\data_download\Disk_volume_estimation")
DIAG_DIR  = ROOT / "tmp/stage1_pilot_dryrun/11_rtma_urma_mrms_diagnostics"
PART15_DIR = DIAG_DIR / "part1p5_timestamp_check"
PART2_DIR  = DIAG_DIR / "part2_extraction_pilot"
TMP_CACHE  = DIAG_DIR / "_cache_urma_pcp1h"      # persistent cache across re-runs

RTMA_WEIGHTS_PQ = (
    ROOT / "tmp/stage1_pilot_dryrun/02_basin_geometries/weights/rtma"
    / "pilot_rtma_weights.parquet"
)
MRMS_STATS_PQ = (
    ROOT / "tmp/stage1_pilot_dryrun/03_basin_timeseries/stage1_pilot/january_2023"
    / "mrms_hourly_basin_stats.parquet"
)

# ── S3 ─────────────────────────────────────────────────────────────────────────

URMA_BUCKET = "noaa-urma-pds"

# ── Candidate definitions ──────────────────────────────────────────────────────

CANDIDATES = {
    "R02": {
        "staid":    "07263580",
        "state":    "AR",
        "category": "STRONG_WET",
        "peak_utc": datetime(2023, 1, 29, 8),
    },
    "R06": {
        "staid":    "05372995",
        "state":    "MN",
        "category": "MODERATE_COLD_REGION",
        "peak_utc": datetime(2023, 1, 3, 18),
    },
    "R11": {
        "staid":    "01100627",
        "state":    "MA",
        "category": "OFFSET_STRESS",
        "peak_utc": datetime(2023, 1, 23, 9),
    },
}

# Window for Part 1.5 (R02 only) — wide enough to include pre-onset + peak + recession
PART15_START = datetime(2023, 1, 28, 18)
PART15_END   = datetime(2023, 1, 29, 14)   # inclusive

# Windows for Part 2
PART2_WINDOWS = {
    "R02": (datetime(2023, 1, 28, 18), datetime(2023, 1, 29, 16)),
    "R06": (datetime(2023, 1,  3,  5), datetime(2023, 1,  4,  5)),
    "R11": (datetime(2023, 1, 22, 20), datetime(2023, 1, 23, 21)),
}

# ── S3 helpers ─────────────────────────────────────────────────────────────────

def _s3():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _urma_key(dt: datetime) -> str:
    """S3 key for URMA pcp_01h.wexp for a given valid-hour datetime."""
    day = dt.strftime("%Y%m%d")
    hh  = dt.strftime("%Y%m%d%H")
    return f"urma2p5.{day}/urma2p5.{hh}.pcp_01h.wexp.grb2"


def _download_urma(s3, dt: datetime) -> Optional[Path]:
    """Download URMA pcp_01h.wexp for dt; return local path (cached)."""
    key   = _urma_key(dt)
    fname = Path(key).name
    local = TMP_CACHE / fname
    if local.exists() and local.stat().st_size > 10_000:
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = s3.get_object(Bucket=URMA_BUCKET, Key=key)
        data = b""
        while chunk := resp["Body"].read(4 * 1024 * 1024):
            data += chunk
        local.write_bytes(data)
        return local
    except Exception as exc:
        print(f"      DOWNLOAD FAILED {dt:%Y-%m-%dT%H}Z  key={key}  err={exc}")
        return None


def _hourly_range(start: datetime, end: datetime) -> list[datetime]:
    """Inclusive hourly range."""
    hrs = []
    t = start
    while t <= end:
        hrs.append(t)
        t += timedelta(hours=1)
    return hrs

# ── GRIB decoding ──────────────────────────────────────────────────────────────

def _load_tp_array(path: Path) -> Optional[np.ndarray]:
    """
    Load the Total Precipitation 2-D array from a URMA pcp_01h.wexp.grb2 file.
    Returns float64 array [1597, 2345] in kg m**-2 (= mm water-equivalent).
    Returns None on failure.
    """
    # cfgrib
    try:
        import cfgrib
        for ds in cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""}):
            for vn in list(ds.data_vars):
                da  = ds[vn]
                arr = np.squeeze(np.asarray(da.values, dtype=np.float64))
                if arr.ndim == 2 and arr.shape == (1597, 2345):
                    arr = np.where(np.isfinite(arr), arr, 0.0)
                    arr = np.maximum(arr, 0.0)
                    return arr
    except Exception as exc:
        print(f"      cfgrib failed: {exc}")

    # eccodes fallback
    try:
        from eccodes import codes_grib_new_from_file, codes_get_values, codes_release
        with path.open("rb") as fh:
            gid = codes_grib_new_from_file(fh)
            if gid is not None:
                try:
                    vals = codes_get_values(gid)
                    if len(vals) == 1597 * 2345:
                        arr = np.array(vals, dtype=np.float64).reshape(1597, 2345)
                        arr = np.where(np.isfinite(arr), arr, 0.0)
                        arr = np.maximum(arr, 0.0)
                        return arr
                finally:
                    codes_release(gid)
    except Exception as exc:
        print(f"      eccodes fallback failed: {exc}")

    return None

# ── Basin extraction ───────────────────────────────────────────────────────────

def _extract_basin_mean(arr: np.ndarray, w_staid: pd.DataFrame) -> float:
    """
    Compute area-weighted basin mean from a 2-D URMA grid using existing RTMA weights.
    w_staid: rows of pilot_rtma_weights for one STAID.
    """
    rows = w_staid["row_idx"].values.astype(int)
    cols = w_staid["col_idx"].values.astype(int)
    wts  = w_staid["normalized_weight"].values.astype(np.float64)
    vals = arr[rows, cols]
    return float(np.dot(vals, wts))


def _download_and_extract(
    s3,
    hours: list[datetime],
    weights_by_staid: dict[str, pd.DataFrame],
    label: str,
) -> pd.DataFrame:
    """
    Download URMA pcp_01h.wexp for each hour, extract basin means for all STAIDs.
    Returns DataFrame with columns: valid_hour_utc, staid, urma_precip_mm.
    """
    records = []
    n = len(hours)
    for i, dt in enumerate(hours):
        print(f"  [{label}] {dt:%Y-%m-%dT%HZ}  ({i+1}/{n}) … ", end="", flush=True)
        t0   = time.time()
        path = _download_urma(s3, dt)
        dl_s = time.time() - t0
        if path is None:
            print("SKIP (download failed)")
            continue
        arr = _load_tp_array(path)
        if arr is None:
            print(f"SKIP (decode failed)  dl={dl_s:.1f}s")
            continue
        for staid, w_df in weights_by_staid.items():
            bmean = _extract_basin_mean(arr, w_df)
            records.append({
                "valid_hour_utc": dt,
                "staid":          staid,
                "urma_precip_mm": round(bmean, 6),
            })
        print(f"ok  dl={dl_s:.1f}s  {arr[arr>0].mean():.2f} mm mean-nonzero")
    return pd.DataFrame(records)

# ── Statistics helpers ─────────────────────────────────────────────────────────

def _lag_stats(urma: np.ndarray, mrms: np.ndarray) -> dict:
    """Pearson r, RMSE, MAE over paired arrays (NaN-safe)."""
    mask = np.isfinite(urma) & np.isfinite(mrms)
    u, m = urma[mask], mrms[mask]
    if len(u) < 3:
        return {"r": np.nan, "rmse": np.nan, "mae": np.nan, "n": int(len(u))}
    r     = float(np.corrcoef(u, m)[0, 1])
    rmse  = float(np.sqrt(np.mean((u - m) ** 2)))
    mae   = float(np.mean(np.abs(u - m)))
    return {"r": round(r, 4), "rmse": round(rmse, 4), "mae": round(mae, 4),
            "n": int(len(u))}


def _peak_time(series: pd.Series) -> Optional[datetime]:
    if series.empty or series.isna().all():
        return None
    return series.idxmax()

# ── MRMS loader ────────────────────────────────────────────────────────────────

def _load_mrms(staid: str) -> pd.Series:
    """Load MRMS basin-mean mm series indexed by tz-naive UTC datetime."""
    df = pd.read_parquet(MRMS_STATS_PQ)
    sub = df[df["STAID"] == staid].copy()
    vt  = pd.to_datetime(sub["valid_time_utc"])
    # Strip timezone so index is tz-naive (all times are UTC by convention)
    if vt.dt.tz is not None:
        vt = vt.dt.tz_convert("UTC").dt.tz_localize(None)
    sub["vt"] = vt
    sub = sub.sort_values("vt").set_index("vt")
    return sub["weighted_mean"].rename("mrms_precip_mm")

# ── Part 1.5 — Timestamp convention check ─────────────────────────────────────

def run_part15(s3, weights: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("Part 1.5 — URMA pcp_01h Timestamp Convention Check (R02)")
    print("=" * 70)

    PART15_DIR.mkdir(parents=True, exist_ok=True)
    staid = CANDIDATES["R02"]["staid"]
    w_r02 = weights[weights["STAID"] == staid]

    hours = _hourly_range(PART15_START, PART15_END)
    print(f"  Downloading {len(hours)} URMA pcp_01h.wexp files for R02 window …")

    df_urma = _download_and_extract(s3, hours, {staid: w_r02}, "R02-ts")
    if df_urma.empty or len(df_urma) < 5:
        return {
            "pass": False,
            "verdict": "STOP_insufficient_urma_data",
            "reason": f"Only {len(df_urma)} URMA hours decoded; need ≥ 5.",
        }

    urma_s = (
        df_urma[df_urma["staid"] == staid]
        .set_index("valid_hour_utc")["urma_precip_mm"]
    )

    mrms_full = _load_mrms(staid)

    # Build comparison table including all lag variants
    mrms_window = mrms_full[
        (mrms_full.index >= PART15_START - timedelta(hours=2)) &
        (mrms_full.index <= PART15_END   + timedelta(hours=2))
    ]

    common_times = sorted(set(urma_s.index) & set(mrms_window.index))
    if len(common_times) < 5:
        return {
            "pass": False,
            "verdict": "STOP_no_overlapping_times",
            "reason": "Cannot align URMA and MRMS time indices.",
        }

    # Convention A: URMA[t] vs MRMS[t]   (filename HH = end of accumulation)
    # Convention B: URMA[t] vs MRMS[t+1] (filename HH = start of accumulation)
    # Convention C: URMA[t] vs MRMS[t-1] (shift URMA 1h earlier)

    def _aligned_pair(lag_hours: int) -> tuple[np.ndarray, np.ndarray]:
        """For lag_hours: MRMS time = URMA filename time + lag_hours."""
        u_vals, m_vals = [], []
        for t_urma in sorted(urma_s.index):
            t_mrms = t_urma + timedelta(hours=lag_hours)
            if t_mrms in mrms_full.index:
                u_vals.append(urma_s[t_urma])
                m_vals.append(mrms_full[t_mrms])
        return np.array(u_vals), np.array(m_vals)

    u_a, m_a = _aligned_pair(0)    # A: filename = end
    u_b, m_b = _aligned_pair(1)    # B: filename = start → compare to MRMS one hour later
    u_c, m_c = _aligned_pair(-1)   # C: compare to MRMS one hour earlier

    stats_a = _lag_stats(u_a, m_a)
    stats_b = _lag_stats(u_b, m_b)
    stats_c = _lag_stats(u_c, m_c)

    # Peak time check for convention A (most likely)
    urma_peak_a = _peak_time(urma_s)
    mrms_peak   = _peak_time(mrms_full[mrms_full.index.isin(urma_s.index)])
    peak_diff_a = None
    if urma_peak_a and mrms_peak:
        peak_diff_a = int((urma_peak_a - mrms_peak).total_seconds() / 3600)

    print(f"\n  Lag comparison results (n = {stats_a['n']} hours):")
    print(f"  {'Convention':<45s}  {'r':>6s}  {'RMSE':>6s}  {'MAE':>6s}")
    print(f"  {'-'*67}")
    for name, s in [("A: URMA[t] vs MRMS[t]   (filename = end)", stats_a),
                    ("B: URMA[t] vs MRMS[t+1] (filename = start)", stats_b),
                    ("C: URMA[t] vs MRMS[t-1] (URMA 1h earlier)", stats_c)]:
        print(f"  {name:<45s}  {s['r']:>6.3f}  {s['rmse']:>6.3f}  {s['mae']:>6.3f}")
    print()
    print(f"  URMA peak hour: {urma_peak_a}  MRMS peak hour: {mrms_peak}  "
          f"Δ = {peak_diff_a:+d}h" if peak_diff_a is not None else "")

    # Verdict logic
    rs  = {"A": stats_a["r"], "B": stats_b["r"], "C": stats_c["r"]}
    best_conv = max(rs, key=lambda k: rs[k] if np.isfinite(rs[k]) else -1)
    best_r    = rs[best_conv]
    second_r  = sorted(rs.values(), reverse=True)[1]
    margin    = best_r - second_r

    if not np.isfinite(best_r):
        ts_verdict = "STOP_no_valid_correlation"
        ts_pass    = False
        reason     = "All correlation values are NaN — series may be all-zero."
    elif best_r < 0.5:
        ts_verdict = "STOP_low_correlation"
        ts_pass    = False
        reason     = f"Best correlation {best_r:.3f} < 0.5 threshold."
    elif margin < 0.05:
        ts_verdict = "AMBIGUOUS"
        ts_pass    = False
        reason     = (
            f"Best convention {best_conv} (r={best_r:.3f}) leads second by "
            f"only {margin:.3f} (<0.05 threshold) — cannot clearly determine convention."
        )
    else:
        ts_verdict = f"PASS_convention_{best_conv}"
        ts_pass    = True
        _best_stats = {"A": stats_a, "B": stats_b, "C": stats_c}[best_conv]
        reason     = (
            f"Convention {best_conv} is clearly best: r={best_r:.3f}, "
            f"RMSE={_best_stats['rmse']:.3f} mm, "
            f"margin over second={margin:.3f}."
        )

    print(f"  VERDICT: {ts_verdict}")
    print(f"  {reason}")

    # Detailed comparison table
    rows = []
    for t_urma in sorted(urma_s.index):
        row = {
            "urma_filename_hour_utc": t_urma.isoformat(),
            "urma_precip_mm":         round(float(urma_s[t_urma]), 4),
            "mrms_t0_mm":  round(float(mrms_full.get(t_urma, np.nan)), 4),
            "mrms_t_plus1_mm": round(float(mrms_full.get(t_urma + timedelta(hours=1), np.nan)), 4),
            "mrms_t_minus1_mm": round(float(mrms_full.get(t_urma - timedelta(hours=1), np.nan)), 4),
        }
        rows.append(row)
    ts_df = pd.DataFrame(rows)
    ts_df.to_csv(PART15_DIR / "r02_urma_vs_mrms_timeseries.csv", index=False)

    # Lag metrics CSV
    lag_df = pd.DataFrame([
        {"convention": "A_filename_eq_end",   **stats_a,
         "peak_urma": str(urma_peak_a), "peak_mrms": str(mrms_peak),
         "peak_diff_h": peak_diff_a},
        {"convention": "B_filename_eq_start",  **stats_b},
        {"convention": "C_urma_1h_earlier",    **stats_c},
    ])
    lag_df.to_csv(PART15_DIR / "r02_lag_comparison.csv", index=False)

    # Figure
    _plot_part15(urma_s, mrms_full, PART15_DIR / "r02_timeseries_comparison.png",
                 best_conv, stats_a, stats_b, stats_c)

    result = {
        "pass":           ts_pass,
        "verdict":        ts_verdict,
        "reason":         reason,
        "best_convention": best_conv,
        "best_r":         best_r,
        "margin":         round(margin, 4),
        "stats_A":        stats_a,
        "stats_B":        stats_b,
        "stats_C":        stats_c,
        "urma_peak_hour": str(urma_peak_a),
        "mrms_peak_hour": str(mrms_peak),
        "peak_diff_h":    peak_diff_a,
        "n_hours":        len(urma_s),
        "outputs": {
            "timeseries_csv": str(PART15_DIR / "r02_urma_vs_mrms_timeseries.csv"),
            "lag_csv":        str(PART15_DIR / "r02_lag_comparison.csv"),
            "figure_png":     str(PART15_DIR / "r02_timeseries_comparison.png"),
        },
    }

    _write_part15_md(result, ts_df, PART15_DIR / "timestamp_check_report.md")
    json_path = PART15_DIR / "timestamp_check_report.json"
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(f"\n  Outputs written to {PART15_DIR}")
    return result


def _plot_part15(
    urma_s: pd.Series,
    mrms_full: pd.Series,
    out_path: Path,
    best_conv: str,
    stats_a: dict,
    stats_b: dict,
    stats_c: dict,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig.suptitle(
        "Part 1.5 — URMA pcp_01h vs MRMS QPE: timestamp convention check (R02, AR)\n"
        "Diagnostic only. URMA is NOT a Stage 1 model input.",
        fontsize=11,
    )

    mrms_win = mrms_full[
        (mrms_full.index >= PART15_START - timedelta(hours=1)) &
        (mrms_full.index <= PART15_END   + timedelta(hours=1))
    ]

    conv_labels = {
        "A": f"Conv A: URMA[t] vs MRMS[t]  r={stats_a['r']:.3f}  RMSE={stats_a['rmse']:.2f} mm",
        "B": f"Conv B: URMA[t] vs MRMS[t+1] r={stats_b['r']:.3f}  RMSE={stats_b['rmse']:.2f} mm",
        "C": f"Conv C: URMA[t] vs MRMS[t-1] r={stats_c['r']:.3f}  RMSE={stats_c['rmse']:.2f} mm",
    }
    offsets = {"A": 0, "B": 1, "C": -1}

    for ax, (conv, offset) in zip(axes, offsets.items()):
        mrms_shifted = mrms_full.copy()
        mrms_shifted.index = mrms_shifted.index - timedelta(hours=offset)
        mrms_aligned = mrms_shifted[mrms_shifted.index.isin(urma_s.index)]
        common       = sorted(set(urma_s.index) & set(mrms_aligned.index))
        if not common:
            continue
        u_vals = urma_s.reindex(common).values
        m_vals = mrms_aligned.reindex(common).values

        ax.bar(common, u_vals, width=0.035, label="URMA pcp_01h", color="#4878CF",
               alpha=0.7)
        ax.plot(common, m_vals, "o-", color="#E87722", lw=1.5,
                ms=4, label="MRMS QPE 1h")
        ax.set_ylabel("Precipitation (mm)", fontsize=8)
        ax.set_title(conv_labels[conv] + (" ← BEST" if conv == best_conv else ""),
                     fontsize=9, fontweight="bold" if conv == best_conv else "normal")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %HZ"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.text(0.5, 0.01,
             "Note: URMA precipitation is a qualitative RTMA/URMA-family consistency check. "
             "It does not prove RTMA wind/temperature physics.",
             ha="center", fontsize=7, color="gray")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {out_path.name}")


def _write_part15_md(result: dict, ts_df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Part 1.5 — URMA pcp_01h Timestamp Convention Check",
        "",
        f"**Verdict**: `{result['verdict']}`  ",
        f"**Pass**: {result['pass']}  ",
        f"**Reason**: {result['reason']}",
        "",
        "## Lag statistics",
        "",
        "| Convention | r | RMSE (mm) | MAE (mm) | n |",
        "|---|---|---|---|---|",
        f"| A — filename HH = end of accumulation | **{result['stats_A']['r']}** | "
        f"**{result['stats_A']['rmse']}** | {result['stats_A']['mae']} | {result['stats_A']['n']} |",
        f"| B — filename HH = start of accumulation | {result['stats_B']['r']} | "
        f"{result['stats_B']['rmse']} | {result['stats_B']['mae']} | {result['stats_B']['n']} |",
        f"| C — URMA 1h earlier | {result['stats_C']['r']} | "
        f"{result['stats_C']['rmse']} | {result['stats_C']['mae']} | {result['stats_C']['n']} |",
        "",
        f"Best convention: **{result['best_convention']}**  "
        f"(best r={result['best_r']}, margin={result['margin']})",
        "",
        "## Peak-time agreement",
        "",
        f"- URMA basin-mean peak: {result['urma_peak_hour']}",
        f"- MRMS basin-mean peak: {result['mrms_peak_hour']}",
        f"- Difference: {result['peak_diff_h']:+d} h" if result['peak_diff_h'] is not None else
        "- Difference: N/A",
        "",
        "## URMA vs MRMS time series (Convention A)",
        "",
        "| URMA filename hour | URMA (mm) | MRMS[t] (mm) | MRMS[t+1] (mm) | MRMS[t-1] (mm) |",
        "|---|---|---|---|---|",
    ]
    for _, row in ts_df.iterrows():
        lines.append(
            f"| {row['urma_filename_hour_utc']} "
            f"| {row['urma_precip_mm']:.3f} "
            f"| {row['mrms_t0_mm']:.3f} "
            f"| {row['mrms_t_plus1_mm']:.3f} "
            f"| {row['mrms_t_minus1_mm']:.3f} |"
        )
    lines += [
        "",
        "---",
        "*Diagnostic-only. URMA precipitation is NOT a Stage 1 model input.*",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

# ── Part 2 — Tiny extraction pilot ────────────────────────────────────────────

def run_part2(s3, weights: pd.DataFrame, timestamp_convention: str) -> dict:
    print("\n" + "=" * 70)
    print("Part 2 — Tiny URMA/MRMS Extraction Pilot (R02, R06, R11)")
    print("=" * 70)
    print(f"  Using timestamp convention: {timestamp_convention}")
    print("  Diagnostic-only. URMA precipitation is NOT a Stage 1 model input.")

    PART2_DIR.mkdir(parents=True, exist_ok=True)

    lag_hours = {"A": 0, "B": 1, "C": -1}[timestamp_convention]

    all_records: list[dict] = []
    all_metrics: list[dict] = []
    figures: dict[str, str] = {}

    for rid, cfg in CANDIDATES.items():
        staid  = cfg["staid"]
        w_rid  = weights[weights["STAID"] == staid]
        start, end = PART2_WINDOWS[rid]
        hours  = _hourly_range(start, end)

        print(f"\n  {rid} ({cfg['state']}, {staid}) — {len(hours)} hours "
              f"  [{start:%Y-%m-%dT%HZ} – {end:%Y-%m-%dT%HZ}]")

        df_u = _download_and_extract(s3, hours, {staid: w_rid}, rid)
        if df_u.empty:
            print(f"  WARN: no URMA data for {rid}, skipping")
            continue

        urma_s = (
            df_u[df_u["staid"] == staid]
            .set_index("valid_hour_utc")["urma_precip_mm"]
        )
        mrms_full = _load_mrms(staid)

        # Apply timestamp convention
        mrms_aligned = pd.Series(
            {t_urma: mrms_full.get(t_urma + timedelta(hours=lag_hours), np.nan)
             for t_urma in urma_s.index},
            name="mrms_precip_mm",
        )

        # Metrics
        u_arr = urma_s.values
        m_arr = mrms_aligned.values
        valid = np.isfinite(u_arr) & np.isfinite(m_arr)
        stats = _lag_stats(u_arr[valid], m_arr[valid])
        all_metrics.append({
            "candidate_id":   rid,
            "staid":          staid,
            "state":          cfg["state"],
            "category":       cfg["category"],
            "n_hours":        int(valid.sum()),
            "urma_peak_mm":   round(float(u_arr.max()), 4),
            "mrms_peak_mm":   round(float(np.nanmax(m_arr)), 4),
            "urma_total_mm":  round(float(u_arr.sum()), 4),
            "mrms_total_mm":  round(float(np.nansum(m_arr)), 4),
            "correlation_r":  stats["r"],
            "rmse_mm":        stats["rmse"],
            "mae_mm":         stats["mae"],
            "convention":     timestamp_convention,
            "lag_hours":      lag_hours,
        })

        # Records for parquet
        for t_urma in sorted(urma_s.index):
            all_records.append({
                "candidate_id":        rid,
                "staid":               staid,
                "state":               cfg["state"],
                "valid_hour_urma_utc": t_urma.isoformat(),
                "urma_precip_mm":      round(float(urma_s[t_urma]), 6),
                "mrms_precip_mm":      round(float(mrms_aligned.get(t_urma, np.nan)), 6),
                "timestamp_convention": timestamp_convention,
                "lag_hours":           lag_hours,
            })

        # Figure
        fig_path = PART2_DIR / f"{rid.lower()}_urma_mrms_comparison.png"
        _plot_comparison(rid, cfg, urma_s, mrms_aligned, stats, lag_hours,
                         timestamp_convention, fig_path)
        figures[rid] = str(fig_path)
        print(f"  {rid}: r={stats['r']:.3f}  RMSE={stats['rmse']:.3f} mm  "
              f"URMA-peak={u_arr.max():.2f} mm  MRMS-peak={np.nanmax(m_arr):.2f} mm")

    if not all_records:
        return {"pass": False, "reason": "No URMA records extracted for any candidate."}

    # Write parquet
    df_all = pd.DataFrame(all_records)
    pq_path = PART2_DIR / "urma_basin_means_all_candidates.parquet"
    df_all.to_parquet(pq_path, index=False)
    preview_path = PART2_DIR / "urma_basin_means_preview.csv"
    df_all.head(50).to_csv(preview_path, index=False)

    # Metrics CSV/MD
    df_met = pd.DataFrame(all_metrics)
    df_met.to_csv(PART2_DIR / "metrics_summary.csv", index=False)
    _write_metrics_md(df_met, figures, timestamp_convention, lag_hours,
                      PART2_DIR / "metrics_summary.md")

    # Pilot report
    _write_pilot_report(df_met, figures, timestamp_convention,
                        PART2_DIR / "pilot_report.md")

    return {
        "pass":    True,
        "records": len(all_records),
        "metrics": all_metrics,
        "outputs": {
            "parquet":       str(pq_path),
            "preview_csv":   str(preview_path),
            "metrics_csv":   str(PART2_DIR / "metrics_summary.csv"),
            "metrics_md":    str(PART2_DIR / "metrics_summary.md"),
            "pilot_report":  str(PART2_DIR / "pilot_report.md"),
            "figures":       figures,
        },
    }


def _plot_comparison(
    rid: str,
    cfg: dict,
    urma_s: pd.Series,
    mrms_s: pd.Series,
    stats: dict,
    lag_hours: int,
    convention: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    times   = sorted(urma_s.index)
    u_vals  = [float(urma_s.get(t, np.nan)) for t in times]
    m_vals  = [float(mrms_s.get(t, np.nan)) for t in times]

    bar_w = 0.032
    ax.bar(times, u_vals, width=bar_w, color="#4878CF", alpha=0.75,
           label="URMA QPE 1h (pcp_01h.wexp)")
    ax.plot(times, m_vals, "o-", color="#E87722", lw=1.8, ms=4,
            label="MRMS QPE 1h Pass1")

    lag_label = (f"Conv {convention}: URMA filename HH = "
                 + ("end of accumulation" if lag_hours == 0 else
                    "start of accumulation" if lag_hours == 1 else
                    "1h earlier"))
    ax.set_title(
        f"{rid} ({cfg['state']}, STAID {cfg['staid']}) — URMA QPE vs MRMS QPE\n"
        f"{lag_label}  |  r={stats['r']:.3f}  RMSE={stats['rmse']:.2f} mm  "
        f"n={stats['n']} hours",
        fontsize=10,
    )
    ax.set_ylabel("Basin-mean precipitation (mm)", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %HZ"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.text(0.5, 0.01,
             "Diagnostic-only: URMA precipitation is NOT a Stage 1 model input and "
             "does not prove RTMA wind/temperature physics.",
             ha="center", fontsize=7, color="gray", style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {out_path.name}")


def _write_metrics_md(df: pd.DataFrame, figures: dict,
                      convention: str, lag_hours: int, path: Path) -> None:
    lag_desc = (
        "URMA filename HH = end of 1h accumulation (URMA[t] aligned to MRMS[t])"
        if lag_hours == 0 else
        f"URMA filename HH = start of accumulation (lag={lag_hours:+d}h)"
    )
    lines = [
        "# URMA QPE vs MRMS QPE — Metrics Summary",
        "",
        f"**Timestamp convention {convention}**: {lag_desc}  ",
        "**Diagnostic-only. URMA precipitation is NOT a Stage 1 model input.**",
        "",
        "## Per-candidate metrics",
        "",
        "| Candidate | STAID | State | n hrs | r | RMSE (mm) | MAE (mm) | "
        "URMA peak (mm) | MRMS peak (mm) | URMA total (mm) | MRMS total (mm) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['candidate_id']} | {row['staid']} | {row['state']} "
            f"| {row['n_hours']} "
            f"| **{row['correlation_r']:.3f}** "
            f"| {row['rmse_mm']:.3f} "
            f"| {row['mae_mm']:.3f} "
            f"| {row['urma_peak_mm']:.2f} "
            f"| {row['mrms_peak_mm']:.2f} "
            f"| {row['urma_total_mm']:.1f} "
            f"| {row['mrms_total_mm']:.1f} |"
        )
    lines += ["", "## Figures", ""]
    for rid, fig_path in figures.items():
        lines.append(f"- [{rid}]({Path(fig_path).name})")
    lines += ["", "---", "*Diagnostic-only output.*"]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_pilot_report(df: pd.DataFrame, figures: dict,
                        convention: str, path: Path) -> None:
    lines = [
        "# URMA QPE vs MRMS QPE — Pilot Diagnostic Report",
        "",
        "## What this diagnostic is",
        "",
        "This is a **URMA QPE vs MRMS QPE diagnostic** for RTMA/URMA-family "
        "grid, weight, and timing consistency in the Flash-NH Stage 1 pilot.",
        "",
        "**What it shows**: whether URMA hourly precipitation (pcp_01h) produces "
        "basin-mean values that correlate with MRMS QPE 1h Pass1 for the same "
        "basins and time windows as the event candidates.",
        "",
        "**What it does NOT show**:",
        "- It does not make URMA precipitation a Stage 1 model input.",
        "- It does not validate RTMA temperature, wind, or humidity physics.",
        "- It does not replace MRMS QPE as the precipitation forcing.",
        "- Correlation may reflect the spatial weight table, not meteorological skill.",
        "",
        "## Product interpretation",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| URMA product | `urma2p5.YYYYMMDDHH.pcp_01h.wexp.grb2` |",
        f"| Variable | `tp` — Total Precipitation, `kg m**-2` (= mm water-equivalent) |",
        f"| Accumulation | 1 hour |",
        f"| Timestamp convention used | {convention} — filename HH = end of accumulation |",
        f"| Grid | 1597 × 2345 Lambert Conformal Conic 2.5 km CONUS |",
        f"| Weight table | `pilot_rtma_weights.parquet` (reused exactly, unchanged) |",
        "",
        "## Limitations",
        "",
        "- `cfgrib` cannot decode `stepRange` from these GRIB files (`step_range='?'`).",
        "  The 1h accumulation window is inferred from the filename suffix `pcp_01h`.",
        "- URMA QPE assimilates Stage IV and gauge data; it is not an independent product.",
        "- The timestamp convention was confirmed empirically from R02 only.",
        "  R06 and R11 are treated as independent validation.",
        "",
        "## Results",
        "",
    ]
    for _, row in df.iterrows():
        lines += [
            f"### {row['candidate_id']} — {row['state']} ({row['category']})",
            "",
            f"- r = **{row['correlation_r']:.3f}**  RMSE = {row['rmse_mm']:.3f} mm",
            f"- URMA peak = {row['urma_peak_mm']:.2f} mm  |  "
            f"MRMS peak = {row['mrms_peak_mm']:.2f} mm",
            f"- URMA window total = {row['urma_total_mm']:.1f} mm  |  "
            f"MRMS window total = {row['mrms_total_mm']:.1f} mm",
            "",
        ]
    lines += [
        "## Next steps (pending user approval)",
        "",
        "If correlation is sufficiently high (r > 0.7) for all three candidates "
        "and the timestamp convention is confirmed, URMA precipitation could be "
        "used as a qualitative diagnostic overlay in future event animations. "
        "This requires explicit approval before any animation or model-input changes.",
        "",
        "---",
        "*Diagnostic-only. Model inputs UNCHANGED. Do not merge URMA precipitation "
        "into Stage 1 forcing.*",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    t_wall = time.time()

    for d in (PART15_DIR, PART2_DIR, TMP_CACHE):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("URMA pcp_01h Timestamp Gate + Tiny Extraction Pilot")
    print("Diagnostic-only. Model inputs UNCHANGED.")
    print("=" * 70)

    # Load weight table
    print("\nLoading RTMA weight table …")
    weights = pd.read_parquet(RTMA_WEIGHTS_PQ)
    weights["STAID"] = weights["STAID"].astype(str)
    print(f"  {len(weights)} rows, {weights['STAID'].nunique()} basins")

    s3 = _s3()

    # ── Part 1.5 ────────────────────────────────────────────────────────────
    p15_result = run_part15(s3, weights)

    if not p15_result.get("pass"):
        verdict = p15_result.get("verdict", "UNKNOWN")
        reason  = p15_result.get("reason", "")
        print(f"\nSTOP after Part 1.5: {verdict}")
        print(f"Reason: {reason}")
        print("Do not proceed to extraction.")
        wall_s = round(time.time() - t_wall, 1)
        summary = {
            "overall_pass":     False,
            "stop_stage":       "part1p5",
            "part1p5_verdict":  verdict,
            "part1p5_reason":   reason,
            "runtime_s":        wall_s,
        }
        _save_top_summary(summary)
        return 1

    best_conv = p15_result["best_convention"]
    print(f"\nPart 1.5 PASS — proceeding to Part 2 with convention {best_conv}")

    # ── Part 2 ──────────────────────────────────────────────────────────────
    p2_result = run_part2(s3, weights, best_conv)

    wall_s = round(time.time() - t_wall, 1)
    overall_pass = p2_result.get("pass", False)

    summary = {
        "overall_pass":     overall_pass,
        "runtime_s":        wall_s,
        "part1p5":          p15_result,
        "part2":            p2_result,
        "timestamp_convention": best_conv,
        "convention_meaning": (
            "filename HH = end of 1h accumulation (URMA[t] aligns to MRMS[t])"
            if best_conv == "A" else
            "filename HH = start of accumulation" if best_conv == "B" else
            "URMA 1h earlier than MRMS"
        ),
        "diagnostic_only":       True,
        "modifies_model_inputs": False,
    }
    _save_top_summary(summary)

    # ── Terminal summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Runtime            : {wall_s} s")
    print(f"  Part 1.5 verdict   : {p15_result['verdict']}")
    print(f"  Timestamp convention: {best_conv}  "
          f"(r={p15_result['best_r']:.3f}, margin={p15_result['margin']:.3f})")
    if "metrics" in p2_result:
        for m in p2_result["metrics"]:
            print(f"  Part 2 {m['candidate_id']}        : r={m['correlation_r']:.3f}  "
                  f"RMSE={m['rmse_mm']:.3f} mm  "
                  f"URMA-peak={m['urma_peak_mm']:.2f}  MRMS-peak={m['mrms_peak_mm']:.2f}")
    print()
    print("  Outputs:")
    print(f"    Part 1.5 dir : {PART15_DIR}")
    print(f"    Part 2 dir   : {PART2_DIR}")
    if "outputs" in p2_result:
        for k, v in p2_result["outputs"].items():
            if isinstance(v, str):
                print(f"      {k}: {Path(v).name}")
            elif isinstance(v, dict):
                for rid, fp in v.items():
                    print(f"      figure {rid}: {Path(fp).name}")
    print("=" * 70)

    return 0 if overall_pass else 1


def _save_top_summary(summary: dict) -> None:
    p = DIAG_DIR / "timestamp_and_pilot_summary.json"
    p.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nTop-level summary: {p}")


if __name__ == "__main__":
    sys.exit(main())

# scripts/plot_flash_diagnostics.py
from __future__ import annotations

from pathlib import Path
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.data_usgs import fetch_usgs_iv_discharge
from src.aggregation import to_hourly_candidates
from src.events import detect_pot_events

import json
from datetime import datetime
import textwrap


# ----------------------------
# Helpers
# ----------------------------
def zfill8(x: str) -> str:
    return str(x).strip().zfill(8)


def max_slope_native(series: pd.Series) -> float:
    """Max dQ/dt using native sampling intervals (Q units per second)."""
    s = series.dropna().sort_index()
    if len(s) < 3:
        return np.nan
    dq = s.diff().dropna()
    dt = s.index.to_series().diff().dropna().dt.total_seconds()
    if len(dq) != len(dt) or len(dt) == 0:
        return np.nan
    slopes = dq.values / dt.values
    return float(np.nanmax(slopes))


def max_slope_1h_from_hi(q_hi: pd.Series) -> float:
    """
    Max 1-hour slope computed from high-res series:
    max over (Q(t) - Q(t-1h))/3600 across timestamps where t-1h aligns.
    Uses cadence-aware fallback if not 15-min.
    """
    s = q_hi.dropna().sort_index()
    if len(s) < 10:
        return np.nan

    # detect common cadence
    diffs = s.index.to_series().diff().dropna()
    dt_mode = diffs.mode()
    dt_min = float(dt_mode.iloc[0].total_seconds() / 60.0) if len(dt_mode) else np.nan

    # fast path for 15-min cadence
    if np.isfinite(dt_min) and abs(dt_min - 15) < 1e-6:
        dq1h = (s - s.shift(4)).dropna()
        slopes = dq1h.values / 3600.0
        return float(np.nanmax(slopes)) if len(slopes) else np.nan

    # time-based fallback: reindex at t-1h exactly
    idx = s.index
    s_prev = s.reindex(idx - pd.Timedelta("1h"))
    dq1h = (s.values - s_prev.values)
    slopes = dq1h / 3600.0
    return float(np.nanmax(slopes)) if np.isfinite(slopes).any() else np.nan


def max_slope_1h_from_hourly(q_hr: pd.Series) -> float:
    """Max 1-hour slope from hourly series (diff / 3600)."""
    s = q_hr.dropna().sort_index()
    if len(s) < 3:
        return np.nan
    slopes = s.diff().dropna().values / 3600.0
    return float(np.nanmax(slopes)) if len(slopes) else np.nan


def pick_hourly_peak(q_hr: pd.Series, t0: pd.Timestamp, win: pd.Timedelta) -> tuple[pd.Timestamp, float] | tuple[None, float]:
    """Return (t_peak, q_peak) within +/- win; (None, nan) if empty."""
    hr_win = q_hr.loc[t0 - win : t0 + win].dropna()
    if hr_win.empty:
        return None, np.nan
    t1 = hr_win.idxmax()
    return t1, float(hr_win.loc[t1])


def event_table_for_quantile(
    gauge: str,
    year: int,
    area_km2: float,
    q_iv: pd.Series,
    q_hr_near: pd.Series,
    q_hr_mean: pd.Series,
    quantile: float,
    min_sep: str,
    search_window: str,
) -> pd.DataFrame:
    """
    Event-level table (one row per POT event detected on hi-res series).
    Includes:
      - hi peak time/value
      - hourly (nearest/mean) matched peaks
      - peak errors/timing errors
      - extreme preservation (hourly exceeds hourly quantile threshold)
    """
    q_hi = q_iv.loc[f"{year}-01-01":f"{year}-12-31 23:59:59"].dropna()
    if q_hi.empty:
        return pd.DataFrame()

    events = detect_pot_events(q_hi, quantile=quantile, min_separation=min_sep)
    if events.empty:
        return pd.DataFrame()

    win = pd.Timedelta(search_window)

    # independent thresholds
    thr_hi = float(q_hi.quantile(quantile))
    thr_near = float(q_hr_near.dropna().quantile(quantile)) if len(q_hr_near.dropna()) else np.nan
    thr_mean = float(q_hr_mean.dropna().quantile(quantile)) if len(q_hr_mean.dropna()) else np.nan

    rows = []
    for _, ev in events.iterrows():
        t0 = ev["peak_time"]
        q0 = float(ev["peak_q"])

        t1n, q1n = pick_hourly_peak(q_hr_near, t0, win)
        t1m, q1m = pick_hourly_peak(q_hr_mean, t0, win)

        rows.append({
            "gauge": gauge,
            "year": year,
            "drainage_km2": area_km2,
            "quantile": quantile,

            "thr_hi": thr_hi,
            "thr_hr_near": thr_near,
            "thr_hr_mean": thr_mean,

            "t_peak_hi": t0,
            "q_peak_hi": q0,

            "t_peak_hr_near": t1n,
            "q_peak_hr_near": q1n,
            "detected_near": bool(t1n is not None),
            "rel_peak_error_near": (q1n - q0) / q0 if (np.isfinite(q1n) and q0 > 0) else np.nan,
            "dt_minutes_near": (t1n - t0).total_seconds() / 60.0 if t1n is not None else np.nan,
            "extreme_preserved_near": bool(q1n > thr_near) if (np.isfinite(q1n) and np.isfinite(thr_near)) else np.nan,

            "t_peak_hr_mean": t1m,
            "q_peak_hr_mean": q1m,
            "detected_mean": bool(t1m is not None),
            "rel_peak_error_mean": (q1m - q0) / q0 if (np.isfinite(q1m) and q0 > 0) else np.nan,
            "dt_minutes_mean": (t1m - t0).total_seconds() / 60.0 if t1m is not None else np.nan,
            "extreme_preserved_mean": bool(q1m > thr_mean) if (np.isfinite(q1m) and np.isfinite(thr_mean)) else np.nan,
        })

    return pd.DataFrame(rows)


def basin_summary_from_events(ev: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event table to basin-wise summaries per quantile."""
    if ev.empty:
        return pd.DataFrame()

    def agg_one(df: pd.DataFrame) -> dict:
        out = {}
        out["n_events"] = len(df)
        out["recall_near"] = float(df["detected_near"].mean())
        out["recall_mean"] = float(df["detected_mean"].mean())

        # extreme recall among detected
        dn = df[df["detected_near"]]
        dm = df[df["detected_mean"]]
        out["extreme_recall_near"] = float(dn["extreme_preserved_near"].dropna().astype(bool).mean()) if len(dn) else np.nan
        out["extreme_recall_mean"] = float(dm["extreme_preserved_mean"].dropna().astype(bool).mean()) if len(dm) else np.nan

        # errors among detected
        for col in ["rel_peak_error_near", "rel_peak_error_mean", "dt_minutes_near", "dt_minutes_mean"]:
            s = df.loc[df[col.replace("rel_", "detected_").replace("dt_", "detected_")] if "detected" in col else df.index, col]

        # simpler:
        out["med_rel_err_near"] = float(dn["rel_peak_error_near"].median()) if len(dn) else np.nan
        out["med_rel_err_mean"] = float(dm["rel_peak_error_mean"].median()) if len(dm) else np.nan
        out["p90_abs_dt_min_near"] = float(dn["dt_minutes_near"].abs().quantile(0.9)) if len(dn) else np.nan
        out["p90_abs_dt_min_mean"] = float(dm["dt_minutes_mean"].abs().quantile(0.9)) if len(dm) else np.nan
        return out

    gb = ev.groupby(["gauge", "quantile"], as_index=False)
    rows = []
    for (g, q), df in gb:
        base = {
            "gauge": g,
            "quantile": q,
            "drainage_km2": float(df["drainage_km2"].iloc[0]),
        }
        base.update(agg_one(df))
        rows.append(base)
    return pd.DataFrame(rows)


def panel_scatter(
    dfs_by_q: dict[float, pd.DataFrame],
    xcol: str,
    ycol: str,
    ccol: str,
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: Path,
    logxy: bool = True,
):
    qs = sorted(dfs_by_q.keys())
    k = len(qs)
    fig, axes = plt.subplots(1, k, figsize=(5 * k, 4), constrained_layout=True)

    # global ranges for consistent axes
    allx = np.concatenate([dfs_by_q[q][xcol].to_numpy() for q in qs])
    ally = np.concatenate([dfs_by_q[q][ycol].to_numpy() for q in qs])
    finite = np.isfinite(allx) & np.isfinite(ally) & (allx > 0) & (ally > 0)
    if finite.any():
        xmin, xmax = np.min(allx[finite]), np.max(allx[finite])
        ymin, ymax = np.min(ally[finite]), np.max(ally[finite])
    else:
        xmin = ymin = 1e-6
        xmax = ymax = 1.0

    # color range consistent
    allc = np.concatenate([dfs_by_q[q][ccol].to_numpy() for q in qs])
    cfinite = np.isfinite(allc)
    vmin = float(np.min(allc[cfinite])) if cfinite.any() else None
    vmax = float(np.max(allc[cfinite])) if cfinite.any() else None

    mappable = None
    for ax, q in zip(np.atleast_1d(axes), qs):
        d = dfs_by_q[q]
        x = d[xcol].to_numpy()
        y = d[ycol].to_numpy()
        c = d[ccol].to_numpy()

        if logxy:
            ax.set_xscale("log")
            ax.set_yscale("log")
        mappable = ax.scatter(x, y, c=c, vmin=vmin, vmax=vmax)
        # 1:1 line
        mn = min(xmin, ymin)
        mx = max(xmax, ymax)
        ax.plot([mn, mx], [mn, mx])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"Q={q}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    fig.suptitle(title)

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=np.atleast_1d(axes), shrink=0.9)
        cbar.set_label("drainage_km2")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def panel_cdf(
    dfs_by_q: dict[float, pd.DataFrame],
    series_specs: list[tuple[str, str]],  # (label, column)
    xlabel: str,
    title: str,
    outpath: Path,
    xscale: str | None = None,
):
    qs = sorted(dfs_by_q.keys())
    k = len(qs)
    fig, axes = plt.subplots(1, k, figsize=(5 * k, 4), constrained_layout=True)

    # determine x-limits globally for comparability
    allvals = []
    for q in qs:
        d = dfs_by_q[q]
        for _, col in series_specs:
            v = d[col].to_numpy()
            v = v[np.isfinite(v)]
            allvals.append(v)
    allvals = np.concatenate(allvals) if allvals else np.array([])
    if len(allvals):
        xmin, xmax = np.quantile(allvals, 0.01), np.quantile(allvals, 0.99)
        if xmin == xmax:
            xmin, xmax = float(np.min(allvals)), float(np.max(allvals))
    else:
        xmin, xmax = -1, 1

    for ax, q in zip(np.atleast_1d(axes), qs):
        d = dfs_by_q[q]
        for label, col in series_specs:
            v = d[col].to_numpy()
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            xs = np.sort(v)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.plot(xs, ys, label=label)

        ax.set_xlim(xmin, xmax)
        if xscale:
            ax.set_xscale(xscale)
        ax.set_title(f"Q={q}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Empirical CDF")
        ax.legend()

    fig.suptitle(title)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def format_time_axis(ax):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def plot_hydrograph_with_thresholds(
    gauge: str,
    year: int,
    quantile: float,
    q_iv: pd.Series,
    q_hr_near: pd.Series,
    q_hr_mean: pd.Series,
    outpath: Path,
    window_days: int = 2,
):
    q_hi_year = q_iv.loc[f"{year}-01-01":f"{year}-12-31 23:59:59"].dropna()
    if q_hi_year.empty:
        return

    # choose a representative event time: annual peak (simple and persuasive)
    t_peak = q_hi_year.idxmax()
    t0 = t_peak - pd.Timedelta(days=window_days)
    t1 = t_peak + pd.Timedelta(days=window_days)

    hi = q_iv.loc[t0:t1]
    hn = q_hr_near.loc[t0:t1]
    hm = q_hr_mean.loc[t0:t1]

    thr_hi = float(q_hi_year.quantile(quantile))
    thr_near = float(q_hr_near.dropna().quantile(quantile)) if len(q_hr_near.dropna()) else np.nan
    thr_mean = float(q_hr_mean.dropna().quantile(quantile)) if len(q_hr_mean.dropna()) else np.nan

    fig, ax = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)
    ax.plot(hi.index, hi.values, label="USGS IV (hi-res)")
    ax.plot(hn.index, hn.values, label="Hourly nearest")
    ax.plot(hm.index, hm.values, label="Hourly mean")

    # thresholds (distinct styles/colors)
    ax.axhline(thr_hi,   color="k",        linestyle="--", linewidth=1.2, label=f"Thr hi Q{quantile}")
    if np.isfinite(thr_near):
        ax.axhline(thr_near, color="tab:orange", linestyle=":",  linewidth=1.2, label=f"Thr near Q{quantile}")
    if np.isfinite(thr_mean):
        ax.axhline(thr_mean, color="tab:green",  linestyle="-.", linewidth=1.2, label=f"Thr mean Q{quantile}")

    ax.set_title(f"{gauge} window around annual peak ({year}), thresholds at Q={quantile}")
    ax.set_xlabel("time")
    ax.set_ylabel("Q (m3/s)")
    format_time_axis(ax)
    ax.legend(ncol=2)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_event_hydrograph(
    gauge: str,
    year: int,
    quantile: float,
    method: str,  # "near" or "mean"
    t_peak_hi: pd.Timestamp,
    q_iv: pd.Series,
    q_hr_near: pd.Series,
    q_hr_mean: pd.Series,
    outpath: Path,
    hours_before: int = 12,
    hours_after: int = 24,
):
    """
    Plot a window around a specific POT event peak time (hi-res peak).
    Shows hi-res, hourly nearest, hourly mean, and thresholds.
    Also marks the matched hourly peak time for the chosen method.
    """
    t0 = t_peak_hi - pd.Timedelta(hours=hours_before)
    t1 = t_peak_hi + pd.Timedelta(hours=hours_after)

    hi = q_iv.loc[t0:t1]
    hn = q_hr_near.loc[t0:t1]
    hm = q_hr_mean.loc[t0:t1]

    # thresholds computed on full-year series (consistent with event table)
    q_hi_year = q_iv.loc[f"{year}-01-01":f"{year}-12-31 23:59:59"].dropna()
    thr_hi = float(q_hi_year.quantile(quantile)) if len(q_hi_year) else np.nan
    thr_near = float(q_hr_near.dropna().quantile(quantile)) if len(q_hr_near.dropna()) else np.nan
    thr_mean = float(q_hr_mean.dropna().quantile(quantile)) if len(q_hr_mean.dropna()) else np.nan

    fig, ax = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)

    ax.plot(hi.index, hi.values, label="USGS IV (hi-res)")
    ax.plot(hn.index, hn.values, label="Hourly nearest")
    ax.plot(hm.index, hm.values, label="Hourly mean")

    # thresholds with distinct styles
    if np.isfinite(thr_hi):
        ax.axhline(thr_hi, color="k", linestyle="--", linewidth=1.2, label=f"Thr hi Q{quantile}")
    if np.isfinite(thr_near):
        ax.axhline(thr_near, color="tab:orange", linestyle=":", linewidth=1.2, label=f"Thr near Q{quantile}")
    if np.isfinite(thr_mean):
        ax.axhline(thr_mean, color="tab:green", linestyle="-.", linewidth=1.2, label=f"Thr mean Q{quantile}")

    # mark hi peak time
    ax.axvline(t_peak_hi, color="k", linestyle=":", linewidth=1.0, alpha=0.7, label="Hi peak time")

    # mark matched hourly peak for chosen method
    q_hr = q_hr_near if method == "near" else q_hr_mean
    t_match, q_match = pick_hourly_peak(q_hr, t_peak_hi, pd.Timedelta("1h"))
    if t_match is not None:
        ax.axvline(t_match, color=("tab:orange" if method == "near" else "tab:green"),
                   linestyle="--", linewidth=1.0, alpha=0.8,
                   label=f"Matched hourly peak ({method})")

    ax.set_title(f"{gauge} event hydrograph ({year}) Q={quantile} method={method}")
    ax.set_xlabel("time")
    ax.set_ylabel("Q (m3/s)")
    format_time_axis(ax)
    ax.legend(ncol=2)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2016)
    ap.add_argument("--n_success", type=int, default=40)  # bumped default
    ap.add_argument("--max_tries", type=int, default=400)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--nearest_tol", type=str, default="10min")
    ap.add_argument("--min_sep", type=str, default="12h")
    ap.add_argument("--search_window", type=str, default="1h")

    ap.add_argument("--attr_csv", type=str, default=r"C:\PhD\Python\neuralhydrology\US_data\iv_scan_results.csv")
    ap.add_argument("--camelsh_dir", type=str, default="data/raw/camelsh")
    ap.add_argument("--quantiles", type=str, default="0.95,0.98,0.99,0.995")

    ap.add_argument("--hydro_quantile", type=float, default=0.99)
    args = ap.parse_args()

    year = args.year
    quantiles = [float(x.strip()) for x in args.quantiles.split(",") if x.strip()]

    # pick hydro quantile once, early
    q_hydro = args.hydro_quantile
    if q_hydro not in quantiles:
        q_hydro = sorted(quantiles, key=lambda x: abs(x - q_hydro))[0]


    start = f"{year-1}-12-31"
    end = f"{year+1}-01-02"

    # ---- Run folder ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    qtag = "-".join([str(q) for q in quantiles])
    run_name = (
        f"flashdiag_{year}"
        f"_n{args.n_success}"
        f"_seed{args.seed}"
        f"_tol{args.nearest_tol}"
        f"_sep{args.min_sep}"
        f"_win{args.search_window}"
        f"_q{qtag}"
        f"_{ts}"
    )
    run_dir = Path("reports/runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    params = vars(args).copy()
    params.update({
        "year": year,
        "quantiles": quantiles,
        "start": start,
        "end": end,
        "run_name": run_name,
    })
    with open(run_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)


    # Use this everywhere instead of reports/figures
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)


    # Load area
    attr = pd.read_csv(args.attr_csv)
    if "site_id" not in attr.columns or "drainage_km2" not in attr.columns:
        raise SystemExit("attr_csv must include columns: site_id, drainage_km2")
    attr["gauge"] = attr["site_id"].astype(str).map(zfill8)
    attr = attr[["gauge", "drainage_km2"]].drop_duplicates("gauge")
    area_map = dict(zip(attr["gauge"], attr["drainage_km2"]))

    # Gauge list from CAMELSH filenames
    camelsh_dir = Path(args.camelsh_dir)
    files = sorted(camelsh_dir.glob("*_hourly.nc"))
    if not files:
        raise SystemExit(f"No *_hourly.nc files found in {camelsh_dir}")
    gauges = [zfill8(f.name.replace("_hourly.nc", "")) for f in files]

    random.seed(args.seed)
    random.shuffle(gauges)

    # keep only gauges with area
    gauges = [g for g in gauges if g in area_map]
    if not gauges:
        raise SystemExit("No gauges overlap between CAMELSH directory and attr_csv after zfill8.")


    # ----------------------------
    # Stage 1: collect basins + basin-wide slopes (quantile-independent)
    # ----------------------------
    basin_rows = []
    basin_cache = {}  # gauge -> (q_iv, q_hr_near, q_hr_mean)
    success = []

    for i, gauge in enumerate(gauges[:args.max_tries], 1):
        if len(success) >= args.n_success:
            break
        print(f"[{i}] {gauge} (success={len(success)})")

        try:
            q_iv = fetch_usgs_iv_discharge(gauge, start, end)
            q_hi = q_iv.loc[f"{year}-01-01":f"{year}-12-31 23:59:59"].dropna()
            if len(q_hi) < 2000:
                raise RuntimeError("Too little IV data in target year")

            cands = to_hourly_candidates(q_iv, nearest_tolerance=args.nearest_tol)
            q_hr_near = cands[f"nearest_{args.nearest_tol}"].loc[f"{year}-01-01":f"{year}-12-31 23:59:59"]
            q_hr_mean = cands["mean"].loc[f"{year}-01-01":f"{year}-12-31 23:59:59"]

            area = float(area_map[gauge])

            basin_rows.append({
                "gauge": gauge,
                "year": year,
                "drainage_km2": area,
                "slope_hi_native_max": max_slope_native(q_hi),
                "slope_hr_native_near_max": max_slope_native(q_hr_near),
                "slope_hr_native_mean_max": max_slope_native(q_hr_mean),
                "slope_hi_1h_max": max_slope_1h_from_hi(q_hi),
                "slope_hr_1h_near_max": max_slope_1h_from_hourly(q_hr_near),
                "slope_hr_1h_mean_max": max_slope_1h_from_hourly(q_hr_mean),
            })

            basin_cache[gauge] = (q_iv, q_hr_near, q_hr_mean)
            success.append(gauge)

        except Exception as e:
            print("  FAIL:", e)

    if not basin_rows:
        raise SystemExit("No successful basins.")

    df_basin = pd.DataFrame(basin_rows)
    df_basin.to_csv(out_dir / f"basin_slopes_{year}_n{len(success)}.csv", index=False)

    # Basin-wide slope scatter plots (once)
    plt.figure(figsize=(6, 5))
    plt.xscale("log"); plt.yscale("log")
    sc = plt.scatter(df_basin["slope_hi_native_max"], df_basin["slope_hr_native_near_max"], c=df_basin["drainage_km2"])
    plt.colorbar(sc, label="drainage_km2")
    finite = (df_basin["slope_hi_native_max"] > 0) & (df_basin["slope_hr_native_near_max"] > 0)
    if finite.any():
        mn = min(df_basin.loc[finite, "slope_hi_native_max"].min(), df_basin.loc[finite, "slope_hr_native_near_max"].min())
        mx = max(df_basin.loc[finite, "slope_hi_native_max"].max(), df_basin.loc[finite, "slope_hr_native_near_max"].max())
        plt.plot([mn, mx], [mn, mx])
    plt.xlabel("max dQ/dt (15-min native)")
    plt.ylabel("max dQ/dt (hourly nearest native)")
    plt.title(f"Native flashiness: nearest (n={len(df_basin)})")
    plt.savefig(out_dir / f"scatter_basin_slope_native_near_{year}.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.xscale("log"); plt.yscale("log")
    sc = plt.scatter(df_basin["slope_hi_1h_max"], df_basin["slope_hr_1h_near_max"], c=df_basin["drainage_km2"])
    plt.colorbar(sc, label="drainage_km2")
    finite = (df_basin["slope_hi_1h_max"] > 0) & (df_basin["slope_hr_1h_near_max"] > 0)
    if finite.any():
        mn = min(df_basin.loc[finite, "slope_hi_1h_max"].min(), df_basin.loc[finite, "slope_hr_1h_near_max"].min())
        mx = max(df_basin.loc[finite, "slope_hi_1h_max"].max(), df_basin.loc[finite, "slope_hr_1h_near_max"].max())
        plt.plot([mn, mx], [mn, mx])
    plt.xlabel("max ΔQ/Δ(1h) from 15-min")
    plt.ylabel("max ΔQ/Δ(1h) from hourly nearest")
    plt.title(f"Hour-scale slopes: nearest (n={len(df_basin)})")
    plt.savefig(out_dir / f"scatter_basin_slope_1h_near_{year}.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ----------------------------
    # Stage 2: event-level recompute per quantile
    # ----------------------------
    all_events = []
    for q in quantiles:
        print(f"\nComputing event tables for quantile {q} ...")
        for gauge in success:
            q_iv, q_hr_near, q_hr_mean = basin_cache[gauge]
            area = float(area_map[gauge])

            ev = event_table_for_quantile(
                gauge=gauge, year=year, area_km2=area,
                q_iv=q_iv, q_hr_near=q_hr_near, q_hr_mean=q_hr_mean,
                quantile=q, min_sep=args.min_sep, search_window=args.search_window
            )
            if len(ev):
                all_events.append(ev)

    if not all_events:
        raise SystemExit("No events detected across basins/quantiles.")

    df_events = pd.concat(all_events, ignore_index=True)
    df_events.to_csv(out_dir / f"events_recomputed_{year}_n{len(success)}.csv", index=False)


    # ----------------------------
    # Worst-event hydrographs (separate PNG per event)
    # ----------------------------
    event_fig_dir = out_dir / "event_hydrographs"
    event_fig_dir.mkdir(parents=True, exist_ok=True)

    # choose which quantile to diagnose visually (default: your hydro quantile)
    q_vis = q_hydro

    dvis = df_events[df_events["quantile"] == q_vis].copy()
    # keep only detected for each method when ranking
    d_near = dvis[dvis["detected_near"]].copy()
    d_mean = dvis[dvis["detected_mean"]].copy()

    def safe_tag(ts: pd.Timestamp) -> str:
        return pd.Timestamp(ts).strftime("%Y%m%dT%H%M%S")

    # 1) Most negative peak error (worst underestimation)
    worst_near_pe = d_near.sort_values("rel_peak_error_near").head(5)
    worst_mean_pe = d_mean.sort_values("rel_peak_error_mean").head(5)

    # 2) Largest absolute timing error
    d_near["abs_dt_near"] = d_near["dt_minutes_near"].abs()
    d_mean["abs_dt_mean"] = d_mean["dt_minutes_mean"].abs()
    worst_near_dt = d_near.sort_values("abs_dt_near", ascending=False).head(5)
    worst_mean_dt = d_mean.sort_values("abs_dt_mean", ascending=False).head(5)

    # 3) Extreme not preserved (False)
    miss_near = d_near[d_near["extreme_preserved_near"] == False].head(5)
    miss_mean = d_mean[d_mean["extreme_preserved_mean"] == False].head(5)

    # bundle all requested cases
    cases = []
    for _, r in worst_near_pe.iterrows():
        cases.append(("worst_peakerr", "near", r))
    for _, r in worst_mean_pe.iterrows():
        cases.append(("worst_peakerr", "mean", r))
    for _, r in worst_near_dt.iterrows():
        cases.append(("worst_timing", "near", r))
    for _, r in worst_mean_dt.iterrows():
        cases.append(("worst_timing", "mean", r))
    for _, r in miss_near.iterrows():
        cases.append(("not_preserved", "near", r))
    for _, r in miss_mean.iterrows():
        cases.append(("not_preserved", "mean", r))

    # de-duplicate by (gauge, method, event time, case_type)
    seen = set()
    cases2 = []
    for case_type, method, r in cases:
        key = (case_type, method, r["gauge"], r["t_peak_hi"])
        if key not in seen:
            cases2.append((case_type, method, r))
            seen.add(key)

    print(f"\nPlotting {len(cases2)} worst-event hydrographs at Q={q_vis} ...")

    for case_type, method, r in cases2:
        g = r["gauge"]
        t_hi = pd.Timestamp(r["t_peak_hi"])
        q_iv, q_hr_near, q_hr_mean = basin_cache[g]

        fname = f"{case_type}_{method}_{g}_{safe_tag(t_hi)}_Q{q_vis}.png"
        plot_event_hydrograph(
            gauge=g, year=year, quantile=q_vis, method=method,
            t_peak_hi=t_hi,
            q_iv=q_iv, q_hr_near=q_hr_near, q_hr_mean=q_hr_mean,
            outpath=event_fig_dir / fname,
            hours_before=12, hours_after=24,
        )


    # Basin summaries derived from event tables
    df_bsum = basin_summary_from_events(df_events)
    df_bsum.to_csv(out_dir / f"basin_summary_from_events_{year}_n{len(success)}.csv", index=False)

    # ----------------------------
    # Panel plots across quantiles (POT-dependent)
    # ----------------------------
    dfs_by_q_events = {q: df_events[df_events["quantile"] == q].copy() for q in sorted(df_events["quantile"].unique())}
    dfs_by_q_bsum = {q: df_bsum[df_bsum["quantile"] == q].copy() for q in sorted(df_bsum["quantile"].unique())}

    # Event-level scatter: peak_hi vs peak_hourly (nearest/mean)
    panel_scatter(
        dfs_by_q=dfs_by_q_events,
        xcol="q_peak_hi",
        ycol="q_peak_hr_near",
        ccol="drainage_km2",
        title="Event peaks: 15-min vs hourly nearest (colored by area)",
        xlabel="Q_peak 15-min (m3/s)",
        ylabel="Q_peak hourly nearest (m3/s)",
        outpath=out_dir / f"panel_event_peak_scatter_near_{year}.png",
        logxy=True,
    )

    panel_scatter(
        dfs_by_q=dfs_by_q_events,
        xcol="q_peak_hi",
        ycol="q_peak_hr_mean",
        ccol="drainage_km2",
        title="Event peaks: 15-min vs hourly mean (colored by area)",
        xlabel="Q_peak 15-min (m3/s)",
        ylabel="Q_peak hourly mean (m3/s)",
        outpath=out_dir / f"panel_event_peak_scatter_mean_{year}.png",
        logxy=True,
    )

    # Event-level CDFs: relative peak error
    panel_cdf(
        dfs_by_q=dfs_by_q_events,
        series_specs=[("nearest", "rel_peak_error_near"), ("mean", "rel_peak_error_mean")],
        xlabel="relative peak error (Q_hr - Q_hi)/Q_hi",
        title="Event-level CDF of relative peak error",
        outpath=out_dir / f"panel_cdf_event_rel_peak_error_{year}.png",
        xscale=None,
    )

    # Event-level CDFs: timing error (absolute minutes)
    # (use abs columns computed on the fly)
    tmp = {}
    for q, d in dfs_by_q_events.items():
        d = d.copy()
        d["abs_dt_near"] = np.abs(d["dt_minutes_near"])
        d["abs_dt_mean"] = np.abs(d["dt_minutes_mean"])
        tmp[q] = d
    panel_cdf(
        dfs_by_q=tmp,
        series_specs=[("nearest", "abs_dt_near"), ("mean", "abs_dt_mean")],
        xlabel="|timing error| (minutes)",
        title="Event-level CDF of absolute timing error",
        outpath=out_dir / f"panel_cdf_event_abs_timing_error_{year}.png",
        xscale=None,
    )

    # Basin-wise scatter (POT-dependent): extreme recall vs area
    # panel_scatter expects positive x/y; extreme recall is [0,1], so we do custom panel
    qs = sorted(dfs_by_q_bsum.keys())
    fig, axes = plt.subplots(1, len(qs), figsize=(5 * len(qs), 4), constrained_layout=True)
    for ax, q in zip(np.atleast_1d(axes), qs):
        d = dfs_by_q_bsum[q]
        ax.set_xscale("log")
        ax.scatter(d["drainage_km2"], d["extreme_recall_near"])
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Q={q}")
        ax.set_xlabel("drainage_km2")
        ax.set_ylabel("extreme recall (nearest)")
    fig.suptitle("Basin-wise extreme recall vs area (nearest)")
    fig.savefig(out_dir / f"panel_extreme_recall_vs_area_near_{year}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(qs), figsize=(5 * len(qs), 4), constrained_layout=True)
    for ax, q in zip(np.atleast_1d(axes), qs):
        d = dfs_by_q_bsum[q]
        ax.set_xscale("log")
        ax.scatter(d["drainage_km2"], d["extreme_recall_mean"])
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Q={q}")
        ax.set_xlabel("drainage_km2")
        ax.set_ylabel("extreme recall (mean)")
    fig.suptitle("Basin-wise extreme recall vs area (mean)")
    fig.savefig(out_dir / f"panel_extreme_recall_vs_area_mean_{year}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ----------------------------
    # Hydrograph examples (better ticks + thresholds)
    # ----------------------------


    # pick gauges for hydro examples based on basin-wide slopes
    dsel = df_basin.copy()
    dsel["slope_ratio_native_near"] = dsel["slope_hr_native_near_max"] / dsel["slope_hi_native_max"]
    dsel = dsel.replace([np.inf, -np.inf], np.nan).dropna(subset=["drainage_km2", "slope_ratio_native_near"])

    picks = []
    if len(dsel):
        picks.append(("smallest_area", dsel.sort_values("drainage_km2").iloc[0]["gauge"]))
        picks.append(("largest_area", dsel.sort_values("drainage_km2").iloc[-1]["gauge"]))
        picks.append(("lowest_slope_ratio", dsel.sort_values("slope_ratio_native_near").iloc[0]["gauge"]))
        picks.append(("highest_slope_ratio", dsel.sort_values("slope_ratio_native_near").iloc[-1]["gauge"]))

    # unique
    seen = set()
    picks2 = []
    for tag, g in picks:
        if g not in seen:
            picks2.append((tag, g))
            seen.add(g)

    print("\nHydrograph example gauges:", picks2)

    for tag, g in picks2:
        try:
            q_iv, q_hr_near, q_hr_mean = basin_cache[g]
            plot_hydrograph_with_thresholds(
                gauge=g, year=year, quantile=q_hydro,
                q_iv=q_iv, q_hr_near=q_hr_near, q_hr_mean=q_hr_mean,
                outpath=out_dir / f"hydro_{tag}_{g}_year{year}_q{q_hydro}.png",
                window_days=2
            )
        except Exception as e:
            print("  hydro FAIL", g, e)

    readme = textwrap.dedent(f"""
    # Flash-resolution diagnostics run

    **Run name:** {run_name}

    ## Goal
    Compare USGS IV high-frequency discharge to hourly representations (nearest-to-hour within tolerance, and hourly mean)
    to quantify how moving to hourly affects:
    - peak magnitude and timing of POT events detected in high-res series
    - whether events remain extreme under hourly quantile thresholds (extreme recall)
    - flashiness/steepness metrics (computed basin-wide, quantile-independent)

    ## Key methodological choices
    - POT events are detected on high-res IV using quantiles: {quantiles}
    - Minimum separation between peaks: {args.min_sep}
    - Hourly matching window around high-res peak: ±{args.search_window}
    - Hourly "nearest" created by selecting the observation nearest to each hourly stamp within {args.nearest_tol}
    - Hourly "mean" created by averaging within hourly bins

    ## Outputs
    ### Tables
    - `basin_slopes_{year}_n*.csv`:
    One row per basin. Basin-wide (quantile-independent) maxima of:
    - native max dQ/dt (hi-res, hourly nearest, hourly mean)
    - 1-hour-scale max ΔQ/Δ(1h) derived from hi-res and hourly series

    - `events_recomputed_{year}_n*.csv`:
    One row per POT event (detected on high-res). Includes matched hourly peaks for nearest/mean, timing errors,
    relative peak errors, and extreme preservation flags based on hourly quantile thresholds.

    - `basin_summary_from_events_{year}_n*.csv`:
    Basin-wise summaries per quantile (recall, extreme recall, median peak error, p90 abs timing error).

    ### Figures
    All figures are saved under `figures/` and are generated from the tables above. Panel plots use shared axes across quantiles
    to enable direct comparison.
    """).strip()

    with open(run_dir / "RUN_README.md", "w", encoding="utf-8") as f:
        f.write(readme + "\n")


    print(f"\nDone. Run saved under: {run_dir.resolve()}")
    print(f"Figures: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

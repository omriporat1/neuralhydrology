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


def _find_threshold_crossing_time(
    q_iv: pd.Series,
    thr: float,
    t_peak: pd.Timestamp,
    lookback: pd.Timedelta,
) -> pd.Timestamp:
    """
    Find the *first* time the IV series crosses thr on the rising limb.
    We search from (t_peak - lookback) .. t_peak.
    Fallback: t_peak.
    """
    s = q_iv.loc[t_peak - lookback : t_peak].dropna()
    if s.empty:
        return t_peak
    crossed = s[s >= thr]
    if crossed.empty:
        return t_peak
    return crossed.index[0]


def _add_threshold_lines(ax, thr_map, method="hi"):
    """
    thr_map: dict[quantile -> dict of thresholds], e.g.
      thr_map[q]["hi"], thr_map[q]["near"], thr_map[q]["mean"]

    method controls which threshold we draw (hi/near/mean).
    """
    # Different styles per quantile so they are clearly distinct
    style = {
        0.95:  dict(linestyle="--", linewidth=1.6),
        0.98:  dict(linestyle=":",  linewidth=2.0),
        0.99:  dict(linestyle="-.", linewidth=1.8),
        0.995: dict(linestyle=(0, (5, 2)), linewidth=1.8),
    }

    # Use consistent ordering
    for q in [0.95, 0.98, 0.99, 0.995]:
        if q not in thr_map:
            continue
        y = thr_map[q].get(method, None)
        if y is None or not np.isfinite(y):
            continue
        ax.axhline(y, label=f"Thr {method} Q={q}", **style[q])


def plot_random_crossing_hydrographs(
    events_df: pd.DataFrame,
    gauge: str,
    year: int,
    q_iv: pd.Series,
    q_hr_near: pd.Series,
    q_hr_mean: pd.Series,
    out_dir: str,
    n_per_q: int = 3,
    half_window_hours: int = 12,
    seed: int = 123,
):
    """
    Makes 3 hydrographs per quantile, centered on the *threshold crossing* time,
    and includes all thresholds (Q=0.95/0.98/0.99/0.995) on each plot.

    events_df must include columns:
      gauge, year, quantile, thr_hi, thr_hr_near, thr_hr_mean, t_peak_hi
    """
    rng = np.random.default_rng(seed)
    half_win = pd.Timedelta(hours=half_window_hours)

    ev_g = events_df[(events_df["gauge"] == gauge) & (events_df["year"] == year)].copy()
    if ev_g.empty:
        return

    # Build threshold map for this gauge-year so every plot shows the "range"
    thr_map = {}
    for q, sub in ev_g.groupby("quantile"):
        # should be constant per gauge-year-quantile, but take first
        row0 = sub.iloc[0]
        thr_map[float(q)] = {
            "hi": float(row0["thr_hi"]),
            "near": float(row0["thr_hr_near"]),
            "mean": float(row0["thr_hr_mean"]),
        }

    # Ensure timestamp column is Timestamp
    ev_g["t_peak_hi"] = pd.to_datetime(ev_g["t_peak_hi"], utc=True, errors="coerce")

    for q in [0.95, 0.98, 0.99, 0.995]:
        sub = ev_g[ev_g["quantile"].astype(float) == float(q)].copy()
        if sub.empty:
            continue

        # Keep only events that actually cross the high-res threshold
        # (usually true, but safe)
        sub = sub[np.isfinite(sub["thr_hi"]) & np.isfinite(sub["q_peak_hi"])]
        sub = sub[sub["q_peak_hi"] >= sub["thr_hi"]]
        if sub.empty:
            continue

        take = min(n_per_q, len(sub))
        picks = sub.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1)))

        for i, row in enumerate(picks.itertuples(index=False), start=1):
            t_peak = getattr(row, "t_peak_hi")
            thr_hi = float(getattr(row, "thr_hi"))

            # Center around crossing time (not peak time)
            t0 = _find_threshold_crossing_time(
                q_iv=q_iv,
                thr=thr_hi,
                t_peak=t_peak,
                lookback=half_win,  # search up to 12h before peak
            )

            tL, tR = t0 - half_win, t0 + half_win

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(q_iv.loc[tL:tR], label="USGS IV (hi-res)")
            ax.plot(q_hr_near.loc[tL:tR], label="Hourly nearest")
            ax.plot(q_hr_mean.loc[tL:tR], label="Hourly mean (centered)")

            # Vertical line at crossing time
            ax.axvline(t0, linestyle="--", linewidth=1.5, label="Crossing time")

            # Show ALL thresholds (range) — high-res thresholds are the most interpretable here
            _add_threshold_lines(ax, thr_map, method="hi")

            ax.set_title(f"{gauge} random crossing event ({year}) Q={q}  [{i}/{take}]")
            ax.set_xlabel("time")
            ax.set_ylabel("Q (m3/s)")
            ax.grid(True, alpha=0.25)

            # Legend outside so it never covers the hydrograph
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
            fig.tight_layout(rect=(0, 0, 0.82, 1))

            out_dir_p = Path(out_dir)
            out_dir_p.mkdir(parents=True, exist_ok=True)

            # Keep filename SHORT to avoid Windows MAX_PATH issues
            fname = f"rnd_{gauge}_q{q}_{i}.png"
            out_path = out_dir_p / fname

            try:
                fig.savefig(out_path, dpi=200, bbox_inches="tight")
            finally:
                plt.close(fig)



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

def event_bounded_window_from_threshold(
    q_hi: pd.Series,
    t_peak: pd.Timestamp,
    thr: float,
    max_expand: pd.Timedelta,
    pad: pd.Timedelta = pd.Timedelta("2h"),
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Define event window as the contiguous period around t_peak where q_hi >= thr,
    limited to +/- max_expand, then padded.
    """
    if not np.isfinite(thr) or t_peak not in q_hi.index:
        # fallback: just use +/- max_expand
        return (t_peak - max_expand, t_peak + max_expand)

    left_lim = t_peak - max_expand
    right_lim = t_peak + max_expand

    w = q_hi.loc[left_lim:right_lim].dropna()
    if w.empty:
        return (t_peak - max_expand, t_peak + max_expand)

    # boolean mask of "above threshold"
    above = (w >= thr)
    if not above.any():
        return (t_peak - max_expand, t_peak + max_expand)

    # walk left/right from the peak to find contiguous above-threshold block
    # find nearest index position to peak in w
    i0 = w.index.get_indexer([t_peak], method="nearest")[0]

    iL = i0
    while iL > 0 and above.iloc[iL - 1]:
        iL -= 1

    iR = i0
    while iR < len(w) - 1 and above.iloc[iR + 1]:
        iR += 1

    t0 = w.index[iL] - pad
    t1 = w.index[iR] + pad
    return (t0, t1)

def event_table_for_quantile(
    gauge: str,
    analysis_start: str,
    analysis_end: str,
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
    q_hi = q_iv.loc[analysis_start:analysis_end].dropna()

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
            "year": pd.Timestamp(t0).year,   # optional metadata only
            "start_year": pd.Timestamp(analysis_start).year,
            "end_year": pd.Timestamp(analysis_end).year,
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

    for ax, q in zip(np.atleast_1d(axes), qs):
        d = dfs_by_q[q].copy()
        x = d[xcol].to_numpy()
        y = d[ycol].to_numpy()
        c = d[ccol].to_numpy()

        valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        x_valid = x[valid]
        y_valid = y[valid]
        c_valid = c[valid]

        if logxy:
            ax.set_xscale("log")
            ax.set_yscale("log")

        mappable = ax.scatter(x_valid, y_valid, c=c_valid, vmin=vmin, vmax=vmax, s=25, edgecolors="none", alpha=0.9)

        # 1:1 line
        mn = min(xmin, ymin)
        mx = max(xmax, ymax)
        ax.plot([mn, mx], [mn, mx], color="tab:blue", linewidth=1)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"Q={q}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # compute simple stats on linear (not log) values if we have data
        if len(x_valid):
            # RMSE (absolute on original units)
            dif = y_valid - x_valid
            rmse = float(np.sqrt(np.nanmean(dif ** 2)))
            # relative bias = mean((y-x)/x)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_err = (y_valid - x_valid) / x_valid
            bias = float(np.nanmean(rel_err))
            # Pearson r
            try:
                r = float(np.corrcoef(x_valid, y_valid)[0, 1])
            except Exception:
                r = np.nan

            txt = f"N={len(x_valid)}\nRMSE={rmse:.3g}\nrel_bias={bias:.3g}\nr={r:.3f}"
            ax.text(0.98, 0.02, txt, ha="right", va="bottom", transform=ax.transAxes,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=4), fontsize=9)

    fig.suptitle(title)

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=np.atleast_1d(axes), shrink=0.9)
        cbar.set_label("drainage_km2")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def panel_cdf(
    dfs_by_q: dict[float, pd.DataFrame],
    series_specs: list[tuple[str, str]],
    xlabel: str,
    title: str,
    outpath: Path,
    xscale: str | None = None,
):
    qs = sorted(dfs_by_q.keys())
    k = len(qs)
    fig, axes = plt.subplots(1, k, figsize=(5 * k, 4), constrained_layout=True)

    # ---- compute global x-limits safely ----
    allvals = []
    for q in qs:
        d = dfs_by_q[q]
        for _, col in series_specs:
            if col in d.columns:
                v = d[col].to_numpy()
                v = v[np.isfinite(v)]
                if len(v):
                    allvals.append(v)

    if not allvals:
        # still produce a figure so your pipeline “has outputs”
        for ax, q in zip(np.atleast_1d(axes), qs):
            ax.text(0.5, 0.5, "No finite data", ha="center", va="center")
            ax.set_title(f"Q={q}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Empirical CDF")
        fig.suptitle(title)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    allvals = np.concatenate(allvals)
    xmin, xmax = np.nanmin(allvals), np.nanmax(allvals)
    if np.isclose(xmin, xmax):
        xmin -= 1.0
        xmax += 1.0

    for ax, q in zip(np.atleast_1d(axes), qs):
        d = dfs_by_q[q]
        for label, col in series_specs:
            if col not in d.columns:
                continue
            xs = d[col].to_numpy()
            xs = xs[np.isfinite(xs)]
            if len(xs) == 0:
                continue
            xs = np.sort(xs)
            ys = (np.arange(len(xs)) + 1) / len(xs)
            ax.plot(xs, ys, label=label)

        # compute x-limits PER PANEL (per q)
        panel_vals = []
        for _, col in series_specs:
            v = d[col].to_numpy()
            v = v[np.isfinite(v)]
            if len(v):
                panel_vals.append(v)

        if panel_vals:
            vv = np.concatenate(panel_vals)
            xmin, xmax = np.quantile(vv, 0.01), np.quantile(vv, 0.99)
            if xmin == xmax:
                xmin, xmax = float(np.min(vv)), float(np.max(vv))
            ax.set_xlim(xmin, xmax)

        N = int(np.sum(np.isfinite(d[series_specs[0][1]].to_numpy())))  # or better: union across cols
        ax.set_title(f"Q={q} (N≈{N})")


        if xscale:
            ax.set_xscale(xscale)
        ax.set_title(f"Q={q}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Empirical CDF")
        # Put legend outside the axes, on the right
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )
        fig.tight_layout(rect=(0, 0, 0.82, 1))  # leave room on the right for legend

    fig.suptitle(title, y=1.02)          # put suptitle a bit above
    fig.subplots_adjust(top=0.86)        # reserve headroom for it    outpath.parent.mkdir(parents=True, exist_ok=True)
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
    window_hours: float | None = None,
):
    """
    Plot hydrograph around representative peak.
    If window_hours is provided, use ±window_hours around the hi-res peak.
    If not provided, fall back to ±48h.
    """
    q_hi_year = q_iv.loc[f"{year}-01-01":f"{year}-12-31 23:59:59"].dropna()
    if q_hi_year.empty:
        return

    # choose a representative event time: annual peak (simple and persuasive)
    t_peak = q_hi_year.idxmax()

    if window_hours is None:
        window_hours = 48.0
    win_td = pd.Timedelta(hours=float(window_hours))
    t0 = t_peak - win_td
    t1 = t_peak + win_td

    hi = q_iv.loc[t0:t1]
    hn = q_hr_near.loc[t0:t1]
    hm = q_hr_mean.loc[t0:t1]

    thr_hi = float(q_hi_year.quantile(quantile))
    thr_near = float(q_hr_near.dropna().quantile(quantile)) if len(q_hr_near.dropna()) else np.nan
    thr_mean = float(q_hr_mean.dropna().quantile(quantile)) if len(q_hr_mean.dropna()) else np.nan

    fig, ax = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)
    ax.plot(hi.index, hi.values, label="USGS IV (hi-res)", linewidth=1)
    ax.plot(hn.index, hn.values, label="Hourly nearest", linewidth=1)
    ax.plot(hm.index, hm.values, label="Hourly mean (centered)", linewidth=1)

    # thresholds: draw with distinct linestyle & color; label includes quantile
    ax.axhline(thr_hi, color="black", linestyle="--", linewidth=1, label=f"Thr hi Q{quantile}")
    if np.isfinite(thr_near):
        ax.axhline(thr_near, color="tab:orange", linestyle=":", linewidth=1, label=f"Thr near Q{quantile}")
    if np.isfinite(thr_mean):
        ax.axhline(thr_mean, color="tab:green", linestyle="-.", linewidth=1, label=f"Thr mean Q{quantile}")

    # try to mark the hi peak + matched hourly peaks (if they exist)
    try:
        t_peak_hi = hi.idxmax()
        ax.axvline(t_peak_hi, color="k", linestyle=":", linewidth=1, label="Hi peak time")
    except Exception:
        t_peak_hi = None

    # find matched hourly peaks inside the plot range (use pick_hourly_peak with same window)
    try:
        # convert win_td to pandas Timedelta for pick_hourly_peak call
        matched_t_near, matched_q_near = pick_hourly_peak(q_hr_near, t_peak, win_td)
        if matched_t_near is not None:
            ax.axvline(matched_t_near, color="tab:orange", linestyle="--", linewidth=1, label="Matched hourly peak (near)")
    except Exception:
        matched_t_near = None

    try:
        matched_t_mean, matched_q_mean = pick_hourly_peak(q_hr_mean, t_peak, win_td)
        if matched_t_mean is not None:
            ax.axvline(matched_t_mean, color="tab:green", linestyle="--", linewidth=1, label="Matched hourly peak (mean)")
    except Exception:
        matched_t_mean = None

    ax.set_title(f"{gauge} window around annual peak ({year}) Q={quantile}")
    ax.set_xlabel("time")
    ax.set_ylabel("Q (m3/s)")
    format_time_axis(ax)

    handles, labels = ax.get_legend_handles_labels()

    # Put legend below the axes (outside), 2–3 columns usually looks good
    ax.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        fontsize=9,
        frameon=True,
    )

    # Give room at bottom for legend
    fig.subplots_adjust(bottom=0.28)


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
    use_event_bounded_window: bool = True,
    max_expand_hours: int = 12,

):
    # thresholds computed on full-year series (consistent with event table)
    q_hi_year = q_iv.loc[f"{year}-01-01":f"{year}-12-31 23:59:59"].dropna()
    thr_hi = float(q_hi_year.quantile(quantile)) if len(q_hi_year) else np.nan
    thr_near = float(q_hr_near.dropna().quantile(quantile)) if len(q_hr_near.dropna()) else np.nan
    thr_mean = float(q_hr_mean.dropna().quantile(quantile)) if len(q_hr_mean.dropna()) else np.nan

    if use_event_bounded_window:
        q_hi = q_iv.loc[f"{year}-01-01":f"{year}-12-31 23:59:59"].dropna()
        t0, t1 = event_bounded_window_from_threshold(
            q_hi=q_hi,
            t_peak=t_peak_hi,
            thr=thr_hi,
            max_expand=pd.Timedelta(hours=max_expand_hours),
            pad=pd.Timedelta("2h"),
        )
    else:
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

    handles, labels = ax.get_legend_handles_labels()

    # Put legend below the axes (outside), 2–3 columns usually looks good
    ax.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        fontsize=9,
        frameon=True,
    )

    # Give room at bottom for legend
    fig.subplots_adjust(bottom=0.28)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=None,
                    help="Single year run (ignored if --year_start/--year_end are provided)")
    ap.add_argument("--start_year", type=int, default=None)
    ap.add_argument("--end_year", type=int, default=None)
    ap.add_argument("--n_success", type=int, default=40)  # bumped default
    ap.add_argument("--max_tries", type=int, default=400)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--nearest_tol", type=str, default="10min")
    ap.add_argument("--min_sep", type=str, default="12h")

    ap.add_argument("--attr_csv", type=str, default=r"C:\PhD\Python\neuralhydrology\US_data\iv_scan_results.csv")
    ap.add_argument("--camelsh_dir", type=str, default="data/raw/camelsh")
    ap.add_argument("--quantiles", type=str, default="0.95,0.98,0.99,0.995")

    ap.add_argument("--hydro_quantile", type=float, default=0.99)
    ap.add_argument("--search_windows", type=str, default="")
    ap.add_argument("--search_window", type=str, default="6h",
                help="search window around hi-res peak for matching hourly peak (e.g. '1h','6h')")
    ap.add_argument("--max_area_km2", type=float, default=1000.0,
                help="Keep only basins with drainage_km2 <= this value")
    ap.add_argument(
        "--no_hydro",
        action="store_true",
        help="Disable all hydrograph plotting (worst-event + random-crossing)."
    )



    args = ap.parse_args()

    start_year = args.start_year
    end_year = args.end_year
    if end_year < start_year:
        raise SystemExit("end_year must be >= start_year")

    analysis_start = f"{start_year}-01-01"
    analysis_end   = f"{end_year}-12-31 23:59:59"

    # fetch buffer (optional but helpful near boundaries)
    fetch_start = (pd.Timestamp(analysis_start) - pd.Timedelta("1D")).strftime("%Y-%m-%d")
    fetch_end   = (pd.Timestamp(analysis_end) + pd.Timedelta("1D")).strftime("%Y-%m-%d")

    year_tag = f"{start_year}-{end_year}"

    quantiles = [float(x.strip()) for x in args.quantiles.split(",") if x.strip()]

    # pick hydro quantile once, early
    q_hydro = args.hydro_quantile
    if q_hydro not in quantiles:
        q_hydro = sorted(quantiles, key=lambda x: abs(x - q_hydro))[0]



    # event-bounded hydrograph window: use half the min_sep (so min_sep=12h -> ±6h)
    try:
        min_sep_td = pd.Timedelta(args.min_sep)
        hydro_win_hours = float(min_sep_td / 2) / 3600.0 if isinstance(min_sep_td, pd.Timedelta) else (12.0 / 2)
        # pd.Timedelta division may already yield Timedelta; simpler:
        hydro_win_hours = min_sep_td.total_seconds() / 3600.0 / 2.0
    except Exception:
        # fallback to 2 days (as previously)
        hydro_win_hours = 48.0

    # ---- Run folder ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    qtag = "-".join([str(q) for q in quantiles])
    run_name = (
        f"{start_year}-{end_year}"
        f"_n{args.n_success}"
        f"_sep{args.min_sep}"
        f"_win{args.search_window}"
        f"_q{qtag}"
        f"_{ts}"
    )
    run_dir = Path("reports/runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    params = vars(args).copy()
    params.update({
        "start_year": start_year,
        "end_year": end_year,
        "analysis_start": analysis_start,
        "analysis_end": analysis_end,
        "fetch_start": fetch_start,
        "fetch_end": fetch_end,
        "year_tag": year_tag,
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

    # filter by drainage area
    gauges = [g for g in gauges if float(area_map[g]) <= args.max_area_km2]
    if not gauges:
        raise SystemExit(f"No gauges left after max_area_km2={args.max_area_km2} filter.")

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
            q_iv = fetch_usgs_iv_discharge(gauge, fetch_start, fetch_end)
            q_hi = q_iv.loc[analysis_start:analysis_end].dropna()
            if len(q_hi) < 2000:
                raise RuntimeError("Too little IV data in target year")

            cands = to_hourly_candidates(q_iv, nearest_tolerance=args.nearest_tol)
            q_hr_near = cands[f"nearest_{args.nearest_tol}"].loc[analysis_start:analysis_end]
            q_hr_mean = cands["mean"].loc[analysis_start:analysis_end]

            area = float(area_map[gauge])

            basin_rows.append({
                "gauge": gauge,
                "start_year": start_year,
                "end_year": end_year,
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
    df_basin.to_csv(out_dir / f"basin_slopes_{start_year}-{end_year}_n{len(success)}.csv", index=False)

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
    plt.savefig(out_dir / f"scatter_basin_slope_native_near_{year_tag}.png", dpi=200, bbox_inches="tight")
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
    plt.savefig(out_dir / f"scatter_basin_slope_1h_near_{year_tag}.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ----------------------------
    # Stage 2: event-level recompute per quantile
    # ----------------------------
    all_events = []


    windows = []
    if args.search_windows.strip():
        windows = [w.strip() for w in args.search_windows.split(",") if w.strip()]
    else:
        windows = [args.search_window]

    all_events = []
    for win_str in windows:
        print(f"\n=== Matching window = ±{win_str} ===")
        for q in quantiles:
            print(f"Computing event tables for quantile {q} ...")
            for gauge in success:
                q_iv, q_hr_near, q_hr_mean = basin_cache[gauge]
                area = float(area_map[gauge])

                ev = event_table_for_quantile(
                    gauge=gauge,
                    analysis_start=analysis_start,
                    analysis_end=analysis_end,
                    area_km2=area,
                    q_iv=q_iv,
                    q_hr_near=q_hr_near,
                    q_hr_mean=q_hr_mean,
                    quantile=q,
                    min_sep=args.min_sep,
                    search_window=win_str,
                )

                if len(ev):
                    ev = ev.copy()
                    ev["match_window"] = win_str
                    all_events.append(ev)

    if not all_events:
        raise SystemExit("No events detected across basins/quantiles.")
    df_events = pd.concat(all_events, ignore_index=True)
    df_events.to_csv(out_dir / f"events_recomputed_{start_year}-{end_year}_n{len(success)}_wins.csv", index=False)


    if not all_events:
        raise SystemExit("No events detected across basins/quantiles.")

    df_events = pd.concat(all_events, ignore_index=True)
    df_events.to_csv(out_dir / f"events_recomputed_{start_year}-{end_year}_n{len(success)}.csv", index=False)

    if not args.no_hydro:

        # ----------------------------
        # Random crossing hydrographs: 3 per threshold (±12h around crossing)
        # ----------------------------
        rand_dir = out_dir / "random_crossing_hydrographs"
        rand_dir.mkdir(parents=True, exist_ok=True)

        for g in success:
            try:
                q_iv, q_hr_near, q_hr_mean = basin_cache[g]

                plot_random_crossing_hydrographs(
                    events_df=df_events,
                    gauge=g,
                    year=year,
                    q_iv=q_iv,
                    q_hr_near=q_hr_near,
                    q_hr_mean=q_hr_mean,
                    out_dir=str(rand_dir),
                    n_per_q=3,
                    half_window_hours=12,
                    seed=args.seed,
                )
            except Exception as e:
                print("  random-cross hydro FAIL", g, e)



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
                gauge=g, year=year_tag, quantile=q_vis, method=method,
                t_peak_hi=t_hi,
                q_iv=q_iv, q_hr_near=q_hr_near, q_hr_mean=q_hr_mean,
                outpath=event_fig_dir / fname,
                hours_before=12, hours_after=24,
            )


    # Basin summaries derived from event tables
    df_bsum = basin_summary_from_events(df_events)
    df_bsum.to_csv(out_dir / f"basin_summary_from_events_{year_tag}_n{len(success)}.csv", index=False)

    # ----------------------------
    # Panel plots across quantiles (POT-dependent)
    # ----------------------------

    for win_str in windows:
        dfw = df_events[df_events["match_window"] == win_str].copy()
        dfs_by_q_events = {q: dfw[dfw["quantile"] == q].copy()
                        for q in sorted(dfw["quantile"].unique())}

        panel_cdf(
            dfs_by_q=dfs_by_q_events,
            series_specs=[("nearest", "rel_peak_error_near"), ("mean", "rel_peak_error_mean")],
            xlabel="relative peak error (Q_hr - Q_hi)/Q_hi",
            title=f"Event-level CDF of relative peak error (±{win_str})",
            outpath=out_dir / f"panel_cdf_event_rel_peak_error_{year_tag}_win{win_str}.png",
            xscale=None,
        )

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
            title=f"Event-level CDF of absolute timing error (±{win_str})",
            outpath=out_dir / f"panel_cdf_event_abs_timing_error_{year_tag}_win{win_str}.png",
            xscale=None,
        )


    
    dfs_by_q_events = {q: df_events[df_events["quantile"] == q].copy() for q in sorted(df_events["quantile"].unique())}
    dfs_by_q_bsum = {q: df_bsum[df_bsum["quantile"] == q].copy() for q in sorted(df_bsum["quantile"].unique())}


    # ----------------------------
    # Aggregate statistics per quantile (event-level)
    # ----------------------------
    stats_rows = []

    for q, d in dfs_by_q_events.items():
        x = d["q_peak_hi"].to_numpy()

        # ---- nearest ----
        y_near = d["q_peak_hr_near"].to_numpy()
        valid = np.isfinite(x) & np.isfinite(y_near) & (x > 0)
        if valid.sum() > 0:
            xv = x[valid]
            yv = y_near[valid]
            rmse = float(np.sqrt(np.mean((yv - xv)**2)))
            rel_bias = float(np.mean((yv - xv) / xv))
            r = float(np.corrcoef(xv, yv)[0, 1])

            stats_rows.append({
                "quantile": q,
                "method": "nearest",
                "N": len(xv),
                "rmse": rmse,
                "rel_bias": rel_bias,
                "pearson_r": r
            })

        # ---- mean ----
        y_mean = d["q_peak_hr_mean"].to_numpy()
        valid = np.isfinite(x) & np.isfinite(y_mean) & (x > 0)
        if valid.sum() > 0:
            xv = x[valid]
            yv = y_mean[valid]
            rmse = float(np.sqrt(np.mean((yv - xv)**2)))
            rel_bias = float(np.mean((yv - xv) / xv))
            r = float(np.corrcoef(xv, yv)[0, 1])

            stats_rows.append({
                "quantile": q,
                "method": "mean",
                "N": len(xv),
                "rmse": rmse,
                "rel_bias": rel_bias,
                "pearson_r": r
            })

    # Save table
    pd.DataFrame(stats_rows).to_csv(
        out_dir / f"event_peak_stats_{year_tag}_n{len(success)}.csv",
        index=False
    )

    # Event-level scatter: peak_hi vs peak_hourly (nearest/mean)
    panel_scatter(
        dfs_by_q=dfs_by_q_events,
        xcol="q_peak_hi",
        ycol="q_peak_hr_near",
        ccol="drainage_km2",
        title="Event peaks: 15-min vs hourly nearest (colored by area)",
        xlabel="Q_peak 15-min (m3/s)",
        ylabel="Q_peak hourly nearest (m3/s)",
        outpath=out_dir / f"panel_event_peak_scatter_near_{year_tag}.png",
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
        outpath=out_dir / f"panel_event_peak_scatter_mean_{year_tag}.png",
        logxy=True,
    )

    # Event-level CDFs: relative peak error
    panel_cdf(
        dfs_by_q=dfs_by_q_events,
        series_specs=[("nearest", "rel_peak_error_near"), ("mean", "rel_peak_error_mean")],
        xlabel="relative peak error (Q_hr - Q_hi)/Q_hi",
        title="Event-level CDF of relative peak error",
        outpath=out_dir / f"panel_cdf_event_rel_peak_error_{year_tag}.png",
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
        outpath=out_dir / f"panel_cdf_event_abs_timing_error_{year_tag}.png",
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
    fig.savefig(out_dir / f"panel_extreme_recall_vs_area_near_{start_year}-{end_year}.png", dpi=200, bbox_inches="tight")
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
    fig.savefig(out_dir / f"panel_extreme_recall_vs_area_mean_{start_year}-{end_year}.png", dpi=200, bbox_inches="tight")
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
            outpath=out_dir / f"hydro_{tag}_{g}_year{start_year}-{end_year}_q{q_hydro}.png",
            window_hours=hydro_win_hours
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
    - `basin_slopes_{year_tag}_n*.csv`:
    One row per basin. Basin-wide (quantile-independent) maxima of:
    - native max dQ/dt (hi-res, hourly nearest, hourly mean)
    - 1-hour-scale max ΔQ/Δ(1h) derived from hi-res and hourly series

    - `events_recomputed_{year_tag}_n*.csv`:
    One row per POT event (detected on high-res). Includes matched hourly peaks for nearest/mean, timing errors,
    relative peak errors, and extreme preservation flags based on hourly quantile thresholds.

    - `basin_summary_from_events_{year_tag}_n*.csv`:
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

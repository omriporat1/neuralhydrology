from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ----------------------------
# Small utilities
# ----------------------------
def zfill8(x: str) -> str:
    return str(x).strip().zfill(8)


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y[m] - x[m]) ** 2)))


def mae(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y[m] - x[m])))


def bias(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() == 0:
        return np.nan
    return float(np.mean(y[m] - x[m]))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """No scipy dependency. Spearman = Pearson of ranks."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    xr = pd.Series(x[m]).rank().to_numpy()
    yr = pd.Series(y[m]).rank().to_numpy()
    return float(np.corrcoef(xr, yr)[0, 1])


def format_time_axis(ax):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def area_bin(area_km2: float) -> str:
    # simple, interpretable bins (edit anytime)
    if area_km2 < 100:
        return "<100"
    if area_km2 < 1000:
        return "100-1000"
    return ">=1000"


# ----------------------------
# Slopes (flashiness)
# ----------------------------
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
    """Max 1-hour slope computed from hi-res series: (Q(t)-Q(t-1h))/3600."""
    s = q_hi.dropna().sort_index()
    if len(s) < 10:
        return np.nan

    diffs = s.index.to_series().diff().dropna()
    dt_mode = diffs.mode()
    dt_min = float(dt_mode.iloc[0].total_seconds() / 60.0) if len(dt_mode) else np.nan

    if np.isfinite(dt_min) and abs(dt_min - 15) < 1e-6:
        dq1h = (s - s.shift(4)).dropna()
        slopes = dq1h.values / 3600.0
        return float(np.nanmax(slopes)) if len(slopes) else np.nan

    idx = s.index
    s_prev = s.reindex(idx - pd.Timedelta("1h"))
    dq1h = (s.values - s_prev.values)
    slopes = dq1h / 3600.0
    return float(np.nanmax(slopes)) if np.isfinite(slopes).any() else np.nan


def max_slope_1h_from_hourly(q_hr: pd.Series) -> float:
    s = q_hr.dropna().sort_index()
    if len(s) < 3:
        return np.nan
    slopes = s.diff().dropna().values / 3600.0
    return float(np.nanmax(slopes)) if len(slopes) else np.nan


# ----------------------------
# Event matching
# ----------------------------
def pick_hourly_peak(
    q_hr: pd.Series,
    t0: pd.Timestamp,
    win: pd.Timedelta,
) -> tuple[pd.Timestamp | None, float]:
    hr_win = q_hr.loc[t0 - win : t0 + win].dropna()
    if hr_win.empty:
        return None, np.nan

    qmax = hr_win.max()

    # all timestamps attaining the maximum (handles plateaus / ties)
    t_candidates = hr_win.index[hr_win.values == qmax]
    if len(t_candidates) == 1:
        t1 = t_candidates[0]
        return t1, float(qmax)

    # break ties by choosing the time closest to the hi-res peak time t0
    # (if still tied, keep the earlier one deterministically)
    dt = np.abs((t_candidates - t0).astype("timedelta64[s]").astype(np.int64))
    best = np.where(dt == dt.min())[0][0]
    t1 = t_candidates[best]
    return t1, float(qmax)

def make_event_table(
    gauge: str,
    year: int,
    area_km2: float,
    q_iv_year: pd.Series,
    q_hr_near_year: pd.Series,
    q_hr_mean_year: pd.Series,
    detect_pot_events_fn,
    quantile: float,
    min_sep: str,
    search_window: str,
) -> pd.DataFrame:
    """One row per hi-res POT event; attach matched hourly peaks + errors + thresholds."""
    q_hi = q_iv_year.dropna().sort_index()
    if q_hi.empty:
        return pd.DataFrame()

    events = detect_pot_events_fn(q_hi, quantile=quantile, min_separation=min_sep)
    if events.empty:
        return pd.DataFrame()

    win = pd.Timedelta(search_window)

    thr_hi = float(q_hi.quantile(quantile))
    thr_near = float(q_hr_near_year.dropna().quantile(quantile)) if len(q_hr_near_year.dropna()) else np.nan
    thr_mean = float(q_hr_mean_year.dropna().quantile(quantile)) if len(q_hr_mean_year.dropna()) else np.nan

    rows = []
    for _, ev in events.iterrows():
        t0 = pd.Timestamp(ev["peak_time"])
        q0 = float(ev["peak_q"])

        t1n, q1n = pick_hourly_peak(q_hr_near_year, t0, win)
        t1m, q1m = pick_hourly_peak(q_hr_mean_year, t0, win)

        rows.append({
            "gauge": gauge,
            "year": year,
            "drainage_km2": area_km2,
            "area_bin": area_bin(area_km2),
            "quantile": quantile,

            "thr_hi": thr_hi,
            "thr_hr_near": thr_near,
            "thr_hr_mean": thr_mean,

            "t_peak_hi": t0,
            "q_peak_hi": q0,

            "t_peak_hr_near": t1n,
            "q_peak_hr_near": q1n,
            "detected_near": bool(t1n is not None),
            "dt_minutes_near": (t1n - t0).total_seconds() / 60.0 if t1n is not None else np.nan,
            "rel_peak_error_near": (q1n - q0) / q0 if (np.isfinite(q1n) and q0 > 0) else np.nan,
            "extreme_preserved_near": bool(q1n > thr_near) if (np.isfinite(q1n) and np.isfinite(thr_near)) else np.nan,

            "t_peak_hr_mean": t1m,
            "q_peak_hr_mean": q1m,
            "detected_mean": bool(t1m is not None),
            "dt_minutes_mean": (t1m - t0).total_seconds() / 60.0 if t1m is not None else np.nan,
            "rel_peak_error_mean": (q1m - q0) / q0 if (np.isfinite(q1m) and q0 > 0) else np.nan,
            "extreme_preserved_mean": bool(q1m > thr_mean) if (np.isfinite(q1m) and np.isfinite(thr_mean)) else np.nan,
        })

    return pd.DataFrame(rows)


def basin_summary_from_events(df_events: pd.DataFrame) -> pd.DataFrame:
    if df_events.empty:
        return pd.DataFrame()

    rows = []
    for (g, q), d in df_events.groupby(["gauge", "quantile"]):
        base = {
            "gauge": g,
            "quantile": float(q),
            "drainage_km2": float(d["drainage_km2"].iloc[0]),
            "area_bin": str(d["area_bin"].iloc[0]),
            "n_events": int(len(d)),
        }

        dn = d[d["detected_near"]]
        dm = d[d["detected_mean"]]

        base["recall_near"] = float(d["detected_near"].mean())
        base["recall_mean"] = float(d["detected_mean"].mean())

        base["extreme_recall_near"] = float(dn["extreme_preserved_near"].dropna().astype(bool).mean()) if len(dn) else np.nan
        base["extreme_recall_mean"] = float(dm["extreme_preserved_mean"].dropna().astype(bool).mean()) if len(dm) else np.nan

        base["med_rel_err_near"] = float(dn["rel_peak_error_near"].median()) if len(dn) else np.nan
        base["med_rel_err_mean"] = float(dm["rel_peak_error_mean"].median()) if len(dm) else np.nan

        base["p90_abs_dt_min_near"] = float(dn["dt_minutes_near"].abs().quantile(0.9)) if len(dn) else np.nan
        base["p90_abs_dt_min_mean"] = float(dm["dt_minutes_mean"].abs().quantile(0.9)) if len(dm) else np.nan

        rows.append(base)

    return pd.DataFrame(rows)


# ----------------------------
# Plotting
# ----------------------------
def panel_event_scatter(
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

    # global axis limits for comparability
    allx = np.concatenate([dfs_by_q[q][xcol].to_numpy() for q in qs])
    ally = np.concatenate([dfs_by_q[q][ycol].to_numpy() for q in qs])
    finite = np.isfinite(allx) & np.isfinite(ally) & (allx > 0) & (ally > 0)
    xmin = float(np.min(allx[finite])) if finite.any() else 1e-6
    xmax = float(np.max(allx[finite])) if finite.any() else 1.0
    ymin = float(np.min(ally[finite])) if finite.any() else 1e-6
    ymax = float(np.max(ally[finite])) if finite.any() else 1.0

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

        mn = min(xmin, ymin)
        mx = max(xmax, ymax)
        ax.plot([mn, mx], [mn, mx])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # stats + N
        n = int((np.isfinite(x) & np.isfinite(y)).sum())
        ax.text(
            0.05, 0.95,
            f"N={n}\nRMSE={rmse(x,y):.2f}\nρ={spearman(x,y):.3f}\nBias={bias(x,y):.2f}",
            transform=ax.transAxes,
            va="top"
        )

        ax.set_title(f"Q={q}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    fig.suptitle(title)
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=np.atleast_1d(axes), shrink=0.9)
        cbar.set_label(ccol)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_event_hydrograph(
    gauge: str,
    year: int,
    quantile: float,
    method: str,  # "near" or "mean"
    t_peak_hi: pd.Timestamp,
    q_iv_year: pd.Series,
    q_hr_near_year: pd.Series,
    q_hr_mean_year: pd.Series,
    outpath: Path,
    hours_before: int = 12,
    hours_after: int = 24,
):
    t0 = t_peak_hi - pd.Timedelta(hours=hours_before)
    t1 = t_peak_hi + pd.Timedelta(hours=hours_after)

    hi = q_iv_year.loc[t0:t1]
    hn = q_hr_near_year.loc[t0:t1]
    hm = q_hr_mean_year.loc[t0:t1]

    thr_hi = float(q_iv_year.dropna().quantile(quantile)) if len(q_iv_year.dropna()) else np.nan
    thr_near = float(q_hr_near_year.dropna().quantile(quantile)) if len(q_hr_near_year.dropna()) else np.nan
    thr_mean = float(q_hr_mean_year.dropna().quantile(quantile)) if len(q_hr_mean_year.dropna()) else np.nan

    fig, ax = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)

    ax.plot(hi.index, hi.values, label="USGS IV (hi-res)")
    ax.plot(hn.index, hn.values, label="Hourly nearest")
    ax.plot(hm.index, hm.values, label="Hourly mean (centered)")

    # thresholds: distinct
    if np.isfinite(thr_hi):
        ax.axhline(thr_hi, color="k", linestyle="--", linewidth=1.2, label=f"Thr hi Q{quantile}")
    if np.isfinite(thr_near):
        ax.axhline(thr_near, color="tab:orange", linestyle=":", linewidth=1.2, label=f"Thr near Q{quantile}")
    if np.isfinite(thr_mean):
        ax.axhline(thr_mean, color="tab:green", linestyle="-.", linewidth=1.2, label=f"Thr mean Q{quantile}")

    ax.axvline(t_peak_hi, color="k", linestyle=":", linewidth=1.0, alpha=0.7, label="Hi peak time")

    q_hr = q_hr_near_year if method == "near" else q_hr_mean_year
    t_match, _ = pick_hourly_peak(q_hr, t_peak_hi, pd.Timedelta("1h"))
    if t_match is not None:
        ax.axvline(
            t_match,
            color=("tab:orange" if method == "near" else "tab:green"),
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label=f"Matched hourly peak ({method})",
        )

    ax.set_title(f"{gauge} event hydrograph ({year}) Q={quantile} method={method}")
    ax.set_xlabel("time")
    ax.set_ylabel("Q (m3/s)")
    format_time_axis(ax)
    ax.legend(ncol=2)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Run artifact helpers
# ----------------------------
def write_run_artifacts(run_dir: Path, params: dict, readme_text: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    with open(run_dir / "RUN_README.md", "w", encoding="utf-8") as f:
        f.write(readme_text.strip() + "\n")

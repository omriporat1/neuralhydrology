# src/flash_metrics.py
from __future__ import annotations
import pandas as pd
import numpy as np


from scipy.stats import spearmanr


def event_peak_comparison(
    q_hi: pd.Series,
    q_hourly: pd.Series,
    events: pd.DataFrame,
    search_window: str = "1h",
    hourly_quantile: float | None = None,
):
    """
    Compare event peaks and compute:

    - rel_peak_error
    - dt_minutes
    - detected
    - extreme_preserved (if hourly_quantile provided)
    - slope_ratio (max dQ/dt hourly / max dQ/dt hi-res)
    """

    if events.empty:
        return pd.DataFrame()

    win = pd.Timedelta(search_window)

    # Compute independent hourly threshold if requested
    if hourly_quantile is not None:
        qh_nonan = q_hourly.dropna()
        if len(qh_nonan):
            thr_hourly = float(qh_nonan.quantile(hourly_quantile))
        else:
            thr_hourly = np.nan
    else:
        thr_hourly = None

    rows = []

    for _, row in events.iterrows():
        t0 = row["peak_time"]
        q0 = float(row["peak_q"])

        hr_win = q_hourly.loc[t0 - win : t0 + win].dropna()

        if hr_win.empty:
            rows.append({
                "t_peak_hi": t0,
                "q_peak_hi": q0,
                "detected": False,
                "extreme_preserved": False,
                "rel_peak_error": np.nan,
                "dt_minutes": np.nan,
                "slope_ratio": np.nan,
            })
            continue

        t1 = hr_win.idxmax()
        q1 = float(hr_win.loc[t1])

        # relative error
        rel_err = (q1 - q0) / q0 if q0 > 0 else np.nan

        # timing error
        dt_min = (t1 - t0).total_seconds() / 60.0

        # extreme preservation
        if thr_hourly is not None and not np.isnan(thr_hourly):
            extreme_preserved = bool(q1 > thr_hourly)
        else:
            extreme_preserved = np.nan

        # Rising limb steepness ratio
        # compute max dQ/dt in +/- search_window
        hi_win = q_hi.loc[t0 - win : t0]
        hr_win2 = q_hourly.loc[t0 - win : t0]

        def max_slope(series):
            if len(series) < 2:
                return np.nan
            diffs = series.diff().dropna()
            dt = series.index.to_series().diff().dropna().dt.total_seconds()
            slopes = diffs / dt.values
            return slopes.max()

        slope_hi = max_slope(hi_win)
        slope_hr = max_slope(hr_win2)

        slope_ratio = slope_hr / slope_hi if slope_hi and slope_hi > 0 else np.nan

        rows.append({
            "t_peak_hi": t0,
            "q_peak_hi": q0,
            "t_peak_hr": t1,
            "q_peak_hr": q1,
            "detected": True,
            "extreme_preserved": extreme_preserved,
            "rel_peak_error": rel_err,
            "dt_minutes": dt_min,
            "slope_ratio": slope_ratio,
        })

    return pd.DataFrame(rows)

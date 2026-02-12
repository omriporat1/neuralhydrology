# src/flash_metrics.py
from __future__ import annotations
import pandas as pd
import numpy as np


def event_peak_comparison(
    q_hi: pd.Series,
    q_hourly: pd.Series,
    events: pd.DataFrame,
    search_window: str = "1h",
) -> pd.DataFrame:
    """
    For each event peak time from high-res series, compare with hourly series.

    - q_hi: high-res discharge (e.g., 15-min IV) (tz-aware recommended)
    - q_hourly: hourly discharge (e.g., nearest-to-hour or hourly mean) on hourly grid
    - events: output of detect_pot_events on q_hi
    - search_window: look for hourly peak within +/- this window around hi-res peak time

    Returns per-event table with:
    - t_peak_hi, q_peak_hi
    - t_peak_hr, q_peak_hr
    - dt_minutes (hr - hi)
    - rel_peak_error = (q_hr - q_hi) / q_hi
    - detected (bool)
    """
    if events.empty:
        return pd.DataFrame(columns=[
            "t_peak_hi","q_peak_hi","t_peak_hr","q_peak_hr",
            "dt_minutes","rel_peak_error","detected"
        ])

    win = pd.Timedelta(search_window)

    rows = []
    for _, row in events.iterrows():
        t0 = row["peak_time"]
        q0 = float(row["peak_q"])

        # search in hourly within +/- win (use available points only)
        hr_win = q_hourly.loc[t0 - win : t0 + win].dropna()

        if hr_win.empty:
            rows.append({
                "t_peak_hi": t0, "q_peak_hi": q0,
                "t_peak_hr": pd.NaT, "q_peak_hr": np.nan,
                "dt_minutes": np.nan,
                "rel_peak_error": np.nan,
                "detected": False,
            })
            continue

        t1 = hr_win.idxmax()
        q1 = float(hr_win.loc[t1])

        rows.append({
            "t_peak_hi": t0, "q_peak_hi": q0,
            "t_peak_hr": t1, "q_peak_hr": q1,
            "dt_minutes": float((t1 - t0).total_seconds() / 60.0),
            "rel_peak_error": float((q1 - q0) / q0) if q0 > 0 else np.nan,
            "detected": True,
        })

    return pd.DataFrame(rows)

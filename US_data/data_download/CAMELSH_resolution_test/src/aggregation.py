# src/aggregation.py
from __future__ import annotations
import pandas as pd


def to_hourly_candidates(q_iv: pd.Series, nearest_tolerance: str = "10min") -> dict[str, pd.Series]:
    """
    Create multiple plausible hourly series from an IV (sub-hourly) discharge series.

    q_iv:
      - index must be datetime-like
      - can be irregular (missing 15-min points)
      - may be tz-aware (recommended)

    Returns dict: name -> hourly pd.Series.
    - Aggregates (median/max/min/first/last) are labeled at the start of each hour [t, t+1h).
    - mean is centered: timestamp t represents ~[t-30min, t+30min).
    - nearest_* is sampled at exact hour stamps (t).
    """
    if not isinstance(q_iv.index, pd.DatetimeIndex):
        raise TypeError("q_iv must have a DatetimeIndex")

    q = q_iv.sort_index()

    # Resample to hourly bins. We choose label='left', closed='left'
    # so the timestamp represents the hour starting at that time: [t, t+1h)
    r = q.resample("1h", label="left", closed="left")

    candidates: dict[str, pd.Series] = {}
    
    # Centered 1-hour mean: timestamp t represents approximately [t-30min, t+30min)
    mean_centered = (
        q.resample("1h", label="right", closed="right")
        .mean()
    )
    mean_centered.index = mean_centered.index - pd.Timedelta("30min")
    
    candidates["mean"] = mean_centered
    candidates["median"] = r.median()
    candidates["max"] = r.max()
    candidates["min"] = r.min()
    candidates["last"] = r.last()
    candidates["first"] = r.first()

    # Nearest-to-hour sampling
    hourly_index = pd.date_range(
        start=q.index.min().floor("h"),
        end=q.index.max().ceil("h"),
        freq="1h",
        tz=q.index.tz,
    )
    sampled = q.reindex(
        hourly_index,
        method="nearest",
        tolerance=pd.Timedelta(nearest_tolerance),
    )
    sampled.name = q.name
    candidates[f"nearest_{nearest_tolerance}"] = sampled


    # Clean naming
    for k, s in candidates.items():
        s.name = f"q_usgs_hourly_{k}"
    return candidates

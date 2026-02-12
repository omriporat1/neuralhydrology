# src/events.py
from __future__ import annotations
import pandas as pd
import numpy as np


def detect_pot_events(
    q: pd.Series,
    quantile: float = 0.95,
    min_separation: str = "12h",
    min_event_points: int = 1,
) -> pd.DataFrame:
    """
    Detect Peak-Over-Threshold (POT) events from a discharge series.

    Approach:
    - threshold = q.quantile(quantile) (computed on non-NaN values)
    - candidate peaks are local maxima above threshold
    - enforce independence by requiring peaks to be at least `min_separation` apart
      (keep the larger peak if two are too close)

    Returns a DataFrame with one row per event:
    - peak_time, peak_q, threshold
    """
    if not isinstance(q.index, pd.DatetimeIndex):
        raise TypeError("q must have a DatetimeIndex")
    q = q.sort_index()

    qn = q.dropna()
    if qn.empty:
        return pd.DataFrame(columns=["peak_time", "peak_q", "threshold"])

    thr = float(qn.quantile(quantile))

    # candidate points above threshold
    qa = qn[qn > thr]
    if qa.empty:
        return pd.DataFrame(columns=["peak_time", "peak_q", "threshold"])

    # find local maxima among above-threshold points
    # A point is a local max if it is >= neighbors (in time order).
    vals = qa.values
    times = qa.index

    is_peak = np.zeros(len(qa), dtype=bool)
    for i in range(len(qa)):
        left = vals[i - 1] if i - 1 >= 0 else -np.inf
        right = vals[i + 1] if i + 1 < len(qa) else -np.inf
        if vals[i] >= left and vals[i] >= right:
            is_peak[i] = True

    peaks = qa[is_peak].sort_values(ascending=False)  # sort by magnitude (desc)

    # enforce independence (greedy by magnitude)
    min_sep = pd.Timedelta(min_separation)
    chosen = []
    for t, v in peaks.items():
        if all(abs(t - tc) >= min_sep for tc, _ in chosen):
            chosen.append((t, float(v)))

    # sort chosen peaks by time
    chosen.sort(key=lambda x: x[0])

    df = pd.DataFrame(chosen, columns=["peak_time", "peak_q"])
    df["threshold"] = thr

    # optional: require enough points above threshold around peak (very light filter)
    if min_event_points > 1:
        # count above-threshold points within +/- min_sep/2
        half = min_sep / 2
        keep = []
        for t in df["peak_time"]:
            window = qn.loc[t - half : t + half]
            keep.append(int((window > thr).sum()) >= min_event_points)
        df = df[pd.Series(keep, index=df.index)].reset_index(drop=True)

    return df

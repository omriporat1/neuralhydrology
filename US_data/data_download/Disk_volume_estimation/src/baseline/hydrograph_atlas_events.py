"""Stage 1 hydrograph-atlas event selection (Part C, section 7.3 of
docs/stage1_validation_optimization_foundation.md).

Small, deterministic, generic functions operating on a single basin's
*observed* discharge time series only. Per section 7.2/7.3, event selection
must use observed flow only -- no function in this module accepts a
predicted/simulated discharge argument, so it is structurally impossible for
model error to influence which events are selected (see
tests/test_hydrograph_atlas_events.py's signature-inspection tests).

Design decisions (recorded here because section 7.3 requires them to be
specified explicitly):

- Observed peak definition: a local maximum of the observed series, found by
  greedy iterative "declustering" -- repeatedly take the largest remaining
  finite observed value, then exclude a window of +/- min_separation_hours
  around its timestamp before looking for the next peak. This guarantees no
  two returned peaks are closer than min_separation_hours together.
- Event ranking statistic: the observed peak value itself (not volume, not
  duration).
- Minimum event separation: min_separation_hours, default 72 h
  (pre_hours + post_hours of the preferred standard window below), so
  distinct events' standard windows never overlap by construction.
- Tie-breaking: equal peak values -> earliest timestamp wins (deterministic,
  no reliance on pandas' unspecified stable-sort behavior across versions).
- Missing observations: a timestep with a non-finite (NaN) observed value is
  never itself a candidate peak; a window's pre/post span may still include
  NaN timesteps (recorded via n_missing_in_window, not excluded).
- Period-boundary handling: a window that would extend before the series'
  first timestamp or after its last is clipped to the available range and
  flagged window_clipped=True -- never dropped, never silently extended.
- Overlapping windows: NOT allowed. min_separation_hours >= pre_hours +
  post_hours guarantees this by construction for the declustered peak
  population; select_atlas_events additionally verifies no two selected
  windows overlap before returning (defensive, not merely by convention).
- Magnitude-stratum targets: among the declustered peak population (ranked
  descending by observed value), four magnitude classes are picked by
  nearest-value match to specific quantiles of that *peak population*
  (not of the full time series): moderate -> p50, upper_middle -> p75,
  high -> p90, extreme -> the single largest peak of record. Nearest-value
  ties break to the earliest timestamp; each peak can satisfy only one
  magnitude class (first-assigned wins, in moderate/upper_middle/high/
  extreme order), so a sparse peak population never double-counts one event.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "EventSelectionError",
    "EventWindow",
    "find_observed_peaks",
    "select_atlas_events",
    "select_high_flow_events",
]

_MAGNITUDE_CLASSES_IN_ORDER = ["moderate", "upper_middle", "high", "extreme"]
_MAGNITUDE_TARGET_QUANTILES = {"moderate": 0.50, "upper_middle": 0.75, "high": 0.90, "extreme": 1.00}


class EventSelectionError(ValueError):
    """Raised when event-selection inputs are invalid (e.g. no finite observations)."""


@dataclass(frozen=True)
class EventWindow:
    magnitude_class: str
    peak_time: pd.Timestamp
    peak_value: float
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    window_clipped: bool
    n_missing_in_window: int


def find_observed_peaks(observed: pd.Series, min_separation_hours: int) -> pd.Series:
    """Greedy declustered local maxima of an observed discharge series.

    observed: pandas Series with a DatetimeIndex, values in physical units
    (e.g. m^3/s). NaN values are ignored as candidates.

    Returns a Series of peak values indexed by their timestamp, sorted by
    timestamp ascending. Deterministic: ties on value are broken by earliest
    timestamp (the candidate pool is index-sorted before each argmax pass).
    """
    if not isinstance(observed.index, pd.DatetimeIndex):
        raise EventSelectionError("find_observed_peaks requires a DatetimeIndex")
    if min_separation_hours <= 0:
        raise EventSelectionError(f"min_separation_hours must be positive, got {min_separation_hours}")

    finite = observed.dropna().sort_index()
    if finite.empty:
        raise EventSelectionError("observed series has no finite values")

    separation = pd.Timedelta(hours=min_separation_hours)
    remaining = finite.copy()
    peaks: list[tuple[pd.Timestamp, float]] = []

    while not remaining.empty:
        # Deterministic tie-break: among ties for the max value, the
        # earliest timestamp is chosen (remaining is index-sorted ascending,
        # and idxmax() returns the first occurrence of the max on ties).
        peak_time = remaining.idxmax()
        peak_value = float(remaining.loc[peak_time])
        peaks.append((peak_time, peak_value))
        lo = peak_time - separation
        hi = peak_time + separation
        remaining = remaining.loc[(remaining.index < lo) | (remaining.index > hi)]

    peaks.sort(key=lambda t: t[0])
    idx = pd.DatetimeIndex([t for t, _ in peaks])
    return pd.Series([v for _, v in peaks], index=idx, name=observed.name or "observed")


def _windows_overlap(a_start, a_end, b_start, b_end) -> bool:
    return a_start < b_end and b_start < a_end


def select_atlas_events(
    observed: pd.Series,
    min_separation_hours: int = 72,
    pre_hours: int = 24,
    post_hours: int = 48,
) -> dict[str, EventWindow]:
    """Select up to four observed-flow events (moderate/upper_middle/high/
    extreme) for one basin, using observed flow only.

    Returns a dict keyed by magnitude class; a class is omitted (not present
    as a key) only if the declustered peak population has fewer candidates
    than magnitude classes -- never silently substituted.
    """
    peaks = find_observed_peaks(observed, min_separation_hours)
    # Rank descending by value; ties break to the earliest timestamp.
    tie_break = pd.DataFrame({"value": peaks.values, "time": peaks.index})
    tie_break = tie_break.sort_values(["value", "time"], ascending=[False, True])
    ranked = pd.Series(tie_break["value"].values, index=pd.DatetimeIndex(tie_break["time"]))

    series_start = observed.index.min()
    series_end = observed.index.max()

    selected: dict[str, EventWindow] = {}
    used_times: set = set()
    used_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for magnitude_class in _MAGNITUDE_CLASSES_IN_ORDER:
        target_q = _MAGNITUDE_TARGET_QUANTILES[magnitude_class]
        target_value = float(np.quantile(ranked.values, target_q))

        candidates = ranked.loc[[t for t in ranked.index if t not in used_times]]
        if candidates.empty:
            continue

        distances = (candidates.values - target_value)
        abs_distances = np.abs(distances)
        min_dist = abs_distances.min()
        tie_mask = abs_distances == min_dist
        tie_times = candidates.index[tie_mask]
        chosen_time = min(tie_times)  # earliest timestamp among nearest-value ties
        chosen_value = float(candidates.loc[chosen_time])

        raw_start = chosen_time - pd.Timedelta(hours=pre_hours)
        raw_end = chosen_time + pd.Timedelta(hours=post_hours)
        window_start = max(raw_start, series_start)
        window_end = min(raw_end, series_end)
        clipped = (window_start != raw_start) or (window_end != raw_end)

        if any(_windows_overlap(window_start, window_end, s, e) for s, e in used_windows):
            # Should not occur when min_separation_hours >= pre_hours +
            # post_hours; defensive fallback: drop this candidate and retry
            # with the next-nearest one rather than silently returning an
            # overlapping window.
            remaining_candidates = candidates.drop(index=chosen_time)
            found = False
            while not remaining_candidates.empty:
                distances = np.abs(remaining_candidates.values - target_value)
                min_dist = distances.min()
                tie_times = remaining_candidates.index[distances == min_dist]
                chosen_time = min(tie_times)
                chosen_value = float(remaining_candidates.loc[chosen_time])
                raw_start = chosen_time - pd.Timedelta(hours=pre_hours)
                raw_end = chosen_time + pd.Timedelta(hours=post_hours)
                window_start = max(raw_start, series_start)
                window_end = min(raw_end, series_end)
                clipped = (window_start != raw_start) or (window_end != raw_end)
                if not any(_windows_overlap(window_start, window_end, s, e) for s, e in used_windows):
                    found = True
                    break
                remaining_candidates = remaining_candidates.drop(index=chosen_time)
            if not found:
                continue

        n_missing_in_window = int(observed.loc[window_start:window_end].isna().sum())

        selected[magnitude_class] = EventWindow(
            magnitude_class=magnitude_class,
            peak_time=chosen_time,
            peak_value=chosen_value,
            window_start=window_start,
            window_end=window_end,
            window_clipped=bool(clipped),
            n_missing_in_window=n_missing_in_window,
        )
        used_times.add(chosen_time)
        used_windows.append((window_start, window_end))

    return selected


def select_high_flow_events(
    observed: pd.Series,
    *,
    threshold: float,
    min_separation_hours: int = 72,
    top_n: int = 3,
    pre_hours: int = 24,
    post_hours: int = 48,
) -> list[EventWindow]:
    """Select up to ``top_n`` observed high-flow events (declustered peak
    value >= ``threshold``) for one basin, using observed flow only.

    Distinct from :func:`select_atlas_events` (four magnitude-stratum
    quantile targets, used only for the frozen 8-basin qualitative overlay
    panels): this is a simple absolute-threshold top-N ranking, added for
    the Dynamic-Input-Family-A event/high-flow population audit -- see
    ``.scratch_local/moriah_evidence/dynamic_input_family_a_event_audit/
    METHODOLOGY_preregistered.md`` section 2 for the pre-registered rule
    this implements. ``select_atlas_events`` itself is untouched.

    Every returned :class:`EventWindow` has ``magnitude_class="high_flow"``
    (not one of ``select_atlas_events``'s four strata). Returned list is
    ordered by descending observed peak value (index 0 = largest). A basin
    with zero qualifying peaks returns an empty list (never raises).
    """
    if top_n <= 0:
        raise EventSelectionError(f"top_n must be positive, got {top_n}")
    if not np.isfinite(threshold):
        raise EventSelectionError(f"threshold must be finite, got {threshold}")

    peaks = find_observed_peaks(observed, min_separation_hours)
    qualifying = peaks[peaks.values >= threshold]
    if qualifying.empty:
        return []

    tie_break = pd.DataFrame({"value": qualifying.values, "time": qualifying.index})
    tie_break = tie_break.sort_values(["value", "time"], ascending=[False, True])
    ranked_times = list(pd.DatetimeIndex(tie_break["time"]))
    ranked_values = [float(v) for v in tie_break["value"]]

    series_start = observed.index.min()
    series_end = observed.index.max()

    selected: list[EventWindow] = []
    used_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for chosen_time, chosen_value in zip(ranked_times, ranked_values):
        if len(selected) >= top_n:
            break
        raw_start = chosen_time - pd.Timedelta(hours=pre_hours)
        raw_end = chosen_time + pd.Timedelta(hours=post_hours)
        window_start = max(raw_start, series_start)
        window_end = min(raw_end, series_end)
        clipped = (window_start != raw_start) or (window_end != raw_end)
        if any(_windows_overlap(window_start, window_end, s, e) for s, e in used_windows):
            # Already-declustered by min_separation_hours >= pre_hours +
            # post_hours (default), so this should not occur in practice;
            # skip rather than silently return an overlapping window.
            continue
        n_missing_in_window = int(observed.loc[window_start:window_end].isna().sum())
        selected.append(EventWindow(
            magnitude_class="high_flow",
            peak_time=chosen_time,
            peak_value=chosen_value,
            window_start=window_start,
            window_end=window_end,
            window_clipped=bool(clipped),
            n_missing_in_window=n_missing_in_window,
        ))
        used_windows.append((window_start, window_end))

    return selected

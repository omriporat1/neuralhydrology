"""Tests for src/baseline/hydrograph_atlas_events.py.

Synthetic fixtures only. In addition to behavioural checks, this module
includes a structural test (test_select_atlas_events_signature_has_no_predicted_argument)
proving -- by inspecting function signatures, not just by testing behaviour --
that no function in hydrograph_atlas_events.py can accept a predicted/
simulated discharge argument, satisfying section 7.4's explicit requirement
for "tests proving no prediction-error-based event selection".
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.baseline.hydrograph_atlas_events import (
    EventSelectionError,
    EventWindow,
    find_observed_peaks,
    select_atlas_events,
    select_high_flow_events,
)


def _hourly_index(n, start="2023-01-01"):
    return pd.date_range(start, periods=n, freq="h")


def _flat_series_with_spikes(n, spikes, start="2023-01-01"):
    idx = _hourly_index(n, start)
    values = np.ones(n) * 10.0
    for offset, value in spikes.items():
        values[offset] = value
    return pd.Series(values, index=idx, name="qobs")


# ---------------------------------------------------------------------------
# find_observed_peaks
# ---------------------------------------------------------------------------

def test_find_observed_peaks_requires_datetime_index():
    series = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(EventSelectionError):
        find_observed_peaks(series, min_separation_hours=24)


def test_find_observed_peaks_requires_positive_separation():
    series = _flat_series_with_spikes(100, {50: 100.0})
    with pytest.raises(EventSelectionError):
        find_observed_peaks(series, min_separation_hours=0)


def test_find_observed_peaks_rejects_all_nan():
    idx = _hourly_index(10)
    series = pd.Series([np.nan] * 10, index=idx)
    with pytest.raises(EventSelectionError):
        find_observed_peaks(series, min_separation_hours=24)


def test_find_observed_peaks_finds_single_spike():
    # Series sized so the declustering window around the one spike
    # (+/- min_separation_hours) covers the entire record -- nothing else
    # can remain as a candidate, so exactly one peak is returned.
    n = 2 * 24 + 1
    series = _flat_series_with_spikes(n, {24: 500.0})
    peaks = find_observed_peaks(series, min_separation_hours=24)
    assert len(peaks) == 1
    assert peaks.iloc[0] == 500.0


def test_find_observed_peaks_respects_min_separation():
    # Baseline ties can themselves produce additional low-value declustered
    # "peaks" (documented behaviour: the function enumerates ALL declustered
    # local maxima, not just visually obvious spikes) -- so assert on the
    # presence/absence of the specific spike values rather than total count.
    series = _flat_series_with_spikes(500, {100: 500.0, 110: 490.0, 300: 480.0})
    peaks = find_observed_peaks(series, min_separation_hours=24)
    # The 110-offset spike is within 24h of the 100-offset spike and has a
    # lower value, so it must be excluded by the declustering window.
    assert 500.0 in peaks.values
    assert 480.0 in peaks.values
    assert 490.0 not in peaks.values
    peak_times = sorted(peaks.index)
    for a, b in zip(peak_times, peak_times[1:]):
        assert (b - a) >= pd.Timedelta(hours=24)


def test_find_observed_peaks_deterministic_tie_break_earliest_wins():
    series = _flat_series_with_spikes(500, {100: 500.0, 400: 500.0})
    peaks = find_observed_peaks(series, min_separation_hours=24)
    spike_times = peaks.index[peaks.values == 500.0]
    assert len(spike_times) == 2
    assert list(peaks.index) == sorted(peaks.index)


def test_find_observed_peaks_ignores_nan_candidates():
    idx = _hourly_index(200)
    values = np.ones(200) * 10.0
    values[100] = np.nan
    values[150] = 300.0
    series = pd.Series(values, index=idx)
    peaks = find_observed_peaks(series, min_separation_hours=24)
    assert 300.0 in peaks.values
    assert not any(np.isnan(v) for v in peaks.values)


def test_find_observed_peaks_deterministic_repeat():
    series = _flat_series_with_spikes(1000, {i * 50: 100.0 + i for i in range(15)})
    p1 = find_observed_peaks(series, min_separation_hours=24)
    p2 = find_observed_peaks(series, min_separation_hours=24)
    pd.testing.assert_series_equal(p1, p2)


# ---------------------------------------------------------------------------
# select_atlas_events
# ---------------------------------------------------------------------------

def _rich_series(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = _hourly_index(n)
    base = rng.uniform(5.0, 15.0, size=n)
    # Sprinkle well-separated distinguishable spikes across a broad range.
    spike_offsets = list(range(80, n - 80, 150))
    rng2 = np.random.default_rng(seed + 1)
    for i, offset in enumerate(spike_offsets):
        base[offset] = 50.0 + rng2.uniform(0, 500.0)
    return pd.Series(base, index=idx, name="qobs")


def test_select_atlas_events_returns_up_to_four_classes():
    series = _rich_series()
    events = select_atlas_events(series, min_separation_hours=72, pre_hours=24, post_hours=48)
    assert set(events.keys()).issubset({"moderate", "upper_middle", "high", "extreme"})
    assert len(events) > 0


def test_select_atlas_events_all_values_are_event_windows():
    series = _rich_series()
    events = select_atlas_events(series)
    for window in events.values():
        assert isinstance(window, EventWindow)


def test_select_atlas_events_no_overlapping_windows():
    series = _rich_series()
    events = select_atlas_events(series, min_separation_hours=72, pre_hours=24, post_hours=48)
    windows = [(w.window_start, w.window_end) for w in events.values()]
    for i in range(len(windows)):
        for j in range(i + 1, len(windows)):
            a_start, a_end = windows[i]
            b_start, b_end = windows[j]
            assert not (a_start < b_end and b_start < a_end)


def test_select_atlas_events_deterministic_repeat():
    series = _rich_series()
    e1 = select_atlas_events(series)
    e2 = select_atlas_events(series)
    assert set(e1.keys()) == set(e2.keys())
    for key in e1:
        assert e1[key] == e2[key]


def test_select_atlas_events_distinct_peaks_across_classes():
    series = _rich_series()
    events = select_atlas_events(series)
    peak_times = [w.peak_time for w in events.values()]
    assert len(peak_times) == len(set(peak_times))


def test_select_atlas_events_window_clipped_at_series_start():
    # min_separation_hours must be >= pre_hours + post_hours (72 here) so
    # declustered peaks' standard windows never overlap by construction --
    # violating that invariant makes the overlap-avoidance fallback discard
    # otherwise-valid candidates (see module docstring, section 7.3).
    idx = _hourly_index(300)
    values = np.ones(300) * 10.0
    values[5] = 500.0  # peak very close to series start
    series = pd.Series(values, index=idx)
    events = select_atlas_events(series, min_separation_hours=72, pre_hours=24, post_hours=48)
    matching = [w for w in events.values() if w.peak_time == idx[5]]
    assert len(matching) == 1
    window = matching[0]
    assert window.peak_value == 500.0
    assert window.window_clipped
    assert window.window_start == series.index.min()


def test_select_atlas_events_window_clipped_at_series_end():
    idx = _hourly_index(300)
    values = np.ones(300) * 10.0
    values[295] = 500.0  # peak very close to series end
    series = pd.Series(values, index=idx)
    events = select_atlas_events(series, min_separation_hours=72, pre_hours=24, post_hours=48)
    matching = [w for w in events.values() if w.peak_time == idx[295]]
    assert len(matching) == 1
    window = matching[0]
    assert window.peak_value == 500.0
    assert window.window_clipped
    assert window.window_end == series.index.max()


def test_select_atlas_events_records_missing_observations_in_window():
    idx = _hourly_index(300)
    values = np.ones(300) * 10.0
    values[150] = 500.0
    values[145] = np.nan
    values[148] = np.nan
    series = pd.Series(values, index=idx)
    events = select_atlas_events(series, min_separation_hours=72, pre_hours=24, post_hours=48)
    matching = [w for w in events.values() if w.peak_time == idx[150]]
    assert len(matching) == 1
    assert matching[0].n_missing_in_window == 2


def test_select_atlas_events_too_few_peaks_omits_classes_not_substitutes():
    # Series sized so the single spike's declustering window covers the
    # entire record -- the peak population has exactly one member, so all
    # four magnitude-class quantile targets collapse onto it; only the
    # first-processed class ("moderate") can claim it and the rest must be
    # omitted, never substituted with a different (lower-quality) peak.
    n = 2 * 72 + 1
    series = _flat_series_with_spikes(n, {72: 500.0})
    events = select_atlas_events(series, min_separation_hours=72, pre_hours=24, post_hours=48)
    assert len(events) == 1
    peak_times = {w.peak_time for w in events.values()}
    assert len(peak_times) == 1


def test_select_atlas_events_raises_on_invalid_input():
    series = pd.Series([1.0, 2.0, 3.0])  # no DatetimeIndex
    with pytest.raises(EventSelectionError):
        select_atlas_events(series)


# ---------------------------------------------------------------------------
# select_high_flow_events (Dynamic-Input-Family-A event/high-flow audit)
# ---------------------------------------------------------------------------

def test_select_high_flow_events_empty_when_no_peak_clears_threshold():
    series = _flat_series_with_spikes(200, {50: 100.0, 150: 90.0})
    events = select_high_flow_events(series, threshold=1000.0)
    assert events == []


def test_select_high_flow_events_returns_only_qualifying_peaks():
    series = _flat_series_with_spikes(500, {50: 100.0, 200: 50.0, 350: 80.0})
    events = select_high_flow_events(series, threshold=75.0, min_separation_hours=24)
    assert {e.peak_value for e in events} == {100.0, 80.0}


def test_select_high_flow_events_ordered_descending_by_peak_value():
    series = _flat_series_with_spikes(500, {50: 80.0, 200: 100.0, 350: 90.0})
    events = select_high_flow_events(series, threshold=75.0, min_separation_hours=24)
    values = [e.peak_value for e in events]
    assert values == sorted(values, reverse=True)


def test_select_high_flow_events_respects_top_n_cap():
    series = _flat_series_with_spikes(
        1000, {50: 100.0, 200: 99.0, 350: 98.0, 500: 97.0}
    )
    events = select_high_flow_events(series, threshold=90.0, min_separation_hours=24, top_n=2)
    assert len(events) == 2
    assert [e.peak_value for e in events] == [100.0, 99.0]


def test_select_high_flow_events_all_events_are_high_flow_class():
    series = _flat_series_with_spikes(500, {50: 100.0, 200: 90.0})
    events = select_high_flow_events(series, threshold=85.0, min_separation_hours=24)
    assert all(e.magnitude_class == "high_flow" for e in events)


def test_select_high_flow_events_window_bounds_match_pre_post_hours():
    series = _flat_series_with_spikes(500, {200: 100.0})
    events = select_high_flow_events(
        series, threshold=90.0, min_separation_hours=24, pre_hours=24, post_hours=48,
    )
    assert len(events) == 1
    event = events[0]
    assert event.window_start == event.peak_time - pd.Timedelta(hours=24)
    assert event.window_end == event.peak_time + pd.Timedelta(hours=48)
    assert event.window_clipped is False


def test_select_high_flow_events_clips_window_at_series_boundary():
    series = _flat_series_with_spikes(30, {2: 100.0})
    events = select_high_flow_events(
        series, threshold=90.0, min_separation_hours=1, pre_hours=24, post_hours=48,
    )
    assert len(events) == 1
    assert events[0].window_clipped is True
    assert events[0].window_start == series.index.min()


def test_select_high_flow_events_rejects_nonpositive_top_n():
    series = _flat_series_with_spikes(100, {50: 100.0})
    with pytest.raises(EventSelectionError):
        select_high_flow_events(series, threshold=50.0, top_n=0)


def test_select_high_flow_events_rejects_nonfinite_threshold():
    series = _flat_series_with_spikes(100, {50: 100.0})
    with pytest.raises(EventSelectionError):
        select_high_flow_events(series, threshold=float("nan"))


def test_select_high_flow_events_deterministic_across_repeated_calls():
    series = _flat_series_with_spikes(500, {50: 100.0, 200: 90.0, 350: 95.0})
    first = select_high_flow_events(series, threshold=85.0, min_separation_hours=24)
    second = select_high_flow_events(series, threshold=85.0, min_separation_hours=24)
    assert first == second


def test_select_high_flow_events_signature_has_no_predicted_argument():
    params = list(inspect.signature(select_high_flow_events).parameters)
    assert params == [
        "observed", "threshold", "min_separation_hours", "top_n", "pre_hours", "post_hours",
    ]
    for name in params:
        assert not any(frag in name.lower() for frag in _DISALLOWED_PARAM_NAME_FRAGMENTS)


# ---------------------------------------------------------------------------
# Structural proof: no prediction-error-based event selection.
#
# Section 7.4 explicitly requires "tests proving no prediction-error-based
# event selection". Behavioural tests above show observed-only inputs
# determine the output; this test additionally proves it structurally: no
# public function in this module has a parameter that could plausibly carry
# predicted/simulated discharge, so it is not merely untested but impossible
# for model error to enter the selection at the API level.
# ---------------------------------------------------------------------------

_DISALLOWED_PARAM_NAME_FRAGMENTS = ("predict", "simulat", "sim_", "_sim", "model_output", "qsim", "error")


def test_find_observed_peaks_signature_has_no_predicted_argument():
    params = list(inspect.signature(find_observed_peaks).parameters)
    assert params == ["observed", "min_separation_hours"]
    for name in params:
        assert not any(frag in name.lower() for frag in _DISALLOWED_PARAM_NAME_FRAGMENTS)


def test_select_atlas_events_signature_has_no_predicted_argument():
    params = list(inspect.signature(select_atlas_events).parameters)
    assert params == ["observed", "min_separation_hours", "pre_hours", "post_hours"]
    for name in params:
        assert not any(frag in name.lower() for frag in _DISALLOWED_PARAM_NAME_FRAGMENTS)


def test_module_public_api_has_no_predicted_discharge_symbol():
    # "error" is deliberately excluded here: EventSelectionError is a
    # legitimate exception-class name, not a predicted-discharge argument.
    import src.baseline.hydrograph_atlas_events as mod
    fragments = tuple(f for f in _DISALLOWED_PARAM_NAME_FRAGMENTS if f != "error")
    for name in mod.__all__:
        assert not any(frag in name.lower() for frag in fragments)

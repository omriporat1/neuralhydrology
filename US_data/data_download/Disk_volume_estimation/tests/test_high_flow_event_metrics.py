"""Tests for src/baseline/high_flow_event_metrics.py.

Synthetic fixtures only. Covers section 1 (high-flow conditional metrics)
and section 3 (event-window metrics) of the pre-registered methodology at
.scratch_local/moriah_evidence/dynamic_input_family_a_event_audit/
METHODOLOGY_preregistered.md.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.baseline.hydrograph_atlas_events import EventWindow
from src.baseline.high_flow_event_metrics import (
    HighFlowEventMetricsError,
    basin_high_flow_threshold,
    event_metrics,
    high_flow_conditional_metrics,
)


# ---------------------------------------------------------------------------
# basin_high_flow_threshold
# ---------------------------------------------------------------------------

def test_basin_high_flow_threshold_matches_numpy_quantile_on_finite_values():
    obs = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    expected = float(np.quantile(obs[np.isfinite(obs)], 0.9))
    assert basin_high_flow_threshold(obs, 0.9) == pytest.approx(expected)


def test_basin_high_flow_threshold_ignores_nan():
    obs = np.array([np.nan, np.nan, 5.0, 5.0, 5.0])
    assert basin_high_flow_threshold(obs, 0.5) == pytest.approx(5.0)


def test_basin_high_flow_threshold_rejects_all_nan():
    obs = np.array([np.nan, np.nan])
    with pytest.raises(HighFlowEventMetricsError):
        basin_high_flow_threshold(obs, 0.9)


@pytest.mark.parametrize("quantile", [0.0, 1.0, -0.1, 1.5])
def test_basin_high_flow_threshold_rejects_out_of_range_quantile(quantile):
    obs = np.array([1.0, 2.0, 3.0])
    with pytest.raises(HighFlowEventMetricsError):
        basin_high_flow_threshold(obs, quantile)


# ---------------------------------------------------------------------------
# high_flow_conditional_metrics
# ---------------------------------------------------------------------------

def test_high_flow_conditional_metrics_never_returns_nse_key():
    obs = np.array([1.0, 2.0, 100.0, 110.0, 120.0])
    sim = np.array([1.1, 2.1, 105.0, 108.0, 118.0])
    result = high_flow_conditional_metrics(obs, sim, threshold=50.0)
    assert "nse" not in result


def test_high_flow_conditional_metrics_restricts_to_subset_above_threshold():
    obs = np.array([1.0, 2.0, 100.0, 100.0])
    sim = np.array([1.0, 2.0, 110.0, 90.0])  # errors of +10 and -10 above threshold
    result = high_flow_conditional_metrics(obs, sim, threshold=50.0)
    assert result["n"] == 2
    assert result["bias"] == pytest.approx(0.0)
    assert result["mae"] == pytest.approx(10.0)
    assert result["rmse"] == pytest.approx(10.0)


def test_high_flow_conditional_metrics_nrmse_uses_conditional_subset_mean():
    obs = np.array([100.0, 200.0])
    sim = np.array([110.0, 190.0])
    result = high_flow_conditional_metrics(obs, sim, threshold=50.0)
    obs_mean = 150.0
    expected_rmse = math.sqrt(np.mean((np.array([110.0, 190.0]) - obs) ** 2))
    assert result["nrmse"] == pytest.approx(expected_rmse / obs_mean)


def test_high_flow_conditional_metrics_zero_n_when_nothing_clears_threshold():
    obs = np.array([1.0, 2.0, 3.0])
    sim = np.array([1.0, 2.0, 3.0])
    result = high_flow_conditional_metrics(obs, sim, threshold=1000.0)
    assert result["n"] == 0
    assert math.isnan(result["rmse"])
    assert math.isnan(result["pearson_r"])


def test_high_flow_conditional_metrics_suppresses_correlation_below_min_n():
    rng = np.random.default_rng(0)
    obs = 100.0 + rng.normal(size=5)
    sim = obs + rng.normal(scale=0.1, size=5)
    result = high_flow_conditional_metrics(obs, sim, threshold=0.0, min_n_for_correlation=10)
    assert result["n"] == 5
    assert math.isnan(result["pearson_r"])
    assert math.isnan(result["kge"])


def test_high_flow_conditional_metrics_populates_correlation_above_min_n():
    rng = np.random.default_rng(0)
    obs = 100.0 + rng.normal(size=20)
    sim = obs + rng.normal(scale=0.1, size=20)
    result = high_flow_conditional_metrics(obs, sim, threshold=0.0, min_n_for_correlation=10)
    assert result["n"] == 20
    assert not math.isnan(result["pearson_r"])
    assert result["pearson_r"] > 0.9


def test_high_flow_conditional_metrics_ignores_non_admitted_nan_pairs():
    obs = np.array([100.0, 100.0, np.nan, 100.0])
    sim = np.array([100.0, np.nan, 100.0, 100.0])
    result = high_flow_conditional_metrics(obs, sim, threshold=50.0)
    assert result["n"] == 2  # only index 0 and 3 have both finite


def test_high_flow_conditional_metrics_rejects_shape_mismatch():
    with pytest.raises(HighFlowEventMetricsError):
        high_flow_conditional_metrics(np.array([1.0, 2.0]), np.array([1.0]), threshold=0.0)


# ---------------------------------------------------------------------------
# event_metrics
# ---------------------------------------------------------------------------

def _hourly_index(n, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="h")


def _make_event(peak_time, peak_value):
    return EventWindow(
        magnitude_class="high_flow",
        peak_time=peak_time,
        peak_value=peak_value,
        window_start=peak_time - pd.Timedelta(hours=2),
        window_end=peak_time + pd.Timedelta(hours=2),
        window_clipped=False,
        n_missing_in_window=0,
    )


def test_event_metrics_peak_magnitude_and_timing_exact_match():
    dates = _hourly_index(5)
    obs = np.array([10.0, 20.0, 100.0, 20.0, 10.0])
    sim = np.array([10.0, 20.0, 100.0, 20.0, 10.0])
    event = _make_event(dates[2], 100.0)
    result = event_metrics(dates, obs, sim, event=event)
    assert result["sim_peak"] == pytest.approx(100.0)
    assert result["abs_peak_error"] == pytest.approx(0.0)
    assert result["abs_timing_error_hours"] == pytest.approx(0.0)


def test_event_metrics_detects_timing_offset():
    dates = _hourly_index(5)
    obs = np.array([10.0, 20.0, 100.0, 20.0, 10.0])
    sim = np.array([10.0, 100.0, 90.0, 20.0, 10.0])  # sim peak one hour early
    event = _make_event(dates[2], 100.0)
    result = event_metrics(dates, obs, sim, event=event)
    assert result["sim_peak_time"] == dates[1].isoformat()
    assert result["abs_timing_error_hours"] == pytest.approx(1.0)


def test_event_metrics_relative_peak_error_sign_convention():
    dates = _hourly_index(3)
    obs = np.array([10.0, 100.0, 10.0])
    sim = np.array([10.0, 120.0, 10.0])  # overpredicts peak by 20%
    event = _make_event(dates[1], 100.0)
    result = event_metrics(dates, obs, sim, event=event)
    assert result["signed_peak_bias"] == pytest.approx(20.0)
    assert result["relative_peak_error"] == pytest.approx(0.2)


def test_event_metrics_volume_is_rectangular_sum_times_3600s():
    dates = _hourly_index(3)
    obs = np.array([1.0, 1.0, 1.0])
    sim = np.array([2.0, 2.0, 2.0])
    event = _make_event(dates[1], 1.0)
    result = event_metrics(dates, obs, sim, event=event)
    assert result["obs_volume_m3"] == pytest.approx(3.0 * 3600.0)
    assert result["sim_volume_m3"] == pytest.approx(6.0 * 3600.0)
    assert result["relative_volume_bias"] == pytest.approx(1.0)


def test_event_metrics_excludes_non_admitted_samples_from_volume_and_shape():
    dates = _hourly_index(4)
    obs = np.array([1.0, np.nan, 1.0, 1.0])
    sim = np.array([1.0, 1.0, np.nan, 1.0])
    event = _make_event(dates[0], 1.0)
    result = event_metrics(dates, obs, sim, event=event)
    # Only indices 0 and 3 have both finite -> n_admitted=2, volume=2*3600
    assert result["n_admitted"] == 2
    assert result["obs_volume_m3"] == pytest.approx(2.0 * 3600.0)


def test_event_metrics_zero_admitted_returns_nan_not_raise():
    dates = _hourly_index(3)
    obs = np.array([np.nan, np.nan, np.nan])
    sim = np.array([1.0, 2.0, 3.0])
    event = _make_event(dates[1], 5.0)
    result = event_metrics(dates, obs, sim, event=event)
    assert result["n_admitted"] == 0
    assert math.isnan(result["sim_peak"])
    assert math.isnan(result["rmse"])


def test_event_metrics_supplementary_nse_kge_present_but_labeled():
    dates = _hourly_index(10)
    rng = np.random.default_rng(1)
    obs = 10.0 + rng.normal(size=10)
    sim = obs + rng.normal(scale=0.01, size=10)
    event = _make_event(dates[5], float(obs[5]))
    result = event_metrics(dates, obs, sim, event=event)
    assert "nse_supplementary" in result
    assert "kge_supplementary" in result
    assert result["nse_supplementary"] > 0.9


def test_event_metrics_rejects_length_mismatch():
    dates = _hourly_index(3)
    event = _make_event(dates[1], 1.0)
    with pytest.raises(HighFlowEventMetricsError):
        event_metrics(dates, np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), event=event)

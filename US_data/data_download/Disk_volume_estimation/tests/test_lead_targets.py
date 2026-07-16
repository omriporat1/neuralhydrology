"""Tests for src/baseline/lead_targets.py (Milestone 2K-G-I I-C1).

Expected values are hand-computed following build_lead_target's defined
convert-then-shift API (convert m3/s -> mm/h, then shift by -lead_hours)
-- never derived by re-deriving the answer from the function under test.
Convert-then-shift is tested here as the one public API to call; it is not
implied that shift-then-convert would give a different (or wrong) result
-- basin area is constant per series, so the two linear operations commute.
"""
import math

import numpy as np
import pandas as pd
import pytest

from src.baseline.lead_targets import (
    DEFAULT_LEADS_HOURS,
    LeadTargetError,
    LeadTargetMetadata,
    build_lead_target,
    build_lead_targets,
    variable_name_for_lead,
)

HOURLY = "h"


def _hourly_index(n, start="2023-01-01"):
    return pd.date_range(start, periods=n, freq=HOURLY)


# ---------------------------------------------------------------------------
# Asymmetric synthetic ramp: proves shift direction (t -> t+lead, not t-lead)
# ---------------------------------------------------------------------------


def test_ramp_shift_direction_lead01():
    # ramp in m3/s: 0,10,20,...,90 over area=3.6 -> mm/h equals the ramp value
    # itself (q*3.6/3.6=q). target(t) must equal source(t+1), not source(t-1).
    idx = _hourly_index(10)
    q = pd.Series(np.arange(0.0, 100.0, 10.0), index=idx)
    target, _ = build_lead_target(q, area_km2=3.6, lead_hours=1)
    # hand: target[0] = mm[1] = 10.0; target[1] = mm[2] = 20.0; ...
    expected = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, np.nan]
    np.testing.assert_allclose(target.to_numpy(), expected, rtol=1e-15, equal_nan=True)
    # Explicitly rule out the reversed (t-lead) direction.
    reversed_wrong = [np.nan, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    assert not np.allclose(
        np.nan_to_num(target.to_numpy(), nan=-999.0),
        np.nan_to_num(reversed_wrong, nan=-999.0),
    )


# ---------------------------------------------------------------------------
# All four binding leads + exact trailing boundary NaN counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lead_hours", [1, 3, 6, 12])
def test_all_four_leads_hand_computed_and_trailing_nans(lead_hours):
    n = 20
    idx = _hourly_index(n)
    # area=3.6 -> mm/h numerically equals the m3/s ramp value.
    q = pd.Series(np.arange(n, dtype=np.float64), index=idx)
    target, metadata = build_lead_target(q, area_km2=3.6, lead_hours=lead_hours)

    valid_len = n - lead_hours
    # hand: target[i] = q[i + lead_hours] for i in [0, valid_len)
    expected_valid = np.arange(lead_hours, n, dtype=np.float64)
    np.testing.assert_allclose(
        target.to_numpy()[:valid_len], expected_valid, rtol=1e-15
    )
    # Exactly lead_hours trailing NaNs, no more, no fewer.
    trailing = target.to_numpy()[valid_len:]
    assert len(trailing) == lead_hours
    assert np.all(np.isnan(trailing))
    assert not np.isnan(target.to_numpy()[valid_len - 1])

    assert metadata.lead_hours == lead_hours
    assert metadata.variable_name == f"qobs_mm_per_h_lead{lead_hours:02d}"


def test_default_leads_hours_constant():
    assert DEFAULT_LEADS_HOURS == (1, 3, 6, 12)


# ---------------------------------------------------------------------------
# Source NaN propagation to the correct (earlier) target index
# ---------------------------------------------------------------------------


def test_source_nan_propagates_to_correct_earlier_index():
    idx = _hourly_index(8)
    q = pd.Series([0.0, 1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0], index=idx)
    lead_hours = 3
    target, _ = build_lead_target(q, area_km2=3.6, lead_hours=lead_hours)
    # source NaN sits at position 3 -> target[3 - 3] = target[0] must be NaN.
    assert math.isnan(target.iloc[0])
    # No other position should be NaN except the trailing boundary (last 3).
    non_boundary = target.iloc[1 : len(q) - lead_hours]
    assert not non_boundary.isna().any()
    # hand: target[1] = q[4] = 4.0
    assert target.iloc[1] == pytest.approx(4.0, rel=1e-15)


def test_multiple_source_nans_each_propagate_independently():
    idx = _hourly_index(6)
    q = pd.Series([np.nan, 1.0, np.nan, 3.0, 4.0, 5.0], index=idx)
    target, _ = build_lead_target(q, area_km2=3.6, lead_hours=1)
    # target[t] = q[t+1]: target[0]=q[1]=1.0 (not nan);
    # target[1]=q[2]=NaN; target[2]=q[3]=3.0; target[3]=q[4]=4.0;
    # target[4]=q[5]=5.0; target[5]=NaN (boundary, q[6] doesn't exist).
    expected = [1.0, np.nan, 3.0, 4.0, 5.0, np.nan]
    np.testing.assert_allclose(target.to_numpy(), expected, rtol=1e-15, equal_nan=True)


# ---------------------------------------------------------------------------
# Timestamps preserved unchanged
# ---------------------------------------------------------------------------


def test_timestamps_and_length_preserved():
    idx = _hourly_index(5)
    q = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
    target, _ = build_lead_target(q, area_km2=3.6, lead_hours=2)
    assert len(target) == len(q)
    assert target.index.equals(idx)


# ---------------------------------------------------------------------------
# Variable names and metadata
# ---------------------------------------------------------------------------


def test_variable_name_for_lead_default_template():
    assert variable_name_for_lead(1) == "qobs_mm_per_h_lead01"
    assert variable_name_for_lead(3) == "qobs_mm_per_h_lead03"
    assert variable_name_for_lead(6) == "qobs_mm_per_h_lead06"
    assert variable_name_for_lead(12) == "qobs_mm_per_h_lead12"


def test_metadata_fields_populated():
    idx = _hourly_index(20)
    q = pd.Series(np.arange(20.0), index=idx)
    _, metadata = build_lead_target(
        q, area_km2=100.0, lead_hours=6, source_variable="qobs_m3s"
    )
    assert isinstance(metadata, LeadTargetMetadata)
    assert metadata.variable_name == "qobs_mm_per_h_lead06"
    assert metadata.lead_hours == 6
    assert metadata.units == "mm/h"
    assert metadata.source_variable == "qobs_m3s"
    assert "6-hour" in metadata.long_name
    assert "qobs_m3s" in metadata.long_name


def test_build_lead_targets_batch_matches_individual_calls():
    idx = _hourly_index(20)
    q = pd.Series(np.arange(20.0), index=idx)
    batch = build_lead_targets(q, area_km2=3.6, leads_hours=(1, 3, 6, 12))
    assert set(batch.keys()) == {
        "qobs_mm_per_h_lead01",
        "qobs_mm_per_h_lead03",
        "qobs_mm_per_h_lead06",
        "qobs_mm_per_h_lead12",
    }
    for lead_hours in (1, 3, 6, 12):
        name = variable_name_for_lead(lead_hours)
        batch_series, batch_meta = batch[name]
        single_series, single_meta = build_lead_target(q, 3.6, lead_hours)
        pd.testing.assert_series_equal(batch_series, single_series)
        assert batch_meta == single_meta


# ---------------------------------------------------------------------------
# Invalid lead rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_lead", [0, -1, -6])
def test_nonpositive_lead_rejected(bad_lead):
    idx = _hourly_index(5)
    q = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
    with pytest.raises(LeadTargetError):
        build_lead_target(q, area_km2=3.6, lead_hours=bad_lead)


def test_non_integer_lead_rejected():
    idx = _hourly_index(5)
    q = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
    with pytest.raises(LeadTargetError):
        build_lead_target(q, area_km2=3.6, lead_hours=6.0)
    with pytest.raises(LeadTargetError):
        build_lead_target(q, area_km2=3.6, lead_hours=True)


# ---------------------------------------------------------------------------
# Rejection of irregular / duplicate / descending / non-hourly time indexes
# ---------------------------------------------------------------------------


def test_non_datetimeindex_rejected():
    q = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
    with pytest.raises(LeadTargetError):
        build_lead_target(q, area_km2=3.6, lead_hours=1)


def test_too_short_index_rejected():
    idx = _hourly_index(1)
    q = pd.Series([1.0], index=idx)
    with pytest.raises(LeadTargetError):
        build_lead_target(q, area_km2=3.6, lead_hours=1)


def test_duplicate_timestamps_rejected():
    idx = pd.DatetimeIndex(
        ["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 01:00", "2023-01-01 02:00"]
    )
    q = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    with pytest.raises(LeadTargetError):
        build_lead_target(q, area_km2=3.6, lead_hours=1)


def test_descending_timestamps_rejected():
    idx = _hourly_index(5)[::-1]
    q = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
    with pytest.raises(LeadTargetError):
        build_lead_target(q, area_km2=3.6, lead_hours=1)


def test_gap_in_hourly_index_rejected():
    idx = pd.DatetimeIndex(
        ["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 03:00", "2023-01-01 04:00"]
    )
    q = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    with pytest.raises(LeadTargetError):
        build_lead_target(q, area_km2=3.6, lead_hours=1)


def test_non_hourly_frequency_rejected():
    idx = pd.date_range("2023-01-01", periods=5, freq="30min")
    q = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
    with pytest.raises(LeadTargetError):
        build_lead_target(q, area_km2=3.6, lead_hours=1)


def test_daily_frequency_rejected():
    idx = pd.date_range("2023-01-01", periods=5, freq="D")
    q = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
    with pytest.raises(LeadTargetError):
        build_lead_target(q, area_km2=3.6, lead_hours=1)


def test_non_series_input_rejected():
    idx = _hourly_index(5)
    with pytest.raises(LeadTargetError):
        build_lead_target(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), area_km2=3.6, lead_hours=1)


# ---------------------------------------------------------------------------
# Invalid area is still rejected (delegated to units.py, exercised here too)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_area", [0.0, -5.0, float("nan"), float("inf")])
def test_invalid_area_rejected(bad_area):
    idx = _hourly_index(5)
    q = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
    with pytest.raises(ValueError):
        build_lead_target(q, area_km2=bad_area, lead_hours=1)

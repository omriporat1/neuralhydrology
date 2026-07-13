"""Tests for src/baseline/units.py (Milestone 2K-G-I I-B).

Expected values are hand-computed from the binding formulas
(q_mm_per_h = q_m3s * 3.6 / area_km2) — never derived by calling the
opposite conversion function (round-trip tests excepted, where the
round trip itself is the property under test).
"""
import math

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.baseline.units import (
    discharge_m3s_to_runoff_mm_per_h,
    runoff_mm_per_h_to_discharge_m3s,
)

# ---------------------------------------------------------------------------
# Hand-calculated scalar values
# ---------------------------------------------------------------------------


def test_hand_calc_forward_unit_case():
    # 1 m3/s = 3600 m3/h over 3.6 km2 = 3.6e6 m2 -> 1e-3 m/h = 1.0 mm/h
    assert discharge_m3s_to_runoff_mm_per_h(1.0, 3.6) == pytest.approx(1.0, rel=1e-15)


def test_hand_calc_inverse_unit_case():
    # 1.0 mm/h over 3.6 km2 -> 1.0 m3/s (hand: 1.0 * 3.6 / 3.6)
    assert runoff_mm_per_h_to_discharge_m3s(1.0, 3.6) == pytest.approx(1.0, rel=1e-15)


def test_small_basin_forward():
    # hand: 0.5 * 3.6 / 5.0 = 1.8 / 5.0 = 0.36 mm/h
    assert discharge_m3s_to_runoff_mm_per_h(0.5, 5.0) == pytest.approx(0.36, rel=1e-12)


def test_large_basin_forward():
    # hand: 1234.5 * 3.6 / 25000.0 = 4444.2 / 25000.0 = 0.177768 mm/h
    result = discharge_m3s_to_runoff_mm_per_h(1234.5, 25000.0)
    assert result == pytest.approx(0.177768, rel=1e-12)


def test_small_basin_inverse():
    # hand: 0.36 * 5.0 / 3.6 = 1.8 / 3.6 = 0.5 m3/s
    assert runoff_mm_per_h_to_discharge_m3s(0.36, 5.0) == pytest.approx(0.5, rel=1e-12)


def test_zero_discharge_is_exact_zero_both_directions():
    assert discharge_m3s_to_runoff_mm_per_h(0.0, 123.4) == 0.0
    assert runoff_mm_per_h_to_discharge_m3s(0.0, 123.4) == 0.0


def test_negative_discharge_converts_arithmetically():
    # Utility-level documented behavior; the builder enforces the cleaned-
    # target policy separately. hand: -1.0 * 3.6 / 3.6 = -1.0
    assert discharge_m3s_to_runoff_mm_per_h(-1.0, 3.6) == pytest.approx(-1.0, rel=1e-15)


def test_nan_discharge_preserved_scalar():
    assert math.isnan(discharge_m3s_to_runoff_mm_per_h(float("nan"), 10.0))
    assert math.isnan(runoff_mm_per_h_to_discharge_m3s(float("nan"), 10.0))


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_float64_round_trip_tight_tolerance():
    q = np.array([0.0, 1e-6, 0.5, 1.0, 1234.5, 98765.4321], dtype=np.float64)
    area = 537.25
    back = runoff_mm_per_h_to_discharge_m3s(
        discharge_m3s_to_runoff_mm_per_h(q, area), area
    )
    np.testing.assert_allclose(back, q, rtol=1e-12, atol=0.0)


def test_float32_write_read_round_trip():
    # Simulate the package pathway: convert in float64, write float32,
    # read back, invert. float32 has ~7 significant digits -> rtol 1e-5.
    q = np.array([1e-4, 0.5, 1.0, 1234.5, 98765.4], dtype=np.float64)
    area = 42.42
    mm_f32 = discharge_m3s_to_runoff_mm_per_h(q, area).astype(np.float32)
    back = runoff_mm_per_h_to_discharge_m3s(mm_f32.astype(np.float64), area)
    np.testing.assert_allclose(back, q, rtol=1e-5, atol=0.0)


# ---------------------------------------------------------------------------
# Container types and broadcasting
# ---------------------------------------------------------------------------


def test_numpy_array_input_values_and_type():
    q = np.array([0.0, 1.0, 2.0])
    result = discharge_m3s_to_runoff_mm_per_h(q, 3.6)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    # hand: [0.0, 1.0, 2.0] * 3.6 / 3.6
    np.testing.assert_allclose(result, [0.0, 1.0, 2.0], rtol=1e-15)


def test_broadcasting_array_discharge_with_array_area():
    q = np.array([1.0, 1.0, 5.0])
    area = np.array([3.6, 7.2, 3.6])
    result = discharge_m3s_to_runoff_mm_per_h(q, area)
    # hand: 1*3.6/3.6=1.0; 1*3.6/7.2=0.5; 5*3.6/3.6=5.0
    np.testing.assert_allclose(result, [1.0, 0.5, 5.0], rtol=1e-15)


def test_broadcasting_scalar_discharge_with_array_area():
    result = discharge_m3s_to_runoff_mm_per_h(1.0, np.array([3.6, 7.2]))
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [1.0, 0.5], rtol=1e-15)


def test_pandas_series_values_index_name_preserved():
    idx = pd.date_range("2023-01-01", periods=3, freq="h", name="date")
    q = pd.Series([1.0, np.nan, 7.2], index=idx, name="qobs_m3s")
    result = discharge_m3s_to_runoff_mm_per_h(q, 7.2)
    assert isinstance(result, pd.Series)
    assert result.name == "qobs_m3s"
    assert result.index.equals(idx)
    # hand: 1*3.6/7.2=0.5; NaN; 7.2*3.6/7.2=3.6
    assert result.iloc[0] == pytest.approx(0.5, rel=1e-15)
    assert math.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(3.6, rel=1e-15)


def test_pandas_series_inverse_preserves_structure():
    q = pd.Series([0.5, 3.6], index=["a", "b"], name="qobs_mm_per_h_lead06")
    result = runoff_mm_per_h_to_discharge_m3s(q, 7.2)
    assert isinstance(result, pd.Series)
    assert result.name == "qobs_mm_per_h_lead06"
    assert list(result.index) == ["a", "b"]
    # hand: 0.5*7.2/3.6=1.0; 3.6*7.2/3.6=7.2
    assert result.loc["a"] == pytest.approx(1.0, rel=1e-15)
    assert result.loc["b"] == pytest.approx(7.2, rel=1e-15)


def test_xarray_dataarray_dims_coords_preserved():
    dates = pd.date_range("2023-01-01", periods=4, freq="h")
    da = xr.DataArray(
        np.array([0.0, 1.0, np.nan, 7.2]),
        dims=("date",),
        coords={"date": dates},
        name="qobs_m3s",
    )
    result = discharge_m3s_to_runoff_mm_per_h(da, 7.2)
    assert isinstance(result, xr.DataArray)
    assert result.dims == ("date",)
    assert np.array_equal(result["date"].values, dates.values)
    # hand: 0; 1*3.6/7.2=0.5; NaN; 7.2*3.6/7.2=3.6
    np.testing.assert_allclose(
        result.values, [0.0, 0.5, np.nan, 3.6], rtol=1e-15, equal_nan=True
    )


def test_nan_preserved_in_array():
    q = np.array([1.0, np.nan, 2.0])
    result = discharge_m3s_to_runoff_mm_per_h(q, 3.6)
    assert math.isnan(result[1])
    assert not math.isnan(result[0]) and not math.isnan(result[2])


# ---------------------------------------------------------------------------
# Invalid area rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_area", [0.0, -5.0, float("nan"), float("inf"), float("-inf")]
)
def test_invalid_scalar_area_rejected_forward(bad_area):
    with pytest.raises(ValueError):
        discharge_m3s_to_runoff_mm_per_h(1.0, bad_area)


@pytest.mark.parametrize(
    "bad_area", [0.0, -5.0, float("nan"), float("inf"), float("-inf")]
)
def test_invalid_scalar_area_rejected_inverse(bad_area):
    with pytest.raises(ValueError):
        runoff_mm_per_h_to_discharge_m3s(1.0, bad_area)


def test_invalid_area_inside_numpy_array_rejected():
    with pytest.raises(ValueError):
        discharge_m3s_to_runoff_mm_per_h(np.array([1.0, 1.0]), np.array([10.0, 0.0]))


def test_invalid_area_inside_series_rejected():
    with pytest.raises(ValueError):
        discharge_m3s_to_runoff_mm_per_h(1.0, pd.Series([10.0, np.nan]))


def test_invalid_area_inside_dataarray_rejected():
    bad = xr.DataArray(np.array([10.0, -3.0]), dims=("basin",))
    with pytest.raises(ValueError):
        runoff_mm_per_h_to_discharge_m3s(1.0, bad)


# ---------------------------------------------------------------------------
# Infinite discharge rejection (documented behavior)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_q", [float("inf"), float("-inf")])
def test_infinite_scalar_discharge_rejected_both_directions(bad_q):
    with pytest.raises(ValueError):
        discharge_m3s_to_runoff_mm_per_h(bad_q, 10.0)
    with pytest.raises(ValueError):
        runoff_mm_per_h_to_discharge_m3s(bad_q, 10.0)


def test_infinite_discharge_inside_array_rejected():
    with pytest.raises(ValueError):
        discharge_m3s_to_runoff_mm_per_h(np.array([1.0, np.inf]), 10.0)

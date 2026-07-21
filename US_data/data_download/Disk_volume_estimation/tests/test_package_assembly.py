"""Tests for src/baseline/package_assembly.py (Compact Scientific Package
builder, first code increment).

Expected values are hand-computed from the binding formulas exactly as in
tests/test_units.py and tests/test_lead_targets.py (area=3.6 makes mm/h
numerically equal the m3/s ramp value) -- never derived by re-deriving the
answer from the function under test.
"""
import copy
import math

import numpy as np
import pandas as pd
import pytest

from src.baseline.package_assembly import (
    DYNAMIC_INPUTS,
    RAW_TARGET_VARIABLE,
    PackageAssemblyError,
    assemble_basin_package_table,
)
from src.baseline.policy import load_stage1_baseline_policy

REPO_ROOT_POLICY = (
    __import__("pathlib").Path(__file__).resolve().parents[1]
    / "config"
    / "stage1_scientific_baseline_v001.yaml"
)

EXPECTED_LEAD_COLUMNS = [
    "qobs_mm_per_h_lead01",
    "qobs_mm_per_h_lead03",
    "qobs_mm_per_h_lead06",
    "qobs_mm_per_h_lead12",
]
EXPECTED_COLUMN_ORDER = list(DYNAMIC_INPUTS) + [RAW_TARGET_VARIABLE] + EXPECTED_LEAD_COLUMNS


def _hourly_index(n, start="2023-01-01"):
    return pd.date_range(start, periods=n, freq="h")


def _valid_forcing(idx, **overrides):
    n = len(idx)
    data = {
        "mrms_qpe_1h_mm": np.linspace(0.0, 1.0, n),
        "rtma_2t_K": np.linspace(270.0, 280.0, n),
        "rtma_2d_K": np.linspace(260.0, 270.0, n),
        "rtma_2sh_kgkg": np.linspace(0.001, 0.002, n),
        "rtma_10u_ms": np.linspace(-2.0, 2.0, n),
        "rtma_10v_ms": np.linspace(-1.0, 1.0, n),
        "mrms_qpe_1h_mm_gap": np.zeros(n, dtype=np.float32),
        "rtma_gap": np.zeros(n, dtype=np.float32),
    }
    data.update(overrides)
    return pd.DataFrame(data, index=idx)


def _valid_qobs(idx, values=None):
    n = len(idx)
    if values is None:
        values = np.arange(n, dtype=np.float64)
    return pd.Series(values, index=idx, name="qobs_m3s")


# ---------------------------------------------------------------------------
# 1. Column order / 2. exactly eight dynamic inputs
# ---------------------------------------------------------------------------


def test_output_column_order_exact():
    idx = _hourly_index(20)
    table = assemble_basin_package_table(_valid_forcing(idx), _valid_qobs(idx), area_km2=3.6)
    assert list(table.columns) == EXPECTED_COLUMN_ORDER


def test_exactly_eight_dynamic_input_columns():
    assert len(DYNAMIC_INPUTS) == 8
    assert list(DYNAMIC_INPUTS) == [
        "mrms_qpe_1h_mm",
        "rtma_2t_K",
        "rtma_2d_K",
        "rtma_2sh_kgkg",
        "rtma_10u_ms",
        "rtma_10v_ms",
        "mrms_qpe_1h_mm_gap",
        "rtma_gap",
    ]


def test_dynamic_inputs_matches_committed_policy():
    policy = load_stage1_baseline_policy(REPO_ROOT_POLICY)
    assert list(DYNAMIC_INPUTS) == policy["dynamic_inputs"]


# ---------------------------------------------------------------------------
# 3. Raw qobs_m3s preserved exactly, including NaNs
# ---------------------------------------------------------------------------


def test_raw_qobs_preserved_exactly_including_nan():
    idx = _hourly_index(8)
    q = pd.Series([0.0, 1.0, np.nan, 3.0, 4.0, np.nan, 6.0, 7.0], index=idx)
    table = assemble_basin_package_table(_valid_forcing(idx), q, area_km2=3.6)
    np.testing.assert_array_equal(table["qobs_m3s"].to_numpy(), q.to_numpy())


# ---------------------------------------------------------------------------
# 4. Correct m3/s -> mm/h conversion for a known area and known flows
# ---------------------------------------------------------------------------


def test_hand_calculated_conversion_via_lead_target():
    idx = _hourly_index(5)
    # area=5.0: hand: 10.0 * 3.6 / 5.0 = 7.2 mm/h
    q = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0], index=idx)
    table = assemble_basin_package_table(_valid_forcing(idx), q, area_km2=5.0)
    assert table["qobs_mm_per_h_lead01"].iloc[0] == pytest.approx(7.2, rel=1e-12)


# ---------------------------------------------------------------------------
# 5. Future-target lead direction for leads 1/3/6/12 (asymmetric ramp)
# 6. Final lead-tail NaN counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lead_hours,column", list(zip([1, 3, 6, 12], EXPECTED_LEAD_COLUMNS)))
def test_lead_direction_and_trailing_nan_count(lead_hours, column):
    n = 20
    idx = _hourly_index(n)
    # area=3.6 -> mm/h numerically equals the m3/s ramp value.
    q = pd.Series(np.arange(n, dtype=np.float64), index=idx)
    table = assemble_basin_package_table(_valid_forcing(idx), q, area_km2=3.6)

    valid_len = n - lead_hours
    expected_valid = np.arange(lead_hours, n, dtype=np.float64)
    np.testing.assert_allclose(
        table[column].to_numpy()[:valid_len], expected_valid, rtol=1e-15
    )
    trailing = table[column].to_numpy()[valid_len:]
    assert len(trailing) == lead_hours
    assert np.all(np.isnan(trailing))
    assert not np.isnan(table[column].to_numpy()[valid_len - 1])


def test_reversed_lead_direction_would_be_caught():
    # Catches the common incorrect implementation lead_target[t] = qobs[t - lead].
    idx = _hourly_index(10)
    q = pd.Series(np.arange(0.0, 100.0, 10.0), index=idx)  # area=3.6 -> mm/h == value
    table = assemble_basin_package_table(_valid_forcing(idx), q, area_km2=3.6)
    correct = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, np.nan]
    reversed_wrong = [np.nan, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    actual = table["qobs_mm_per_h_lead01"].to_numpy()
    np.testing.assert_allclose(actual, correct, rtol=1e-15, equal_nan=True)
    assert not np.allclose(
        np.nan_to_num(actual, nan=-999.0), np.nan_to_num(reversed_wrong, nan=-999.0)
    )


# ---------------------------------------------------------------------------
# 7. Source qobs NaN propagates to the correct earlier issue timestamp
# ---------------------------------------------------------------------------


def test_qobs_nan_propagates_to_correct_earlier_issue_time():
    idx = _hourly_index(8)
    q = pd.Series([0.0, 1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0], index=idx)
    table = assemble_basin_package_table(_valid_forcing(idx), q, area_km2=3.6)
    lead3 = table["qobs_mm_per_h_lead03"]
    # source NaN sits at position 3 -> target[3 - 3] = target[0] must be NaN.
    assert math.isnan(lead3.iloc[0])
    non_boundary = lead3.iloc[1 : len(q) - 3]
    assert not non_boundary.isna().any()


# ---------------------------------------------------------------------------
# 8. Forcing NaNs preserved rather than filled
# ---------------------------------------------------------------------------


def test_forcing_nans_preserved_not_filled():
    idx = _hourly_index(6)
    mrms = np.array([0.0, np.nan, 2.0, 3.0, np.nan, 5.0])
    table = assemble_basin_package_table(
        _valid_forcing(idx, mrms_qpe_1h_mm=mrms), _valid_qobs(idx), area_km2=3.6
    )
    np.testing.assert_array_equal(table["mrms_qpe_1h_mm"].to_numpy(), mrms)


# ---------------------------------------------------------------------------
# 9. Gap flags preserved exactly
# ---------------------------------------------------------------------------


def test_gap_flags_preserved_exactly():
    idx = _hourly_index(6)
    mrms_gap = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    rtma_gap = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    table = assemble_basin_package_table(
        _valid_forcing(idx, mrms_qpe_1h_mm_gap=mrms_gap, rtma_gap=rtma_gap),
        _valid_qobs(idx),
        area_km2=3.6,
    )
    np.testing.assert_array_equal(table["mrms_qpe_1h_mm_gap"].to_numpy(), mrms_gap)
    np.testing.assert_array_equal(table["rtma_gap"].to_numpy(), rtma_gap)


def test_boolean_gap_flags_accepted():
    idx = _hourly_index(4)
    mrms_gap = np.array([True, False, False, True])
    table = assemble_basin_package_table(
        _valid_forcing(idx, mrms_qpe_1h_mm_gap=mrms_gap), _valid_qobs(idx), area_km2=3.6
    )
    np.testing.assert_array_equal(table["mrms_qpe_1h_mm_gap"].to_numpy(), mrms_gap)


# ---------------------------------------------------------------------------
# 10. Extra / missing dynamic columns rejected
# ---------------------------------------------------------------------------


def test_extra_dynamic_column_rejected():
    idx = _hourly_index(5)
    forcing = _valid_forcing(idx)
    forcing["rtma_sp_Pa"] = np.linspace(90000.0, 91000.0, len(idx))  # Smoke-era extra
    with pytest.raises(PackageAssemblyError, match="unapproved"):
        assemble_basin_package_table(forcing, _valid_qobs(idx), area_km2=3.6)


def test_missing_dynamic_column_rejected():
    idx = _hourly_index(5)
    forcing = _valid_forcing(idx).drop(columns=["rtma_gap"])
    with pytest.raises(PackageAssemblyError, match="missing"):
        assemble_basin_package_table(forcing, _valid_qobs(idx), area_km2=3.6)


def test_non_numeric_dynamic_column_rejected():
    idx = _hourly_index(4)
    forcing = _valid_forcing(idx, rtma_gap=["a", "b", "c", "d"])
    with pytest.raises(PackageAssemblyError):
        assemble_basin_package_table(forcing, _valid_qobs(idx), area_km2=3.6)


# ---------------------------------------------------------------------------
# 11. Duplicate timestamp rejection / 12. Non-hourly timestamp rejection
# ---------------------------------------------------------------------------


def test_duplicate_forcing_timestamps_rejected():
    idx = pd.DatetimeIndex(
        ["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 01:00", "2023-01-01 02:00"]
    )
    forcing = _valid_forcing(idx)
    with pytest.raises(PackageAssemblyError, match="duplicate"):
        assemble_basin_package_table(forcing, _valid_qobs(idx), area_km2=3.6)


def test_non_hourly_forcing_timestamps_rejected():
    idx = pd.date_range("2023-01-01", periods=5, freq="30min")
    forcing = _valid_forcing(idx)
    with pytest.raises(PackageAssemblyError, match="hourly"):
        assemble_basin_package_table(forcing, _valid_qobs(idx), area_km2=3.6)


def test_gap_in_hourly_forcing_index_rejected():
    idx = pd.DatetimeIndex(
        ["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 03:00", "2023-01-01 04:00"]
    )
    forcing = _valid_forcing(idx)
    with pytest.raises(PackageAssemblyError, match="hourly"):
        assemble_basin_package_table(forcing, _valid_qobs(idx), area_km2=3.6)


def test_descending_forcing_timestamps_rejected():
    idx = _hourly_index(5)[::-1]
    forcing = _valid_forcing(idx)
    with pytest.raises(PackageAssemblyError, match="increasing"):
        assemble_basin_package_table(forcing, _valid_qobs(idx), area_km2=3.6)


# ---------------------------------------------------------------------------
# 13. Mismatched forcing/qobs timeline rejected (no silent reindex)
# ---------------------------------------------------------------------------


def test_mismatched_qobs_timeline_rejected():
    idx = _hourly_index(10)
    shifted_idx = _hourly_index(10, start="2023-01-01 01:00")
    q = pd.Series(np.arange(10.0), index=shifted_idx)
    with pytest.raises(PackageAssemblyError, match="index must match"):
        assemble_basin_package_table(_valid_forcing(idx), q, area_km2=3.6)


def test_shorter_qobs_series_rejected():
    idx = _hourly_index(10)
    q = pd.Series(np.arange(5.0), index=_hourly_index(5))
    with pytest.raises(PackageAssemblyError, match="index must match"):
        assemble_basin_package_table(_valid_forcing(idx), q, area_km2=3.6)


# ---------------------------------------------------------------------------
# 14. Zero, negative, NaN, infinite area rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_area", [0.0, -5.0, float("nan"), float("inf"), float("-inf")])
def test_invalid_area_rejected(bad_area):
    idx = _hourly_index(5)
    with pytest.raises(ValueError):
        assemble_basin_package_table(_valid_forcing(idx), _valid_qobs(idx), area_km2=bad_area)


def test_array_like_area_rejected():
    idx = _hourly_index(5)
    with pytest.raises(PackageAssemblyError, match="scalar"):
        assemble_basin_package_table(
            _valid_forcing(idx), _valid_qobs(idx), area_km2=np.array([3.6, 3.6])
        )


# ---------------------------------------------------------------------------
# 15. Negative finite qobs rejection
# ---------------------------------------------------------------------------


def test_negative_finite_qobs_rejected():
    idx = _hourly_index(5)
    q = pd.Series([1.0, -2.0, 3.0, 4.0, 5.0], index=idx)
    with pytest.raises(PackageAssemblyError, match="negative"):
        assemble_basin_package_table(_valid_forcing(idx), q, area_km2=3.6)


def test_negative_infinite_qobs_still_rejected_as_infinite():
    # -inf is neither a permitted NaN nor a "negative finite" value; the
    # underlying units.py layer rejects all infinities, and that failure must
    # surface as this module's own PackageAssemblyError (not a bare ValueError).
    idx = _hourly_index(5)
    q = pd.Series([1.0, float("-inf"), 3.0, 4.0, 5.0], index=idx)
    with pytest.raises(PackageAssemblyError):
        assemble_basin_package_table(_valid_forcing(idx), q, area_km2=3.6)


def test_positive_infinite_qobs_rejected_as_package_assembly_error():
    idx = _hourly_index(5)
    q = pd.Series([1.0, float("inf"), 3.0, 4.0, 5.0], index=idx)
    with pytest.raises(PackageAssemblyError):
        assemble_basin_package_table(_valid_forcing(idx), q, area_km2=3.6)


# ---------------------------------------------------------------------------
# qobs dtype: numeric (non-boolean) required
# ---------------------------------------------------------------------------


def test_object_dtype_qobs_rejected():
    idx = _hourly_index(4)
    q = pd.Series(["a", "b", "c", "d"], index=idx, name="qobs_m3s")
    with pytest.raises(PackageAssemblyError, match="numeric"):
        assemble_basin_package_table(_valid_forcing(idx), q, area_km2=3.6)


def test_boolean_dtype_qobs_rejected():
    idx = _hourly_index(4)
    q = pd.Series([True, False, True, False], index=idx, name="qobs_m3s")
    with pytest.raises(PackageAssemblyError, match="numeric"):
        assemble_basin_package_table(_valid_forcing(idx), q, area_km2=3.6)


# ---------------------------------------------------------------------------
# expected_index: optional exact canonical timeline enforcement
# ---------------------------------------------------------------------------


def test_expected_index_matching_short_index_accepted():
    idx = _hourly_index(6)
    table = assemble_basin_package_table(
        _valid_forcing(idx), _valid_qobs(idx), area_km2=3.6, expected_index=idx
    )
    assert list(table.index) == list(idx)


def test_expected_index_wrong_start_same_length_rejected():
    idx = _hourly_index(6)
    expected = _hourly_index(6, start="2023-01-02")
    with pytest.raises(PackageAssemblyError, match="expected_index"):
        assemble_basin_package_table(
            _valid_forcing(idx), _valid_qobs(idx), area_km2=3.6, expected_index=expected
        )


def test_expected_index_wrong_length_rejected():
    idx = _hourly_index(6)
    expected = _hourly_index(10)
    with pytest.raises(PackageAssemblyError, match="expected_index"):
        assemble_basin_package_table(
            _valid_forcing(idx), _valid_qobs(idx), area_km2=3.6, expected_index=expected
        )


def test_structurally_valid_forcing_still_rejected_against_mismatched_expected_index():
    # forcing passes every general hourly/monotonic/duplicate-free structural
    # check on its own, but must still be rejected once an expected_index
    # covering a different period is supplied -- no reindexing/repair.
    idx = _hourly_index(24, start="2023-06-01")
    expected = _hourly_index(24, start="2023-01-01")
    with pytest.raises(PackageAssemblyError, match="expected_index"):
        assemble_basin_package_table(
            _valid_forcing(idx), _valid_qobs(idx), area_km2=3.6, expected_index=expected
        )


# ---------------------------------------------------------------------------
# Timezone-naive enforcement (never silently converted)
# ---------------------------------------------------------------------------


def test_tz_aware_forcing_index_rejected():
    idx = _hourly_index(5).tz_localize("UTC")
    with pytest.raises(PackageAssemblyError, match="timezone-naive"):
        assemble_basin_package_table(_valid_forcing(idx), _valid_qobs(idx), area_km2=3.6)


def test_tz_aware_expected_index_rejected():
    idx = _hourly_index(5)
    expected = idx.tz_localize("UTC")
    with pytest.raises(PackageAssemblyError, match="timezone-naive"):
        assemble_basin_package_table(
            _valid_forcing(idx), _valid_qobs(idx), area_km2=3.6, expected_index=expected
        )


# ---------------------------------------------------------------------------
# Gap-flag binary contract: boolean, or numeric containing only finite 0/1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gap_column", ["mrms_qpe_1h_mm_gap", "rtma_gap"])
@pytest.mark.parametrize(
    "values",
    [
        [True, False, False, True],
        [0.0, 1.0, 0.0, 1.0],
        [0, 1, 0, 1],
    ],
    ids=["boolean", "numeric_float_0_1", "numeric_int_0_1"],
)
def test_gap_flag_valid_categories_accepted(gap_column, values):
    idx = _hourly_index(4)
    forcing = _valid_forcing(idx, **{gap_column: values})
    table = assemble_basin_package_table(forcing, _valid_qobs(idx), area_km2=3.6)
    np.testing.assert_array_equal(table[gap_column].to_numpy(), np.array(values))


@pytest.mark.parametrize("gap_column", ["mrms_qpe_1h_mm_gap", "rtma_gap"])
@pytest.mark.parametrize(
    "values",
    [
        [0.0, -1.0, 0.0, 1.0],
        [0.0, 2.0, 0.0, 1.0],
        [0.0, 0.5, 0.0, 1.0],
        [0.0, float("nan"), 0.0, 1.0],
        [0.0, float("inf"), 0.0, 1.0],
        [0.0, float("-inf"), 0.0, 1.0],
    ],
    ids=["negative", "greater_than_one", "fractional", "nan", "positive_inf", "negative_inf"],
)
def test_gap_flag_invalid_categories_rejected(gap_column, values):
    idx = _hourly_index(4)
    forcing = _valid_forcing(idx, **{gap_column: values})
    with pytest.raises(PackageAssemblyError, match="gap-flag"):
        assemble_basin_package_table(forcing, _valid_qobs(idx), area_km2=3.6)


# ---------------------------------------------------------------------------
# Policy validation: delegates to policy.py, never duplicates its invariants
# ---------------------------------------------------------------------------


def test_committed_policy_succeeds():
    idx = _hourly_index(6)
    policy = load_stage1_baseline_policy(REPO_ROOT_POLICY)
    table = assemble_basin_package_table(
        _valid_forcing(idx), _valid_qobs(idx), area_km2=3.6, policy=policy
    )
    assert list(table.columns) == EXPECTED_COLUMN_ORDER


def test_policy_with_mutated_dynamic_inputs_rejected():
    idx = _hourly_index(6)
    policy = copy.deepcopy(load_stage1_baseline_policy(REPO_ROOT_POLICY))
    policy["dynamic_inputs"] = policy["dynamic_inputs"] + ["rtma_sp_Pa"]
    with pytest.raises(PackageAssemblyError, match="policy"):
        assemble_basin_package_table(
            _valid_forcing(idx), _valid_qobs(idx), area_km2=3.6, policy=policy
        )


def test_policy_with_mutated_leads_hours_rejected():
    idx = _hourly_index(6)
    policy = copy.deepcopy(load_stage1_baseline_policy(REPO_ROOT_POLICY))
    policy["target"]["leads_hours"] = [1, 3, 6, 24]
    with pytest.raises(PackageAssemblyError, match="policy"):
        assemble_basin_package_table(
            _valid_forcing(idx), _valid_qobs(idx), area_km2=3.6, policy=policy
        )


# ---------------------------------------------------------------------------
# 16. Deterministic output for identical inputs
# ---------------------------------------------------------------------------


def test_deterministic_output_for_identical_inputs():
    idx = _hourly_index(15)
    forcing = _valid_forcing(idx)
    q = _valid_qobs(idx)
    table1 = assemble_basin_package_table(forcing.copy(), q.copy(), area_km2=42.0)
    table2 = assemble_basin_package_table(forcing.copy(), q.copy(), area_km2=42.0)
    pd.testing.assert_frame_equal(table1, table2)


# ---------------------------------------------------------------------------
# qobs accepted as a DataFrame too
# ---------------------------------------------------------------------------


def test_qobs_accepted_as_dataframe_with_source_column():
    idx = _hourly_index(6)
    q_df = pd.DataFrame({"qobs_m3s": np.arange(6.0), "other_col": np.zeros(6)}, index=idx)
    table = assemble_basin_package_table(_valid_forcing(idx), q_df, area_km2=3.6)
    np.testing.assert_array_equal(table["qobs_m3s"].to_numpy(), np.arange(6.0))


def test_qobs_dataframe_missing_source_column_rejected():
    idx = _hourly_index(6)
    q_df = pd.DataFrame({"other_col": np.zeros(6)}, index=idx)
    with pytest.raises(PackageAssemblyError, match="qobs_m3s"):
        assemble_basin_package_table(_valid_forcing(idx), q_df, area_km2=3.6)

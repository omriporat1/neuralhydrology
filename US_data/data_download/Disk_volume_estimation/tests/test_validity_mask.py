"""Tests for src/baseline/validity_mask.py (Milestone 2K-G-I I-D1).

Expected vectors are hand-verified against the binding definition
sample_valid(t) = history_valid_seq(t) & target_boundary_valid_lead(t) on
small synthetic timelines, never re-derived from the function under test.
"""
import numpy as np
import pandas as pd
import pytest

from src.baseline.validity_mask import (
    ValidityMaskError,
    bad_hour_mask_from_timestamps,
    compute_boundary_valid,
    compute_history_valid,
    compute_validity_mask,
)

HOURLY = "h"


def _hourly_index(n, start="2023-01-01"):
    return pd.date_range(start, periods=n, freq=HOURLY)


def _brute_force_history_valid(index, bad_hour_mask, seq_length):
    n = len(index)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        start = i - seq_length + 1
        if start < 0:
            continue
        window_ok = True
        for j in range(start, i + 1):
            if bad_hour_mask[j]:
                window_ok = False
                break
        valid[i] = window_ok
    return valid


# ---------------------------------------------------------------------------
# history_valid_seq: no-gap case, warm-up loss, exact inclusive window
# ---------------------------------------------------------------------------


def test_no_gap_case_only_warmup_excluded():
    idx = _hourly_index(10)
    bad = np.zeros(10, dtype=bool)
    hv = compute_history_valid(idx, bad, seq_length=3)
    expected = [False, False, True, True, True, True, True, True, True, True]
    np.testing.assert_array_equal(hv, expected)
    assert hv.sum() == 10 - (3 - 1)


def test_warmup_loss_is_exactly_seq_length_minus_one():
    idx = _hourly_index(20)
    bad = np.zeros(20, dtype=bool)
    for seq_length in (1, 2, 5, 12, 20):
        hv = compute_history_valid(idx, bad, seq_length=seq_length)
        n_invalid = (~hv).sum()
        assert n_invalid == seq_length - 1
        # the first (seq_length - 1) positions are exactly the invalid ones
        np.testing.assert_array_equal(hv[: seq_length - 1], False)
        np.testing.assert_array_equal(hv[seq_length - 1 :], True)


def test_bad_hour_at_start_invalidates_only_windows_containing_it():
    idx = _hourly_index(10)
    bad = np.zeros(10, dtype=bool)
    bad[0] = True
    hv = compute_history_valid(idx, bad, seq_length=3)
    # windows: t=2 -> [0,1,2] contains bad -> False
    #          t=3 -> [1,2,3] clean -> True (once warmup satisfied)
    expected = [False, False, False, True, True, True, True, True, True, True]
    np.testing.assert_array_equal(hv, expected)


def test_bad_hour_in_middle_invalidates_only_windows_containing_it():
    idx = _hourly_index(10)
    bad = np.zeros(10, dtype=bool)
    bad[5] = True
    hv = compute_history_valid(idx, bad, seq_length=3)
    # window [i-2,i-1,i] contains index 5 for i in {5,6,7}
    expected = [False, False, True, True, True, False, False, False, True, True]
    np.testing.assert_array_equal(hv, expected)


def test_bad_hour_at_end_invalidates_only_windows_containing_it():
    idx = _hourly_index(10)
    bad = np.zeros(10, dtype=bool)
    bad[9] = True
    hv = compute_history_valid(idx, bad, seq_length=3)
    # window [7,8,9] and [8,9,-] -> for i=9: [7,8,9] contains bad
    expected = [False, False, True, True, True, True, True, True, True, False]
    np.testing.assert_array_equal(hv, expected)


def test_exact_inclusive_window_boundaries_seq_length_one():
    # seq_length=1 -> window is just [t,t]; bad_hour_mask should equal
    # history_valid exactly (no warmup loss beyond position 0 requirement).
    idx = _hourly_index(6)
    bad = np.array([False, True, False, False, True, False])
    hv = compute_history_valid(idx, bad, seq_length=1)
    expected = ~bad
    np.testing.assert_array_equal(hv, expected)


# ---------------------------------------------------------------------------
# target_boundary_valid_lead: exact loss of `lead` issue times; history
# independence from lead; gap-at-t+lead does not invalidate
# ---------------------------------------------------------------------------


def test_boundary_loss_is_exactly_lead_issue_times():
    idx = _hourly_index(20)
    for lead in (1, 3, 6, 12):
        bv = compute_boundary_valid(idx, lead_hours=lead)
        assert (~bv).sum() == lead
        np.testing.assert_array_equal(bv[-lead:], False)
        np.testing.assert_array_equal(bv[: 20 - lead], True)


def test_lead_changes_only_boundary_validity_not_history_validity():
    idx = _hourly_index(20)
    bad = np.zeros(20, dtype=bool)
    bad[10] = True
    hv_ref = compute_history_valid(idx, bad, seq_length=5)
    for lead in (1, 3, 6, 12):
        hv = compute_history_valid(idx, bad, seq_length=5)
        np.testing.assert_array_equal(hv, hv_ref)
        result = compute_validity_mask(idx, bad, seq_length=5, lead_hours=lead)
        np.testing.assert_array_equal(result.history_valid, hv_ref)


def test_gap_at_target_hour_alone_does_not_invalidate_sample():
    idx = _hourly_index(20)
    lead = 6
    seq_length = 4
    bad = np.zeros(20, dtype=bool)
    # Put a bad hour only at position t+lead for some issue time t whose
    # history window is otherwise clean.
    t = 5
    bad[t + lead] = True
    result = compute_validity_mask(idx, bad, seq_length=seq_length, lead_hours=lead)
    assert result.history_valid[t]
    assert result.boundary_valid[t]
    assert result.combined_valid[t]


# ---------------------------------------------------------------------------
# Multiple / consecutive gap hours
# ---------------------------------------------------------------------------


def test_multiple_and_consecutive_gap_hours():
    idx = _hourly_index(15)
    bad = np.zeros(15, dtype=bool)
    bad[3] = True
    bad[7] = True
    bad[8] = True  # consecutive pair
    hv = compute_history_valid(idx, bad, seq_length=4)
    brute = _brute_force_history_valid(idx, bad, seq_length=4)
    np.testing.assert_array_equal(hv, brute)
    # sanity: windows touching {3}, {7,8} should be invalid, others valid
    # (post warmup)
    assert not hv[3]
    assert not hv[6]  # window [3,4,5,6] contains bad at 3
    assert not hv[7]
    assert not hv[8]
    assert not hv[9]  # window [6,7,8,9] contains bad at 7,8
    assert not hv[10]  # window [7,8,9,10] contains bad at 7,8
    assert not hv[11]  # window [8,9,10,11] contains bad at 8
    assert hv[12]  # window [9,10,11,12] clean


# ---------------------------------------------------------------------------
# MRMS-only vs combined MRMS+RTMA masks
# ---------------------------------------------------------------------------


def test_mrms_only_vs_combined_mrms_rtma_masks():
    idx = _hourly_index(15)
    mrms_bad = np.zeros(15, dtype=bool)
    mrms_bad[4] = True
    rtma_bad = np.zeros(15, dtype=bool)
    rtma_bad[10] = True
    combined_bad = mrms_bad | rtma_bad

    mrms_result = compute_validity_mask(idx, mrms_bad, seq_length=3, lead_hours=1)
    combined_result = compute_validity_mask(idx, combined_bad, seq_length=3, lead_hours=1)

    # Combined mask can only exclude at least as many samples as MRMS-only.
    assert combined_result.n_history_valid <= mrms_result.n_history_valid
    # Position 10 area differs between the two.
    assert mrms_result.history_valid[10]
    assert not combined_result.history_valid[10]


# ---------------------------------------------------------------------------
# All 4 seq_lengths x all 4 leads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_length", [12, 24, 48, 72])
@pytest.mark.parametrize("lead", [1, 3, 6, 12])
def test_all_seq_length_lead_combinations(seq_length, lead):
    n = 200
    idx = _hourly_index(n)
    rng = np.random.default_rng(seq_length * 100 + lead)
    bad = rng.random(n) < 0.02
    result = compute_validity_mask(idx, bad, seq_length=seq_length, lead_hours=lead)

    brute_hv = _brute_force_history_valid(idx, bad, seq_length)
    np.testing.assert_array_equal(result.history_valid, brute_hv)

    expected_boundary_invalid = lead
    assert (~result.boundary_valid).sum() == expected_boundary_invalid
    np.testing.assert_array_equal(
        result.combined_valid, result.history_valid & result.boundary_valid
    )
    assert result.seq_length == seq_length
    assert result.lead_hours == lead
    assert result.n_timeline == n


# ---------------------------------------------------------------------------
# Brute-force vs efficient implementation, dedicated comparison test
# ---------------------------------------------------------------------------


def test_efficient_matches_brute_force_reference():
    n = 500
    idx = _hourly_index(n)
    rng = np.random.default_rng(12345)
    bad = rng.random(n) < 0.05
    for seq_length in (1, 2, 12, 24, 48, 72, 100):
        efficient = compute_history_valid(idx, bad, seq_length=seq_length)
        brute = _brute_force_history_valid(idx, bad, seq_length=seq_length)
        np.testing.assert_array_equal(efficient, brute)


# ---------------------------------------------------------------------------
# Deterministic counts
# ---------------------------------------------------------------------------


def test_counts_are_deterministic_across_repeated_calls():
    idx = _hourly_index(100)
    rng = np.random.default_rng(7)
    bad = rng.random(100) < 0.03
    r1 = compute_validity_mask(idx, bad, seq_length=24, lead_hours=6)
    r2 = compute_validity_mask(idx, bad, seq_length=24, lead_hours=6)
    assert r1.n_history_valid == r2.n_history_valid
    assert r1.n_boundary_valid == r2.n_boundary_valid
    assert r1.n_combined_valid == r2.n_combined_valid
    np.testing.assert_array_equal(r1.combined_valid, r2.combined_valid)


# ---------------------------------------------------------------------------
# bad_hour_mask_from_timestamps: conversion + strictness policy
# ---------------------------------------------------------------------------


def test_bad_hour_mask_from_timestamps_builds_correct_vector():
    idx = _hourly_index(10)
    bad_ts = [idx[2], idx[7]]
    mask = bad_hour_mask_from_timestamps(idx, bad_ts)
    expected = np.zeros(10, dtype=bool)
    expected[[2, 7]] = True
    np.testing.assert_array_equal(mask, expected)


def test_bad_hour_mask_from_timestamps_out_of_range_errors_by_default():
    idx = _hourly_index(10)
    out_of_range_ts = idx[0] - pd.Timedelta(hours=5)
    with pytest.raises(ValidityMaskError):
        bad_hour_mask_from_timestamps(idx, [idx[1], out_of_range_ts])


def test_bad_hour_mask_from_timestamps_out_of_range_ignored_when_explicit():
    idx = _hourly_index(10)
    out_of_range_ts = idx[0] - pd.Timedelta(hours=5)
    mask = bad_hour_mask_from_timestamps(
        idx, [idx[1], out_of_range_ts], on_out_of_range="ignore"
    )
    expected = np.zeros(10, dtype=bool)
    expected[1] = True
    np.testing.assert_array_equal(mask, expected)


def test_bad_hour_mask_from_timestamps_invalid_strictness_policy_rejected():
    idx = _hourly_index(10)
    with pytest.raises(ValidityMaskError):
        bad_hour_mask_from_timestamps(idx, [idx[1]], on_out_of_range="bogus")


# ---------------------------------------------------------------------------
# Rejection: irregular / duplicate / descending / non-hourly timeline
# ---------------------------------------------------------------------------


def test_non_datetimeindex_rejected():
    bad = np.zeros(3, dtype=bool)
    with pytest.raises(ValidityMaskError):
        compute_history_valid([0, 1, 2], bad, seq_length=1)


def test_duplicate_timestamps_rejected():
    idx = pd.DatetimeIndex(
        ["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 01:00", "2023-01-01 02:00"]
    )
    bad = np.zeros(4, dtype=bool)
    with pytest.raises(ValidityMaskError):
        compute_history_valid(idx, bad, seq_length=1)


def test_descending_timestamps_rejected():
    idx = _hourly_index(5)[::-1]
    bad = np.zeros(5, dtype=bool)
    with pytest.raises(ValidityMaskError):
        compute_history_valid(idx, bad, seq_length=1)


def test_irregular_gap_in_hourly_index_rejected():
    idx = pd.DatetimeIndex(
        ["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 03:00", "2023-01-01 04:00"]
    )
    bad = np.zeros(4, dtype=bool)
    with pytest.raises(ValidityMaskError):
        compute_history_valid(idx, bad, seq_length=1)


def test_non_hourly_frequency_rejected():
    idx = pd.date_range("2023-01-01", periods=5, freq="30min")
    bad = np.zeros(5, dtype=bool)
    with pytest.raises(ValidityMaskError):
        compute_history_valid(idx, bad, seq_length=1)


def test_daily_frequency_rejected():
    idx = pd.date_range("2023-01-01", periods=5, freq="D")
    bad = np.zeros(5, dtype=bool)
    with pytest.raises(ValidityMaskError):
        compute_history_valid(idx, bad, seq_length=1)


def test_descending_rejected_in_compute_boundary_valid_and_validity_mask():
    idx = _hourly_index(5)[::-1]
    bad = np.zeros(5, dtype=bool)
    with pytest.raises(ValidityMaskError):
        compute_boundary_valid(idx, lead_hours=1)
    with pytest.raises(ValidityMaskError):
        compute_validity_mask(idx, bad, seq_length=1, lead_hours=1)


# ---------------------------------------------------------------------------
# Invalid scalar arguments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_seq_length", [0, -1, -12])
def test_nonpositive_seq_length_rejected(bad_seq_length):
    idx = _hourly_index(10)
    bad = np.zeros(10, dtype=bool)
    with pytest.raises(ValidityMaskError):
        compute_history_valid(idx, bad, seq_length=bad_seq_length)


def test_non_integer_seq_length_rejected():
    idx = _hourly_index(10)
    bad = np.zeros(10, dtype=bool)
    with pytest.raises(ValidityMaskError):
        compute_history_valid(idx, bad, seq_length=3.5)
    with pytest.raises(ValidityMaskError):
        compute_history_valid(idx, bad, seq_length=True)


@pytest.mark.parametrize("bad_lead", [0, -1, -6])
def test_nonpositive_lead_rejected(bad_lead):
    idx = _hourly_index(10)
    with pytest.raises(ValidityMaskError):
        compute_boundary_valid(idx, lead_hours=bad_lead)


def test_non_integer_lead_rejected():
    idx = _hourly_index(10)
    with pytest.raises(ValidityMaskError):
        compute_boundary_valid(idx, lead_hours=6.0)
    with pytest.raises(ValidityMaskError):
        compute_boundary_valid(idx, lead_hours=True)


# ---------------------------------------------------------------------------
# Boolean-vector length mismatch
# ---------------------------------------------------------------------------


def test_bad_hour_mask_length_mismatch_rejected():
    idx = _hourly_index(10)
    bad = np.zeros(9, dtype=bool)
    with pytest.raises(ValidityMaskError):
        compute_history_valid(idx, bad, seq_length=3)
    with pytest.raises(ValidityMaskError):
        compute_validity_mask(idx, bad, seq_length=3, lead_hours=1)


def test_bad_hour_mask_too_long_rejected():
    idx = _hourly_index(10)
    bad = np.zeros(11, dtype=bool)
    with pytest.raises(ValidityMaskError):
        compute_history_valid(idx, bad, seq_length=3)

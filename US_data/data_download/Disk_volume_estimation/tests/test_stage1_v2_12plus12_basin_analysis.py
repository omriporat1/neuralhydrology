"""Tests for src/baseline/stage1_v2_12plus12_basin_analysis.py (RD1-C3).

All fixtures are small and hand-checkable. Percentile expectations are
derived independently of the module under test (via ``numpy.percentile``
reasoning identical to the already-qualified
``percentile_diagnostics.compute_percentile_table``), never by calling the
function under test with different arguments.
"""
import math

import numpy as np
import pandas as pd
import pytest

from src.baseline.percentile_diagnostics import compute_percentile_table
from src.baseline.stage1_v2_12plus12_basin_analysis import (
    NUMERICAL_TIE_TOLERANCE,
    BasinAnalysisContractError,
    analyze_basin_distribution,
    analyze_paired_basin_difference,
    compute_ecdf_table,
)

EXPECTED_120 = [f"b{i:03d}" for i in range(120)]


def _rows(basin_ids, values, extra=None):
    rows = [{"basin_id": b, "nse": v} for b, v in zip(basin_ids, values)]
    if extra:
        for row, ex in zip(rows, extra):
            row.update(ex)
    return rows


# ---------------------------------------------------------------------------
# 1-2: frozen seven-quantile core + IQR, reuse of compute_percentile_table
# ---------------------------------------------------------------------------


def test_frozen_quantile_core_matches_percentile_diagnostics_reuse():
    values = np.arange(0, 101, dtype=np.float64)  # 101 basins, 0..100
    basin_ids = [f"b{i:03d}" for i in range(101)]
    result = analyze_basin_distribution("cfgA", basin_ids, _rows(basin_ids, values))

    reference = compute_percentile_table(values, metric_name="nse")
    core = result.frozen_core
    assert core.q1 == pytest.approx(reference.percentiles["p1"])
    assert core.q5 == pytest.approx(reference.percentiles["p5"])
    assert core.q25 == pytest.approx(reference.percentiles["p25"])
    assert core.q50 == pytest.approx(reference.percentiles["p50"])
    assert core.q75 == pytest.approx(reference.percentiles["p75"])
    assert core.q95 == pytest.approx(reference.percentiles["p95"])
    assert core.q99 == pytest.approx(reference.percentiles["p99"])
    assert core.iqr == pytest.approx(reference.percentiles["p75"] - reference.percentiles["p25"])
    # for 0..100 linear interpolation, percentile p sits exactly at value p.
    assert core.q50 == pytest.approx(50.0, abs=1e-9)
    assert result.percentile_table is not None
    # result reuses the exact same PercentileTable object semantics (p10/p90 retained)
    assert "p10" in result.percentile_table.percentiles
    assert "p90" in result.percentile_table.percentiles


# ---------------------------------------------------------------------------
# 3: total/finite/nonfinite counting with NaN and infinities
# ---------------------------------------------------------------------------


def test_finite_nonfinite_counting_with_nan_and_inf():
    basin_ids = ["a", "b", "c", "d", "e", "f"]
    values = [1.0, 2.0, np.nan, np.inf, -np.inf, 3.0]
    result = analyze_basin_distribution("cfgA", basin_ids, _rows(basin_ids, values))
    assert result.n_total_basins == 6
    assert result.n_finite_basins == 3
    assert result.n_nonfinite_basins == 3
    assert result.frozen_core.q50 == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 4-5: ECDF empty / ordering / duplicate jumps
# ---------------------------------------------------------------------------


def test_ecdf_stable_empty_when_no_finite_values():
    table = compute_ecdf_table([np.nan, np.inf, -np.inf])
    assert table.n_finite == 0
    assert list(table.value) == []
    assert list(table.count) == []
    assert list(table.cumulative_count) == []
    assert list(table.cumulative_fraction) == []


def test_ecdf_ordering_and_duplicate_value_jump():
    # values: 1, 1, 2, 3, 3, 3 (6 finite) -> unique 1,2,3 with counts 2,1,3
    values = [3.0, 1.0, 3.0, 2.0, 1.0, 3.0]
    table = compute_ecdf_table(values)
    assert list(table.value) == [1.0, 2.0, 3.0]
    assert list(table.count) == [2, 1, 3]
    assert list(table.cumulative_count) == [2, 3, 6]
    assert table.cumulative_fraction == pytest.approx([2 / 6, 3 / 6, 6 / 6])
    assert table.n_finite == 6


# ---------------------------------------------------------------------------
# 6-9: basin-identity and metric-value contract failures
# ---------------------------------------------------------------------------


def test_invalid_blank_basin_id_rejected():
    with pytest.raises(BasinAnalysisContractError):
        analyze_basin_distribution("cfgA", ["a", ""], _rows(["a", ""], [0.1, 0.2]))


def test_non_string_basin_id_rejected():
    with pytest.raises(BasinAnalysisContractError):
        analyze_basin_distribution("cfgA", ["a", "b"], _rows(["a", 123], [0.1, 0.2]))


def test_duplicate_basin_id_rejected():
    with pytest.raises(BasinAnalysisContractError):
        analyze_basin_distribution("cfgA", ["a", "b"], _rows(["a", "a"], [0.1, 0.2]))


def test_missing_expected_basin_id_rejected():
    with pytest.raises(BasinAnalysisContractError):
        analyze_basin_distribution("cfgA", ["a", "b", "c"], _rows(["a", "b"], [0.1, 0.2]))


def test_unexpected_basin_id_rejected():
    with pytest.raises(BasinAnalysisContractError):
        analyze_basin_distribution("cfgA", ["a", "b"], _rows(["a", "b", "z"], [0.1, 0.2, 0.3]))


def test_malformed_bool_nse_rejected():
    with pytest.raises(BasinAnalysisContractError):
        analyze_basin_distribution("cfgA", ["a", "b"], _rows(["a", "b"], [0.1, True]))


def test_malformed_numeric_looking_string_nse_rejected():
    with pytest.raises(BasinAnalysisContractError):
        analyze_basin_distribution("cfgA", ["a", "b"], _rows(["a", "b"], [0.1, "0.2"]))


# ---------------------------------------------------------------------------
# 10: row-order independence
# ---------------------------------------------------------------------------


def test_row_order_independence():
    ids_a = ["a", "b", "c"]
    values_a = [0.1, 0.2, 0.3]
    ids_b = ["c", "a", "b"]
    values_b = [0.3, 0.1, 0.2]
    result_a = analyze_basin_distribution("cfg", ["a", "b", "c"], _rows(ids_a, values_a))
    result_b = analyze_basin_distribution("cfg", ["a", "b", "c"], _rows(ids_b, values_b))
    pd.testing.assert_frame_equal(
        result_a.per_basin.reset_index(drop=True), result_b.per_basin.reset_index(drop=True)
    )
    assert list(result_a.per_basin["basin_id"]) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# 11-12: paired alignment by ID + fail-closed identity mismatch
# ---------------------------------------------------------------------------


def test_paired_alignment_by_basin_id_not_row_position():
    expected = ["a", "b", "c"]
    candidate = analyze_basin_distribution(
        "cand", expected, _rows(["c", "a", "b"], [0.9, 0.1, 0.5])
    )
    reference = analyze_basin_distribution(
        "ref", expected, _rows(["a", "b", "c"], [0.1, 0.4, 0.9])
    )
    paired = analyze_paired_basin_difference(candidate, reference)
    row_a = paired.per_basin[paired.per_basin["basin_id"] == "a"].iloc[0]
    assert row_a["nse_candidate"] == pytest.approx(0.1)
    assert row_a["nse_reference"] == pytest.approx(0.1)
    assert row_a["delta"] == pytest.approx(0.0)


def test_paired_identity_mismatch_fails_closed():
    candidate = analyze_basin_distribution("cand", ["a", "b"], _rows(["a", "b"], [0.1, 0.2]))
    reference = analyze_basin_distribution("ref", ["a", "c"], _rows(["a", "c"], [0.1, 0.2]))
    with pytest.raises(BasinAnalysisContractError):
        analyze_paired_basin_difference(candidate, reference)


# ---------------------------------------------------------------------------
# 13: paired finite-mask accounting when one side is nonfinite
# ---------------------------------------------------------------------------


def test_paired_finite_accounting_one_side_nonfinite():
    expected = ["a", "b", "c"]
    candidate = analyze_basin_distribution("cand", expected, _rows(expected, [0.1, np.nan, 0.3]))
    reference = analyze_basin_distribution("ref", expected, _rows(expected, [0.05, 0.2, np.nan]))
    paired = analyze_paired_basin_difference(candidate, reference)
    assert paired.n_total_basins == 3
    assert paired.n_candidate_finite == 2
    assert paired.n_reference_finite == 2
    assert paired.n_pairwise_finite == 1  # only "a"
    assert paired.n_pairwise_nonfinite == 2
    row_b = paired.per_basin[paired.per_basin["basin_id"] == "b"].iloc[0]
    row_c = paired.per_basin[paired.per_basin["basin_id"] == "c"].iloc[0]
    assert row_b["classification"] == "nonfinite_unavailable"
    assert math.isnan(row_b["delta"])
    assert row_c["classification"] == "nonfinite_unavailable"
    assert math.isnan(row_c["delta"])


# ---------------------------------------------------------------------------
# 14-16: classification boundaries at +-1e-12, forbidden 0.01 convention
# ---------------------------------------------------------------------------


def test_classification_improved_worse_tied():
    expected = ["a", "b", "c"]
    candidate = analyze_basin_distribution("cand", expected, _rows(expected, [0.5, 0.2, 0.3]))
    reference = analyze_basin_distribution("ref", expected, _rows(expected, [0.3, 0.4, 0.3]))
    paired = analyze_paired_basin_difference(candidate, reference)
    by_id = paired.per_basin.set_index("basin_id")
    assert by_id.loc["a", "classification"] == "improved"
    assert by_id.loc["b", "classification"] == "worse"
    assert by_id.loc["c", "classification"] == "tied"
    assert paired.n_improved == 1
    assert paired.n_worse == 1
    assert paired.n_tied == 1


def test_classification_exact_boundary_at_tolerance():
    # Use a reference of 0.0 so delta == candidate value exactly (no
    # floating-point rounding from an intermediate addition), letting the
    # boundary sit at exactly +-NUMERICAL_TIE_TOLERANCE.
    expected = ["a", "b", "c"]
    candidate_values = [
        2 * NUMERICAL_TIE_TOLERANCE,  # clearly above -> improved
        -2 * NUMERICAL_TIE_TOLERANCE,  # clearly below -> worse
        NUMERICAL_TIE_TOLERANCE / 2,  # inside tolerance -> tied
    ]
    candidate = analyze_basin_distribution("cand", expected, _rows(expected, candidate_values))
    reference = analyze_basin_distribution("ref", expected, _rows(expected, [0.0, 0.0, 0.0]))
    paired = analyze_paired_basin_difference(candidate, reference)
    by_id = paired.per_basin.set_index("basin_id")
    assert by_id.loc["a", "classification"] == "improved"
    assert by_id.loc["b", "classification"] == "worse"
    assert by_id.loc["c", "classification"] == "tied"


def test_small_delta_not_tied_under_forbidden_001_convention():
    # A delta of 0.005 would have been "tied" under the old scratch +-0.01
    # convention, but must be classified "improved" under the frozen
    # 1e-12 numerical-tolerance rule.
    expected = ["a"]
    candidate = analyze_basin_distribution("cand", expected, _rows(expected, [0.505]))
    reference = analyze_basin_distribution("ref", expected, _rows(expected, [0.5]))
    paired = analyze_paired_basin_difference(candidate, reference)
    assert paired.per_basin.iloc[0]["classification"] == "improved"
    assert paired.n_improved == 1
    assert paired.n_tied == 0


# ---------------------------------------------------------------------------
# 17: NaN fractions when no pairwise-finite basins exist
# ---------------------------------------------------------------------------


def test_nan_fractions_when_no_pairwise_finite_basins():
    expected = ["a", "b"]
    candidate = analyze_basin_distribution("cand", expected, _rows(expected, [np.nan, 0.2]))
    reference = analyze_basin_distribution("ref", expected, _rows(expected, [0.1, np.nan]))
    paired = analyze_paired_basin_difference(candidate, reference)
    assert paired.n_pairwise_finite == 0
    assert math.isnan(paired.frac_improved)
    assert math.isnan(paired.frac_worse)
    assert math.isnan(paired.frac_tied)


# ---------------------------------------------------------------------------
# 18: full paired-difference quantiles, IQR, ECDF, per-basin delta rows
# ---------------------------------------------------------------------------


def test_paired_full_quantiles_iqr_ecdf_and_delta_rows():
    expected = [f"b{i:03d}" for i in range(101)]
    candidate_values = np.arange(0, 101, dtype=np.float64)
    reference_values = np.zeros(101, dtype=np.float64)
    candidate = analyze_basin_distribution("cand", expected, _rows(expected, candidate_values))
    reference = analyze_basin_distribution("ref", expected, _rows(expected, reference_values))
    paired = analyze_paired_basin_difference(candidate, reference)

    # delta == candidate values (0..100), so frozen core mirrors the raw case.
    assert paired.frozen_core.q50 == pytest.approx(50.0, abs=1e-9)
    assert paired.frozen_core.iqr == pytest.approx(paired.frozen_core.q75 - paired.frozen_core.q25)
    assert len(paired.per_basin) == 101
    assert set(paired.per_basin.columns) >= {"basin_id", "nse_candidate", "nse_reference", "delta", "classification"}
    # ECDF must cover the same finite support as delta.
    assert paired.ecdf.n_finite == 101
    assert paired.ecdf.value[0] == pytest.approx(0.0)
    assert paired.ecdf.value[-1] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 19: vertical synthetic consumer-contract test
# ---------------------------------------------------------------------------


def test_vertical_consumer_contract_with_producer_shaped_rows():
    """Simulates the future RD1-C4 consumer handing fixed-support-contract-v2
    -shaped per_basin rows (extra columns like the real evaluator emits) to
    the public C3 entry points, with deliberately different row orders for
    candidate vs. reference."""
    expected_basin_ids = [f"gauge_{i:03d}" for i in range(12)]

    rng_candidate_nse = [0.1, 0.9, 0.3, 0.75, np.nan, 0.6, 0.55, 0.2, 0.85, 0.4, 0.65, 0.05]
    candidate_rows = [
        {
            "basin_id": bid,
            "nse": nse,
            "freq": "1h",
            "n_fixed_support_eligible": 8760,
            "n_sim_nonfinite_at_admitted": 0,
        }
        for bid, nse in zip(expected_basin_ids, rng_candidate_nse)
    ]
    # shuffle candidate row order
    candidate_rows_shuffled = candidate_rows[::-1]

    reference_nse = [0.05, 0.85, 0.35, 0.70, 0.5, 0.55, 0.5, 0.25, 0.80, 0.45, np.inf, 0.02]
    reference_rows = [
        {
            "basin_id": bid,
            "nse": nse,
            "freq": "1h",
            "n_fixed_support_eligible": 8760,
            "n_sim_nonfinite_at_admitted": 0,
        }
        for bid, nse in zip(expected_basin_ids, reference_nse)
    ]
    # different shuffle for reference (not reversed the same way)
    import random

    reference_rows_shuffled = reference_rows.copy()
    random.Random(42).shuffle(reference_rows_shuffled)

    candidate_result = analyze_basin_distribution(
        "P2_candidate", expected_basin_ids, pd.DataFrame(candidate_rows_shuffled)
    )
    reference_result = analyze_basin_distribution(
        "R2_reference", expected_basin_ids, pd.DataFrame(reference_rows_shuffled)
    )

    assert candidate_result.configuration_id == "P2_candidate"
    assert reference_result.configuration_id == "R2_reference"
    assert list(candidate_result.per_basin["basin_id"]) == sorted(expected_basin_ids)
    assert list(reference_result.per_basin["basin_id"]) == sorted(expected_basin_ids)
    assert candidate_result.n_total_basins == 12
    assert candidate_result.n_finite_basins == 11
    assert reference_result.n_finite_basins == 11
    # extra producer columns survive untouched
    assert "n_fixed_support_eligible" in candidate_result.per_basin.columns

    paired = analyze_paired_basin_difference(candidate_result, reference_result)
    assert paired.candidate_id == "P2_candidate"
    assert paired.reference_id == "R2_reference"
    assert paired.n_total_basins == 12
    # gauge_004 nonfinite in candidate, gauge_010 nonfinite in reference -> 10 pairwise finite
    assert paired.n_pairwise_finite == 10
    assert paired.n_pairwise_nonfinite == 2
    assert paired.n_improved + paired.n_worse + paired.n_tied == paired.n_pairwise_finite
    assert paired.frac_improved == pytest.approx(paired.n_improved / paired.n_pairwise_finite)
    assert not math.isnan(paired.frozen_core.q50)
    assert paired.ecdf.n_finite == paired.n_pairwise_finite
    by_id = paired.per_basin.set_index("basin_id")
    assert by_id.loc["gauge_004", "classification"] == "nonfinite_unavailable"
    assert by_id.loc["gauge_010", "classification"] == "nonfinite_unavailable"
    # gauge_001: candidate 0.9 vs reference 0.85 -> improved
    assert by_id.loc["gauge_001", "classification"] == "improved"


# ---------------------------------------------------------------------------
# Correction pass, finding 1: recognized missing-value markers (None, pd.NA)
# ---------------------------------------------------------------------------


def test_distribution_none_marker_is_nonfinite_not_zero():
    expected = ["a", "b", "c"]
    result = analyze_basin_distribution("cfg", expected, _rows(expected, [0.1, None, 0.3]))
    assert result.n_total_basins == 3
    assert result.n_finite_basins == 2
    assert result.n_nonfinite_basins == 1
    row_b = result.per_basin.set_index("basin_id").loc["b"]
    assert math.isnan(row_b["nse"])  # never silently coerced to zero


def test_distribution_pd_na_marker_is_nonfinite_not_zero():
    expected = ["a", "b", "c"]
    result = analyze_basin_distribution("cfg", expected, _rows(expected, [0.1, pd.NA, 0.3]))
    assert result.n_total_basins == 3
    assert result.n_finite_basins == 2
    assert result.n_nonfinite_basins == 1
    row_b = result.per_basin.set_index("basin_id").loc["b"]
    assert math.isnan(row_b["nse"])


def test_paired_alignment_retains_none_and_pd_na_basins_as_nonfinite_unavailable():
    expected = ["a", "b", "c", "d"]
    candidate = analyze_basin_distribution("cand", expected, _rows(expected, [0.5, None, 0.3, pd.NA]))
    reference = analyze_basin_distribution("ref", expected, _rows(expected, [0.4, 0.2, 0.3, 0.1]))
    paired = analyze_paired_basin_difference(candidate, reference)
    by_id = paired.per_basin.set_index("basin_id")
    assert by_id.loc["b", "classification"] == "nonfinite_unavailable"
    assert math.isnan(by_id.loc["b", "delta"])
    assert by_id.loc["d", "classification"] == "nonfinite_unavailable"
    assert math.isnan(by_id.loc["d", "delta"])
    # a: 0.5 vs 0.4 -> improved; c: 0.3 vs 0.3 -> tied
    assert by_id.loc["a", "classification"] == "improved"
    assert by_id.loc["c", "classification"] == "tied"
    assert paired.n_total_basins == 4
    assert paired.n_candidate_finite == 2
    assert paired.n_reference_finite == 4
    assert paired.n_pairwise_finite == 2
    assert paired.n_pairwise_nonfinite == 2


def test_boolean_and_string_still_rejected_alongside_missing_markers():
    expected = ["a", "b"]
    with pytest.raises(BasinAnalysisContractError):
        analyze_basin_distribution("cfg", expected, _rows(expected, [None, True]))
    with pytest.raises(BasinAnalysisContractError):
        analyze_basin_distribution("cfg", expected, _rows(expected, [None, "0.2"]))


def test_malformed_container_value_still_rejected_no_ambiguous_truth_value():
    expected = ["a", "b"]
    # A list/dict is neither a recognized missing marker nor a numeric
    # scalar; must raise cleanly, never trigger a numpy/pandas ambiguous
    # truth-value error.
    with pytest.raises(BasinAnalysisContractError):
        analyze_basin_distribution("cfg", expected, _rows(expected, [0.1, [1, 2, 3]]))
    with pytest.raises(BasinAnalysisContractError):
        analyze_basin_distribution("cfg", expected, _rows(expected, [0.1, {"x": 1}]))


# ---------------------------------------------------------------------------
# Correction pass, finding 2: ECDF scalar validation via the shared contract
# ---------------------------------------------------------------------------


def test_ecdf_rejects_numeric_looking_string():
    with pytest.raises(BasinAnalysisContractError):
        compute_ecdf_table([0.1, "0.2", 0.3])


def test_ecdf_rejects_bool():
    with pytest.raises(BasinAnalysisContractError):
        compute_ecdf_table([0.1, True, 0.3])


def test_ecdf_none_counts_as_nonfinite():
    table = compute_ecdf_table([0.1, None, 0.3])
    assert table.n_total == 3
    assert table.n_finite == 2
    assert table.n_nonfinite == 1


def test_ecdf_pd_na_counts_as_nonfinite():
    table = compute_ecdf_table([0.1, pd.NA, 0.3])
    assert table.n_total == 3
    assert table.n_finite == 2
    assert table.n_nonfinite == 1


def test_ecdf_mixed_finite_and_nonfinite_input():
    table = compute_ecdf_table([1.0, np.nan, 2.0, np.inf, -np.inf, 1.0, None])
    assert table.n_total == 7
    assert table.n_finite == 3
    assert table.n_nonfinite == 4
    assert list(table.value) == [1.0, 2.0]
    assert list(table.count) == [2, 1]
    assert table.cumulative_fraction == pytest.approx([2 / 3, 3 / 3])


# ---------------------------------------------------------------------------
# Correction pass, finding 3: complete standalone ECDF accounting
# ---------------------------------------------------------------------------


def test_ecdf_accounting_empty_input():
    table = compute_ecdf_table([])
    assert table.n_total == 0
    assert table.n_finite == 0
    assert table.n_nonfinite == 0


def test_ecdf_accounting_all_finite_input():
    table = compute_ecdf_table([3.0, 1.0, 2.0])
    assert table.n_total == 3
    assert table.n_finite == 3
    assert table.n_nonfinite == 0


def test_ecdf_accounting_mixed_input():
    table = compute_ecdf_table([1.0, np.nan, 2.0])
    assert table.n_total == 3
    assert table.n_finite == 2
    assert table.n_nonfinite == 1


def test_ecdf_accounting_all_nonfinite_input():
    table = compute_ecdf_table([np.nan, np.inf, -np.inf])
    assert table.n_total == 3
    assert table.n_finite == 0
    assert table.n_nonfinite == 3
    # stable empty schema even though n_total > 0.
    assert list(table.value) == []


# ---------------------------------------------------------------------------
# Correction pass, finding 4: linear-interpolation regression via a sparse,
# hand-computed fixture reused through compute_percentile_table.
# ---------------------------------------------------------------------------


def test_frozen_core_sparse_two_point_linear_interpolation_fixture():
    # [0.0, 10.0]: NumPy linear-percentile interpolation places percentile p
    # at index (n - 1) * p / 100 = p / 100 for n=2, i.e. value = 10 * p/100.
    expected = ["a", "b"]
    result = analyze_basin_distribution("cfg", expected, _rows(expected, [0.0, 10.0]))
    core = result.frozen_core
    assert core.q1 == pytest.approx(0.1)
    assert core.q5 == pytest.approx(0.5)
    assert core.q25 == pytest.approx(2.5)
    assert core.q50 == pytest.approx(5.0)
    assert core.q75 == pytest.approx(7.5)
    assert core.q95 == pytest.approx(9.5)
    assert core.q99 == pytest.approx(9.9)
    assert core.iqr == pytest.approx(5.0)
    # Cross-check directly against the reused qualified helper's own output,
    # proving this is the same linear-interpolation path, not a parallel one.
    reference = compute_percentile_table([0.0, 10.0], metric_name="nse")
    assert core.q1 == pytest.approx(reference.percentiles["p1"])
    assert core.q99 == pytest.approx(reference.percentiles["p99"])


# ---------------------------------------------------------------------------
# Correction pass, finding 5: exact +-1e-12 tolerance boundary regions
# ---------------------------------------------------------------------------


def test_exact_tolerance_boundary_all_five_regions():
    # Reference fixed at 0.0 so delta == candidate value exactly (a
    # subtraction from zero introduces no floating-point rounding), letting
    # each region sit at exactly the intended distance from the boundary.
    expected = ["above", "at_plus", "inside", "at_minus", "below"]
    candidate_values = [
        2 * NUMERICAL_TIE_TOLERANCE,   # delta > +1e-12 -> improved
        NUMERICAL_TIE_TOLERANCE,        # delta == +1e-12 -> tied
        NUMERICAL_TIE_TOLERANCE / 2,    # -1e-12 < delta < +1e-12 -> tied
        -NUMERICAL_TIE_TOLERANCE,       # delta == -1e-12 -> tied
        -2 * NUMERICAL_TIE_TOLERANCE,   # delta < -1e-12 -> worse
    ]
    candidate = analyze_basin_distribution("cand", expected, _rows(expected, candidate_values))
    reference = analyze_basin_distribution("ref", expected, _rows(expected, [0.0] * 5))
    paired = analyze_paired_basin_difference(candidate, reference)
    by_id = paired.per_basin.set_index("basin_id")
    assert by_id.loc["above", "classification"] == "improved"
    assert by_id.loc["at_plus", "classification"] == "tied"
    assert by_id.loc["inside", "classification"] == "tied"
    assert by_id.loc["at_minus", "classification"] == "tied"
    assert by_id.loc["below", "classification"] == "worse"
    assert paired.n_improved == 1
    assert paired.n_worse == 1
    assert paired.n_tied == 3

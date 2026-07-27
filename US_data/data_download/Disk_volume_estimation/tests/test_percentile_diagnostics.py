"""Tests for src/baseline/percentile_diagnostics.py (Stage 1 validation and
optimization foundation, Part A).

Expected percentile/statistic values are hand-computed against small,
fully-enumerable synthetic arrays -- never derived by calling the function
under test with different arguments.
"""
import math

import numpy as np
import pytest

from src.baseline.percentile_diagnostics import (
    PERCENTILES,
    basin_consistency_diagnostic,
    build_epoch_percentile_tables,
    compute_percentile_table,
    percentile_change_table,
)

# ---------------------------------------------------------------------------
# compute_percentile_table
# ---------------------------------------------------------------------------


def test_percentile_table_basic_values():
    # 0..100 (101 points): linear-interpolation percentile p sits exactly at
    # index p (0-indexed step of 1 per unit), so np.percentile(p) == p.
    values = np.arange(0, 101, dtype=np.float64)
    table = compute_percentile_table(values, metric_name="nse").to_dict()
    assert table["metric_name"] == "nse"
    assert table["n_total_basins"] == 101
    assert table["n_finite_basins"] == 101
    for p in PERCENTILES:
        assert table[f"p{p}"] == pytest.approx(float(p), abs=1e-9)
    assert table["min"] == pytest.approx(0.0)
    assert table["max"] == pytest.approx(100.0)


def test_percentile_table_sign_fractions():
    # 2 negative, 1 zero, 3 positive (1 of which > 0.5) -> 6 finite total.
    values = np.array([-2.0, -1.0, 0.0, 0.1, 0.5, 1.0])
    table = compute_percentile_table(values).to_dict()
    assert table["frac_lt_0"] == pytest.approx(2 / 6)
    assert table["frac_gt_0"] == pytest.approx(3 / 6)
    assert table["frac_gt_0p5"] == pytest.approx(1 / 6)


def test_percentile_table_excludes_nonfinite_but_counts_total():
    values = np.array([1.0, 2.0, np.nan, np.inf, -np.inf, 3.0])
    table = compute_percentile_table(values).to_dict()
    assert table["n_total_basins"] == 6
    assert table["n_finite_basins"] == 3
    assert table["p50"] == pytest.approx(2.0)


def test_percentile_table_all_nonfinite_returns_nan_but_no_crash():
    values = np.array([np.nan, np.nan])
    table = compute_percentile_table(values).to_dict()
    assert table["n_total_basins"] == 2
    assert table["n_finite_basins"] == 0
    assert all(math.isnan(table[f"p{p}"]) for p in PERCENTILES)
    assert math.isnan(table["min"])
    assert math.isnan(table["frac_gt_0"])


def test_percentile_table_empty_array():
    table = compute_percentile_table(np.array([])).to_dict()
    assert table["n_total_basins"] == 0
    assert table["n_finite_basins"] == 0


# ---------------------------------------------------------------------------
# build_epoch_percentile_tables
# ---------------------------------------------------------------------------


def test_build_epoch_percentile_tables_sorted_and_keyed_by_epoch():
    per_epoch = {
        3: np.array([1.0, 2.0, 3.0]),
        1: np.array([4.0, 5.0, 6.0]),
    }
    tables = build_epoch_percentile_tables(per_epoch)
    assert list(tables.keys()) == [1, 3]
    assert tables[1]["p50"] == pytest.approx(5.0)
    assert tables[3]["p50"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# percentile_change_table
# ---------------------------------------------------------------------------


def _make_epoch_tables(per_epoch):
    return build_epoch_percentile_tables(per_epoch)


def test_percentile_change_table_monotonic_increasing():
    per_epoch = {
        1: np.full(10, 0.1),
        2: np.full(10, 0.2),
        3: np.full(10, 0.3),
    }
    tables = _make_epoch_tables(per_epoch)
    rows = percentile_change_table(tables)
    p50_row = next(r for r in rows if r["percentile"] == "p50")
    assert p50_row["epochs"] == [1, 2, 3]
    assert p50_row["values"] == pytest.approx([0.1, 0.2, 0.3])
    assert p50_row["deltas"] == pytest.approx([0.1, 0.1])
    assert p50_row["monotonic"] == "increasing"


def test_percentile_change_table_non_monotonic():
    per_epoch = {
        1: np.full(10, 0.1),
        2: np.full(10, 0.3),
        3: np.full(10, 0.2),
    }
    tables = _make_epoch_tables(per_epoch)
    rows = percentile_change_table(tables)
    p50_row = next(r for r in rows if r["percentile"] == "p50")
    assert p50_row["monotonic"] == "non_monotonic"


def test_percentile_change_table_constant():
    per_epoch = {1: np.full(10, 0.5), 2: np.full(10, 0.5)}
    tables = _make_epoch_tables(per_epoch)
    rows = percentile_change_table(tables)
    p50_row = next(r for r in rows if r["percentile"] == "p50")
    assert p50_row["monotonic"] == "constant"


def test_percentile_change_table_undefined_when_all_nan():
    per_epoch = {1: np.array([np.nan, np.nan]), 2: np.array([np.nan, np.nan])}
    tables = _make_epoch_tables(per_epoch)
    rows = percentile_change_table(tables)
    p50_row = next(r for r in rows if r["percentile"] == "p50")
    assert p50_row["monotonic"] == "undefined"
    assert all(math.isnan(d) for d in p50_row["deltas"])


# ---------------------------------------------------------------------------
# basin_consistency_diagnostic
# ---------------------------------------------------------------------------


def test_basin_consistency_perfect_agreement():
    per_epoch = {
        1: {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4, "E": 0.5, "F": 0.6, "G": 0.7, "H": 0.8},
        2: {"A": 0.15, "B": 0.25, "C": 0.35, "D": 0.45, "E": 0.55, "F": 0.65, "G": 0.75, "H": 0.85},
    }
    rows = basin_consistency_diagnostic(per_epoch)
    assert len(rows) == 1
    row = rows[0]
    assert row["epoch_from"] == 1
    assert row["epoch_to"] == 2
    assert row["n_common_finite_basins"] == 8
    # identical ranks across epochs -> perfect rank correlation.
    assert row["spearman_r"] == pytest.approx(1.0)
    # every basin's relative ordering (and hence quartile) is unchanged.
    assert row["frac_same_quartile"] == pytest.approx(1.0)


def test_basin_consistency_reversed_ranks_gives_negative_spearman():
    per_epoch = {
        1: {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4},
        2: {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1},
    }
    rows = basin_consistency_diagnostic(per_epoch)
    assert rows[0]["spearman_r"] == pytest.approx(-1.0)


def test_basin_consistency_excludes_basins_missing_or_nonfinite_in_either_epoch():
    per_epoch = {
        1: {"A": 0.1, "B": 0.2, "C": np.nan, "D": 0.4},
        2: {"A": 0.15, "B": 0.25, "D": 0.45},  # C missing entirely from epoch 2
    }
    rows = basin_consistency_diagnostic(per_epoch)
    # Only A, B, D are common and finite on both sides (C excluded: absent
    # from epoch 2 AND NaN in epoch 1).
    assert rows[0]["n_common_finite_basins"] == 3


def test_basin_consistency_too_few_common_basins_returns_nan_not_crash():
    per_epoch = {1: {"A": 0.1}, 2: {"A": 0.2}}
    rows = basin_consistency_diagnostic(per_epoch)
    assert rows[0]["n_common_finite_basins"] == 1
    assert math.isnan(rows[0]["frac_same_quartile"])
    assert math.isnan(rows[0]["spearman_r"])


def test_basin_consistency_multiple_epoch_pairs():
    per_epoch = {
        1: {"A": 0.1, "B": 0.2, "C": 0.3},
        2: {"A": 0.2, "B": 0.3, "C": 0.4},
        3: {"A": 0.3, "B": 0.4, "C": 0.5},
    }
    rows = basin_consistency_diagnostic(per_epoch)
    assert len(rows) == 2
    assert [(r["epoch_from"], r["epoch_to"]) for r in rows] == [(1, 2), (2, 3)]

import io
import json

import numpy as np
import pandas as pd
import pytest

from src.baseline.stage1_rd1_c4_f_synthesis import (
    BasinIdFormatError,
    RosterMismatchError,
    add_raw_axis_columns,
    basin_arm_medians,
    build_configuration_table,
    select_percentile_basins,
    sign_counts,
    validate_canonical_basin_ids,
    validate_roster_against_trial_ids,
)

# The three real RD1-C4-F selected basin IDs (10th/50th/90th percentile of
# Bayesian-minus-random-control median-NSE difference). Canonical, fixed-width,
# zero-padded USGS identifiers -- must never lose their leading zero.
SELECTED_BASIN_IDS = ("06600100", "01464000", "02303205")


def _tiny_roster():
    rows = []
    for arm in ("bayesian", "random_control"):
        for order in (1, 2):
            rows.append(
                {
                    "proposal_id": f"trial_{arm}_{order}",
                    "search_arm": arm,
                    "proposal_order": order,
                    "coord_learning_rate": 0.001 * order,
                    "coord_hidden_size": 128,
                    "coord_embedding_dropout": 0.1,
                    "coord_output_dropout": 0.2,
                    "coord_batch_size": 256,
                    "coord_seq_length": 60,
                    "canonical_coordinate_key": json.dumps(
                        {
                            "learning_rate": str(0.0001 * order),
                            "hidden_size": 128,
                            "embedding_dropout": "0.1",
                            "output_dropout": "0.2",
                            "batch_size": 256,
                            "seq_length": 60,
                        }
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_add_raw_axis_columns_parses_canonical_coordinate_key():
    roster = _tiny_roster()
    out = add_raw_axis_columns(roster)
    assert out["raw_learning_rate"].tolist() == pytest.approx([0.0001, 0.0002, 0.0001, 0.0002])
    assert out["raw_hidden_size"].tolist() == [128.0, 128.0, 128.0, 128.0]


def test_validate_roster_against_trial_ids_passes_on_exact_match():
    roster = _tiny_roster()
    validate_roster_against_trial_ids(roster, roster["proposal_id"])


def test_validate_roster_against_trial_ids_raises_on_missing_and_extra():
    roster = _tiny_roster()
    expected = set(roster["proposal_id"]) - {"trial_bayesian_1"} | {"trial_bayesian_99"}
    with pytest.raises(RosterMismatchError):
        validate_roster_against_trial_ids(roster, expected)


def test_validate_roster_against_trial_ids_raises_on_duplicate():
    roster = pd.concat([_tiny_roster(), _tiny_roster().iloc[[0]]], ignore_index=True)
    with pytest.raises(RosterMismatchError):
        validate_roster_against_trial_ids(roster, roster["proposal_id"])


def _tiny_per_basin_metrics():
    rows = []
    for arm in ("bayesian", "random_control"):
        for order in (1, 2):
            trial_id = f"trial_{arm}_{order}"
            for basin, nse, kge in (("A", 0.5, 0.6), ("B", 0.7, 0.8)):
                rows.append(
                    {
                        "trial_id": trial_id,
                        "search_arm": arm,
                        "configuration_id": f"cfg_{arm}_{order}",
                        "proposal_order": order,
                        "official_objective": 0.1 * order,
                        "basin_id": basin,
                        "nse": nse + 0.01 * order,
                        "kge": kge + 0.01 * order,
                    }
                )
    return pd.DataFrame(rows)


def _tiny_q98_diagnostics(per_basin_metrics):
    rows = []
    for _, r in per_basin_metrics.iterrows():
        rows.append(
            {
                "trial_id": r["trial_id"],
                "basin_id": r["basin_id"],
                "q98_normalized_rmse": 0.3,
                "relative_volume_bias": 0.05,
                "observed_peak_time_magnitude_error": 0.02,
                "high_flow_nse": 0.4,
            }
        )
    return pd.DataFrame(rows)


def test_build_configuration_table_one_row_per_trial_with_axes():
    pbm = _tiny_per_basin_metrics()
    q98 = _tiny_q98_diagnostics(pbm)
    roster = _tiny_roster()

    table = build_configuration_table(pbm, q98, roster)

    assert len(table) == 4
    assert set(table["trial_id"]) == set(roster["proposal_id"])
    assert list(table["search_arm"].unique()) == ["bayesian", "random_control"]
    for axis in ("coord_learning_rate", "coord_hidden_size", "raw_learning_rate", "raw_hidden_size"):
        assert axis in table.columns
        assert table[axis].notna().all()
    # basin A has nse=0.5+0.01*order, basin B has nse=0.7+0.01*order -> median is midpoint
    row = table[table["trial_id"] == "trial_bayesian_1"].iloc[0]
    assert row["median_nse"] == pytest.approx((0.51 + 0.71) / 2)


def test_build_configuration_table_raises_when_roster_missing_a_trial():
    pbm = _tiny_per_basin_metrics()
    q98 = _tiny_q98_diagnostics(pbm)
    roster = _tiny_roster().iloc[1:].reset_index(drop=True)  # drop trial_bayesian_1

    with pytest.raises(RosterMismatchError):
        build_configuration_table(pbm, q98, roster)


def test_basin_arm_medians_and_sign_counts():
    pbm = _tiny_per_basin_metrics()
    medians = basin_arm_medians(pbm, "nse")

    assert set(medians.index) == {"A", "B"}
    assert "diff_bayesian_minus_random" in medians.columns

    counts = sign_counts(medians["diff_bayesian_minus_random"])
    assert counts["n_positive"] + counts["n_negative"] + counts["n_zero"] == 2


def test_select_percentile_basins_is_deterministic_and_picks_closest():
    diff = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=["b", "a", "c", "d", "e"])
    selected = select_percentile_basins(diff, percentiles=(10, 50, 90))
    # median of [1..5] is 3.0 -> basin "c"
    assert selected[50] == "c"
    # result must be stable across repeated calls (no RNG / dict-order dependence)
    selected_again = select_percentile_basins(diff, percentiles=(10, 50, 90))
    assert selected == selected_again


def test_select_percentile_basins_breaks_ties_lexicographically():
    # two basins equidistant from the target percentile
    diff = pd.Series([1.0, 3.0], index=["zzz", "aaa"])
    target_percentile = 50  # midpoint 2.0 is equidistant from both 1.0 and 3.0
    selected = select_percentile_basins(diff, percentiles=(target_percentile,))
    assert selected[target_percentile] == "aaa"


def test_select_percentile_basins_raises_on_empty_series():
    with pytest.raises(ValueError):
        select_percentile_basins(pd.Series(dtype=float), percentiles=(50,))


def test_validate_canonical_basin_ids_accepts_the_three_selected_ids():
    validate_canonical_basin_ids(SELECTED_BASIN_IDS)


def test_validate_canonical_basin_ids_rejects_zero_padding_loss():
    # exactly the defect this fix repairs: leading zeros silently stripped
    unpadded = ("6600100", "1464000", "2303205")
    with pytest.raises(BasinIdFormatError):
        validate_canonical_basin_ids(unpadded)


def test_validate_canonical_basin_ids_rejects_non_string_numeric():
    with pytest.raises(BasinIdFormatError):
        validate_canonical_basin_ids([6600100, 1464000, 2303205])


def test_read_csv_with_dtype_str_preserves_leading_zeros_for_selected_ids():
    # Reproduces the real defect end-to-end: pd.read_csv without an explicit
    # dtype silently coerces an all-digit basin_id column to int64 and drops
    # leading zeros; dtype={"basin_id": str} preserves the canonical string.
    csv_text = "basin_id,nse\n06600100,0.32\n01464000,0.47\n02303205,0.62\n"

    coerced = pd.read_csv(io.StringIO(csv_text))
    assert coerced["basin_id"].tolist() != list(SELECTED_BASIN_IDS)  # demonstrates the defect

    fixed = pd.read_csv(io.StringIO(csv_text), dtype={"basin_id": str})
    assert fixed["basin_id"].tolist() == list(SELECTED_BASIN_IDS)
    validate_canonical_basin_ids(fixed["basin_id"])


def test_select_percentile_basins_preserves_canonical_ids_of_real_selection():
    # Synthetic diff series keyed by the three real selected basin IDs, spaced
    # so that percentile selection deterministically picks each one at its
    # documented percentile, confirming the zero-padded string identity
    # survives the full selection path untouched.
    diff = pd.Series(
        {"06600100": -100.0, "01464000": 0.0, "02303205": 100.0}
    )

    selected = select_percentile_basins(diff, percentiles=(10, 50, 90))
    for basin_id in selected.values():
        assert isinstance(basin_id, str)
        assert len(basin_id) == 8
    assert selected[10] == "06600100"
    assert selected[50] == "01464000"
    assert selected[90] == "02303205"

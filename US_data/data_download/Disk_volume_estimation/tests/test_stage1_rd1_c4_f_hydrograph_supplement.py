import numpy as np
import pandas as pd
import pytest

from src.baseline.stage1_rd1_c4_f_hydrograph_supplement import (
    IncumbentResolutionError,
    PeakWindowError,
    build_incumbent_order_mapping,
    incumbent_trial_at_proposal_order,
    select_observed_peak_window,
    validate_selected_basins_canonical,
)
from src.baseline.stage1_rd1_c4_f_synthesis import BasinIdFormatError

SELECTED_BASIN_IDS = {"10": "06600100", "50": "01464000", "90": "02303205"}


# ---------------------------------------------------------------------------
# select_observed_peak_window
# ---------------------------------------------------------------------------

def _hourly_series(n_hours, start="2020-01-01"):
    return pd.date_range(start, periods=n_hours, freq="h")


def test_select_observed_peak_window_picks_global_max_and_centers_window():
    dates = _hourly_series(200)
    obs = np.full(200, 1.0)
    obs[100] = 50.0  # single unambiguous peak, well inside the series
    admitted = np.ones(200, dtype=bool)

    pw = select_observed_peak_window(dates, obs, admitted, basin_id="01464000")

    assert pw.peak_time == dates[100]
    assert pw.peak_value == 50.0
    assert pw.window_start == dates[100] - pd.Timedelta(hours=36)
    assert pw.window_end == dates[100] + pd.Timedelta(hours=36)
    assert pw.actual_window_hours == pytest.approx(72.0)
    assert not pw.clipped_left
    assert not pw.clipped_right


def test_select_observed_peak_window_earliest_tie_rule():
    dates = _hourly_series(200)
    obs = np.full(200, 1.0)
    obs[80] = 50.0
    obs[120] = 50.0  # exact tie -- earliest (index 80) must win
    admitted = np.ones(200, dtype=bool)

    pw = select_observed_peak_window(dates, obs, admitted, basin_id="01464000")

    assert pw.peak_time == dates[80]


def test_select_observed_peak_window_ignores_non_admitted_and_nonfinite():
    dates = _hourly_series(200)
    obs = np.full(200, 1.0)
    obs[150] = 999.0  # would be the max, but not admitted
    obs[100] = 50.0   # true admitted max
    admitted = np.ones(200, dtype=bool)
    admitted[150] = False

    pw = select_observed_peak_window(dates, obs, admitted, basin_id="01464000")
    assert pw.peak_time == dates[100]
    assert pw.peak_value == 50.0


def test_select_observed_peak_window_clips_at_series_start_without_shifting():
    dates = _hourly_series(200)
    obs = np.full(200, 1.0)
    obs[10] = 50.0  # only 10 admitted hours before the peak, less than 36
    admitted = np.ones(200, dtype=bool)

    pw = select_observed_peak_window(dates, obs, admitted, basin_id="01464000")

    assert pw.window_start == dates[0]
    assert pw.clipped_left
    assert not pw.clipped_right
    assert pw.window_end == dates[10] + pd.Timedelta(hours=36)
    assert pw.actual_window_hours < 72.0


def test_select_observed_peak_window_clips_at_series_end_without_shifting():
    dates = _hourly_series(200)
    obs = np.full(200, 1.0)
    obs[195] = 50.0  # only 4 admitted hours after the peak
    admitted = np.ones(200, dtype=bool)

    pw = select_observed_peak_window(dates, obs, admitted, basin_id="01464000")

    assert pw.window_end == dates[-1]
    assert pw.clipped_right
    assert not pw.clipped_left


def test_select_observed_peak_window_raises_on_no_admitted_samples():
    dates = _hourly_series(10)
    obs = np.full(10, 1.0)
    admitted = np.zeros(10, dtype=bool)
    with pytest.raises(PeakWindowError):
        select_observed_peak_window(dates, obs, admitted, basin_id="01464000")


# ---------------------------------------------------------------------------
# incumbent_trial_at_proposal_order / build_incumbent_order_mapping
# ---------------------------------------------------------------------------

def _tiny_config_table():
    # Mirrors the real roster's structure: proposal 1 and 9 are new
    # incumbents for "bayesian"; proposal 1 and 2 for "random_control".
    rows = [
        ("bayesian", 1, True),
        ("bayesian", 2, False),
        ("bayesian", 3, False),
        ("bayesian", 9, True),
        ("bayesian", 12, False),
        ("random_control", 1, True),
        ("random_control", 2, True),
        ("random_control", 3, False),
        ("random_control", 9, False),
    ]
    return pd.DataFrame(
        [
            {
                "trial_id": f"trial_{arm}_{order}",
                "search_arm": arm,
                "proposal_order": order,
                "is_new_incumbent": is_new,
            }
            for arm, order, is_new in rows
        ]
    )


def test_incumbent_trial_at_proposal_order_picks_most_recent_incumbent():
    table = _tiny_config_table()
    assert incumbent_trial_at_proposal_order(table, "bayesian", 1) == "trial_bayesian_1"
    assert incumbent_trial_at_proposal_order(table, "bayesian", 3) == "trial_bayesian_1"
    assert incumbent_trial_at_proposal_order(table, "bayesian", 9) == "trial_bayesian_9"
    assert incumbent_trial_at_proposal_order(table, "bayesian", 12) == "trial_bayesian_9"


def test_incumbent_trial_at_proposal_order_raises_when_none_exists_yet():
    table = pd.DataFrame(
        [{"trial_id": "t1", "search_arm": "bayesian", "proposal_order": 5, "is_new_incumbent": True}]
    )
    with pytest.raises(IncumbentResolutionError):
        incumbent_trial_at_proposal_order(table, "bayesian", 1)


def test_build_incumbent_order_mapping_dedups_and_preserves_full_mapping():
    table = _tiny_config_table()
    manifest = build_incumbent_order_mapping(table, ("bayesian", "random_control"), (1, 3, 6, 9, 12))

    # complete mapping: 2 arms x 5 orders = 10 lookups
    assert manifest["n_lookups"] == 10
    assert manifest["order_to_incumbent_trial_id"]["bayesian__proposal_order_01"] == "trial_bayesian_1"
    assert manifest["order_to_incumbent_trial_id"]["bayesian__proposal_order_06"] == "trial_bayesian_1"
    assert manifest["order_to_incumbent_trial_id"]["bayesian__proposal_order_09"] == "trial_bayesian_9"
    assert manifest["order_to_incumbent_trial_id"]["random_control__proposal_order_03"] == "trial_random_control_2"

    # dedup: bayesian has 2 unique incumbents, random_control has 2 unique -> 4 total
    assert manifest["n_unique_trials"] == 4
    assert set(manifest["unique_incumbent_trial_ids"]) == {
        "trial_bayesian_1", "trial_bayesian_9", "trial_random_control_1", "trial_random_control_2",
    }


# ---------------------------------------------------------------------------
# validate_selected_basins_canonical (thin boundary re-check)
# ---------------------------------------------------------------------------

def test_validate_selected_basins_canonical_accepts_real_selection():
    validate_selected_basins_canonical(SELECTED_BASIN_IDS)


def test_validate_selected_basins_canonical_rejects_unpadded_ids():
    with pytest.raises(BasinIdFormatError):
        validate_selected_basins_canonical({"10": "6600100", "50": "1464000", "90": "2303205"})

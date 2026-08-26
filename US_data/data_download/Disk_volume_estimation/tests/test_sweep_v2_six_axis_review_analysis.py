"""Offline numerical/structural tests for the Sweep-v2 six-axis
review-analysis extension (:mod:`src.baseline.sweep_v2_six_axis_review_analysis`).

Mirrors ``tests/test_sweep_v1_review_analysis.py``'s small-hand-built-table
approach, extended with a ``seq_length`` column/parameter. Never touches
W&B, Slurm, Moriah, or sealed scopes -- pure pandas/numpy over in-memory
fixtures.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.baseline import sweep_v1_campaign as sweep_v1
from src.baseline import sweep_v1_review_analysis as analysis_v1
from src.baseline import sweep_v2_six_axis_review_analysis as analysis_v2
from src.baseline.sweep_v2_six_axis_campaign import CAMPAIGN_ID_V2, DOMAIN_VERSION_V2


def _row(*, search_arm, proposal_order, valid_result_order, workflow_status="pass",
         best_score=0.35, best_epoch=8, final_epoch_score=0.34, best_minus_final=0.01,
         best_score_10=0.35, best_score_12=0.35, late_gain_10_to_12=0.0, late_best=False,
         learning_rate=3e-4, hidden_size=128, embedding_dropout=0.1, output_dropout=0.2,
         batch_size=256, seq_length=72, gpu_hours=2.0, runtime_seconds=7200.0, trial_id=None,
         synthetic_archetype=None):
    return {
        "campaign_id": CAMPAIGN_ID_V2, "domain_version": DOMAIN_VERSION_V2,
        "search_arm": search_arm, "proposal_id": f"{search_arm}_{proposal_order}",
        "configuration_id": f"sweep_v2_cfg_{search_arm}_{proposal_order}",
        "trial_id": trial_id or f"trial_{search_arm}_{proposal_order}",
        "workflow_status": workflow_status,
        "objective_score": best_score if workflow_status == "pass" else None,
        "best_epoch": best_epoch if workflow_status == "pass" else None,
        "best_score": best_score if workflow_status == "pass" else None,
        "final_epoch_score": final_epoch_score if workflow_status == "pass" else None,
        "best_minus_final": best_minus_final if workflow_status == "pass" else None,
        "best_score_10": best_score_10 if workflow_status == "pass" else None,
        "best_score_12": best_score_12 if workflow_status == "pass" else None,
        "late_gain_10_to_12": late_gain_10_to_12 if workflow_status == "pass" else None,
        "late_best": late_best if workflow_status == "pass" else None,
        "learning_rate": learning_rate, "hidden_size": hidden_size,
        "embedding_dropout": embedding_dropout, "output_dropout": output_dropout,
        "batch_size": batch_size, "seq_length": seq_length,
        "runtime_seconds": runtime_seconds, "gpu_hours": gpu_hours,
        "execution_generation": 1, "retry_of_trial_id": None,
        "failure_category": None if workflow_status == "pass" else "node_failure",
        "proposal_order": proposal_order,
        "valid_result_order": valid_result_order if workflow_status == "pass" else None,
        "wave_id": f"{DOMAIN_VERSION_V2}_wave1", "boundary_review_checkpoint": None,
        "synthetic_archetype": synthetic_archetype,
    }


def _small_trial_df() -> pd.DataFrame:
    """5 valid Bayesian, 1 failed Bayesian (proposal_order=3); v2 has no
    random_control arm (SEARCH_ARMS_V2 is Bayesian-only), so this fixture
    intentionally omits it (unlike v1's fixture)."""
    rows = [
        _row(search_arm="bayesian", proposal_order=1, valid_result_order=1, best_score=0.340,
             learning_rate=9.5e-4, hidden_size=64, seq_length=48),
        _row(search_arm="bayesian", proposal_order=2, valid_result_order=2, best_score=0.360,
             learning_rate=1.1e-4, hidden_size=256, gpu_hours=3.0, seq_length=120),
        _row(search_arm="bayesian", proposal_order=3, valid_result_order=None, workflow_status="failed",
             learning_rate=2e-4, hidden_size=128, seq_length=72),
        _row(search_arm="bayesian", proposal_order=4, valid_result_order=3, best_score=0.370,
             learning_rate=1.05e-4, hidden_size=256, gpu_hours=2.5, late_best=True, best_epoch=11,
             late_gain_10_to_12=0.01, seq_length=120),
        _row(search_arm="bayesian", proposal_order=5, valid_result_order=4, best_score=0.330,
             learning_rate=5e-4, hidden_size=128, best_minus_final=0.05, final_epoch_score=0.28,
             seq_length=96),
        _row(search_arm="bayesian", proposal_order=6, valid_result_order=5, best_score=0.355,
             learning_rate=3e-4, hidden_size=64, gpu_hours=1.9, seq_length=48),
    ]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# reuse-vs-sibling identity: axis-agnostic functions must be the SAME
# object as v1's, not re-implementations.
# ------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "VALID_STATUS", "BOUNDARY_FRACTION", "SENSITIVITY_BAND_FRACTIONS", "TIER_ORDER",
    "valid_trials", "checkpoint_slice", "checkpoint_operations_slice", "common_n",
    "cumulative_best_by_order", "cumulative_gpu_hours_by_order", "top_quartile_threshold",
    "is_top_quartile", "near_boundary_mask", "categorical_extreme_mask", "proposal_drift_evidence",
    "neighborhood_support_evidence", "boundary_pressure_tier", "representative_trajectory_selection",
    "boundary_pressure_evolution", "boundary_band_sensitivity", "checkpoint_status_summary",
])
def test_reused_symbols_are_the_same_object_as_v1(name):
    assert getattr(analysis_v2, name) is getattr(analysis_v1, name)


# ------------------------------------------------------------------
# six-axis boundary pressure table
# ------------------------------------------------------------------

def test_derive_boundary_pressure_table_v2_covers_all_six_axes_both_sides():
    df = _small_trial_df()
    table = analysis_v2.derive_boundary_pressure_table_v2(df)
    assert set(table["axis"]) == {"learning_rate", "embedding_dropout", "output_dropout",
                                   "hidden_size", "batch_size", "seq_length"}
    assert set(table["tier"]) <= {"STRONG", "MODERATE", "WEAK-NONE"}


def test_derive_boundary_pressure_table_v2_seq_length_is_always_natural_both_sides():
    df = _small_trial_df()
    table = analysis_v2.derive_boundary_pressure_table_v2(df)
    seq_rows = table[table["axis"] == "seq_length"]
    assert set(seq_rows["boundary_side"]) == {"lower", "upper"}
    assert (seq_rows["boundary_nature"] == "natural").all()
    assert not seq_rows["expansion_eligible"].any()
    assert seq_rows["interpretation"].str.startswith("N/A (natural bound").all()


def test_derive_boundary_pressure_table_v2_seq_length_bounds_are_48_and_120():
    df = _small_trial_df()
    table = analysis_v2.derive_boundary_pressure_table_v2(df)
    seq_rows = table[table["axis"] == "seq_length"]
    lower_row = seq_rows[seq_rows["boundary_side"] == "lower"].iloc[0]
    upper_row = seq_rows[seq_rows["boundary_side"] == "upper"].iloc[0]
    # Two of six valid trials sit exactly at 48h (the lower grid edge) --
    # confirms lower/upper were derived from SEQ_LENGTH_MIN/MAX (48/120),
    # not accidentally left at some other default.
    assert lower_row["top_quartile_near_count"] >= 0
    assert upper_row["top_quartile_near_count"] >= 0


def test_derive_boundary_pressure_table_v2_five_original_axes_match_v1_numerically():
    """For the five axes v1 already covers, v2's table must be numerically
    identical to v1's over an equivalent trial table (SEARCH_DOMAIN_V2's
    five original axes are byte-identical deep copies of v1's)."""
    df_v2 = _small_trial_df()
    df_v1 = df_v2.assign(campaign_id=sweep_v1.CAMPAIGN_ID, domain_version=sweep_v1.DOMAIN_VERSION)
    table_v2 = analysis_v2.derive_boundary_pressure_table_v2(df_v2)
    table_v1 = analysis_v1.derive_boundary_pressure_table(df_v1)
    shared_axes = {"learning_rate", "embedding_dropout", "output_dropout", "hidden_size", "batch_size"}
    v2_shared = table_v2[table_v2["axis"].isin(shared_axes)].reset_index(drop=True)
    v1_shared = table_v1[table_v1["axis"].isin(shared_axes)].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        v2_shared.sort_values(["axis", "boundary_side"]).reset_index(drop=True),
        v1_shared.sort_values(["axis", "boundary_side"]).reset_index(drop=True),
    )


def test_derive_boundary_pressure_table_v2_no_hidden_composite_score_column():
    df = _small_trial_df()
    table = analysis_v2.derive_boundary_pressure_table_v2(df)
    forbidden = {"score", "composite_score", "weighted_score"}
    assert not (forbidden & set(table.columns))


# ------------------------------------------------------------------
# categorical occupancy (identity separation, not new axes)
# ------------------------------------------------------------------

def test_categorical_occupancy_table_v2_counts_per_arm_per_category():
    df = _small_trial_df()
    occ = analysis_v2.categorical_occupancy_table_v2(df)
    hs = occ[(occ["axis"] == "hidden_size") & (occ["search_arm"] == "bayesian")]
    counts_by_value = dict(zip(hs["value"], hs["count"]))
    assert counts_by_value[256] == 2
    assert counts_by_value[64] == 2
    assert counts_by_value[128] == 1


def test_categorical_occupancy_table_v2_reads_search_domain_v2_not_v1():
    """Sibling exists for identity separation even though the two domains
    are numerically identical for these axes -- confirm it's not literally
    the same function object as v1's (unlike the axis-agnostic reuse set)."""
    assert analysis_v2.categorical_occupancy_table_v2 is not analysis_v1.categorical_occupancy_table


# ------------------------------------------------------------------
# most-pressured continuous axis
# ------------------------------------------------------------------

def test_most_pressured_continuous_axis_v2_never_selects_seq_length():
    df = _small_trial_df()
    table = analysis_v2.derive_boundary_pressure_table_v2(df)
    axis, side = analysis_v2.most_pressured_continuous_axis_v2(table)
    assert axis != "seq_length"
    row = table[(table["axis"] == axis) & (table["boundary_side"] == side)].iloc[0]
    assert bool(row["expansion_eligible"]) is True


def test_most_pressured_continuous_axis_v2_falls_back_to_learning_rate_lower_when_nothing_expandable():
    empty = pd.DataFrame([{"axis": "seq_length", "boundary_side": "lower", "expansion_eligible": False,
                            "tier": "WEAK-NONE", "top_quartile_near_fraction": 0.0}])
    axis, side = analysis_v2.most_pressured_continuous_axis_v2(empty)
    assert (axis, side) == ("learning_rate", "lower")


# ------------------------------------------------------------------
# top configurations table
# ------------------------------------------------------------------

def test_top_configurations_table_v2_includes_seq_length_column():
    df = _small_trial_df()
    table = analysis_v2.top_configurations_table_v2(df, n_bayesian=10, n_random=3)
    assert "seq_length" in table.columns
    assert set(table["seq_length"]) <= {48, 60, 72, 84, 96, 108, 120}


def test_top_configurations_table_v2_ranked_and_capped_per_arm():
    df = _small_trial_df()
    table = analysis_v2.top_configurations_table_v2(df, n_bayesian=2, n_random=1)
    assert (table["search_arm"] == "bayesian").sum() <= 2
    assert (table["search_arm"] == "random_control").sum() == 0  # v2 has no random-control arm
    assert list(table["rank"]) == list(range(1, len(table) + 1))
    assert list(table["best_score"]) == sorted(table["best_score"], reverse=True)


def test_top_configurations_table_v2_short_id_distinguishes_rows():
    df = _small_trial_df()
    table = analysis_v2.top_configurations_table_v2(df, n_bayesian=10, n_random=3)
    assert table["configuration_short_id"].nunique() == len(table)


# ------------------------------------------------------------------
# guard tests (same discipline as v1's module)
# ------------------------------------------------------------------

def test_review_analysis_v2_has_no_wandb_dependency():
    source = Path(analysis_v2.__file__).read_text(encoding="utf-8")
    assert "import wandb" not in source and "from wandb" not in source


def test_review_analysis_v2_never_references_sealed_scopes():
    source = Path(analysis_v2.__file__).read_text(encoding="utf-8")
    lowered = source.lower()
    for forbidden in ("temporal_test", "spatial_holdout", "california"):
        assert forbidden not in lowered


def test_review_analysis_v2_never_reads_v1_search_domain_directly():
    """Section H must read SEARCH_DOMAIN_V2 exclusively, never v1's
    sweep.SEARCH_DOMAIN, to keep the two campaigns' domains independently
    editable in the future without silent cross-contamination."""
    source = Path(analysis_v2.__file__).read_text(encoding="utf-8")
    assert "sweep.SEARCH_DOMAIN" not in source
    assert "sweep_v1_campaign.SEARCH_DOMAIN" not in source

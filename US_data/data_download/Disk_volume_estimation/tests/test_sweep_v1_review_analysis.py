"""Offline numerical/structural tests for the Sweep-v1 durable review-analysis layer.

Uses small hand-built trial tables (not the full synthetic fixture) so each
test isolates one derivation. Never touches W&B, Slurm, Moriah, or sealed
scopes -- pure pandas/numpy over in-memory fixtures.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.baseline import sweep_v1_campaign as sweep
from src.baseline import sweep_v1_review_analysis as analysis


def _row(*, search_arm, proposal_order, valid_result_order, workflow_status="pass",
         best_score=0.35, best_epoch=8, final_epoch_score=0.34, best_minus_final=0.01,
         best_score_10=0.35, best_score_12=0.35, late_gain_10_to_12=0.0, late_best=False,
         learning_rate=3e-4, hidden_size=128, embedding_dropout=0.1, output_dropout=0.2,
         batch_size=256, gpu_hours=2.0, runtime_seconds=7200.0, trial_id=None,
         synthetic_archetype=None):
    return {
        "campaign_id": sweep.CAMPAIGN_ID, "domain_version": sweep.DOMAIN_VERSION,
        "search_arm": search_arm, "proposal_id": f"{search_arm}_{proposal_order}",
        "configuration_id": f"cfg_{search_arm}_{proposal_order}",
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
        "batch_size": batch_size,
        "runtime_seconds": runtime_seconds, "gpu_hours": gpu_hours,
        "execution_generation": 1, "retry_of_trial_id": None,
        "failure_category": None if workflow_status == "pass" else "node_failure",
        "proposal_order": proposal_order,
        "valid_result_order": valid_result_order if workflow_status == "pass" else None,
        "wave_id": f"{sweep.DOMAIN_VERSION}_wave1", "boundary_review_checkpoint": None,
        "synthetic_archetype": synthetic_archetype,
    }


def _small_trial_df() -> pd.DataFrame:
    """4 valid Bayesian, 1 failed Bayesian (proposal_order=3), 3 valid random."""
    rows = [
        _row(search_arm="bayesian", proposal_order=1, valid_result_order=1, best_score=0.340,
             learning_rate=9.5e-4, hidden_size=64),
        _row(search_arm="bayesian", proposal_order=2, valid_result_order=2, best_score=0.360,
             learning_rate=1.1e-4, hidden_size=256, gpu_hours=3.0),
        _row(search_arm="bayesian", proposal_order=3, valid_result_order=None, workflow_status="failed",
             learning_rate=2e-4, hidden_size=128),
        _row(search_arm="bayesian", proposal_order=4, valid_result_order=3, best_score=0.370,
             learning_rate=1.05e-4, hidden_size=256, gpu_hours=2.5, late_best=True, best_epoch=11,
             late_gain_10_to_12=0.01),
        _row(search_arm="bayesian", proposal_order=5, valid_result_order=4, best_score=0.330,
             learning_rate=5e-4, hidden_size=128, best_minus_final=0.05, final_epoch_score=0.28),
        _row(search_arm="random_control", proposal_order=1, valid_result_order=1, best_score=0.345,
             learning_rate=4e-4, hidden_size=128, gpu_hours=1.8),
        _row(search_arm="random_control", proposal_order=2, valid_result_order=2, best_score=0.350,
             learning_rate=3e-4, hidden_size=64, gpu_hours=1.9),
        _row(search_arm="random_control", proposal_order=3, valid_result_order=3, best_score=0.355,
             learning_rate=2e-4, hidden_size=256, gpu_hours=2.1),
    ]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# valid_trials / failures / retries
# ------------------------------------------------------------------

def test_valid_trials_excludes_failed_rows():
    df = _small_trial_df()
    valid = analysis.valid_trials(df)
    assert len(valid) == 7
    assert (valid["workflow_status"] == "pass").all()
    assert "cfg_bayesian_3" not in set(valid["configuration_id"])


def test_valid_trials_coerces_late_best_to_real_bool():
    df = _small_trial_df()
    valid = analysis.valid_trials(df)
    assert valid["late_best"].dtype == bool
    # bitwise-invert regression guard: ~True must be False, not -2.
    inverted = ~valid["late_best"]
    assert set(inverted.unique()) <= {True, False}


def test_retry_semantics_failed_and_pass_share_configuration_not_trial_id():
    rng_hp = {"learning_rate": 3e-4, "hidden_size": 128, "embedding_dropout": 0.1,
              "output_dropout": 0.2, "batch_size": 256}
    config_id = sweep.configuration_id(rng_hp)
    failed_id = sweep.trial_id(config_id, execution_generation=1)
    retried_id = sweep.trial_id(config_id, execution_generation=2)
    df = pd.DataFrame([
        _row(search_arm="bayesian", proposal_order=5, valid_result_order=None,
             workflow_status="failed", trial_id=failed_id, **rng_hp),
        _row(search_arm="bayesian", proposal_order=5, valid_result_order=1,
             workflow_status="pass", trial_id=retried_id, **rng_hp),
    ])
    valid = analysis.valid_trials(df)
    assert list(valid["trial_id"]) == [retried_id]
    assert failed_id != retried_id


# ------------------------------------------------------------------
# checkpoint slicing
# ------------------------------------------------------------------

def test_checkpoint_slice_orders_by_arm_then_valid_result_order():
    df = _small_trial_df()
    sliced = analysis.checkpoint_slice(df, checkpoint_valid_bayesian_count=2)
    assert list(sliced["search_arm"]) == ["bayesian", "bayesian", "random_control",
                                           "random_control", "random_control"]
    assert list(sliced[sliced["search_arm"] == "bayesian"]["valid_result_order"]) == [1, 2]


def test_checkpoint_slice_allows_bounded_overshoot_without_truncation():
    df = _small_trial_df()
    sliced = analysis.checkpoint_slice(df, checkpoint_valid_bayesian_count=3)
    bayesian = sliced[sliced["search_arm"] == "bayesian"]
    assert list(bayesian["valid_result_order"]) == [1, 2, 3]
    sliced_over = analysis.checkpoint_slice(df, checkpoint_valid_bayesian_count=100)
    assert len(sliced_over[sliced_over["search_arm"] == "bayesian"]) == 4


def test_checkpoint_slice_caps_random_control_when_requested():
    df = _small_trial_df()
    sliced = analysis.checkpoint_slice(df, checkpoint_valid_bayesian_count=100, random_control_count=1)
    assert len(sliced[sliced["search_arm"] == "random_control"]) == 1


def test_checkpoint_slice_excludes_failures():
    df = _small_trial_df()
    sliced = analysis.checkpoint_slice(df, checkpoint_valid_bayesian_count=100)
    assert (sliced["workflow_status"] == "pass").all()


def test_checkpoint_operations_slice_includes_in_window_failures_only():
    df = _small_trial_df()  # failed bayesian proposal_order=3
    ops = analysis.checkpoint_operations_slice(df, checkpoint_valid_bayesian_count=3)
    failed_rows = ops[ops["workflow_status"] != "pass"]
    assert len(failed_rows) == 1
    assert failed_rows.iloc[0]["proposal_order"] == 3

    ops_narrow = analysis.checkpoint_operations_slice(df, checkpoint_valid_bayesian_count=1)
    # 1 valid bayesian (proposal_order=1) + all 3 valid random-control (uncapped); the
    # failure at proposal_order=3 falls outside this narrower checkpoint window.
    assert (ops_narrow["workflow_status"] == "pass").sum() == 4
    assert (ops_narrow["workflow_status"] != "pass").sum() == 0


def test_checkpoint_operations_slice_scientific_slice_still_excludes_failures():
    df = _small_trial_df()
    ops = analysis.checkpoint_operations_slice(df, checkpoint_valid_bayesian_count=100)
    scientific = analysis.checkpoint_slice(df, checkpoint_valid_bayesian_count=100)
    assert (scientific["workflow_status"] == "pass").all()
    assert (ops["workflow_status"] != "pass").any()


# ------------------------------------------------------------------
# common-N / cumulative-best / cumulative-GPU-hours
# ------------------------------------------------------------------

def test_common_n_is_min_across_arms():
    df = _small_trial_df()
    sliced = analysis.checkpoint_slice(df, checkpoint_valid_bayesian_count=100)
    assert analysis.common_n(sliced) == 3  # 4 bayesian valid, 3 random valid


def test_cumulative_best_by_order_is_running_max():
    values = [0.30, 0.28, 0.35, 0.33, 0.40]
    result = analysis.cumulative_best_by_order(values)
    assert list(result) == [0.30, 0.30, 0.35, 0.35, 0.40]


def test_cumulative_best_by_order_empty_input():
    assert list(analysis.cumulative_best_by_order([])) == []


def test_cumulative_gpu_hours_by_order_is_running_sum():
    values = [1.0, 2.5, 0.5]
    result = analysis.cumulative_gpu_hours_by_order(values)
    assert list(result) == [1.0, 3.5, 4.0]


# ------------------------------------------------------------------
# top-quartile membership
# ------------------------------------------------------------------

def test_top_quartile_threshold_is_75th_percentile():
    scores = [0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44]
    threshold = analysis.top_quartile_threshold(scores)
    assert threshold == pytest.approx(np.percentile(scores, 75))


def test_is_top_quartile_flags_scores_at_or_above_threshold():
    df = pd.DataFrame({"best_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]})
    mask, threshold = analysis.is_top_quartile(df)
    assert list(df.loc[mask, "best_score"]) == [x for x in df["best_score"] if x >= threshold]
    assert mask.sum() >= 1


# ------------------------------------------------------------------
# continuous near-boundary geometry (log for LR, linear for dropout)
# ------------------------------------------------------------------

def test_near_boundary_mask_log_geometry_for_learning_rate():
    lower, upper = 1e-4, 1e-3
    values = [1.0e-4, 1.05e-4, 5e-4, 9.5e-4, 1.0e-3]
    near_lower, near_upper, positions = analysis.near_boundary_mask(values, lower, upper, geometry="log")
    assert near_lower[0] and near_lower[1]
    assert not near_lower[2]
    assert near_upper[-1] and near_upper[-2]
    assert positions[0] == pytest.approx(0.0)
    assert positions[-1] == pytest.approx(1.0)


def test_near_boundary_mask_linear_geometry_for_dropout():
    lower, upper = 0.0, 0.4
    values = [0.0, 0.02, 0.2, 0.38, 0.4]
    near_lower, near_upper, positions = analysis.near_boundary_mask(values, lower, upper, geometry="linear")
    assert near_lower[0] and near_lower[1]
    assert not near_lower[2]
    assert near_upper[-1] and near_upper[-2]
    assert positions[2] == pytest.approx(0.5)


def test_categorical_extreme_mask_matches_only_extreme_value():
    values = [64, 128, 256, 256, 64]
    mask = analysis.categorical_extreme_mask(values, 256)
    assert list(mask) == [False, False, True, True, False]


# ------------------------------------------------------------------
# categorical occupancy
# ------------------------------------------------------------------

def test_categorical_occupancy_table_counts_per_arm_per_category():
    df = _small_trial_df()
    occ = analysis.categorical_occupancy_table(df)
    hs = occ[(occ["axis"] == "hidden_size") & (occ["search_arm"] == "bayesian")]
    counts_by_value = dict(zip(hs["value"], hs["count"]))
    assert counts_by_value[256] == 2  # proposal_order 2 and 4
    assert counts_by_value[64] == 1
    assert counts_by_value[128] == 1


def test_categorical_occupancy_table_fractions_sum_to_one_per_arm():
    df = _small_trial_df()
    occ = analysis.categorical_occupancy_table(df)
    bayesian_hs = occ[(occ["axis"] == "hidden_size") & (occ["search_arm"] == "bayesian")]
    assert bayesian_hs["fraction"].sum() == pytest.approx(1.0)


# ------------------------------------------------------------------
# late-best detection / best-minus-final
# ------------------------------------------------------------------

def test_late_best_flagged_rows_are_visible_in_valid_trials():
    df = _small_trial_df()
    valid = analysis.valid_trials(df)
    late = valid[valid["late_best"]]
    assert len(late) == 1
    assert late.iloc[0]["best_epoch"] == 11


def test_best_minus_final_field_reflects_derived_instability():
    df = _small_trial_df()
    valid = analysis.valid_trials(df)
    unstable_row = valid.sort_values("best_minus_final", ascending=False).iloc[0]
    assert unstable_row["best_minus_final"] == pytest.approx(0.05)


# ------------------------------------------------------------------
# proposal drift / neighborhood support / boundary-pressure tier
# ------------------------------------------------------------------

def test_proposal_drift_evidence_detects_late_half_concentration():
    # Early half far from lower bound, late half concentrated near it.
    rows = []
    for i in range(1, 5):
        rows.append({"proposal_order": i, "learning_rate": 9e-4})
    for i in range(5, 9):
        rows.append({"proposal_order": i, "learning_rate": 1.02e-4})
    bayesian = pd.DataFrame(rows)
    evidence = analysis.proposal_drift_evidence(bayesian, "learning_rate", "lower", geometry="log",
                                                 lower=1e-4, upper=1e-3)
    assert evidence["drift_toward_boundary"] is True
    assert evidence["late_near_boundary_fraction"] > evidence["early_near_boundary_fraction"]


def test_proposal_drift_evidence_no_drift_when_flat():
    rows = [{"proposal_order": i, "learning_rate": 5e-4} for i in range(1, 9)]
    bayesian = pd.DataFrame(rows)
    evidence = analysis.proposal_drift_evidence(bayesian, "learning_rate", "lower", geometry="log",
                                                 lower=1e-4, upper=1e-3)
    assert evidence["drift_toward_boundary"] is False


def test_neighborhood_support_evidence_categorical_supports_when_extreme_scores_higher():
    df = pd.DataFrame({
        "hidden_size": [64, 64, 128, 256, 256, 256],
        "best_score": [0.30, 0.31, 0.33, 0.40, 0.41, 0.42],
    })
    evidence = analysis.neighborhood_support_evidence(df, "hidden_size", "upper", extreme_value=256)
    assert evidence["supports_direction"] is True
    assert evidence["mean_gap"] > 0


def test_boundary_pressure_tier_strong_requires_all_three_signals():
    assert analysis.boundary_pressure_tier(
        top_quartile_near_fraction=0.6, drift_detected=True, neighborhood_supports=True) == "STRONG"
    assert analysis.boundary_pressure_tier(
        top_quartile_near_fraction=0.6, drift_detected=False, neighborhood_supports=True) == "MODERATE"
    assert analysis.boundary_pressure_tier(
        top_quartile_near_fraction=0.0, drift_detected=False, neighborhood_supports=False) == "WEAK-NONE"


def test_derive_boundary_pressure_table_covers_all_five_axes_both_sides():
    df = _small_trial_df()
    table = analysis.derive_boundary_pressure_table(df)
    assert set(table["axis"]) == {"learning_rate", "embedding_dropout", "output_dropout",
                                   "hidden_size", "batch_size"}
    assert set(table["tier"]) <= {"STRONG", "MODERATE", "WEAK-NONE"}
    # Natural bounds must never be marked expansion-eligible.
    natural_rows = table[table["boundary_nature"] == "natural"]
    assert not natural_rows.empty
    assert not natural_rows["expansion_eligible"].any()


def test_derive_boundary_pressure_table_no_hidden_composite_score_column():
    df = _small_trial_df()
    table = analysis.derive_boundary_pressure_table(df)
    forbidden = {"score", "composite_score", "weighted_score"}
    assert not (forbidden & set(table.columns))


# ------------------------------------------------------------------
# representative-trajectory selection
# ------------------------------------------------------------------

def test_representative_trajectory_selection_prefers_synthetic_archetype_tag():
    df = _small_trial_df()
    df.loc[df["proposal_order"].eq(1) & (df["search_arm"] == "bayesian"), "synthetic_archetype"] = "strong_stable"
    df.loc[df["proposal_order"].eq(4) & (df["search_arm"] == "bayesian"), "synthetic_archetype"] = "late_best"
    picks = analysis.representative_trajectory_selection(df)
    assert picks["strong_stable"] == "trial_bayesian_1"
    assert picks["late_best"] == "trial_bayesian_4"


def test_representative_trajectory_selection_falls_back_to_heuristic_without_tags():
    df = _small_trial_df()
    df["synthetic_archetype"] = None
    picks = analysis.representative_trajectory_selection(df)
    assert picks["late_best"] is not None  # trial_bayesian_4 has late_best=True
    assert picks["unstable"] is not None


# ------------------------------------------------------------------
# no sealed-scope fields / no W&B dependency
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# human-review refinement pass (v002): effect-size drift, pooled vs
# arm-specific occupancy, checkpoint evolution, boundary-band sensitivity,
# top-configurations matrix, natural-boundary interpretation, status header
# ------------------------------------------------------------------

def test_proposal_drift_evidence_exposes_effect_size_fields():
    rows = []
    for i in range(1, 5):
        rows.append({"proposal_order": i, "learning_rate": 9e-4})
    for i in range(5, 9):
        rows.append({"proposal_order": i, "learning_rate": 1.02e-4})
    bayesian = pd.DataFrame(rows)
    evidence = analysis.proposal_drift_evidence(bayesian, "learning_rate", "lower", geometry="log",
                                                 lower=1e-4, upper=1e-3)
    assert evidence["early_median_position"] is not None
    assert evidence["late_median_position"] is not None
    assert evidence["early_median_position"] > evidence["late_median_position"]
    assert evidence["position_shift_toward_boundary"] > 0
    assert evidence["spearman_toward_boundary"] > 0
    assert evidence["effect_size_drift_toward_boundary"] is True
    assert evidence["occupancy_drift_toward_boundary"] is True


def test_proposal_drift_evidence_flags_gradual_downward_sequence_toward_lower_not_upper():
    # A gradual, monotone-ish decline that never crosses the 10% near-boundary
    # band densely enough for the old occupancy-only rule to fire, but is a
    # clear, real directional move -- this is the exact v001 LR/fig06
    # mismatch this refinement pass must fix (§2/§3 of the task).
    log_lower, log_upper = -4.0, -3.0
    n = 20
    values = []
    for i in range(n):
        frac = i / (n - 1)
        log_lr = log_upper - 0.55 * frac * (log_upper - log_lower)  # ends well short of the 10% band
        values.append(10 ** log_lr)
    bayesian = pd.DataFrame({"proposal_order": list(range(1, n + 1)), "learning_rate": values})

    lower_evidence = analysis.proposal_drift_evidence(bayesian, "learning_rate", "lower", geometry="log",
                                                        lower=1e-4, upper=1e-3)
    assert lower_evidence["effect_size_drift_toward_boundary"] is True
    assert lower_evidence["drift_toward_boundary"] is True

    upper_evidence = analysis.proposal_drift_evidence(bayesian, "learning_rate", "upper", geometry="log",
                                                        lower=1e-4, upper=1e-3)
    assert upper_evidence["effect_size_drift_toward_boundary"] is False


def test_proposal_drift_evidence_no_drift_when_flat_has_null_effect_size_fields():
    rows = [{"proposal_order": i, "learning_rate": 5e-4} for i in range(1, 9)]
    bayesian = pd.DataFrame(rows)
    evidence = analysis.proposal_drift_evidence(bayesian, "learning_rate", "lower", geometry="log",
                                                 lower=1e-4, upper=1e-3)
    assert evidence["drift_toward_boundary"] is False
    assert evidence["effect_size_drift_toward_boundary"] is False
    assert evidence["position_shift_toward_boundary"] == pytest.approx(0.0)


def test_derive_boundary_pressure_table_exposes_pooled_and_arm_specific_occupancy():
    df = _small_trial_df()
    table = analysis.derive_boundary_pressure_table(df)
    lr_lower = table[(table["axis"] == "learning_rate") & (table["boundary_side"] == "lower")].iloc[0]
    for col in ("bayesian_top_quartile_near_fraction", "bayesian_top_quartile_near_count",
                "bayesian_top_quartile_n", "random_top_quartile_near_fraction",
                "random_top_quartile_near_count", "random_top_quartile_n"):
        assert col in table.columns
    # arm-specific n's must not exceed the pooled top-quartile n.
    assert lr_lower["bayesian_top_quartile_n"] + lr_lower["random_top_quartile_n"] == lr_lower["top_quartile_n"]


def test_derive_boundary_pressure_table_natural_bound_interpretation_is_not_expansion_pressure():
    df = _small_trial_df()
    table = analysis.derive_boundary_pressure_table(df)
    natural_rows = table[table["boundary_nature"] == "natural"]
    assert len(natural_rows)
    for _, row in natural_rows.iterrows():
        assert row["expansion_eligible"] is False
        assert "preference" in row["interpretation"]
        assert "expansion" not in row["interpretation"].lower() or "no expansion possible" in row["interpretation"]
        assert "pressure" not in row["interpretation"].lower()


def test_boundary_pressure_evolution_marks_first_checkpoint_explicitly():
    df = _small_trial_df()
    current = analysis.derive_boundary_pressure_table(df)
    evolution = analysis.boundary_pressure_evolution(current, None)
    assert (evolution["direction"] == "n/a (first checkpoint)").all()
    assert evolution["tier_previous"].isna().all()


def test_boundary_pressure_evolution_detects_strengthening_and_stable():
    df = _small_trial_df()
    previous = analysis.derive_boundary_pressure_table(df)
    current = previous.copy()
    lr_lower_mask = (current["axis"] == "learning_rate") & (current["boundary_side"] == "lower")
    previous.loc[lr_lower_mask, "tier"] = "WEAK-NONE"
    current.loc[lr_lower_mask, "tier"] = "STRONG"
    hs_upper_mask = (current["axis"] == "hidden_size") & (current["boundary_side"].str.startswith("upper"))
    previous.loc[hs_upper_mask, "tier"] = "MODERATE"
    current.loc[hs_upper_mask, "tier"] = "MODERATE"
    evolution = analysis.boundary_pressure_evolution(current, previous)
    lr_row = evolution[(evolution["axis"] == "learning_rate") & (evolution["boundary_side"] == "lower")].iloc[0]
    assert lr_row["direction"] == "strengthening"
    assert lr_row["tier_previous"] == "WEAK-NONE" and lr_row["tier_current"] == "STRONG"
    hs_row = evolution[(evolution["axis"] == "hidden_size")
                        & (evolution["boundary_side"].str.startswith("upper"))].iloc[0]
    assert hs_row["direction"] == "stable"


def test_boundary_band_sensitivity_reports_all_three_canonical_fractions():
    values = [1.0e-4, 1.05e-4, 1.2e-4, 2.0e-4, 5.0e-4, 9.5e-4]
    table = analysis.boundary_band_sensitivity(values, 1e-4, 1e-3, geometry="log")
    assert list(table["band_fraction"]) == list(analysis.SENSITIVITY_BAND_FRACTIONS)
    canonical_row = table[table["canonical"]].iloc[0]
    assert canonical_row["band_fraction"] == pytest.approx(analysis.BOUNDARY_FRACTION)
    # widening the band can only add points, never remove them.
    counts = list(table["near_lower_count"])
    assert counts == sorted(counts)


def test_boundary_band_sensitivity_does_not_mutate_canonical_fraction():
    values = [1.0e-4, 5.0e-4, 9.5e-4]
    analysis.boundary_band_sensitivity(values, 1e-4, 1e-3, geometry="log")
    assert analysis.BOUNDARY_FRACTION == 0.10


def test_top_configurations_table_ranked_and_capped_per_arm():
    df = _small_trial_df()
    table = analysis.top_configurations_table(df, n_bayesian=2, n_random=1)
    assert (table["search_arm"] == "bayesian").sum() <= 2
    assert (table["search_arm"] == "random_control").sum() <= 1
    assert list(table["rank"]) == list(range(1, len(table) + 1))
    assert list(table["best_score"]) == sorted(table["best_score"], reverse=True)
    assert table["configuration_short_id"].str.len().le(10).all()


def test_top_configurations_table_short_id_distinguishes_rows():
    df = _small_trial_df()
    table = analysis.top_configurations_table(df, n_bayesian=10, n_random=3)
    # configuration_id values here share a long common prefix per arm
    # ("cfg_bayesian_"/"cfg_random_control_"); a short id built from a
    # fixed-length prefix slice would be identical for every row in an arm
    # and useless for telling distinct top configurations apart.
    assert table["configuration_short_id"].nunique() == len(table)


def test_most_pressured_continuous_axis_returns_expandable_axis_side():
    df = _small_trial_df()
    table = analysis.derive_boundary_pressure_table(df)
    axis, side = analysis.most_pressured_continuous_axis(table)
    row = table[(table["axis"] == axis) & (table["boundary_side"] == side)].iloc[0]
    assert bool(row["expansion_eligible"]) is True


def test_checkpoint_status_summary_reports_target_actual_and_incumbents():
    df = _small_trial_df()
    trial_slice = analysis.checkpoint_slice(df, checkpoint_valid_bayesian_count=100)
    ops_slice = analysis.checkpoint_operations_slice(df, checkpoint_valid_bayesian_count=100)
    summary = analysis.checkpoint_status_summary(
        trial_slice, ops_slice, checkpoint_valid_bayesian_count=3, review_name="Boundary Review 1")
    assert summary["review_name"] == "Boundary Review 1"
    assert summary["target_valid_bayesian"] == 3
    assert summary["actual_valid_bayesian"] == 4
    assert summary["bounded_overshoot"] == 1
    assert summary["valid_random_control"] == 3
    assert summary["failed_or_retry_attempts"] == 1
    assert summary["bayesian_incumbent_best"] == pytest.approx(0.370)
    assert summary["common_n"] == 3


def test_review_analysis_has_no_wandb_dependency():
    source = Path(analysis.__file__).read_text(encoding="utf-8")
    assert "import wandb" not in source and "from wandb" not in source


def test_review_analysis_never_references_sealed_scopes():
    source = Path(analysis.__file__).read_text(encoding="utf-8")
    lowered = source.lower()
    for forbidden in ("temporal_test", "spatial_holdout", "california"):
        assert forbidden not in lowered

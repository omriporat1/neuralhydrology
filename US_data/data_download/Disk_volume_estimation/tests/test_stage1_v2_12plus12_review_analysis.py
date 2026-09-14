"""Focused offline tests for the RD1-C2 configuration-level analysis core.

Each unit test isolates one derivation over a tiny hand-built table; the
final test drives a hand-built two-arm table through the public consumer
interface (the RD1-C2 Interface / Consumer Contract Gate vertical). Pure
pandas/numpy over in-memory fixtures -- no W&B, Slurm, Moriah, remote
service, sealed scope, or NeuralHydrology evaluation.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.baseline import sweep_v1_review_analysis as v1_analysis
from src.baseline import stage1_v2_12plus12_review_analysis as rd1
from src.baseline.sweep_v2_six_axis_campaign import SweepV2CampaignError

# ---------------------------------------------------------------------------
# Canonical legal configurations used across tests.
# ---------------------------------------------------------------------------

# All six normalized coordinates == 0.0 (every axis at its lower bound).
CONFIG_ZERO = {
    "learning_rate": 1e-4,
    "hidden_size": 64,
    "embedding_dropout": 0.0,
    "output_dropout": 0.0,
    "batch_size": 128,
    "seq_length": 48,
}
# All six normalized coordinates == 1.0 (every axis at its upper bound).
CONFIG_ONE = {
    "learning_rate": 1e-3,
    "hidden_size": 256,
    "embedding_dropout": 0.4,
    "output_dropout": 0.4,
    "batch_size": 512,
    "seq_length": 120,
}
# All six normalized coordinates == 0.5 (axis midpoints, log-aware).
CONFIG_HALF = {
    "learning_rate": 10 ** -3.5,
    "hidden_size": 128,
    "embedding_dropout": 0.2,
    "output_dropout": 0.2,
    "batch_size": 256,
    "seq_length": 84,
}
CONFIG_P1 = {
    "learning_rate": 3e-4,
    "hidden_size": 128,
    "embedding_dropout": 0.1,
    "output_dropout": 0.15,
    "batch_size": 256,
    "seq_length": 72,
}


def _override(base, **changes):
    return {**base, **changes}


# ---------------------------------------------------------------------------
# A. Six-axis normalization -- all six transforms, boundaries, interior.
# ---------------------------------------------------------------------------


def test_all_lower_bounds_map_to_zero_all_upper_bounds_map_to_one():
    assert rd1.normalized_coordinate_map(CONFIG_ZERO) == {axis: 0.0 for axis in rd1.SIX_AXIS_ORDER}
    assert rd1.normalized_coordinate_map(CONFIG_ONE) == {axis: 1.0 for axis in rd1.SIX_AXIS_ORDER}


@pytest.mark.parametrize(
    "axis, raw_value, expected",
    [
        ("learning_rate", 10 ** -3.5, 0.5),   # log10 midpoint of [1e-4, 1e-3]
        ("hidden_size", 128, 0.5),            # log2 midpoint of [64, 256]
        ("embedding_dropout", 0.1, 0.25),     # linear
        ("output_dropout", 0.3, 0.75),        # linear
        ("batch_size", 256, 0.5),             # log2 midpoint of [128, 512]
        ("seq_length", 84, 0.5),              # linear/ordinal midpoint of [48, 120]
    ],
)
def test_each_axis_interior_transform(axis, raw_value, expected):
    coords = rd1.normalized_coordinate_map(_override(CONFIG_ZERO, **{axis: raw_value}))
    assert coords[axis] == pytest.approx(expected, abs=1e-9)
    # the five untouched axes stay pinned at their lower bound
    for other in rd1.SIX_AXIS_ORDER:
        if other != axis:
            assert coords[other] == 0.0


def test_normalized_coordinates_vector_is_in_fixed_axis_order():
    vec = rd1.normalized_coordinates(CONFIG_HALF)
    assert vec.shape == (6,)
    assert list(vec) == pytest.approx([0.5] * 6, abs=1e-9)


# ---------------------------------------------------------------------------
# Canonical validation goes through the authoritative v2 campaign contract.
# ---------------------------------------------------------------------------


def test_missing_axis_rejected():
    with pytest.raises(SweepV2CampaignError):
        rd1.canonical_configuration({k: v for k, v in CONFIG_ZERO.items() if k != "seq_length"})


def test_off_domain_categorical_rejected():
    with pytest.raises(ValueError):
        rd1.canonical_configuration(_override(CONFIG_ZERO, hidden_size=100))


def test_off_grid_seq_length_rejected():
    with pytest.raises(SweepV2CampaignError):
        rd1.canonical_configuration(_override(CONFIG_ZERO, seq_length=50))


def test_out_of_range_continuous_rejected():
    with pytest.raises(ValueError):
        rd1.canonical_configuration(_override(CONFIG_ZERO, learning_rate=5e-3))


def test_bool_value_rejected():
    with pytest.raises(SweepV2CampaignError):
        rd1.canonical_configuration(_override(CONFIG_ZERO, hidden_size=True))


def test_canonical_coordinate_key_is_stable_and_distinguishes_configs():
    assert rd1.canonical_coordinate_key(CONFIG_HALF) == rd1.canonical_coordinate_key(dict(CONFIG_HALF))
    assert rd1.canonical_coordinate_key(CONFIG_HALF) != rd1.canonical_coordinate_key(CONFIG_ONE)


# ---------------------------------------------------------------------------
# B. RMS configuration distance.
# ---------------------------------------------------------------------------


def test_distance_zero_self_symmetry_and_known_values():
    zero = np.zeros(6)
    ones = np.ones(6)
    one_axis = np.array([1.0, 0, 0, 0, 0, 0])
    assert rd1.configuration_distance(zero, zero) == 0.0
    assert rd1.configuration_distance(zero, ones) == pytest.approx(1.0)
    assert rd1.configuration_distance(zero, one_axis) == pytest.approx(math.sqrt(1 / 6))
    assert rd1.configuration_distance(one_axis, zero) == rd1.configuration_distance(zero, one_axis)


def test_distance_between_configs_matches_known_half_to_one():
    # every axis differs by 0.5 -> sqrt(mean(6 * 0.25)) == 0.5
    assert rd1.configuration_distance_between_configs(CONFIG_HALF, CONFIG_ONE) == pytest.approx(0.5, abs=1e-9)


def test_distance_rejects_wrong_length_vector():
    with pytest.raises(ValueError):
        rd1.configuration_distance([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# C. Bayesian novelty.
# ---------------------------------------------------------------------------


def test_first_proposal_novelty_is_missing_and_duplicate_is_zero():
    coords = [np.zeros(6), np.ones(6), np.zeros(6)]
    novelty = rd1.bayesian_novelty_series(coords)
    assert math.isnan(novelty[0])            # first valid proposal: undefined
    assert novelty[1] == pytest.approx(1.0)
    assert novelty[2] == 0.0                 # duplicate coordinate -> novelty 0


# ---------------------------------------------------------------------------
# D. Bayesian incumbent distance -- deterministic first-achieved semantics.
# ---------------------------------------------------------------------------


def test_first_achieved_incumbent_indices_tie_does_not_replace():
    # exact objective tie at positions 1 and 2: incumbent stays the FIRST achiever
    assert rd1.first_achieved_incumbent_indices([0.40, 0.44, 0.44, 0.42]) == [None, 0, 1, 1]


def test_incumbent_distance_frame_first_row_missing_and_identity_exposed():
    coords = [
        rd1.normalized_coordinates(CONFIG_P1),
        rd1.normalized_coordinates(CONFIG_HALF),
        rd1.normalized_coordinates(CONFIG_HALF),
        rd1.normalized_coordinates(CONFIG_ONE),
    ]
    frame = rd1.bayesian_incumbent_distance_frame(
        coords, [0.35, 0.44, 0.44, 0.30], proposal_ids=["P1", "P2", "P3", "P4"],
        proposal_orders=[1, 2, 3, 4],
    )
    assert math.isnan(frame.loc[0, "incumbent_distance"])
    assert pd.isna(frame.loc[0, "incumbent_proposal_id_before"])  # undefined for first proposal
    # P3 duplicates P2 (the first achiever of 0.44) -> distance 0
    assert frame.loc[2, "incumbent_proposal_id_before"] == "P2"
    assert frame.loc[2, "incumbent_distance"] == 0.0
    # P4 incumbent is still P2 (tie at P3 did not replace)
    assert frame.loc[3, "incumbent_proposal_id_before"] == "P2"
    assert frame.loc[3, "incumbent_distance"] == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# E. Threshold-free top-k.
# ---------------------------------------------------------------------------


def test_top3_floor_mean_median_and_completeness():
    summary = rd1.top_k_summary([0.5, 0.9, 0.7, 0.1, 0.8], k=3, identities=["a", "b", "c", "d", "e"])
    assert summary["n_available"] == 5
    assert summary["is_complete"] is True
    assert summary["best"] == pytest.approx(0.9)
    assert summary["floor"] == pytest.approx(0.7)
    assert summary["mean"] == pytest.approx(0.8)
    assert summary["median"] == pytest.approx(0.8)
    assert [row["identity"] for row in summary["selected"]] == ["b", "e", "c"]


def test_top_k_tie_breaks_on_earlier_proposal_order():
    summary = rd1.top_k_summary([0.5, 0.5, 0.3], k=2, identities=["first", "second", "third"])
    assert [row["identity"] for row in summary["selected"]] == ["first", "second"]


def test_top_k_fewer_than_three_is_explicit():
    summary = rd1.top_k_summary([0.4, 0.2], k=3)
    assert summary["n_selected"] == 2
    assert summary["is_complete"] is False
    assert summary["floor"] == pytest.approx(0.2)
    assert summary["mean"] == pytest.approx(0.3)


def test_top_k_empty_is_nan():
    summary = rd1.top_k_summary([], k=3)
    assert summary["n_selected"] == 0
    assert math.isnan(summary["best"]) and math.isnan(summary["floor"])
    assert math.isnan(summary["mean"]) and math.isnan(summary["median"])


# ---------------------------------------------------------------------------
# F. P2-referenced descriptive near-incumbent yield (one-sided).
# ---------------------------------------------------------------------------


def test_near_incumbent_primary_counts_scores_at_or_above_floor_including_above_anchor():
    scores = [0.45, 0.4300, 0.40]  # 0.45 exceeds the anchor and still counts
    result = rd1.near_incumbent_yield(scores, margin=rd1.NEAR_INCUMBENT_PRIMARY_MARGIN)
    assert result["performance_floor"] == pytest.approx(rd1.P2_NEAR_INCUMBENT_ANCHOR - 0.01)
    assert result["margin_kind"] == "descriptive_one_sided_near_or_better"
    assert [m["near_or_better"] for m in result["membership"]] == [True, True, False]
    assert result["n_near_or_better"] == 2
    above_anchor = result["membership"][0]
    assert above_anchor["objective"] > result["anchor"] and above_anchor["near_or_better"] is True


def test_near_incumbent_sensitivity_is_stricter_than_primary():
    scores = [0.45, 0.4300, 0.40]
    result = rd1.near_incumbent_yield(scores, margin=rd1.NEAR_INCUMBENT_SENSITIVITY_MARGIN)
    assert result["performance_floor"] == pytest.approx(rd1.P2_NEAR_INCUMBENT_ANCHOR - 0.005)
    assert [m["near_or_better"] for m in result["membership"]] == [True, False, False]


def test_cumulative_near_incumbent_count_is_monotone():
    counts = rd1.cumulative_near_incumbent_count(
        [0.45, 0.43, 0.40, 0.44], margin=rd1.NEAR_INCUMBENT_PRIMARY_MARGIN
    )
    assert list(counts) == [1, 2, 2, 3]


# ---------------------------------------------------------------------------
# I. Duplicate-coordinate / proposal-vs-unique accounting.
# ---------------------------------------------------------------------------


def test_duplicate_accounting_preserves_every_proposal():
    configs = [CONFIG_HALF, CONFIG_ONE, dict(CONFIG_HALF), CONFIG_ZERO]
    acc = rd1.duplicate_coordinate_accounting(configs, identities=["P1", "P2", "P3", "P4"])
    assert acc["n_proposals"] == 4
    assert acc["n_unique_coordinates"] == 3
    assert acc["n_duplicate_proposals"] == 1
    assert list(acc["duplicate_groups"].values()) == [["P1", "P3"]]


# ---------------------------------------------------------------------------
# J. Exact GPU-hour / compute-coverage semantics.
# ---------------------------------------------------------------------------


def test_complete_scope_gives_authoritative_sum():
    summary = rd1.exact_compute_scope_summary([1.0, 2.0, 3.0])
    assert summary["coverage_complete"] is True
    assert summary["authoritative_exact_gpu_hours"] == pytest.approx(6.0)


def test_missing_compute_is_never_silently_zero():
    summary = rd1.exact_compute_scope_summary([1.0, None, 3.0])
    assert summary["covered_configuration_count"] == 2
    assert summary["coverage_complete"] is False
    assert math.isnan(summary["authoritative_exact_gpu_hours"])
    assert summary["known_partial_gpu_hours_sum"] == pytest.approx(4.0)


def test_prefix_coverage_becomes_unavailable_after_first_gap():
    frame = rd1.exact_compute_coverage_by_prefix([1.0, None, 2.0])
    assert frame.loc[0, "cumulative_exact_gpu_hours"] == pytest.approx(1.0)
    assert math.isnan(frame.loc[1, "cumulative_exact_gpu_hours"])
    assert math.isnan(frame.loc[2, "cumulative_exact_gpu_hours"])


# ---------------------------------------------------------------------------
# H. First-achieved final incumbent / post-incumbent descriptors.
# ---------------------------------------------------------------------------


def test_first_achieved_final_incumbent_is_earliest_attainer_of_final_max():
    assert rd1.first_achieved_final_incumbent_index([0.3, 0.5, 0.4, 0.5, 0.2]) == 1


def test_post_incumbent_summarises_only_strictly_subsequent_proposals():
    result = rd1.post_incumbent_descriptors(
        [0.3, 0.5, 0.4, 0.5, 0.2], identities=["a", "b", "c", "d", "e"]
    )
    assert result["final_incumbent_identity"] == "b"
    assert result["n_subsequent_proposals"] == 3
    assert result["subsequent_objective_top3"]["n_available"] == 3


# ---------------------------------------------------------------------------
# Reuse discipline: the one legacy primitive is reused, not reimplemented.
# ---------------------------------------------------------------------------


def test_cumulative_best_primitive_is_the_shared_v1_object():
    assert rd1.cumulative_best_by_order is v1_analysis.cumulative_best_by_order


def test_no_wandb_or_sealed_scope_dependency_in_module_source():
    source = (rd1.__file__ and open(rd1.__file__, "r", encoding="utf-8").read()).lower()
    assert "import wandb" not in source and "from wandb" not in source
    for sealed in ("temporal_test", "spatial_holdout", "california"):
        assert sealed not in source


# ---------------------------------------------------------------------------
# Consumer-contract failures.
# ---------------------------------------------------------------------------


def _proposal_row(*, arm, order, config, objective, gpu_hours=None, proposal_id=None, configuration_id=None):
    row = {
        "search_arm": arm,
        "proposal_order": order,
        "proposal_id": proposal_id if proposal_id is not None else f"{arm}_{order}",
        "configuration_id": configuration_id if configuration_id is not None else f"cfg_{arm}_{order}",
        "objective_score": objective,
        **{axis: config[axis] for axis in rd1.SIX_AXIS_ORDER},
    }
    if gpu_hours is not None:
        row["exact_gpu_hours"] = gpu_hours
    return row


def test_missing_required_column_fails_clearly():
    rows = [_proposal_row(arm="bayesian", order=1, config=CONFIG_HALF, objective=0.44)]
    frame = pd.DataFrame(rows).drop(columns=["configuration_id"])
    with pytest.raises(rd1.ReviewTableContractError):
        rd1.analyze_review_table(frame)


def test_non_finite_objective_fails_clearly():
    rows = [
        _proposal_row(arm="bayesian", order=1, config=CONFIG_HALF, objective=0.44),
        _proposal_row(arm="bayesian", order=2, config=CONFIG_ONE, objective=float("nan")),
    ]
    with pytest.raises(rd1.ReviewTableContractError):
        rd1.analyze_review_table(rows)


def test_illegal_configuration_row_fails_clearly():
    rows = [_proposal_row(arm="bayesian", order=1, config=_override(CONFIG_HALF, hidden_size=100), objective=0.44)]
    with pytest.raises(rd1.ReviewTableContractError):
        rd1.analyze_review_table(rows)


def test_duplicate_proposal_order_within_arm_fails_clearly():
    rows = [
        _proposal_row(arm="bayesian", order=1, config=CONFIG_HALF, objective=0.44),
        _proposal_row(arm="bayesian", order=1, config=CONFIG_ONE, objective=0.30),
    ]
    with pytest.raises(rd1.ReviewTableContractError):
        rd1.analyze_review_table(rows)


def test_analysis_available_when_exact_compute_column_absent():
    rows = [
        _proposal_row(arm="bayesian", order=1, config=CONFIG_P1, objective=0.35),
        _proposal_row(arm="bayesian", order=2, config=CONFIG_HALF, objective=0.44),
    ]
    analysis = rd1.analyze_review_table(rows)
    arm = analysis.arm("bayesian")
    assert arm.exact_compute_scope_summary["coverage_complete"] is False
    assert math.isnan(arm.exact_compute_scope_summary["authoritative_exact_gpu_hours"])
    assert arm.top3["n_available"] == 2  # objective/geometry analysis still available


# ---------------------------------------------------------------------------
# Section 10 vertical: hand-built two-arm table through the public interface.
# ---------------------------------------------------------------------------


def _vertical_table():
    bayesian = [
        _proposal_row(arm="bayesian", order=1, config=CONFIG_P1, objective=0.35, gpu_hours=2.0, proposal_id="P1"),
        _proposal_row(arm="bayesian", order=2, config=CONFIG_HALF, objective=0.44, gpu_hours=2.1, proposal_id="P2"),
        _proposal_row(arm="bayesian", order=3, config=dict(CONFIG_HALF), objective=0.44, gpu_hours=2.2, proposal_id="P3"),
        _proposal_row(arm="bayesian", order=4, config=CONFIG_ONE, objective=0.30, gpu_hours=2.0, proposal_id="P4"),
        _proposal_row(arm="bayesian", order=5, config=CONFIG_ZERO, objective=0.43, gpu_hours=1.9, proposal_id="P5"),
    ]
    random_control = [
        _proposal_row(arm="random_control", order=1, config=CONFIG_P1, objective=0.36, gpu_hours=2.0, proposal_id="R1"),
        _proposal_row(arm="random_control", order=2, config=CONFIG_HALF,
                      objective=rd1.P2_NEAR_INCUMBENT_ANCHOR, proposal_id="R2"),
        _proposal_row(arm="random_control", order=3, config=CONFIG_ONE, objective=0.41, gpu_hours=2.5, proposal_id="R3"),
        _proposal_row(arm="random_control", order=4, config=CONFIG_ZERO, objective=0.43, proposal_id="R4"),
    ]
    return pd.DataFrame(bayesian + random_control)


def test_vertical_consumer_contract_bayesian_arm():
    analysis = rd1.analyze_review_table(_vertical_table())
    bay = analysis.arm("bayesian")
    per = bay.per_proposal

    # whole-sequence descriptors, in proposal order
    assert list(per["proposal_id"]) == ["P1", "P2", "P3", "P4", "P5"]
    assert list(per["cumulative_best"]) == pytest.approx([0.35, 0.44, 0.44, 0.44, 0.44])
    assert list(per["is_new_incumbent"]) == [True, True, False, False, False]
    assert list(per["cumulative_near_incumbent_primary"]) == [0, 1, 2, 2, 3]
    assert list(per["cumulative_near_incumbent_sensitivity"]) == [0, 1, 2, 2, 2]

    # first valid Bayesian proposal: missing novelty + missing incumbent distance
    assert math.isnan(per.loc[0, "bayesian_novelty"])
    assert math.isnan(per.loc[0, "incumbent_distance"])
    assert pd.isna(per.loc[0, "incumbent_proposal_id_before"])

    # duplicate canonical coordinate at P3 -> novelty 0 and incumbent distance 0
    assert per.loc[2, "bayesian_novelty"] == 0.0
    assert per.loc[2, "incumbent_distance"] == 0.0
    # deterministic first-achieved incumbent: P4/P5 still measure against P2
    assert list(per["incumbent_proposal_id_before"])[1:] == ["P1", "P2", "P2", "P2"]
    assert per.loc[3, "incumbent_distance"] == pytest.approx(0.5, abs=1e-9)
    assert per.loc[4, "incumbent_distance"] == pytest.approx(0.5, abs=1e-9)

    # P2 six-axis coords are the known all-0.5 point
    assert [per.loc[1, f"coord_{axis}"] for axis in rd1.SIX_AXIS_ORDER] == pytest.approx([0.5] * 6, abs=1e-9)

    # threshold-free top-3 over the whole arm
    assert bay.top3["best"] == pytest.approx(0.44)
    assert bay.top3["floor"] == pytest.approx(0.43)
    assert bay.top3["median"] == pytest.approx(0.44)

    # P2-referenced descriptive near-incumbent yield
    assert bay.near_incumbent_primary["n_near_or_better"] == 3
    assert bay.near_incumbent_sensitivity["n_near_or_better"] == 2

    # duplicate / proposal-vs-unique accounting (duplicates preserved)
    assert bay.duplicate_coordinate_accounting["n_proposals"] == 5
    assert bay.duplicate_coordinate_accounting["n_unique_coordinates"] == 4
    assert list(bay.duplicate_coordinate_accounting["duplicate_groups"].values()) == [["P2", "P3"]]

    # first-achieved final incumbent + post-incumbent descriptors
    assert bay.post_incumbent["final_incumbent_identity"] == "P2"
    assert bay.post_incumbent["n_subsequent_proposals"] == 3
    assert bay.post_incumbent["subsequent_near_incumbent_primary"]["n_near_or_better"] == 2
    assert bay.post_incumbent["n_subsequent_new_coordinates"] == 2

    # complete exact-compute scope -> authoritative GPU-hour sum
    assert bay.exact_compute_scope_summary["coverage_complete"] is True
    assert bay.exact_compute_scope_summary["authoritative_exact_gpu_hours"] == pytest.approx(10.2)
    assert bay.post_incumbent["subsequent_exact_gpu_hours"] == pytest.approx(6.1)

    # ---- RD1-C2 corrective-pass demonstrations, same public interface ----

    # (9)+(11) proposal-aligned cumulative exact compute, identity attached,
    # authoritative because the whole Bayesian prefix has exact accounting.
    assert bay.exact_compute_available is True
    assert list(per["exact_gpu_hours"]) == pytest.approx([2.0, 2.1, 2.2, 2.0, 1.9])
    assert list(per["exact_gpu_hours_prefix_complete"]) == [True, True, True, True, True]
    assert list(per["cumulative_exact_gpu_hours"]) == pytest.approx([2.0, 4.1, 6.3, 8.3, 10.2])

    # (12) transparent cumulative near-incumbent-yield per GPU-hour: count / GPU
    yld = list(per["cumulative_near_incumbent_primary_yield_per_gpu_hour"])
    assert yld[0] == pytest.approx(0.0)             # 0 near / 2.0 GPU-h
    assert yld[1] == pytest.approx(1 / 4.1)
    assert yld[4] == pytest.approx(3 / 10.2)

    # (13) champion-discovery compute: first achiever of the arm's eventual best
    champ = bay.champion_discovery_compute
    assert champ["champion_proposal_id"] == "P2" and champ["champion_proposal_order"] == 2
    assert champ["champion_objective"] == pytest.approx(0.44)
    assert champ["prefix_complete_through_champion"] is True
    assert champ["cumulative_exact_gpu_hours_through_champion"] == pytest.approx(4.1)

    # (14) compute through discovery of the complete selected final top-3 set
    disc = bay.top_k_discovery_compute
    assert [m["proposal_id"] for m in disc["members"]] == ["P2", "P3", "P5"]
    assert disc["is_complete"] is True
    assert disc["latest_discovery_proposal_order"] == 5
    assert disc["prefix_complete_through_full_top_k_discovery"] is True
    assert disc["cumulative_exact_gpu_hours_through_full_top_k_discovery"] == pytest.approx(10.2)

    # (15) post-incumbent compute stays strictly subsequent to P2 and exact
    assert bay.post_incumbent["subsequent_exact_compute"]["coverage_complete"] is True


def test_vertical_consumer_contract_pooled_p2_margins_across_both_arms():
    # (7)+(8) both frozen P2 margins, pooled arm-neutral over the whole table
    analysis = rd1.analyze_review_table(_vertical_table())

    pooled_primary = analysis.pooled_near_incumbent("primary")
    pooled_sensitivity = analysis.pooled_near_incumbent("sensitivity")

    assert pooled_primary["margin"] == pytest.approx(0.01)
    assert pooled_sensitivity["margin"] == pytest.approx(0.005)
    assert pooled_primary["scope"] == "pooled_across_arms"

    # pooled counts across both arms for each margin
    assert pooled_primary["n_total"] == 9
    assert pooled_primary["n_near_or_better"] == 5     # P2,P3,P5 + R2,R4
    assert pooled_sensitivity["n_near_or_better"] == 3  # P2,P3 + R2

    # membership carries the arm plus proposal / configuration identities
    members = {(m["search_arm"], m["proposal_id"]) for m in pooled_primary["near_or_better_members"]}
    assert members == {
        ("bayesian", "P2"), ("bayesian", "P3"), ("bayesian", "P5"),
        ("random_control", "R2"), ("random_control", "R4"),
    }
    for m in pooled_primary["near_or_better_members"]:
        assert m["configuration_id"] and m["search_arm"] in {"bayesian", "random_control"}

    # a score that *exceeds* P2 is still pooled-included (one-sided floor)
    r2 = next(m for m in pooled_primary["membership"] if m["proposal_id"] == "R2")
    assert r2["objective"] >= pooled_primary["anchor"] and r2["near_or_better"] is True

    # no fabricated pooled ordering / pooled cumulative sequence
    assert "cumulative" not in pooled_primary
    assert "pooled_proposal_order" not in pooled_primary


def test_vertical_consumer_contract_random_arm_has_no_incumbent_learning_semantics():
    analysis = rd1.analyze_review_table(_vertical_table())
    rnd = analysis.arm("random_control")

    # random-control order carries no optimizer-learning interpretation
    assert rnd.novelty is None
    assert rnd.incumbent_distance_frame is None
    assert "bayesian_novelty" not in rnd.per_proposal.columns
    assert "incumbent_distance" not in rnd.per_proposal.columns

    # descriptive coverage/yield still available for the random arm
    assert rnd.near_incumbent_primary["n_near_or_better"] == 2
    assert rnd.near_incumbent_sensitivity["n_near_or_better"] == 1

    # (10) deliberately incomplete exact GPU-hour coverage -> unavailable, never zero
    assert rnd.exact_compute_scope_summary["coverage_complete"] is False
    assert math.isnan(rnd.exact_compute_scope_summary["authoritative_exact_gpu_hours"])
    assert rnd.exact_compute_scope_summary["known_partial_gpu_hours_sum"] == pytest.approx(4.5)
    coverage = rnd.exact_compute_coverage_by_prefix
    assert coverage.loc[0, "cumulative_exact_gpu_hours"] == pytest.approx(2.0)
    assert math.isnan(coverage.loc[1, "cumulative_exact_gpu_hours"])

    # (11) proposal-aligned cumulative compute does not "resume" after the gap
    per = rnd.per_proposal
    assert per.loc[0, "cumulative_exact_gpu_hours"] == pytest.approx(2.0)
    assert list(per["exact_gpu_hours_prefix_complete"]) == [True, False, False, False]
    assert all(math.isnan(v) for v in list(per["cumulative_exact_gpu_hours"])[1:])

    # (12) near-incumbent yield/GPU unavailable (not fabricated) once compute is
    assert all(math.isnan(v) for v in list(per["cumulative_near_incumbent_primary_yield_per_gpu_hour"])[1:])

    # (13)+(14) champion / final-top-3 discovery compute: champion R2's own GPU
    # value is missing, so cumulative-through-champion is explicitly unavailable
    assert rnd.champion_discovery_compute["champion_proposal_id"] == "R2"
    assert rnd.champion_discovery_compute["prefix_complete_through_champion"] is False
    assert math.isnan(rnd.champion_discovery_compute["cumulative_exact_gpu_hours_through_champion"])
    assert rnd.top_k_discovery_compute["prefix_complete_through_full_top_k_discovery"] is False
    assert math.isnan(
        rnd.top_k_discovery_compute["cumulative_exact_gpu_hours_through_full_top_k_discovery"]
    )

    # (16) still no Bayesian-learning semantics anywhere on the random arm
    assert "bayesian_novelty" not in per.columns and "incumbent_distance" not in per.columns


# ---------------------------------------------------------------------------
# Finding A -- exact GPU-hour validation at the review-table boundary.
# ---------------------------------------------------------------------------


def _gpu_table(second_gpu):
    rows = [
        _proposal_row(arm="bayesian", order=1, config=CONFIG_P1, objective=0.35, gpu_hours=2.0),
        _proposal_row(arm="bayesian", order=2, config=CONFIG_HALF, objective=0.44, gpu_hours=1.0),
    ]
    frame = pd.DataFrame(rows)
    frame["exact_gpu_hours"] = frame["exact_gpu_hours"].astype(object)
    frame.loc[1, "exact_gpu_hours"] = second_gpu
    return frame


@pytest.mark.parametrize(
    "bad", [-1.0, float("inf"), float("-inf"), True, "2.5", "n/a"]
)
def test_populated_exact_gpu_hours_must_be_finite_nonnegative_number(bad):
    # negative / +inf / -inf / boolean / numeric-looking string / non-numeric
    # string are all populated-but-malformed -> clear consumer-contract failure
    with pytest.raises(rd1.ReviewTableContractError):
        rd1.analyze_review_table(_gpu_table(bad))


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA])
def test_missing_exact_gpu_hours_marker_is_accepted_as_unavailable_not_zero(missing):
    analysis = rd1.analyze_review_table(_gpu_table(missing))
    arm = analysis.arm("bayesian")
    scope = arm.exact_compute_scope_summary
    assert scope["covered_configuration_count"] == 1
    assert scope["coverage_complete"] is False
    assert math.isnan(scope["authoritative_exact_gpu_hours"])       # never 0.0
    assert scope["known_partial_gpu_hours_sum"] == pytest.approx(2.0)
    assert arm.top3["n_available"] == 2                             # objective analysis intact
    assert math.isnan(arm.per_proposal.loc[1, "exact_gpu_hours"])


def test_ordinary_finite_nonnegative_exact_gpu_hours_is_accepted():
    arm = rd1.analyze_review_table(_gpu_table(2.5)).arm("bayesian")
    assert arm.exact_compute_scope_summary["coverage_complete"] is True
    assert arm.exact_compute_scope_summary["authoritative_exact_gpu_hours"] == pytest.approx(4.5)
    assert list(arm.per_proposal["cumulative_exact_gpu_hours"]) == pytest.approx([2.0, 4.5])


# ---------------------------------------------------------------------------
# Finding D -- proposal / configuration identity validation.
# ---------------------------------------------------------------------------


def _identity_rows():
    return [
        _proposal_row(arm="bayesian", order=1, config=CONFIG_P1, objective=0.35,
                      proposal_id="P1", configuration_id="C1"),
        _proposal_row(arm="bayesian", order=2, config=CONFIG_HALF, objective=0.44,
                      proposal_id="P2", configuration_id="C2"),
    ]


@pytest.mark.parametrize("bad_id", [None, "", "   "])
def test_missing_or_blank_proposal_id_fails_clearly(bad_id):
    frame = pd.DataFrame(_identity_rows())
    frame["proposal_id"] = frame["proposal_id"].astype(object)
    frame.loc[1, "proposal_id"] = bad_id
    with pytest.raises(rd1.ReviewTableContractError):
        rd1.analyze_review_table(frame)


@pytest.mark.parametrize("bad_id", [None, "", "   "])
def test_missing_or_blank_configuration_id_fails_clearly(bad_id):
    frame = pd.DataFrame(_identity_rows())
    frame["configuration_id"] = frame["configuration_id"].astype(object)
    frame.loc[1, "configuration_id"] = bad_id
    with pytest.raises(rd1.ReviewTableContractError):
        rd1.analyze_review_table(frame)


def test_duplicate_proposal_id_within_arm_fails_clearly():
    rows = [
        _proposal_row(arm="bayesian", order=1, config=CONFIG_P1, objective=0.35, proposal_id="DUP"),
        _proposal_row(arm="bayesian", order=2, config=CONFIG_HALF, objective=0.44, proposal_id="DUP"),
    ]
    with pytest.raises(rd1.ReviewTableContractError):
        rd1.analyze_review_table(rows)


def test_same_proposal_id_across_different_arms_is_allowed():
    rows = [
        _proposal_row(arm="bayesian", order=1, config=CONFIG_P1, objective=0.35, proposal_id="X1"),
        _proposal_row(arm="random_control", order=1, config=CONFIG_HALF, objective=0.44, proposal_id="X1"),
    ]
    analysis = rd1.analyze_review_table(rows)  # uniqueness is required *within* an arm only
    assert set(analysis.arms) == {"bayesian", "random_control"}


def test_configuration_id_with_conflicting_canonical_coordinates_fails():
    rows = [
        _proposal_row(arm="bayesian", order=1, config=CONFIG_HALF, objective=0.35,
                      proposal_id="P1", configuration_id="SHARED"),
        _proposal_row(arm="bayesian", order=2, config=CONFIG_ONE, objective=0.44,
                      proposal_id="P2", configuration_id="SHARED"),
    ]
    with pytest.raises(rd1.ReviewTableContractError):
        rd1.analyze_review_table(rows)


def test_repeated_configuration_id_with_same_canonical_coordinate_is_accepted():
    rows = [
        _proposal_row(arm="bayesian", order=1, config=CONFIG_HALF, objective=0.35,
                      proposal_id="P1", configuration_id="SAME"),
        _proposal_row(arm="random_control", order=1, config=dict(CONFIG_HALF), objective=0.44,
                      proposal_id="R1", configuration_id="SAME"),
    ]
    analysis = rd1.analyze_review_table(rows)
    assert analysis.arm("bayesian").per_proposal.loc[0, "configuration_id"] == "SAME"


def test_distinct_configuration_ids_sharing_a_canonical_coordinate_are_duplicate_proposals():
    rows = [
        _proposal_row(arm="bayesian", order=1, config=CONFIG_HALF, objective=0.35,
                      proposal_id="P1", configuration_id="CFG_A"),
        _proposal_row(arm="bayesian", order=2, config=dict(CONFIG_HALF), objective=0.44,
                      proposal_id="P2", configuration_id="CFG_B"),
    ]
    acc = rd1.analyze_review_table(rows).arm("bayesian").duplicate_coordinate_accounting
    assert acc["n_proposals"] == 2
    assert acc["n_unique_coordinates"] == 1
    assert list(acc["duplicate_groups"].values()) == [["P1", "P2"]]


# ---------------------------------------------------------------------------
# Finding C -- compute / yield consumer contract (proposal-identified).
# ---------------------------------------------------------------------------


def test_proposal_aligned_cumulative_compute_stays_unavailable_after_a_gap():
    rows = [
        _proposal_row(arm="bayesian", order=1, config=CONFIG_P1, objective=0.35, gpu_hours=2.0, proposal_id="P1"),
        _proposal_row(arm="bayesian", order=2, config=CONFIG_HALF, objective=0.44, proposal_id="P2"),  # no GPU
        _proposal_row(arm="bayesian", order=3, config=CONFIG_ONE, objective=0.30, gpu_hours=3.0, proposal_id="P3"),
    ]
    per = rd1.analyze_review_table(pd.DataFrame(rows)).arm("bayesian").per_proposal
    assert list(per["proposal_id"]) == ["P1", "P2", "P3"]           # identity / order retained
    assert per.loc[0, "cumulative_exact_gpu_hours"] == pytest.approx(2.0)
    assert bool(per.loc[0, "exact_gpu_hours_prefix_complete"]) is True
    assert math.isnan(per.loc[1, "cumulative_exact_gpu_hours"])
    assert math.isnan(per.loc[2, "cumulative_exact_gpu_hours"])     # never resumes after the gap
    assert per.loc[2, "exact_gpu_hours"] == pytest.approx(3.0)      # proposal's own value retained


def test_top_k_discovery_compute_handles_fewer_than_three_members():
    rows = [
        _proposal_row(arm="bayesian", order=1, config=CONFIG_P1, objective=0.35, gpu_hours=2.0, proposal_id="P1"),
        _proposal_row(arm="bayesian", order=2, config=CONFIG_HALF, objective=0.44, gpu_hours=2.0, proposal_id="P2"),
    ]
    disc = rd1.analyze_review_table(pd.DataFrame(rows)).arm("bayesian").top_k_discovery_compute
    assert disc["requested_k"] == 3
    assert disc["n_selected"] == 2 and disc["is_complete"] is False
    assert [m["proposal_id"] for m in disc["members"]] == ["P2", "P1"]
    assert disc["cumulative_exact_gpu_hours_through_full_top_k_discovery"] == pytest.approx(4.0)


def test_exact_compute_summaries_report_unavailable_when_gpu_column_absent():
    rows = [
        _proposal_row(arm="bayesian", order=1, config=CONFIG_P1, objective=0.35, proposal_id="P1"),
        _proposal_row(arm="bayesian", order=2, config=CONFIG_HALF, objective=0.44, proposal_id="P2"),
    ]
    analysis = rd1.analyze_review_table(rows)
    arm = analysis.arm("bayesian")
    assert analysis.exact_compute_available is False
    assert arm.exact_compute_available is False
    assert all(math.isnan(v) for v in arm.per_proposal["cumulative_exact_gpu_hours"])
    assert not any(arm.per_proposal["exact_gpu_hours_prefix_complete"])
    assert math.isnan(arm.champion_discovery_compute["cumulative_exact_gpu_hours_through_champion"])
    assert math.isnan(
        arm.top_k_discovery_compute["cumulative_exact_gpu_hours_through_full_top_k_discovery"]
    )
    # objective / geometry / yield analyses remain fully available
    assert arm.top3["best"] == pytest.approx(0.44)
    assert arm.near_incumbent_primary["n_near_or_better"] == 1

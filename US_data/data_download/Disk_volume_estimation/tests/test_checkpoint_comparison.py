"""Section 10-13 tests for src/baseline/checkpoint_comparison.py.

Pure-function tests against small synthetic per-basin NSE fixtures shaped
exactly like src.baseline.pilot_diagnostic_eval.evaluate_diagnostic_checkpoint's
return value -- no real NH data, no real training, no real checkpoint I/O.
"""
import math

import numpy as np
import pytest

from src.baseline.checkpoint_comparison import (
    CheckpointComparisonError,
    build_n_vs_one_comparison,
    cadence_sensitivity_view,
    derive_trajectory_summary,
    paired_diff_challenger_vs_reference,
    summarize_candidate_epoch,
)


def _payload(epoch, basin_nse, *, evaluation_role="official", authoritative=True):
    return {
        "epoch": epoch,
        "evaluation_role": evaluation_role,
        "authoritative": authoritative,
        "raw_space_metrics": {
            "per_basin": [{"basin_id": basin_id, "nse": nse} for basin_id, nse in basin_nse.items()],
        },
    }


# ----------------------------------------------------------------------
# summarize_candidate_epoch
# ----------------------------------------------------------------------


def test_summarize_candidate_epoch_basic_stats():
    payload = _payload(3, {"A": 0.1, "B": 0.5, "C": 0.9, "D": -0.2})
    row = summarize_candidate_epoch(payload, candidate_id="cand1")

    assert row["candidate_id"] == "cand1"
    assert row["epoch"] == 3
    assert row["n_basins"] == 4
    assert row["median_nse"] == pytest.approx(np.median([0.1, 0.5, 0.9, -0.2]))
    assert row["q25_nse"] == pytest.approx(np.percentile([0.1, 0.5, 0.9, -0.2], 25))
    assert row["q75_nse"] == pytest.approx(np.percentile([0.1, 0.5, 0.9, -0.2], 75))
    assert row["frac_nse_gt_0"] == pytest.approx(0.75)
    assert row["evaluation_role"] == "official"
    assert row["authoritative"] is True


def test_summarize_candidate_epoch_passes_through_optional_metadata():
    payload = _payload(1, {"A": 0.2}, evaluation_role="retrospective_diagnostic", authoritative=False)
    row = summarize_candidate_epoch(
        payload,
        candidate_id="cand1",
        mean_training_loss=0.0123,
        cumulative_updates=25000,
        secondary_diagnostics={"kge_median": 0.3},
    )
    assert row["mean_training_loss"] == pytest.approx(0.0123)
    assert row["cumulative_updates"] == 25000
    assert row["secondary_diagnostics"] == {"kge_median": 0.3}
    assert row["evaluation_role"] == "retrospective_diagnostic"
    assert row["authoritative"] is False


def test_summarize_candidate_epoch_defaults_optional_metadata_to_none():
    payload = _payload(1, {"A": 0.2})
    row = summarize_candidate_epoch(payload, candidate_id="cand1")
    assert row["mean_training_loss"] is None
    assert row["cumulative_updates"] is None
    assert row["secondary_diagnostics"] is None


def test_summarize_candidate_epoch_ignores_non_finite_nse_in_stats():
    payload = _payload(1, {"A": 0.5, "B": float("nan")})
    row = summarize_candidate_epoch(payload, candidate_id="cand1")
    assert row["n_basins"] == 1
    assert row["median_nse"] == pytest.approx(0.5)


def test_summarize_candidate_epoch_all_non_finite_yields_nan_stats():
    payload = _payload(1, {"A": float("nan"), "B": float("-inf")})
    row = summarize_candidate_epoch(payload, candidate_id="cand1")
    assert row["n_basins"] == 0
    assert math.isnan(row["median_nse"])
    assert math.isnan(row["frac_nse_gt_0"])


def test_summarize_candidate_epoch_raises_on_duplicate_basin_id():
    payload = {
        "epoch": 1,
        "evaluation_role": "official",
        "authoritative": True,
        "raw_space_metrics": {
            "per_basin": [{"basin_id": "A", "nse": 0.1}, {"basin_id": "A", "nse": 0.5}],
        },
    }
    with pytest.raises(CheckpointComparisonError, match="duplicate basin_id"):
        summarize_candidate_epoch(payload, candidate_id="cand1")


# ----------------------------------------------------------------------
# paired_diff_challenger_vs_reference
# ----------------------------------------------------------------------


def test_paired_diff_basic_stats():
    challenger = _payload(6, {"A": 0.6, "B": 0.4, "C": 0.2})
    reference = _payload(6, {"A": 0.5, "B": 0.5, "C": 0.1})
    row = paired_diff_challenger_vs_reference(
        challenger, reference, challenger_id="chal", reference_id="ref"
    )
    assert row["challenger_id"] == "chal"
    assert row["reference_id"] == "ref"
    assert row["epoch"] == 6
    assert row["matched_basin_count"] == 3
    diffs = [0.1, -0.1, 0.1]
    assert row["median_diff_nse"] == pytest.approx(np.median(diffs))
    assert row["q25_diff_nse"] == pytest.approx(np.percentile(diffs, 25))
    assert row["q75_diff_nse"] == pytest.approx(np.percentile(diffs, 75))
    assert row["frac_challenger_better"] == pytest.approx(2 / 3)
    assert row["frac_reference_better"] == pytest.approx(1 / 3)
    assert row["frac_tied"] == pytest.approx(0.0)


def test_paired_diff_ties_are_counted():
    challenger = _payload(1, {"A": 0.5, "B": 0.3})
    reference = _payload(1, {"A": 0.5, "B": 0.3})
    row = paired_diff_challenger_vs_reference(
        challenger, reference, challenger_id="chal", reference_id="ref"
    )
    assert row["frac_tied"] == pytest.approx(1.0)
    assert row["median_diff_nse"] == pytest.approx(0.0)


def test_paired_diff_raises_when_challenger_and_reference_ids_are_equal():
    payload = _payload(1, {"A": 0.5})
    with pytest.raises(CheckpointComparisonError, match="must differ"):
        paired_diff_challenger_vs_reference(payload, payload, challenger_id="same", reference_id="same")


def test_paired_diff_raises_on_mixed_epochs():
    challenger = _payload(1, {"A": 0.5})
    reference = _payload(2, {"A": 0.5})
    with pytest.raises(CheckpointComparisonError, match="mixed epochs"):
        paired_diff_challenger_vs_reference(
            challenger, reference, challenger_id="chal", reference_id="ref"
        )


def test_paired_diff_raises_on_missing_basin():
    challenger = _payload(1, {"A": 0.5, "B": 0.3})
    reference = _payload(1, {"A": 0.5, "C": 0.1})
    with pytest.raises(CheckpointComparisonError, match="matched-basin-identity violation"):
        paired_diff_challenger_vs_reference(
            challenger, reference, challenger_id="chal", reference_id="ref"
        )


def test_paired_diff_raises_on_population_size_mismatch():
    challenger = _payload(1, {"A": 0.5})
    reference = _payload(1, {"A": 0.5, "B": 0.1})
    with pytest.raises(CheckpointComparisonError, match="matched-basin-identity violation"):
        paired_diff_challenger_vs_reference(
            challenger, reference, challenger_id="chal", reference_id="ref"
        )


def test_paired_diff_raises_on_duplicate_basin_id_in_either_side():
    good = _payload(1, {"A": 0.5})
    dup = {
        "epoch": 1,
        "evaluation_role": "official",
        "authoritative": True,
        "raw_space_metrics": {
            "per_basin": [{"basin_id": "A", "nse": 0.1}, {"basin_id": "A", "nse": 0.5}],
        },
    }
    with pytest.raises(CheckpointComparisonError, match="duplicate basin_id"):
        paired_diff_challenger_vs_reference(dup, good, challenger_id="chal", reference_id="ref")


# ----------------------------------------------------------------------
# build_n_vs_one_comparison
# ----------------------------------------------------------------------


def _candidate_payloads(basin_nse_by_epoch, **kwargs):
    return [_payload(epoch, basin_nse, **kwargs) for epoch, basin_nse in basin_nse_by_epoch.items()]


def _standard_payloads_by_candidate(epochs=(1, 2)):
    basins = {"A": None, "B": None, "C": None}
    by_candidate = {}
    for candidate_id, offset in [("ref", 0.0), ("chal1", 0.05), ("chal2", -0.05)]:
        per_epoch = {}
        for epoch in epochs:
            per_epoch[epoch] = {b: 0.1 * epoch + offset for b in basins}
        by_candidate[candidate_id] = _candidate_payloads(per_epoch)
    return by_candidate


def test_build_n_vs_one_comparison_assembles_summary_and_diff_tables():
    payloads = _standard_payloads_by_candidate(epochs=(1, 2))
    result = build_n_vs_one_comparison(
        payloads, reference_id="ref", challenger_ids=["chal1", "chal2"], epochs=[1, 2]
    )
    assert result["reference_id"] == "ref"
    assert result["challenger_ids"] == ["chal1", "chal2"]
    assert result["epochs"] == [1, 2]
    assert len(result["candidate_epoch_summaries"]) == 3 * 2
    assert len(result["paired_diffs"]) == 2 * 2
    diff_keys = {(row["challenger_id"], row["epoch"]) for row in result["paired_diffs"]}
    assert diff_keys == {("chal1", 1), ("chal1", 2), ("chal2", 1), ("chal2", 2)}


def test_build_n_vs_one_comparison_threads_loss_and_updates_metadata():
    payloads = _standard_payloads_by_candidate(epochs=(1,))
    result = build_n_vs_one_comparison(
        payloads,
        reference_id="ref",
        challenger_ids=["chal1"],
        epochs=[1],
        loss_by_candidate_epoch={("chal1", 1): 0.02},
        updates_by_candidate_epoch={("chal1", 1): 12500},
    )
    chal_row = next(
        row for row in result["candidate_epoch_summaries"] if row["candidate_id"] == "chal1"
    )
    assert chal_row["mean_training_loss"] == pytest.approx(0.02)
    assert chal_row["cumulative_updates"] == 12500
    ref_row = next(row for row in result["candidate_epoch_summaries"] if row["candidate_id"] == "ref")
    assert ref_row["mean_training_loss"] is None


def test_build_n_vs_one_comparison_raises_when_reference_is_also_a_challenger():
    payloads = _standard_payloads_by_candidate(epochs=(1,))
    with pytest.raises(CheckpointComparisonError, match="must not also appear"):
        build_n_vs_one_comparison(
            payloads, reference_id="ref", challenger_ids=["ref", "chal1"], epochs=[1]
        )


def test_build_n_vs_one_comparison_raises_on_duplicate_challenger_ids():
    payloads = _standard_payloads_by_candidate(epochs=(1,))
    with pytest.raises(CheckpointComparisonError, match="duplicate challenger_ids"):
        build_n_vs_one_comparison(
            payloads, reference_id="ref", challenger_ids=["chal1", "chal1"], epochs=[1]
        )


def test_build_n_vs_one_comparison_raises_on_empty_epochs():
    payloads = _standard_payloads_by_candidate(epochs=(1,))
    with pytest.raises(CheckpointComparisonError, match="epochs must be non-empty"):
        build_n_vs_one_comparison(payloads, reference_id="ref", challenger_ids=["chal1"], epochs=[])


def test_build_n_vs_one_comparison_raises_on_missing_candidate():
    payloads = _standard_payloads_by_candidate(epochs=(1,))
    del payloads["chal2"]
    with pytest.raises(CheckpointComparisonError, match="missing required candidate_id"):
        build_n_vs_one_comparison(
            payloads, reference_id="ref", challenger_ids=["chal1", "chal2"], epochs=[1]
        )


def test_build_n_vs_one_comparison_raises_on_missing_epoch_for_a_candidate():
    payloads = _standard_payloads_by_candidate(epochs=(1, 2))
    payloads["chal1"] = [p for p in payloads["chal1"] if p["epoch"] != 2]
    with pytest.raises(CheckpointComparisonError, match="missing diagnostic payload"):
        build_n_vs_one_comparison(
            payloads, reference_id="ref", challenger_ids=["chal1"], epochs=[1, 2]
        )


def test_build_n_vs_one_comparison_raises_on_duplicate_epoch_payload_for_a_candidate():
    payloads = _standard_payloads_by_candidate(epochs=(1,))
    payloads["chal1"] = payloads["chal1"] + payloads["chal1"]
    with pytest.raises(CheckpointComparisonError, match="more than one diagnostic payload"):
        build_n_vs_one_comparison(payloads, reference_id="ref", challenger_ids=["chal1"], epochs=[1])


def test_build_n_vs_one_comparison_propagates_matched_basin_identity_failure():
    payloads = _standard_payloads_by_candidate(epochs=(1,))
    payloads["chal1"] = [_payload(1, {"A": 0.1, "Z": 0.2})]
    with pytest.raises(CheckpointComparisonError, match="matched-basin-identity violation"):
        build_n_vs_one_comparison(payloads, reference_id="ref", challenger_ids=["chal1"], epochs=[1])


# ----------------------------------------------------------------------
# derive_trajectory_summary
# ----------------------------------------------------------------------


def _summary_rows(candidate_id, median_by_epoch):
    return [
        {
            "candidate_id": candidate_id,
            "epoch": epoch,
            "median_nse": median,
            "q25_nse": median - 0.05,
            "q75_nse": median + 0.05,
            "frac_nse_gt_0": 1.0,
            "evaluation_role": "official",
            "authoritative": True,
            "mean_training_loss": None,
            "cumulative_updates": None,
            "secondary_diagnostics": None,
        }
        for epoch, median in median_by_epoch.items()
    ]


def test_derive_trajectory_summary_increasing_late_window():
    rows = _summary_rows("cand1", {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5, 6: 0.6})
    summary = derive_trajectory_summary(rows, candidate_id="cand1")
    assert summary["best_observed_median_nse_checkpoint_epoch"] == 6
    assert summary["best_observed_median_nse"] == pytest.approx(0.6)
    assert summary["epoch_6_median_nse"] == pytest.approx(0.6)
    assert summary["late_window_epochs_available"] == [4, 5, 6]
    assert summary["median_of_epoch_medians_late_window"] == pytest.approx(0.5)
    assert summary["range_of_epoch_medians_late_window"] == pytest.approx(0.2)
    assert summary["late_trajectory_direction"] == "increasing"


def test_derive_trajectory_summary_decreasing_late_window():
    rows = _summary_rows("cand1", {4: 0.5, 5: 0.4, 6: 0.3})
    summary = derive_trajectory_summary(rows, candidate_id="cand1")
    assert summary["late_trajectory_direction"] == "decreasing"


def test_derive_trajectory_summary_flat_late_window():
    rows = _summary_rows("cand1", {4: 0.4, 5: 0.4, 6: 0.4})
    summary = derive_trajectory_summary(rows, candidate_id="cand1")
    assert summary["late_trajectory_direction"] == "flat"


def test_derive_trajectory_summary_insufficient_data_with_single_late_epoch():
    rows = _summary_rows("cand1", {1: 0.1, 6: 0.6})
    summary = derive_trajectory_summary(rows, candidate_id="cand1")
    assert summary["late_window_epochs_available"] == [6]
    assert summary["late_trajectory_direction"] == "insufficient_data"
    assert summary["epoch_6_median_nse"] == pytest.approx(0.6)


def test_derive_trajectory_summary_best_checkpoint_need_not_be_epoch_6():
    rows = _summary_rows("cand1", {1: 0.1, 2: 0.9, 3: 0.2, 4: 0.3, 5: 0.2, 6: 0.1})
    summary = derive_trajectory_summary(rows, candidate_id="cand1")
    assert summary["best_observed_median_nse_checkpoint_epoch"] == 2
    assert summary["best_observed_median_nse"] == pytest.approx(0.9)
    assert summary["epoch_6_median_nse"] == pytest.approx(0.1)


def test_derive_trajectory_summary_raises_on_unknown_candidate_id():
    rows = _summary_rows("cand1", {1: 0.1})
    with pytest.raises(CheckpointComparisonError, match="no candidate_epoch_summaries rows"):
        derive_trajectory_summary(rows, candidate_id="does-not-exist")


def test_derive_trajectory_summary_never_returns_a_composite_score_or_rank_key():
    rows = _summary_rows("cand1", {4: 0.1, 5: 0.2, 6: 0.3})
    summary = derive_trajectory_summary(rows, candidate_id="cand1")
    forbidden_keys = {"score", "rank", "winner", "composite_score", "is_best"}
    assert forbidden_keys.isdisjoint(summary.keys())


# ----------------------------------------------------------------------
# cadence_sensitivity_view
# ----------------------------------------------------------------------


def test_cadence_sensitivity_view_all_vs_sparse_cadences():
    rows = _summary_rows("cand1", {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5, 6: 0.6})
    view = cadence_sensitivity_view(
        rows,
        candidate_id="cand1",
        cadences={
            "all_epochs": [1, 2, 3, 4, 5, 6],
            "every_other_epoch": [2, 4, 6],
            "screening_cadence_only": [3, 6],
        },
    )
    assert view["candidate_id"] == "cand1"
    all_view = view["cadences"]["all_epochs"]
    assert all_view["epochs_available"] == [1, 2, 3, 4, 5, 6]
    assert all_view["best_observed_median_nse_checkpoint_epoch"] == 6
    assert all_view["median_of_epoch_medians"] == pytest.approx(np.median([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))

    sparse_view = view["cadences"]["screening_cadence_only"]
    assert sparse_view["epochs_available"] == [3, 6]
    assert sparse_view["median_of_epoch_medians"] == pytest.approx(np.median([0.3, 0.6]))


def test_cadence_sensitivity_view_reports_missing_epochs_without_raising():
    rows = _summary_rows("cand1", {2: 0.2, 4: 0.4})
    view = cadence_sensitivity_view(
        rows, candidate_id="cand1", cadences={"all_epochs": [1, 2, 3, 4, 5, 6]}
    )
    cadence_view = view["cadences"]["all_epochs"]
    assert cadence_view["epochs_available"] == [2, 4]
    assert cadence_view["epochs_missing"] == [1, 3, 5, 6]
    assert cadence_view["median_of_epoch_medians"] == pytest.approx(np.median([0.2, 0.4]))


def test_cadence_sensitivity_view_raises_on_unknown_candidate_id():
    rows = _summary_rows("cand1", {1: 0.1})
    with pytest.raises(CheckpointComparisonError, match="no candidate_epoch_summaries rows"):
        cadence_sensitivity_view(rows, candidate_id="does-not-exist", cadences={"all": [1]})

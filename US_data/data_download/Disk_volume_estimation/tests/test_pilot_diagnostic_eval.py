"""Focused tests for src/baseline/pilot_diagnostic_eval.py (Sections 6-7 of
the LR-A learning-rate range-characterization campaign).

Covers: off-cadence epochs (1, 2, 4, 5) getting the explicit
"retrospective_diagnostic" tag with authoritative=False/stopping_eligible=False
unconditionally; on-cadence epochs (3, 6) delegating byte-for-byte to
evaluate_screening_checkpoint with only an added evaluation_role="official"
tag; the same guard rails (non-validation period, empty basin list) as
evaluate_screening_checkpoint; evaluate_all_diagnostic_checkpoints's ordered,
non-sorting, non-deduping list assembly; and that this module never imports
or can call record_screening_event (an off-cadence retrospective evaluation
must never mutate early-stopping state).

Uses the same real-fixture pattern as test_pilot_screening_eval.py -- a
synthetic package plus a hand-written perfect (NSE=1.0) validation_results.p
pickle per epoch -- so ensure_validation_results finds the expected pickle
already present and evaluate_checkpoint_fn (real NH inference) is never
invoked. No torch/NeuralHydrology needed.
"""
from __future__ import annotations

import pytest

from src.baseline import pilot_diagnostic_eval
from src.baseline.pilot_diagnostic_eval import (
    OFFICIAL_EVALUATION_ROLE,
    RETROSPECTIVE_EVALUATION_ROLE,
    evaluate_all_diagnostic_checkpoints,
    evaluate_diagnostic_checkpoint,
)
from src.baseline.pilot_lead06_config import load_pilot_policy
from src.baseline.pilot_screening_eval import PilotScreeningEvalError, SCREENING_METRIC_SCOPE

from tests._pilot_support import (
    PILOT_POLICY_PATH,
    build_full_union_package,
    pick_development_basins,
    write_perfect_validation_results,
)

TARGET_VARIABLE = "qobs_mm_per_h_lead06"
LEAD_HOURS = 6


@pytest.fixture
def policy():
    return load_pilot_policy(PILOT_POLICY_PATH)


def _build_run_dir_with_perfect_checkpoints(tmp_path, basins, epochs):
    package_root = tmp_path / "package"
    build_full_union_package(package_root, ts_basin_ids=basins)
    run_dir = tmp_path / "run"
    for epoch in epochs:
        write_perfect_validation_results(run_dir, epoch, basins, package_root)
    return package_root, run_dir


# --- off-cadence epochs: retrospective, non-authoritative, never stopping ---


@pytest.mark.parametrize("epoch", [1, 2, 4, 5])
def test_off_cadence_epoch_is_tagged_retrospective_and_non_authoritative(tmp_path, policy, epoch):
    basins = pick_development_basins(5)
    package_root, run_dir = _build_run_dir_with_perfect_checkpoints(tmp_path, basins, [epoch])

    result = evaluate_diagnostic_checkpoint(
        nh_run_dir=run_dir,
        epoch=epoch,
        package_root=package_root,
        target_variable=TARGET_VARIABLE,
        lead_hours=LEAD_HOURS,
        screening_basin_ids=basins,
        pilot_policy=policy,
    )

    assert result["scope"] == SCREENING_METRIC_SCOPE
    assert result["authoritative"] is False
    assert result["epoch"] == epoch
    assert result["epoch_role"] == "not_a_screening_epoch"
    assert result["evaluation_role"] == RETROSPECTIVE_EVALUATION_ROLE
    assert result["stopping_eligible"] is False
    assert result["n_screening_basins_requested"] == len(basins)
    assert result["primary_metric_median"] == pytest.approx(1.0, abs=1e-6)
    assert "per_basin" in result["raw_space_metrics"]


# --- on-cadence epochs: delegate to evaluate_screening_checkpoint -----------


def test_diagnostic_only_epoch_3_delegates_with_official_tag(tmp_path, policy):
    basins = pick_development_basins(5)
    package_root, run_dir = _build_run_dir_with_perfect_checkpoints(tmp_path, basins, [3])

    result = evaluate_diagnostic_checkpoint(
        nh_run_dir=run_dir,
        epoch=3,
        package_root=package_root,
        target_variable=TARGET_VARIABLE,
        lead_hours=LEAD_HOURS,
        screening_basin_ids=basins,
        pilot_policy=policy,
    )

    assert result["epoch_role"] == "diagnostic_only"
    assert result["evaluation_role"] == OFFICIAL_EVALUATION_ROLE
    assert result["stopping_eligible"] is False
    assert result["primary_metric_median"] == pytest.approx(1.0, abs=1e-6)


def test_stopping_eligible_epoch_6_delegates_with_official_tag(tmp_path, policy):
    basins = pick_development_basins(5)
    package_root, run_dir = _build_run_dir_with_perfect_checkpoints(tmp_path, basins, [6])

    result = evaluate_diagnostic_checkpoint(
        nh_run_dir=run_dir,
        epoch=6,
        package_root=package_root,
        target_variable=TARGET_VARIABLE,
        lead_hours=LEAD_HOURS,
        screening_basin_ids=basins,
        pilot_policy=policy,
    )

    assert result["epoch_role"] == "stopping_eligible"
    assert result["evaluation_role"] == OFFICIAL_EVALUATION_ROLE
    assert result["stopping_eligible"] is True
    assert result["primary_metric_median"] == pytest.approx(1.0, abs=1e-6)


# --- guard rails, mirroring evaluate_screening_checkpoint's own ------------


def test_rejects_non_validation_period(tmp_path, policy):
    basins = pick_development_basins(5)
    package_root, run_dir = _build_run_dir_with_perfect_checkpoints(tmp_path, basins, [1])

    with pytest.raises(PilotScreeningEvalError):
        evaluate_diagnostic_checkpoint(
            nh_run_dir=run_dir,
            epoch=1,
            package_root=package_root,
            target_variable=TARGET_VARIABLE,
            lead_hours=LEAD_HOURS,
            screening_basin_ids=basins,
            pilot_policy=policy,
            period="test",
        )


def test_rejects_empty_basin_list(tmp_path, policy):
    with pytest.raises(PilotScreeningEvalError):
        evaluate_diagnostic_checkpoint(
            nh_run_dir=tmp_path / "run",
            epoch=1,
            package_root=tmp_path / "package",
            target_variable=TARGET_VARIABLE,
            lead_hours=LEAD_HOURS,
            screening_basin_ids=[],
            pilot_policy=policy,
        )


# --- evaluate_all_diagnostic_checkpoints: ordered assembly ------------------


def test_evaluate_all_diagnostic_checkpoints_covers_epochs_1_through_6(tmp_path, policy):
    basins = pick_development_basins(5)
    epochs = [1, 2, 3, 4, 5, 6]
    package_root, run_dir = _build_run_dir_with_perfect_checkpoints(tmp_path, basins, epochs)

    results = evaluate_all_diagnostic_checkpoints(
        nh_run_dir=run_dir,
        epochs=epochs,
        package_root=package_root,
        target_variable=TARGET_VARIABLE,
        lead_hours=LEAD_HOURS,
        screening_basin_ids=basins,
        pilot_policy=policy,
    )

    assert [r["epoch"] for r in results] == epochs
    expected_roles = [
        RETROSPECTIVE_EVALUATION_ROLE,
        RETROSPECTIVE_EVALUATION_ROLE,
        OFFICIAL_EVALUATION_ROLE,
        RETROSPECTIVE_EVALUATION_ROLE,
        RETROSPECTIVE_EVALUATION_ROLE,
        OFFICIAL_EVALUATION_ROLE,
    ]
    assert [r["evaluation_role"] for r in results] == expected_roles
    expected_stopping_eligible = [False, False, False, False, False, True]
    assert [r["stopping_eligible"] for r in results] == expected_stopping_eligible
    for r in results:
        assert r["primary_metric_median"] == pytest.approx(1.0, abs=1e-6)


def test_evaluate_all_diagnostic_checkpoints_preserves_caller_supplied_order(tmp_path, policy):
    basins = pick_development_basins(5)
    package_root, run_dir = _build_run_dir_with_perfect_checkpoints(tmp_path, basins, [6, 1])

    results = evaluate_all_diagnostic_checkpoints(
        nh_run_dir=run_dir,
        epochs=[6, 1],
        package_root=package_root,
        target_variable=TARGET_VARIABLE,
        lead_hours=LEAD_HOURS,
        screening_basin_ids=basins,
        pilot_policy=policy,
    )

    assert [r["epoch"] for r in results] == [6, 1]
    assert results[0]["evaluation_role"] == OFFICIAL_EVALUATION_ROLE
    assert results[1]["evaluation_role"] == RETROSPECTIVE_EVALUATION_ROLE


# --- never touches early-stopping state ------------------------------------


def test_module_never_imports_record_screening_event():
    assert "record_screening_event" not in vars(pilot_diagnostic_eval)

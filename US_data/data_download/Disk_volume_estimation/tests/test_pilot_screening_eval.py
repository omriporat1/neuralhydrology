"""Focused tests for src/baseline/pilot_screening_eval.py (task item 10).

Covers: epoch-role classification (diagnostic-only / stopping-eligible /
not-a-screening-epoch) across the pilot's fixed 3-epoch cadence;
load_validated_screening_basin_ids's re-derivation-not-trust-caller
behavior and its rejection of a sealed-population basin; and
evaluate_screening_checkpoint's off-cadence/empty-list rejection, its
scope/authoritative tagging, and a deterministic perfect-NSE numeric check
using a synthetic package + hand-written validation_results.p (no NH/torch
needed).
"""
from __future__ import annotations

import dataclasses

import pytest

from src.baseline.pilot_lead06_config import load_pilot_policy
from src.baseline.pilot_screening_eval import (
    PilotScreeningEvalError,
    PRIMARY_METRIC_NAME,
    SCREENING_METRIC_SCOPE,
    classify_screening_epoch_role,
    evaluate_screening_checkpoint,
    load_validated_screening_basin_ids,
)
from src.baseline.splits import sha256_of

from tests._pilot_support import (
    PILOT_POLICY_PATH,
    REAL_DEVELOPMENT,
    REAL_SPATIAL_HOLDOUT,
    SPLITS_DIR,
    build_full_union_package,
    pick_development_basins,
    write_perfect_validation_results,
    write_screening_basin_ids_file,
)


@pytest.fixture
def policy():
    return load_pilot_policy(PILOT_POLICY_PATH)


# --- classify_screening_epoch_role -------------------------------------------

@pytest.mark.parametrize("epoch,expected_role", [
    (1, "not_a_screening_epoch"),
    (2, "not_a_screening_epoch"),
    (3, "diagnostic_only"),
    (4, "not_a_screening_epoch"),
    (5, "not_a_screening_epoch"),
    (6, "stopping_eligible"),
    (7, "not_a_screening_epoch"),
    (9, "stopping_eligible"),
    (12, "stopping_eligible"),
    (36, "stopping_eligible"),
])
def test_classify_screening_epoch_role(policy, epoch, expected_role):
    assert classify_screening_epoch_role(epoch, policy) == expected_role


def test_epoch_3_is_diagnostic_even_though_on_cadence(policy):
    # 3 % 3 == 0 (on-cadence) but < stopping_eligible_from_epoch(6) -- must
    # never be classified stopping_eligible.
    assert policy.diagnostic_only_epoch == 3
    assert policy.stopping_eligible_from_epoch == 6
    assert classify_screening_epoch_role(3, policy) == "diagnostic_only"


# --- load_validated_screening_basin_ids: re-derivation, not trust-caller ----

def test_load_validated_screening_basin_ids_matches_committed_file(tmp_path, policy):
    package_root = tmp_path / "package"
    build_full_union_package(package_root)
    ids = load_validated_screening_basin_ids(
        pilot_policy=policy, package_root=package_root, splits_dir=SPLITS_DIR,
    )
    assert len(ids) > 0
    assert set(ids) <= set(REAL_DEVELOPMENT)


def test_load_validated_screening_basin_ids_rejects_tampered_file_with_holdout_basin(tmp_path, policy):
    package_root = tmp_path / "package"
    build_full_union_package(package_root)

    tampered = REAL_DEVELOPMENT[:350] + [REAL_SPATIAL_HOLDOUT[0]]
    tampered_path = write_screening_basin_ids_file(tmp_path / "tampered_screening.txt", tampered)
    tampered_policy = dataclasses.replace(
        policy,
        screening_basin_ids_path=str(tampered_path),
        screening_expected_count=len(tampered),
        screening_expected_sha256=sha256_of(tampered_path),
    )
    with pytest.raises(Exception):
        load_validated_screening_basin_ids(
            pilot_policy=tampered_policy, package_root=package_root, splits_dir=SPLITS_DIR,
        )


# --- evaluate_screening_checkpoint: guard rails ------------------------------

def test_evaluate_screening_checkpoint_rejects_empty_basin_list(tmp_path, policy):
    with pytest.raises(PilotScreeningEvalError):
        evaluate_screening_checkpoint(
            run_dir=tmp_path / "run",
            epoch=6,
            package_root=tmp_path / "package",
            target_variable="qobs_mm_per_h_lead06",
            lead_hours=6,
            screening_basin_ids=[],
            pilot_policy=policy,
        )


def test_evaluate_screening_checkpoint_rejects_off_cadence_epoch(tmp_path, policy):
    with pytest.raises(PilotScreeningEvalError):
        evaluate_screening_checkpoint(
            run_dir=tmp_path / "run",
            epoch=7,
            package_root=tmp_path / "package",
            target_variable="qobs_mm_per_h_lead06",
            lead_hours=6,
            screening_basin_ids=["01234567"],
            pilot_policy=policy,
        )


def test_evaluate_screening_checkpoint_rejects_non_validation_period(tmp_path, policy):
    # This pilot's screening/evaluation path must never invoke the sealed
    # temporal-test period or a spatial-holdout/California evaluation --
    # only period="validation" is ever legitimate here (task item 6).
    with pytest.raises(PilotScreeningEvalError):
        evaluate_screening_checkpoint(
            run_dir=tmp_path / "run",
            epoch=6,
            package_root=tmp_path / "package",
            target_variable="qobs_mm_per_h_lead06",
            lead_hours=6,
            screening_basin_ids=["01234567"],
            pilot_policy=policy,
            period="test",
        )


# --- evaluate_screening_checkpoint: real numeric check via perfect fixture --

def _build_run_dir_with_perfect_checkpoint(tmp_path, basins, epoch):
    package_root = tmp_path / "package"
    build_full_union_package(package_root, ts_basin_ids=basins)
    run_dir = tmp_path / "run"
    write_perfect_validation_results(run_dir, epoch, basins, package_root)
    return package_root, run_dir


def test_evaluate_screening_checkpoint_diagnostic_epoch_perfect_nse(tmp_path, policy):
    basins = pick_development_basins(5)
    package_root, run_dir = _build_run_dir_with_perfect_checkpoint(tmp_path, basins, epoch=3)

    result = evaluate_screening_checkpoint(
        run_dir=run_dir,
        epoch=3,
        package_root=package_root,
        target_variable="qobs_mm_per_h_lead06",
        lead_hours=6,
        screening_basin_ids=basins,
        pilot_policy=policy,
    )
    assert result["scope"] == SCREENING_METRIC_SCOPE
    assert result["authoritative"] is False
    assert result["epoch_role"] == "diagnostic_only"
    assert result["stopping_eligible"] is False
    assert result["primary_metric_name"] == PRIMARY_METRIC_NAME
    assert result["primary_metric_median"] == pytest.approx(1.0, abs=1e-6)
    assert result["n_screening_basins_requested"] == len(basins)


def test_evaluate_screening_checkpoint_stopping_eligible_epoch_perfect_nse(tmp_path, policy):
    basins = pick_development_basins(5)
    package_root, run_dir = _build_run_dir_with_perfect_checkpoint(tmp_path, basins, epoch=6)

    result = evaluate_screening_checkpoint(
        run_dir=run_dir,
        epoch=6,
        package_root=package_root,
        target_variable="qobs_mm_per_h_lead06",
        lead_hours=6,
        screening_basin_ids=basins,
        pilot_policy=policy,
    )
    assert result["epoch_role"] == "stopping_eligible"
    assert result["stopping_eligible"] is True
    assert result["primary_metric_median"] == pytest.approx(1.0, abs=1e-6)

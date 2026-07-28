"""Focused tests for src/baseline/pilot_full_validation.py (task item 9's
readiness interface, task item 10's tests).

This module is a READINESS interface only -- it is never invoked against
the real certified package or a real training run anywhere in this
project. These tests exercise its code paths purely against synthetic
fixtures (tests/_pilot_support.py), which is exactly the "no test/holdout
access, interface-only" testing the task requires: no sealed data, no real
package, no Moriah/GPU involved.

Covers: load_validated_full_population_basin_ids re-derives the real
development population; build_pilot_full_validation_bundle's train ==
validation == test == full development population (distinct from the
screening bundle's proper-subset validation scope) and its distinct
FULL_VALIDATION_POPULATION_ROLE; evaluate_full_validation_checkpoint's
empty-list rejection, authoritative=True/scope tagging, promoted_from_run_id
passthrough, perfect-NSE numeric check, and the (deliberate) absence of any
screening-style cadence restriction.
"""
from __future__ import annotations

import dataclasses

import pytest

from src.baseline.pilot_lead06_config import (
    SCREENING_VALIDATION_POPULATION_ROLE,
    build_pilot_bundle,
    load_pilot_policy,
)
from src.baseline.pilot_full_validation import (
    FULL_VALIDATION_METRIC_SCOPE,
    FULL_VALIDATION_POPULATION_ROLE,
    PilotFullValidationError,
    build_pilot_full_validation_bundle,
    evaluate_full_validation_checkpoint,
    load_validated_full_population_basin_ids,
)
from src.baseline.splits import sha256_of

from tests._pilot_support import (
    BASELINE_POLICY_PATH,
    PILOT_POLICY_PATH,
    REAL_DEVELOPMENT,
    SPLITS_DIR,
    build_full_union_package,
    pick_development_basins,
    write_perfect_validation_results,
    write_screening_basin_ids_file,
)


@pytest.fixture
def pilot_policy_with_screening(tmp_path):
    base = load_pilot_policy(PILOT_POLICY_PATH)
    screening_path = write_screening_basin_ids_file(tmp_path / "screening.txt", REAL_DEVELOPMENT[:350])
    return dataclasses.replace(
        base,
        screening_basin_ids_path=str(screening_path),
        screening_expected_count=350,
        screening_expected_sha256=sha256_of(screening_path),
    )


# --- load_validated_full_population_basin_ids --------------------------------

def test_load_validated_full_population_basin_ids_matches_development_split(tmp_path):
    package_root = tmp_path / "package"
    build_full_union_package(package_root)
    ids = load_validated_full_population_basin_ids(package_root=package_root, splits_dir=SPLITS_DIR)
    assert sorted(ids) == sorted(REAL_DEVELOPMENT)


# --- build_pilot_full_validation_bundle: full population, distinct role ----

def test_build_pilot_full_validation_bundle_uses_full_population_for_all_three_scopes(
    tmp_path, pilot_policy_with_screening,
):
    package_root = tmp_path / "package"
    build_full_union_package(package_root)

    bundle = build_pilot_full_validation_bundle(
        pilot_policy=pilot_policy_with_screening,
        run_id="raw_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=package_root,
        splits_dir=SPLITS_DIR,
    )
    assert sorted(bundle.train_basin_ids) == sorted(REAL_DEVELOPMENT)
    assert sorted(bundle.validation_basin_ids) == sorted(REAL_DEVELOPMENT)
    assert sorted(bundle.test_basin_ids) == sorted(REAL_DEVELOPMENT)
    assert bundle.population_role == FULL_VALIDATION_POPULATION_ROLE
    assert bundle.population_role != SCREENING_VALIDATION_POPULATION_ROLE
    assert "full_validation" in bundle.package_type


def test_build_pilot_full_validation_bundle_vs_screening_bundle_differ_only_in_validation_scope(
    tmp_path, pilot_policy_with_screening,
):
    package_root = tmp_path / "package"
    build_full_union_package(package_root)

    screening_bundle = build_pilot_bundle(
        pilot_policy=pilot_policy_with_screening, run_id="raw_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
    )
    full_bundle = build_pilot_full_validation_bundle(
        pilot_policy=pilot_policy_with_screening, run_id="raw_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
    )
    assert sorted(screening_bundle.train_basin_ids) == sorted(full_bundle.train_basin_ids)
    assert sorted(screening_bundle.test_basin_ids) == sorted(full_bundle.test_basin_ids)
    assert set(screening_bundle.validation_basin_ids) < set(full_bundle.validation_basin_ids)
    assert sorted(full_bundle.validation_basin_ids) == sorted(REAL_DEVELOPMENT)


# --- evaluate_full_validation_checkpoint: guard rails + scope tagging ------

def test_evaluate_full_validation_checkpoint_rejects_empty_basin_list(tmp_path):
    with pytest.raises(PilotFullValidationError):
        evaluate_full_validation_checkpoint(
            run_dir=tmp_path / "run",
            epoch=6,
            package_root=tmp_path / "package",
            target_variable="qobs_mm_per_h_lead06",
            lead_hours=6,
            development_basin_ids=[],
            promoted_from_run_id="raw_seedA",
        )


def test_evaluate_full_validation_checkpoint_perfect_nse_and_scope(tmp_path):
    basins = pick_development_basins(5)
    package_root = tmp_path / "package"
    build_full_union_package(package_root, ts_basin_ids=basins)
    run_dir = tmp_path / "run"
    write_perfect_validation_results(run_dir, 6, basins, package_root)

    result = evaluate_full_validation_checkpoint(
        run_dir=run_dir,
        epoch=6,
        package_root=package_root,
        target_variable="qobs_mm_per_h_lead06",
        lead_hours=6,
        development_basin_ids=basins,
        promoted_from_run_id="emb128x64_seedA",
    )
    assert result["scope"] == FULL_VALIDATION_METRIC_SCOPE
    assert result["authoritative"] is True
    assert result["promoted_from_run_id"] == "emb128x64_seedA"
    assert result["n_development_basins_requested"] == len(basins)
    assert result["primary_metric_median"] == pytest.approx(1.0, abs=1e-6)


def test_evaluate_full_validation_checkpoint_has_no_screening_style_cadence_restriction(tmp_path):
    # Deliberately not a screening-cadence epoch (7 is off-cadence for the
    # pilot's 3-epoch screening schedule) -- the full-validation readiness
    # interface has no cadence concept of its own, unlike
    # evaluate_screening_checkpoint.
    basins = pick_development_basins(5)
    package_root = tmp_path / "package"
    build_full_union_package(package_root, ts_basin_ids=basins)
    run_dir = tmp_path / "run"
    write_perfect_validation_results(run_dir, 7, basins, package_root)

    result = evaluate_full_validation_checkpoint(
        run_dir=run_dir,
        epoch=7,
        package_root=package_root,
        target_variable="qobs_mm_per_h_lead06",
        lead_hours=6,
        development_basin_ids=basins,
        promoted_from_run_id="raw_seedA",
    )
    assert result["epoch"] == 7
    assert result["primary_metric_median"] == pytest.approx(1.0, abs=1e-6)

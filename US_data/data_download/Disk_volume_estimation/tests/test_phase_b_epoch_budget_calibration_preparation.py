"""Focused, preparation-only contract tests for Phase-B epoch calibration."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path

from src.baseline.pilot_early_stopping import build_effective_policy, record_screening_event
from src.baseline.pilot_lead06_config import load_pilot_policy
from src.baseline.pilot_screening_eval import classify_screening_epoch_role
from src.baseline.pilot_orchestration import chunk_epoch_targets, screening_epochs_in_chunk
from tests._pilot_support import BASELINE_POLICY_PATH, PILOT_POLICY_PATH, REAL_DEVELOPMENT, SPLITS_DIR, build_full_union_package, write_screening_basin_ids_file

_SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_phase_b_epoch_budget_calibration.py"
_SPEC = importlib.util.spec_from_file_location("phase_b_epoch_budget_calibration", _SCRIPT)
campaign = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(campaign)


def _policy(tmp_path):
    base = load_pilot_policy(PILOT_POLICY_PATH)
    screening = write_screening_basin_ids_file(tmp_path / "screening.txt", REAL_DEVELOPMENT[:350])
    base = dataclasses.replace(base, screening_basin_ids_path=str(screening), screening_expected_count=350)
    from src.baseline.splits import sha256_of
    base = dataclasses.replace(base, screening_expected_sha256=sha256_of(screening))
    return campaign.build_epoch_budget_calibration_policy(base)


def test_frozen_five_candidate_cohort_and_exact_tuples():
    assert list(campaign.EPOCH_BUDGET_CALIBRATION_RUN_SPECS) == ["C1_anchor", "C2_low_lr", "C3_high_lr", "C4_late_h64", "C5_convergence_stress"]
    assert [(s.learning_rate, s.hidden_size, s.batch_size) for s in campaign.EPOCH_BUDGET_CALIBRATION_RUN_SPECS.values()] == [
        (3e-4, 128, 256), (1e-4, 128, 256), (1e-3, 128, 256), (3e-4, 64, 256), (3e-4, 256, 128)]
    # The literal run IDs/specs are the complete scientific identities; no
    # execution-order or Slurm-job input participates in their construction.
    assert {spec.run_id for spec in reversed(list(campaign.EPOCH_BUDGET_CALIBRATION_RUN_SPECS.values()))} == set(campaign.EPOCH_BUDGET_CALIBRATION_RUN_SPECS)
    assert not hasattr(campaign, "run_pilot") and not hasattr(campaign, "init_pilot_tracking_run")


def test_campaign_policy_requests_full_every_epoch_trajectory_without_performance_stop(tmp_path):
    policy = _policy(tmp_path)
    assert policy.pilot_max_epoch_budget == 14
    assert policy.initial_training_epochs == 14
    assert policy.performance_early_stopping_enabled is False
    assert chunk_epoch_targets(policy, 14) == [14]
    assert screening_epochs_in_chunk(0, 14, policy) == list(range(1, 15))
    assert [classify_screening_epoch_role(epoch, policy) for epoch in range(1, 15)] == ["stopping_eligible"] * 14
    effective = build_effective_policy(policy)
    assert effective["max_epoch_budget"] == 14
    assert effective["performance_early_stopping_enabled"] is False


def test_historical_default_policy_is_unchanged():
    historical = load_pilot_policy(PILOT_POLICY_PATH)
    assert (historical.pilot_max_epoch_budget, historical.initial_training_epochs, historical.screening_validation_every_n_epochs,
            historical.diagnostic_only_epoch, historical.stopping_eligible_from_epoch, historical.performance_early_stopping_enabled) == (36, 6, 3, 3, 6, True)
    assert chunk_epoch_targets(historical, 36) == [6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]
    assert [classify_screening_epoch_role(epoch, historical) for epoch in (1, 3, 6)] == ["not_a_screening_epoch", "diagnostic_only", "stopping_eligible"]


def test_no_performance_stop_records_every_epoch_through_target(tmp_path):
    policy = _policy(tmp_path)
    effective = build_effective_policy(policy)
    run_dir = tmp_path / "run"; run_dir.mkdir()
    for epoch in range(1, 14):
        state = record_screening_event(run_dir=run_dir, epoch=epoch, epoch_role="stopping_eligible",
                                       primary_metric_median=0.5, effective_policy=effective)
        assert state["stopped"] is False
    state = record_screening_event(run_dir=run_dir, epoch=14, epoch_role="stopping_eligible",
                                   primary_metric_median=0.5, effective_policy=effective)
    assert len(state["history"]) == 14
    assert state["stop_reason"] == "max_epoch_budget_reached"


def test_prepare_campaign_writes_auditable_configs_only(tmp_path):
    package = tmp_path / "package"; build_full_union_package(package)
    audit_path = campaign.prepare_campaign(pilot_policy_path=PILOT_POLICY_PATH, baseline_policy_path=BASELINE_POLICY_PATH,
                                           package_root=package, splits_dir=SPLITS_DIR, output_dir=tmp_path / "audit")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["candidate_count"] == 5
    assert audit["audit_scope"] == "LOCAL_STRUCTURAL_AUDIT_ONLY"
    assert audit["canonical_package_validation_required_before_training"] is True
    assert audit["training_cadence"] == "one_uninterrupted_segment_through_target_epoch"
    assert audit["checkpoint_retention"] == "save_weights_every_epoch"
    assert audit["evaluation_scope"] == "development_validation_2024_only"
    assert audit["sealed_scopes_not_accessed"] == ["temporal_test_2025", "non_ca_spatial_holdout", "california"]
    assert audit["screening_epochs"] == list(range(1, 15))
    assert audit["performance_early_stopping_enabled"] is False
    assert audit["no_wandb_or_hpo"] is True and audit["no_sealed_scope"] is True
    assert all(row["max_updates_per_epoch"] == 50_000 and row["training_segment_epochs"] == 14
               and row["checkpoint_save_every_epochs"] == 1 for row in audit["candidates"])
    assert all(row["evaluation_period"] == "validation_2024_only" and row["screening_validation_basin_count"] == 400 for row in audit["candidates"])
    assert all("slurm" not in row and "wandb" not in row and row["sealed_scope"] is False for row in audit["candidates"])

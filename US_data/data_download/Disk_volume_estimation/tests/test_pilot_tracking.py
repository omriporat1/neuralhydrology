"""Focused tests for src/baseline/pilot_tracking.py (task item 10).

Covers: build_pilot_run_identity/build_pilot_hyperparameters payload shape
using a real GeneratedConfigBundle built from build_pilot_bundle; W&B
disabled (real committed default policy) returns a null-backend TrackingRun
without ever importing wandb; an enabled=True policy with wandb not
installed is caught and downgraded to a null sink with a warning (never
raised); log_pilot_screening_event's scope assertion and metric shaping;
finish_pilot_run's terminal-status recording. No real W&B network/package
required (wandb is confirmed not installed in this environment, which is
itself what exercises the failure-downgrade path).
"""
from __future__ import annotations

import warnings

import pytest
import yaml

from src.baseline.pilot_lead06_config import build_pilot_bundle, load_pilot_policy, resolve_pilot_run_spec
from src.baseline.pilot_early_stopping import build_effective_policy
from src.baseline.pilot_screening_eval import SCREENING_METRIC_SCOPE, evaluate_screening_checkpoint
from src.baseline.pilot_tracking import (
    build_pilot_hyperparameters,
    build_pilot_run_identity,
    finish_pilot_run,
    init_pilot_tracking_run,
    log_pilot_checkpoint_reference,
    log_pilot_epoch_training_metrics,
    log_pilot_screening_event,
)
from src.baseline.wandb_tracking import TrackingError
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
def pilot_policy():
    return load_pilot_policy(PILOT_POLICY_PATH)


@pytest.fixture
def bundle_and_effective_policy(tmp_path, pilot_policy):
    import dataclasses

    package_root = tmp_path / "package"
    build_full_union_package(package_root)
    screening_path = write_screening_basin_ids_file(tmp_path / "screening.txt", REAL_DEVELOPMENT[:350])
    policy = dataclasses.replace(
        pilot_policy,
        screening_basin_ids_path=str(screening_path),
        screening_expected_count=350,
        screening_expected_sha256=sha256_of(screening_path),
    )
    bundle = build_pilot_bundle(
        pilot_policy=policy, run_id="raw_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
    )
    effective = build_effective_policy(policy)
    return policy, bundle, effective, package_root


# --- build_pilot_run_identity / build_pilot_hyperparameters ------------------

def test_build_pilot_run_identity_shape(bundle_and_effective_policy):
    policy, bundle, effective, _ = bundle_and_effective_policy
    run_spec = resolve_pilot_run_spec(policy, "raw_seedA")
    identity = build_pilot_run_identity(
        pilot_policy=policy, run_spec=run_spec, bundle=bundle, effective_early_stopping_policy=effective,
    )
    assert identity["run_id"] == "raw_seedA"
    assert identity["seed"] == 967139
    assert identity["static_pathway"] == run_spec.static_pathway
    assert identity["n_train_basins"] == len(bundle.train_basin_ids)
    assert identity["n_validation_basins"] == len(bundle.validation_basin_ids)
    assert identity["is_workflow_qualification_run"] is False
    assert identity["effective_max_epoch_budget"] == effective["max_epoch_budget"]
    # no credential-shaped or sealed-population keys anywhere in the identity
    for k in identity:
        kl = k.lower()
        assert "api_key" not in kl and "secret" not in kl and "password" not in kl


def test_build_pilot_run_identity_flags_workflow_qualification_run(bundle_and_effective_policy):
    policy, bundle, effective, _ = bundle_and_effective_policy
    run_spec = resolve_pilot_run_spec(policy, "emb128x64_seedA")
    identity = build_pilot_run_identity(
        pilot_policy=policy, run_spec=run_spec, bundle=bundle, effective_early_stopping_policy=effective,
    )
    assert identity["is_workflow_qualification_run"] is True


def test_build_pilot_hyperparameters_subset(bundle_and_effective_policy):
    _, bundle, _, _ = bundle_and_effective_policy
    hyperparams = build_pilot_hyperparameters(bundle)
    assert hyperparams["model"] == "cudalstm"
    assert hyperparams["hidden_size"] == 128
    assert hyperparams["output_dropout"] == 0.25
    assert hyperparams["batch_size"] == 256
    assert hyperparams["seed"] == 967139
    # never carries basin lists or filesystem paths
    assert "train_basin_ids" not in hyperparams
    assert "validation_basin_ids" not in hyperparams


# --- init_pilot_tracking_run: disabled-by-default + failure downgrade ------

def test_init_pilot_tracking_run_disabled_by_default_returns_null_backend(pilot_policy):
    run = init_pilot_tracking_run(pilot_policy, run_identity={"run_id": "raw_seedA"})
    assert run.backend == "null"
    assert run.finished is False


def test_init_pilot_tracking_run_downgrades_on_enabled_but_wandb_missing(tmp_path, pilot_policy):
    import dataclasses

    enabled_policy_raw = {
        "policy_name": "test_enabled_wandb_policy",
        "enabled": True,
        "mode": "online",
        "project": "flashnh-stage1-test",
        "entity": None,
        "tags": ["test"],
        "max_artifact_reference_bytes": 1048576,
    }
    enabled_policy_path = tmp_path / "enabled_wandb_policy.yaml"
    enabled_policy_path.write_text(yaml.safe_dump(enabled_policy_raw), encoding="utf-8")
    policy = dataclasses.replace(pilot_policy, wandb_policy_path=str(enabled_policy_path))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = init_pilot_tracking_run(policy, run_identity={"run_id": "raw_seedA"})
    assert run.backend == "null"
    assert any("W&B tracking init failed" in str(w.message) for w in caught)


# --- log_pilot_screening_event: scope assertion + metric shaping -----------

def test_log_pilot_screening_event_rejects_wrong_scope(pilot_policy):
    run = init_pilot_tracking_run(pilot_policy, run_identity={"run_id": "raw_seedA"})
    with pytest.raises(TrackingError):
        log_pilot_screening_event(
            run, epoch=6, screening_result={"scope": "some_other_scope"},
        )


def test_log_pilot_screening_event_logs_real_screening_result(bundle_and_effective_policy, pilot_policy):
    policy, bundle, effective, package_root = bundle_and_effective_policy
    basins = pick_development_basins(5)
    # rebuild package with actual NetCDF for these basins + a perfect checkpoint
    build_full_union_package(package_root, ts_basin_ids=basins)
    run_dir = package_root.parent / "run"
    write_perfect_validation_results(run_dir, 6, basins, package_root)

    screening_result = evaluate_screening_checkpoint(
        run_dir=run_dir, epoch=6, package_root=package_root,
        target_variable=bundle.target_variable, lead_hours=policy.lead_hours,
        screening_basin_ids=basins, pilot_policy=policy,
    )
    assert screening_result["scope"] == SCREENING_METRIC_SCOPE

    run = init_pilot_tracking_run(policy, run_identity={"run_id": "raw_seedA"})
    log_pilot_screening_event(run, epoch=6, screening_result=screening_result)
    assert len(run.scientific_metrics) == 1
    logged_epoch, logged_metrics = run.scientific_metrics[0]
    assert logged_epoch == 6
    assert logged_metrics["screening/primary_metric_median"] == pytest.approx(1.0, abs=1e-6)
    assert logged_metrics["screening/epoch_role"] == "stopping_eligible"
    # no temporal-test/spatial-holdout-shaped key ever logged
    for k in logged_metrics:
        kl = k.lower()
        assert "holdout" not in kl and "temporal" not in kl and "test" not in kl.replace("_lead", "")


# --- log_pilot_epoch_training_metrics / log_pilot_checkpoint_reference -----

def test_log_pilot_epoch_training_metrics_all_none_is_noop(pilot_policy):
    run = init_pilot_tracking_run(pilot_policy, run_identity={"run_id": "raw_seedA"})
    log_pilot_epoch_training_metrics(run, epoch=1)
    assert run.resource_metrics == []


def test_log_pilot_epoch_training_metrics_logs_provided_fields(pilot_policy):
    run = init_pilot_tracking_run(pilot_policy, run_identity={"run_id": "raw_seedA"})
    log_pilot_epoch_training_metrics(run, epoch=1, training_loss=0.5, wall_time_s=12.3)
    assert len(run.resource_metrics) == 1
    epoch, metrics = run.resource_metrics[0]
    assert epoch == 1
    assert metrics == {"training_loss": 0.5, "wall_time_s": 12.3}


def test_log_pilot_checkpoint_reference_records_path_and_checksum(tmp_path, pilot_policy):
    run = init_pilot_tracking_run(pilot_policy, run_identity={"run_id": "raw_seedA"})
    ckpt = tmp_path / "model_epoch006.pt"
    ckpt.write_bytes(b"fake checkpoint bytes")
    log_pilot_checkpoint_reference(run, epoch=6, path=ckpt, checksum="deadbeef")
    assert len(run.artifact_references) == 1
    assert run.artifact_references[0]["name"] == "checkpoint_epoch_006"
    assert run.artifact_references[0]["checksum"] == "deadbeef"


# --- finish_pilot_run ---------------------------------------------------------

def test_finish_pilot_run_records_terminal_status(pilot_policy):
    run = init_pilot_tracking_run(pilot_policy, run_identity={"run_id": "raw_seedA"})
    finish_pilot_run(run, final_status="stopped_patience_exhausted", best_epoch=6)
    assert run.finished is True
    assert run.run_identity["final_status"] == "stopped_patience_exhausted"
    assert run.run_identity["best_checkpoint_epoch"] == 6

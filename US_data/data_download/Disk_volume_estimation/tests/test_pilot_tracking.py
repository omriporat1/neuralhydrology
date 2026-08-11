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

import sys
import types
import warnings

import pytest
import yaml

from src.baseline.pilot_lead06_config import build_pilot_bundle, load_pilot_policy, resolve_pilot_run_spec
from src.baseline.pilot_early_stopping import build_effective_policy
from src.baseline.pilot_screening_eval import SCREENING_METRIC_SCOPE, evaluate_screening_checkpoint
from src.baseline.pilot_tracking import (
    WANDB_RUN_ID_STATE_FILENAME,
    build_pilot_hyperparameters,
    build_pilot_run_identity,
    derive_pilot_wandb_run_id,
    finish_pilot_run,
    init_pilot_tracking_run,
    log_pilot_checkpoint_reference,
    log_pilot_epoch_training_metrics,
    log_pilot_screening_event,
    resolve_pilot_wandb_run_id,
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


def test_build_pilot_run_identity_carries_no_override_learning_rate_and_hidden_size(bundle_and_effective_policy):
    # For every pre-existing pilot run (no learning_rate/hidden_size override
    # declared on its PilotRunSpec), the *_override fields must be None while
    # the resolved_* fields must still carry the run's actual profile-derived
    # values -- these are never omitted/None for a real generated config.
    policy, bundle, effective, _ = bundle_and_effective_policy
    run_spec = resolve_pilot_run_spec(policy, "raw_seedA")
    identity = build_pilot_run_identity(
        pilot_policy=policy, run_spec=run_spec, bundle=bundle, effective_early_stopping_policy=effective,
    )
    assert identity["learning_rate_override"] is None
    assert identity["resolved_learning_rate"] is not None
    assert identity["hidden_size_override"] is None
    assert identity["resolved_hidden_size"] == 128


def test_build_pilot_run_identity_carries_explicit_hidden_size_override(tmp_path, pilot_policy):
    # Hidden-size-A range-characterization campaign: a run_id whose
    # PilotRunSpec explicitly overrides hidden_size must surface that value
    # in both build_pilot_run_identity's hidden_size_override (the raw
    # override) and resolved_hidden_size (the value actually written into
    # this run's config.yaml) -- see nh_config_generation.
    # validate_hidden_size_override and pilot_orchestration.
    # enforce_pilot_hidden_size_identity, which key off exactly these fields.
    import dataclasses

    package_root = tmp_path / "package"
    build_full_union_package(package_root)
    screening_path = write_screening_basin_ids_file(tmp_path / "screening.txt", REAL_DEVELOPMENT[:350])
    overridden_run_spec = dataclasses.replace(pilot_policy.runs["raw_seedA"], hidden_size=256)
    overridden_runs = dict(pilot_policy.runs)
    overridden_runs["raw_seedA"] = overridden_run_spec
    policy = dataclasses.replace(
        pilot_policy,
        runs=overridden_runs,
        screening_basin_ids_path=str(screening_path),
        screening_expected_count=350,
        screening_expected_sha256=sha256_of(screening_path),
    )
    bundle = build_pilot_bundle(
        pilot_policy=policy, run_id="raw_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
    )
    effective = build_effective_policy(policy)
    run_spec = resolve_pilot_run_spec(policy, "raw_seedA")
    identity = build_pilot_run_identity(
        pilot_policy=policy, run_spec=run_spec, bundle=bundle, effective_early_stopping_policy=effective,
    )
    assert identity["hidden_size_override"] == 256
    assert identity["resolved_hidden_size"] == 256


def test_build_pilot_run_identity_carries_no_override_embedding_dropout_on_raw_pathway(
    bundle_and_effective_policy,
):
    # raw_seedA's profile has no statics_embedding section at all -- both the
    # override and resolved fields must be None (never a stray 0.0 or a
    # KeyError), distinguishing "no embedding pathway" from "embedding
    # pathway with dropout=0.0" (the drop00 candidate).
    policy, bundle, effective, _ = bundle_and_effective_policy
    run_spec = resolve_pilot_run_spec(policy, "raw_seedA")
    identity = build_pilot_run_identity(
        pilot_policy=policy, run_spec=run_spec, bundle=bundle, effective_early_stopping_policy=effective,
    )
    assert identity["embedding_dropout_override"] is None
    assert identity["resolved_embedding_dropout"] is None


def test_build_pilot_run_identity_carries_no_override_embedding_dropout_on_embedding_pathway(tmp_path, pilot_policy):
    # emb128x64_seedA's profile does have a statics_embedding section -- with
    # no override declared, embedding_dropout_override must be None while
    # resolved_embedding_dropout must still carry the profile's own dropout
    # (0.1, the frozen untuned value -- see config/stage1_lead06_pilot_v001.yaml).
    package_root = tmp_path / "package"
    build_full_union_package(package_root)
    screening_path = write_screening_basin_ids_file(tmp_path / "screening.txt", REAL_DEVELOPMENT[:350])
    import dataclasses

    policy = dataclasses.replace(
        pilot_policy,
        screening_basin_ids_path=str(screening_path),
        screening_expected_count=350,
        screening_expected_sha256=sha256_of(screening_path),
    )
    bundle = build_pilot_bundle(
        pilot_policy=policy, run_id="emb128x64_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
    )
    effective = build_effective_policy(policy)
    run_spec = resolve_pilot_run_spec(policy, "emb128x64_seedA")
    identity = build_pilot_run_identity(
        pilot_policy=policy, run_spec=run_spec, bundle=bundle, effective_early_stopping_policy=effective,
    )
    assert identity["embedding_dropout_override"] is None
    assert identity["resolved_embedding_dropout"] == pytest.approx(0.1)


@pytest.mark.parametrize("override_value", [0.0, 0.05, 0.10, 0.20, 0.40])
def test_build_pilot_run_identity_carries_explicit_embedding_dropout_override(
    tmp_path, pilot_policy, override_value
):
    # Embedding-Dropout-A range-characterization campaign: a run_id whose
    # PilotRunSpec explicitly overrides embedding_dropout must surface that
    # value in both build_pilot_run_identity's embedding_dropout_override
    # (the raw override) and resolved_embedding_dropout (the value actually
    # written into this run's config.yaml) -- see nh_config_generation.
    # validate_embedding_dropout_override and pilot_orchestration.
    # enforce_pilot_embedding_dropout_identity, which key off exactly these
    # fields. Parametrized over all five Embedding-Dropout-A candidate
    # values, including the drop00 candidate's explicit 0.0 -- proving it is
    # never lost/confused with the "no override" None case exercised above.
    import dataclasses

    package_root = tmp_path / "package"
    build_full_union_package(package_root)
    screening_path = write_screening_basin_ids_file(tmp_path / "screening.txt", REAL_DEVELOPMENT[:350])
    overridden_run_spec = dataclasses.replace(
        pilot_policy.runs["emb128x64_seedA"], embedding_dropout=override_value
    )
    overridden_runs = dict(pilot_policy.runs)
    overridden_runs["emb128x64_seedA"] = overridden_run_spec
    policy = dataclasses.replace(
        pilot_policy,
        runs=overridden_runs,
        screening_basin_ids_path=str(screening_path),
        screening_expected_count=350,
        screening_expected_sha256=sha256_of(screening_path),
    )
    bundle = build_pilot_bundle(
        pilot_policy=policy, run_id="emb128x64_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
    )
    effective = build_effective_policy(policy)
    run_spec = resolve_pilot_run_spec(policy, "emb128x64_seedA")
    identity = build_pilot_run_identity(
        pilot_policy=policy, run_spec=run_spec, bundle=bundle, effective_early_stopping_policy=effective,
    )
    assert identity["embedding_dropout_override"] == pytest.approx(override_value)
    assert identity["resolved_embedding_dropout"] == pytest.approx(override_value)
    assert identity["embedding_dropout_override"] is not None
    assert identity["resolved_embedding_dropout"] is not None


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


# --- init_pilot_tracking_run: require_tracking strict launch contract ------
# (Hidden-size-A campaign's W&B launch contract -- see
# docs/decision_log.md's 2026-08-09 Hidden-size-A design-freeze entry and
# init_pilot_tracking_run's own docstring. Default require_tracking=False
# preserves every above test's null-fallback behavior unchanged; these tests
# cover the opt-in require_tracking=True hard-fail contract.)

def test_init_pilot_tracking_run_require_tracking_true_raises_when_policy_disabled(pilot_policy):
    # the real committed default pilot policy is disabled -- require_tracking=True
    # must raise immediately, before attempting any init call at all.
    with pytest.raises(TrackingError):
        init_pilot_tracking_run(
            pilot_policy, run_identity={"run_id": "raw_seedA"}, require_tracking=True
        )


def test_init_pilot_tracking_run_require_tracking_true_raises_on_init_failure(tmp_path, pilot_policy):
    # same enabled-but-wandb-missing policy as
    # test_init_pilot_tracking_run_downgrades_on_enabled_but_wandb_missing --
    # with require_tracking=True, this must now raise (wrapping the original
    # failure) instead of downgrading to a null sink.
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

    with pytest.raises(TrackingError):
        init_pilot_tracking_run(policy, run_identity={"run_id": "raw_seedA"}, require_tracking=True)


def test_init_pilot_tracking_run_require_tracking_true_raises_when_resolved_backend_null(
    tmp_path, pilot_policy, monkeypatch
):
    # Defense-in-depth: even if init_tracking_run itself succeeds but somehow
    # still resolves to a null/untracked run, require_tracking=True must
    # raise rather than silently returning it.
    import dataclasses

    import src.baseline.pilot_tracking as pilot_tracking_module

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

    def _fake_init_tracking_run(resolved_policy, run_identity, *, run_id, resume):
        from src.baseline.wandb_tracking import TrackingRun

        return TrackingRun(
            backend="null",
            max_artifact_reference_bytes=resolved_policy["max_artifact_reference_bytes"],
            run_identity=run_identity,
        )

    monkeypatch.setattr(pilot_tracking_module, "init_tracking_run", _fake_init_tracking_run)

    with pytest.raises(TrackingError):
        init_pilot_tracking_run(policy, run_identity={"run_id": "raw_seedA"}, require_tracking=True)


def test_init_pilot_tracking_run_require_tracking_true_succeeds_with_real_run(
    tmp_path, pilot_policy, monkeypatch
):
    # Positive path: a real (non-null, wandb_run_id-bearing) resolved run
    # must be returned normally, without raising, even with require_tracking=True.
    import dataclasses

    import src.baseline.pilot_tracking as pilot_tracking_module

    enabled_policy_raw = {
        "policy_name": "test_enabled_wandb_policy",
        "enabled": True,
        "mode": "offline",
        "project": "flashnh-stage1-test",
        "entity": None,
        "tags": ["test"],
        "max_artifact_reference_bytes": 1048576,
    }
    enabled_policy_path = tmp_path / "enabled_wandb_policy.yaml"
    enabled_policy_path.write_text(yaml.safe_dump(enabled_policy_raw), encoding="utf-8")
    policy = dataclasses.replace(pilot_policy, wandb_policy_path=str(enabled_policy_path))

    def _fake_init_tracking_run(resolved_policy, run_identity, *, run_id, resume):
        from src.baseline.wandb_tracking import TrackingRun

        return TrackingRun(
            backend="wandb",
            max_artifact_reference_bytes=resolved_policy["max_artifact_reference_bytes"],
            run_identity=run_identity,
            wandb_run_id=run_id or "fake_wandb_run_id",
        )

    monkeypatch.setattr(pilot_tracking_module, "init_tracking_run", _fake_init_tracking_run)

    run = init_pilot_tracking_run(policy, run_identity={"run_id": "raw_seedA"}, require_tracking=True)
    assert run.backend == "wandb"
    assert run.wandb_run_id is not None


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
    assert run.artifact_references[0]["checkpoint_type"] == "nh_model_checkpoint"


def test_log_pilot_checkpoint_reference_survives_real_oversized_checkpoint(
    fake_wandb, enabled_offline_policy, pilot_policy, tmp_path
):
    """Reproduces the exact job 45731908 failure shape: a real-sized (>1MB)
    NH checkpoint logged through log_pilot_checkpoint_reference under the
    real committed 1,048,576-byte max_artifact_reference_bytes policy value
    must not raise -- this used to route through log_artifact_reference,
    whose size ceiling raised TrackingError uncaught and killed the pilot
    mid-screening."""
    policy = enabled_offline_policy(pilot_policy)
    run = init_pilot_tracking_run(
        policy, run_identity={"pilot_policy_name": "stage1_lead06_pilot_v001", "run_id": "raw_seedA"}
    )
    assert run.max_artifact_reference_bytes == 1_048_576

    ckpt = tmp_path / "model_epoch003.pt"
    ckpt.write_bytes(b"0" * 1_300_000)  # ~1.25 MB, matching the real checkpoint size

    log_pilot_checkpoint_reference(run, epoch=3, path=ckpt, checksum="deadbeef")

    assert run.degraded is False
    assert run.artifact_references[0]["size_bytes"] == 1_300_000
    assert run.artifact_references[0]["epoch"] == 3


# --- finish_pilot_run ---------------------------------------------------------

def test_finish_pilot_run_records_terminal_status(pilot_policy):
    run = init_pilot_tracking_run(pilot_policy, run_identity={"run_id": "raw_seedA"})
    finish_pilot_run(run, final_status="stopped_patience_exhausted", best_epoch=6)
    assert run.finished is True
    assert run.run_identity["final_status"] == "stopped_patience_exhausted"
    assert run.run_identity["best_checkpoint_epoch"] == 6


# --- extended metadata contract: nullable multi-fidelity + baseline/split ids

def test_build_pilot_run_identity_includes_extended_metadata_contract_fields(bundle_and_effective_policy):
    policy, bundle, effective, _ = bundle_and_effective_policy
    run_spec = resolve_pilot_run_spec(policy, "raw_seedA")
    identity = build_pilot_run_identity(
        pilot_policy=policy, run_spec=run_spec, bundle=bundle, effective_early_stopping_policy=effective,
    )
    # Multi-fidelity is a not-yet-implemented direction (Part L.5): this
    # pilot only ever runs at full fidelity, so the field must be present
    # and explicitly null, never omitted and never a fabricated value.
    assert identity["max_updates_per_epoch"] is None
    assert identity["baseline_policy_sha256"] == bundle.policy_sha256
    assert identity["splits_dir"] == bundle.splits_dir
    assert identity["tracking_generation"] == "g1"
    assert identity["wandb_run_id"] == derive_pilot_wandb_run_id(
        policy.raw.get("policy_name"), run_spec.run_id, "g1"
    )

    bumped_identity = build_pilot_run_identity(
        pilot_policy=policy, run_spec=run_spec, bundle=bundle, effective_early_stopping_policy=effective,
        tracking_generation="g2",
    )
    assert bumped_identity["tracking_generation"] == "g2"
    assert bumped_identity["wandb_run_id"] != identity["wandb_run_id"]


def test_build_pilot_run_identity_wandb_policy_sha256_reflects_override(bundle_and_effective_policy, tmp_path):
    # run_identity carries only the checksum of whichever W&B policy file
    # actually took effect (committed default, or an explicit per-run
    # --wandb-policy-path override) -- never the raw path itself, mirroring
    # pilot_policy_sha256/baseline_policy_sha256 above. The raw override path
    # is machine-local and already captured verbatim in commands_used/the
    # evidence bundle instead (see scripts/run_stage1_lead06_pilot.py).
    import dataclasses

    policy, bundle, effective, _ = bundle_and_effective_policy
    run_spec = resolve_pilot_run_spec(policy, "raw_seedA")
    default_identity = build_pilot_run_identity(
        pilot_policy=policy, run_spec=run_spec, bundle=bundle, effective_early_stopping_policy=effective,
    )
    assert default_identity["wandb_policy_sha256"] == sha256_of(policy.wandb_policy_path)

    override_raw = {
        "policy_name": "test_override_wandb_policy",
        "enabled": True,
        "mode": "offline",
        "project": "flashnh-stage1-test",
        "entity": None,
        "tags": ["test"],
        "max_artifact_reference_bytes": 1048576,
    }
    override_path = tmp_path / "override_wandb_policy.yaml"
    override_path.write_text(yaml.safe_dump(override_raw), encoding="utf-8")
    overridden_policy = dataclasses.replace(policy, wandb_policy_path=str(override_path))
    overridden_identity = build_pilot_run_identity(
        pilot_policy=overridden_policy, run_spec=run_spec, bundle=bundle,
        effective_early_stopping_policy=effective,
    )
    assert overridden_identity["wandb_policy_sha256"] == sha256_of(override_path)
    assert overridden_identity["wandb_policy_sha256"] != default_identity["wandb_policy_sha256"]
    # the raw path never leaks into run_identity
    for v in overridden_identity.values():
        assert str(override_path) != v


# --- stable W&B run identity across restarts --------------------------------

def test_derive_pilot_wandb_run_id_is_deterministic():
    id_a = derive_pilot_wandb_run_id("stage1_lead06_pilot_v001", "raw_seedA")
    id_b = derive_pilot_wandb_run_id("stage1_lead06_pilot_v001", "raw_seedA")
    assert id_a == id_b
    assert id_a != derive_pilot_wandb_run_id("stage1_lead06_pilot_v001", "raw_seedB")


def test_resolve_pilot_wandb_run_id_without_nh_run_dir_is_pure_deterministic():
    resolved = resolve_pilot_wandb_run_id(pilot_policy_name="pol", run_id="raw_seedA", nh_run_dir=None)
    assert resolved == derive_pilot_wandb_run_id("pol", "raw_seedA")


def test_resolve_pilot_wandb_run_id_persists_and_reuses_across_calls(tmp_path):
    run_dir = tmp_path / "nh_run"
    run_dir.mkdir()
    first = resolve_pilot_wandb_run_id(pilot_policy_name="pol", run_id="raw_seedA", nh_run_dir=run_dir)
    assert (run_dir / WANDB_RUN_ID_STATE_FILENAME).is_file()
    second = resolve_pilot_wandb_run_id(pilot_policy_name="pol", run_id="raw_seedA", nh_run_dir=run_dir)
    assert first == second


def test_resolve_pilot_wandb_run_id_missing_nh_run_dir_does_not_write_or_fail(tmp_path):
    run_dir = tmp_path / "does_not_exist_yet"
    resolved = resolve_pilot_wandb_run_id(pilot_policy_name="pol", run_id="raw_seedA", nh_run_dir=run_dir)
    assert resolved == derive_pilot_wandb_run_id("pol", "raw_seedA")
    assert not run_dir.exists()


def test_resolve_pilot_wandb_run_id_contradiction_raises(tmp_path):
    run_dir = tmp_path / "nh_run"
    run_dir.mkdir()
    resolve_pilot_wandb_run_id(pilot_policy_name="pol", run_id="raw_seedA", nh_run_dir=run_dir)
    with pytest.raises(TrackingError):
        # Same run_dir, but a different candidate -- a real bug (stale run
        # directory reused), must fail loudly rather than silently mixing
        # two candidates' histories into one W&B run.
        resolve_pilot_wandb_run_id(pilot_policy_name="pol", run_id="raw_seedB", nh_run_dir=run_dir)


# --- tracking_generation: disambiguates a deliberate restart-from-scratch
# under the same run_id from an ordinary bounded-Slurm continuation, since
# NH run directories are timestamped/prefix-matched rather than fixed paths
# (see module docstring and derive_pilot_wandb_run_id) ------------------------

def test_derive_pilot_wandb_run_id_default_generation_matches_omitted_argument():
    # Ordinary continuations never pass tracking_generation explicitly --
    # the default must be indistinguishable from the pre-existing 2-arg
    # call shape so this fix is invisible to every caller that doesn't
    # deliberately opt in to a new generation.
    assert derive_pilot_wandb_run_id("pol", "raw_seedA") == derive_pilot_wandb_run_id(
        "pol", "raw_seedA", "g1"
    )


def test_derive_pilot_wandb_run_id_different_generation_yields_different_id():
    default_gen = derive_pilot_wandb_run_id("pol", "raw_seedA")
    bumped_gen = derive_pilot_wandb_run_id("pol", "raw_seedA", "g2")
    assert default_gen != bumped_gen
    # Deterministic and reproducible for the bumped generation too -- this
    # is what actually closes the collision (a fresh nh_run_dir=None call,
    # e.g. after the operator deletes an abandoned NH run directory and
    # deliberately restarts raw_seedA from scratch, gets a genuinely new
    # W&B run id rather than silently resuming the abandoned attempt).
    assert bumped_gen == derive_pilot_wandb_run_id("pol", "raw_seedA", "g2")


def test_resolve_pilot_wandb_run_id_different_generation_same_dir_contradicts(tmp_path):
    run_dir = tmp_path / "nh_run"
    run_dir.mkdir()
    resolve_pilot_wandb_run_id(
        pilot_policy_name="pol", run_id="raw_seedA", tracking_generation="g1", nh_run_dir=run_dir
    )
    with pytest.raises(TrackingError):
        # Same run_dir, same candidate/run_id, but a different generation --
        # e.g. an operator who deleted and restarted this run directory
        # without bumping tracking_generation. This must fail loudly rather
        # than silently reusing the prior attempt's W&B history.
        resolve_pilot_wandb_run_id(
            pilot_policy_name="pol", run_id="raw_seedA", tracking_generation="g2", nh_run_dir=run_dir
        )


# --- init_pilot_tracking_run with a real (fake) wandb backend --------------

class _FakeWandbConfig(dict):
    def update(self, other=None, allow_val_change=None, **kwargs):
        if other:
            dict.update(self, other)


class _FakeWandbRun:
    def __init__(self):
        self.config = _FakeWandbConfig()
        self.summary = {}
        self.logged = []
        self.finished = False

    def log(self, data, step=None):
        self.logged.append((step, dict(data)))

    def finish(self):
        self.finished = True


class _FakeWandbModule(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.init_calls = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return _FakeWandbRun()


@pytest.fixture
def fake_wandb(monkeypatch):
    fake = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


@pytest.fixture
def enabled_offline_policy(tmp_path):
    import dataclasses

    def _make(pilot_policy):
        raw = {
            "policy_name": "test_enabled_offline_policy",
            "enabled": True,
            "mode": "offline",
            "project": "flashnh-stage1-test",
            "entity": None,
            "tags": ["test"],
            "max_artifact_reference_bytes": 1048576,
        }
        p = tmp_path / "enabled_offline_policy.yaml"
        p.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return dataclasses.replace(pilot_policy, wandb_policy_path=str(p))

    return _make


def test_init_pilot_tracking_run_offline_passes_deterministic_run_id(
    fake_wandb, enabled_offline_policy, pilot_policy
):
    policy = enabled_offline_policy(pilot_policy)
    run = init_pilot_tracking_run(
        policy, run_identity={"pilot_policy_name": "stage1_lead06_pilot_v001", "run_id": "raw_seedA"}
    )
    assert run.backend == "wandb"
    expected_id = derive_pilot_wandb_run_id("stage1_lead06_pilot_v001", "raw_seedA")
    assert fake_wandb.init_calls[0]["id"] == expected_id
    assert fake_wandb.init_calls[0]["resume"] == "allow"
    assert run.wandb_run_id == expected_id


def test_init_pilot_tracking_run_reuses_same_run_id_across_continuations(
    fake_wandb, enabled_offline_policy, pilot_policy, tmp_path
):
    policy = enabled_offline_policy(pilot_policy)
    run_identity = {"pilot_policy_name": "stage1_lead06_pilot_v001", "run_id": "raw_seedA"}
    nh_run_dir = tmp_path / "nh_run"
    nh_run_dir.mkdir()

    run_1 = init_pilot_tracking_run(policy, run_identity=run_identity, nh_run_dir=nh_run_dir)
    run_2 = init_pilot_tracking_run(policy, run_identity=run_identity, nh_run_dir=nh_run_dir)

    assert run_1.wandb_run_id == run_2.wandb_run_id
    assert fake_wandb.init_calls[0]["id"] == fake_wandb.init_calls[1]["id"]
    assert (nh_run_dir / WANDB_RUN_ID_STATE_FILENAME).is_file()


def test_init_pilot_tracking_run_threads_tracking_generation_from_run_identity(
    fake_wandb, enabled_offline_policy, pilot_policy
):
    policy = enabled_offline_policy(pilot_policy)
    run = init_pilot_tracking_run(
        policy,
        run_identity={
            "pilot_policy_name": "stage1_lead06_pilot_v001",
            "run_id": "raw_seedA",
            "tracking_generation": "g2",
        },
    )
    expected_id = derive_pilot_wandb_run_id("stage1_lead06_pilot_v001", "raw_seedA", "g2")
    default_gen_id = derive_pilot_wandb_run_id("stage1_lead06_pilot_v001", "raw_seedA")
    assert expected_id != default_gen_id
    assert fake_wandb.init_calls[0]["id"] == expected_id
    assert run.wandb_run_id == expected_id


def test_init_pilot_tracking_run_disabled_policy_never_writes_identity_file(tmp_path, pilot_policy):
    nh_run_dir = tmp_path / "nh_run"
    nh_run_dir.mkdir()
    init_pilot_tracking_run(
        pilot_policy,
        run_identity={"pilot_policy_name": "stage1_lead06_pilot_v001", "run_id": "raw_seedA"},
        nh_run_dir=nh_run_dir,
    )
    assert not (nh_run_dir / WANDB_RUN_ID_STATE_FILENAME).exists()


# ---------------------------------------------------------------------------
# max_updates_per_epoch: run-identity + hyperparameter threading for a capped
# (early-fidelity-screening) bundle, distinguished from the uncapped default
# already covered by test_build_pilot_run_identity_includes_extended_metadata_
# contract_fields above.
# ---------------------------------------------------------------------------

def _build_capped_bundle_and_effective_policy(tmp_path, pilot_policy, *, cap):
    import dataclasses

    package_root = tmp_path / "package"
    build_full_union_package(package_root)
    screening_path = write_screening_basin_ids_file(tmp_path / "screening.txt", REAL_DEVELOPMENT[:350])

    with open(PILOT_POLICY_PATH, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    entry = next(r for r in raw["runs"] if r["run_id"] == "raw_seedA")
    entry["max_updates_per_epoch"] = cap
    policy_path = tmp_path / "capped_policy.yaml"
    policy_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    policy = load_pilot_policy(policy_path)
    policy = dataclasses.replace(
        policy,
        screening_basin_ids_path=str(screening_path),
        screening_expected_count=350,
        screening_expected_sha256=sha256_of(screening_path),
    )
    bundle = build_pilot_bundle(
        pilot_policy=policy, run_id="raw_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
    )
    effective = build_effective_policy(policy)
    return policy, bundle, effective


def test_build_pilot_run_identity_max_updates_per_epoch_reflects_capped_bundle(tmp_path, pilot_policy):
    policy, bundle, effective = _build_capped_bundle_and_effective_policy(tmp_path, pilot_policy, cap=10)
    assert bundle.max_updates_per_epoch == 10
    run_spec = resolve_pilot_run_spec(policy, "raw_seedA")
    identity = build_pilot_run_identity(
        pilot_policy=policy, run_spec=run_spec, bundle=bundle, effective_early_stopping_policy=effective,
    )
    assert identity["max_updates_per_epoch"] == 10


def test_build_pilot_hyperparameters_omits_cap_key_when_bundle_uncapped(bundle_and_effective_policy):
    _, bundle, _, _ = bundle_and_effective_policy
    assert bundle.max_updates_per_epoch is None
    hyperparams = build_pilot_hyperparameters(bundle)
    assert "max_updates_per_epoch" not in hyperparams


def test_build_pilot_hyperparameters_includes_cap_key_when_bundle_capped(tmp_path, pilot_policy):
    _, bundle, _ = _build_capped_bundle_and_effective_policy(tmp_path, pilot_policy, cap=15)
    hyperparams = build_pilot_hyperparameters(bundle)
    assert hyperparams["max_updates_per_epoch"] == 15

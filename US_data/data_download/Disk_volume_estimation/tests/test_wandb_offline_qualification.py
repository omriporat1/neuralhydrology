"""W&B OFFLINE QUALIFICATION harness (Flash-NH task: "W&B offline
qualification and the Flash-NH-specific user guide").

A single, traceable, numbered harness exercising the 15 scenarios the task
specifies before ordinary W&B tracking may be used on a real structural
candidate (``raw_seedA``). Every scenario below uses W&B's OFFLINE mode
only, via an in-process fake ``wandb`` module (monkeypatched into
``sys.modules``, never the real package) -- no API key, no network call, no
change to any user-level global W&B config. This mirrors the same
fake-module pattern already used in ``tests/test_wandb_tracking.py`` and
``tests/test_pilot_tracking.py``; this file exists to give one place where
all 15 required scenarios are visibly enumerated together, not to duplicate
those files' deeper integration coverage (cross-referenced below where
relevant).

Scenario -> underlying mechanism:
  1.  Policy parsing with offline mode           -> wandb_tracking.load_tracking_policy
  2.  Real W&B init                              -> wandb_tracking.init_tracking_run
  3.  Full config logging                        -> wandb_tracking.log_hyperparameters
  4.  Epoch/training metric events                -> log_scientific_metrics + log_resource_metrics
  5.  Screening metric + early-stopping-state log -> pilot_tracking.log_pilot_screening_event
  6.  Checkpoint-reference metadata, no bytes     -> pilot_tracking.log_pilot_checkpoint_reference
  7.  Clean finish                                -> wandb_tracking.finish_tracking_run
  8.  Continuation using same persisted run ID    -> pilot_tracking.resolve_pilot_wandb_run_id
  9.  No duplicate screening event on replay      -> pilot_orchestration's logged_screening_epochs guard
  10. Simulated logging failure, state intact     -> wandb_tracking._guard_backend_call
  11. Simulated finish failure, nonfatal, recorded -> wandb_tracking._guard_backend_call
  12. Disabled mode never imports wandb           -> wandb_tracking.init_tracking_run
  13. Absent W&B package behavior                 -> wandb_tracking.init_tracking_run / pilot_tracking downgrade
  14. Forbidden credential-like keys               -> _reject_credential_like_keys
  15. No temporal-test/spatial-holdout metadata    -> _reject_disallowed_metric_keys
"""
from __future__ import annotations

import sys
import types
import warnings

import pytest

from src.baseline.pilot_screening_eval import SCREENING_METRIC_SCOPE
from src.baseline.pilot_tracking import (
    init_pilot_tracking_run,
    log_pilot_checkpoint_reference,
    log_pilot_screening_event,
    resolve_pilot_wandb_run_id,
)
from src.baseline.pilot_lead06_config import load_pilot_policy
from src.baseline.wandb_tracking import (
    TrackingError,
    finish_tracking_run,
    init_tracking_run,
    load_tracking_policy,
    log_hyperparameters,
    log_resource_metrics,
    log_scientific_metrics,
)

_REAL_DEFAULT_POLICY_PATH = "config/stage1_wandb_tracking_policy_v001.yaml"
_REAL_PILOT_POLICY_PATH = "config/stage1_lead06_pilot_v001.yaml"


def _offline_policy(**overrides):
    base = {
        "policy_name": "offline_qualification_test",
        "enabled": True,
        "mode": "offline",
        "project": "flashnh-stage1-qualification",
        "entity": None,
        "tags": ["stage1", "offline_qualification"],
        "max_artifact_reference_bytes": 1_048_576,
    }
    base.update(overrides)
    return base


class _FakeWandbConfig(dict):
    def update(self, other=None, allow_val_change=None, **kwargs):
        if other:
            dict.update(self, other)


class _FakeWandbRun:
    def __init__(self, fail_ops: frozenset = frozenset()):
        self.config = _FakeWandbConfig()
        self.summary = {}
        self.logged: list = []
        self.finished = False
        self._fail_ops = fail_ops

    def log(self, data, step=None):
        if "log" in self._fail_ops:
            raise RuntimeError("simulated wandb.log failure")
        self.logged.append((step, dict(data)))

    def finish(self):
        if "finish" in self._fail_ops:
            raise RuntimeError("simulated wandb.finish failure")
        self.finished = True


class _FakeWandbModule(types.ModuleType):
    def __init__(self, fail_ops: frozenset = frozenset()):
        super().__init__("wandb")
        self.init_calls: list = []
        self.last_run: "_FakeWandbRun | None" = None
        self._fail_ops = fail_ops

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        run = _FakeWandbRun(fail_ops=self._fail_ops)
        self.last_run = run
        return run


@pytest.fixture
def fake_wandb(monkeypatch):
    fake = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


@pytest.fixture
def pilot_policy():
    return load_pilot_policy(_REAL_PILOT_POLICY_PATH)


# ---------------------------------------------------------------------------
# 1. Policy parsing with offline mode
# ---------------------------------------------------------------------------

def test_scenario_01_policy_parsing_with_offline_mode(tmp_path):
    import yaml

    p = tmp_path / "offline_policy.yaml"
    p.write_text(yaml.safe_dump(_offline_policy()), encoding="utf-8")
    policy = load_tracking_policy(p)
    assert policy["mode"] == "offline"
    assert policy["enabled"] is True
    # The shipped, committed default remains disabled -- offline mode is
    # opt-in per this qualification, never the real running default.
    real_default = load_tracking_policy(_REAL_DEFAULT_POLICY_PATH)
    assert real_default["enabled"] is False
    assert real_default["mode"] == "disabled"


# ---------------------------------------------------------------------------
# 2. Real W&B init (through the real code path, fake in-process package)
# ---------------------------------------------------------------------------

def test_scenario_02_real_wandb_init_offline_no_network_module(fake_wandb):
    run = init_tracking_run(_offline_policy(), {"run_name": "qualification"})
    assert run.backend == "wandb"
    assert run.mode == "offline"
    assert len(fake_wandb.init_calls) == 1
    assert fake_wandb.init_calls[0]["mode"] == "offline"


# ---------------------------------------------------------------------------
# 3. Full config logging
# ---------------------------------------------------------------------------

def test_scenario_03_full_config_logging(fake_wandb):
    run = init_tracking_run(_offline_policy(), {"run_name": "qualification"})
    hyperparameters = {"model": "cudalstm", "hidden_size": 128, "seed": 42}
    log_hyperparameters(run, hyperparameters)
    assert run.hyperparameters == hyperparameters
    assert dict(run._wandb_run.config) == hyperparameters


# ---------------------------------------------------------------------------
# 4. Epoch / training metric events
# ---------------------------------------------------------------------------

def test_scenario_04_epoch_and_training_metric_events(fake_wandb):
    run = init_tracking_run(_offline_policy(), {"run_name": "qualification"})
    log_scientific_metrics(run, 6, {"median_nse": 0.42})
    log_resource_metrics(run, 6, {"epoch_wall_seconds": 88.0})
    assert run.scientific_metrics == [(6, {"median_nse": 0.42})]
    assert run.resource_metrics == [(6, {"epoch_wall_seconds": 88.0})]
    assert (6, {"median_nse": 0.42}) in run._wandb_run.logged
    assert (6, {"epoch_wall_seconds": 88.0}) in run._wandb_run.logged


# ---------------------------------------------------------------------------
# 5. Screening metric + early-stopping-state logging
# ---------------------------------------------------------------------------

def test_scenario_05_screening_metric_and_early_stopping_state_logging(fake_wandb, pilot_policy, tmp_path, monkeypatch):
    import dataclasses
    import yaml

    p = tmp_path / "offline_policy.yaml"
    p.write_text(yaml.safe_dump(_offline_policy()), encoding="utf-8")
    policy = dataclasses.replace(pilot_policy, wandb_policy_path=str(p))
    run = init_pilot_tracking_run(policy, run_identity={"pilot_policy_name": "test", "run_id": "raw_seedA"})
    assert run.backend == "wandb"

    screening_result = {
        "scope": SCREENING_METRIC_SCOPE,
        "primary_metric_name": "median_per_basin_raw_space_nse",
        "primary_metric_median": 0.31,
        "epoch_role": "stopping_eligible",
        "stopping_eligible": True,
        "n_screening_basins_requested": 350,
        "raw_space_metrics": {"aggregate": {"metrics": {}}},
        "primary_metric_distribution": {},
    }
    early_stopping_state = {
        "best_epoch": 6, "best_metric_value": 0.31,
        "events_since_best_improvement": 0, "stopped": False, "stop_reason": None,
    }
    log_pilot_screening_event(run, epoch=6, screening_result=screening_result, early_stopping_state=early_stopping_state)

    logged_epoch, logged_metrics = run.scientific_metrics[0]
    assert logged_epoch == 6
    assert logged_metrics["screening/primary_metric_median"] == 0.31
    assert logged_metrics["early_stopping/best_epoch"] == 6
    assert logged_metrics["early_stopping/stopped"] is False


# ---------------------------------------------------------------------------
# 6. Checkpoint-reference metadata, never the checkpoint bytes
# ---------------------------------------------------------------------------

def test_scenario_06_checkpoint_reference_metadata_without_uploading_bytes(fake_wandb, tmp_path):
    run = init_tracking_run(_offline_policy(), {"run_name": "qualification"})
    ckpt = tmp_path / "model_epoch006.pt"
    ckpt.write_bytes(b"fake checkpoint bytes, never logged")
    log_pilot_checkpoint_reference(run, epoch=6, path=ckpt, checksum="deadbeef")

    ref = run.artifact_references[0]
    assert ref["checksum"] == "deadbeef"
    assert ref["path"] == str(ckpt)
    assert "size_bytes" in ref
    logged_summary_value = run._wandb_run.summary["checkpoint_ref/epoch_006"]
    assert logged_summary_value == ref
    assert isinstance(logged_summary_value["path"], str)  # a path string, never file content


# ---------------------------------------------------------------------------
# 7. Clean finish
# ---------------------------------------------------------------------------

def test_scenario_07_clean_finish(fake_wandb):
    run = init_tracking_run(_offline_policy(), {"run_name": "qualification"})
    finish_tracking_run(run)
    assert run.finished is True
    assert run.degraded is False
    assert run._wandb_run.finished is True


# ---------------------------------------------------------------------------
# 8. Continuation using the same persisted W&B run ID
# ---------------------------------------------------------------------------

def test_scenario_08_continuation_reuses_same_persisted_run_id(fake_wandb, tmp_path, pilot_policy):
    import dataclasses
    import yaml

    p = tmp_path / "offline_policy.yaml"
    p.write_text(yaml.safe_dump(_offline_policy()), encoding="utf-8")
    policy = dataclasses.replace(pilot_policy, wandb_policy_path=str(p))
    nh_run_dir = tmp_path / "nh_run"
    nh_run_dir.mkdir()
    run_identity = {"pilot_policy_name": "stage1_lead06_pilot_v001", "run_id": "raw_seedA"}

    # Simulates two separate, bounded Slurm jobs continuing one candidate.
    run_job1 = init_pilot_tracking_run(policy, run_identity=run_identity, nh_run_dir=nh_run_dir)
    run_job2 = init_pilot_tracking_run(policy, run_identity=run_identity, nh_run_dir=nh_run_dir)

    assert run_job1.wandb_run_id == run_job2.wandb_run_id
    assert fake_wandb.init_calls[0]["id"] == fake_wandb.init_calls[1]["id"]
    assert fake_wandb.init_calls[1]["resume"] == "allow"


# ---------------------------------------------------------------------------
# 9. No duplicate screening event on replay/resume
# ---------------------------------------------------------------------------

def test_scenario_09_replay_guard_skips_already_logged_epoch(fake_wandb):
    """This exact skip mechanism (an epoch already present in a run's
    persisted ``logged_screening_epochs`` is never re-evaluated or
    re-logged) is pilot_orchestration.run_pilot_chunk's existing,
    unmodified-by-this-task guard (see src/baseline/pilot_orchestration.py,
    "Trust the persisted logged_screening_epochs contract"). Full
    end-to-end integration proof (real chunk replay, real early-stopping
    history, exactly the shape of Moriah job 45718742) lives in
    tests/test_pilot_orchestration.py::
    test_run_pilot_end_to_end_rerun_of_fully_screened_earlier_chunks_is_idempotent
    and
    test_run_pilot_chunk_screening_log_failure_does_not_break_orchestration_state's
    sibling tests. This test isolates just the guard's effect on tracking:
    a screening event logged once is never logged twice for the same
    epoch."""
    run = init_tracking_run(_offline_policy(), {"run_name": "qualification"})
    logged_epochs: set = set()

    def maybe_log_screening_event(epoch):
        if epoch in logged_epochs:
            return  # the orchestration-level guard this test isolates
        log_scientific_metrics(run, epoch, {"screening/primary_metric_median": 0.31})
        logged_epochs.add(epoch)

    maybe_log_screening_event(6)
    maybe_log_screening_event(6)  # replay/resume re-attempt
    assert len(run.scientific_metrics) == 1
    assert len(run._wandb_run.logged) == 1


# ---------------------------------------------------------------------------
# 10. Simulated logging failure leaves scientific/orchestration state intact
# ---------------------------------------------------------------------------

def test_scenario_10_simulated_logging_failure_leaves_state_intact(monkeypatch):
    fake = _FakeWandbModule(fail_ops=frozenset({"log"}))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    run = init_tracking_run(_offline_policy(), {"run_name": "qualification"})

    with pytest.warns(RuntimeWarning):
        log_scientific_metrics(run, 6, {"median_nse": 0.31})

    # The scientific record survives the backend failure unchanged.
    assert run.scientific_metrics == [(6, {"median_nse": 0.31})]
    assert run.degraded is True
    assert "log_scientific_metrics" in run.degraded_operations
    # Full orchestration-level proof (early-stopping state + orchestration
    # state files both still written correctly despite every W&B call
    # failing) lives in tests/test_pilot_orchestration.py::
    # test_run_pilot_chunk_screening_log_failure_does_not_break_orchestration_state.


# ---------------------------------------------------------------------------
# 11. Simulated finish failure is nonfatal but recorded
# ---------------------------------------------------------------------------

def test_scenario_11_simulated_finish_failure_nonfatal_but_recorded(monkeypatch):
    fake = _FakeWandbModule(fail_ops=frozenset({"finish"}))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    run = init_tracking_run(_offline_policy(), {"run_name": "qualification"})

    with pytest.warns(RuntimeWarning):
        finish_tracking_run(run)  # must not raise

    assert run.finished is True
    assert run.degraded is True
    assert "finish_tracking_run" in run.degraded_operations


# ---------------------------------------------------------------------------
# 12. Disabled mode never imports wandb
# ---------------------------------------------------------------------------

def test_scenario_12_disabled_mode_never_imports_wandb(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)  # any import attempt raises ImportError
    policy = {
        "policy_name": "disabled_test", "enabled": False, "mode": "disabled",
        "project": "x", "entity": None, "tags": [], "max_artifact_reference_bytes": 1024,
    }
    run = init_tracking_run(policy, {"run_name": "qualification"})
    assert run.backend == "null"
    assert run.mode == "disabled"
    log_scientific_metrics(run, 1, {"median_nse": 0.1})
    finish_tracking_run(run)
    assert run.finished is True


# ---------------------------------------------------------------------------
# 13. Absent W&B package behavior
# ---------------------------------------------------------------------------

def test_scenario_13_absent_wandb_package_raises_at_wrapper_layer(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(TrackingError):
        init_tracking_run(_offline_policy(), {"run_name": "qualification"})


def test_scenario_13_absent_wandb_package_downgrades_gracefully_at_pilot_layer(monkeypatch, pilot_policy, tmp_path):
    import dataclasses
    import yaml

    monkeypatch.setitem(sys.modules, "wandb", None)
    p = tmp_path / "offline_policy.yaml"
    p.write_text(yaml.safe_dump(_offline_policy()), encoding="utf-8")
    policy = dataclasses.replace(pilot_policy, wandb_policy_path=str(p))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = init_pilot_tracking_run(policy, run_identity={"pilot_policy_name": "test", "run_id": "raw_seedA"})
    assert run.backend == "null"  # training is never blocked by a missing wandb package
    assert any("W&B tracking init failed" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# 14. Forbidden credential-like config/metric keys
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_key", ["api_key", "WANDB_API_KEY", "secret_token", "password", "auth_token"])
def test_scenario_14_forbidden_credential_like_keys_rejected(fake_wandb, bad_key):
    with pytest.raises(TrackingError):
        init_tracking_run(_offline_policy(), {"run_name": "x", bad_key: "should-not-be-here"})
    run = init_tracking_run(_offline_policy(), {"run_name": "x"})
    with pytest.raises(TrackingError):
        log_hyperparameters(run, {bad_key: "should-not-be-here"})
    with pytest.raises(TrackingError):
        log_resource_metrics(run, 1, {bad_key: "should-not-be-here"})


# ---------------------------------------------------------------------------
# 15. No temporal-test / spatial-holdout metadata
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_key", ["temporal_test_nse", "spatial_holdout_nse", "test_nse", "holdout_median"])
def test_scenario_15_no_temporal_test_or_spatial_holdout_metadata(fake_wandb, bad_key):
    run = init_tracking_run(_offline_policy(), {"run_name": "x"})
    with pytest.raises(TrackingError):
        log_scientific_metrics(run, 1, {bad_key: 0.5})

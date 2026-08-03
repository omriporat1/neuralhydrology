"""Tests for src/baseline/wandb_tracking.py.

Synthetic fixtures only, plus a fake in-process ``wandb`` module (installed
via monkeypatching ``sys.modules``) so the real-backend code path can be
exercised deterministically without the wandb package installed or any
network access -- consistent with section 10's "offline-mode-supported, no
credential exposure, tracking only" requirements.

In addition to behavioural checks, this module includes a structural test
proving -- by inspecting function signatures, not just by testing behaviour
-- that no public function in wandb_tracking.py has a parameter shaped like
a credential, and a behavioural test proving no function accepts or forwards
temporal-test/spatial-holdout-shaped metric keys.
"""
from __future__ import annotations

import inspect
import sys
import types

import pytest

from src.baseline.wandb_tracking import (
    TrackingError,
    TrackingRun,
    finish_tracking_run,
    init_tracking_run,
    load_tracking_policy,
    log_artifact_reference,
    log_checkpoint_reference,
    log_hyperparameters,
    log_resource_metrics,
    log_scientific_metrics,
)

_REAL_POLICY_PATH = "config/stage1_wandb_tracking_policy_v001.yaml"


def _policy(**overrides):
    base = {
        "policy_name": "test_policy",
        "enabled": False,
        "mode": "disabled",
        "project": "flashnh-stage1-test",
        "entity": None,
        "tags": ["stage1"],
        "max_artifact_reference_bytes": 1024,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Fake wandb module for exercising the "wandb" backend without the real
# package installed and without any network access.
# ---------------------------------------------------------------------------

class _FakeWandbConfig(dict):
    def update(self, other=None, allow_val_change=None, **kwargs):
        if other:
            dict.update(self, other)
        if kwargs:
            dict.update(self, kwargs)


class _FailingSummaryDict(dict):
    """A wandb ``run.summary``-shaped dict whose ``__setitem__`` always
    raises -- used to prove checkpoint-reference/artifact-reference backend
    writes are isolated the same way as every other backend call."""

    def __setitem__(self, key, value):
        raise RuntimeError("simulated wandb summary write failure")


class _FakeWandbRun:
    def __init__(self, fail_ops: frozenset[str] = frozenset()):
        self.config = _FakeWandbConfig()
        self.summary = _FailingSummaryDict() if "summary" in fail_ops else {}
        self.logged: list[tuple[int | None, dict]] = []
        self.finished = False
        self.log_call_count = 0
        self._fail_ops = fail_ops

    def log(self, data, step=None):
        self.log_call_count += 1
        if "log" in self._fail_ops:
            raise RuntimeError("simulated wandb.log failure")
        self.logged.append((step, dict(data)))

    def finish(self):
        if "finish" in self._fail_ops:
            raise RuntimeError("simulated wandb.finish failure")
        self.finished = True


class _FakeWandbModule(types.ModuleType):
    def __init__(self, fail_ops: frozenset[str] = frozenset()):
        super().__init__("wandb")
        self.init_calls: list[dict] = []
        self.last_run: _FakeWandbRun | None = None
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
def fake_wandb_failing_log(monkeypatch):
    """A fake wandb backend whose run.log() always raises -- used to prove
    the failure-isolation boundary around log_scientific_metrics/
    log_resource_metrics."""
    fake = _FakeWandbModule(fail_ops=frozenset({"log"}))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


@pytest.fixture
def fake_wandb_failing_finish(monkeypatch):
    """A fake wandb backend whose run.finish() always raises -- used to
    prove finish_tracking_run stays non-fatal."""
    fake = _FakeWandbModule(fail_ops=frozenset({"finish"}))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


@pytest.fixture
def fake_wandb_failing_summary(monkeypatch):
    """A fake wandb backend whose run.summary[...] = ... always raises --
    used to prove log_checkpoint_reference's backend write is isolated."""
    fake = _FakeWandbModule(fail_ops=frozenset({"summary"}))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


# ---------------------------------------------------------------------------
# load_tracking_policy
# ---------------------------------------------------------------------------

def test_load_tracking_policy_real_config():
    policy = load_tracking_policy(_REAL_POLICY_PATH)
    assert policy["enabled"] is False
    assert policy["mode"] == "disabled"
    assert policy["max_artifact_reference_bytes"] > 0


def test_load_tracking_policy_missing_file():
    with pytest.raises(TrackingError):
        load_tracking_policy("does/not/exist.yaml")


def test_load_tracking_policy_missing_key(tmp_path):
    import yaml

    bad = _policy()
    del bad["mode"]
    p = tmp_path / "policy.yaml"
    with open(p, "w", encoding="utf-8") as fh:
        yaml.safe_dump(bad, fh)
    with pytest.raises(TrackingError):
        load_tracking_policy(p)


def test_load_tracking_policy_rejects_invalid_mode(tmp_path):
    import yaml

    bad = _policy(mode="sometimes")
    p = tmp_path / "policy.yaml"
    with open(p, "w", encoding="utf-8") as fh:
        yaml.safe_dump(bad, fh)
    with pytest.raises(TrackingError):
        load_tracking_policy(p)


def test_load_tracking_policy_rejects_nonpositive_artifact_limit(tmp_path):
    import yaml

    bad = _policy(max_artifact_reference_bytes=0)
    p = tmp_path / "policy.yaml"
    with open(p, "w", encoding="utf-8") as fh:
        yaml.safe_dump(bad, fh)
    with pytest.raises(TrackingError):
        load_tracking_policy(p)


# ---------------------------------------------------------------------------
# Disabled / null-sink backend (the shipped default)
# ---------------------------------------------------------------------------

def test_disabled_policy_never_imports_wandb(monkeypatch):
    # Simulate wandb genuinely not being installed: any accidental import
    # attempt raises ImportError immediately.
    monkeypatch.setitem(sys.modules, "wandb", None)
    policy = _policy(enabled=False, mode="disabled")
    run = init_tracking_run(policy, {"run_name": "r1"})
    assert run.backend == "null"
    log_hyperparameters(run, {"hidden_size": 128})
    log_scientific_metrics(run, 1, {"median_nse": 0.20})
    finish_tracking_run(run)
    assert run.finished is True


def test_null_run_records_full_local_mirror(tmp_path):
    policy = _policy()
    run = init_tracking_run(policy, {"run_name": "r1", "seed": 42})
    log_hyperparameters(run, {"hidden_size": 128, "batch_size": 256})
    log_scientific_metrics(run, 1, {"median_nse": 0.20, "p50_nse": 0.20})
    log_scientific_metrics(run, 2, {"median_nse": 0.23, "p50_nse": 0.23})
    log_resource_metrics(run, 1, {"epoch_wall_seconds": 120.5})

    artifact_path = tmp_path / "small_manifest.json"
    artifact_path.write_text('{"ok": true}', encoding="utf-8")
    log_artifact_reference(run, "manifest", artifact_path, checksum="abc123")

    finish_tracking_run(run)

    assert run.hyperparameters == {"hidden_size": 128, "batch_size": 256}
    assert [e for e in run.scientific_metrics] == [
        (1, {"median_nse": 0.20, "p50_nse": 0.20}),
        (2, {"median_nse": 0.23, "p50_nse": 0.23}),
    ]
    assert run.resource_metrics == [(1, {"epoch_wall_seconds": 120.5})]
    assert len(run.artifact_references) == 1
    assert run.artifact_references[0]["name"] == "manifest"
    assert run.artifact_references[0]["checksum"] == "abc123"
    assert run.finished is True


def test_resource_metrics_no_op_when_nothing_captured():
    policy = _policy()
    run = init_tracking_run(policy, {"run_name": "r1"})
    log_resource_metrics(run, 1, {})
    log_resource_metrics(run, 2, None)
    assert run.resource_metrics == []


# ---------------------------------------------------------------------------
# Real (fake) wandb backend
# ---------------------------------------------------------------------------

def test_enabled_offline_mode_routes_through_wandb(fake_wandb):
    policy = _policy(enabled=True, mode="offline")
    run = init_tracking_run(policy, {"run_name": "r1"})
    assert run.backend == "wandb"
    assert len(fake_wandb.init_calls) == 1
    assert fake_wandb.init_calls[0]["mode"] == "offline"
    assert fake_wandb.init_calls[0]["project"] == "flashnh-stage1-test"

    log_hyperparameters(run, {"hidden_size": 128})
    assert run._wandb_run.config["hidden_size"] == 128

    log_scientific_metrics(run, 3, {"median_nse": 0.24})
    assert run._wandb_run.logged[-1] == (3, {"median_nse": 0.24})

    finish_tracking_run(run)
    assert run._wandb_run.finished is True


def test_enabled_but_wandb_not_installed_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    policy = _policy(enabled=True, mode="offline")
    with pytest.raises(TrackingError):
        init_tracking_run(policy, {"run_name": "r1"})


# ---------------------------------------------------------------------------
# Credential-exposure guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_key", ["api_key", "WANDB_API_KEY", "secret_token", "password", "auth_token"])
def test_run_identity_rejects_credential_like_keys(bad_key):
    policy = _policy()
    with pytest.raises(TrackingError):
        init_tracking_run(policy, {"run_name": "r1", bad_key: "should-not-be-here"})


@pytest.mark.parametrize("bad_key", ["api_key", "secret", "password"])
def test_hyperparameters_rejects_credential_like_keys(bad_key):
    policy = _policy()
    run = init_tracking_run(policy, {"run_name": "r1"})
    with pytest.raises(TrackingError):
        log_hyperparameters(run, {bad_key: "x"})


def test_resource_metrics_rejects_credential_like_keys():
    policy = _policy()
    run = init_tracking_run(policy, {"run_name": "r1"})
    with pytest.raises(TrackingError):
        log_resource_metrics(run, 1, {"api_key": "x"})


# ---------------------------------------------------------------------------
# Artifact size ceiling (never large prediction/checkpoint/NetCDF/Parquet)
# ---------------------------------------------------------------------------

def test_artifact_reference_rejects_oversized_file(tmp_path):
    policy = _policy(max_artifact_reference_bytes=10)
    run = init_tracking_run(policy, {"run_name": "r1"})
    big = tmp_path / "big.bin"
    big.write_bytes(b"0123456789ABCDEF")  # 16 bytes > 10-byte limit
    with pytest.raises(TrackingError):
        log_artifact_reference(run, "big", big, checksum="deadbeef")


def test_artifact_reference_rejects_missing_file(tmp_path):
    policy = _policy()
    run = init_tracking_run(policy, {"run_name": "r1"})
    with pytest.raises(TrackingError):
        log_artifact_reference(run, "missing", tmp_path / "nope.bin", checksum="x")


# ---------------------------------------------------------------------------
# Checkpoint references: metadata-only, never subject to the compact-artifact
# size ceiling, and never fatal (job 45731908 postmortem).
# ---------------------------------------------------------------------------

def test_checkpoint_reference_ignores_artifact_size_ceiling(tmp_path):
    # A real NH checkpoint (~1.25 MB in the job 45731908 postmortem) is
    # always far above a "compact artifact" ceiling -- log_artifact_reference
    # would refuse it; log_checkpoint_reference must not.
    policy = _policy(max_artifact_reference_bytes=10)
    run = init_tracking_run(policy, {"run_name": "r1"})
    ckpt = tmp_path / "model_epoch006.pt"
    ckpt.write_bytes(b"0" * 5000)
    log_checkpoint_reference(run, epoch=6, path=ckpt, checksum="deadbeef")
    assert run.degraded is False
    assert len(run.artifact_references) == 1
    record = run.artifact_references[0]
    assert record["epoch"] == 6
    assert record["size_bytes"] == 5000
    assert record["checkpoint_type"] == "nh_model_checkpoint"
    assert record["checksum"] == "deadbeef"
    assert record["path"] == str(ckpt)


def test_checkpoint_reference_missing_file_degrades_not_raises(tmp_path):
    policy = _policy()
    run = init_tracking_run(policy, {"run_name": "r1"})
    with pytest.warns(RuntimeWarning, match="log_checkpoint_reference"):
        log_checkpoint_reference(run, epoch=6, path=tmp_path / "nope.pt", checksum="x")
    assert run.degraded is True
    assert "log_checkpoint_reference" in run.degraded_operations
    assert run.artifact_references == []


def test_checkpoint_reference_backend_failure_is_nonfatal(fake_wandb_failing_summary, tmp_path):
    policy = _policy(enabled=True, mode="offline", max_artifact_reference_bytes=10)
    run = init_tracking_run(policy, {"run_name": "r1"})
    ckpt = tmp_path / "model_epoch006.pt"
    ckpt.write_bytes(b"0" * 5000)

    with pytest.warns(RuntimeWarning, match="log_checkpoint_reference"):
        log_checkpoint_reference(run, epoch=6, path=ckpt, checksum="deadbeef")

    assert run.degraded is True
    assert "log_checkpoint_reference" in run.degraded_operations
    # Local mirror is still recorded even though the backend write failed --
    # this is optional telemetry, so scientific/evidence state must not be
    # lost just because wandb itself failed.
    assert run.artifact_references[0]["epoch"] == 6


# ---------------------------------------------------------------------------
# Structural proof: no temporal-test / spatial-holdout metric keys.
# ---------------------------------------------------------------------------

def test_scientific_metrics_rejects_disallowed_keys():
    policy = _policy()
    run = init_tracking_run(policy, {"run_name": "r1"})
    with pytest.raises(TrackingError):
        log_scientific_metrics(run, 1, {"temporal_test_nse": 0.5})
    with pytest.raises(TrackingError):
        log_scientific_metrics(run, 1, {"spatial_holdout_nse": 0.5})


# ---------------------------------------------------------------------------
# Structural proof: no credential-shaped parameter in any public function.
# ---------------------------------------------------------------------------

_DISALLOWED_PARAM_NAME_FRAGMENTS = ("api_key", "apikey", "secret", "password", "token", "credential")


@pytest.mark.parametrize(
    "func",
    [
        load_tracking_policy,
        init_tracking_run,
        log_hyperparameters,
        log_scientific_metrics,
        log_resource_metrics,
        log_artifact_reference,
        log_checkpoint_reference,
        finish_tracking_run,
    ],
)
def test_public_function_signatures_have_no_credential_argument(func):
    params = list(inspect.signature(func).parameters)
    for name in params:
        assert not any(frag in name.lower() for frag in _DISALLOWED_PARAM_NAME_FRAGMENTS), (
            f"{func.__name__} has disallowed parameter name {name!r}"
        )


def test_module_public_api_has_no_credential_symbol():
    import src.baseline.wandb_tracking as mod

    for name in mod.__all__:
        assert not any(frag in name.lower() for frag in _DISALLOWED_PARAM_NAME_FRAGMENTS)


# ---------------------------------------------------------------------------
# Failure isolation: a backend failure after init must never propagate, and
# must never silently claim tracking succeeded.
# ---------------------------------------------------------------------------

def test_log_scientific_metrics_failure_isolated_and_recorded(fake_wandb_failing_log):
    policy = _policy(enabled=True, mode="offline")
    run = init_tracking_run(policy, {"run_name": "r1"})
    assert run.degraded is False

    with pytest.warns(RuntimeWarning, match="log_scientific_metrics"):
        log_scientific_metrics(run, 1, {"median_nse": 0.2})

    # Scientific/local state is preserved even though the backend call failed.
    assert run.scientific_metrics == [(1, {"median_nse": 0.2})]
    assert run.degraded is True
    assert "log_scientific_metrics" in run.degraded_operations


def test_repeated_logging_failure_warns_only_once(fake_wandb_failing_log):
    policy = _policy(enabled=True, mode="offline")
    run = init_tracking_run(policy, {"run_name": "r1"})

    with pytest.warns(RuntimeWarning):
        log_scientific_metrics(run, 1, {"median_nse": 0.2})

    import warnings as warnings_mod

    with warnings_mod.catch_warnings(record=True) as caught:
        warnings_mod.simplefilter("always")
        log_scientific_metrics(run, 2, {"median_nse": 0.25})
        log_scientific_metrics(run, 3, {"median_nse": 0.30})
    assert len(caught) == 0  # no repeat warning for the same operation
    assert run.scientific_metrics[-1] == (3, {"median_nse": 0.30})


def test_finish_tracking_run_failure_is_nonfatal_and_recorded(fake_wandb_failing_finish):
    policy = _policy(enabled=True, mode="offline")
    run = init_tracking_run(policy, {"run_name": "r1"})

    with pytest.warns(RuntimeWarning, match="finish_tracking_run"):
        finish_tracking_run(run)

    assert run.finished is True  # local/scientific completion state still recorded
    assert run.degraded is True
    assert "finish_tracking_run" in run.degraded_operations


def test_resource_metrics_failure_does_not_affect_scientific_metrics(fake_wandb_failing_log):
    policy = _policy(enabled=True, mode="offline")
    run = init_tracking_run(policy, {"run_name": "r1"})

    with pytest.warns(RuntimeWarning):
        log_resource_metrics(run, 1, {"epoch_wall_seconds": 12.0})
    # A different, unrelated (non-backend) call proceeds normally.
    log_scientific_metrics_call_ok = True
    try:
        with pytest.warns(RuntimeWarning):
            log_scientific_metrics(run, 1, {"median_nse": 0.2})
    except AssertionError:
        log_scientific_metrics_call_ok = False
    assert log_scientific_metrics_call_ok
    assert run.resource_metrics == [(1, {"epoch_wall_seconds": 12.0})]
    assert run.scientific_metrics == [(1, {"median_nse": 0.2})]
    # Two distinct operations both failed -- per-operation warning dedup must
    # not collapse them into a single hidden failure.
    assert run.degraded_operations == {"log_resource_metrics", "log_scientific_metrics"}


def test_null_backend_never_degrades():
    policy = _policy()
    run = init_tracking_run(policy, {"run_name": "r1"})
    log_scientific_metrics(run, 1, {"median_nse": 0.2})
    finish_tracking_run(run)
    assert run.degraded is False
    assert run.degraded_operations == set()


# ---------------------------------------------------------------------------
# Stable run identity across restarts: run_id / resume pass-through.
# ---------------------------------------------------------------------------

def test_init_tracking_run_passes_run_id_and_default_resume(fake_wandb):
    policy = _policy(enabled=True, mode="offline")
    run = init_tracking_run(policy, {"run_name": "r1"}, run_id="flashnh-emb128x64-seedA")
    assert fake_wandb.init_calls[0]["id"] == "flashnh-emb128x64-seedA"
    assert fake_wandb.init_calls[0]["resume"] == "allow"
    assert run.wandb_run_id == "flashnh-emb128x64-seedA"


def test_init_tracking_run_without_run_id_passes_none(fake_wandb):
    policy = _policy(enabled=True, mode="offline")
    run = init_tracking_run(policy, {"run_name": "r1"})
    assert fake_wandb.init_calls[0]["id"] is None
    assert fake_wandb.init_calls[0]["resume"] is None
    assert run.wandb_run_id is None


def test_init_tracking_run_explicit_resume_overrides_default(fake_wandb):
    policy = _policy(enabled=True, mode="offline")
    init_tracking_run(policy, {"run_name": "r1"}, run_id="rid-1", resume="must")
    assert fake_wandb.init_calls[0]["resume"] == "must"


def test_same_run_id_reused_across_two_init_calls_is_recorded_identically(fake_wandb):
    policy = _policy(enabled=True, mode="offline")
    run_a = init_tracking_run(policy, {"run_name": "r1"}, run_id="rid-stable")
    run_b = init_tracking_run(policy, {"run_name": "r1"}, run_id="rid-stable")
    assert run_a.wandb_run_id == run_b.wandb_run_id == "rid-stable"
    assert fake_wandb.init_calls[0]["id"] == fake_wandb.init_calls[1]["id"] == "rid-stable"


def test_disabled_mode_does_not_require_run_id(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    policy = _policy(enabled=False, mode="disabled")
    run = init_tracking_run(policy, {"run_name": "r1"})
    assert run.backend == "null"
    assert run.wandb_run_id is None
    assert run.mode == "disabled"


def test_tracking_run_mode_field_matches_policy(fake_wandb):
    policy = _policy(enabled=True, mode="offline")
    run = init_tracking_run(policy, {"run_name": "r1"})
    assert run.mode == "offline"

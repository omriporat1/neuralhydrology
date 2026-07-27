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


class _FakeWandbRun:
    def __init__(self):
        self.config = _FakeWandbConfig()
        self.summary = {}
        self.logged: list[tuple[int | None, dict]] = []
        self.finished = False

    def log(self, data, step=None):
        self.logged.append((step, dict(data)))

    def finish(self):
        self.finished = True


class _FakeWandbModule(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.init_calls: list[dict] = []
        self.last_run: _FakeWandbRun | None = None

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        run = _FakeWandbRun()
        self.last_run = run
        return run


@pytest.fixture
def fake_wandb(monkeypatch):
    fake = _FakeWandbModule()
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

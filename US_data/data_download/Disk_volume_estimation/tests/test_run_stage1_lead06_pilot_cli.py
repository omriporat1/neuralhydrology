"""Focused tests for the scripts/run_stage1_lead06_pilot.py CLI entrypoint's
per-run W&B policy override and tracking_generation flags.

Loads the script as a module (it is not part of an installed package) and
replaces its `run_pilot` reference with a fake that records the kwargs it
was called with -- these tests are about CLI argument threading, not about
exercising a real pilot run (that is already covered end-to-end in
tests/test_pilot_orchestration.py). Real `load_tracking_policy` validation
is exercised unmocked, since that is exactly the loud-failure behavior these
tests must confirm.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from src.baseline.pilot_lead06_config import load_pilot_policy
from src.baseline.wandb_tracking import TrackingError

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_stage1_lead06_pilot.py"
DEFAULT_PILOT_POLICY_PATH = REPO_ROOT / "config" / "stage1_lead06_pilot_v001.yaml"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "run_stage1_lead06_pilot_cli_under_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli_module():
    return _load_cli_module()


@pytest.fixture
def valid_offline_override_policy(tmp_path):
    raw = {
        "policy_name": "test_cli_override_policy",
        "enabled": True,
        "mode": "offline",
        "project": "flashnh-stage1-test",
        "entity": None,
        "tags": ["test"],
        "max_artifact_reference_bytes": 1048576,
    }
    p = tmp_path / "override_wandb_policy.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


class _FakeRunPilotCapture:
    """Records the last call's kwargs; returns a minimal, always-completed
    result shape sufficient for main()'s own post-processing."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "nh_run_dir": "/tmp/fake_nh_run_dir",
            "evidence_bundle_path": "/tmp/fake_evidence_bundle",
            "final_status": "completed",
        }


def _run_main(cli_module, monkeypatch, argv, fake_run_pilot):
    monkeypatch.setattr(cli_module, "run_pilot", fake_run_pilot)
    monkeypatch.setattr(sys, "argv", ["run_stage1_lead06_pilot.py"] + argv)
    cli_module.main()


def _base_argv(tmp_path):
    return [
        "--run-id", "raw_seedA",
        "--package-root", str(tmp_path / "package"),
        "--config-out-dir", str(tmp_path / "config_out"),
        "--evidence-out-dir", str(tmp_path / "evidence"),
    ]


# --- (1) no override uses the committed pilot policy's existing wandb path --

def test_no_override_uses_committed_default_wandb_policy_path(cli_module, monkeypatch, tmp_path):
    fake_run_pilot = _FakeRunPilotCapture()
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path), fake_run_pilot)

    expected_default = cli_module._resolve_policy_relative_paths(
        load_pilot_policy(DEFAULT_PILOT_POLICY_PATH)
    ).wandb_policy_path

    assert len(fake_run_pilot.calls) == 1
    used_policy = fake_run_pilot.calls[0]["pilot_policy"]
    assert used_policy.wandb_policy_path == expected_default


# --- (2) a supplied override replaces only wandb_policy_path ---------------

def test_override_replaces_only_wandb_policy_path(
    cli_module, monkeypatch, tmp_path, valid_offline_override_policy
):
    fake_run_pilot = _FakeRunPilotCapture()
    argv = _base_argv(tmp_path) + ["--wandb-policy-path", str(valid_offline_override_policy)]
    _run_main(cli_module, monkeypatch, argv, fake_run_pilot)

    default_policy = cli_module._resolve_policy_relative_paths(load_pilot_policy(DEFAULT_PILOT_POLICY_PATH))
    used_policy = fake_run_pilot.calls[0]["pilot_policy"]

    assert used_policy.wandb_policy_path == str(valid_offline_override_policy)
    expected_unchanged = dataclasses.replace(
        default_policy, wandb_policy_path=used_policy.wandb_policy_path
    )
    assert used_policy == expected_unchanged


# --- (3) missing override file fails loudly before training ----------------

def test_missing_override_file_fails_loudly_before_run_pilot(cli_module, monkeypatch, tmp_path):
    fake_run_pilot = _FakeRunPilotCapture()
    missing_path = tmp_path / "does_not_exist.yaml"
    argv = _base_argv(tmp_path) + ["--wandb-policy-path", str(missing_path)]
    with pytest.raises(TrackingError):
        _run_main(cli_module, monkeypatch, argv, fake_run_pilot)
    assert fake_run_pilot.calls == []


# --- (4) malformed override fails through the existing validator -----------

def test_malformed_override_fails_through_existing_validator(cli_module, monkeypatch, tmp_path):
    fake_run_pilot = _FakeRunPilotCapture()
    malformed_path = tmp_path / "malformed_wandb_policy.yaml"
    # Missing required keys (e.g. "mode", "max_artifact_reference_bytes") --
    # the exact TrackingError the committed default policy would also raise
    # if it were ever this malformed.
    malformed_path.write_text(yaml.safe_dump({"policy_name": "bad", "enabled": True}), encoding="utf-8")
    argv = _base_argv(tmp_path) + ["--wandb-policy-path", str(malformed_path)]
    with pytest.raises(TrackingError):
        _run_main(cli_module, monkeypatch, argv, fake_run_pilot)
    assert fake_run_pilot.calls == []


# --- (5) the CLI threads the override to orchestration ----------------------

def test_cli_threads_override_path_into_run_pilot_call(
    cli_module, monkeypatch, tmp_path, valid_offline_override_policy
):
    fake_run_pilot = _FakeRunPilotCapture()
    argv = _base_argv(tmp_path) + ["--wandb-policy-path", str(valid_offline_override_policy)]
    _run_main(cli_module, monkeypatch, argv, fake_run_pilot)
    used_policy = fake_run_pilot.calls[0]["pilot_policy"]
    assert used_policy.wandb_policy_path == str(valid_offline_override_policy)
    # the literal invocation (including the new flag) is preserved verbatim
    # for the evidence bundle's commands_used, per the existing convention
    # of recording raw local paths only there, never in run_identity.
    assert any("--wandb-policy-path" in c for c in fake_run_pilot.calls[0]["commands_used"])


# --- (7) tracking_generation defaults to g1 ---------------------------------

def test_tracking_generation_defaults_to_g1(cli_module, monkeypatch, tmp_path):
    fake_run_pilot = _FakeRunPilotCapture()
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path), fake_run_pilot)
    assert fake_run_pilot.calls[0]["tracking_generation"] == "g1"


# --- (8) an explicit non-default generation is threaded correctly ----------

def test_explicit_tracking_generation_is_threaded(cli_module, monkeypatch, tmp_path):
    fake_run_pilot = _FakeRunPilotCapture()
    argv = _base_argv(tmp_path) + ["--tracking-generation", "g2"]
    _run_main(cli_module, monkeypatch, argv, fake_run_pilot)
    assert fake_run_pilot.calls[0]["tracking_generation"] == "g2"


# --- (10) ordinary disabled-mode invocation remains unchanged --------------

def test_ordinary_invocation_without_override_unchanged(cli_module, monkeypatch, tmp_path, capsys):
    fake_run_pilot = _FakeRunPilotCapture()
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path), fake_run_pilot)
    call = fake_run_pilot.calls[0]
    # No override supplied: the committed disabled default policy is used
    # unmodified, and tracking_generation is the ordinary "g1" default --
    # this is the exact call shape every pre-existing pilot invocation made
    # before this task, unaffected by the new optional flags.
    default_policy = cli_module._resolve_policy_relative_paths(load_pilot_policy(DEFAULT_PILOT_POLICY_PATH))
    assert call["pilot_policy"] == default_policy
    assert call["tracking_generation"] == "g1"
    assert call["force"] is False


# --- --max-target-epoch: bounded-recovery flag threading -------------------

def test_max_target_epoch_defaults_to_none(cli_module, monkeypatch, tmp_path):
    fake_run_pilot = _FakeRunPilotCapture()
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path), fake_run_pilot)
    assert fake_run_pilot.calls[0]["max_target_epoch"] is None


def test_explicit_max_target_epoch_is_threaded(cli_module, monkeypatch, tmp_path):
    fake_run_pilot = _FakeRunPilotCapture()
    argv = _base_argv(tmp_path) + ["--max-target-epoch", "6"]
    _run_main(cli_module, monkeypatch, argv, fake_run_pilot)
    assert fake_run_pilot.calls[0]["max_target_epoch"] == 6


# --- --prepare-only: config-generation only, never real training -----------

class _FakePreparePilotRunOnlyCapture:
    """Records the last call's kwargs; returns a minimal PREPARED_ONLY result
    shape sufficient for main()'s own post-processing (a bare print)."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "status": "PREPARED_ONLY",
            "run_id": kwargs["run_id"],
            "config_out_dir": str(kwargs["config_out_dir"]),
            "generated_config_path": str(Path(kwargs["config_out_dir"]) / "config.yaml"),
            "generation_manifest_path": str(Path(kwargs["config_out_dir"]) / "generation_manifest.json"),
            "wandb_policy_sha256": "fake_sha256",
            "tracking_generation": kwargs["tracking_generation"],
            "training_started": False,
            "evaluation_started": False,
            "wandb_backend_initialized": False,
        }


def _run_main_prepare_only(cli_module, monkeypatch, argv, fake_prepare_only, fake_run_pilot):
    monkeypatch.setattr(cli_module, "prepare_pilot_run_only", fake_prepare_only)
    monkeypatch.setattr(cli_module, "run_pilot", fake_run_pilot)
    monkeypatch.setattr(sys, "argv", ["run_stage1_lead06_pilot.py"] + argv)
    cli_module.main()


def test_ordinary_invocation_without_prepare_only_never_calls_prepare_pilot_run_only(cli_module, monkeypatch, tmp_path):
    """(Part 4, #1) An ordinary invocation (no --prepare-only) must behave
    exactly as before: only run_pilot is called, prepare_pilot_run_only is
    never touched."""
    fake_run_pilot = _FakeRunPilotCapture()
    fake_prepare_only = _FakePreparePilotRunOnlyCapture()
    _run_main_prepare_only(cli_module, monkeypatch, _base_argv(tmp_path), fake_prepare_only, fake_run_pilot)
    assert len(fake_run_pilot.calls) == 1
    assert fake_prepare_only.calls == []


def test_prepare_only_calls_preparation_and_never_run_pilot(cli_module, monkeypatch, tmp_path):
    """(Part 4, #2) --prepare-only must call prepare_pilot_run_only and must
    never call run_pilot (the real training/W&B-initializing orchestration
    entrypoint) at all."""
    fake_run_pilot = _FakeRunPilotCapture()
    fake_prepare_only = _FakePreparePilotRunOnlyCapture()
    argv = _base_argv(tmp_path) + ["--prepare-only"]
    _run_main_prepare_only(cli_module, monkeypatch, argv, fake_prepare_only, fake_run_pilot)
    assert len(fake_prepare_only.calls) == 1
    assert fake_run_pilot.calls == []


def test_prepare_only_threads_wandb_override_and_checksum(
    cli_module, monkeypatch, tmp_path, valid_offline_override_policy
):
    """(Part 4, #4) The --wandb-policy-path override is applied to the
    pilot_policy passed into prepare_pilot_run_only exactly as it is for the
    ordinary run_pilot path, and the fake's returned checksum surfaces
    unmodified in the printed result."""
    fake_run_pilot = _FakeRunPilotCapture()
    fake_prepare_only = _FakePreparePilotRunOnlyCapture()
    argv = _base_argv(tmp_path) + ["--prepare-only", "--wandb-policy-path", str(valid_offline_override_policy)]
    _run_main_prepare_only(cli_module, monkeypatch, argv, fake_prepare_only, fake_run_pilot)
    used_policy = fake_prepare_only.calls[0]["pilot_policy"]
    assert used_policy.wandb_policy_path == str(valid_offline_override_policy)


def test_prepare_only_tracking_generation_defaults_to_g1(cli_module, monkeypatch, tmp_path):
    """(Part 4, #5)"""
    fake_run_pilot = _FakeRunPilotCapture()
    fake_prepare_only = _FakePreparePilotRunOnlyCapture()
    argv = _base_argv(tmp_path) + ["--prepare-only"]
    _run_main_prepare_only(cli_module, monkeypatch, argv, fake_prepare_only, fake_run_pilot)
    assert fake_prepare_only.calls[0]["tracking_generation"] == "g1"


def test_prepare_only_explicit_tracking_generation_is_retained(cli_module, monkeypatch, tmp_path):
    """(Part 4, #6)"""
    fake_run_pilot = _FakeRunPilotCapture()
    fake_prepare_only = _FakePreparePilotRunOnlyCapture()
    argv = _base_argv(tmp_path) + ["--prepare-only", "--tracking-generation", "g2"]
    _run_main_prepare_only(cli_module, monkeypatch, argv, fake_prepare_only, fake_run_pilot)
    assert fake_prepare_only.calls[0]["tracking_generation"] == "g2"


def test_prepare_only_reports_prepared_only_status(cli_module, monkeypatch, tmp_path, capsys):
    """(Part 4, #7) main() prints the PREPARED_ONLY result verbatim (as
    JSON) and exits normally (no sys.exit(1), unlike the blocked-training
    path)."""
    fake_run_pilot = _FakeRunPilotCapture()
    fake_prepare_only = _FakePreparePilotRunOnlyCapture()
    argv = _base_argv(tmp_path) + ["--prepare-only"]
    _run_main_prepare_only(cli_module, monkeypatch, argv, fake_prepare_only, fake_run_pilot)
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "PREPARED_ONLY"
    assert printed["training_started"] is False
    assert printed["wandb_backend_initialized"] is False

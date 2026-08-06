"""Focused tests for scripts/run_stage1_cap50k_closure.py -- the Seed-A 50k
embedding-structure closure CLI covering exactly the two approved candidates
(emb128x64_seedA_cap_low_cal incumbent, emb128x32_seedA_cap_low_cal
challenger).

Loads the script as a module and replaces its run_pilot/
prepare_pilot_run_only/discover_nh_run_dir/compute_pilot_status_fields
references with fakes that record kwargs -- these tests are about CLI
argument threading, allowlist enforcement, and mode routing, never about
exercising real training (mirrors the convention in
tests/test_run_stage1_lead06_pilot_cli.py).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.baseline.pilot_lead06_config import load_pilot_policy

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_stage1_cap50k_closure.py"
DEFAULT_PILOT_POLICY_PATH = REPO_ROOT / "config" / "stage1_lead06_pilot_v001.yaml"

INCUMBENT = "emb128x64_seedA_cap_low_cal"
CHALLENGER = "emb128x32_seedA_cap_low_cal"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "run_stage1_cap50k_closure_cli_under_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli_module():
    return _load_cli_module()


def _base_argv(tmp_path, run_id=INCUMBENT):
    return [
        "--run-id", run_id,
        "--package-root", str(tmp_path / "package"),
        "--config-out-dir", str(tmp_path / "config_out"),
        "--evidence-out-dir", str(tmp_path / "evidence"),
    ]


class _FakeRunPilotCapture:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "nh_run_dir": "/tmp/fake_nh_run_dir",
            "evidence_bundle_path": "/tmp/fake_evidence_bundle",
            "final_status": "completed",
        }


class _FakePreparePilotRunOnlyCapture:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "status": "PREPARED_ONLY",
            "run_id": kwargs["run_id"],
            "generated_config_path": str(Path(kwargs["config_out_dir"]) / "config.yaml"),
            "generation_manifest_path": str(Path(kwargs["config_out_dir"]) / "generation_manifest.json"),
            "wandb_policy_sha256": "fake_sha256",
            "tracking_generation": kwargs["tracking_generation"],
            "training_started": False,
            "evaluation_started": False,
            "wandb_backend_initialized": False,
        }


def _install_fakes(cli_module, monkeypatch, *, run_pilot=None, prepare_only=None,
                    discover_nh_run_dir=None, compute_pilot_status_fields=None):
    run_pilot = run_pilot if run_pilot is not None else _FakeRunPilotCapture()
    prepare_only = prepare_only if prepare_only is not None else _FakePreparePilotRunOnlyCapture()
    monkeypatch.setattr(cli_module, "run_pilot", run_pilot)
    monkeypatch.setattr(cli_module, "prepare_pilot_run_only", prepare_only)
    if discover_nh_run_dir is not None:
        monkeypatch.setattr(cli_module, "discover_nh_run_dir", discover_nh_run_dir)
    if compute_pilot_status_fields is not None:
        monkeypatch.setattr(cli_module, "compute_pilot_status_fields", compute_pilot_status_fields)
    return run_pilot, prepare_only


def _run_main(cli_module, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["run_stage1_cap50k_closure.py"] + argv)
    cli_module.main()


# --- (1) exact two-run allowlist --------------------------------------------

def test_closure_run_specs_is_exactly_the_two_approved_candidates(cli_module):
    assert set(cli_module.CLOSURE_RUN_SPECS) == {INCUMBENT, CHALLENGER}


def test_argparse_choices_are_built_from_the_two_entry_mapping(cli_module):
    # main() builds --run-id's argparse `choices` as sorted(CLOSURE_RUN_SPECS)
    # -- this is the single source of truth the allowlist tests above and
    # below both exercise indirectly through main(); pin its exact value here.
    assert sorted(cli_module.CLOSURE_RUN_SPECS) == [CHALLENGER, INCUMBENT]


def test_no_25k_or_out_of_scope_candidate_is_selectable(cli_module):
    forbidden = {
        "raw_seedA_cap_medium_cal", "raw_seedA_cap_low_cal",
        "emb128x64_seedA_cap_25k_cal", "emb128x32_seedA_cap_25k_cal",
        "emb128x64_seedA", "emb128x32_seedA",
    }
    assert forbidden.isdisjoint(cli_module.CLOSURE_RUN_SPECS)


# --- (2) correct run-ID-to-driver mapping -----------------------------------

def test_incumbent_run_spec_has_correct_embedding_shape_and_profile(cli_module):
    spec = cli_module.CLOSURE_RUN_SPECS[INCUMBENT]
    assert spec.embedding_hiddens == [128, 64]
    assert spec.run_profile_name == "pilot_lead06_emb128x64_seedA_v001"
    assert spec.seed_name == "seed_a"
    assert spec.seed == 967139
    assert spec.max_updates_per_epoch == 50_000
    assert spec.static_pathway == "learned_fc_embedding"


def test_challenger_run_spec_has_correct_embedding_shape_and_profile(cli_module):
    spec = cli_module.CLOSURE_RUN_SPECS[CHALLENGER]
    assert spec.embedding_hiddens == [128, 32]
    assert spec.run_profile_name == "pilot_lead06_emb128x32_seedA_v001"
    assert spec.seed_name == "seed_a"
    assert spec.seed == 967139
    assert spec.max_updates_per_epoch == 50_000
    assert spec.static_pathway == "learned_fc_embedding"


def test_incumbent_and_challenger_share_the_same_seed_and_cap_differ_only_in_embedding(cli_module):
    incumbent = cli_module.CLOSURE_RUN_SPECS[INCUMBENT]
    challenger = cli_module.CLOSURE_RUN_SPECS[CHALLENGER]
    assert incumbent.seed == challenger.seed
    assert incumbent.max_updates_per_epoch == challenger.max_updates_per_epoch
    assert incumbent.embedding_hiddens != challenger.embedding_hiddens


def test_build_closure_policy_splices_without_mutating_committed_policy(cli_module):
    real_policy = load_pilot_policy(DEFAULT_PILOT_POLICY_PATH)
    real_run_ids_before = set(real_policy.runs)
    closure_policy = cli_module._build_closure_policy(real_policy)
    assert INCUMBENT in closure_policy.runs
    assert CHALLENGER in closure_policy.runs
    # the real, loaded policy object itself is untouched
    assert set(real_policy.runs) == real_run_ids_before
    assert INCUMBENT not in real_policy.runs


def test_build_closure_policy_raises_on_collision_with_a_real_pilot_run_id(cli_module):
    real_policy = load_pilot_policy(DEFAULT_PILOT_POLICY_PATH)
    colliding_runs = dict(real_policy.runs)
    colliding_runs[INCUMBENT] = next(iter(real_policy.runs.values()))
    mutated = dataclasses.replace(real_policy, runs=colliding_runs)
    with pytest.raises(RuntimeError):
        cli_module._build_closure_policy(mutated)


# --- (3) rejection of unknown run_ids before any Python policy loading -----

def test_unknown_run_id_rejected_by_argparse_before_policy_loading(cli_module, monkeypatch, tmp_path):
    argv = _base_argv(tmp_path, run_id="emb128x64_seedA")
    monkeypatch.setattr(sys, "argv", ["run_stage1_cap50k_closure.py"] + argv)
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2


def test_unknown_run_id_never_reaches_run_pilot_or_prepare_only(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, fake_prepare_only = _install_fakes(cli_module, monkeypatch)
    argv = _base_argv(tmp_path, run_id="not_a_real_run_id")
    monkeypatch.setattr(sys, "argv", ["run_stage1_cap50k_closure.py"] + argv)
    with pytest.raises(SystemExit):
        cli_module.main()
    assert fake_run_pilot.calls == []
    assert fake_prepare_only.calls == []


# --- --force absence in routine continuation --------------------------------

def test_ordinary_invocation_defaults_force_to_false(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, _ = _install_fakes(cli_module, monkeypatch)
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path))
    assert fake_run_pilot.calls[0]["force"] is False


def test_explicit_force_flag_is_threaded_when_supplied(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, _ = _install_fakes(cli_module, monkeypatch)
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path) + ["--force"])
    assert fake_run_pilot.calls[0]["force"] is True


# --- W&B offline default / tracking_generation ------------------------------

def test_tracking_generation_defaults_to_g1_for_run_pilot(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, _ = _install_fakes(cli_module, monkeypatch)
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path))
    assert fake_run_pilot.calls[0]["tracking_generation"] == "g1"


def test_no_override_uses_committed_default_wandb_policy_path(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, _ = _install_fakes(cli_module, monkeypatch)
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path))
    expected_default = cli_module._resolve_policy_relative_paths(
        load_pilot_policy(DEFAULT_PILOT_POLICY_PATH)
    ).wandb_policy_path
    assert fake_run_pilot.calls[0]["pilot_policy"].wandb_policy_path == expected_default


# --- correct run-pilot vs prepare-only vs status-only routing ---------------

def test_ordinary_invocation_calls_only_run_pilot(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, fake_prepare_only = _install_fakes(cli_module, monkeypatch)
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path))
    assert len(fake_run_pilot.calls) == 1
    assert fake_prepare_only.calls == []


def test_prepare_only_calls_only_prepare_pilot_run_only(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, fake_prepare_only = _install_fakes(cli_module, monkeypatch)
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path, run_id=CHALLENGER) + ["--prepare-only"])
    assert len(fake_prepare_only.calls) == 1
    assert fake_run_pilot.calls == []
    assert fake_prepare_only.calls[0]["run_id"] == CHALLENGER


def test_prepare_only_and_status_only_are_mutually_exclusive(cli_module, monkeypatch, tmp_path):
    _install_fakes(cli_module, monkeypatch)
    argv = _base_argv(tmp_path) + ["--prepare-only", "--status-only"]
    monkeypatch.setattr(sys, "argv", ["run_stage1_cap50k_closure.py"] + argv)
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2


# --- --status-only: read-only, never generates config or calls NH ----------

def test_status_only_never_requires_package_root_or_evidence_out_dir(cli_module, monkeypatch, tmp_path, capsys):
    fake_run_pilot, fake_prepare_only = _install_fakes(
        cli_module, monkeypatch,
        discover_nh_run_dir=lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no run dir")),
    )
    argv = ["--run-id", INCUMBENT, "--config-out-dir", str(tmp_path / "config_out"), "--status-only"]
    _run_main(cli_module, monkeypatch, argv)
    assert fake_run_pilot.calls == []
    assert fake_prepare_only.calls == []
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "NO_EXISTING_NH_RUN_DIRECTORY"


def test_status_only_reports_existing_run_dir_when_discovered(cli_module, monkeypatch, tmp_path, capsys):
    fake_status_fields = {
        "highest_physical_checkpoint_epoch": 6,
        "highest_screened_epoch": 6,
        "next_intended_screening_epoch": 9,
        "overshoot_epochs": [],
        "safe_to_continue_automatically": True,
    }
    fake_run_pilot, fake_prepare_only = _install_fakes(
        cli_module, monkeypatch,
        discover_nh_run_dir=lambda *a, **k: tmp_path / "nh_run",
        compute_pilot_status_fields=lambda *a, **k: dict(fake_status_fields),
    )
    argv = ["--run-id", INCUMBENT, "--config-out-dir", str(tmp_path / "config_out"), "--status-only"]
    _run_main(cli_module, monkeypatch, argv)
    assert fake_run_pilot.calls == []
    assert fake_prepare_only.calls == []
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "EXISTING_NH_RUN_DIRECTORY_FOUND"
    assert printed["safe_to_continue_automatically"] is True
    assert printed["next_intended_screening_epoch"] == 9


def test_status_only_never_writes_a_config_or_evidence_file(cli_module, monkeypatch, tmp_path):
    _install_fakes(
        cli_module, monkeypatch,
        discover_nh_run_dir=lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no run dir")),
    )
    config_out_dir = tmp_path / "config_out"
    argv = ["--run-id", CHALLENGER, "--config-out-dir", str(config_out_dir), "--status-only"]
    _run_main(cli_module, monkeypatch, argv)
    assert not config_out_dir.exists()


# --- ordinary invocation still requires package-root / evidence-out-dir ----

def test_ordinary_invocation_without_package_root_fails_before_run_pilot(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, _ = _install_fakes(cli_module, monkeypatch)
    argv = [
        "--run-id", INCUMBENT,
        "--config-out-dir", str(tmp_path / "config_out"),
        "--evidence-out-dir", str(tmp_path / "evidence"),
    ]
    monkeypatch.setattr(sys, "argv", ["run_stage1_cap50k_closure.py"] + argv)
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2
    assert fake_run_pilot.calls == []


# --- epoch-12 closure horizon: fixed, non-overridable -----------------------

def test_closure_max_target_epoch_constant_is_12(cli_module):
    assert cli_module.CLOSURE_MAX_TARGET_EPOCH == 12


def test_no_max_target_epoch_cli_flag_exists(cli_module, monkeypatch, tmp_path):
    # A caller must not be able to name any other target epoch -- the flag
    # itself is gone, so argparse rejects it as an unrecognized argument.
    argv = _base_argv(tmp_path) + ["--max-target-epoch", "36"]
    monkeypatch.setattr(sys, "argv", ["run_stage1_cap50k_closure.py"] + argv)
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2


def test_ordinary_invocation_always_passes_fixed_max_target_epoch_to_run_pilot(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, _ = _install_fakes(cli_module, monkeypatch)
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path))
    assert fake_run_pilot.calls[0]["max_target_epoch"] == 12


def test_ordinary_invocation_for_challenger_also_gets_fixed_max_target_epoch(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, _ = _install_fakes(cli_module, monkeypatch)
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path, run_id=CHALLENGER))
    assert fake_run_pilot.calls[0]["max_target_epoch"] == 12


def test_ordinary_invocation_prints_closure_max_target_epoch(cli_module, monkeypatch, tmp_path, capsys):
    _install_fakes(cli_module, monkeypatch)
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path))
    printed = json.loads(capsys.readouterr().out)
    assert printed["closure_max_target_epoch"] == 12


def test_prepare_only_output_reports_closure_max_target_epoch(cli_module, monkeypatch, tmp_path, capsys):
    _install_fakes(cli_module, monkeypatch)
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path, run_id=CHALLENGER) + ["--prepare-only"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["closure_max_target_epoch"] == 12


def test_status_only_output_reports_closure_max_target_epoch(cli_module, monkeypatch, tmp_path, capsys):
    _install_fakes(
        cli_module, monkeypatch,
        discover_nh_run_dir=lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no run dir")),
    )
    argv = ["--run-id", INCUMBENT, "--config-out-dir", str(tmp_path / "config_out"), "--status-only"]
    _run_main(cli_module, monkeypatch, argv)
    printed = json.loads(capsys.readouterr().out)
    assert printed["closure_max_target_epoch"] == 12

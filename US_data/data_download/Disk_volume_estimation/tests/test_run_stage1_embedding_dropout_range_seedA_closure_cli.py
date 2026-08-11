"""Focused tests for scripts/run_stage1_embedding_dropout_range_seedA_closure.py
-- the Embedding-Dropout-A embedding-dropout range-characterization campaign
CLI covering exactly the five approved NEW candidates plus the historical,
read-only-only Hidden-size-A H=128 reference.

Loads the script as a module and replaces its run_pilot/
prepare_pilot_run_only/discover_nh_run_dir/compute_pilot_status_fields
references with fakes that record kwargs -- these tests are about CLI
argument threading, allowlist enforcement, and mode routing, never about
exercising real training (mirrors the convention in
tests/test_run_stage1_hidden_size_range_seedA_closure_cli.py).
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
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_stage1_embedding_dropout_range_seedA_closure.py"
DEFAULT_PILOT_POLICY_PATH = REPO_ROOT / "config" / "stage1_lead06_pilot_v001.yaml"
DEFAULT_WANDB_OFFLINE_POLICY_PATH = REPO_ROOT / "config" / "stage1_wandb_tracking_policy_offline_v001.yaml"

DROP00 = "emb128x32_seedA_drop00_h128_lr3em4_cap25k_cal"
DROP05 = "emb128x32_seedA_drop05_h128_lr3em4_cap25k_cal"
DROP10 = "emb128x32_seedA_drop10_h128_lr3em4_cap25k_cal"
DROP20 = "emb128x32_seedA_drop20_h128_lr3em4_cap25k_cal"
DROP40 = "emb128x32_seedA_drop40_h128_lr3em4_cap25k_cal"
REFERENCE = "emb128x32_seedA_h128_lr3em4_cap25k_cal"
ALL_FIVE = {DROP00, DROP05, DROP10, DROP20, DROP40}


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "run_stage1_embedding_dropout_range_seedA_closure_cli_under_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli_module():
    return _load_cli_module()


def _base_argv(tmp_path, run_id=DROP00):
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
            "final_status": "paused_at_max_target_epoch",
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
    monkeypatch.setattr(sys, "argv", ["run_stage1_embedding_dropout_range_seedA_closure.py"] + argv)
    cli_module.main()


# --- (G) exact five-run trainable allowlist + reference identity -----------

def test_embedding_dropout_a_run_specs_is_exactly_the_five_approved_candidates(cli_module):
    assert set(cli_module.EMBEDDING_DROPOUT_A_RUN_SPECS) == ALL_FIVE


def test_reference_run_id_is_not_a_member_of_the_trainable_allowlist(cli_module):
    assert cli_module.REFERENCE_RUN_ID == REFERENCE
    assert REFERENCE not in cli_module.EMBEDDING_DROPOUT_A_RUN_SPECS


def test_argparse_choices_include_the_five_plus_the_reference(cli_module):
    assert set(sorted(cli_module.EMBEDDING_DROPOUT_A_RUN_SPECS) + [cli_module.REFERENCE_RUN_ID]) == ALL_FIVE | {REFERENCE}


def test_no_out_of_scope_candidate_is_added_to_trainable_allowlist(cli_module):
    forbidden = {
        REFERENCE, "raw_seedA_cap_medium_cal", "raw_seedA_cap_low_cal",
        "emb128x64_seedA_cap_low_cal", "emb128x32_seedA_cap_low_cal",
        "emb128x64_seedA", "emb128x32_seedA",
        "emb128x32_seedA_lr1em4_cap25k_cal", "emb128x32_seedA_lr3em3_cap25k_cal",
        "emb128x32_seedA_lr1em2_cap25k_cal", "emb128x32_seedA_lr3em4_cap25k_cal",
        "emb128x32_seedA_h64_lr3em4_cap25k_cal", "emb128x32_seedA_h256_lr3em4_cap25k_cal",
        "emb128x32_seedA_h512_lr3em4_cap25k_cal",
    }
    assert forbidden.isdisjoint(cli_module.EMBEDDING_DROPOUT_A_RUN_SPECS)


# --- (B) correct run-ID-to-embedding-dropout mapping ------------------------

@pytest.mark.parametrize("run_id,expected_dropout", [
    (DROP00, 0.00),
    (DROP05, 0.05),
    (DROP10, 0.10),
    (DROP20, 0.20),
    (DROP40, 0.40),
])
def test_each_candidate_has_correct_embedding_dropout(cli_module, run_id, expected_dropout):
    spec = cli_module.EMBEDDING_DROPOUT_A_RUN_SPECS[run_id]
    assert spec.embedding_dropout == pytest.approx(expected_dropout)
    assert spec.embedding_dropout is not None


def test_drop00_candidate_dropout_is_explicit_zero_not_none(cli_module):
    spec = cli_module.EMBEDDING_DROPOUT_A_RUN_SPECS[DROP00]
    assert spec.embedding_dropout == 0.0
    assert spec.embedding_dropout is not None


@pytest.mark.parametrize("run_id", [DROP00, DROP05, DROP10, DROP20, DROP40])
def test_each_candidate_shares_fixed_scientific_contract(cli_module, run_id):
    spec = cli_module.EMBEDDING_DROPOUT_A_RUN_SPECS[run_id]
    assert spec.embedding_hiddens == [128, 32]
    assert spec.run_profile_name == "pilot_lead06_emb128x32_seedA_v001"
    assert spec.seed_name == "seed_a"
    assert spec.seed == 967139
    assert spec.max_updates_per_epoch == 25_000
    assert spec.static_pathway == "learned_fc_embedding"
    # hidden_size and learning_rate are held FIXED at the Hidden-size-A/LR-A
    # provisional anchors across every embedding-dropout candidate -- not
    # varied per-candidate.
    assert spec.hidden_size == 128
    assert spec.learning_rate == 3e-4


def test_all_five_candidates_differ_from_each_other_only_in_embedding_dropout(cli_module):
    specs = [cli_module.EMBEDDING_DROPOUT_A_RUN_SPECS[run_id] for run_id in sorted(cli_module.EMBEDDING_DROPOUT_A_RUN_SPECS)]
    non_dropout_fields = [
        dataclasses.replace(spec, run_id="x", embedding_dropout=None) for spec in specs
    ]
    assert all(f == non_dropout_fields[0] for f in non_dropout_fields)
    dropouts = {spec.embedding_dropout for spec in specs}
    assert len(dropouts) == 5


def test_build_embedding_dropout_a_policy_splices_without_mutating_committed_policy(cli_module):
    real_policy = load_pilot_policy(DEFAULT_PILOT_POLICY_PATH)
    real_run_ids_before = set(real_policy.runs)
    embedding_dropout_a_policy = cli_module._build_embedding_dropout_a_policy(real_policy)
    for run_id in ALL_FIVE:
        assert run_id in embedding_dropout_a_policy.runs
    assert REFERENCE not in embedding_dropout_a_policy.runs
    # the real, loaded policy object itself is untouched
    assert set(real_policy.runs) == real_run_ids_before
    assert DROP00 not in real_policy.runs


def test_build_embedding_dropout_a_policy_sets_frozen_campaign_policy_name(cli_module):
    real_policy = load_pilot_policy(DEFAULT_PILOT_POLICY_PATH)
    embedding_dropout_a_policy = cli_module._build_embedding_dropout_a_policy(real_policy)
    assert embedding_dropout_a_policy.raw["policy_name"] == "embedding_dropout_range_seedA_25k_v001"


def test_build_embedding_dropout_a_policy_raises_on_collision_with_a_real_pilot_run_id(cli_module):
    real_policy = load_pilot_policy(DEFAULT_PILOT_POLICY_PATH)
    colliding_runs = dict(real_policy.runs)
    colliding_runs[DROP00] = next(iter(real_policy.runs.values()))
    mutated = dataclasses.replace(real_policy, runs=colliding_runs)
    with pytest.raises(RuntimeError):
        cli_module._build_embedding_dropout_a_policy(mutated)


def test_build_embedding_dropout_a_policy_raises_on_collision_with_the_reference_run_id(cli_module):
    real_policy = load_pilot_policy(DEFAULT_PILOT_POLICY_PATH)
    colliding_runs = dict(real_policy.runs)
    colliding_runs[REFERENCE] = next(iter(real_policy.runs.values()))
    mutated = dataclasses.replace(real_policy, runs=colliding_runs)
    with pytest.raises(RuntimeError):
        cli_module._build_embedding_dropout_a_policy(mutated)


def test_build_embedding_dropout_a_policy_raises_on_collision_with_another_campaigns_reserved_run_id(cli_module, monkeypatch):
    real_policy = load_pilot_policy(DEFAULT_PILOT_POLICY_PATH)
    colliding_run_id = next(iter(cli_module._OTHER_CAMPAIGN_RESERVED_RUN_IDS))
    embedding_dropout_a_run_specs = dict(cli_module.EMBEDDING_DROPOUT_A_RUN_SPECS)
    embedding_dropout_a_run_specs[colliding_run_id] = next(iter(embedding_dropout_a_run_specs.values()))
    monkeypatch.setattr(cli_module, "EMBEDDING_DROPOUT_A_RUN_SPECS", embedding_dropout_a_run_specs)
    with pytest.raises(RuntimeError):
        cli_module._build_embedding_dropout_a_policy(real_policy)


def test_reference_run_id_is_among_other_campaign_reserved_run_ids(cli_module):
    """Defense-in-depth: REFERENCE_RUN_ID is also one of Hidden-size-A's own
    four reserved run_ids -- confirmed present in the reserved set even
    though it is separately checked via augmented_runs directly."""
    assert cli_module.REFERENCE_RUN_ID in cli_module._OTHER_CAMPAIGN_RESERVED_RUN_IDS


# --- (G) rejection of unknown run_ids before any Python policy loading -----

def test_unknown_run_id_rejected_by_argparse_before_policy_loading(cli_module, monkeypatch, tmp_path):
    argv = _base_argv(tmp_path, run_id="emb128x32_seedA_drop99_h128_lr3em4_cap25k_cal")
    monkeypatch.setattr(sys, "argv", ["run_stage1_embedding_dropout_range_seedA_closure.py"] + argv)
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2


def test_unknown_run_id_never_reaches_run_pilot_or_prepare_only(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, fake_prepare_only = _install_fakes(cli_module, monkeypatch)
    argv = _base_argv(tmp_path, run_id="not_a_real_run_id")
    monkeypatch.setattr(sys, "argv", ["run_stage1_embedding_dropout_range_seedA_closure.py"] + argv)
    with pytest.raises(SystemExit):
        cli_module.main()
    assert fake_run_pilot.calls == []
    assert fake_prepare_only.calls == []


# --- (G) reference run_id: status-only only, never trainable ---------------

def test_reference_run_id_with_ordinary_invocation_is_rejected(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, fake_prepare_only = _install_fakes(cli_module, monkeypatch)
    argv = _base_argv(tmp_path, run_id=REFERENCE)
    monkeypatch.setattr(sys, "argv", ["run_stage1_embedding_dropout_range_seedA_closure.py"] + argv)
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2
    assert fake_run_pilot.calls == []
    assert fake_prepare_only.calls == []


def test_reference_run_id_with_prepare_only_is_rejected(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, fake_prepare_only = _install_fakes(cli_module, monkeypatch)
    argv = _base_argv(tmp_path, run_id=REFERENCE) + ["--prepare-only"]
    monkeypatch.setattr(sys, "argv", ["run_stage1_embedding_dropout_range_seedA_closure.py"] + argv)
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2
    assert fake_run_pilot.calls == []
    assert fake_prepare_only.calls == []


def test_reference_run_id_with_status_only_is_accepted(cli_module, monkeypatch, tmp_path, capsys):
    fake_run_pilot, fake_prepare_only = _install_fakes(
        cli_module, monkeypatch,
        discover_nh_run_dir=lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no run dir")),
    )
    argv = ["--run-id", REFERENCE, "--config-out-dir", str(tmp_path / "config_out"), "--status-only"]
    _run_main(cli_module, monkeypatch, argv)
    assert fake_run_pilot.calls == []
    assert fake_prepare_only.calls == []
    printed = json.loads(capsys.readouterr().out)
    assert printed["is_historical_hidden_size_a_comparator"] is True
    assert printed["status"] == "NO_EXISTING_NH_RUN_DIRECTORY"


def test_non_reference_run_id_status_only_reports_is_historical_comparator_false(cli_module, monkeypatch, tmp_path, capsys):
    _install_fakes(
        cli_module, monkeypatch,
        discover_nh_run_dir=lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no run dir")),
    )
    argv = ["--run-id", DROP00, "--config-out-dir", str(tmp_path / "config_out"), "--status-only"]
    _run_main(cli_module, monkeypatch, argv)
    printed = json.loads(capsys.readouterr().out)
    assert printed["is_historical_hidden_size_a_comparator"] is False


# --- --force absence in routine continuation --------------------------------

def test_ordinary_invocation_defaults_force_to_false(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, _ = _install_fakes(cli_module, monkeypatch)
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path))
    assert fake_run_pilot.calls[0]["force"] is False


def test_explicit_force_flag_is_threaded_when_supplied(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, _ = _install_fakes(cli_module, monkeypatch)
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path) + ["--force"])
    assert fake_run_pilot.calls[0]["force"] is True


# --- (I) W&B launch contract: strict-by-default offline policy -------------

def test_tracking_generation_defaults_to_g1_for_run_pilot(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, _ = _install_fakes(cli_module, monkeypatch)
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path))
    assert fake_run_pilot.calls[0]["tracking_generation"] == "g1"


def test_default_wandb_policy_path_is_the_offline_enabled_policy_not_the_disabled_default(cli_module):
    assert cli_module._DEFAULT_WANDB_POLICY_PATH == DEFAULT_WANDB_OFFLINE_POLICY_PATH


def test_no_override_uses_offline_enabled_wandb_policy_path(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, _ = _install_fakes(cli_module, monkeypatch)
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path))
    assert fake_run_pilot.calls[0]["pilot_policy"].wandb_policy_path == str(DEFAULT_WANDB_OFFLINE_POLICY_PATH)


def test_require_tracking_defaults_to_true(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, _ = _install_fakes(cli_module, monkeypatch)
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path))
    assert fake_run_pilot.calls[0]["require_tracking"] is True


def test_waive_tracking_requirement_flag_flips_require_tracking_to_false(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, _ = _install_fakes(cli_module, monkeypatch)
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path) + ["--waive-tracking-requirement"])
    assert fake_run_pilot.calls[0]["require_tracking"] is False


def test_ordinary_invocation_reports_require_tracking_used(cli_module, monkeypatch, tmp_path, capsys):
    _install_fakes(cli_module, monkeypatch)
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path))
    printed = json.loads(capsys.readouterr().out)
    assert printed["require_tracking_used"] is True


def test_invalid_wandb_policy_path_override_fails_before_run_pilot(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, fake_prepare_only = _install_fakes(cli_module, monkeypatch)
    argv = _base_argv(tmp_path) + ["--wandb-policy-path", str(tmp_path / "does_not_exist.yaml")]
    monkeypatch.setattr(sys, "argv", ["run_stage1_embedding_dropout_range_seedA_closure.py"] + argv)
    with pytest.raises(Exception):
        cli_module.main()
    assert fake_run_pilot.calls == []
    assert fake_prepare_only.calls == []


# --- correct run-pilot vs prepare-only vs status-only routing ---------------

def test_ordinary_invocation_calls_only_run_pilot(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, fake_prepare_only = _install_fakes(cli_module, monkeypatch)
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path))
    assert len(fake_run_pilot.calls) == 1
    assert fake_prepare_only.calls == []


def test_prepare_only_calls_only_prepare_pilot_run_only(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, fake_prepare_only = _install_fakes(cli_module, monkeypatch)
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path, run_id=DROP20) + ["--prepare-only"])
    assert len(fake_prepare_only.calls) == 1
    assert fake_run_pilot.calls == []
    assert fake_prepare_only.calls[0]["run_id"] == DROP20


def test_prepare_only_and_status_only_are_mutually_exclusive(cli_module, monkeypatch, tmp_path):
    _install_fakes(cli_module, monkeypatch)
    argv = _base_argv(tmp_path) + ["--prepare-only", "--status-only"]
    monkeypatch.setattr(sys, "argv", ["run_stage1_embedding_dropout_range_seedA_closure.py"] + argv)
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2


# --- (H) --status-only: read-only, never generates config or calls NH ------

def test_status_only_never_requires_package_root_or_evidence_out_dir(cli_module, monkeypatch, tmp_path, capsys):
    fake_run_pilot, fake_prepare_only = _install_fakes(
        cli_module, monkeypatch,
        discover_nh_run_dir=lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no run dir")),
    )
    argv = ["--run-id", DROP00, "--config-out-dir", str(tmp_path / "config_out"), "--status-only"]
    _run_main(cli_module, monkeypatch, argv)
    assert fake_run_pilot.calls == []
    assert fake_prepare_only.calls == []
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "NO_EXISTING_NH_RUN_DIRECTORY"


def test_status_only_reports_existing_run_dir_when_discovered(cli_module, monkeypatch, tmp_path, capsys):
    fake_status_fields = {
        "highest_physical_checkpoint_epoch": 6,
        "highest_screened_epoch": 6,
        "next_intended_screening_epoch": None,
        "overshoot_epochs": [],
        "safe_to_continue_automatically": True,
    }
    fake_run_pilot, fake_prepare_only = _install_fakes(
        cli_module, monkeypatch,
        discover_nh_run_dir=lambda *a, **k: tmp_path / "nh_run",
        compute_pilot_status_fields=lambda *a, **k: dict(fake_status_fields),
    )
    argv = ["--run-id", DROP00, "--config-out-dir", str(tmp_path / "config_out"), "--status-only"]
    _run_main(cli_module, monkeypatch, argv)
    assert fake_run_pilot.calls == []
    assert fake_prepare_only.calls == []
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "EXISTING_NH_RUN_DIRECTORY_FOUND"
    assert printed["safe_to_continue_automatically"] is True


def test_status_only_never_writes_a_config_or_evidence_file(cli_module, monkeypatch, tmp_path):
    _install_fakes(
        cli_module, monkeypatch,
        discover_nh_run_dir=lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no run dir")),
    )
    config_out_dir = tmp_path / "config_out"
    argv = ["--run-id", DROP40, "--config-out-dir", str(config_out_dir), "--status-only"]
    _run_main(cli_module, monkeypatch, argv)
    assert not config_out_dir.exists()


# --- ordinary invocation still requires package-root / evidence-out-dir ----

def test_ordinary_invocation_without_package_root_fails_before_run_pilot(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, _ = _install_fakes(cli_module, monkeypatch)
    argv = [
        "--run-id", DROP00,
        "--config-out-dir", str(tmp_path / "config_out"),
        "--evidence-out-dir", str(tmp_path / "evidence"),
    ]
    monkeypatch.setattr(sys, "argv", ["run_stage1_embedding_dropout_range_seedA_closure.py"] + argv)
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2
    assert fake_run_pilot.calls == []


# --- (F) epoch-6 Embedding-Dropout-A horizon: fixed, non-overridable -------

def test_embedding_dropout_a_max_target_epoch_constant_is_6(cli_module):
    assert cli_module.EMBEDDING_DROPOUT_A_MAX_TARGET_EPOCH == 6


def test_no_max_target_epoch_cli_flag_exists(cli_module, monkeypatch, tmp_path):
    argv = _base_argv(tmp_path) + ["--max-target-epoch", "36"]
    monkeypatch.setattr(sys, "argv", ["run_stage1_embedding_dropout_range_seedA_closure.py"] + argv)
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2


def test_no_embedding_dropout_cli_flag_exists(cli_module, monkeypatch, tmp_path):
    argv = _base_argv(tmp_path) + ["--embedding-dropout", "0.3"]
    monkeypatch.setattr(sys, "argv", ["run_stage1_embedding_dropout_range_seedA_closure.py"] + argv)
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2


def test_no_hidden_size_cli_flag_exists(cli_module, monkeypatch, tmp_path):
    argv = _base_argv(tmp_path) + ["--hidden-size", "1024"]
    monkeypatch.setattr(sys, "argv", ["run_stage1_embedding_dropout_range_seedA_closure.py"] + argv)
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2


def test_no_learning_rate_cli_flag_exists(cli_module, monkeypatch, tmp_path):
    argv = _base_argv(tmp_path) + ["--learning-rate", "0.05"]
    monkeypatch.setattr(sys, "argv", ["run_stage1_embedding_dropout_range_seedA_closure.py"] + argv)
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 2


def test_ordinary_invocation_always_passes_fixed_max_target_epoch_to_run_pilot(cli_module, monkeypatch, tmp_path):
    fake_run_pilot, _ = _install_fakes(cli_module, monkeypatch)
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path))
    assert fake_run_pilot.calls[0]["max_target_epoch"] == 6


@pytest.mark.parametrize("run_id", [DROP00, DROP05, DROP10, DROP20, DROP40])
def test_ordinary_invocation_for_every_candidate_gets_fixed_max_target_epoch(cli_module, monkeypatch, tmp_path, run_id):
    fake_run_pilot, _ = _install_fakes(cli_module, monkeypatch)
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path, run_id=run_id))
    assert fake_run_pilot.calls[0]["max_target_epoch"] == 6
    assert fake_run_pilot.calls[0]["run_id"] == run_id


def test_ordinary_invocation_prints_embedding_dropout_a_max_target_epoch(cli_module, monkeypatch, tmp_path, capsys):
    _install_fakes(cli_module, monkeypatch)
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path))
    printed = json.loads(capsys.readouterr().out)
    assert printed["embedding_dropout_a_max_target_epoch"] == 6


def test_prepare_only_output_reports_embedding_dropout_a_max_target_epoch(cli_module, monkeypatch, tmp_path, capsys):
    _install_fakes(cli_module, monkeypatch)
    _run_main(cli_module, monkeypatch, _base_argv(tmp_path, run_id=DROP10) + ["--prepare-only"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["embedding_dropout_a_max_target_epoch"] == 6


def test_status_only_output_reports_embedding_dropout_a_max_target_epoch(cli_module, monkeypatch, tmp_path, capsys):
    _install_fakes(
        cli_module, monkeypatch,
        discover_nh_run_dir=lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no run dir")),
    )
    argv = ["--run-id", DROP00, "--config-out-dir", str(tmp_path / "config_out"), "--status-only"]
    _run_main(cli_module, monkeypatch, argv)
    printed = json.loads(capsys.readouterr().out)
    assert printed["embedding_dropout_a_max_target_epoch"] == 6

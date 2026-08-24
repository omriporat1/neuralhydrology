"""Tests for the Sweep-v1 launch-command repair.

Closes the gap the launch-readiness review found: the production W&B sweep
config had no explicit ``command``, so W&B's default ``${args}`` behavior
would have appended the five swept hyperparameters as CLI flags that
``scripts/run_sweep_v1_wandb_bridge.py``'s argparse does not accept, and no
mechanism existed for the sbatch launcher to hand the bridge its four
operational inputs once a real W&B agent -- not the launcher -- constructs
the bridge's argv directly from that command template.

Covers: production sweep-config shape (explicit ``command``, no ``${args}``);
real config-builder serialization and no-overwrite safety; the bridge's
strict CLI/environment operational-input resolution; the OS-level
agent-to-bridge argv contract (the "crucial missing test" -- a real
subprocess run of the exact argv a production ``wandb agent`` will
construct); and the sbatch launcher's static resource/env/concurrency
contract. Never imports wandb, starts an agent, creates a sweep, or runs
training -- every subprocess here uses
``FLASHNH_SWEEP_V1_BRIDGE_SELFTEST=resolve_only``, which returns before the
bridge's own ``import wandb``.

Companion to tests/test_sweep_v1_wandb_bridge_provenance.py (CLI-only golden
bridge runs and provenance progression, unaffected by this repair -- none of
its fixtures set any ``FLASHNH_SWEEP_V1_*`` environment variable, so the new
optional-CLI-flag resolution there still resolves purely from CLI, exactly
as before).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.run_sweep_v1_wandb_bridge as bridge
from scripts.build_sweep_v1_production_sweep_config import main as build_config_main
from src.baseline.sweep_v1_execution import build_production_sweep_config

ROOT = Path(__file__).resolve().parents[1]
BRIDGE_SCRIPT = ROOT / "scripts" / "run_sweep_v1_wandb_bridge.py"
BUILD_CONFIG_SCRIPT = ROOT / "scripts" / "build_sweep_v1_production_sweep_config.py"
SBATCH_SCRIPT = ROOT / "scripts" / "run_sweep_v1_wandb_agent_moriah.sbatch"


# --- 1. production sweep-config shape ---------------------------------------

def test_production_sweep_config_has_explicit_command_and_no_args_macro():
    config = build_production_sweep_config(program="scripts/run_sweep_v1_wandb_bridge.py")
    assert config["command"] == ["${interpreter}", "${program}"]
    serialized = json.dumps(config["command"])
    assert "${args}" not in serialized
    assert "${env}" not in serialized
    assert config["method"] == "bayes"
    assert config["metric"] == {"name": "flashnh/best_score", "goal": "maximize"}
    assert config["program"] == "scripts/run_sweep_v1_wandb_bridge.py"
    assert config["parameters"] == {
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-3},
        "hidden_size": {"values": [64, 128, 256]},
        "embedding_dropout": {"distribution": "uniform", "min": 0.0, "max": 0.4},
        "output_dropout": {"distribution": "uniform", "min": 0.0, "max": 0.4},
        "batch_size": {"values": [128, 256, 512]},
    }


# --- 2. real serialization + no-overwrite safety ----------------------------

def test_builder_script_serializes_real_config_with_exact_command(tmp_path, monkeypatch):
    output = tmp_path / "sweep_config.json"
    monkeypatch.setattr(sys, "argv", ["build_sweep_v1_production_sweep_config.py", "--output", str(output)])
    build_config_main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["command"] == ["${interpreter}", "${program}"]
    assert payload["program"] == "scripts/run_sweep_v1_wandb_bridge.py"


def test_builder_script_refuses_to_silently_overwrite(tmp_path, monkeypatch):
    output = tmp_path / "sweep_config.json"
    monkeypatch.setattr(sys, "argv",
                        ["build_sweep_v1_production_sweep_config.py", "--output", str(output), "--program", "first"])
    build_config_main()
    first_content = output.read_text(encoding="utf-8")

    monkeypatch.setattr(sys, "argv",
                        ["build_sweep_v1_production_sweep_config.py", "--output", str(output), "--program", "second"])
    with pytest.raises(SystemExit):
        build_config_main()
    assert output.read_text(encoding="utf-8") == first_content


def test_builder_script_force_flag_permits_deliberate_overwrite(tmp_path, monkeypatch):
    output = tmp_path / "sweep_config.json"
    monkeypatch.setattr(sys, "argv",
                        ["build_sweep_v1_production_sweep_config.py", "--output", str(output), "--program", "first"])
    build_config_main()

    monkeypatch.setattr(sys, "argv",
                        ["build_sweep_v1_production_sweep_config.py", "--output", str(output),
                         "--program", "second", "--force"])
    build_config_main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["program"] == "second"


# --- 3. bridge CLI/environment operational-input resolver precedence -------

def test_resolve_path_input_cli_only():
    resolved = bridge._resolve_path_operational_input(
        flag="--package-root", cli_value=Path("/cli/root"), env_name="FLASHNH_SWEEP_V1_TEST_UNSET_A")
    assert resolved == Path("/cli/root")


def test_resolve_path_input_env_only(monkeypatch):
    monkeypatch.setenv("FLASHNH_SWEEP_V1_TEST_PATH", "/env/root")
    resolved = bridge._resolve_path_operational_input(
        flag="--package-root", cli_value=None, env_name="FLASHNH_SWEEP_V1_TEST_PATH")
    assert resolved == Path("/env/root")


def test_resolve_path_input_identical_cli_and_env(monkeypatch):
    monkeypatch.setenv("FLASHNH_SWEEP_V1_TEST_PATH", "/same/root")
    resolved = bridge._resolve_path_operational_input(
        flag="--package-root", cli_value=Path("/same/root"), env_name="FLASHNH_SWEEP_V1_TEST_PATH")
    assert resolved == Path("/same/root")


def test_resolve_path_input_contradiction_hard_fails(monkeypatch):
    monkeypatch.setenv("FLASHNH_SWEEP_V1_TEST_PATH", "/env/root")
    with pytest.raises(SystemExit):
        bridge._resolve_path_operational_input(
            flag="--package-root", cli_value=Path("/cli/root"), env_name="FLASHNH_SWEEP_V1_TEST_PATH")


def test_resolve_path_input_neither_supplied_hard_fails(monkeypatch):
    monkeypatch.delenv("FLASHNH_SWEEP_V1_TEST_UNSET_B", raising=False)
    with pytest.raises(SystemExit):
        bridge._resolve_path_operational_input(
            flag="--package-root", cli_value=None, env_name="FLASHNH_SWEEP_V1_TEST_UNSET_B")


def test_resolve_proposal_order_cli_only():
    assert bridge._resolve_proposal_order(cli_value=3, env_name="FLASHNH_SWEEP_V1_TEST_UNSET_C") == 3


def test_resolve_proposal_order_env_only(monkeypatch):
    monkeypatch.setenv("FLASHNH_SWEEP_V1_TEST_ORDER", "5")
    assert bridge._resolve_proposal_order(cli_value=None, env_name="FLASHNH_SWEEP_V1_TEST_ORDER") == 5


def test_resolve_proposal_order_identical_cli_and_env(monkeypatch):
    monkeypatch.setenv("FLASHNH_SWEEP_V1_TEST_ORDER", "4")
    assert bridge._resolve_proposal_order(cli_value=4, env_name="FLASHNH_SWEEP_V1_TEST_ORDER") == 4


def test_resolve_proposal_order_contradiction_hard_fails(monkeypatch):
    monkeypatch.setenv("FLASHNH_SWEEP_V1_TEST_ORDER", "4")
    with pytest.raises(SystemExit):
        bridge._resolve_proposal_order(cli_value=5, env_name="FLASHNH_SWEEP_V1_TEST_ORDER")


def test_resolve_proposal_order_neither_supplied_hard_fails(monkeypatch):
    monkeypatch.delenv("FLASHNH_SWEEP_V1_TEST_UNSET_D", raising=False)
    with pytest.raises(SystemExit):
        bridge._resolve_proposal_order(cli_value=None, env_name="FLASHNH_SWEEP_V1_TEST_UNSET_D")


def test_resolve_proposal_order_malformed_env_hard_fails(monkeypatch):
    monkeypatch.setenv("FLASHNH_SWEEP_V1_TEST_ORDER", "not-an-int")
    with pytest.raises(SystemExit):
        bridge._resolve_proposal_order(cli_value=None, env_name="FLASHNH_SWEEP_V1_TEST_ORDER")


def test_resolve_proposal_order_non_positive_hard_fails():
    with pytest.raises(SystemExit):
        bridge._resolve_proposal_order(cli_value=0, env_name="FLASHNH_SWEEP_V1_TEST_UNSET_E")


# --- 4. the crucial real OS-level agent-to-bridge argv contract ------------

def _expand_command(command, *, interpreter: str, program: str) -> list:
    return [interpreter if token == "${interpreter}" else program if token == "${program}" else token
            for token in command]


def _subprocess_env(overrides: dict) -> dict:
    env = dict(os.environ)
    for key in ("FLASHNH_SWEEP_V1_PACKAGE_ROOT", "FLASHNH_SWEEP_V1_SCREENING_BASIN_IDS",
                "FLASHNH_SWEEP_V1_OUTPUT_ROOT", "FLASHNH_SWEEP_V1_PROPOSAL_ORDER"):
        env.pop(key, None)
    env.update(overrides)
    return env


def test_agent_to_bridge_os_level_argv_contract(tmp_path):
    """Derives argv from the REAL serialized production sweep config (never
    manually constructed) and runs it as a real subprocess: proves the
    bridge's argparse accepts the exact argv W&B will construct, no
    swept-hyperparameter flags are appended, the four operational inputs
    resolve exactly from environment, and proposal order resolves to the
    explicit test value -- with no wandb import, network call, or training."""
    config_path = tmp_path / "sweep_config.json"
    subprocess.run(
        [sys.executable, str(BUILD_CONFIG_SCRIPT), "--output", str(config_path)],
        check=True, cwd=str(ROOT),
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["command"] == ["${interpreter}", "${program}"]

    program = str(BRIDGE_SCRIPT)
    argv = _expand_command(config["command"], interpreter=sys.executable, program=program)
    assert argv == [sys.executable, program]

    package_root = tmp_path / "package_root"
    screening_ids = tmp_path / "screening.txt"
    output_root = tmp_path / "output_root"
    env = _subprocess_env({
        "FLASHNH_SWEEP_V1_PACKAGE_ROOT": str(package_root),
        "FLASHNH_SWEEP_V1_SCREENING_BASIN_IDS": str(screening_ids),
        "FLASHNH_SWEEP_V1_OUTPUT_ROOT": str(output_root),
        "FLASHNH_SWEEP_V1_PROPOSAL_ORDER": "1",
        "FLASHNH_SWEEP_V1_BRIDGE_SELFTEST": "resolve_only",
    })

    result = subprocess.run(argv, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
    resolved = json.loads(result.stdout)
    assert resolved == {
        "package_root": str(package_root), "screening_basin_ids": str(screening_ids),
        "output_root": str(output_root), "proposal_order": 1,
    }


def test_agent_to_bridge_missing_required_input_fails_clearly(tmp_path):
    env = _subprocess_env({
        "FLASHNH_SWEEP_V1_SCREENING_BASIN_IDS": str(tmp_path / "screening.txt"),
        "FLASHNH_SWEEP_V1_OUTPUT_ROOT": str(tmp_path / "output_root"),
        "FLASHNH_SWEEP_V1_PROPOSAL_ORDER": "1",
        "FLASHNH_SWEEP_V1_BRIDGE_SELFTEST": "resolve_only",
    })
    result = subprocess.run([sys.executable, str(BRIDGE_SCRIPT)], cwd=str(ROOT), env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "missing required operational input" in result.stderr
    assert "package-root" in result.stderr


def test_agent_to_bridge_malformed_proposal_order_fails_clearly(tmp_path):
    env = _subprocess_env({
        "FLASHNH_SWEEP_V1_PACKAGE_ROOT": str(tmp_path / "package_root"),
        "FLASHNH_SWEEP_V1_SCREENING_BASIN_IDS": str(tmp_path / "screening.txt"),
        "FLASHNH_SWEEP_V1_OUTPUT_ROOT": str(tmp_path / "output_root"),
        "FLASHNH_SWEEP_V1_PROPOSAL_ORDER": "not-an-int",
        "FLASHNH_SWEEP_V1_BRIDGE_SELFTEST": "resolve_only",
    })
    result = subprocess.run([sys.executable, str(BRIDGE_SCRIPT)], cwd=str(ROOT), env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "is not an integer" in result.stderr


# --- 5. sbatch launcher static contract -------------------------------------

def test_sbatch_launcher_static_contract():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    code_lines = [line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    code_text = "\n".join(code_lines)

    assert "#SBATCH --gres=gpu:l4:1" in text
    assert "#SBATCH --cpus-per-task=8" in text
    assert "#SBATCH --mem=128G" in text
    assert "#SBATCH --time=08:00:00" in text
    assert "--array" not in text

    for env_name in ("WANDB_SWEEP_ID", "FLASHNH_SWEEP_V1_PACKAGE_ROOT", "FLASHNH_SWEEP_V1_SCREENING_BASIN_IDS",
                     "FLASHNH_SWEEP_V1_OUTPUT_ROOT", "FLASHNH_SWEEP_V1_PROPOSAL_ORDER"):
        assert f'"${{{env_name}:?' in text, f"{env_name} must be a required (:?) declaration"

    assert code_text.count("wandb agent") == 1
    assert "wandb agent --count 1" in code_text
    assert "while " not in code_text and "done" not in code_text
    assert "sbatch " not in code_text  # no self-resubmission / job chaining
    assert "WANDB_API_KEY" not in text
    assert "wandb login" not in text

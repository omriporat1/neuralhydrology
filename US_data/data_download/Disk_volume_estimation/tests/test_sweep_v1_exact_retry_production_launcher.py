"""Static contract tests for the production exact-retry Slurm launcher,
``scripts/run_sweep_v1_exact_retry_moriah.sbatch``.

Mirrors ``test_sweep_v1_launch_command_contract.py``'s section-5 static
sbatch-contract pattern for ``run_sweep_v1_wandb_agent_moriah.sbatch``, but
covers the exact-retry launcher's distinct shape: it invokes
``scripts/run_sweep_v1_exact_retry_bridge.py`` directly (never `wandb
agent`), so there is no W&B-constructed argv to test at the OS level here --
only this launcher's own static resource/env/concurrency/no-controller/
no-credential-exposure contract.

Never submits Slurm, never imports wandb, never touches the network.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SBATCH_SCRIPT = ROOT / "scripts" / "run_sweep_v1_exact_retry_moriah.sbatch"


def _code_text() -> str:
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    code_lines = [line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    return "\n".join(code_lines)


def test_exact_resources():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:l4:1" in text
    assert "#SBATCH --cpus-per-task=8" in text
    assert "#SBATCH --mem=128G" in text
    assert "#SBATCH --time=08:00:00" in text


def test_no_array_or_concurrency():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    code_text = _code_text()
    assert "--array" not in text
    assert "while " not in code_text and "done" not in code_text
    assert "sbatch " not in code_text  # no self-resubmission / job chaining


def test_no_wandb_agent_and_no_controller_proposal_request():
    code_text = _code_text()
    assert "wandb agent" not in code_text
    assert "--count" not in code_text  # `wandb agent --count N` is the only proposal-request idiom in this repo
    assert "proposal-order" not in code_text  # the bridge derives proposal_order from the frozen record; never settable here


def test_exactly_one_bridge_invocation():
    code_text = _code_text()
    assert code_text.count("run_sweep_v1_exact_retry_bridge.py") == 1


def test_execution_generation_is_required_and_explicit_with_no_default():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    assert '"${FLASHNH_RETRY_EXECUTION_GENERATION:?' in text
    assert ":-2" not in text and ":-1" not in text  # no silent default value
    assert '"$FLASHNH_RETRY_EXECUTION_GENERATION"' in text  # passed through verbatim to --execution-generation


def test_execution_generation_guard_rejects_original_and_non_positive_values():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    assert "0|1)" in text  # generation 0 and 1 (the original attempt's own generation) are both rejected


def test_fresh_output_root_is_required_with_no_default():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    assert '"${FLASHNH_RETRY_OUTPUT_ROOT:?' in text
    assert "attempt001" not in text  # no hardcoded default pointing at the original attempt's own directory


def test_all_required_inputs_hard_fail_when_missing():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    for env_name in (
        "FLASHNH_RETRY_FROZEN_PROPOSAL_RECORD", "FLASHNH_RETRY_EXPECTED_IDENTITY",
        "FLASHNH_RETRY_EXECUTION_GENERATION", "FLASHNH_RETRY_PACKAGE_ROOT",
        "FLASHNH_RETRY_SCREENING_BASIN_IDS", "FLASHNH_RETRY_OUTPUT_ROOT",
        "FLASHNH_RETRY_PROJECT", "FLASHNH_RETRY_ENTITY",
    ):
        assert f'"${{{env_name}:?' in text, f"{env_name} must be a required (:?) declaration"


def test_no_credential_exposure():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    assert "WANDB_API_KEY" not in text
    assert "wandb login" not in text


def test_prior_attempts_is_optional_and_passed_through_when_set():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    # Optional (":-", never ":?") -- generation 2 (the first retry) legitimately has no prior attempts.
    assert '"${FLASHNH_RETRY_PRIOR_ATTEMPTS:-}"' in text
    assert "--prior-attempts" in text
    assert '"$FLASHNH_RETRY_PRIOR_ATTEMPTS"' in text


def test_execution_generation_3_passes_the_guard():
    # The guard's only specific-value rejection pattern is "0|1)" (the
    # original attempt's own generation and below); anything else numeric,
    # including 3 (attempt003's generation), falls through unmodified to the
    # bridge invocation with no launcher code change required.
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    case_block = text.split("case ", 1)[1].split("esac", 1)[0]
    assert case_block.count(";;") == 2  # exactly two arms: the non-numeric guard and "0|1)" -- no arm for 2/3+
    assert "0|1)" in case_block

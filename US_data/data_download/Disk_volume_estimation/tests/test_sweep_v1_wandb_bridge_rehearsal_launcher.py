"""Static contract tests for the disposable fresh-proposal W&B-agent
bridge rehearsal's Slurm launcher,
``scripts/run_sweep_v1_wandb_bridge_rehearsal_moriah.sbatch``.

This is the fresh-bridge sibling of both
``tests/test_sweep_v1_exact_retry_rehearsal_launcher.py`` (the retry
bridge's rehearsal launcher, which deliberately never calls ``wandb agent``)
and ``tests/test_sweep_v1_launch_command_contract.py``'s section 5 (the
fresh bridge's PRODUCTION agent launcher). Unlike the retry-rehearsal
launcher, this one MUST call ``wandb agent --count 1`` for real -- that
real controller round trip is the entire point of qualifying the fresh
bridge (Section G). Asserts CPU-only resources, exactly one bounded
``wandb agent --count 1`` invocation, an explicit shell-level refusal of the
production sweep id (belt-and-suspenders on top of the manifest schema's own
refusal), and no credential exposure.

Never submits Slurm, never imports wandb, never touches the network.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SBATCH_SCRIPT = ROOT / "scripts" / "run_sweep_v1_wandb_bridge_rehearsal_moriah.sbatch"

_PRODUCTION_SWEEP_ID = "4x3btz2s"


def _code_text() -> str:
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    code_lines = [line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    return "\n".join(code_lines)


def test_cpu_only_resources_no_gpu():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    assert "--gres" not in text
    assert "gpu" not in text.lower()
    assert "#SBATCH --cpus-per-task=2" in text
    assert "#SBATCH --mem=8G" in text
    assert "#SBATCH --time=00:30:00" in text
    assert "#SBATCH --partition=glacier" in text


def test_no_array_or_concurrency_or_self_resubmission():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    code_text = _code_text()
    assert "--array" not in text
    assert "while " not in code_text and "for " not in code_text and "done" not in code_text
    invocation_lines = [line for line in code_text.splitlines() if "sbatch " in line]
    assert invocation_lines == []


def test_exactly_one_bounded_wandb_agent_invocation():
    code_text = _code_text()
    assert code_text.count("wandb agent") == 1
    assert "wandb agent --count 1" in code_text


def test_wandb_sweep_id_required_and_production_sweep_explicitly_refused():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    code_text = _code_text()
    assert '"${WANDB_SWEEP_ID:?' in text
    assert _PRODUCTION_SWEEP_ID in code_text
    # The refusal must be an executable guard, not just a comment mention.
    guard_lines = [
        line for line in code_text.splitlines()
        if _PRODUCTION_SWEEP_ID in line and ("if " in line or "FORBIDDEN" in line)
    ]
    assert guard_lines, "expected an executable guard referencing the production sweep id"
    assert "exit 1" in code_text


def test_wandb_project_and_entity_required_and_exported():
    # wandb agent's internal sweep-lookup query needs an explicit
    # project/entity to resolve WANDB_SWEEP_ID against -- there is no local
    # `wandb/settings` file on a fresh Slurm allocation to infer them from.
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    code_text = _code_text()
    assert '"${WANDB_PROJECT:?' in text
    assert '"${WANDB_ENTITY:?' in text
    assert "export WANDB_PROJECT WANDB_ENTITY" in code_text


def test_no_credential_exposure():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    assert "WANDB_API_KEY" not in text
    assert "wandb login" not in text
    assert ".netrc" not in text


def test_job_name_and_output_paths_identify_this_as_the_rehearsal_launcher():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    assert "#SBATCH --job-name=flashnh-sweep-v1-wandb-bridge-rehearsal" in text
    assert "sweep-v1-wandb-bridge-rehearsal-%j.out" in text
    assert "sweep-v1-wandb-bridge-rehearsal-%j.err" in text


def test_shell_level_checks_are_cheap_existence_checks_only():
    code_text = _code_text()
    assert "git rev-parse" not in code_text
    assert "git status" not in code_text
    assert 'test -d "${REPO_WORKDIR}"' in code_text
    assert 'test -x "${CANONICAL_PYTHON}"' in code_text


def test_path_export_prefers_canonical_python_directory():
    code_text = _code_text()
    assert 'export PATH="$(dirname "${CANONICAL_PYTHON}")' in code_text


def test_no_proposal_order_or_manifest_argument_on_this_launchers_own_cli():
    # Operational routing to rehearsal mode lives entirely in the disposable
    # sweep config's own `command` field (see
    # build_wandb_bridge_rehearsal_sweep_config); this launcher itself takes
    # no positional arguments and sets no FLASHNH_SWEEP_V1_* variables.
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    assert "FLASHNH_SWEEP_V1_" not in text
    assert '"${1' not in text

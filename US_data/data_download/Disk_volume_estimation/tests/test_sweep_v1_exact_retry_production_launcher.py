"""Static contract tests for the PRODUCTION exact-retry Slurm launcher,
``scripts/run_sweep_v1_exact_retry_moriah.sbatch``.

This launcher was migrated from a long ``FLASHNH_RETRY_*`` environment-
variable / multi-flag CLI interface to the same manifest-driven, one-
positional-argument shape as the disposable rehearsal launcher
(``tests/test_sweep_v1_exact_retry_rehearsal_launcher.py`` is the structural
model this file mirrors). Production and rehearsal now share the same
Python entry (``run_sweep_v1_exact_retry_bridge.py``'s ``main_from_manifest``),
manifest loader, runtime contract, identity resolution, W&B initialization
helper, config preparation, and executor selector -- they differ only in
Slurm resource directives and in the manifest's own ``mode`` field.

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


def test_production_gpu_and_resources_preserved():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    assert "#SBATCH --partition=catfish" in text
    assert "#SBATCH --gres=gpu:l4:1" in text
    assert "#SBATCH --cpus-per-task=8" in text
    assert "#SBATCH --mem=128G" in text
    assert "#SBATCH --time=08:00:00" in text
    assert "#SBATCH --job-name=flashnh-sweep-v1-exact-retry" in text
    assert "#SBATCH --job-name=flashnh-sweep-v1-exact-retry-rehearsal" not in text


def test_job_name_and_output_paths_identify_this_as_the_production_launcher():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    assert "sweep-v1-exact-retry-%j.out" in text
    assert "sweep-v1-exact-retry-%j.err" in text
    assert "sweep-v1-exact-retry-rehearsal-%j.out" not in text
    assert "sweep-v1-exact-retry-rehearsal-%j.err" not in text


def test_no_array_or_concurrency_or_self_resubmission():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    code_text = _code_text()
    assert "--array" not in text
    assert "while " not in code_text and "for " not in code_text
    invocation_lines = [
        line for line in code_text.splitlines() if "sbatch " in line and "usage:" not in line
    ]
    assert invocation_lines == []


def test_no_wandb_agent_and_no_controller_proposal_request():
    code_text = _code_text()
    assert "wandb agent" not in code_text
    assert "--count" not in code_text
    assert "proposal-order" not in code_text


def _bridge_invocation_lines(code_text: str) -> "list[str]":
    return [
        line for line in code_text.splitlines()
        if "run_sweep_v1_exact_retry_bridge.py" in line and "CANONICAL_PYTHON" in line
    ]


def test_exactly_one_bridge_invocation():
    code_text = _code_text()
    assert len(_bridge_invocation_lines(code_text)) == 1


def test_bridge_invocation_uses_exec_for_direct_exit_propagation():
    code_text = _code_text()
    bridge_line = _bridge_invocation_lines(code_text)[0]
    assert bridge_line.strip().startswith("exec ")


def test_single_positional_manifest_argument_no_long_export_or_flag_list():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    code_text = _code_text()
    bridge_line = _bridge_invocation_lines(code_text)[0]
    assert '"${MANIFEST_PATH}"' in bridge_line
    # No leftover legacy long-flag interface anywhere in the script.
    for legacy_flag in (
        "--frozen-proposal-record", "--expected-identity", "--execution-generation",
        "--prior-attempts", "--package-root", "--screening-basin-ids", "--output-root",
        "--baseline-policy-path", "--base-pilot-policy-path", "--project", "--entity",
    ):
        assert legacy_flag not in text
    assert "FLASHNH_RETRY_" not in text
    assert "--export=ALL" not in code_text


def test_rejects_missing_positional_argument():
    code_text = _code_text()
    assert 'if [ "$#" -ne 1 ]' in code_text


def test_rejects_extra_positional_arguments():
    # The strict "$#" -ne 1 guard rejects BOTH zero and two-or-more
    # positional arguments -- a strictly stronger check than the rehearsal
    # launcher's "${1:?...}"-only guard (which only catches a missing arg).
    code_text = _code_text()
    assert 'if [ "$#" -ne 1 ]' in code_text
    assert 'MANIFEST_PATH="$1"' in code_text


def test_strict_shell_mode_set_euo_pipefail():
    code_text = _code_text()
    assert "set -euo pipefail" in code_text


def test_no_credential_exposure():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    assert "WANDB_API_KEY" not in text
    assert "wandb login" not in text
    assert ".netrc" not in text


def test_no_hardcoded_sweep_id_manifest_is_sole_authority():
    # Unlike the rehearsal launcher (which is structurally forbidden from
    # ever reaching the production sweep by the manifest schema), the
    # production launcher CAN reach production -- but only via whatever
    # sweep id the supplied manifest itself carries, never a literal baked
    # into the shell script.
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    code_text = _code_text()
    assert "4x3btz2s" not in text
    assert "wandb_sweep_id" not in code_text


def test_shell_level_checks_are_cheap_existence_checks_only():
    # The authoritative commit/dirty-tree/interpreter pin must live in the
    # shared Python runtime contract (run_full_runtime_contract, called from
    # inside main_from_manifest), never duplicated in shell here.
    code_text = _code_text()
    assert "git rev-parse" not in code_text
    assert "git status" not in code_text
    assert 'test -f "${MANIFEST_PATH}"' in code_text
    assert 'test -d "${REPO_WORKDIR}"' in code_text
    assert 'test -x "${CANONICAL_PYTHON}"' in code_text

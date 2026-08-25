"""Static contract tests for the disposable exact-retry startup rehearsal's
Slurm launcher, ``scripts/run_sweep_v1_exact_retry_rehearsal_moriah.sbatch``.

Mirrors ``tests/test_sweep_v1_exact_retry_production_launcher.py``'s pattern
for the production launcher, but asserts the rehearsal's distinct shape per
Binding Design Decisions 1-2: one positional launch-manifest path (never a
long ``--export=ALL,VAR=value,...`` list), CPU-only resources (no GPU/GRES),
and no possible way to target the production sweep or start training.

Never submits Slurm, never imports wandb, never touches the network.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SBATCH_SCRIPT = ROOT / "scripts" / "run_sweep_v1_exact_retry_rehearsal_moriah.sbatch"


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


def test_no_array_or_concurrency_or_self_resubmission():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    code_text = _code_text()
    assert "--array" not in text
    assert "while " not in code_text and "for " not in code_text
    # The one "sbatch " occurrence in the code is inside the :? usage-message
    # string, not an actual self-resubmission invocation.
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
    # Distinguish the actual invocation (which runs the canonical
    # interpreter against the script) from any echoed/logged mention of the
    # script's filename.
    return [
        line for line in code_text.splitlines()
        if "run_sweep_v1_exact_retry_bridge.py" in line and "CANONICAL_PYTHON" in line
    ]


def test_exactly_one_bridge_invocation():
    code_text = _code_text()
    assert len(_bridge_invocation_lines(code_text)) == 1


def test_single_positional_manifest_argument_no_long_export_list():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    code_text = _code_text()
    assert 'MANIFEST_PATH="${1:?' in text
    # The single positional manifest path is the ONLY argument the bridge
    # invocation line passes -- no FLASHNH_RETRY_* long env-var interface.
    bridge_line = _bridge_invocation_lines(code_text)[0]
    assert '"${MANIFEST_PATH}"' in bridge_line
    assert "FLASHNH_RETRY_" not in text
    assert "--export=ALL" not in code_text


def test_no_credential_exposure():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    assert "WANDB_API_KEY" not in text
    assert "wandb login" not in text
    assert ".netrc" not in text


def test_cannot_be_pointed_at_production_sweep_by_construction():
    # This launcher's executable code never mentions a sweep id at all --
    # production-sweep refusal lives entirely in the manifest schema
    # (sweep_v1_launch_manifest.PRODUCTION_WANDB_SWEEP_ID), which is only
    # referenced here in an explanatory comment. Assert the executable code
    # stays silent on sweep identity, consistent with that single-authority
    # design.
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    code_text = _code_text()
    assert "4x3btz2s" not in text
    assert "wandb_sweep_id" not in code_text


def test_job_name_and_output_paths_identify_this_as_the_rehearsal_launcher():
    text = SBATCH_SCRIPT.read_text(encoding="utf-8")
    assert "#SBATCH --job-name=flashnh-sweep-v1-exact-retry-rehearsal" in text
    assert "sweep-v1-exact-retry-rehearsal-%j.out" in text
    assert "sweep-v1-exact-retry-rehearsal-%j.err" in text


def test_shell_level_checks_are_cheap_existence_checks_only():
    # The authoritative commit/dirty-tree/interpreter pin must live in the
    # shared Python runtime contract, never duplicated in shell here.
    code_text = _code_text()
    assert "git rev-parse" not in code_text
    assert "git status" not in code_text
    assert 'test -f "${MANIFEST_PATH}"' in code_text
    assert 'test -d "${REPO_WORKDIR}"' in code_text
    assert 'test -x "${CANONICAL_PYTHON}"' in code_text

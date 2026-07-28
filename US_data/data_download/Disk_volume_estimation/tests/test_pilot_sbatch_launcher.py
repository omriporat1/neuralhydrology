"""Focused tests for scripts/run_stage1_lead06_pilot_moriah.sbatch (task
item 7's launcher, task item 10's "generated paths outside tracked dirs"
checklist item).

This script is never submitted or executed by any test (per the task's
binding remote-access constraints) -- these are purely static/structural
checks on the committed script text, plus a `bash -n` syntax check (skipped
if bash is unavailable in the test environment).
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from tests._pilot_support import REPO_ROOT

SBATCH_PATH = REPO_ROOT / "scripts" / "run_stage1_lead06_pilot_moriah.sbatch"

ALL_RUN_IDS = ("raw_seedA", "raw_seedB", "emb128x64_seedA", "emb128x64_seedB", "emb64_seedA", "emb128_seedA")


def _text() -> str:
    return SBATCH_PATH.read_text(encoding="utf-8")


def test_sbatch_script_exists():
    assert SBATCH_PATH.is_file()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available in this environment")
def test_sbatch_script_bash_syntax_ok():
    result = subprocess.run(["bash", "-n", str(SBATCH_PATH)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_sbatch_script_not_hardcoded_to_a_single_run_id():
    text = _text()
    for run_id in ALL_RUN_IDS:
        assert run_id in text, f"{run_id} not referenced in sbatch script"
    # accepts run_id via $1 / RUN_ID env var, not a fixed literal passed to the CLI wrapper
    assert '--run-id "${RUN_ID}"' in text


def test_sbatch_script_default_output_roots_are_outside_the_tracked_repo_clone():
    text = _text()
    assert 'CONFIG_OUT_DIR="${PILOT_CONFIG_OUT_DIR:-${FLASHNH_BASE}/runs/' in text
    assert 'EVIDENCE_OUT_DIR="${PILOT_EVIDENCE_OUT_DIR:-${FLASHNH_BASE}/evidence/' in text
    # neither default output root is rooted under REPO_CLONE_DIR/REPO_WORKDIR
    assert '${REPO_CLONE_DIR}/runs' not in text
    assert '${REPO_WORKDIR}/runs' not in text
    assert '${REPO_CLONE_DIR}/evidence' not in text
    assert '${REPO_WORKDIR}/evidence' not in text


def test_sbatch_script_repo_clone_dir_is_the_authoritative_clean_clone_not_flashnh_base():
    text = _text()
    # code location is independent of FLASHNH_BASE (data/run storage root).
    assert (
        'PILOT_REPO_CLONE_DIR="${PILOT_REPO_CLONE_DIR:-/sci/labs/efratmorin/omripo/PhD/Python/neuralhydrology}"'
        in text
    )
    assert 'REPO_CLONE_DIR="${PILOT_REPO_CLONE_DIR}"' in text
    assert 'REPO_CLONE_DIR="${FLASHNH_BASE}' not in text
    # never the dirty secondary clone.
    assert "omripo/code/omriporat1/neuralhydrology" not in text


def test_sbatch_script_has_prelaunch_repository_guard():
    text = _text()
    assert 'test -d "${REPO_WORKDIR}"' in text
    assert "git rev-parse HEAD" in text
    assert "git status --porcelain" in text
    # refuses to launch when tracked files are locally modified (untracked
    # generated outputs, matched by ^??, are tolerated).
    assert "grep -v '^??'" in text
    assert "REFUSING to launch" in text


def test_sbatch_script_handles_term_signal_and_forwards_to_child():
    text = _text()
    # an actual trap, not just the bare --signal declaration.
    assert "trap _on_term TERM INT" in text
    assert "kill -TERM \"${PILOT_PID}\"" in text
    assert "TERM_REQUESTED=0" in text


def test_sbatch_script_interrupted_resumable_exits_nonzero_distinct_from_failed():
    text = _text()
    assert "INTERRUPTED_RESUMABLE) exit 2 ;;" in text
    assert "COMPLETED) exit 0 ;;" in text


def test_sbatch_script_never_passes_force_to_cli_wrapper():
    # --force is discussed in a comment (explaining why it's deliberately
    # omitted) but must never appear on an actual (non-comment) command line.
    code_lines = [
        line for line in _text().splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    assert not any("--force" in line for line in code_lines)


def test_sbatch_script_resource_defaults_match_starting_policy_not_corrected_seed_train_values():
    text = _text()
    assert "--cpus-per-task=8" in text
    assert "--mem=128G" in text
    # deliberately NOT the historical seed-train script's later OOM corrections
    assert "--cpus-per-task=16" not in text
    assert "--mem=224G" not in text


def test_sbatch_script_uses_the_only_confirmed_working_gpu_class():
    text = _text()
    assert "--gres=gpu:l4:1" in text

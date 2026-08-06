"""Focused tests for scripts/run_stage1_cap50k_closure_moriah.sbatch -- the
Seed-A 50k embedding-structure closure Slurm launcher.

This script's #SBATCH job is never submitted or executed by any test (per
this task's binding remote-access constraints: no Slurm submissions outside
the one explicitly authorized Part 4 read-only TB-loss-extraction carve-out,
which this file has nothing to do with). Checks here are either purely
static/structural on the committed script text, a `bash -n` syntax check, or
-- for the two-run allowlist, the required EXPECTED_COMMIT input, and the
commit-pin refusal/dirty-tree-refusal behavior -- standalone execution of
just the relevant extracted bash fragment (never Slurm, never GPU, never NH),
mirroring the convention already established in
tests/test_pilot_sbatch_launcher.py.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SBATCH_PATH = REPO_ROOT / "scripts" / "run_stage1_cap50k_closure_moriah.sbatch"

INCUMBENT = "emb128x64_seedA_cap_low_cal"
CHALLENGER = "emb128x32_seedA_cap_low_cal"

_BASH_AVAILABLE = shutil.which("bash") is not None
_GIT_AVAILABLE = shutil.which("git") is not None


def _text() -> str:
    return SBATCH_PATH.read_text(encoding="utf-8")


def test_sbatch_script_exists():
    assert SBATCH_PATH.is_file()


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash not available in this environment")
def test_sbatch_script_bash_syntax_ok():
    result = subprocess.run(["bash", "-n", str(SBATCH_PATH)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_sbatch_script_delegates_to_closure_cli_not_the_six_run_pilot_cli():
    text = _text()
    assert "python scripts/run_stage1_cap50k_closure.py" in text
    code_lines = [
        line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    assert not any("run_stage1_lead06_pilot.py" in line for line in code_lines)


# --- exact two-run allowlist -------------------------------------------------

def test_sbatch_script_allowlist_is_exactly_the_two_closure_run_ids():
    text = _text()
    assert f"{INCUMBENT}|{CHALLENGER}" in text
    assert "raw_seedA" not in text
    assert "emb128x64_seedA_cap_25k_cal" not in text


def _extract_run_id_case_block() -> str:
    text = _text()
    start_marker = 'RUN_ID="${RUN_ID:-${1:-}}"'
    start = text.find(start_marker)
    assert start != -1
    end_marker = "\nesac"
    end = text.find(end_marker, start)
    assert end != -1
    return text[start:end + len(end_marker)]


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash not available in this environment")
@pytest.mark.parametrize("run_id", [INCUMBENT, CHALLENGER])
def test_run_id_case_block_accepts_the_two_approved_run_ids(tmp_path, run_id):
    script_path = tmp_path / "run_id_case.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\nset -uo pipefail\n" + _extract_run_id_case_block() + '\necho "RUN_ID_OK=${RUN_ID}"\n',
        encoding="utf-8",
    )
    result = subprocess.run(["bash", str(script_path), run_id], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert f"RUN_ID_OK={run_id}" in result.stdout


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash not available in this environment")
@pytest.mark.parametrize("bad_run_id", ["raw_seedA", "emb128x64_seedA", "not_a_real_run_id", ""])
def test_run_id_case_block_rejects_any_other_run_id(tmp_path, bad_run_id):
    script_path = tmp_path / "run_id_case.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\nset -uo pipefail\n" + _extract_run_id_case_block() + '\necho "RUN_ID_OK=${RUN_ID}"\n',
        encoding="utf-8",
    )
    args = [bad_run_id] if bad_run_id else []
    result = subprocess.run(["bash", str(script_path), *args], capture_output=True, text=True)
    assert result.returncode == 2
    assert "USAGE" in result.stderr
    assert "RUN_ID_OK" not in result.stdout


# --- required explicit EXPECTED_COMMIT input --------------------------------

def _extract_expected_commit_block() -> str:
    text = _text()
    start_marker = 'EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"'
    start = text.find(start_marker)
    assert start != -1
    end_marker = "\nfi\n"
    end = text.find(end_marker, start)
    assert end != -1
    return text[start:end + len(end_marker)]


def test_sbatch_script_has_no_default_for_expected_commit():
    text = _text()
    assert 'EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"' in text
    # never a hardcoded fallback hash (the self-reference problem this
    # design deliberately avoids -- see the untracked calibration script's
    # stale EXPECTED_HEAD, discussed in the module header comment).
    assert 'EXPECTED_COMMIT:-' in text
    import re
    default_value = re.search(r'EXPECTED_COMMIT:-([^}]*)\}', text).group(1)
    assert default_value == ""


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash not available in this environment")
def test_expected_commit_block_refuses_when_unset(tmp_path):
    script_path = tmp_path / "expected_commit.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\nset -uo pipefail\nRUN_ID=test\n"
        + _extract_expected_commit_block() + '\necho "PASSED"\n',
        encoding="utf-8",
    )
    import os
    env = os.environ.copy()
    env.pop("EXPECTED_COMMIT", None)
    result = subprocess.run(["bash", str(script_path)], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "EXPECTED_COMMIT is required" in result.stderr
    assert "PASSED" not in result.stdout


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash not available in this environment")
def test_expected_commit_block_passes_when_set(tmp_path):
    script_path = tmp_path / "expected_commit.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\nset -uo pipefail\nRUN_ID=test\n"
        + _extract_expected_commit_block() + '\necho "PASSED"\n',
        encoding="utf-8",
    )
    import os
    env = os.environ.copy()
    env["EXPECTED_COMMIT"] = "deadbeefcafef00d"
    result = subprocess.run(["bash", str(script_path)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "PASSED" in result.stdout


# --- commit-pin + dirty-tree refusal, exercised behaviorally against a real
# temp git repo (never touches the actual project repo) ---------------------

def _extract_commit_pin_block() -> str:
    text = _text()
    start_marker = 'test -d "${REPO_WORKDIR}"'
    start = text.find(start_marker)
    assert start != -1
    end_marker = 'echo "Commit pin OK: HEAD matches EXPECTED_COMMIT (${EXPECTED_COMMIT})."'
    end = text.find(end_marker, start)
    assert end != -1
    return text[start:end + len(end_marker)]


def _make_temp_git_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    run = lambda *args: subprocess.run(
        ["git", *args], cwd=str(repo_dir), capture_output=True, text=True, check=True
    )
    run("init", "-q")
    run("config", "user.email", "test@example.com")
    run("config", "user.name", "Test")
    (repo_dir / "tracked.txt").write_text("v1\n", encoding="utf-8")
    run("add", "tracked.txt")
    run("commit", "-q", "-m", "init")
    return repo_dir


def _real_head(repo_dir: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_dir), capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def _run_commit_pin_block(tmp_path: Path, repo_dir: Path, expected_commit: str) -> subprocess.CompletedProcess:
    script_path = tmp_path / "commit_pin.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\nset -uo pipefail\n" + _extract_commit_pin_block() + "\n",
        encoding="utf-8",
    )
    import os
    env = os.environ.copy()
    env["REPO_WORKDIR"] = str(repo_dir)
    env["EXPECTED_COMMIT"] = expected_commit
    return subprocess.run(["bash", str(script_path)], env=env, capture_output=True, text=True)


@pytest.mark.skipif(not (_BASH_AVAILABLE and _GIT_AVAILABLE), reason="bash/git not available in this environment")
def test_commit_pin_block_passes_on_clean_tree_and_matching_commit(tmp_path):
    repo_dir = _make_temp_git_repo(tmp_path)
    result = _run_commit_pin_block(tmp_path, repo_dir, _real_head(repo_dir))
    assert result.returncode == 0, result.stderr
    assert "Commit pin OK" in result.stdout


@pytest.mark.skipif(not (_BASH_AVAILABLE and _GIT_AVAILABLE), reason="bash/git not available in this environment")
def test_commit_pin_block_refuses_on_head_mismatch(tmp_path):
    repo_dir = _make_temp_git_repo(tmp_path)
    result = _run_commit_pin_block(tmp_path, repo_dir, "0000000000000000000000000000000000000000")
    assert result.returncode == 1
    assert "REFUSING to launch" in result.stderr
    assert "does not match" in result.stderr
    assert "Commit pin OK" not in result.stdout


@pytest.mark.skipif(not (_BASH_AVAILABLE and _GIT_AVAILABLE), reason="bash/git not available in this environment")
def test_commit_pin_block_refuses_on_dirty_tracked_working_tree(tmp_path):
    repo_dir = _make_temp_git_repo(tmp_path)
    head = _real_head(repo_dir)
    (repo_dir / "tracked.txt").write_text("v2 -- locally modified\n", encoding="utf-8")
    result = _run_commit_pin_block(tmp_path, repo_dir, head)
    assert result.returncode == 1
    assert "REFUSING to launch" in result.stderr
    assert "locally modified" in result.stderr
    assert "Commit pin OK" not in result.stdout


@pytest.mark.skipif(not (_BASH_AVAILABLE and _GIT_AVAILABLE), reason="bash/git not available in this environment")
def test_commit_pin_block_tolerates_untracked_files_when_commit_matches(tmp_path):
    repo_dir = _make_temp_git_repo(tmp_path)
    head = _real_head(repo_dir)
    (repo_dir / "generated_evidence.json").write_text("{}\n", encoding="utf-8")  # untracked, not staged
    result = _run_commit_pin_block(tmp_path, repo_dir, head)
    assert result.returncode == 0, result.stderr
    assert "Commit pin OK" in result.stdout


# --- absence of --force / wandb sync in routine continuation ---------------

def test_sbatch_script_never_passes_force_to_cli_wrapper():
    code_lines = [
        line for line in _text().splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    assert not any("--force" in line for line in code_lines)


def test_sbatch_script_never_runs_wandb_sync():
    code_lines = [
        line for line in _text().splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    assert not any("wandb sync" in line for line in code_lines)


def test_sbatch_script_never_sets_wandb_mode_online():
    text = _text()
    assert "WANDB_MODE=online" not in text


def test_sbatch_script_exports_offline_mode_unconditionally_before_policy_check():
    # WANDB_MODE=offline must be exported outside/before the WANDB_POLICY_PATH
    # conditional -- never gated on a caller supplying a policy override.
    text = _text()
    export_idx = text.index("export WANDB_MODE=offline")
    policy_if_idx = text.index('if [ -n "${WANDB_POLICY_PATH}" ]; then')
    assert export_idx < policy_if_idx
    # the export line itself must not be inside any `if` body -- confirm no
    # unmatched conditional opens between the start of the script and the
    # export line by checking it precedes the first WANDB_POLICY_PATH read.
    policy_default_idx = text.index('WANDB_POLICY_PATH="${WANDB_POLICY_PATH:-}"')
    assert export_idx < policy_default_idx


def test_sbatch_script_never_uses_gated_wandb_mode_default_pattern():
    # A `${WANDB_MODE:-offline}` default would silently respect an inherited
    # online value; this pattern must never appear in actual code (comments
    # may reference it only to explain why it's avoided).
    code_lines = [
        line for line in _text().splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    assert not any("WANDB_MODE:-offline" in line for line in code_lines)


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash not available in this environment")
def test_wandb_offline_export_overrides_a_caller_supplied_online_value(tmp_path):
    start_marker = "# Approved closure horizon: fixed at epoch 12, enforced Python-side"
    end_marker = "export WANDB_MODE=offline"
    text = _text()
    start = text.index(start_marker)
    end = text.index(end_marker, start) + len(end_marker)
    fragment = text[start:end]
    script_path = tmp_path / "wandb_offline.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\nset -uo pipefail\n" + fragment + '\necho "WANDB_MODE=${WANDB_MODE}"\n',
        encoding="utf-8",
    )
    import os
    env = os.environ.copy()
    env["WANDB_MODE"] = "online"
    result = subprocess.run(["bash", str(script_path)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "WANDB_MODE=offline" in result.stdout


def test_sbatch_script_echoes_effective_wandb_mode_in_always_printed_info_block():
    text = _text()
    echo_idx = text.index('echo "WANDB_MODE (effective, enforced offline, non-overridable): ${WANDB_MODE}"')
    # must appear in the unconditional info block, i.e. before any
    # PREPARE_ONLY/STATUS_ONLY branching.
    prepare_only_branch_idx = text.index('if [ "${PREPARE_ONLY}" = "1" ] || [ "${STATUS_ONLY}" = "1" ]; then')
    assert echo_idx < prepare_only_branch_idx


# --- epoch-12 closure horizon echoed for every mode, including PREPARE_ONLY/
# STATUS_ONLY (no training) -----------------------------------------------

def test_sbatch_script_defines_fixed_closure_max_target_epoch_constant():
    text = _text()
    assert "CLOSURE_MAX_TARGET_EPOCH=12" in text
    code_lines = [
        line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    assert not any('MAX_TARGET_EPOCH="${MAX_TARGET_EPOCH' in line for line in code_lines)
    assert not any("--max-target-epoch" in line for line in code_lines)


def test_sbatch_script_echoes_closure_horizon_in_always_printed_info_block():
    text = _text()
    echo_idx = text.index(
        'echo "Approved closure horizon (fixed, non-overridable): epoch ${CLOSURE_MAX_TARGET_EPOCH}"'
    )
    prepare_only_branch_idx = text.index('if [ "${PREPARE_ONLY}" = "1" ] || [ "${STATUS_ONLY}" = "1" ]; then')
    assert echo_idx < prepare_only_branch_idx


# --- STATUS_ONLY / PREPARE_ONLY mutual exclusion ----------------------------

def test_sbatch_script_status_only_and_prepare_only_are_mutually_exclusive():
    text = _text()
    assert 'STATUS_ONLY="${STATUS_ONLY:-0}"' in text
    assert 'if [ "${PREPARE_ONLY}" = "1" ] && [ "${STATUS_ONLY}" = "1" ]; then' in text


def _extract_status_prepare_mutex_block() -> str:
    text = _text()
    start_marker = 'PREPARE_ONLY="${PREPARE_ONLY:-0}"'
    start = text.find(start_marker)
    assert start != -1
    end_marker = "\nfi\n"
    end = text.find(end_marker, start)
    assert end != -1
    return text[start:end + len(end_marker)]


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash not available in this environment")
def test_status_only_and_prepare_only_mutual_exclusion_block_behaves(tmp_path):
    script_path = tmp_path / "mutex.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\nset -uo pipefail\n" + _extract_status_prepare_mutex_block() + '\necho "PASSED"\n',
        encoding="utf-8",
    )
    import os
    env = os.environ.copy()
    env["PREPARE_ONLY"] = "1"
    env["STATUS_ONLY"] = "1"
    result = subprocess.run(["bash", str(script_path)], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "mutually exclusive" in result.stderr
    assert "PASSED" not in result.stdout


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash not available in this environment")
def test_status_only_alone_passes_mutex_block(tmp_path):
    script_path = tmp_path / "mutex.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\nset -uo pipefail\n" + _extract_status_prepare_mutex_block() + '\necho "PASSED"\n',
        encoding="utf-8",
    )
    import os
    env = os.environ.copy()
    env["PREPARE_ONLY"] = "0"
    env["STATUS_ONLY"] = "1"
    result = subprocess.run(["bash", str(script_path)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "PASSED" in result.stdout


def test_sbatch_script_skips_gpu_and_package_root_checks_in_status_only_mode():
    text = _text()
    assert 'if [ "${STATUS_ONLY}" != "1" ]; then' in text
    assert "MISSING transferred package" in text
    assert 'if [ "${PREPARE_ONLY}" = "1" ] || [ "${STATUS_ONLY}" = "1" ]; then' in text


def test_sbatch_script_status_only_uses_synchronous_call_not_background_pid():
    text = _text()
    status_idx = text.index('if [ "${STATUS_ONLY}" = "1" ]; then\n    # Status-only')
    background_idx = text.index("PILOT_PID=$!")
    assert status_idx < background_idx


# --- output roots outside the tracked repo clone ----------------------------

def test_sbatch_script_default_output_roots_are_outside_the_tracked_repo_clone():
    text = _text()
    assert 'CONFIG_OUT_DIR="${PILOT_CONFIG_OUT_DIR:-${FLASHNH_BASE}/runs/' in text
    assert 'EVIDENCE_OUT_DIR="${PILOT_EVIDENCE_OUT_DIR:-${FLASHNH_BASE}/evidence/' in text
    assert "${REPO_CLONE_DIR}/runs" not in text
    assert "${REPO_WORKDIR}/runs" not in text

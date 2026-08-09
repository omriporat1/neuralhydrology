"""Focused tests for scripts/
wandb_offline_launch_contract_qualification_moriah.sbatch -- the generic W&B
offline tracking launch-contract qualification's Moriah Slurm launcher.

This script's #SBATCH job is never submitted or executed by any test. Checks
here are either purely static/structural on the committed script text, a
`bash -n` syntax check, or -- for the required EXPECTED_COMMIT input and the
commit-pin/dirty-tree refusal behavior -- standalone execution of just the
relevant extracted bash fragment (never Slurm, never GPU, never NH),
mirroring the convention established in
tests/test_lr_range_seedA_closure_sbatch_launcher.py.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SBATCH_PATH = REPO_ROOT / "scripts" / "wandb_offline_launch_contract_qualification_moriah.sbatch"

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


# --- delegates only to the qualification script, never a campaign/training CLI --

def test_sbatch_script_delegates_to_qualification_script_not_other_clis():
    text = _text()
    assert "python scripts/wandb_offline_launch_contract_qualification.py" in text
    code_lines = [
        line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    assert not any("run_stage1_lr_range_seedA_closure.py" in line for line in code_lines)
    assert not any("run_stage1_lead06_pilot.py" in line for line in code_lines)
    assert not any("run_stage1_cap50k_closure.py" in line for line in code_lines)
    assert not any("import neuralhydrology" in line for line in code_lines)
    assert not any(" train.py" in line for line in code_lines)


# --- CPU-only: confirmed no-GRES partition, no GPU request -------------------

def test_sbatch_script_uses_cpu_only_glacier_partition_with_no_gres():
    text = _text()
    assert "#SBATCH --partition=glacier" in text
    assert "--gres" not in text
    assert "gpu:" not in text


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
    import re
    default_value = re.search(r'EXPECTED_COMMIT:-([^}]*)\}', text).group(1)
    assert default_value == ""


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash not available in this environment")
def test_expected_commit_block_refuses_when_unset(tmp_path):
    script_path = tmp_path / "expected_commit.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\nset -uo pipefail\n"
        + _extract_expected_commit_block() + '\necho "PASSED"\n',
        encoding="utf-8",
    )
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
        "#!/usr/bin/env bash\nset -uo pipefail\n"
        + _extract_expected_commit_block() + '\necho "PASSED"\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["EXPECTED_COMMIT"] = "deadbeefcafef00d"
    result = subprocess.run(["bash", str(script_path)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "PASSED" in result.stdout


# --- WANDB_MODE forced offline, unconditionally, before anything else -------

def test_sbatch_script_never_sets_wandb_mode_online():
    text = _text()
    assert "WANDB_MODE=online" not in text


def test_sbatch_script_exports_offline_mode_unconditionally_before_policy_default():
    text = _text()
    export_idx = text.index("export WANDB_MODE=offline")
    policy_default_idx = text.index(
        'WANDB_POLICY_PATH="${WANDB_POLICY_PATH:-config/stage1_wandb_tracking_policy_offline_v001.yaml}"'
    )
    assert export_idx < policy_default_idx


def test_sbatch_script_never_uses_gated_wandb_mode_default_pattern():
    code_lines = [
        line for line in _text().splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    assert not any("WANDB_MODE:-offline" in line for line in code_lines)


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash not available in this environment")
def test_wandb_offline_export_overrides_a_caller_supplied_online_value(tmp_path):
    start_marker = "export WANDB_MODE=offline"
    text = _text()
    start = text.index(start_marker)
    fragment = text[start:start + len(start_marker)]
    script_path = tmp_path / "wandb_offline.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\nset -uo pipefail\n" + fragment + '\necho "WANDB_MODE=${WANDB_MODE}"\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["WANDB_MODE"] = "online"
    result = subprocess.run(["bash", str(script_path)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "WANDB_MODE=offline" in result.stdout


def test_sbatch_script_echoes_effective_wandb_mode():
    text = _text()
    assert 'echo "WANDB_MODE (effective, enforced offline, non-overridable): ${WANDB_MODE}"' in text


def test_sbatch_script_never_runs_wandb_sync():
    code_lines = [
        line for line in _text().splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    assert not any("wandb sync" in line for line in code_lines)


# --- WANDB_POLICY_PATH defaults to the reviewed offline-enabled override, ---
# --- still overridable like every other Stage 1 launcher --------------------

def test_sbatch_script_defaults_wandb_policy_path_to_the_reviewed_offline_override():
    text = _text()
    assert (
        'WANDB_POLICY_PATH="${WANDB_POLICY_PATH:-config/stage1_wandb_tracking_policy_offline_v001.yaml}"'
        in text
    )


def test_sbatch_script_passes_wandb_policy_path_through_to_the_qualification_cli():
    text = _text()
    assert '--wandb-policy-path "${WANDB_POLICY_PATH}"' in text
    assert '--wandb-dir "${WANDB_DIR}"' in text


# --- commit-pin + dirty-tree refusal, exercised against a real temp git ----

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
    assert "REFUSING to run" in result.stderr
    assert "does not match" in result.stderr
    assert "Commit pin OK" not in result.stdout


@pytest.mark.skipif(not (_BASH_AVAILABLE and _GIT_AVAILABLE), reason="bash/git not available in this environment")
def test_commit_pin_block_refuses_on_dirty_tracked_working_tree(tmp_path):
    repo_dir = _make_temp_git_repo(tmp_path)
    head = _real_head(repo_dir)
    (repo_dir / "tracked.txt").write_text("v2 -- locally modified\n", encoding="utf-8")
    result = _run_commit_pin_block(tmp_path, repo_dir, head)
    assert result.returncode == 1
    assert "REFUSING to run" in result.stderr
    assert "Commit pin OK" not in result.stdout


@pytest.mark.skipif(not (_BASH_AVAILABLE and _GIT_AVAILABLE), reason="bash/git not available in this environment")
def test_commit_pin_block_tolerates_untracked_files_when_commit_matches(tmp_path):
    repo_dir = _make_temp_git_repo(tmp_path)
    head = _real_head(repo_dir)
    (repo_dir / "generated_evidence.json").write_text("{}\n", encoding="utf-8")
    result = _run_commit_pin_block(tmp_path, repo_dir, head)
    assert result.returncode == 0, result.stderr
    assert "Commit pin OK" in result.stdout


# --- ephemeral venv: created and torn down within the job, never mutates ---
# --- the shared flashnh-moriah conda env ------------------------------------

def test_sbatch_script_installs_wandb_into_an_ephemeral_venv_not_the_shared_env():
    text = _text()
    assert "python3 -m venv" in text
    assert "pip install --quiet wandb pyyaml" in text
    assert "rm -rf \"${VENV_DIR}\"" in text
    assert "conda activate flashnh-moriah" not in text
    assert "conda install" not in text


def test_sbatch_script_never_defines_a_gres_or_gpu_module():
    code_lines = [
        line for line in _text().splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    assert not any("cuda" in line.lower() for line in code_lines)
    assert not any("nvidia" in line.lower() for line in code_lines)


# --- exits with the qualification script's own exit code -------------------

def test_sbatch_script_propagates_qualification_script_exit_code():
    text = _text()
    assert "QUAL_STATUS=$?" in text
    assert 'exit "${QUAL_STATUS}"' in text


# --- output/result directory outside the tracked repo clone -----------------

def test_sbatch_script_result_dir_is_outside_the_tracked_repo_clone():
    text = _text()
    assert 'RESULT_DIR="${FLASHNH_BASE}/evidence/wandb_offline_launch_contract_qualification_${SLURM_JOB_ID}"' in text
    assert "${REPO_CLONE_DIR}/evidence" not in text
    assert "${REPO_WORKDIR}/evidence" not in text

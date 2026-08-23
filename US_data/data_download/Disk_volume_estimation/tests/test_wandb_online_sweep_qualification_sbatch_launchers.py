"""Focused structural tests for the three Phase-B online W&B sweep
qualification Moriah Slurm launchers:
  * scripts/wandb_online_sweep_qualification_preflight_moriah.sbatch
  * scripts/wandb_online_sweep_qualification_run_moriah.sbatch
  * scripts/wandb_online_sweep_qualification_lifecycle_moriah.sbatch

None of these #SBATCH jobs are ever submitted or executed by any test. Checks
are static/structural on the committed script text, a `bash -n` syntax
check, and standalone execution of the extracted EXPECTED_COMMIT/commit-pin
bash fragments -- mirroring tests/
test_wandb_offline_launch_contract_qualification_sbatch_launcher.py.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PREFLIGHT_SBATCH = REPO_ROOT / "scripts" / "wandb_online_sweep_qualification_preflight_moriah.sbatch"
RUN_SBATCH = REPO_ROOT / "scripts" / "wandb_online_sweep_qualification_run_moriah.sbatch"
LIFECYCLE_SBATCH = REPO_ROOT / "scripts" / "wandb_online_sweep_qualification_lifecycle_moriah.sbatch"
ALL_SBATCH = (PREFLIGHT_SBATCH, RUN_SBATCH, LIFECYCLE_SBATCH)

_BASH_AVAILABLE = shutil.which("bash") is not None
_GIT_AVAILABLE = shutil.which("git") is not None


# ---------------------------------------------------------------------------
# existence + syntax
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ALL_SBATCH)
def test_sbatch_script_exists(path):
    assert path.is_file()


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash not available in this environment")
@pytest.mark.parametrize("path", ALL_SBATCH)
def test_sbatch_script_bash_syntax_ok(path):
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# CPU-only: confirmed no-GRES partition, no GPU request, on all three
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ALL_SBATCH)
def test_sbatch_script_uses_cpu_only_glacier_partition_with_no_gres(path):
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --partition=glacier" in text
    assert "--gres" not in text
    assert "gpu:" not in text
    code_lines = [line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    assert not any("cuda" in line.lower() for line in code_lines)
    assert not any("nvidia" in line.lower() for line in code_lines)


# ---------------------------------------------------------------------------
# never forces offline mode -- the whole point is ONLINE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ALL_SBATCH)
def test_sbatch_script_never_forces_offline_mode(path):
    code_lines = [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert not any("WANDB_MODE=offline" in line for line in code_lines)


# ---------------------------------------------------------------------------
# uses the canonical Moriah runtime directly -- no ephemeral venv
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ALL_SBATCH)
def test_sbatch_script_never_creates_an_ephemeral_venv(path):
    text = path.read_text(encoding="utf-8")
    assert "python3 -m venv" not in text
    assert "pip install" not in text


@pytest.mark.parametrize("path", ALL_SBATCH)
def test_sbatch_script_references_canonical_runtime_python(path):
    text = path.read_text(encoding="utf-8")
    assert "envs/flashnh-moriah/bin/python" in text


# ---------------------------------------------------------------------------
# never imports neuralhydrology / touches training CLIs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ALL_SBATCH)
def test_sbatch_script_never_touches_training_or_nh(path):
    code_lines = [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert not any("import neuralhydrology" in line for line in code_lines)
    assert not any(" train.py" in line for line in code_lines)
    assert not any("run_stage1_lr_range_seedA_closure.py" in line for line in code_lines)


# ---------------------------------------------------------------------------
# each delegates to its own qualification script only
# ---------------------------------------------------------------------------

def test_preflight_sbatch_delegates_to_preflight_script():
    text = PREFLIGHT_SBATCH.read_text(encoding="utf-8")
    assert "wandb_online_sweep_qualification_preflight.py" in text
    assert "wandb_online_sweep_qualification_run.py" not in text
    assert "wandb_online_sweep_qualification_lifecycle.py" not in text


def test_run_sbatch_delegates_to_run_script():
    text = RUN_SBATCH.read_text(encoding="utf-8")
    assert "wandb_online_sweep_qualification_run.py" in text
    assert "wandb_online_sweep_qualification_preflight.py" not in text
    assert "wandb_online_sweep_qualification_lifecycle.py" not in text


def test_lifecycle_sbatch_delegates_to_lifecycle_script():
    text = LIFECYCLE_SBATCH.read_text(encoding="utf-8")
    assert "wandb_online_sweep_qualification_lifecycle.py" in text
    assert "wandb_online_sweep_qualification_preflight.py" not in text
    assert "wandb_online_sweep_qualification_run.py" not in text


# ---------------------------------------------------------------------------
# required-input guards: EXPECTED_COMMIT (preflight + run), PROPOSAL_LABEL
# (run), SWEEP_ID + ACTION (lifecycle) -- all default to empty, never a
# silently-accepted value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", (PREFLIGHT_SBATCH, RUN_SBATCH))
def test_expected_commit_has_no_default(path):
    text = path.read_text(encoding="utf-8")
    assert 'EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"' in text
    import re
    default_value = re.search(r'EXPECTED_COMMIT:-([^}]*)\}', text).group(1)
    assert default_value == ""


def test_run_sbatch_requires_proposal_label():
    text = RUN_SBATCH.read_text(encoding="utf-8")
    assert 'PROPOSAL_LABEL="${PROPOSAL_LABEL:-}"' in text
    assert "PROPOSAL_LABEL is required" in text


def test_lifecycle_sbatch_requires_sweep_id_and_restricts_action():
    text = LIFECYCLE_SBATCH.read_text(encoding="utf-8")
    assert 'SWEEP_ID="${SWEEP_ID:-}"' in text
    assert "SWEEP_ID is required" in text
    assert "status|pause|resume|stop)" in text


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash not available in this environment")
def test_run_sbatch_sweep_id_is_optional_and_maps_to_cli_flag():
    text = RUN_SBATCH.read_text(encoding="utf-8")
    assert "SWEEP_ID_ARGS=()" in text
    assert '--sweep-id "${SWEEP_ID}"' in text


# ---------------------------------------------------------------------------
# commit-pin + dirty-tree refusal, exercised against a real temp git repo
# (preflight and run launchers only -- lifecycle never touches the repo
# clone's git state)
# ---------------------------------------------------------------------------

def _extract_commit_pin_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
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
    run = lambda *args: subprocess.run(["git", *args], cwd=str(repo_dir), capture_output=True, text=True, check=True)
    run("init", "-q")
    run("config", "user.email", "test@example.com")
    run("config", "user.name", "Test")
    (repo_dir / "tracked.txt").write_text("v1\n", encoding="utf-8")
    run("add", "tracked.txt")
    run("commit", "-q", "-m", "init")
    return repo_dir


def _real_head(repo_dir: Path) -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def _run_commit_pin_block(tmp_path: Path, sbatch_path: Path, repo_dir: Path, expected_commit: str) -> subprocess.CompletedProcess:
    script_path = tmp_path / "commit_pin.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\nset -uo pipefail\n" + _extract_commit_pin_block(sbatch_path) + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["REPO_WORKDIR"] = str(repo_dir)
    env["EXPECTED_COMMIT"] = expected_commit
    return subprocess.run(["bash", str(script_path)], env=env, capture_output=True, text=True)


@pytest.mark.skipif(not (_BASH_AVAILABLE and _GIT_AVAILABLE), reason="bash/git not available in this environment")
@pytest.mark.parametrize("sbatch_path", (PREFLIGHT_SBATCH, RUN_SBATCH))
def test_commit_pin_block_passes_on_clean_tree_and_matching_commit(tmp_path, sbatch_path):
    repo_dir = _make_temp_git_repo(tmp_path)
    result = _run_commit_pin_block(tmp_path, sbatch_path, repo_dir, _real_head(repo_dir))
    assert result.returncode == 0, result.stderr
    assert "Commit pin OK" in result.stdout


@pytest.mark.skipif(not (_BASH_AVAILABLE and _GIT_AVAILABLE), reason="bash/git not available in this environment")
@pytest.mark.parametrize("sbatch_path", (PREFLIGHT_SBATCH, RUN_SBATCH))
def test_commit_pin_block_refuses_on_head_mismatch(tmp_path, sbatch_path):
    repo_dir = _make_temp_git_repo(tmp_path)
    result = _run_commit_pin_block(tmp_path, sbatch_path, repo_dir, "0" * 40)
    assert result.returncode == 1
    assert "REFUSING to run" in result.stderr
    assert "Commit pin OK" not in result.stdout


@pytest.mark.skipif(not (_BASH_AVAILABLE and _GIT_AVAILABLE), reason="bash/git not available in this environment")
@pytest.mark.parametrize("sbatch_path", (PREFLIGHT_SBATCH, RUN_SBATCH))
def test_commit_pin_block_refuses_on_dirty_tracked_working_tree(tmp_path, sbatch_path):
    repo_dir = _make_temp_git_repo(tmp_path)
    head = _real_head(repo_dir)
    (repo_dir / "tracked.txt").write_text("v2 -- locally modified\n", encoding="utf-8")
    result = _run_commit_pin_block(tmp_path, sbatch_path, repo_dir, head)
    assert result.returncode == 1
    assert "REFUSING to run" in result.stderr


# ---------------------------------------------------------------------------
# exit-code propagation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ALL_SBATCH)
def test_sbatch_script_propagates_qualification_script_exit_code(path):
    text = path.read_text(encoding="utf-8")
    assert "QUAL_STATUS=$?" in text
    assert 'exit "${QUAL_STATUS}"' in text


# ---------------------------------------------------------------------------
# evidence written outside the tracked repo clone
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ALL_SBATCH)
def test_sbatch_script_result_dir_is_outside_the_tracked_repo_clone(path):
    text = path.read_text(encoding="utf-8")
    assert "evidence/phase_b_wandb_online_sweep_qualification_v001" in text
    assert "${REPO_WORKDIR}/evidence" not in text

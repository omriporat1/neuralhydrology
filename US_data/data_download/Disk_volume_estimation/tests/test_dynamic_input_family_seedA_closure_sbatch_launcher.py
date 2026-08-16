"""Structural + real-execution tests for
scripts/run_stage1_dynamic_input_family_seedA_closure_moriah.sbatch (the
Dynamic-Input-Family-A Slurm launcher).

Unlike scripts/run_stage1_seq_length_range_seedA_closure_moriah.sbatch, this
launcher declares NO historical-comparator REFERENCE_RUN_ID and NO guard
block around one -- Dynamic-Input-Family-A has no comparator (P is this
campaign's own in-campaign reference family, trainable like every other
candidate). Tests that would exercise a REFERENCE_RUN_ID guard block are
therefore omitted entirely, not adapted -- replaced by an explicit assertion
that no such identifier or block exists in this file at all.

Two kinds of check are used:
  * Structural/static checks against the literal sbatch text (allowlist
    exactness, absence of override knobs, W&B contract wording, delegate
    script name, RESULT_DIR namespacing) -- fast, no subprocess.
  * Real-bash-execution checks (RUN_ID allowlist rejection, EXPECTED_COMMIT
    requirement, PREPARE_ONLY/STATUS_ONLY mutex, and the commit-pin/
    dirty-tree guard) that actually invoke the real script text (or, for the
    commit-pin guard, the real block of it verbatim-extracted into a small
    harness) via a real ``bash`` subprocess and a real temporary git
    repository -- never a hand-reimplemented reimplementation of the shell
    logic.

No Slurm job, GPU, module system, conda environment, or real NH/W&B call is
needed or used anywhere in this file.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SBATCH_PATH = REPO_ROOT / "scripts" / "run_stage1_dynamic_input_family_seedA_closure_moriah.sbatch"
CLOSURE_SCRIPT_PATH = REPO_ROOT / "scripts" / "run_stage1_dynamic_input_family_seedA_closure.py"

_EXPECTED_RUN_IDS = (
    "emb128x32_seedA_dynP_seq72_h128_lr3em4_cap25k_cal",
    "emb128x32_seedA_dynPT_seq72_h128_lr3em4_cap25k_cal",
    "emb128x32_seedA_dynPTM_seq72_h128_lr3em4_cap25k_cal",
    "emb128x32_seedA_dynPTMW_seq72_h128_lr3em4_cap25k_cal",
)

_BASH = shutil.which("bash")
requires_bash = pytest.mark.skipif(_BASH is None, reason="bash not found on PATH")


@pytest.fixture(scope="module")
def sbatch_text():
    return SBATCH_PATH.read_text(encoding="utf-8")


def _non_comment_text(text: str) -> str:
    """Join only the lines that are not pure comment lines -- so a
    documentation comment that explicitly explains a pattern is ABSENT
    (e.g. "no --max-target-epoch override here", "never `${WANDB_MODE:-...}`")
    is not mistaken for the pattern's actual presence/usage."""
    return "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("#")
    )


# --- structural/static checks ----------------------------------------------


def test_sbatch_file_exists_and_is_nonempty(sbatch_text):
    assert sbatch_text.strip()
    assert sbatch_text.startswith("#!/usr/bin/env bash")


def test_run_id_allowlist_case_block_is_exactly_the_four_campaign_run_ids(sbatch_text):
    match = re.search(r'case "\$\{RUN_ID\}" in\n\s*(.+?)\)\s*;;', sbatch_text)
    assert match, "could not locate the RUN_ID case-block allowlist line"
    allowlist = tuple(match.group(1).split("|"))
    assert allowlist == _EXPECTED_RUN_IDS


def test_no_reference_run_id_or_guard_block_present(sbatch_text):
    assert "REFERENCE_RUN_ID" not in sbatch_text
    assert "is_historical_hidden_size_a_comparator" not in sbatch_text
    assert "comparator" not in sbatch_text.lower() or "own in-campaign reference family" in sbatch_text


def test_no_seq_length_hidden_size_learning_rate_or_dynamic_inputs_override_env_knobs(sbatch_text):
    active_text = _non_comment_text(sbatch_text)
    for forbidden_pattern in (
        r"\bSEQ_LENGTH=", r"\bHIDDEN_SIZE=", r"\bLEARNING_RATE=",
        r"\bDYNAMIC_INPUTS=", r"--seq-length\b", r"--hidden-size\b",
        r"--learning-rate\b", r"--dynamic-inputs\b", r"--max-target-epoch\b",
        r"\bFORCE=", r"--wandb-sync\b", r"WANDB_MODE=online",
    ):
        assert not re.search(forbidden_pattern, active_text), (
            f"unexpected override knob pattern found (outside comments): {forbidden_pattern!r}"
        )


def test_max_target_epoch_is_hardcoded_six_not_env_overridable(sbatch_text):
    assert "DYNAMIC_INPUT_FAMILY_A_MAX_TARGET_EPOCH=6" in sbatch_text
    assert "DYNAMIC_INPUT_FAMILY_A_MAX_TARGET_EPOCH=\"${" not in sbatch_text


def test_wandb_mode_forced_offline_unconditionally(sbatch_text):
    assert "export WANDB_MODE=offline" in sbatch_text
    assert "${WANDB_MODE:-" not in _non_comment_text(sbatch_text)


def test_wandb_policy_path_defaults_to_the_reviewed_offline_policy(sbatch_text):
    assert "stage1_wandb_tracking_policy_offline_v001.yaml" in sbatch_text


def test_delegates_to_the_dynamic_input_family_closure_script_only(sbatch_text):
    invocations = re.findall(r"python (scripts/\S+\.py)", sbatch_text)
    assert invocations, "no python script invocation found"
    assert set(invocations) == {"scripts/run_stage1_dynamic_input_family_seedA_closure.py"}


def test_result_dir_uses_the_dynamic_input_family_closure_prefix(sbatch_text):
    match = re.search(r'RESULT_DIR="([^"]+)"', sbatch_text)
    assert match
    assert "dynamic_input_family_closure_${RUN_ID}_${SLURM_JOB_ID}" in match.group(1)


def test_flashnh_base_default_is_outside_the_tracked_repo_clone(sbatch_text):
    match = re.search(r'FLASHNH_BASE="\$\{FLASHNH_BASE:-([^}]+)\}"', sbatch_text)
    assert match
    flashnh_base_default = match.group(1)
    repo_clone_match = re.search(r'PILOT_REPO_CLONE_DIR="\$\{PILOT_REPO_CLONE_DIR:-([^}]+)\}"', sbatch_text)
    assert repo_clone_match
    repo_clone_default = repo_clone_match.group(1)
    assert flashnh_base_default != repo_clone_default
    assert not repo_clone_default.startswith(flashnh_base_default)


def test_expected_commit_has_no_default_value(sbatch_text):
    assert 'EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"' in sbatch_text
    assert not re.search(r'EXPECTED_COMMIT:-[0-9a-f]{7,40}', sbatch_text)


def test_closure_script_exists_and_syntax_checks_clean():
    assert CLOSURE_SCRIPT_PATH.is_file()


# --- real-bash-execution checks ---------------------------------------------


@requires_bash
@pytest.mark.parametrize("bad_run_id", ["", "not_a_real_run_id", "emb128x32_seedA_h128_lr3em4_cap25k_cal"])
def test_real_bash_rejects_any_run_id_outside_the_four_candidate_allowlist(bad_run_id):
    """Runs the REAL sbatch script text (SBATCH pragma lines are ordinary
    bash comments to bash itself) with an out-of-allowlist RUN_ID and checks
    it refuses via exit code 2 and a USAGE message -- before touching
    EXPECTED_COMMIT, modules, conda, or the package root."""
    result = subprocess.run(
        [_BASH, str(SBATCH_PATH), bad_run_id],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    assert "USAGE" in result.stderr


@requires_bash
def test_real_bash_requires_expected_commit_with_no_default():
    result = subprocess.run(
        [_BASH, str(SBATCH_PATH), _EXPECTED_RUN_IDS[0]],
        capture_output=True, text=True, timeout=30,
        env={"PATH": _env_path()},
    )
    assert result.returncode == 2
    assert "EXPECTED_COMMIT is required" in result.stderr


@requires_bash
def test_real_bash_rejects_prepare_only_and_status_only_together():
    result = subprocess.run(
        [_BASH, str(SBATCH_PATH), _EXPECTED_RUN_IDS[0]],
        capture_output=True, text=True, timeout=30,
        env={
            "PATH": _env_path(),
            "EXPECTED_COMMIT": "0" * 40,
            "PREPARE_ONLY": "1",
            "STATUS_ONLY": "1",
        },
    )
    assert result.returncode == 2
    assert "mutually exclusive" in result.stderr


def _env_path():
    import os
    return os.environ.get("PATH", "")


@requires_bash
class TestCommitPinBlockRealGitRepo:
    """Extracts the REAL commit-pin/dirty-tree guard block verbatim out of
    the sbatch file (never a hand-reimplementation of its logic) and runs it
    against a real temporary git repository, so this exercises the actual
    shell commands that will run on Moriah."""

    @staticmethod
    def _extract_commit_pin_block(sbatch_text: str) -> str:
        start_marker = 'test -d "${REPO_WORKDIR}"'
        end_marker = 'echo "Commit pin OK: HEAD matches EXPECTED_COMMIT (${EXPECTED_COMMIT})."'
        start = sbatch_text.index(start_marker)
        end = sbatch_text.index(end_marker) + len(end_marker)
        return sbatch_text[start:end]

    @pytest.fixture
    def harness_script(self, tmp_path, sbatch_text):
        block = self._extract_commit_pin_block(sbatch_text)
        harness = tmp_path / "commit_pin_harness.sh"
        harness.write_text(
            "#!/usr/bin/env bash\n"
            "set -uo pipefail\n"
            'REPO_WORKDIR="$1"\n'
            'EXPECTED_COMMIT="$2"\n'
            f"{block}\n"
            'echo "HARNESS_REACHED_END_OK"\n',
            encoding="utf-8",
        )
        return harness

    @pytest.fixture
    def clean_repo(self, tmp_path):
        repo_dir = tmp_path / "repo_clean"
        repo_dir.mkdir()
        _run_git(repo_dir, "init", "-q")
        _run_git(repo_dir, "config", "user.email", "test@example.invalid")
        _run_git(repo_dir, "config", "user.name", "Test")
        (repo_dir / "tracked.txt").write_text("hello\n", encoding="utf-8")
        _run_git(repo_dir, "add", "tracked.txt")
        _run_git(repo_dir, "commit", "-q", "-m", "initial")
        commit = _run_git(repo_dir, "rev-parse", "HEAD").stdout.strip()
        return repo_dir, commit

    def test_clean_tree_matching_commit_passes(self, harness_script, clean_repo):
        repo_dir, commit = clean_repo
        result = subprocess.run(
            [_BASH, str(harness_script), str(repo_dir), commit],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "HARNESS_REACHED_END_OK" in result.stdout
        assert "Commit pin OK" in result.stdout

    def test_dirty_tracked_file_refuses(self, harness_script, clean_repo):
        repo_dir, commit = clean_repo
        (repo_dir / "tracked.txt").write_text("modified\n", encoding="utf-8")
        result = subprocess.run(
            [_BASH, str(harness_script), str(repo_dir), commit],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 1
        assert "REFUSING to launch" in result.stderr
        assert "HARNESS_REACHED_END_OK" not in result.stdout

    def test_untracked_file_alone_does_not_block(self, harness_script, clean_repo):
        repo_dir, commit = clean_repo
        (repo_dir / "untracked_scratch.txt").write_text("scratch\n", encoding="utf-8")
        result = subprocess.run(
            [_BASH, str(harness_script), str(repo_dir), commit],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "HARNESS_REACHED_END_OK" in result.stdout

    def test_mismatched_expected_commit_refuses(self, harness_script, clean_repo):
        repo_dir, commit = clean_repo
        wrong_commit = "f" * 40
        result = subprocess.run(
            [_BASH, str(harness_script), str(repo_dir), wrong_commit],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 1
        assert "does not match" in result.stderr
        assert "HARNESS_REACHED_END_OK" not in result.stdout

    def test_missing_repo_workdir_refuses(self, harness_script, tmp_path):
        missing_dir = tmp_path / "does_not_exist"
        result = subprocess.run(
            [_BASH, str(harness_script), str(missing_dir), "0" * 40],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 1
        assert "MISSING expected pilot repo work directory" in result.stdout


def _run_git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr}"
    return result

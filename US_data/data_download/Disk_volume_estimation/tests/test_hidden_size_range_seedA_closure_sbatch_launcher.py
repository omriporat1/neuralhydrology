"""Focused tests for scripts/run_stage1_hidden_size_range_seedA_closure_moriah.sbatch
-- the Hidden-size-A hidden-size range-characterization campaign Slurm launcher.

This script's #SBATCH job is never submitted or executed by any test. Checks
here are either purely static/structural on the committed script text, a
`bash -n` syntax check, or -- for the five-run_id allowlist, the reused-
reference guard, the required EXPECTED_COMMIT input, the commit-pin
refusal/dirty-tree-refusal behavior, and the strict W&B launch contract --
standalone execution of just the relevant extracted bash fragment (never
Slurm, never GPU, never NH), mirroring the convention established in
tests/test_lr_range_seedA_closure_sbatch_launcher.py.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SBATCH_PATH = REPO_ROOT / "scripts" / "run_stage1_hidden_size_range_seedA_closure_moriah.sbatch"

H64 = "emb128x32_seedA_h64_lr3em4_cap25k_cal"
H128 = "emb128x32_seedA_h128_lr3em4_cap25k_cal"
H256 = "emb128x32_seedA_h256_lr3em4_cap25k_cal"
H512 = "emb128x32_seedA_h512_lr3em4_cap25k_cal"
REFERENCE = "emb128x32_seedA_lr3em4_cap25k_cal"
ALL_FIVE = [H64, H128, H256, H512, REFERENCE]

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


def test_sbatch_script_delegates_to_hidden_size_closure_cli_not_other_pilot_clis():
    text = _text()
    assert "python scripts/run_stage1_hidden_size_range_seedA_closure.py" in text
    code_lines = [
        line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    assert not any("run_stage1_lead06_pilot.py" in line for line in code_lines)
    assert not any("run_stage1_cap50k_closure.py" in line for line in code_lines)
    assert not any("run_stage1_lr_range_seedA_closure.py" in line for line in code_lines)


# --- (5) exact five-entry allowlist (four trainable + one status-only-only) -

def test_sbatch_script_allowlist_is_exactly_the_five_hidden_size_a_run_ids():
    text = _text()
    assert f"{H64}|{H128}|{H256}|{H512}|{REFERENCE}" in text
    assert "raw_seedA" not in text
    assert "emb128x64_seedA" not in text


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
@pytest.mark.parametrize("run_id", ALL_FIVE)
def test_run_id_case_block_accepts_all_five_approved_run_ids(tmp_path, run_id):
    script_path = tmp_path / "run_id_case.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\nset -uo pipefail\n" + _extract_run_id_case_block() + '\necho "RUN_ID_OK=${RUN_ID}"\n',
        encoding="utf-8",
    )
    result = subprocess.run(["bash", str(script_path), run_id], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert f"RUN_ID_OK={run_id}" in result.stdout


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash not available in this environment")
@pytest.mark.parametrize("bad_run_id", [
    "raw_seedA", "emb128x64_seedA", "emb128x32_seedA_h999_lr3em4_cap25k_cal",
    "not_a_real_run_id", "",
])
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


# --- reused-reference guard (STATUS_ONLY only, never trainable) -------------

def _extract_reference_guard_block() -> str:
    text = _text()
    start_marker = f'REFERENCE_RUN_ID="{REFERENCE}"'
    start = text.find(start_marker)
    assert start != -1
    end_marker = "\nfi\n"
    end = text.find(end_marker, start)
    assert end != -1
    return text[start:end + len(end_marker)]


def test_reference_guard_block_is_placed_immediately_after_the_case_block():
    text = _text()
    case_block = _extract_run_id_case_block()
    case_end = text.find(case_block) + len(case_block)
    guard_start = text.find(f'REFERENCE_RUN_ID="{REFERENCE}"')
    assert guard_start != -1
    between = text[case_end:guard_start]
    non_comment_lines = [
        line for line in between.splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    assert non_comment_lines == []


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash not available in this environment")
def test_reference_guard_block_refuses_ordinary_training_of_the_reference(tmp_path):
    script_path = tmp_path / "reference_guard.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\nset -uo pipefail\n"
        f'RUN_ID="{REFERENCE}"\n'
        + _extract_reference_guard_block()
        + '\necho "PASSED"\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("STATUS_ONLY", None)
    result = subprocess.run(["bash", str(script_path)], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "REFUSING to launch" in result.stderr
    assert "never trainable/configurable" in result.stderr
    assert "PASSED" not in result.stdout


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash not available in this environment")
def test_reference_guard_block_allows_status_only_of_the_reference(tmp_path):
    script_path = tmp_path / "reference_guard.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\nset -uo pipefail\n"
        f'RUN_ID="{REFERENCE}"\n'
        + _extract_reference_guard_block()
        + '\necho "PASSED"\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["STATUS_ONLY"] = "1"
    result = subprocess.run(["bash", str(script_path)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "PASSED" in result.stdout


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash not available in this environment")
@pytest.mark.parametrize("run_id", [H64, H128, H256, H512])
def test_reference_guard_block_never_blocks_the_four_real_candidates(tmp_path, run_id):
    script_path = tmp_path / "reference_guard.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\nset -uo pipefail\n"
        f'RUN_ID="{run_id}"\n'
        + _extract_reference_guard_block()
        + '\necho "PASSED"\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("STATUS_ONLY", None)
    result = subprocess.run(["bash", str(script_path)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "PASSED" in result.stdout


def test_reference_run_id_only_appears_paired_with_status_only_in_usage_examples():
    text = _text()
    usage_block_start = text.index("# Usage:")
    usage_block_end = text.index("# --- Commit-pin design")
    usage_lines = text[usage_block_start:usage_block_end].splitlines()
    reference_line_indices = [i for i, line in enumerate(usage_lines) if REFERENCE in line]
    assert reference_line_indices
    for idx in reference_line_indices:
        preceding_and_current = "\n".join(usage_lines[max(0, idx - 1):idx + 1])
        assert "STATUS_ONLY=1" in preceding_and_current


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
        "#!/usr/bin/env bash\nset -uo pipefail\nRUN_ID=test\n"
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
        "#!/usr/bin/env bash\nset -uo pipefail\nRUN_ID=test\n"
        + _extract_expected_commit_block() + '\necho "PASSED"\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["EXPECTED_COMMIT"] = "deadbeefcafef00d"
    result = subprocess.run(["bash", str(script_path)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "PASSED" in result.stdout


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
    (repo_dir / "generated_evidence.json").write_text("{}\n", encoding="utf-8")
    result = _run_commit_pin_block(tmp_path, repo_dir, head)
    assert result.returncode == 0, result.stderr
    assert "Commit pin OK" in result.stdout


# --- absence of --force / wandb sync / HIDDEN_SIZE|LEARNING_RATE override --

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


def test_sbatch_script_exports_offline_mode_unconditionally():
    text = _text()
    export_idx = text.index("export WANDB_MODE=offline")
    policy_default_idx = text.index('WANDB_POLICY_PATH="${WANDB_POLICY_PATH:-')
    assert export_idx < policy_default_idx


def test_sbatch_script_never_uses_gated_wandb_mode_default_pattern():
    code_lines = [
        line for line in _text().splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    assert not any("WANDB_MODE:-offline" in line for line in code_lines)


def test_sbatch_script_never_defines_a_hidden_size_or_learning_rate_override_knob():
    code_lines = [
        line for line in _text().splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    assert not any("LEARNING_RATE" in line for line in code_lines)
    assert not any("--learning-rate" in line for line in code_lines)
    assert not any(line.strip().startswith("HIDDEN_SIZE=") for line in code_lines)
    assert not any("--hidden-size" in line for line in code_lines)


# --- strict W&B launch contract: unconditional offline-policy default, ------
# --- WAIVE_TRACKING_REQUIREMENT default-0 threading -------------------------

def test_wandb_policy_path_default_is_unconditional_not_empty_string_gated():
    text = _text()
    # Unlike LR-A's optional `WANDB_POLICY_PATH="${WANDB_POLICY_PATH:-}"`
    # (empty-string default, conditionally applied later), Hidden-size-A's
    # default must resolve to a real, non-empty policy file path directly in
    # the parameter-expansion default itself.
    assert 'WANDB_POLICY_PATH="${WANDB_POLICY_PATH:-}"' not in text
    assert "WANDB_POLICY_PATH=\"${WANDB_POLICY_PATH:-${REPO_WORKDIR}/config/stage1_wandb_tracking_policy_offline_v001.yaml}\"" in text


def test_wandb_policy_path_always_passed_to_cli_wrapper_unconditionally():
    text = _text()
    # --wandb-policy-path must be inside the CLOSURE_CLI_ARGS base array
    # (before any conditional `if` branch appends to it), not gated behind
    # a `[ -n "${WANDB_POLICY_PATH}" ]` check the way LR-A's optional
    # override was.
    args_block_start = text.index("CLOSURE_CLI_ARGS=(")
    args_block_end = text.index(")\n", args_block_start)
    base_args_block = text[args_block_start:args_block_end]
    assert "--wandb-policy-path" in base_args_block
    assert 'if [ -n "${WANDB_POLICY_PATH}" ]; then' not in text


def test_waive_tracking_requirement_defaults_to_0():
    text = _text()
    assert 'WAIVE_TRACKING_REQUIREMENT="${WAIVE_TRACKING_REQUIREMENT:-0}"' in text


def _extract_waive_tracking_append_block() -> str:
    text = _text()
    start_marker = 'if [ "${WAIVE_TRACKING_REQUIREMENT}" = "1" ]; then'
    start = text.find(start_marker)
    assert start != -1
    end_marker = "\nfi\n"
    end = text.find(end_marker, start)
    assert end != -1
    return text[start:end + len(end_marker)]


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash not available in this environment")
def test_waive_tracking_requirement_flag_appended_only_when_set_to_1(tmp_path):
    script_path = tmp_path / "waive.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\nset -uo pipefail\n"
        "CLOSURE_CLI_ARGS=(--run-id fake)\n"
        + _extract_waive_tracking_append_block()
        + '\necho "ARGS=${CLOSURE_CLI_ARGS[@]}"\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["WAIVE_TRACKING_REQUIREMENT"] = "1"
    result = subprocess.run(["bash", str(script_path)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--waive-tracking-requirement" in result.stdout


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash not available in this environment")
def test_waive_tracking_requirement_flag_absent_by_default(tmp_path):
    script_path = tmp_path / "waive.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\nset -uo pipefail\n"
        'WAIVE_TRACKING_REQUIREMENT="${WAIVE_TRACKING_REQUIREMENT:-0}"\n'
        "CLOSURE_CLI_ARGS=(--run-id fake)\n"
        + _extract_waive_tracking_append_block()
        + '\necho "ARGS=${CLOSURE_CLI_ARGS[@]}"\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("WAIVE_TRACKING_REQUIREMENT", None)
    result = subprocess.run(["bash", str(script_path)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--waive-tracking-requirement" not in result.stdout


def test_sbatch_script_echoes_wandb_policy_path_and_waiver_in_info_block():
    text = _text()
    policy_echo_idx = text.index('echo "WANDB_POLICY_PATH (effective): ${WANDB_POLICY_PATH}"')
    waiver_echo_idx = text.index('echo "WAIVE_TRACKING_REQUIREMENT: ${WAIVE_TRACKING_REQUIREMENT}"')
    prepare_only_branch_idx = text.index('if [ "${PREPARE_ONLY}" = "1" ] || [ "${STATUS_ONLY}" = "1" ]; then')
    assert policy_echo_idx < prepare_only_branch_idx
    assert waiver_echo_idx < prepare_only_branch_idx


# --- epoch-6 Hidden-size-A horizon: fixed, non-overridable ------------------

def test_sbatch_script_defines_fixed_hidden_size_a_max_target_epoch_constant():
    text = _text()
    assert "HIDDEN_SIZE_A_MAX_TARGET_EPOCH=6" in text
    code_lines = [
        line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    assert not any('MAX_TARGET_EPOCH="${MAX_TARGET_EPOCH' in line for line in code_lines)
    assert not any("--max-target-epoch" in line for line in code_lines)


def test_sbatch_script_echoes_hidden_size_a_horizon_in_always_printed_info_block():
    text = _text()
    echo_idx = text.index(
        'echo "Approved Hidden-size-A horizon (fixed, non-overridable): epoch ${HIDDEN_SIZE_A_MAX_TARGET_EPOCH}"'
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

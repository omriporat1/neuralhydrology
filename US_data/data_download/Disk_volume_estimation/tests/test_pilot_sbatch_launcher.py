"""Focused tests for scripts/run_stage1_lead06_pilot_moriah.sbatch (task
item 7's launcher, task item 10's "generated paths outside tracked dirs"
checklist item).

This script's #SBATCH job is never submitted or executed by any test (per
the task's binding remote-access constraints) -- most checks here are
purely static/structural checks on the committed script text, plus a
`bash -n` syntax check (skipped if bash is unavailable in the test
environment). The status-classification tests below are the one exception:
they extract the second embedded ``python -c "..."`` block (the pure
on-disk/stdout status-derivation snippet, no Slurm/GPU/NH involved) and run
it standalone against constructed env vars -- this never touches Slurm and
is the only way to behaviorally prove the launcher's own status
classification (see job 45718473, docs/decision_log.md), not just that
certain substrings are present in the script text.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tests._pilot_support import REPO_ROOT

SBATCH_PATH = REPO_ROOT / "scripts" / "run_stage1_lead06_pilot_moriah.sbatch"

ALL_RUN_IDS = ("raw_seedA", "raw_seedB", "emb128x64_seedA", "emb128x64_seedB", "emb64_seedA", "emb128_seedA")


def _text() -> str:
    return SBATCH_PATH.read_text(encoding="utf-8")


def _extract_status_classification_block() -> str:
    """Pull out the second ``python -c "..."`` block in the sbatch script
    (the status-derivation/classification snippet run after the pilot CLI
    exits -- the first ``python -c "..."`` block is the earlier, unrelated
    NH_RUN_DIR discovery snippet)."""
    text = _text()
    marker = 'python -c "'
    first = text.find(marker)
    second = text.find(marker, first + 1)
    assert second != -1, "expected two python -c blocks in the sbatch script"
    start = second + len(marker)
    end = text.find('\n" | tee', start)
    assert end != -1, "could not find the end of the status-classification python -c block"
    return text[start:end]


def _run_status_classification_block(tmp_path: Path, env_overrides: dict) -> dict:
    """Execute the extracted status-classification snippet as a standalone
    script with the given environment, returning the parsed
    ``pilot_result.json``-equivalent dict it prints. Never invokes bash,
    sbatch, or Slurm -- only the pure-Python fragment."""
    import os

    block_path = tmp_path / "status_block.py"
    block_path.write_text(_extract_status_classification_block(), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.update(env_overrides)

    proc = subprocess.run(
        [sys.executable, str(block_path)], env=env, capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert proc.returncode == 0, proc.stderr
    json_text = proc.stdout.split("\nRUN_STATUS_FOR_SHELL")[0]
    return json.loads(json_text)


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


def test_sbatch_script_finds_checkpoints_recursively_not_just_maxdepth_1():
    # second real Moriah failure (job 45705457): a -maxdepth 1 find missed
    # checkpoints written into a nested continue_training_from_epoch###/
    # continuation directory and so reported a stale "latest epoch". The
    # unrelated -mindepth 1 -maxdepth 1 find just above (locating the NH
    # run directory itself among sibling run directories, never a
    # checkpoint file search) is untouched and deliberately excluded here.
    text = _text()
    assert "find \"${NH_RUN_DIR}\" -name 'model_epoch*.pt'" in text
    assert "find \"${NH_RUN_DIR}\" -maxdepth 1 -name 'model_epoch*.pt'" not in text


def test_sbatch_script_sorts_latest_checkpoint_numerically_not_alphabetically():
    text = _text()
    # extracts the numeric epoch as an explicit sort key (sort -n) rather
    # than relying on path-string ordering, which is not epoch-numeric
    # across differently-nested continuation directories.
    assert "sort -n" in text


def test_sbatch_script_reports_distinct_physical_vs_screened_vs_next_intended_fields():
    text = _text()
    for field in (
        "highest_physical_checkpoint_epoch",
        "highest_screened_epoch",
        "next_intended_screening_epoch",
        "overshoot_epochs",
        "safe_to_continue_automatically",
        "blocked_reason",
    ):
        assert field in text, f"{field} not reported by the sbatch script"


def test_sbatch_script_has_a_distinct_blocked_status_not_folded_into_completed():
    text = _text()
    assert "BLOCKED_MANUAL_REVIEW_REQUIRED" in text
    assert "blocked_continuation_overshoot_conflict" in text


def test_sbatch_script_status_fallback_uses_compute_pilot_status_fields_not_a_reimplementation():
    text = _text()
    # falls back to the single canonical resolver rather than re-deriving
    # highest-screened/overshoot logic a second time in shell/python here.
    assert "from src.baseline.pilot_orchestration import compute_pilot_status_fields" in text


def test_sbatch_script_status_json_fields_passed_via_environment_not_string_interpolation():
    text = _text()
    # a free-text blocked_reason could contain quotes/newlines; must never
    # be spliced directly into the embedded Python source as a string
    # literal (the earlier fields like run_id/nh_run_dir were interpolated
    # this way, which is what this new block deliberately avoids for the
    # new free-text field).
    assert "env['blocked_reason']" not in text
    assert "os.environ" in text


# --- status-classification behavior (job 45718473) --------------------------
#
# Real Moriah recovery job 45718473 (commit fbf7eea): the pilot correctly
# reused the trusted epoch-9 checkpoint and refused to advance into the
# untrusted 10-15 overshoot range (pilot_orchestration.run_pilot() itself
# returns "final_status": "blocked_continuation_overshoot_conflict" and a
# non-null "blocked_reason" for this exact scenario -- see
# test_run_pilot_end_to_end_propagates_blocked_continuation_overshoot_conflict
# in tests/test_pilot_orchestration.py). But the launcher's own primary
# pilot_stdout.json.log was empty when read, so its status-derivation
# fallback (computed directly from on-disk state) restored
# overshoot_epochs/safe_to_continue_automatically but left
# pilot_final_status/blocked_reason as None -- and the classification below
# only checked pilot_final_status, so a blocked run was reported as
# COMPLETED. The tests below exercise the actual extracted classification
# snippet (see _run_status_classification_block above), not just its text.

def _base_env(tmp_path: Path, *, nh_run_dir: Path, stdout_json_path: Path, latest_ckpt: str, latest_epoch: str):
    return {
        "RUN_ID": "raw_seedA",
        "NH_RUN_DIR": str(nh_run_dir),
        "STDOUT_JSON_PATH": str(stdout_json_path),
        "PILOT_STATUS": "0",
        "PACKAGE_ROOT": "/fake/package",
        "CONFIG_OUT_DIR": "/fake/config_out",
        "EVIDENCE_OUT_DIR": "/fake/evidence_out",
        "LATEST_CKPT_AFTER": latest_ckpt,
        "LATEST_EPOCH_AFTER": latest_epoch,
        "TERM_REQUESTED": "0",
        "SLURM_JOB_ID": "45718473",
        "SLURM_JOB_PARTITION": "catfish",
        "SOURCE_COMMIT": "fbf7eea",
        "RESULT_JSON_PATH": str(tmp_path / "pilot_result.json"),
    }


def test_sbatch_status_block_reports_blocked_not_completed_when_stdout_empty_but_overshoot_on_disk(tmp_path):
    """Job 45718473's exact shape: primary stdout JSON empty, but on-disk
    state (checkpoints 1-6 flat, 7-15 in a continue_training_from_epoch006/
    continuation directory, only epochs 6 and 9 actually screened) shows an
    untrusted overshoot. The launcher must classify this as
    BLOCKED_MANUAL_REVIEW_REQUIRED with a non-null pilot_final_status and
    blocked_reason -- never COMPLETED."""
    nh_run_dir = tmp_path / "nh_run"
    nh_run_dir.mkdir()
    for epoch in range(1, 7):
        (nh_run_dir / f"model_epoch{epoch:03d}.pt").write_bytes(b"x")
    cont_dir = nh_run_dir / "continue_training_from_epoch006"
    cont_dir.mkdir()
    for epoch in range(7, 16):
        (cont_dir / f"model_epoch{epoch:03d}.pt").write_bytes(b"x")
    (nh_run_dir / "pilot_early_stopping_state.json").write_text(
        json.dumps({"history": [{"epoch": 6, "median": 0.2}, {"epoch": 9, "median": 0.18}], "stopped": False})
    )

    stdout_json_path = tmp_path / "pilot_stdout.json.log"
    stdout_json_path.write_text("")  # primary stdout unavailable, exactly as observed

    result = _run_status_classification_block(
        tmp_path,
        _base_env(
            tmp_path,
            nh_run_dir=nh_run_dir,
            stdout_json_path=stdout_json_path,
            latest_ckpt=str(cont_dir / "model_epoch015.pt"),
            latest_epoch="15",
        ),
    )

    assert result["status"] == "BLOCKED_MANUAL_REVIEW_REQUIRED"
    assert result["status"] != "COMPLETED"
    assert result["pilot_final_status"] == "blocked_continuation_overshoot_conflict"
    assert result["blocked_reason"] is not None
    assert result["overshoot_epochs"] == [10, 11, 12, 13, 14, 15]
    assert result["safe_to_continue_automatically"] is False


def test_sbatch_status_block_ordinary_completed_run_remains_completed(tmp_path):
    """Regression: a genuinely completed/stopped run (primary stdout JSON
    present and populated, no overshoot) must remain classified COMPLETED --
    the new fallback-derived blocked classification must not fire when the
    primary status is already known."""
    nh_run_dir = tmp_path / "nh_run"
    nh_run_dir.mkdir()

    stdout_json_path = tmp_path / "pilot_stdout.json.log"
    stdout_json_path.write_text(
        json.dumps(
            {
                "run_id": "raw_seedA",
                "final_status": "stopped_patience_exhausted",
                "best_checkpoint_epoch": 6,
                "nh_run_dir": str(nh_run_dir),
                "evidence_bundle_path": str(tmp_path / "evidence"),
                "highest_physical_checkpoint_epoch": 15,
                "highest_screened_epoch": 15,
                "next_intended_screening_epoch": None,
                "overshoot_epochs": [],
                "safe_to_continue_automatically": True,
                "blocked_reason": None,
            }
        )
    )

    result = _run_status_classification_block(
        tmp_path,
        _base_env(
            tmp_path, nh_run_dir=nh_run_dir, stdout_json_path=stdout_json_path, latest_ckpt="", latest_epoch=""
        ),
    )

    assert result["status"] == "COMPLETED"
    assert result["pilot_final_status"] == "stopped_patience_exhausted"

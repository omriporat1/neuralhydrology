"""Structural checks for the unsubmitted Attempt-2 combined retry launcher."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RETRY = REPO_ROOT / "scripts" / "run_phase_b_batch_size_operational_retry_moriah.sbatch"
SINGLE = REPO_ROOT / "scripts" / "run_phase_b_batch_size_operational_qualification_moriah.sbatch"
PREPARER = REPO_ROOT / "scripts" / "prepare_phase_b_batch_size_operational_qualification.py"


def test_retry_is_one_l4_allocation_with_deterministic_fresh_batch_processes():
    text = RETRY.read_text(encoding="utf-8")
    assert 'readonly BATCH_SIZES=(128 256 512)' in text
    assert "#SBATCH --partition=catfish" in text
    assert "#SBATCH --gres=gpu:l4:1" in text
    assert "#SBATCH --cpus-per-task=8" in text
    assert "#SBATCH --mem=64G" in text
    assert "#SBATCH --time=00:30:00" in text
    assert 'bash "${SINGLE_BATCH_LAUNCHER}" "${batch_size}"' in text
    assert "BATCH_EXIT_STATUS" in text
    assert "set +e" in text and "set -e" in text
    assert "PARTIAL_FAILURE_OR_FAIL" in text
    assert "attempt2_combined_sequential_operational_retry_v001" in text
    executable_lines = [line for line in text.splitlines() if not line.startswith("#")]
    assert not any(re.match(r"^\s*sbatch\s", line) for line in executable_lines)


def test_retry_preserves_separate_attempt_and_batch_evidence_identities():
    text = RETRY.read_text(encoding="utf-8")
    assert "OPQUAL_ATTEMPT2_EVIDENCE_ROOT" in text
    assert "phase_b_batch_size_operational_qualification_attempt2_combined_v001" in text
    assert 'export OPQUAL_ATTEMPT_ID="${ATTEMPT_ID}"' in text
    assert 'f"bs{batch_size}_job{os.environ.get(\'SLURM_JOB_ID\', \'unknown\')}"' in text
    assert "attempt2_combined_summary.json" in text
    assert "intended_config_sha256" in text
    assert "operational_config_sha256" in text
    assert "phase_b_batch_size_operational_qualification_only" not in text


def test_retry_retains_guard_and_reuses_approved_smoke_contract_without_metrics():
    retry = RETRY.read_text(encoding="utf-8")
    single = SINGLE.read_text(encoding="utf-8")
    preparer = PREPARER.read_text(encoding="utf-8")
    assert 'REPO_CLONE_DIR="${FLASHNH_REPO_CLONE_DIR:-${FLASHNH_BASE}/repos/flash-nh}"' in retry
    assert 'git -C "${REPO_WORKDIR}" rev-parse --show-toplevel' in retry
    assert 'git -C "${RESOLVED_REPOSITORY}" rev-parse HEAD' in retry
    assert 'test "${ACTUAL_COMMIT}" = "${EXPECTED_COMMIT}"' in retry
    assert "ACTUAL_COMMIT does not match EXPECTED_COMMIT" in retry
    assert "OPERATIONAL_SMOKE_UPDATES = 8" in preparer
    assert "SWEEP_V1_UPDATES_PER_EPOCH = 50_000" in preparer
    assert "start_run(config_file=config)" in single
    for forbidden in ("wandb", "NSE", "KGE"):
        assert forbidden.lower() not in retry.lower()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_retry_bash_syntax_is_valid():
    result = subprocess.run(["bash", "-n", str(RETRY)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

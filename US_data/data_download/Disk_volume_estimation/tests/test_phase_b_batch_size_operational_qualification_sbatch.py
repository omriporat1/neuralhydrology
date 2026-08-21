"""Static safety checks for the unsubmitted Phase-B operational launcher."""
from __future__ import annotations

import shutil
import subprocess
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SBATCH = REPO_ROOT / "scripts" / "run_phase_b_batch_size_operational_qualification_moriah.sbatch"


def test_launcher_has_expected_l4_scope_and_no_wandb_or_submission():
    text = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:l4:1" in text
    assert 'case "${BATCH_SIZE}" in 128|256|512)' in text
    assert "EXPECTED_COMMIT" in text
    assert "OPERATIONAL QUALIFICATION ONLY" in text
    assert "launcher_python_exit_status" in text
    assert "peak_gpu_memory_bytes" in text
    assert 'REPO_CLONE_DIR="${FLASHNH_REPO_CLONE_DIR:-${FLASHNH_BASE}/repos/flash-nh}"' in text
    assert "/sci/labs/efratmorin/omripo/PhD/Python/neuralhydrology" not in text
    assert 'git -C "${REPO_WORKDIR}" rev-parse --show-toplevel' in text
    assert 'git -C "${RESOLVED_REPOSITORY}" rev-parse HEAD' in text
    for diagnostic in ("SUBMIT_WORKDIR:", "RESOLVED_REPOSITORY:", "EXPECTED_COMMIT:", "ACTUAL_COMMIT:"):
        assert diagnostic in text
    assert "ACTUAL_COMMIT does not match EXPECTED_COMMIT" in text
    assert "WANDB" not in text
    executable_lines = [line for line in text.splitlines() if not line.startswith("#")]
    assert not any(re.match(r"^\s*sbatch\s", line) for line in executable_lines)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_launcher_bash_syntax_is_valid():
    result = subprocess.run(["bash", "-n", str(SBATCH)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

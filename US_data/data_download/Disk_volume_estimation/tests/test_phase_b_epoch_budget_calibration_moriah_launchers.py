"""Static safety tests for the unsubmitted Phase-B Moriah launch layer."""
from __future__ import annotations

import shutil
import subprocess
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PREFLIGHT = ROOT / "scripts" / "run_phase_b_epoch_budget_calibration_preflight_moriah.sbatch"
TRAINING = ROOT / "scripts" / "run_phase_b_epoch_budget_calibration_moriah.sbatch"
PYTHON = "/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah/bin/python"


@pytest.mark.parametrize("launcher", [PREFLIGHT, TRAINING])
def test_launchers_use_commit_guard_and_exact_moriah_python(launcher):
    text = launcher.read_text(encoding="utf-8")
    assert "EXPECTED_COMMIT" in text
    assert PYTHON in text
    assert "PILOT_PACKAGE_ROOT" in text
    executable = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    assert not re.search(r"^\s*sbatch\s", executable, flags=re.MULTILINE)


def test_preflight_is_cpu_only_and_generates_all_configs():
    text = PREFLIGHT.read_text(encoding="utf-8")
    assert "#SBATCH --partition=glacier" in text and "#SBATCH --gres" not in text
    assert "prepare_phase_b_epoch_budget_calibration.py" in text
    assert "--canonical-package-preflight" in text
    assert "from neuralhydrology.nh_run" not in text and "torch.cuda.is_initialized" in text
    assert "canonical_preflight_summary.json" in text
    assert "PHASEB_SCREENING_ARTIFACT" in text
    assert "d4395d93ebc567cf09e149c0121463d75cf4f7ecc02c07a7c4a7999763baa372" in text
    assert "--screening-artifact-path" in text


def test_training_launcher_is_exactly_one_frozen_candidate_and_no_wandb():
    text = TRAINING.read_text(encoding="utf-8")
    assert "#SBATCH --partition=catfish" in text and "#SBATCH --gres=gpu:l4:1" in text
    assert "C1_anchor|C2_low_lr|C3_high_lr|C4_late_h64|C5_convergence_stress" in text
    assert "W&B" in text and "require_tracking" not in text
    assert "PHASEB_SCREENING_ARTIFACT" in text


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
@pytest.mark.parametrize("launcher", [PREFLIGHT, TRAINING])
def test_launcher_bash_syntax(launcher):
    result = subprocess.run(["bash", "-n", str(launcher)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

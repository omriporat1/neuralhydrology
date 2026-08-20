"""Preparation invariants for the Phase-B batch-size operational smoke only."""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "prepare_phase_b_batch_size_operational_qualification.py"


def _load_module_constants():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return values


def test_qualification_scope_is_exact_and_non_scientific():
    values = _load_module_constants()
    assert values["APPROVED_BATCH_SIZES"] == (128, 256, 512)
    assert values["HIDDEN_SIZE_STRESS_POINT"] == 256
    assert values["SWEEP_V1_UPDATES_PER_EPOCH"] == 50_000
    assert values["OPERATIONAL_SMOKE_UPDATES"] == 8
    assert values["OPERATIONAL_SMOKE_EPOCHS"] == 1
    assert "OPERATIONAL QUALIFICATION ONLY" in SCRIPT.read_text(encoding="utf-8")


def test_cli_only_accepts_the_three_provisional_batch_sizes(tmp_path):
    common = [sys.executable, str(SCRIPT), "--package-root", str(tmp_path), "--out-dir", str(tmp_path / "out")]
    result = subprocess.run([*common, "--batch-size", "64"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_smoke_mapping_never_validates_or_uses_the_sweep_cap():
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'mapping.update({"epochs": OPERATIONAL_SMOKE_EPOCHS, "validate_every": 100' in text
    assert "max_updates_per_epoch=OPERATIONAL_SMOKE_UPDATES" in text
    assert "max_updates_per_epoch=SWEEP_V1_UPDATES_PER_EPOCH" in text

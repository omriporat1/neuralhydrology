"""Vertical, offline-only contract tests for the v2 rehearsal serializer."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from src.baseline.sweep_v2_six_axis_config import (
    V2_METRIC_NAME,
    V2_REHEARSAL_PLACEHOLDER_METRIC_NAME,
    build_wandb_bridge_rehearsal_sweep_config_v2,
)


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "build_sweep_v2_six_axis_wandb_bridge_rehearsal_sweep_config.py"


def _run(*arguments: str, poison_wandb_at: Path | None = None) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    python_path = [str(ROOT)]
    if poison_wandb_at is not None:
        python_path.insert(0, str(poison_wandb_at))
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_serializer_subprocess_writes_exact_authoritative_rehearsal_config(tmp_path):
    output = tmp_path / "nested" / "config.json"
    manifest = tmp_path / "future-rehearsal-manifest.json"
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "wandb.py").write_text("raise AssertionError('wandb import is forbidden')\n", encoding="utf-8")

    result = _run(
        "--output", str(output), "--manifest-path", str(manifest), poison_wandb_at=poison,
    )

    assert result.returncode == 0, result.stderr
    serialized = json.loads(output.read_text(encoding="utf-8"))
    assert serialized == build_wandb_bridge_rehearsal_sweep_config_v2(
        program="scripts/run_sweep_v2_six_axis_wandb_bridge.py",
        manifest_path=str(manifest),
    )
    assert output.read_bytes().endswith(b"\n")
    assert serialized["method"] == "bayes"
    assert set(serialized["parameters"]) == {
        "learning_rate", "hidden_size", "embedding_dropout", "output_dropout", "batch_size", "seq_length",
    }
    assert serialized["parameters"]["seq_length"] == {
        "distribution": "q_uniform", "min": 48, "max": 120, "q": 12,
    }
    assert serialized["metric"]["name"] == V2_REHEARSAL_PLACEHOLDER_METRIC_NAME
    assert serialized["metric"]["name"] != V2_METRIC_NAME
    assert serialized["command"] == ["${interpreter}", "${program}", str(manifest)]
    assert "4x3btz2s" not in json.dumps(serialized)


def test_serializer_rejects_relative_manifest_before_creating_output(tmp_path):
    output = tmp_path / "would-be-created" / "config.json"

    result = _run("--output", str(output), "--manifest-path", "relative/manifest.json")

    assert result.returncode != 0
    assert "--manifest-path must be absolute" in result.stderr
    assert not output.exists()
    assert not output.parent.exists()


def test_serializer_preserves_an_absolute_moriah_manifest_path(tmp_path):
    output = tmp_path / "config.json"
    manifest = "/sci/labs/efratmorin/omripo/Flash-NH/rehearsal-manifest.json"

    result = _run("--output", str(output), "--manifest-path", manifest)

    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["command"][-1] == manifest


def test_serializer_refuses_existing_file_without_modifying_bytes(tmp_path):
    output = tmp_path / "config.json"
    original = b"do not replace\r\n"
    output.write_bytes(original)

    result = _run("--output", str(output), "--manifest-path", str(tmp_path / "manifest.json"))

    assert result.returncode != 0
    assert output.read_bytes() == original


def test_serializer_refuses_existing_directory_without_modifying_it(tmp_path):
    output = tmp_path / "existing-directory"
    output.mkdir()

    result = _run("--output", str(output), "--manifest-path", str(tmp_path / "manifest.json"))

    assert result.returncode != 0
    assert output.is_dir()
    assert list(output.iterdir()) == []


def test_serializer_source_never_imports_wandb():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import wandb" not in source
    assert "wandb.sweep" not in source

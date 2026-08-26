"""Tests for the disposable REHEARSAL Sweep-v1 W&B sweep config builder,
``src.baseline.sweep_v1_execution.build_wandb_bridge_rehearsal_sweep_config``
and its CLI wrapper
``scripts/build_sweep_v1_wandb_bridge_rehearsal_sweep_config.py`` -- the
rehearsal sibling of ``tests/test_sweep_v1_launch_command_contract.py``'s
sections 1-2 for the production sweep config.

Covers: the config keeps the same five-axis parameter domain as production;
the disposable metric name never collides with the production metric name;
`command` carries exactly one extra static positional token (the rehearsal
manifest path) beyond the interpreter/program pair, still with no
``${args}``/``${env}`` macro; the CLI wrapper serializes this correctly and
enforces the same no-silent-overwrite safety as the production builder.

Never imports wandb, never touches the network, never starts training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.build_sweep_v1_wandb_bridge_rehearsal_sweep_config import main as build_rehearsal_config_main
from src.baseline.sweep_v1_execution import (
    build_production_sweep_config, build_wandb_bridge_rehearsal_sweep_config,
)

ROOT = Path(__file__).resolve().parent.parent
BUILD_CONFIG_SCRIPT = ROOT / "scripts" / "build_sweep_v1_wandb_bridge_rehearsal_sweep_config.py"

_MANIFEST_PATH = "/sci/labs/efratmorin/omripo/Flash-NH/repos/flash-nh/.scratch_local/rehearsal_manifest.json"


# --- 1. rehearsal sweep-config shape ----------------------------------------

def test_rehearsal_sweep_config_reuses_production_parameter_domain():
    production = build_production_sweep_config(program="scripts/run_sweep_v1_wandb_bridge.py")
    rehearsal = build_wandb_bridge_rehearsal_sweep_config(
        program="scripts/run_sweep_v1_wandb_bridge.py", manifest_path=_MANIFEST_PATH,
    )
    assert rehearsal["parameters"] == production["parameters"]
    assert rehearsal["method"] == production["method"] == "bayes"


def test_rehearsal_sweep_config_uses_a_disposable_metric_distinct_from_production():
    production = build_production_sweep_config(program="scripts/run_sweep_v1_wandb_bridge.py")
    rehearsal = build_wandb_bridge_rehearsal_sweep_config(
        program="scripts/run_sweep_v1_wandb_bridge.py", manifest_path=_MANIFEST_PATH,
    )
    assert rehearsal["metric"]["name"] != production["metric"]["name"]
    assert rehearsal["metric"]["name"] == "qualification/rehearsal_placeholder_metric"
    assert rehearsal["metric"]["goal"] == "maximize"


def test_rehearsal_sweep_config_command_carries_exactly_one_extra_static_manifest_token():
    rehearsal = build_wandb_bridge_rehearsal_sweep_config(
        program="scripts/run_sweep_v1_wandb_bridge.py", manifest_path=_MANIFEST_PATH,
    )
    assert rehearsal["command"] == ["${interpreter}", "${program}", _MANIFEST_PATH]
    serialized = json.dumps(rehearsal["command"])
    assert "${args}" not in serialized
    assert "${env}" not in serialized


def test_rehearsal_sweep_config_manifest_path_is_absolute_and_literal():
    relative_path = "relative/manifest.json"
    # The builder itself does not validate absoluteness (that is the
    # manifest schema's job at load time); this test documents that the
    # config faithfully embeds whatever string it is given, so callers must
    # pass an already-resolved absolute path.
    rehearsal = build_wandb_bridge_rehearsal_sweep_config(
        program="scripts/run_sweep_v1_wandb_bridge.py", manifest_path=relative_path,
    )
    assert rehearsal["command"][-1] == relative_path


# --- 2. CLI wrapper: real serialization + no-overwrite safety ---------------

def test_rehearsal_builder_script_serializes_real_config_with_exact_command(tmp_path, monkeypatch):
    output = tmp_path / "rehearsal_sweep_config.json"
    monkeypatch.setattr(
        sys, "argv",
        ["build_sweep_v1_wandb_bridge_rehearsal_sweep_config.py", "--output", str(output),
         "--manifest-path", _MANIFEST_PATH],
    )
    build_rehearsal_config_main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["command"] == ["${interpreter}", "${program}", _MANIFEST_PATH]
    assert payload["program"] == "scripts/run_sweep_v1_wandb_bridge.py"
    assert payload["metric"]["name"] == "qualification/rehearsal_placeholder_metric"


def test_rehearsal_builder_script_refuses_to_silently_overwrite(tmp_path, monkeypatch):
    output = tmp_path / "rehearsal_sweep_config.json"
    monkeypatch.setattr(
        sys, "argv",
        ["build_sweep_v1_wandb_bridge_rehearsal_sweep_config.py", "--output", str(output),
         "--manifest-path", _MANIFEST_PATH],
    )
    build_rehearsal_config_main()
    first_content = output.read_text(encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv",
        ["build_sweep_v1_wandb_bridge_rehearsal_sweep_config.py", "--output", str(output),
         "--manifest-path", _MANIFEST_PATH + ".other"],
    )
    with pytest.raises(SystemExit):
        build_rehearsal_config_main()
    assert output.read_text(encoding="utf-8") == first_content


def test_rehearsal_builder_script_force_flag_permits_deliberate_overwrite(tmp_path, monkeypatch):
    output = tmp_path / "rehearsal_sweep_config.json"
    monkeypatch.setattr(
        sys, "argv",
        ["build_sweep_v1_wandb_bridge_rehearsal_sweep_config.py", "--output", str(output),
         "--manifest-path", _MANIFEST_PATH],
    )
    build_rehearsal_config_main()

    monkeypatch.setattr(
        sys, "argv",
        ["build_sweep_v1_wandb_bridge_rehearsal_sweep_config.py", "--output", str(output),
         "--manifest-path", _MANIFEST_PATH + ".other", "--force"],
    )
    build_rehearsal_config_main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["command"][-1] == _MANIFEST_PATH + ".other"

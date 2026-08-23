"""Tests for scripts/wandb_online_sweep_qualification_lifecycle.py.

Uses a fake in-process ``wandb`` module (for ``wandb.Api().sweep(...).state``
read-only state queries) and a fake CLI executable (a tiny script standing
in for the real ``wandb`` CLI binary) to deterministically exercise the
pause/resume/stop/status state-transition recording without a real network
or the real wandb package/CLI installed.
"""
from __future__ import annotations

import json
import os
import stat
import sys
import types

import pytest

import scripts.wandb_online_sweep_qualification_lifecycle as lifecycle


class _FakeSweep:
    def __init__(self, state: str):
        self.state = state


class _FakeApi:
    def __init__(self, states_by_call):
        self._states = iter(states_by_call)

    def sweep(self, sweep_id):
        return _FakeSweep(next(self._states))


class _FakeWandbModule(types.ModuleType):
    def __init__(self, states_by_call):
        super().__init__("wandb")
        # One shared _FakeApi across every wandb.Api() call in this test, so
        # the states_by_call iterator actually advances call-to-call instead
        # of restarting -- the real script calls wandb.Api() fresh for each
        # state query, exactly like this.
        self._api = _FakeApi(states_by_call)

    def Api(self):
        return self._api


@pytest.fixture(autouse=True)
def _no_real_wandb_module(monkeypatch):
    monkeypatch.delitem(sys.modules, "wandb", raising=False)


def _install_fake_wandb(monkeypatch, states_by_call):
    fake = _FakeWandbModule(states_by_call)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def _write_fake_cli_dir(tmp_path_factory_dir, *, returncode: int = 0, stdout: str = "Sweep updated"):
    """Writes a fake `wandb` CLI into its own directory and returns that
    directory, for prepending to PATH. Exercises ``_wandb_cli_path()``'s
    ``shutil.which`` fallback branch (portable across POSIX/Windows) rather
    than its sibling-of-interpreter branch, which is exact-filename-only
    (production layout: canonical runtime's ``bin/python`` + ``bin/wandb``
    side by side on Linux) and not worth faking cross-platform here.
    """
    cli_dir = tmp_path_factory_dir / "fake_cli_bin"
    cli_dir.mkdir()
    if sys.platform == "win32":
        cli_path = cli_dir / "wandb.cmd"
        cli_path.write_text(f"@echo {stdout}\r\n@exit /b {returncode}\r\n", encoding="utf-8")
    else:
        cli_path = cli_dir / "wandb"
        cli_path.write_text(f"#!/usr/bin/env bash\necho '{stdout}'\nexit {returncode}\n", encoding="utf-8")
        cli_path.chmod(cli_path.stat().st_mode | stat.S_IEXEC)
    return cli_dir


@pytest.fixture
def fake_cli_on_path(tmp_path, monkeypatch):
    """Prepends a directory containing a fake `wandb` CLI to PATH, and points
    ``sys.executable`` somewhere with no sibling ``wandb``, forcing
    ``_wandb_cli_path()`` through its ``shutil.which`` fallback branch."""
    cli_dir = _write_fake_cli_dir(tmp_path)
    monkeypatch.setenv("PATH", str(cli_dir) + os.pathsep + os.environ.get("PATH", ""))
    no_sibling_dir = tmp_path / "interpreter_with_no_sibling_cli"
    no_sibling_dir.mkdir()
    fake_interpreter = no_sibling_dir / ("python.exe" if sys.platform == "win32" else "python")
    fake_interpreter.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(fake_interpreter))
    return cli_dir / ("wandb.cmd" if sys.platform == "win32" else "wandb")


def _run_main(monkeypatch, tmp_path, *, sweep_id, action):
    out_dir = tmp_path / "out"
    argv = [
        "wandb_online_sweep_qualification_lifecycle.py",
        "--sweep-id", sweep_id,
        "--action", action,
        "--out-dir", str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    exit_code = lifecycle.main()
    record = json.loads((out_dir / f"lifecycle_record_{action}.json").read_text(encoding="utf-8"))
    return exit_code, record


def test_status_action_makes_no_cli_call(monkeypatch, tmp_path, fake_cli_on_path):
    _install_fake_wandb(monkeypatch, states_by_call=["running", "running"])
    exit_code, record = _run_main(monkeypatch, tmp_path, sweep_id="e/p/s1", action="status")
    assert exit_code == 0
    assert record["cli_result"] is None
    assert record["state_before"] == "running"
    assert record["state_after"] == "running"


def test_pause_action_records_state_before_and_after(monkeypatch, tmp_path, fake_cli_on_path):
    _install_fake_wandb(monkeypatch, states_by_call=["running", "paused"])
    exit_code, record = _run_main(monkeypatch, tmp_path, sweep_id="e/p/s1", action="pause")
    assert exit_code == 0
    assert record["state_before"] == "running"
    assert record["state_after"] == "paused"
    assert record["cli_result"]["returncode"] == 0


def test_resume_action_records_state_transition(monkeypatch, tmp_path, fake_cli_on_path):
    _install_fake_wandb(monkeypatch, states_by_call=["paused", "running"])
    exit_code, record = _run_main(monkeypatch, tmp_path, sweep_id="e/p/s1", action="resume")
    assert exit_code == 0
    assert record["state_before"] == "paused"
    assert record["state_after"] == "running"


def test_stop_action_records_final_state(monkeypatch, tmp_path, fake_cli_on_path):
    _install_fake_wandb(monkeypatch, states_by_call=["running", "finished"])
    exit_code, record = _run_main(monkeypatch, tmp_path, sweep_id="e/p/s1", action="stop")
    assert exit_code == 0
    assert record["state_after"] == "finished"


def test_cli_failure_is_reported_and_causes_nonzero_exit(monkeypatch, tmp_path):
    _install_fake_wandb(monkeypatch, states_by_call=["running", "running"])
    cli_dir = _write_fake_cli_dir(tmp_path, returncode=1, stdout="boom")
    monkeypatch.setenv("PATH", str(cli_dir) + os.pathsep + os.environ.get("PATH", ""))
    no_sibling_dir = tmp_path / "interpreter_with_no_sibling_cli"
    no_sibling_dir.mkdir()
    fake_interpreter = no_sibling_dir / ("python.exe" if sys.platform == "win32" else "python")
    fake_interpreter.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(fake_interpreter))

    exit_code, record = _run_main(monkeypatch, tmp_path, sweep_id="e/p/s1", action="pause")
    assert exit_code == 1
    assert record["cli_result"]["returncode"] == 1


def test_action_choices_are_restricted(monkeypatch, tmp_path):
    argv = [
        "wandb_online_sweep_qualification_lifecycle.py",
        "--sweep-id", "e/p/s1",
        "--action", "not_a_real_action",
        "--out-dir", str(tmp_path / "out"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        lifecycle.main()

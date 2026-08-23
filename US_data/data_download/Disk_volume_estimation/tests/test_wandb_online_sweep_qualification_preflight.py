"""Tests for scripts/wandb_online_sweep_qualification_preflight.py.

Uses a fake in-process ``wandb`` module (monkeypatched into ``sys.modules``,
same convention as tests/test_wandb_offline_launch_contract_qualification.py)
to deterministically exercise the import/version check, and real temp-dir
netrc/env fixtures to exercise the CREDENTIAL-PRESENT BOOLEAN-ONLY logic --
never asserting on any actual secret value.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

import scripts.wandb_online_sweep_qualification_preflight as preflight


class _FakeWandbModule(types.ModuleType):
    def __init__(self, version: str = "0.28.1"):
        super().__init__("wandb")
        self.__version__ = version


@pytest.fixture
def fake_wandb(monkeypatch):
    fake = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


@pytest.fixture(autouse=True)
def _isolate_credentials_and_wandb_module(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delitem(sys.modules, "wandb", raising=False)


# ---------------------------------------------------------------------------
# _credential_present: boolean only, never returns/stores the secret value
# ---------------------------------------------------------------------------

def test_credential_present_true_from_env_var(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "totally-fake-test-value-not-a-real-secret")
    assert preflight._credential_present() is True


def _set_home(monkeypatch, home_dir) -> None:
    # netrc.netrc() resolves "~" via os.path.expanduser, which reads HOME on
    # POSIX and USERPROFILE on Windows -- set both so this test is portable
    # across the Linux (Moriah) and Windows (local) runtimes this repo uses.
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("USERPROFILE", str(home_dir))


def test_credential_present_false_when_neither_env_nor_netrc(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    _set_home(monkeypatch, tmp_path)
    assert preflight._credential_present() is False


def test_credential_present_true_from_netrc(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    _set_home(monkeypatch, tmp_path)
    netrc_path = tmp_path / ".netrc"
    netrc_path.write_text(
        f"machine {preflight._WANDB_NETRC_HOST}\nlogin user\npassword fake-test-password\n",
        encoding="utf-8",
    )
    assert preflight._credential_present() is True


def test_credential_present_return_value_is_a_plain_bool(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_API_KEY", "fake-test-value")
    result = preflight._credential_present()
    assert result is True
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# main(): checks dict / evidence record shape, never leaks the secret value
# ---------------------------------------------------------------------------

def _run_main(monkeypatch, tmp_path, *, expected_commit, expected_runtime_python):
    out_dir = tmp_path / "out"
    argv = [
        "wandb_online_sweep_qualification_preflight.py",
        "--expected-commit",
        expected_commit,
        "--expected-runtime-python",
        expected_runtime_python,
        "--out-dir",
        str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    exit_code = preflight.main()
    record = json.loads((out_dir / "preflight_record.json").read_text(encoding="utf-8"))
    return exit_code, record


def test_main_all_checks_pass_when_everything_matches(fake_wandb, monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_API_KEY", "fake-test-secret-value")
    real_head = preflight._git_head(preflight._REPO_ROOT)
    exit_code, record = _run_main(
        monkeypatch, tmp_path, expected_commit=real_head, expected_runtime_python=sys.executable
    )
    assert record["checks"]["git_commit_matches_expected"] is True
    assert record["checks"]["wandb_import_ok"] is True
    assert record["checks"]["credential_available"] is True
    assert record["wandb_version"] == "0.28.1"
    # Secret value must never appear anywhere in the written record.
    dumped = json.dumps(record)
    assert "fake-test-secret-value" not in dumped
    assert exit_code == (0 if record["all_checks_passed"] else 1)


def test_main_fails_on_commit_mismatch(fake_wandb, monkeypatch, tmp_path):
    exit_code, record = _run_main(
        monkeypatch, tmp_path,
        expected_commit="0" * 40,
        expected_runtime_python=sys.executable,
    )
    assert record["checks"]["git_commit_matches_expected"] is False
    assert exit_code == 1


def test_main_fails_when_wandb_import_fails(monkeypatch, tmp_path):
    def _raise_import_error(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("no module named wandb")
        return _real_import(name, *args, **kwargs)

    import builtins

    _real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", _raise_import_error)

    real_head = preflight._git_head(preflight._REPO_ROOT)
    exit_code, record = _run_main(
        monkeypatch, tmp_path, expected_commit=real_head, expected_runtime_python=sys.executable
    )
    assert record["checks"]["wandb_import_ok"] is False
    assert record["wandb_import_error"] is not None
    assert exit_code == 1


def test_main_records_credential_available_as_boolean_never_the_value(fake_wandb, monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_API_KEY", "another-fake-secret-xyz")
    real_head = preflight._git_head(preflight._REPO_ROOT)
    _, record = _run_main(
        monkeypatch, tmp_path, expected_commit=real_head, expected_runtime_python=sys.executable
    )
    assert record["credential_available"] is True
    assert isinstance(record["credential_available"], bool)
    dumped = json.dumps(record)
    assert "another-fake-secret-xyz" not in dumped

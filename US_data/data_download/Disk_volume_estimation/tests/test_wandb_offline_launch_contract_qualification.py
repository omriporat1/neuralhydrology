"""Tests for scripts/wandb_offline_launch_contract_qualification.py.

Uses the same fake in-process ``wandb`` module (monkeypatched into
``sys.modules``) as tests/test_wandb_tracking.py, so these tests prove the
script's own logic (policy-path resolution precedence, run-identity shape,
evidence-record/checks construction) deterministically, without a real
network or the real wandb package installed. The separate real-package run
(scripts/wandb_real_offline_qualification_smoke.py's sibling qualification
step) is what proves this same code path against the real backend -- see
docs/stage1_wandb_user_guide.md.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

import scripts.wandb_offline_launch_contract_qualification as qual

_REAL_OFFLINE_POLICY_PATH = "config/stage1_wandb_tracking_policy_offline_v001.yaml"
_REAL_DISABLED_POLICY_PATH = "config/stage1_wandb_tracking_policy_v001.yaml"


class _FakeWandbConfig(dict):
    def update(self, other=None, allow_val_change=None, **kwargs):
        if other:
            dict.update(self, other)
        if kwargs:
            dict.update(self, kwargs)


class _FakeWandbRun:
    def __init__(self, run_id: str = "fake-generated-run-id-0001"):
        self.id = run_id
        self.config = _FakeWandbConfig()
        self.summary = {}
        self.logged: list[tuple[int | None, dict]] = []
        self.finished = False

    def log(self, data, step=None):
        self.logged.append((step, dict(data)))

    def finish(self):
        self.finished = True


class _FakeWandbModule(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.init_calls: list[dict] = []
        self.last_run: "_FakeWandbRun | None" = None

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        run = _FakeWandbRun()
        self.last_run = run
        return run


@pytest.fixture
def fake_wandb(monkeypatch):
    fake = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


# ---------------------------------------------------------------------------
# Policy-path resolution precedence (mirrors the real launcher's
# --wandb-policy-path / WANDB_POLICY_PATH contract)
# ---------------------------------------------------------------------------

def _args(wandb_policy_path=None, wandb_dir=None):
    return types.SimpleNamespace(wandb_policy_path=wandb_policy_path, wandb_dir=wandb_dir)


def test_resolve_policy_path_prefers_cli_flag_over_env(monkeypatch):
    monkeypatch.setenv("WANDB_POLICY_PATH", "from/env.yaml")
    path, source = qual._resolve_policy_path(_args(wandb_policy_path="from/cli.yaml"))
    assert path == "from/cli.yaml"
    assert source == "cli_flag"


def test_resolve_policy_path_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("WANDB_POLICY_PATH", "from/env.yaml")
    path, source = qual._resolve_policy_path(_args(wandb_policy_path=None))
    assert path == "from/env.yaml"
    assert source == "env_var"


def test_resolve_policy_path_requires_one_or_the_other(monkeypatch):
    monkeypatch.delenv("WANDB_POLICY_PATH", raising=False)
    with pytest.raises(SystemExit):
        qual._resolve_policy_path(_args(wandb_policy_path=None))


def test_resolve_wandb_dir_precedence(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_DIR", raising=False)
    default_dir = qual._resolve_wandb_dir(_args())
    assert default_dir == qual._DEFAULT_OUT_DIR / "wandb_dir"

    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "from_env"))
    env_dir = qual._resolve_wandb_dir(_args())
    assert env_dir == tmp_path / "from_env"

    cli_dir = qual._resolve_wandb_dir(_args(wandb_dir=str(tmp_path / "from_cli")))
    assert cli_dir == tmp_path / "from_cli"


# ---------------------------------------------------------------------------
# End-to-end main() against the fake wandb backend
# ---------------------------------------------------------------------------

def test_main_against_offline_policy_all_checks_pass(fake_wandb, monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    out_dir = tmp_path / "out"
    argv = [
        "wandb_offline_launch_contract_qualification.py",
        "--wandb-policy-path",
        _REAL_OFFLINE_POLICY_PATH,
        "--wandb-dir",
        str(tmp_path / "wandb_dir"),
        "--out-dir",
        str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    (tmp_path / "wandb_dir").mkdir()
    # The fake backend never actually writes files to wandb_dir, so seed one
    # to exercise the "offline files created" inventory/check path the same
    # way a real wandb backend would populate it.
    (tmp_path / "wandb_dir" / "run-fake.wandb").write_bytes(b"fake offline run file")

    exit_code = qual.main()

    assert exit_code == 0
    record = json.loads((out_dir / "qualification_record.json").read_text(encoding="utf-8"))
    assert record["backend"] == "wandb"
    assert record["mode"] == "offline"
    assert record["checks"]["policy_enabled_true"] is True
    assert record["checks"]["policy_mode_offline"] is True
    assert record["checks"]["backend_is_wandb"] is True
    assert record["checks"]["offline_run_files_created"] is True
    assert record["checks"]["no_online_dependency"] is True
    assert record["checks"]["qualification_identity_non_scientific"] is True
    assert record["checks"]["wandb_run_id_non_null"] is True
    assert record["wandb_run_id"] == "fake-generated-run-id-0001"
    assert record["all_checks_passed"] is True
    assert record["run_identity"]["qualification_kind"] == "wandb_offline_launch_contract"
    assert record["run_identity"]["launch_contract_qualification"] is True
    # Never a real/pilot-scientific identity field.
    assert "pilot_policy_name" not in record["run_identity"]
    assert "run_spec" not in record["run_identity"]

    init_kwargs = fake_wandb.init_calls[0]
    assert init_kwargs["mode"] == "offline"
    assert init_kwargs["project"] == "flashnh-stage1"


def test_main_against_disabled_default_policy_fails_checks(fake_wandb, monkeypatch, tmp_path):
    """Pointing this qualification at the committed DISABLED default must
    resolve backend='null' and fail the checks -- proving the script
    actually exercises the policy it is given rather than assuming
    offline."""
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    out_dir = tmp_path / "out"
    argv = [
        "wandb_offline_launch_contract_qualification.py",
        "--wandb-policy-path",
        _REAL_DISABLED_POLICY_PATH,
        "--wandb-dir",
        str(tmp_path / "wandb_dir"),
        "--out-dir",
        str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    exit_code = qual.main()

    assert exit_code == 1
    record = json.loads((out_dir / "qualification_record.json").read_text(encoding="utf-8"))
    assert record["backend"] == "null"
    assert record["checks"]["policy_enabled_true"] is False
    assert record["checks"]["backend_is_wandb"] is False
    assert record["all_checks_passed"] is False
    assert fake_wandb.init_calls == []


def test_main_requires_explicit_policy_path(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_POLICY_PATH", raising=False)
    argv = ["wandb_offline_launch_contract_qualification.py", "--out-dir", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        qual.main()

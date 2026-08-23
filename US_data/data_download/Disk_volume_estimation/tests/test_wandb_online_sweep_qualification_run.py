"""Tests for scripts/wandb_online_sweep_qualification_run.py.

Uses a fake in-process ``wandb`` module (monkeypatched into ``sys.modules``)
to deterministically exercise: sweep creation vs. reuse, the
one-proposal-per-``wandb.agent(..., count=1)`` call contract, online-mode
verification (including the mandatory FAIL-on-silent-offline-fallback
path), toy-objective/legality wiring, and the commit/runtime guard refusal
behavior -- all without a real network or the real wandb package.
"""
from __future__ import annotations

import json
import subprocess
import sys
import types

import pytest

import scripts.wandb_online_sweep_qualification_run as run_script


class _FakeSettings:
    def __init__(self, mode: str):
        self.mode = mode


class _FakeWandbRun:
    def __init__(self, run_id: str, mode: str, url: "str | None"):
        self.id = run_id
        self.settings = _FakeSettings(mode)
        self._url = url
        self.config = {}
        self.logged: list[dict] = []
        self.finished = False

    def get_url(self):
        if self._url is None:
            raise RuntimeError("no url available (offline run)")
        return self._url

    def log(self, data):
        self.logged.append(dict(data))

    def finish(self):
        self.finished = True


class _FakeWandbModule(types.ModuleType):
    def __init__(self, *, run_mode: str = "online", run_url: "str | None" = "https://wandb.ai/fake/entity/runs/abc123"):
        super().__init__("wandb")
        self.run_mode = run_mode
        self.run_url = run_url
        self.sweep_calls: list[dict] = []
        self.agent_calls: list[dict] = []
        self.created_sweep_id = "fake_entity/fake_project/fakesweepid0001"
        self._next_run_id = 0
        self.runs: list[_FakeWandbRun] = []

        proposals = [
            {"learning_rate": 3.1e-4, "hidden_size": 128, "embedding_dropout": 0.1, "output_dropout": 0.1, "batch_size": 256},
            {"learning_rate": 5.5e-4, "hidden_size": 64, "embedding_dropout": 0.2, "output_dropout": 0.0, "batch_size": 512},
        ]
        self._proposals = iter(proposals)

    def sweep(self, config, project=None, entity=None):
        self.sweep_calls.append({"config": config, "project": project, "entity": entity})
        return self.created_sweep_id

    def init(self, **kwargs):
        self._next_run_id += 1
        run = _FakeWandbRun(run_id=f"fake-run-{self._next_run_id:04d}", mode=self.run_mode, url=self.run_url)
        try:
            run.config.update(next(self._proposals))
        except StopIteration:
            run.config.update({"learning_rate": 3e-4, "hidden_size": 128, "embedding_dropout": 0.1, "output_dropout": 0.1, "batch_size": 256})
        self.runs.append(run)
        return run

    def agent(self, sweep_id, function=None, project=None, entity=None, count=None):
        self.agent_calls.append({"sweep_id": sweep_id, "project": project, "entity": entity, "count": count})
        assert count == 1, "production architecture requires exactly one proposal per allocation"
        function()


@pytest.fixture
def fake_wandb(monkeypatch):
    fake = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


@pytest.fixture(autouse=True)
def _no_real_wandb_module(monkeypatch):
    monkeypatch.delitem(sys.modules, "wandb", raising=False)


def _run_main(monkeypatch, tmp_path, *, proposal_label, sweep_id=None, expected_commit=None):
    if expected_commit is None:
        expected_commit = run_script._git_head(run_script._REPO_ROOT)
    out_dir = tmp_path / "out"
    argv = [
        "wandb_online_sweep_qualification_run.py",
        "--expected-commit",
        expected_commit,
        "--expected-runtime-python",
        sys.executable,
        "--proposal-label",
        proposal_label,
        "--out-dir",
        str(out_dir),
    ]
    if sweep_id is not None:
        argv += ["--sweep-id", sweep_id]
    monkeypatch.setattr(sys, "argv", argv)
    exit_code = run_script.main()
    record = json.loads((out_dir / f"run_record_{proposal_label}.json").read_text(encoding="utf-8"))
    return exit_code, record


# ---------------------------------------------------------------------------
# guard refusal
# ---------------------------------------------------------------------------

def test_refuses_on_commit_mismatch(fake_wandb, monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="REFUSING"):
        _run_main(monkeypatch, tmp_path, proposal_label="first", expected_commit="0" * 40)


def test_refuses_on_non_canonical_runtime(fake_wandb, monkeypatch, tmp_path):
    real_head = run_script._git_head(run_script._REPO_ROOT)
    out_dir = tmp_path / "out"
    argv = [
        "wandb_online_sweep_qualification_run.py",
        "--expected-commit", real_head,
        "--expected-runtime-python", "/some/other/python",
        "--proposal-label", "first",
        "--out-dir", str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="REFUSING"):
        run_script.main()


# ---------------------------------------------------------------------------
# sweep create vs. reuse
# ---------------------------------------------------------------------------

def test_first_job_creates_a_new_sweep(fake_wandb, monkeypatch, tmp_path):
    exit_code, record = _run_main(monkeypatch, tmp_path, proposal_label="first")
    assert record["created_new_sweep"] is True
    assert len(fake_wandb.sweep_calls) == 1
    assert record["sweep_id"] == fake_wandb.created_sweep_id
    assert exit_code == 0


def test_second_job_reuses_the_given_sweep_id(fake_wandb, monkeypatch, tmp_path):
    exit_code, record = _run_main(monkeypatch, tmp_path, proposal_label="second", sweep_id="some/existing/sweep")
    assert record["created_new_sweep"] is False
    assert len(fake_wandb.sweep_calls) == 0
    assert record["sweep_id"] == "some/existing/sweep"
    assert exit_code == 0


# ---------------------------------------------------------------------------
# exactly one proposal per allocation
# ---------------------------------------------------------------------------

def test_agent_is_called_with_count_one(fake_wandb, monkeypatch, tmp_path):
    _run_main(monkeypatch, tmp_path, proposal_label="first")
    assert len(fake_wandb.agent_calls) == 1
    assert fake_wandb.agent_calls[0]["count"] == 1


def test_exactly_one_run_is_initialized(fake_wandb, monkeypatch, tmp_path):
    _run_main(monkeypatch, tmp_path, proposal_label="first")
    assert len(fake_wandb.runs) == 1
    assert fake_wandb.runs[0].finished is True


# ---------------------------------------------------------------------------
# online-mode verification, including mandatory FAIL on silent offline
# ---------------------------------------------------------------------------

def test_online_run_passes_and_logs_toy_metric(fake_wandb, monkeypatch, tmp_path):
    exit_code, record = _run_main(monkeypatch, tmp_path, proposal_label="first")
    assert exit_code == 0
    assert record["online_confirmed"] is True
    assert record["reported_mode"] == "online"
    assert record["run_url"] == fake_wandb.run_url
    assert "toy_objective" in record
    assert isinstance(record["toy_objective"], float)
    assert fake_wandb.runs[0].logged == [{"qualification/toy_objective": record["toy_objective"]}]


def test_silent_offline_fallback_is_a_hard_failure_not_partial_success(monkeypatch, tmp_path):
    fake = _FakeWandbModule(run_mode="offline", run_url=None)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    with pytest.raises(SystemExit, match="QUALIFICATION FAIL"):
        _run_main(monkeypatch, tmp_path, proposal_label="first")


def test_missing_url_with_online_mode_string_still_fails(monkeypatch, tmp_path):
    """Defence in depth: even if ``settings.mode`` claims online, a missing
    hosted URL (no server-visible identity) must still fail -- the task
    requires a non-null hosted URL as an independent online signal, not
    trusting a single field."""
    fake = _FakeWandbModule(run_mode="online", run_url=None)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    with pytest.raises(SystemExit, match="QUALIFICATION FAIL"):
        _run_main(monkeypatch, tmp_path, proposal_label="first")


# ---------------------------------------------------------------------------
# Flash-NH legality/configuration_id wiring
# ---------------------------------------------------------------------------

def test_flashnh_legality_is_recorded_for_the_proposal(fake_wandb, monkeypatch, tmp_path):
    _, record = _run_main(monkeypatch, tmp_path, proposal_label="first")
    legality = record["flashnh_legality"]
    assert legality["legality_pass"] is True
    assert legality["configuration_id"].startswith("sweep_v1_cfg_")


# ---------------------------------------------------------------------------
# distinct proposals across two separate jobs
# ---------------------------------------------------------------------------

def test_two_separate_jobs_receive_distinct_run_ids(fake_wandb, monkeypatch, tmp_path):
    _, first_record = _run_main(monkeypatch, tmp_path, proposal_label="first")
    _, second_record = _run_main(monkeypatch, tmp_path, proposal_label="second", sweep_id=first_record["sweep_id"])
    assert first_record["run_id"] != second_record["run_id"]
    assert first_record["proposed_config"] != second_record["proposed_config"]

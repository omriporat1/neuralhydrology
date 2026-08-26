"""Tests for ``src.baseline.sweep_v2_six_axis_objective_recovery`` (Section G,
additive six-axis campaign foundation) -- the v2 sibling of
``sweep_v1_objective_recovery``.

Most of the underlying machinery (``load_immutable_trial_record``,
``assert_recovery_eligible``, ``build_objective_publication_payload``,
``is_already_published``, ``record_publication``) is literally the same
imported v1 function, already covered exhaustively by
tests/test_sweep_v1_objective_recovery.py -- that coverage is not repeated
here. This file focuses on what v2 adds: refusing a foreign (e.g. v1)
campaign/domain via ``assert_v2_campaign_identity``, the six-axis pinned-
identity delegation via ``assert_matches_expected_identity_v2``, and the
full ``recover_and_publish_objective_v2`` orchestration (idempotent
short-circuit, pre-wandb refusal paths, full publish path against a fake
in-process ``wandb`` module, sweep-mismatch refusal) -- exactly mirroring
v1's own fake-wandb pattern, never the real package, never the network.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from src.baseline import sweep_v1_campaign as sweep_v1
from src.baseline.sweep_v2_six_axis_campaign import CAMPAIGN_ID_V2, DOMAIN_VERSION_V2
from src.baseline.sweep_v2_six_axis_objective_recovery import (
    ObjectiveRecoveryError,
    SweepV2ObjectiveRecoveryError,
    assert_matches_expected_identity_v2,
    assert_v2_campaign_identity,
    build_objective_publication_payload,
    is_already_published,
    record_publication,
    recover_and_publish_objective_v2,
)

_VALID_RECORD = {
    "campaign_id": CAMPAIGN_ID_V2,
    "domain_version": DOMAIN_VERSION_V2,
    "configuration_id": "sweep_v2_cfg_deadbeefdeadbeefdead0",
    "proposal_id": f"{CAMPAIGN_ID_V2}__bayesian__proposal007",
    "proposal_order": 7,
    "search_arm": "bayesian",
    "trial_id": f"{CAMPAIGN_ID_V2}__sweep_v2_cfg_deadbeefdeadbeefdead0__seedA967139__attempt002",
    "retry_of_trial_id": f"{CAMPAIGN_ID_V2}__sweep_v2_cfg_deadbeefdeadbeefdead0__seedA967139__attempt001",
    "execution_generation": 2,
    "execution_status": "VALID",
    "objective_score": 0.40,
    "generated_nh_config_sha256": "a" * 64,
    "wandb_run_id": "fake-run-0001",
    "wandb_sweep_id": "rehearsal-sweep-v2-abc",
    "hyperparameters": {
        "learning_rate": 3e-4, "hidden_size": 128, "embedding_dropout": 0.10,
        "output_dropout": 0.25, "batch_size": 256, "seq_length": 96,
    },
    "support_contract_version": "common120_raw_space_nse_v001",
    "support_contract_sha256": "b" * 64,
}

_INVALID_RECORD = {**_VALID_RECORD, "execution_status": "INVALID", "objective_score": None}

_V1_SHAPED_RECORD = {
    **_VALID_RECORD,
    "campaign_id": sweep_v1.CAMPAIGN_ID,
    "domain_version": sweep_v1.DOMAIN_VERSION,
}


def _write_record(tmp_path, record) -> "str":
    path = tmp_path / "execution_provenance.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return str(path)


# --- assert_v2_campaign_identity ------------------------------------------------

def test_assert_v2_campaign_identity_passes_on_a_genuine_v2_record():
    assert_v2_campaign_identity(_VALID_RECORD)  # must not raise


def test_assert_v2_campaign_identity_refuses_a_v1_campaign_record():
    with pytest.raises(SweepV2ObjectiveRecoveryError, match="Sweep-v2 six-axis"):
        assert_v2_campaign_identity(_V1_SHAPED_RECORD)


def test_assert_v2_campaign_identity_refuses_a_matching_campaign_but_foreign_domain():
    record = {**_VALID_RECORD, "domain_version": "some_other_domain_v999"}
    with pytest.raises(SweepV2ObjectiveRecoveryError, match="Sweep-v2 six-axis"):
        assert_v2_campaign_identity(record)


# --- assert_matches_expected_identity_v2 -----------------------------------------

def test_assert_matches_expected_identity_v2_passes_on_matching_subset():
    assert_matches_expected_identity_v2(
        _VALID_RECORD, {"trial_id": _VALID_RECORD["trial_id"], "configuration_id": _VALID_RECORD["configuration_id"]},
    )


def test_assert_matches_expected_identity_v2_raises_on_seq_length_mismatch():
    from src.baseline.sweep_v2_six_axis_retry import SweepV2RetryError

    with pytest.raises(SweepV2RetryError, match="seq_length"):
        assert_matches_expected_identity_v2(_VALID_RECORD, {"seq_length": 48})


def test_assert_matches_expected_identity_v2_raises_on_mismatched_wandb_run_id():
    from src.baseline.sweep_v2_six_axis_retry import SweepV2RetryError

    with pytest.raises(SweepV2RetryError, match="wandb_run_id"):
        assert_matches_expected_identity_v2(_VALID_RECORD, {"wandb_run_id": "some-other-run-id"})


def test_assert_matches_expected_identity_v2_raises_on_support_contract_mismatch():
    from src.baseline.sweep_v2_six_axis_retry import SweepV2RetryError

    with pytest.raises(SweepV2RetryError, match="support_contract_version"):
        assert_matches_expected_identity_v2(_VALID_RECORD, {"support_contract_version": "some_other_version"})


# --- recover_and_publish_objective_v2: pre-wandb refusal paths -------------------

def test_recover_and_publish_objective_v2_refuses_a_v1_record_before_any_wandb_import(tmp_path, monkeypatch):
    record_path = _write_record(tmp_path, _V1_SHAPED_RECORD)
    marker_path = tmp_path / "marker.json"

    monkeypatch.setitem(sys.modules, "wandb", None)

    with pytest.raises(SweepV2ObjectiveRecoveryError, match="Sweep-v2 six-axis"):
        recover_and_publish_objective_v2(
            execution_provenance_path=record_path, expected_identity={},
            marker_path=marker_path, project="flashnh-stage1-test",
        )
    assert not marker_path.exists()


def test_recover_and_publish_objective_v2_refuses_invalid_record_before_any_wandb_import(tmp_path, monkeypatch):
    record_path = _write_record(tmp_path, _INVALID_RECORD)
    marker_path = tmp_path / "marker.json"

    monkeypatch.setitem(sys.modules, "wandb", None)

    with pytest.raises(ObjectiveRecoveryError, match="non-VALID"):
        recover_and_publish_objective_v2(
            execution_provenance_path=record_path, expected_identity={},
            marker_path=marker_path, project="flashnh-stage1-test",
        )
    assert not marker_path.exists()


def test_recover_and_publish_objective_v2_refuses_identity_mismatch_before_any_wandb_import(tmp_path, monkeypatch):
    record_path = _write_record(tmp_path, _VALID_RECORD)
    marker_path = tmp_path / "marker.json"

    monkeypatch.setitem(sys.modules, "wandb", None)

    from src.baseline.sweep_v2_six_axis_retry import SweepV2RetryError

    with pytest.raises(SweepV2RetryError):
        recover_and_publish_objective_v2(
            execution_provenance_path=record_path,
            expected_identity={"configuration_id": "sweep_v2_cfg_wrongwrongwrongwrong0"},
            marker_path=marker_path, project="flashnh-stage1-test",
        )
    assert not marker_path.exists()


# --- recover_and_publish_objective_v2: idempotent short-circuit ------------------

def test_recover_and_publish_objective_v2_is_idempotent_and_never_imports_wandb_when_already_published(
    tmp_path, monkeypatch,
):
    record_path = _write_record(tmp_path, _VALID_RECORD)
    marker_path = tmp_path / "marker.json"
    record_publication(
        marker_path, wandb_run_id=_VALID_RECORD["wandb_run_id"],
        payload=build_objective_publication_payload(_VALID_RECORD),
    )

    monkeypatch.setitem(sys.modules, "wandb", None)  # poison: any import attempt raises ImportError

    result = recover_and_publish_objective_v2(
        execution_provenance_path=record_path, expected_identity={"trial_id": _VALID_RECORD["trial_id"]},
        marker_path=marker_path, project="flashnh-stage1-test",
    )

    assert result == {"status": "already_published", "wandb_run_id": _VALID_RECORD["wandb_run_id"]}


def test_recover_and_publish_objective_v2_refuses_a_changed_objective_under_the_same_marker(tmp_path, monkeypatch):
    record_path = _write_record(tmp_path, _VALID_RECORD)
    marker_path = tmp_path / "marker.json"
    stale_payload = build_objective_publication_payload(_VALID_RECORD)
    stale_payload["flashnh/objective_score"] = 0.999999
    record_publication(marker_path, wandb_run_id=_VALID_RECORD["wandb_run_id"], payload=stale_payload)

    monkeypatch.setitem(sys.modules, "wandb", None)

    with pytest.raises(SweepV2ObjectiveRecoveryError, match="changed objective"):
        recover_and_publish_objective_v2(
            execution_provenance_path=record_path, expected_identity={"trial_id": _VALID_RECORD["trial_id"]},
            marker_path=marker_path, project="flashnh-stage1-test",
        )


# --- recover_and_publish_objective_v2: full path against a fake wandb -----------

def _fake_wandb_api_module(run):
    module = types.ModuleType("wandb")
    captured: "dict[str, object]" = {}

    class _FakeApi:
        def run(self, path):
            captured["run_path"] = path
            return run

    module.Api = _FakeApi
    module.captured = captured
    return module


class _FakeSweep:
    def __init__(self, sweep_id: str):
        self.id = sweep_id


class _FakeApiRun:
    """Same shape as v1's own fake, mirroring the REAL wandb.apis.public Run
    (``run.sweep.id``, not a fictitious ``run.sweepId``) -- see
    tests/test_sweep_v1_objective_recovery.py's identical fake for the
    documented history of this exact bug class."""

    def __init__(self, run_id: str, sweep_id: "str | None"):
        self.id = run_id
        self.sweep = _FakeSweep(sweep_id) if sweep_id is not None else None
        self.summary: "dict[str, object]" = {}

    def update(self):
        pass


def test_recover_and_publish_objective_v2_full_path_publishes_and_writes_marker(tmp_path, monkeypatch):
    record_path = _write_record(tmp_path, _VALID_RECORD)
    marker_path = tmp_path / "marker.json"
    fake_run = _FakeApiRun(_VALID_RECORD["wandb_run_id"], _VALID_RECORD["wandb_sweep_id"])
    fake_module = _fake_wandb_api_module(fake_run)
    monkeypatch.setitem(sys.modules, "wandb", fake_module)

    result = recover_and_publish_objective_v2(
        execution_provenance_path=record_path, expected_identity={"trial_id": _VALID_RECORD["trial_id"]},
        marker_path=marker_path, project="flashnh-stage1-test",
    )

    assert result["status"] == "published"
    assert fake_module.captured["run_path"] == f"flashnh-stage1-test/{_VALID_RECORD['wandb_run_id']}"
    assert fake_run.summary["flashnh/valid"] is True
    assert fake_run.summary["flashnh/objective_score"] == 0.40
    assert marker_path.is_file()
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["wandb_run_id"] == _VALID_RECORD["wandb_run_id"]

    monkeypatch.setitem(sys.modules, "wandb", None)
    second = recover_and_publish_objective_v2(
        execution_provenance_path=record_path, expected_identity={"trial_id": _VALID_RECORD["trial_id"]},
        marker_path=marker_path, project="flashnh-stage1-test",
    )
    assert second["status"] == "already_published"


def test_recover_and_publish_objective_v2_refuses_sweep_mismatch_and_does_not_write_marker(tmp_path, monkeypatch):
    record_path = _write_record(tmp_path, _VALID_RECORD)
    marker_path = tmp_path / "marker.json"
    fake_run = _FakeApiRun(_VALID_RECORD["wandb_run_id"], "some-unrelated-sweep")
    fake_module = _fake_wandb_api_module(fake_run)
    monkeypatch.setitem(sys.modules, "wandb", fake_module)

    with pytest.raises(SweepV2ObjectiveRecoveryError, match="refusing"):
        recover_and_publish_objective_v2(
            execution_provenance_path=record_path, expected_identity={"trial_id": _VALID_RECORD["trial_id"]},
            marker_path=marker_path, project="flashnh-stage1-test",
        )

    assert fake_run.summary == {}
    assert not marker_path.exists()


def test_recover_and_publish_objective_v2_reuses_v1_payload_shape():
    """build_objective_publication_payload is imported unchanged from v1 --
    confirm identity, not just behavior, to lock in the reuse-vs-sibling
    decision recorded in the module docstring."""
    from src.baseline.sweep_v1_objective_recovery import build_objective_publication_payload as v1_fn

    assert build_objective_publication_payload is v1_fn

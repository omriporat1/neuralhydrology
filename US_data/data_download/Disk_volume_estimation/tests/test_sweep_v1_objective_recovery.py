"""Tests for ``src.baseline.sweep_v1_objective_recovery`` -- the accepted
exact-retry startup rehearsal task's Section F frozen design: an idempotent,
read-only-with-respect-to-the-Flash-NH-record mechanism to republish a
trial's objective to W&B after a transient W&B failure, without retraining
and without inventing any new scientific fact.

Per the task's explicit scope boundary, this module's network-facing
``recover_and_publish_objective`` path is implemented but NOT exercised
against the production sweep or any real W&B backend anywhere in this
session; here it is exercised only against an in-process fake ``wandb``
module (never the real package, never the network).

Covers, in order:
  1. ``load_immutable_trial_record``: accepts a genuinely terminal (VALID or
     INVALID) record with a recorded wandb run/sweep association; refuses a
     missing file, a non-terminal ``execution_status``, and a terminal
     record that never reached ``wandb_associated`` (no run to republish
     onto).
  1b. ``assert_recovery_eligible``: authored in the follow-up closure task to
     close a gap found while attempting Section F qualification -- narrows
     eligibility beyond ``load_immutable_trial_record``'s generic
     terminal-status check. Refuses a non-VALID (e.g. INVALID) record, an
     incomplete record, a record with no ``generated_nh_config_sha256``
     (missing source hash), and a missing/non-finite ``objective_score``.
  2. ``assert_matches_expected_identity``: passes on a matching subset,
     raises on any contradiction (including a pinned-but-mismatched
     ``wandb_run_id``, also authored in the follow-up closure task) --
     delegates to ``sweep_v1_retry.assert_matches_pinned_identity`` with no
     parallel identity rule.
  3. ``build_objective_publication_payload``: pure derivation of exactly the
     W&B summary fields a recovery would publish, for both VALID and
     INVALID records; never imports wandb as a side effect of being called.
  4. ``is_already_published`` / ``record_publication``: the local
     idempotency marker's presence check and write path.
  5. ``recover_and_publish_objective``: idempotent short-circuit (already
     published, with a payload-equality check refusing a changed objective
     under the same marker) and pre-wandb refusal paths (non-terminal
     record, non-VALID record, incomplete record, missing source hash,
     non-finite objective, identity mismatch) never import wandb; the full
     path against a fake ``wandb.Api()`` module publishes the payload and
     writes the marker; a sweep-association mismatch refuses and leaves no
     marker behind.
"""
from __future__ import annotations

import copy
import json
import math
import sys
import types

import pytest

from src.baseline.sweep_v1_objective_recovery import (
    ObjectiveRecoveryError, assert_matches_expected_identity, assert_recovery_eligible,
    build_objective_publication_payload, is_already_published, load_immutable_trial_record, record_publication,
    recover_and_publish_objective,
)
from src.baseline.sweep_v1_retry import SweepV1RetryError

_VALID_RECORD = {
    "campaign_id": "stage1_phase_b_sweep_v1_original_domain_v001",
    "configuration_id": "sweep_v1_cfg_deadbeefdeadbeefdead",
    "proposal_id": "stage1_phase_b_sweep_v1_original_domain_v001__bayesian__proposal007",
    "proposal_order": 7,
    "search_arm": "bayesian",
    "trial_id": "stage1_phase_b_sweep_v1_original_domain_v001__sweep_v1_cfg_deadbeefdeadbeefdead__mf12x50000__seedA967139__attempt002",
    "retry_of_trial_id": "stage1_phase_b_sweep_v1_original_domain_v001__sweep_v1_cfg_deadbeefdeadbeefdead__mf12x50000__seedA967139__attempt001",
    "execution_generation": 2,
    "execution_status": "VALID",
    "objective_score": 0.40,
    "generated_nh_config_sha256": "a" * 64,
    "wandb_run_id": "fake-run-0001",
    "wandb_sweep_id": "rehearsal-sweep-abc",
}

_INVALID_RECORD = {**_VALID_RECORD, "execution_status": "INVALID", "objective_score": None}


def _write_record(tmp_path, record) -> "str":
    path = tmp_path / "execution_provenance.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return str(path)


# --- load_immutable_trial_record ---------------------------------------------

def test_load_immutable_trial_record_accepts_a_valid_terminal_record(tmp_path):
    path = _write_record(tmp_path, _VALID_RECORD)
    record = load_immutable_trial_record(path)
    assert record["execution_status"] == "VALID"
    assert record["wandb_run_id"] == "fake-run-0001"


def test_load_immutable_trial_record_accepts_an_invalid_terminal_record(tmp_path):
    path = _write_record(tmp_path, _INVALID_RECORD)
    record = load_immutable_trial_record(path)
    assert record["execution_status"] == "INVALID"
    assert record["objective_score"] is None


def test_load_immutable_trial_record_raises_when_file_missing(tmp_path):
    with pytest.raises(ObjectiveRecoveryError, match="not found"):
        load_immutable_trial_record(tmp_path / "does_not_exist.json")


@pytest.mark.parametrize("bad_status", [None, "wandb_init_failed", "prepared_with_config", "wandb_tags_rejected"])
def test_load_immutable_trial_record_raises_on_non_terminal_status(tmp_path, bad_status):
    record = dict(_VALID_RECORD)
    if bad_status is None:
        del record["execution_status"]
    else:
        record["provenance_stage"] = bad_status
        record["execution_status"] = None
    path = _write_record(tmp_path, record)
    with pytest.raises(ObjectiveRecoveryError, match="non-terminal"):
        load_immutable_trial_record(path)


@pytest.mark.parametrize("missing_key", ["wandb_run_id", "wandb_sweep_id"])
def test_load_immutable_trial_record_raises_when_wandb_identity_missing(tmp_path, missing_key):
    record = dict(_VALID_RECORD)
    record[missing_key] = None
    path = _write_record(tmp_path, record)
    with pytest.raises(ObjectiveRecoveryError, match="wandb_associated"):
        load_immutable_trial_record(path)


# --- assert_recovery_eligible --------------------------------------------------

def test_assert_recovery_eligible_passes_on_a_complete_valid_record():
    assert_recovery_eligible(_VALID_RECORD)  # must not raise


def test_assert_recovery_eligible_rejects_an_invalid_record():
    with pytest.raises(ObjectiveRecoveryError, match="non-VALID"):
        assert_recovery_eligible(_INVALID_RECORD)


@pytest.mark.parametrize(
    "missing_key",
    ["campaign_id", "proposal_id", "configuration_id", "trial_id", "execution_generation", "search_arm"],
)
def test_assert_recovery_eligible_rejects_an_incomplete_record(missing_key):
    record = dict(_VALID_RECORD)
    record[missing_key] = None
    with pytest.raises(ObjectiveRecoveryError, match="incomplete"):
        assert_recovery_eligible(record)


def test_assert_recovery_eligible_rejects_a_record_missing_the_source_config_checksum():
    record = dict(_VALID_RECORD)
    del record["generated_nh_config_sha256"]
    with pytest.raises(ObjectiveRecoveryError, match="source hash"):
        assert_recovery_eligible(record)


@pytest.mark.parametrize("bad_objective", [None, math.nan, math.inf, -math.inf])
def test_assert_recovery_eligible_rejects_a_missing_or_non_finite_objective(bad_objective):
    record = dict(_VALID_RECORD)
    record["objective_score"] = bad_objective
    with pytest.raises(ObjectiveRecoveryError, match="non-finite"):
        assert_recovery_eligible(record)


def test_recover_and_publish_objective_refuses_invalid_record_before_any_wandb_import(tmp_path, monkeypatch):
    record_path = _write_record(tmp_path, _INVALID_RECORD)
    marker_path = tmp_path / "marker.json"

    monkeypatch.setitem(sys.modules, "wandb", None)

    with pytest.raises(ObjectiveRecoveryError, match="non-VALID"):
        recover_and_publish_objective(
            execution_provenance_path=record_path, expected_identity={},
            marker_path=marker_path, project="flashnh-stage1-test",
        )
    assert not marker_path.exists()


def test_recover_and_publish_objective_refuses_incomplete_record_before_any_wandb_import(tmp_path, monkeypatch):
    record = dict(_VALID_RECORD)
    record["trial_id"] = None
    record_path = _write_record(tmp_path, record)
    marker_path = tmp_path / "marker.json"

    monkeypatch.setitem(sys.modules, "wandb", None)

    with pytest.raises(ObjectiveRecoveryError, match="incomplete"):
        recover_and_publish_objective(
            execution_provenance_path=record_path, expected_identity={},
            marker_path=marker_path, project="flashnh-stage1-test",
        )
    assert not marker_path.exists()


def test_recover_and_publish_objective_refuses_missing_source_hash_before_any_wandb_import(tmp_path, monkeypatch):
    record = dict(_VALID_RECORD)
    del record["generated_nh_config_sha256"]
    record_path = _write_record(tmp_path, record)
    marker_path = tmp_path / "marker.json"

    monkeypatch.setitem(sys.modules, "wandb", None)

    with pytest.raises(ObjectiveRecoveryError, match="source hash"):
        recover_and_publish_objective(
            execution_provenance_path=record_path, expected_identity={},
            marker_path=marker_path, project="flashnh-stage1-test",
        )
    assert not marker_path.exists()


def test_recover_and_publish_objective_refuses_non_finite_objective_before_any_wandb_import(tmp_path, monkeypatch):
    record = dict(_VALID_RECORD)
    record["objective_score"] = math.nan
    record_path = _write_record(tmp_path, record)
    marker_path = tmp_path / "marker.json"

    monkeypatch.setitem(sys.modules, "wandb", None)

    with pytest.raises(ObjectiveRecoveryError, match="non-finite"):
        recover_and_publish_objective(
            execution_provenance_path=record_path, expected_identity={},
            marker_path=marker_path, project="flashnh-stage1-test",
        )
    assert not marker_path.exists()


def test_recover_and_publish_objective_refuses_a_changed_objective_under_the_same_marker(tmp_path, monkeypatch):
    record_path = _write_record(tmp_path, _VALID_RECORD)
    marker_path = tmp_path / "marker.json"
    stale_payload = build_objective_publication_payload(_VALID_RECORD)
    stale_payload["flashnh/objective_score"] = 0.999999  # simulates a since-changed/tampered record
    record_publication(marker_path, wandb_run_id=_VALID_RECORD["wandb_run_id"], payload=stale_payload)

    monkeypatch.setitem(sys.modules, "wandb", None)  # must never even get to a wandb import

    with pytest.raises(ObjectiveRecoveryError, match="changed objective"):
        recover_and_publish_objective(
            execution_provenance_path=record_path, expected_identity={"trial_id": _VALID_RECORD["trial_id"]},
            marker_path=marker_path, project="flashnh-stage1-test",
        )


# --- assert_matches_expected_identity -----------------------------------------

def test_assert_matches_expected_identity_passes_on_matching_subset():
    assert_matches_expected_identity(
        _VALID_RECORD, {"trial_id": _VALID_RECORD["trial_id"], "configuration_id": _VALID_RECORD["configuration_id"]},
    )


def test_assert_matches_expected_identity_raises_on_mismatch():
    with pytest.raises(SweepV1RetryError):
        assert_matches_expected_identity(_VALID_RECORD, {"configuration_id": "sweep_v1_cfg_wrongwrongwrongwrong"})


def test_assert_matches_expected_identity_raises_on_mismatched_wandb_run_id():
    with pytest.raises(SweepV1RetryError, match="wandb_run_id"):
        assert_matches_expected_identity(_VALID_RECORD, {"wandb_run_id": "some-other-run-id"})


# --- build_objective_publication_payload --------------------------------------

def test_build_objective_publication_payload_for_valid_record():
    payload = build_objective_publication_payload(_VALID_RECORD)
    assert payload == {
        "flashnh/valid": True,
        "flashnh/objective_score": 0.40,
        "flashnh/trial_id": _VALID_RECORD["trial_id"],
        "flashnh/retry_of_trial_id": _VALID_RECORD["retry_of_trial_id"],
        "flashnh/execution_generation": 2,
        "flashnh/objective_recovered": True,
    }


def test_build_objective_publication_payload_for_invalid_record():
    payload = build_objective_publication_payload(_INVALID_RECORD)
    assert payload["flashnh/valid"] is False
    assert payload["flashnh/objective_score"] is None
    assert payload["flashnh/objective_recovered"] is True


def test_build_objective_publication_payload_never_imports_wandb(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    before = copy.deepcopy(_VALID_RECORD)
    build_objective_publication_payload(_VALID_RECORD)
    assert _VALID_RECORD == before  # pure: no mutation either


# --- is_already_published / record_publication --------------------------------

def test_is_already_published_reflects_marker_file_presence(tmp_path):
    marker_path = tmp_path / "nested" / "marker.json"
    assert is_already_published(marker_path) is False
    record_publication(marker_path, wandb_run_id="fake-run-0001", payload={"flashnh/valid": True})
    assert is_already_published(marker_path) is True


def test_record_publication_writes_marker_and_creates_parent_dirs(tmp_path):
    marker_path = tmp_path / "does" / "not" / "exist_yet" / "marker.json"
    payload = {"flashnh/valid": True, "flashnh/objective_score": 0.40}
    marker = record_publication(marker_path, wandb_run_id="fake-run-0001", payload=payload)
    assert marker == {"wandb_run_id": "fake-run-0001", "published_payload": payload}
    on_disk = json.loads(marker_path.read_text(encoding="utf-8"))
    assert on_disk == marker


# --- recover_and_publish_objective ---------------------------------------------

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


class _FakeApiRun:
    def __init__(self, run_id: str, sweep_id: "str | None"):
        self.id = run_id
        self.sweepId = sweep_id
        self.summary: "dict[str, object]" = {}

    def update(self):
        # run.summary.update() is called with no arguments by production
        # code -- this method exists only so a plain-dict-like attribute
        # access pattern is not required; see run.summary usage below.
        pass


def test_recover_and_publish_objective_is_idempotent_and_never_imports_wandb_when_already_published(
    tmp_path, monkeypatch,
):
    record_path = _write_record(tmp_path, _VALID_RECORD)
    marker_path = tmp_path / "marker.json"
    record_publication(
        marker_path, wandb_run_id=_VALID_RECORD["wandb_run_id"],
        payload=build_objective_publication_payload(_VALID_RECORD),
    )

    monkeypatch.setitem(sys.modules, "wandb", None)  # poison: any import attempt raises ImportError

    result = recover_and_publish_objective(
        execution_provenance_path=record_path, expected_identity={"trial_id": _VALID_RECORD["trial_id"]},
        marker_path=marker_path, project="flashnh-stage1-test",
    )

    assert result == {"status": "already_published", "wandb_run_id": _VALID_RECORD["wandb_run_id"]}


def test_recover_and_publish_objective_refuses_non_terminal_record_before_any_wandb_import(tmp_path, monkeypatch):
    record = dict(_VALID_RECORD)
    del record["execution_status"]
    record_path = _write_record(tmp_path, record)
    marker_path = tmp_path / "marker.json"

    monkeypatch.setitem(sys.modules, "wandb", None)

    with pytest.raises(ObjectiveRecoveryError, match="non-terminal"):
        recover_and_publish_objective(
            execution_provenance_path=record_path, expected_identity={},
            marker_path=marker_path, project="flashnh-stage1-test",
        )
    assert not marker_path.exists()


def test_recover_and_publish_objective_refuses_identity_mismatch_before_any_wandb_import(tmp_path, monkeypatch):
    record_path = _write_record(tmp_path, _VALID_RECORD)
    marker_path = tmp_path / "marker.json"

    monkeypatch.setitem(sys.modules, "wandb", None)

    with pytest.raises(SweepV1RetryError):
        recover_and_publish_objective(
            execution_provenance_path=record_path,
            expected_identity={"configuration_id": "sweep_v1_cfg_wrongwrongwrongwrong"},
            marker_path=marker_path, project="flashnh-stage1-test",
        )
    assert not marker_path.exists()


def test_recover_and_publish_objective_full_path_publishes_and_writes_marker(tmp_path, monkeypatch):
    record_path = _write_record(tmp_path, _VALID_RECORD)
    marker_path = tmp_path / "marker.json"
    fake_run = _FakeApiRun(_VALID_RECORD["wandb_run_id"], _VALID_RECORD["wandb_sweep_id"])
    fake_module = _fake_wandb_api_module(fake_run)
    monkeypatch.setitem(sys.modules, "wandb", fake_module)

    result = recover_and_publish_objective(
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

    # Idempotent re-invocation: no second Api() call needed -- prove by
    # poisoning wandb entirely and confirming no error and no re-publish.
    monkeypatch.setitem(sys.modules, "wandb", None)
    second = recover_and_publish_objective(
        execution_provenance_path=record_path, expected_identity={"trial_id": _VALID_RECORD["trial_id"]},
        marker_path=marker_path, project="flashnh-stage1-test",
    )
    assert second["status"] == "already_published"


def test_recover_and_publish_objective_refuses_sweep_mismatch_and_does_not_write_marker(tmp_path, monkeypatch):
    record_path = _write_record(tmp_path, _VALID_RECORD)
    marker_path = tmp_path / "marker.json"
    fake_run = _FakeApiRun(_VALID_RECORD["wandb_run_id"], "some-unrelated-sweep")
    fake_module = _fake_wandb_api_module(fake_run)
    monkeypatch.setitem(sys.modules, "wandb", fake_module)

    with pytest.raises(ObjectiveRecoveryError, match="refusing"):
        recover_and_publish_objective(
            execution_provenance_path=record_path, expected_identity={"trial_id": _VALID_RECORD["trial_id"]},
            marker_path=marker_path, project="flashnh-stage1-test",
        )

    assert fake_run.summary == {}  # never mutated
    assert not marker_path.exists()

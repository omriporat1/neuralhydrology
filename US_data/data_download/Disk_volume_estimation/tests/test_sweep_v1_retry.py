"""Tests for the exact-retry identity seam in ``src.baseline.sweep_v1_retry``.

Covers, in order:
  1. ``load_frozen_proposal_record`` happy path off a real durable record
     written by ``write_proposal_intake_provenance``.
  2. ``load_frozen_proposal_record`` raises on a missing file.
  3. ``load_frozen_proposal_record`` raises on a required field missing from
     the record.
  4. ``load_frozen_proposal_record`` raises when the persisted ``trial_id``
     disagrees with the identity re-derived from the record's own axes (a
     tampered or stale file is never silently trusted).
  5. ``load_frozen_proposal_record`` raises when persisted
     ``configuration_id``/``proposal_id`` disagree with re-derived values.
  6. ``load_frozen_proposal_record`` raises on a foreign campaign/domain.
  7. ``assert_matches_pinned_identity`` passes silently when every pinned
     field (identity + all five hyperparameters + model_seed) matches.
  8. ``assert_matches_pinned_identity`` raises, enumerating every mismatch,
     when a hyperparameter contradicts the pinned value.
  9. ``assert_matches_pinned_identity`` raises when pinned ``model_seed``
     disagrees with the campaign-wide ``sweep.MODEL_SEED_A`` constant.
  10. ``derive_exact_retry_identity`` happy path: identical hyperparameters/
      configuration_id/proposal_id, a fresh trial_id, correct
      retry_of_trial_id, requested execution_generation.
  11. ``derive_exact_retry_identity`` raises when the requested
      execution_generation does not strictly exceed the original.
  12. ``derive_exact_retry_identity`` raises on a non-integer (including
      ``bool``) execution_generation.

Never imports wandb; never touches the filesystem beyond ``tmp_path``; never
starts NH training or execution.
"""
from __future__ import annotations

import json

import pytest

from src.baseline import sweep_v1_campaign as sweep
from src.baseline.sweep_v1_execution import write_proposal_intake_provenance
from src.baseline.sweep_v1_retry import (
    SweepV1RetryError, assert_matches_pinned_identity, derive_exact_retry_identity, load_frozen_proposal_record,
)

_AXES = {"learning_rate": 3e-4, "hidden_size": 128, "embedding_dropout": 0.10, "output_dropout": 0.25, "batch_size": 256}


def _write_record(tmp_path, **overrides):
    kwargs = dict(
        output_root=tmp_path / "attempt1", axes=_AXES, search_arm="bayesian",
        proposal_order=1, wandb_sweep_id="prod-sweep", wandb_run_id="run-1",
    )
    kwargs.update(overrides)
    return write_proposal_intake_provenance(**kwargs)


def _record_path(tmp_path, record) -> "object":
    return tmp_path / "attempt1" / record["trial_id"] / "execution_provenance.json"


# --- load_frozen_proposal_record ---------------------------------------------

def test_load_frozen_proposal_record_happy_path_recomputes_identity(tmp_path):
    written = _write_record(tmp_path)
    loaded = load_frozen_proposal_record(_record_path(tmp_path, written))
    assert loaded["configuration_id"] == written["configuration_id"]
    assert loaded["proposal_id"] == written["proposal_id"]
    assert loaded["trial_id"] == written["trial_id"]
    # Persisted hyperparameters are the versioned canonical form (continuous
    # axes serialized as .17g strings, per sweep_v1_campaign.canonical_hyperparameters),
    # not the raw floats originally passed in.
    assert loaded["hyperparameters"] == sweep.canonical_hyperparameters(_AXES)


def test_load_frozen_proposal_record_raises_on_missing_file(tmp_path):
    with pytest.raises(SweepV1RetryError, match="not found"):
        load_frozen_proposal_record(tmp_path / "does_not_exist.json")


def test_load_frozen_proposal_record_raises_on_missing_required_field(tmp_path):
    written = _write_record(tmp_path)
    path = _record_path(tmp_path, written)
    record = json.loads(path.read_text(encoding="utf-8"))
    del record["wandb_sweep_id"]
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(SweepV1RetryError, match="missing required fields"):
        load_frozen_proposal_record(path)


def test_load_frozen_proposal_record_raises_on_tampered_trial_id(tmp_path):
    written = _write_record(tmp_path)
    path = _record_path(tmp_path, written)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["trial_id"] = "some_other_trial_id"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(SweepV1RetryError, match="trial_id"):
        load_frozen_proposal_record(path)


def test_load_frozen_proposal_record_raises_on_tampered_configuration_id(tmp_path):
    written = _write_record(tmp_path)
    path = _record_path(tmp_path, written)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["configuration_id"] = "some_other_configuration_id"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(SweepV1RetryError, match="configuration_id"):
        load_frozen_proposal_record(path)


def test_load_frozen_proposal_record_raises_on_foreign_campaign(tmp_path):
    written = _write_record(tmp_path)
    path = _record_path(tmp_path, written)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["campaign_id"] = "some_other_campaign"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(SweepV1RetryError, match="campaign/domain"):
        load_frozen_proposal_record(path)


# --- assert_matches_pinned_identity -------------------------------------------

def test_assert_matches_pinned_identity_passes_on_full_agreement(tmp_path):
    written = _write_record(tmp_path)
    pinned = {
        "proposal_order": written["proposal_order"], "proposal_id": written["proposal_id"],
        "configuration_id": written["configuration_id"], "trial_id": written["trial_id"],
        "search_arm": written["search_arm"], "wandb_sweep_id": written["wandb_sweep_id"],
        "model_seed": sweep.MODEL_SEED_A, **written["hyperparameters"],
    }
    assert_matches_pinned_identity(written, pinned)  # must not raise


def test_assert_matches_pinned_identity_raises_and_enumerates_hyperparameter_mismatch(tmp_path):
    written = _write_record(tmp_path)
    pinned = {"learning_rate": 9.9e-4, "hidden_size": 64}
    with pytest.raises(SweepV1RetryError) as excinfo:
        assert_matches_pinned_identity(written, pinned)
    message = str(excinfo.value)
    assert "hyperparameters.learning_rate" in message
    assert "hyperparameters.hidden_size" in message


def test_assert_matches_pinned_identity_raises_on_model_seed_mismatch(tmp_path):
    written = _write_record(tmp_path)
    with pytest.raises(SweepV1RetryError, match="model_seed"):
        assert_matches_pinned_identity(written, {"model_seed": sweep.MODEL_SEED_A + 1})


# --- derive_exact_retry_identity ----------------------------------------------

def test_derive_exact_retry_identity_happy_path(tmp_path):
    written = _write_record(tmp_path)
    retry = derive_exact_retry_identity(written, execution_generation=2)

    assert retry["hyperparameters"] == sweep.canonical_hyperparameters(_AXES)
    assert retry["configuration_id"] == written["configuration_id"]
    assert retry["proposal_id"] == written["proposal_id"]
    assert retry["search_arm"] == written["search_arm"]
    assert retry["proposal_order"] == written["proposal_order"]
    assert retry["execution_generation"] == 2
    assert retry["trial_id"] != written["trial_id"]
    assert retry["retry_of_trial_id"] == written["trial_id"]
    assert retry["wandb_sweep_id"] == written["wandb_sweep_id"]
    assert retry["trial_id"] == sweep.trial_id(written["configuration_id"], execution_generation=2)


def test_derive_exact_retry_identity_raises_when_generation_does_not_advance(tmp_path):
    written = _write_record(tmp_path)
    with pytest.raises(SweepV1RetryError, match="strictly exceed"):
        derive_exact_retry_identity(written, execution_generation=1)


def test_derive_exact_retry_identity_raises_on_non_integer_generation(tmp_path):
    written = _write_record(tmp_path)
    with pytest.raises(SweepV1RetryError, match="must be an integer"):
        derive_exact_retry_identity(written, execution_generation=True)
    with pytest.raises(SweepV1RetryError, match="must be an integer"):
        derive_exact_retry_identity(written, execution_generation=2.0)

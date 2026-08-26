"""Tests for the exact-retry identity seam in
``src.baseline.sweep_v2_six_axis_retry`` (Section G, additive six-axis
campaign foundation).

Mirrors tests/test_sweep_v1_retry.py's core coverage (load/recompute
identity, pinned-identity contradiction rejection, exact-retry derivation,
generation-reuse guard, bounded W&B tags), extended to the six-axis field
set and the fixed-support-contract identity fields v2's configuration
identity additionally binds. Unlike v1's test file, there is no real
production run history to draw fixtures from yet (v2 has never launched),
so every record here is built through the real
``write_proposal_intake_provenance_v2`` function rather than a hand-typed
"real production" fixture -- never a manufactured shortcut around the
canonical identity math.

Never imports wandb; never touches the filesystem beyond ``short_tmp_path``
(see tests/conftest.py for why plain ``tmp_path`` overflows Windows'
MAX_PATH for v2's longer trial_id strings); never starts NH training or
execution.
"""
from __future__ import annotations

import copy
import json

import pytest

from src.baseline import sweep_v1_campaign as sweep
from src.baseline.sweep_v2_six_axis_campaign import CAMPAIGN_ID_V2, DOMAIN_VERSION_V2
from src.baseline.sweep_v2_six_axis_execution import write_proposal_intake_provenance_v2
from src.baseline.sweep_v2_six_axis_retry import (
    MAX_WANDB_TAG_LENGTH,
    SweepV2RetryError,
    assert_generation_not_previously_attempted,
    assert_matches_pinned_identity_v2,
    build_bounded_wandb_tags_v2,
    derive_exact_retry_identity_v2,
    load_frozen_proposal_record_v2,
    validate_wandb_tags,
)

_AXES = {
    "learning_rate": 3e-4, "hidden_size": 128, "embedding_dropout": 0.10,
    "output_dropout": 0.25, "batch_size": 256, "seq_length": 96,
}
_SUPPORT_CONTRACT_VERSION = "common120_raw_space_nse_v001"
_SUPPORT_CONTRACT_SHA256 = "b" * 64


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_record(short_tmp_path, **overrides):
    kwargs = dict(
        output_root=short_tmp_path, axes=_AXES, search_arm="bayesian",
        proposal_order=1, wandb_sweep_id="v2-prod-sweep", wandb_run_id="run-1",
        support_contract_version=_SUPPORT_CONTRACT_VERSION, support_contract_sha256=_SUPPORT_CONTRACT_SHA256,
    )
    kwargs.update(overrides)
    return write_proposal_intake_provenance_v2(**kwargs)


def _record_path(short_tmp_path, record):
    return short_tmp_path / record["trial_id"] / "execution_provenance.json"


# --- load_frozen_proposal_record_v2 -------------------------------------------

def test_load_frozen_proposal_record_v2_happy_path_recomputes_identity(short_tmp_path):
    written = _write_record(short_tmp_path)
    loaded = load_frozen_proposal_record_v2(_record_path(short_tmp_path, written))
    assert loaded["configuration_id"] == written["configuration_id"]
    assert loaded["proposal_id"] == written["proposal_id"]
    assert loaded["trial_id"] == written["trial_id"]
    assert loaded["hyperparameters"]["seq_length"] == 96


def test_load_frozen_proposal_record_v2_raises_on_missing_file(short_tmp_path):
    with pytest.raises(SweepV2RetryError, match="not found"):
        load_frozen_proposal_record_v2(short_tmp_path / "does_not_exist.json")


def test_load_frozen_proposal_record_v2_raises_on_missing_required_field(short_tmp_path):
    written = _write_record(short_tmp_path)
    path = _record_path(short_tmp_path, written)
    record = json.loads(path.read_text(encoding="utf-8"))
    del record["support_contract_version"]
    _write_json(path, record)

    with pytest.raises(SweepV2RetryError, match="missing required fields"):
        load_frozen_proposal_record_v2(path)


def test_load_frozen_proposal_record_v2_raises_on_tampered_trial_id(short_tmp_path):
    written = _write_record(short_tmp_path)
    path = _record_path(short_tmp_path, written)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["trial_id"] = "some_other_trial_id"
    _write_json(path, record)

    with pytest.raises(SweepV2RetryError, match="trial_id"):
        load_frozen_proposal_record_v2(path)


def test_load_frozen_proposal_record_v2_raises_on_tampered_configuration_id(short_tmp_path):
    written = _write_record(short_tmp_path)
    path = _record_path(short_tmp_path, written)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["configuration_id"] = "sweep_v2_cfg_deadbeefdeadbeefdead0"
    _write_json(path, record)

    with pytest.raises(SweepV2RetryError, match="configuration_id"):
        load_frozen_proposal_record_v2(path)


def test_load_frozen_proposal_record_v2_raises_on_tampered_support_contract_identity(short_tmp_path):
    """support_contract_version/sha256 are bound into configuration_id_v2's
    hash -- tampering with either without updating configuration_id must be
    caught, exactly like tampering with a hyperparameter."""
    written = _write_record(short_tmp_path)
    path = _record_path(short_tmp_path, written)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["support_contract_sha256"] = "c" * 64
    _write_json(path, record)

    with pytest.raises(SweepV2RetryError, match="configuration_id"):
        load_frozen_proposal_record_v2(path)


def test_load_frozen_proposal_record_v2_raises_on_foreign_campaign(short_tmp_path):
    written = _write_record(short_tmp_path)
    path = _record_path(short_tmp_path, written)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["campaign_id"] = "some_other_campaign"
    _write_json(path, record)

    with pytest.raises(SweepV2RetryError, match="campaign/domain"):
        load_frozen_proposal_record_v2(path)


def test_load_frozen_proposal_record_v2_refuses_a_genuine_v1_record(short_tmp_path):
    """A real v1-shaped record (v1 campaign/domain, five-axis
    hyperparameters, no support-contract fields at all) must never be
    accepted through the v2 loader -- direct v1/v2 cross-contamination
    refusal."""
    v1_record = {
        "hyperparameters": {"learning_rate": 3e-4, "hidden_size": 128, "embedding_dropout": 0.10,
                             "output_dropout": 0.25, "batch_size": 256},
        "search_arm": "bayesian", "proposal_order": 1, "execution_generation": 1,
        "configuration_id": "sweep_v1_cfg_5731e180d1bf9d582afc",
        "proposal_id": "stage1_phase_b_sweep_v1_original_domain_v001__bayesian__proposal001",
        "trial_id": "stage1_phase_b_sweep_v1_original_domain_v001__sweep_v1_cfg_5731e180d1bf9d582afc__mf12x50000__seedA967139__attempt001",
        "campaign_id": sweep.CAMPAIGN_ID, "domain_version": sweep.DOMAIN_VERSION, "wandb_sweep_id": "4x3btz2s",
    }
    path = short_tmp_path / "execution_provenance.json"
    _write_json(path, v1_record)

    # A real v1 record never carries the two fixed-support-contract identity
    # fields v2 requires, so it is refused at the missing-required-fields
    # check (before even reaching the campaign/domain check below, which is
    # covered directly by test_load_frozen_proposal_record_v2_raises_on_foreign_campaign).
    with pytest.raises(SweepV2RetryError, match="missing required fields"):
        load_frozen_proposal_record_v2(path)


# --- assert_matches_pinned_identity_v2 -----------------------------------------

def test_assert_matches_pinned_identity_v2_passes_on_full_agreement(short_tmp_path):
    written = _write_record(short_tmp_path)
    pinned = {
        "proposal_order": written["proposal_order"], "proposal_id": written["proposal_id"],
        "configuration_id": written["configuration_id"], "trial_id": written["trial_id"],
        "search_arm": written["search_arm"], "wandb_sweep_id": written["wandb_sweep_id"],
        "support_contract_version": written["support_contract_version"],
        "support_contract_sha256": written["support_contract_sha256"],
        "model_seed": sweep.MODEL_SEED_A, **written["hyperparameters"],
    }
    assert_matches_pinned_identity_v2(written, pinned)  # must not raise


def test_assert_matches_pinned_identity_v2_raises_and_enumerates_seq_length_mismatch(short_tmp_path):
    written = _write_record(short_tmp_path)
    pinned = {"seq_length": 48, "hidden_size": 64}
    with pytest.raises(SweepV2RetryError) as excinfo:
        assert_matches_pinned_identity_v2(written, pinned)
    message = str(excinfo.value)
    assert "hyperparameters.seq_length" in message
    assert "hyperparameters.hidden_size" in message


def test_assert_matches_pinned_identity_v2_raises_on_support_contract_mismatch(short_tmp_path):
    written = _write_record(short_tmp_path)
    with pytest.raises(SweepV2RetryError, match="support_contract_sha256"):
        assert_matches_pinned_identity_v2(written, {"support_contract_sha256": "f" * 64})


def test_assert_matches_pinned_identity_v2_raises_on_model_seed_mismatch(short_tmp_path):
    written = _write_record(short_tmp_path)
    with pytest.raises(SweepV2RetryError, match="model_seed"):
        assert_matches_pinned_identity_v2(written, {"model_seed": sweep.MODEL_SEED_A + 1})


# --- derive_exact_retry_identity_v2 --------------------------------------------

def test_derive_exact_retry_identity_v2_happy_path(short_tmp_path):
    written = _write_record(short_tmp_path)
    retry = derive_exact_retry_identity_v2(written, execution_generation=2)

    assert retry["hyperparameters"]["seq_length"] == 96
    assert retry["configuration_id"] == written["configuration_id"]
    assert retry["proposal_id"] == written["proposal_id"]
    assert retry["search_arm"] == written["search_arm"]
    assert retry["proposal_order"] == written["proposal_order"]
    assert retry["execution_generation"] == 2
    assert retry["trial_id"] != written["trial_id"]
    assert retry["retry_of_trial_id"] == written["trial_id"]
    assert retry["wandb_sweep_id"] == written["wandb_sweep_id"]
    assert retry["support_contract_version"] == written["support_contract_version"]
    assert retry["support_contract_sha256"] == written["support_contract_sha256"]


def test_derive_exact_retry_identity_v2_raises_when_generation_does_not_advance(short_tmp_path):
    written = _write_record(short_tmp_path)
    with pytest.raises(SweepV2RetryError, match="strictly exceed"):
        derive_exact_retry_identity_v2(written, execution_generation=1)


def test_derive_exact_retry_identity_v2_raises_on_non_integer_generation(short_tmp_path):
    written = _write_record(short_tmp_path)
    with pytest.raises(SweepV2RetryError, match="must be an integer"):
        derive_exact_retry_identity_v2(written, execution_generation=True)
    with pytest.raises(SweepV2RetryError, match="must be an integer"):
        derive_exact_retry_identity_v2(written, execution_generation=2.0)


def test_derive_exact_retry_identity_v2_two_controller_proposals_never_collide(short_tmp_path):
    """Two different proposals landing on identical six-axis coordinates
    have different proposal_id but the same configuration_id; their
    retry-derived trial_ids must never collide -- trial_id_v2's whole
    collision-safety point."""
    first = _write_record(short_tmp_path, proposal_order=1, wandb_run_id="run-1")
    second = _write_record(short_tmp_path, proposal_order=2, wandb_run_id="run-2")
    assert first["configuration_id"] == second["configuration_id"]
    assert first["proposal_id"] != second["proposal_id"]

    retry_first = derive_exact_retry_identity_v2(first, execution_generation=2)
    retry_second = derive_exact_retry_identity_v2(second, execution_generation=2)
    assert retry_first["trial_id"] != retry_second["trial_id"]


def test_derive_exact_retry_identity_v2_refuses_to_reuse_a_reserved_generation(short_tmp_path):
    """assert_generation_not_previously_attempted is imported unchanged from
    v1 (genuinely generation/campaign-agnostic), so it raises v1's own
    SweepV1RetryError, not SweepV2RetryError."""
    from src.baseline.sweep_v1_retry import SweepV1RetryError

    written = _write_record(short_tmp_path)
    prior_attempts = [{"execution_generation": 2, "slurm_job_id": "99999999", "status": "failed_before_wandb_association"}]
    with pytest.raises(SweepV1RetryError, match="already reserved"):
        derive_exact_retry_identity_v2(written, execution_generation=2, prior_attempts=prior_attempts)


def test_assert_generation_not_previously_attempted_reused_directly_from_v1():
    """Confirms this function is imported unchanged from sweep_v1_retry (a
    genuinely axis-agnostic helper) rather than re-implemented."""
    from src.baseline.sweep_v1_retry import assert_generation_not_previously_attempted as v1_fn
    assert assert_generation_not_previously_attempted is v1_fn


# --- executed-attempt envelope normalization (synthetic v2-shaped envelope) ---

def test_load_frozen_proposal_record_v2_accepts_executed_attempt_envelope(short_tmp_path):
    written = _write_record(short_tmp_path)
    flat = json.loads(_record_path(short_tmp_path, written).read_text(encoding="utf-8"))
    envelope = {
        "campaign_id": flat["campaign_id"], "configuration_id": flat["configuration_id"],
        "execution_generation": flat["execution_generation"], "execution_status": "INVALID",
        "objective_score": None, "preparation_record": flat, "proposal_id": flat["proposal_id"],
        "retry_of_trial_id": None, "search_arm": flat["search_arm"], "trial_id": flat["trial_id"],
        "result": {"blocked": True},
    }
    path = short_tmp_path / "envelope_execution_provenance.json"
    _write_json(path, envelope)

    loaded = load_frozen_proposal_record_v2(path)
    assert loaded["trial_id"] == flat["trial_id"]
    assert loaded["configuration_id"] == flat["configuration_id"]


def test_load_frozen_proposal_record_v2_rejects_outer_nested_identity_contradiction(short_tmp_path):
    """This shape-level check happens inside the reused (genuinely
    axis-agnostic) ``_normalize_frozen_record`` helper, upstream of any
    v2-specific validation, so the raised type is v1's own
    ``SweepV1RetryError`` -- not wrapped into ``SweepV2RetryError`` -- per
    the module's documented reuse-vs-sibling convention."""
    from src.baseline.sweep_v1_retry import SweepV1RetryError

    written = _write_record(short_tmp_path)
    flat = json.loads(_record_path(short_tmp_path, written).read_text(encoding="utf-8"))
    envelope = {
        "campaign_id": flat["campaign_id"], "configuration_id": "sweep_v2_cfg_deadbeefdeadbeefdead0",
        "execution_generation": flat["execution_generation"], "execution_status": "INVALID",
        "objective_score": None, "preparation_record": flat, "proposal_id": flat["proposal_id"],
        "retry_of_trial_id": None, "search_arm": flat["search_arm"], "trial_id": flat["trial_id"],
    }
    path = short_tmp_path / "envelope_execution_provenance.json"
    _write_json(path, envelope)

    with pytest.raises(SweepV1RetryError, match="contradicts its own nested preparation_record"):
        load_frozen_proposal_record_v2(path)


def test_derive_exact_retry_identity_v2_from_envelope(short_tmp_path):
    written = _write_record(short_tmp_path)
    flat = json.loads(_record_path(short_tmp_path, written).read_text(encoding="utf-8"))
    envelope = {
        "campaign_id": flat["campaign_id"], "configuration_id": flat["configuration_id"],
        "execution_generation": flat["execution_generation"], "execution_status": "INVALID",
        "objective_score": None, "preparation_record": flat, "proposal_id": flat["proposal_id"],
        "retry_of_trial_id": None, "search_arm": flat["search_arm"], "trial_id": flat["trial_id"],
    }
    path = short_tmp_path / "envelope_execution_provenance.json"
    _write_json(path, envelope)
    loaded = load_frozen_proposal_record_v2(path)

    retry = derive_exact_retry_identity_v2(loaded, execution_generation=2)

    assert retry["execution_generation"] == 2
    assert retry["retry_of_trial_id"] == flat["trial_id"]
    assert retry["trial_id"] != flat["trial_id"]
    assert retry["configuration_id"] == flat["configuration_id"]
    assert retry["proposal_id"] == flat["proposal_id"]


# --- bounded W&B tags -----------------------------------------------------------

def test_build_bounded_wandb_tags_v2_are_all_within_the_max_length(short_tmp_path):
    written = _write_record(short_tmp_path)
    tags = build_bounded_wandb_tags_v2(
        proposal_order=1, execution_generation=2, configuration_id=written["configuration_id"],
    )
    assert all(1 <= len(tag) <= MAX_WANDB_TAG_LENGTH for tag in tags)


def test_build_bounded_wandb_tags_v2_matches_expected_set_and_differs_from_v1_label():
    tags = build_bounded_wandb_tags_v2(
        proposal_order=1, execution_generation=3, configuration_id="sweep_v2_cfg_deadbeefdeadbeefdead0",
    )
    assert tags == [
        "sweep-v2-six-axis", "exact-retry", "proposal-001", "execution-generation-3",
        "sweep_v2_cfg_deadbeefdeadbeefdead0",
    ]
    assert "sweep-v1" not in tags
    validate_wandb_tags(tags)  # must not raise


def test_validate_wandb_tags_reused_directly_from_v1():
    from src.baseline.sweep_v1_retry import validate_wandb_tags as v1_fn
    assert validate_wandb_tags is v1_fn

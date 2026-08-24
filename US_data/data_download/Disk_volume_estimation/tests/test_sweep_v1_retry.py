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

Executed-attempt envelope coverage (the real, post-execution
``execution_provenance.json`` shape ``execute_prepared_trial`` writes, with
identity nested under ``preparation_record``):
  13. A realistic executed-attempt envelope loads successfully.
  14. The normalized identity exactly matches the nested ``preparation_record``.
  15. An outer/nested identity contradiction is rejected.
  16. A non-mapping ``preparation_record`` is rejected.
  17. A ``preparation_record`` missing a required field is rejected.
  18. The real attempt001 golden fixture (INVALID, null objective) loads and
      recovers its frozen proposal identity.
  19. A VALID envelope (non-null outer terminal objective) also loads, and
      the normalized record's own ``objective_score`` is the nested
      (always-null-at-intake) value, never the outer terminal one.
  20. ``derive_exact_retry_identity`` fed a normalized envelope record
      produces the expected fresh attempt002 identity.

Never imports wandb; never touches the filesystem beyond ``tmp_path``; never
starts NH training or execution.
"""
from __future__ import annotations

import copy
import json

import pytest

from src.baseline import sweep_v1_campaign as sweep
from src.baseline.sweep_v1_execution import write_proposal_intake_provenance
from src.baseline.sweep_v1_retry import (
    MAX_WANDB_TAG_LENGTH, SweepV1RetryError, assert_generation_not_previously_attempted,
    assert_matches_pinned_identity, build_bounded_wandb_tags, derive_exact_retry_identity,
    load_frozen_proposal_record, validate_wandb_tags,
)

_AXES = {"learning_rate": 3e-4, "hidden_size": 128, "embedding_dropout": 0.10, "output_dropout": 0.25, "batch_size": 256}

# Sanitized golden fixture matching the REAL Sweep-v1 production attempt001
# ``execution_provenance.json`` envelope shape (fetched directly off Moriah,
# baseline sha256 4da96c34dcaee4cad83c9371c62fccbf609fad05985f8b56e4d69f3633afb052).
# Contains the real non-secret identity/hyperparameter values and real HPC
# paths -- no credentials or machine-private secrets are present in this
# record. Only ``expected_output_dir``/``generated_nh_config_path``/
# ``generation_manifest_path``/``nh_run_dir`` (plain evidence-tree paths, not
# secrets) are reproduced verbatim; nothing here is a token, password, or key.
_REAL_ATTEMPT001_PREPARATION_RECORD = {
    "artifact_identity_status": "PASS",
    "authoritative_screening_epochs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "campaign_id": "stage1_phase_b_sweep_v1_original_domain_v001",
    "configuration_id": "sweep_v1_cfg_5731e180d1bf9d582afc",
    "development_split_sha256": "397ab432564c18c3abc5158a47ada2b28840bbf6f0c213d2475444fded33858f",
    "domain_version": "original_domain_v001",
    "evaluation_scope": "development_validation_2024_only",
    "execution_generation": 1,
    "expected_output_dir": (
        "/sci/labs/efratmorin/omripo/Flash-NH/evidence/sweep_v1_bayesian_production_v001/"
        "stage1_phase_b_sweep_v1_original_domain_v001__sweep_v1_cfg_5731e180d1bf9d582afc__mf12x50000__seedA967139__attempt001"
    ),
    "fidelity_id": "mf12x50000",
    "generated_nh_config_path": (
        "/sci/labs/efratmorin/omripo/Flash-NH/evidence/sweep_v1_bayesian_production_v001/"
        "stage1_phase_b_sweep_v1_original_domain_v001__sweep_v1_cfg_5731e180d1bf9d582afc__mf12x50000__seedA967139__attempt001/config.yaml"
    ),
    "generated_nh_config_sha256": "3bfe06e5f784153630abcad935e6defcebfaeb3f8566e265c8eba403ebd68f77",
    "generation_manifest_path": (
        "/sci/labs/efratmorin/omripo/Flash-NH/evidence/sweep_v1_bayesian_production_v001/"
        "stage1_phase_b_sweep_v1_original_domain_v001__sweep_v1_cfg_5731e180d1bf9d582afc__mf12x50000__seedA967139__attempt001/generation_manifest.json"
    ),
    "hyperparameters": {
        "batch_size": 512,
        "embedding_dropout": "0.08637149503762416",
        "hidden_size": 64,
        "learning_rate": "0.00024474898782657741",
        "output_dropout": "0.20096018948154892",
    },
    "max_updates_per_epoch": 50000,
    "model_seed": 967139,
    "objective_score": None,
    "package_file_checksums_sha256": "83b47374725d418b130a8e28dcf1cb118cee88f99624907238e25ee2a9067d13",
    "package_identity": "stage1_scientific_package_v002",
    "package_manifest_sha256": "6c52fb1b81f6a5f730b805d0c273e9d00cbf5bb93d1cd0da58452f5a0e5bcc4a",
    "package_run_provenance_sha256": "030de2f9458aa40deba74d84910904f02468adb9eb1786ee3a71556bfcb11a8b",
    "performance_early_stopping_enabled": False,
    "prepare_only": True,
    "prepare_status": "PASS",
    "proposal_id": "stage1_phase_b_sweep_v1_original_domain_v001__bayesian__proposal001",
    "proposal_order": 1,
    "save_weights_every": 1,
    "screening_artifact_sha256": "d4395d93ebc567cf09e149c0121463d75cf4f7ecc02c07a7c4a7999763baa372",
    "screening_policy_identity": "stage1_provisional_operational_screening_subset_v001",
    "sealed_scope": False,
    "search_arm": "bayesian",
    "spatial_holdout_split_sha256": "76d1c546e703b1b5aa8f4a3ead971327de0151dae4fcce0c90b1272da0f587b7",
    "target_epoch": 12,
    "trial_id": (
        "stage1_phase_b_sweep_v1_original_domain_v001__sweep_v1_cfg_5731e180d1bf9d582afc__mf12x50000__seedA967139__attempt001"
    ),
    "wandb_run_id": "7nxaim79",
    "wandb_sweep_id": "4x3btz2s",
}

_REAL_ATTEMPT001_ENVELOPE = {
    "campaign_id": "stage1_phase_b_sweep_v1_original_domain_v001",
    "configuration_id": "sweep_v1_cfg_5731e180d1bf9d582afc",
    "execution_generation": 1,
    "execution_status": "INVALID",
    "generated_nh_config_path": _REAL_ATTEMPT001_PREPARATION_RECORD["generated_nh_config_path"],
    "generated_nh_config_sha256": "3bfe06e5f784153630abcad935e6defcebfaeb3f8566e265c8eba403ebd68f77",
    "git_commit": "d8098ee5267e96e4e1b4c8246c210be376760eef",
    "objective_score": None,
    "preparation_record": _REAL_ATTEMPT001_PREPARATION_RECORD,
    "proposal_id": "stage1_phase_b_sweep_v1_original_domain_v001__bayesian__proposal001",
    "result": {
        "blocked": True,
        "blocked_reason": "cannot safely continue training from epoch 1 to 2: refusing duplicate physical claim",
        "checkpoint_epochs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "final_status": "blocked_continuation_overshoot_conflict",
        "screening_epochs": [1],
        "stop_reason": None,
        "stopped": False,
    },
    "retry_of_trial_id": None,
    "search_arm": "bayesian",
    "trial_id": (
        "stage1_phase_b_sweep_v1_original_domain_v001__sweep_v1_cfg_5731e180d1bf9d582afc__mf12x50000__seedA967139__attempt001"
    ),
}


def _write_json(tmp_path, name, payload) -> "object":
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


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


# --- executed-attempt envelope normalization ----------------------------------

def test_load_frozen_proposal_record_accepts_executed_attempt_envelope(tmp_path):
    envelope = copy.deepcopy(_REAL_ATTEMPT001_ENVELOPE)
    path = _write_json(tmp_path, "execution_provenance.json", envelope)

    loaded = load_frozen_proposal_record(path)

    assert loaded["trial_id"] == _REAL_ATTEMPT001_PREPARATION_RECORD["trial_id"]
    assert loaded["configuration_id"] == "sweep_v1_cfg_5731e180d1bf9d582afc"
    assert loaded["proposal_id"] == "stage1_phase_b_sweep_v1_original_domain_v001__bayesian__proposal001"


def test_load_frozen_proposal_record_envelope_identity_matches_nested_preparation_record_exactly(tmp_path):
    envelope = copy.deepcopy(_REAL_ATTEMPT001_ENVELOPE)
    path = _write_json(tmp_path, "execution_provenance.json", envelope)

    loaded = load_frozen_proposal_record(path)

    for key, value in _REAL_ATTEMPT001_PREPARATION_RECORD.items():
        assert loaded[key] == value, key


def test_load_frozen_proposal_record_rejects_outer_nested_identity_contradiction(tmp_path):
    envelope = copy.deepcopy(_REAL_ATTEMPT001_ENVELOPE)
    envelope["configuration_id"] = "sweep_v1_cfg_deadbeefdeadbeefdead"
    path = _write_json(tmp_path, "execution_provenance.json", envelope)

    with pytest.raises(SweepV1RetryError, match="contradicts its own nested preparation_record"):
        load_frozen_proposal_record(path)


def test_load_frozen_proposal_record_rejects_non_mapping_preparation_record(tmp_path):
    envelope = copy.deepcopy(_REAL_ATTEMPT001_ENVELOPE)
    envelope["preparation_record"] = "not-an-object"
    path = _write_json(tmp_path, "execution_provenance.json", envelope)

    with pytest.raises(SweepV1RetryError, match="not a JSON object"):
        load_frozen_proposal_record(path)


def test_load_frozen_proposal_record_rejects_envelope_missing_required_nested_field(tmp_path):
    envelope = copy.deepcopy(_REAL_ATTEMPT001_ENVELOPE)
    del envelope["preparation_record"]["wandb_sweep_id"]
    path = _write_json(tmp_path, "execution_provenance.json", envelope)

    with pytest.raises(SweepV1RetryError, match="missing required fields"):
        load_frozen_proposal_record(path)


def test_load_frozen_proposal_record_real_attempt001_golden_fixture_invalid_status(tmp_path):
    """attempt001's real INVALID execution_status and null outer objective_score
    must not prevent recovering the frozen proposal identity -- the outer
    terminal fields are never load-bearing for identity."""
    envelope = copy.deepcopy(_REAL_ATTEMPT001_ENVELOPE)
    assert envelope["execution_status"] == "INVALID"
    assert envelope["objective_score"] is None
    path = _write_json(tmp_path, "execution_provenance.json", envelope)

    loaded = load_frozen_proposal_record(path)

    assert loaded["proposal_order"] == 1
    assert loaded["model_seed"] == 967139
    assert loaded["hyperparameters"] == {
        "batch_size": 512, "embedding_dropout": "0.08637149503762416", "hidden_size": 64,
        "learning_rate": "0.00024474898782657741", "output_dropout": "0.20096018948154892",
    }


def test_load_frozen_proposal_record_valid_envelope_keeps_nested_null_objective_not_outer_terminal_value(tmp_path):
    """A VALID envelope's outer objective_score legitimately diverges (a real
    finite score) from the nested, always-null-at-intake objective_score --
    the normalized identity must keep the nested (frozen) value, never the
    outer terminal one."""
    envelope = copy.deepcopy(_REAL_ATTEMPT001_ENVELOPE)
    envelope["execution_status"] = "VALID"
    envelope["objective_score"] = 0.412  # outer terminal result -- must not leak into identity
    path = _write_json(tmp_path, "execution_provenance.json", envelope)

    loaded = load_frozen_proposal_record(path)

    assert loaded["objective_score"] is None
    assert loaded["trial_id"] == _REAL_ATTEMPT001_PREPARATION_RECORD["trial_id"]
    assert loaded["configuration_id"] == "sweep_v1_cfg_5731e180d1bf9d582afc"


def test_derive_exact_retry_identity_from_real_attempt001_envelope(tmp_path):
    envelope = copy.deepcopy(_REAL_ATTEMPT001_ENVELOPE)
    path = _write_json(tmp_path, "execution_provenance.json", envelope)
    loaded = load_frozen_proposal_record(path)

    retry = derive_exact_retry_identity(loaded, execution_generation=2)

    assert retry["execution_generation"] == 2
    assert retry["retry_of_trial_id"] == _REAL_ATTEMPT001_PREPARATION_RECORD["trial_id"]
    assert retry["trial_id"] != _REAL_ATTEMPT001_PREPARATION_RECORD["trial_id"]
    assert retry["configuration_id"] == "sweep_v1_cfg_5731e180d1bf9d582afc"
    assert retry["proposal_id"] == "stage1_phase_b_sweep_v1_original_domain_v001__bayesian__proposal001"
    assert retry["proposal_order"] == 1
    assert retry["hyperparameters"] == _REAL_ATTEMPT001_PREPARATION_RECORD["hyperparameters"]
    assert retry["trial_id"] == sweep.trial_id("sweep_v1_cfg_5731e180d1bf9d582afc", execution_generation=2)


# --- attempt002/job 45939764 incident: generation-reuse guard + attempt003 derivation ----

# The real, historical operator-authored prior-attempts record for the
# attempt002/job 45939764 incident: a failed attempt that crashed inside
# wandb.init()'s own tag validation before any durable per-trial evidence of
# its own was ever written, so its only durable trace is this operator
# record (same trust model as pinned-identity).
_REAL_PRIOR_ATTEMPTS_AFTER_ATTEMPT002 = [
    {
        "execution_generation": 2,
        "slurm_job_id": "45939764",
        "source_commit": "12efb3e",
        "status": "failed_before_wandb_association",
        "failure_category": "wandb_tag_length_validation",
    }
]


def test_assert_generation_not_previously_attempted_passes_when_generation_is_fresh():
    assert_generation_not_previously_attempted(3, _REAL_PRIOR_ATTEMPTS_AFTER_ATTEMPT002)  # must not raise


def test_assert_generation_not_previously_attempted_raises_when_generation_2_is_reused():
    with pytest.raises(SweepV1RetryError, match="already reserved"):
        assert_generation_not_previously_attempted(2, _REAL_PRIOR_ATTEMPTS_AFTER_ATTEMPT002)


def test_derive_exact_retry_identity_from_real_attempt001_envelope_to_attempt003(tmp_path):
    """Real attempt001 envelope -> attempt003 derivation (execution_generation=3),
    with the real attempt002/job 45939764 prior-attempts record supplied --
    the direct real-record analogue of the attempt003 launch this repair
    exists to support."""
    envelope = copy.deepcopy(_REAL_ATTEMPT001_ENVELOPE)
    path = _write_json(tmp_path, "execution_provenance.json", envelope)
    loaded = load_frozen_proposal_record(path)

    retry = derive_exact_retry_identity(
        loaded, execution_generation=3, prior_attempts=_REAL_PRIOR_ATTEMPTS_AFTER_ATTEMPT002,
    )

    assert retry["execution_generation"] == 3
    assert retry["retry_of_trial_id"] == _REAL_ATTEMPT001_PREPARATION_RECORD["trial_id"]
    assert retry["trial_id"] != _REAL_ATTEMPT001_PREPARATION_RECORD["trial_id"]
    assert retry["trial_id"] == sweep.trial_id("sweep_v1_cfg_5731e180d1bf9d582afc", execution_generation=3)
    assert retry["configuration_id"] == "sweep_v1_cfg_5731e180d1bf9d582afc"
    assert retry["proposal_id"] == "stage1_phase_b_sweep_v1_original_domain_v001__bayesian__proposal001"
    assert retry["proposal_order"] == 1
    assert retry["hyperparameters"] == _REAL_ATTEMPT001_PREPARATION_RECORD["hyperparameters"]
    assert retry["wandb_sweep_id"] == "4x3btz2s"


def test_derive_exact_retry_identity_refuses_to_reuse_generation_2_after_attempt002(tmp_path):
    """Generation 2 (attempt002's own reserved generation) must never be
    reused for a new attempt, even though attempt002 itself left no output
    directory on disk -- the only detection mechanism is the operator-
    supplied prior-attempts record."""
    envelope = copy.deepcopy(_REAL_ATTEMPT001_ENVELOPE)
    path = _write_json(tmp_path, "execution_provenance.json", envelope)
    loaded = load_frozen_proposal_record(path)

    with pytest.raises(SweepV1RetryError, match="already reserved"):
        derive_exact_retry_identity(
            loaded, execution_generation=2, prior_attempts=_REAL_PRIOR_ATTEMPTS_AFTER_ATTEMPT002,
        )


def test_derive_exact_retry_identity_defaults_to_no_prior_attempts(tmp_path):
    """Omitting prior_attempts entirely (the first retry ever, generation 2)
    must behave exactly as before this repair -- no forced argument, no
    behavior change for the common case."""
    written = _write_record(tmp_path)
    retry = derive_exact_retry_identity(written, execution_generation=2)
    assert retry["execution_generation"] == 2


# --- Repair 1: bounded W&B tags (attempt002/job 45939764 tag-length incident) ----

# The real, historical offending tag from job 45939764: "retry_of_" plus
# attempt001's own real trial_id -- reproduced here (not invented) to prove
# the new bounded tag scheme replaces exactly this failure mode.
_REAL_ATTEMPT001_TRIAL_ID = _REAL_ATTEMPT001_PREPARATION_RECORD["trial_id"]
_REAL_OFFENDING_ATTEMPT002_TAG = f"retry_of_{_REAL_ATTEMPT001_TRIAL_ID}"


def test_real_offending_attempt002_tag_exceeds_the_max_length():
    """Confirms the historical failure mode: the OLD tag construction really
    does exceed MAX_WANDB_TAG_LENGTH for the real attempt001 identity."""
    assert len(_REAL_OFFENDING_ATTEMPT002_TAG) > MAX_WANDB_TAG_LENGTH


def test_build_bounded_wandb_tags_are_all_within_the_max_length():
    tags = build_bounded_wandb_tags(
        proposal_order=1, execution_generation=3, configuration_id="sweep_v1_cfg_5731e180d1bf9d582afc",
    )
    assert all(1 <= len(tag) <= MAX_WANDB_TAG_LENGTH for tag in tags)


def test_build_bounded_wandb_tags_real_attempt003_identity_matches_expected_set():
    tags = build_bounded_wandb_tags(
        proposal_order=1, execution_generation=3, configuration_id="sweep_v1_cfg_5731e180d1bf9d582afc",
    )
    assert tags == [
        "sweep-v1", "exact-retry", "proposal-001", "execution-generation-3", "sweep_v1_cfg_5731e180d1bf9d582afc",
    ]


def test_validate_wandb_tags_accepts_boundary_63_and_64_char_tags():
    validate_wandb_tags(["a" * 63, "b" * 64])  # must not raise


def test_validate_wandb_tags_rejects_boundary_65_char_tag():
    with pytest.raises(SweepV1RetryError, match=r"1-64 character contract"):
        validate_wandb_tags(["a" * 65])


def test_validate_wandb_tags_rejects_empty_tag():
    with pytest.raises(SweepV1RetryError, match=r"1-64 character contract"):
        validate_wandb_tags([""])


def test_validate_wandb_tags_rejects_the_real_historical_offending_tag():
    with pytest.raises(SweepV1RetryError, match=r"1-64 character contract"):
        validate_wandb_tags(["exact_retry", _REAL_OFFENDING_ATTEMPT002_TAG])


def test_validate_wandb_tags_accepts_the_production_bounded_tag_set_for_real_attempt003_identity():
    tags = build_bounded_wandb_tags(
        proposal_order=1, execution_generation=3, configuration_id="sweep_v1_cfg_5731e180d1bf9d582afc",
    )
    validate_wandb_tags(tags)  # must not raise


# --- Repair 2: retry_history carried in durable provenance, never overloading retry_of_trial_id ----

def test_write_proposal_intake_provenance_persists_retry_history_without_touching_retry_of_trial_id(tmp_path):
    written = _write_record(
        tmp_path, execution_generation=3, retry_of_trial_id=_REAL_ATTEMPT001_TRIAL_ID,
        retry_history=_REAL_PRIOR_ATTEMPTS_AFTER_ATTEMPT002,
    )
    assert written["retry_of_trial_id"] == _REAL_ATTEMPT001_TRIAL_ID
    assert written["retry_history"] == _REAL_PRIOR_ATTEMPTS_AFTER_ATTEMPT002


def test_write_proposal_intake_provenance_defaults_retry_history_to_empty_list(tmp_path):
    written = _write_record(tmp_path)
    assert written["retry_history"] == []

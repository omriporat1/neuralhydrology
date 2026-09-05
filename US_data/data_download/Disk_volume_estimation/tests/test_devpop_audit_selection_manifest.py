"""Local tests for the SHARED-A4 frozen seven-checkpoint selection manifest
(:mod:`src.baseline.devpop_audit_selection_manifest`).

Synthetic entries only -- this never populates or references the real
seven-run production manifest (that is SHARED-A5)."""
from __future__ import annotations

import hashlib

import pytest

from src.baseline.devpop_common120_audit_evaluator import DevpopAuditCheckpointIdentity
from src.baseline.sweep_v2_six_axis_campaign import CAMPAIGN_ID_V2, OBJECTIVE_ID_V2
from src.baseline.devpop_audit_selection_manifest import (
    DevpopAuditSelectionManifestError,
    build_devpop_audit_selection_manifest_entry,
    compute_devpop_audit_selection_manifest_sha256,
    load_devpop_audit_selection_manifest,
    selection_manifest_entry_to_checkpoint_identity,
    validate_devpop_audit_selection_manifest,
    validate_devpop_audit_selection_manifest_entry,
    write_devpop_audit_selection_manifest,
)

_SUPPORT_VERSION = "common120_raw_space_nse_v001"
_SUPPORT_SHA256 = "f" * 64


def _hyperparameters(**changes):
    value = {
        "learning_rate": 3e-4, "hidden_size": 128, "embedding_dropout": 0.10,
        "output_dropout": 0.25, "batch_size": 256, "seq_length": 96,
    }
    value.update(changes)
    return value


def _entry(*, search_arm: str, proposal_order: int, seq_length: int, epoch: int = 3, **changes) -> dict:
    ckpt_sha256 = hashlib.sha256(f"ckpt-{search_arm}-{proposal_order}".encode()).hexdigest()
    kwargs = dict(
        search_arm=search_arm,
        proposal_order=proposal_order,
        hyperparameters=_hyperparameters(seq_length=seq_length),
        screening_score=0.62,
        screening_best_epoch=epoch,
        screening_evidence_path=f"reports/screening/{search_arm}_{proposal_order}.json",
        source_run_dir=f"/scratch/runs/{search_arm}_{proposal_order}",
        checkpoint_filename=f"model_epoch{epoch:03d}.pt",
        checkpoint_sha256=ckpt_sha256,
        selection_policy="frozen_screening_best_epoch_v001",
        support_contract_version=_SUPPORT_VERSION,
        support_contract_sha256=_SUPPORT_SHA256,
    )
    kwargs.update(changes)
    return build_devpop_audit_selection_manifest_entry(**kwargs)


def _seven_entries() -> list:
    # Distinct seq_length per slot (48-120h domain has exactly 7 legal steps)
    # so no two of the seven entries coincidentally share a configuration_id.
    entries = []
    for order, seq_length in zip((1, 2, 3), (72, 84, 96)):
        entries.append(_entry(search_arm="bayesian", proposal_order=order, seq_length=seq_length))
    for order, seq_length in zip((1, 2, 3, 4), (48, 60, 108, 120)):
        entries.append(_entry(search_arm="random_control", proposal_order=order, seq_length=seq_length))
    return entries


# --------------------------------------------------------------------------- #
# construction + identity
# --------------------------------------------------------------------------- #

def test_entry_identity_matches_v2_grammar():
    entry = _entry(search_arm="bayesian", proposal_order=1, seq_length=96)
    assert entry["configuration_id"].startswith("sweep_v2_cfg_")
    assert entry["proposal_id"] == f"{CAMPAIGN_ID_V2}__bayesian__proposal001"
    assert entry["trial_id"].startswith(entry["proposal_id"])
    assert entry["objective_id"] == OBJECTIVE_ID_V2


def test_seven_entry_manifest_validates_and_pins_identity():
    entries = _seven_entries()
    manifest = validate_devpop_audit_selection_manifest(entries)
    assert manifest["manifest_sha256"] == compute_devpop_audit_selection_manifest_sha256(entries)
    # order-independence of the identity hash
    reordered = list(reversed(entries))
    assert compute_devpop_audit_selection_manifest_sha256(reordered) == manifest["manifest_sha256"]


def test_bridges_to_existing_checkpoint_identity_class():
    entry = _entry(search_arm="bayesian", proposal_order=1, seq_length=96, epoch=7)
    identity = selection_manifest_entry_to_checkpoint_identity(entry)
    assert isinstance(identity, DevpopAuditCheckpointIdentity)
    assert identity.trial_id == entry["trial_id"]
    assert identity.configuration_id == entry["configuration_id"]
    assert identity.checkpoint_epoch == 7
    assert identity.checkpoint_sha256 == entry["checkpoint_sha256"]


# --------------------------------------------------------------------------- #
# write / load / tamper
# --------------------------------------------------------------------------- #

def test_write_then_load_round_trips(tmp_path):
    entries = _seven_entries()
    path = write_devpop_audit_selection_manifest(entries, tmp_path / "manifest.json")
    loaded = load_devpop_audit_selection_manifest(path)
    assert loaded["manifest_sha256"] == compute_devpop_audit_selection_manifest_sha256(entries)
    assert len(loaded["entries"]) == 7


def test_write_refuses_overwrite(tmp_path):
    entries = _seven_entries()
    path = tmp_path / "manifest.json"
    write_devpop_audit_selection_manifest(entries, path)
    with pytest.raises(DevpopAuditSelectionManifestError):
        write_devpop_audit_selection_manifest(entries, path)


def test_load_rejects_tampered_manifest(tmp_path):
    entries = _seven_entries()
    path = write_devpop_audit_selection_manifest(entries, tmp_path / "manifest.json")
    text = path.read_text(encoding="utf-8")
    tampered = text.replace('"screening_score": 0.62', '"screening_score": 0.99')
    assert tampered != text
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(DevpopAuditSelectionManifestError):
        load_devpop_audit_selection_manifest(path)


# --------------------------------------------------------------------------- #
# fail-closed: composition / count
# --------------------------------------------------------------------------- #

def test_rejects_wrong_entry_count():
    entries = _seven_entries()[:6]
    with pytest.raises(DevpopAuditSelectionManifestError, match="exactly 7"):
        validate_devpop_audit_selection_manifest(entries)


def test_rejects_wrong_arm_composition():
    entries = _seven_entries()
    entries[0] = _entry(search_arm="random_control", proposal_order=5, seq_length=96)
    with pytest.raises(DevpopAuditSelectionManifestError, match="composition"):
        validate_devpop_audit_selection_manifest(entries)


def test_rejects_duplicate_configuration_id_across_distinct_slots():
    # Two coincidentally-identical hyperparameter coordinates in otherwise
    # distinct, validly-composed slots (different search_arm/proposal_order,
    # hence different proposal_id/trial_id) must still be rejected --
    # duplicate scientific configuration identity, not just duplicate slot.
    entries = _seven_entries()
    same_seq_length = entries[2]["hyperparameters"]["seq_length"]
    entries[3] = _entry(search_arm="random_control", proposal_order=4, seq_length=same_seq_length)
    assert entries[3]["configuration_id"] == entries[2]["configuration_id"]
    assert entries[3]["trial_id"] != entries[2]["trial_id"]
    with pytest.raises(DevpopAuditSelectionManifestError):
        validate_devpop_audit_selection_manifest(entries)


# --------------------------------------------------------------------------- #
# fail-closed: per-entry fields
# --------------------------------------------------------------------------- #

def test_rejects_non_finite_screening_score():
    with pytest.raises(DevpopAuditSelectionManifestError):
        _entry(search_arm="bayesian", proposal_order=1, seq_length=96, screening_score=float("nan"))


def test_rejects_invalid_checkpoint_epoch():
    with pytest.raises(DevpopAuditSelectionManifestError):
        _entry(search_arm="bayesian", proposal_order=1, seq_length=96, screening_best_epoch=0)


def test_rejects_epoch_not_bound_to_checkpoint_filename():
    # Defect 2: an entry may not claim screening_best_epoch N while naming
    # checkpoint bytes from a different epoch -- checkpoint_filename must be
    # exactly f"{weight_stem(screening_best_epoch)}.pt". Checked at
    # entry-validation time, before any producer copy/rename.
    with pytest.raises(DevpopAuditSelectionManifestError, match="checkpoint_filename"):
        _entry(
            search_arm="bayesian", proposal_order=1, seq_length=96,
            screening_best_epoch=5, checkpoint_filename="model_epoch012.pt",
        )


def test_rejects_malformed_checkpoint_sha256():
    with pytest.raises(DevpopAuditSelectionManifestError):
        _entry(search_arm="bayesian", proposal_order=1, seq_length=96, checkpoint_sha256="not-a-sha256")


def test_rejects_unknown_search_arm():
    with pytest.raises(DevpopAuditSelectionManifestError):
        _entry(search_arm="grid_search", proposal_order=1, seq_length=96)


def test_rejects_missing_evidence_field():
    with pytest.raises(DevpopAuditSelectionManifestError):
        _entry(search_arm="bayesian", proposal_order=1, seq_length=96, selection_policy="")


def test_rejects_configuration_id_conflicting_with_different_support_contract():
    entry = _entry(search_arm="bayesian", proposal_order=1, seq_length=96)
    with pytest.raises(DevpopAuditSelectionManifestError):
        validate_devpop_audit_selection_manifest_entry(
            entry, support_contract_version=_SUPPORT_VERSION, support_contract_sha256="0" * 64,
        )

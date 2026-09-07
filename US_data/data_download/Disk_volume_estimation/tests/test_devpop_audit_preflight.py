"""SHARED-A5 Workstream A: configuration-generic per-entry audit preflight
(:mod:`src.baseline.devpop_audit_preflight`).

Proves the preflight derives every configuration-specific architecture fact
from the validated frozen manifest entry + the authoritative frozen v2
configuration -- never from P1 constants -- and fails closed on any
inconsistency.  Synthetic manifests only; the real seven-entry frozen manifest
is exercised through *pure* validation logic when its (untracked) A5 bundle is
present.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.baseline.devpop_audit_preflight import (
    DevpopAuditPreflightError,
    campaign_frozen_audit_facts,
    derive_expected_preflight_facts,
    preflight_devpop_audit_entry,
    prepare_and_preflight_devpop_audit_entry,
)
from src.baseline.devpop_audit_selection_manifest import (
    build_devpop_audit_selection_manifest_entry,
    load_devpop_audit_selection_manifest,
    validate_devpop_audit_selection_manifest,
)
from src.baseline.sweep_v2_six_axis_campaign import FROZEN_FIXED_CONFIGURATION_V2
from tests.test_devpop_audit_eval_run_producer import (
    _fixed_support_contract_path,
    _producer_paths,
    _seven_entry_manifest,
    _stage_sources,
)

_REAL_A5_MANIFEST = (
    Path(__file__).parents[1]
    / ".scratch_local"
    / "devpop_audit_a5"
    / "devpop_audit_selection_manifest_v001.json"
)

# Per-slot (search_arm, proposal_order) -> (hidden_size, seq_length, epoch).
# hidden_size is drawn from the frozen Sweep-v1 domain {64, 128, 256}; seq_length
# and epoch deliberately vary per slot.  No slot reproduces P1's full triple
# (256 / 60 / 1), and most slots differ in every component, so a leaked P1
# constant cannot pass the per-entry derivation.
_VARIED = {
    ("bayesian", 1): (64, 48, 2),
    ("bayesian", 2): (128, 120, 9),
    ("bayesian", 3): (256, 72, 4),
    ("random_control", 1): (256, 84, 10),
    ("random_control", 2): (64, 96, 6),
    ("random_control", 3): (128, 108, 5),
    ("random_control", 4): (64, 60, 7),
}


def _varied_manifest(tmp_path):
    contract_path = _fixed_support_contract_path(tmp_path)
    cd = json.loads(contract_path.read_text(encoding="utf-8"))
    support_version, support_sha256 = cd["contract_id"], cd["checksum_sha256"]

    entries = []
    for (arm, order), (hidden_size, seq_length, epoch) in _VARIED.items():
        entries.append(
            build_devpop_audit_selection_manifest_entry(
                search_arm=arm,
                proposal_order=order,
                hyperparameters={
                    "learning_rate": 3e-4,
                    "hidden_size": hidden_size,
                    "embedding_dropout": 0.10,
                    "output_dropout": 0.25,
                    "batch_size": 256,
                    "seq_length": seq_length,
                },
                screening_score=0.70 + 0.001 * order + (0.01 if arm == "bayesian" else 0.0),
                screening_best_epoch=epoch,
                screening_evidence_path=f"reports/screening/{arm}_{order}.json",
                source_run_dir=f"/scratch/runs/{arm}_{order}",
                checkpoint_filename=f"model_epoch{epoch:03d}.pt",
                checkpoint_sha256=hashlib.sha256(f"ckpt-{arm}-{order}".encode()).hexdigest(),
                selection_policy="frozen_screening_best_epoch_v001",
                support_contract_version=support_version,
                support_contract_sha256=support_sha256,
            )
        )
    return validate_devpop_audit_selection_manifest(entries), entries


# --------------------------------------------------------------------------- #
# pure derivation
# --------------------------------------------------------------------------- #

def test_campaign_frozen_facts_derived_from_frozen_v2_not_p1():
    facts = campaign_frozen_audit_facts()
    # every value here comes from FROZEN_FIXED_CONFIGURATION_V2 / audit-contract
    # constants, identical for all seven configurations
    assert facts.dynamic_inputs == tuple(FROZEN_FIXED_CONFIGURATION_V2["dynamic_inputs"])
    assert facts.dynamic_input_count == 2
    assert facts.static_embedding_hiddens == (128, 32)
    assert facts.static_embedding_output_dim == 32
    assert facts.implied_lstm_input_dim == 34
    assert facts.lead_hours == 6
    assert facts.model_seed == 967139
    assert facts.seq_length_floor == 120
    assert facts.expected_population_size == 2307
    assert facts.authoritative_screening_epochs == tuple(range(1, 13))


@pytest.mark.parametrize("slot", list(_VARIED))
def test_derive_uses_entry_specific_architecture(tmp_path, slot):
    _manifest, entries = _varied_manifest(tmp_path)
    entry = next(
        e for e in entries if (e["search_arm"], e["proposal_order"]) == slot
    )
    hidden_size, seq_length, epoch = _VARIED[slot]

    facts = derive_expected_preflight_facts(entry)

    # per-configuration: from THIS entry only
    assert facts.hidden_size == hidden_size
    assert facts.seq_length == seq_length
    assert facts.checkpoint_epoch == epoch
    assert facts.checkpoint_filename == f"model_epoch{epoch:03d}.pt"
    # campaign-frozen: identical regardless of the entry
    assert facts.campaign_frozen.implied_lstm_input_dim == 34
    assert facts.campaign_frozen.static_embedding_output_dim == 32
    assert facts.campaign_frozen.lead_hours == 6


def test_derive_covers_bayesian_and_random_control_slots(tmp_path):
    _manifest, entries = _varied_manifest(tmp_path)
    arms = {e["search_arm"] for e in entries}
    assert arms == {"bayesian", "random_control"}
    for e in entries:
        derive_expected_preflight_facts(e)  # all seven derive without error


def test_derive_fails_closed_on_corrupted_identity(tmp_path):
    _manifest, entries = _varied_manifest(tmp_path)
    entry = dict(entries[1])
    entry["trial_id"] = entry["trial_id"] + "__tampered"
    with pytest.raises(DevpopAuditPreflightError):
        derive_expected_preflight_facts(entry)


def test_derive_fails_closed_on_checkpoint_filename_epoch_mismatch(tmp_path):
    _manifest, entries = _varied_manifest(tmp_path)
    entry = dict(entries[2])
    entry["checkpoint_filename"] = "model_epoch999.pt"
    with pytest.raises(DevpopAuditPreflightError):
        derive_expected_preflight_facts(entry)


def test_derive_fails_closed_on_epoch_outside_authoritative_screening_epochs(tmp_path):
    _manifest, entries = _varied_manifest(tmp_path)
    entry = dict(entries[0])
    entry["screening_best_epoch"] = 13
    entry["checkpoint_filename"] = "model_epoch013.pt"
    with pytest.raises(DevpopAuditPreflightError, match="authoritative screening"):
        derive_expected_preflight_facts(entry)


def test_derive_fails_closed_on_checkpoint_sha_not_hex(tmp_path):
    _manifest, entries = _varied_manifest(tmp_path)
    entry = dict(entries[3])
    entry["checkpoint_sha256"] = "not-a-sha"
    with pytest.raises(DevpopAuditPreflightError):
        derive_expected_preflight_facts(entry)


@pytest.mark.skipif(not _REAL_A5_MANIFEST.is_file(), reason="real A5 selection-manifest bundle not present")
def test_all_seven_real_frozen_entries_derive_through_pure_logic():
    manifest = load_devpop_audit_selection_manifest(_REAL_A5_MANIFEST)
    assert len(manifest["entries"]) == 7
    for entry in manifest["entries"]:
        facts = derive_expected_preflight_facts(entry)
        # per-entry architecture matches the entry's own recorded hyperparameters
        assert facts.hidden_size == int(entry["hyperparameters"]["hidden_size"])
        assert facts.seq_length == int(entry["hyperparameters"]["seq_length"])
        assert facts.checkpoint_epoch == entry["screening_best_epoch"]
        # campaign-frozen architecture is identical across all seven
        assert facts.campaign_frozen.implied_lstm_input_dim == 34
        assert facts.campaign_frozen.static_embedding_output_dim == 32
    # no remote checkpoint bytes were read (pure logic only)


# --------------------------------------------------------------------------- #
# staged-run preflight (real producer path)
# --------------------------------------------------------------------------- #

def _prepare(tmp_path):
    ckpt_bytes = b"synthetic-screened-checkpoint-weights"
    manifest, entry, contract_path, ckpt_src, scaler_src = _stage_sources(tmp_path, ckpt_bytes)
    out = prepare_and_preflight_devpop_audit_entry(
        selection_manifest=manifest,
        entry_trial_id=entry["trial_id"],
        fixed_support_contract_path=contract_path,
        checkpoint_src_path=ckpt_src,
        scaler_src_path=scaler_src,
        out_generated_dir=tmp_path / "generated",
        out_run_dir=tmp_path / "run",
        **_producer_paths(tmp_path),
    )
    return manifest, entry, out


def test_prepare_and_preflight_passes_for_staged_entry(tmp_path):
    manifest, entry, out = _prepare(tmp_path)
    report = out["preflight_report"]
    assert report["ok"] is True
    assert report["failed_checks"] == []
    # entry-specific values from _seven_entry_manifest (hidden_size 128, the
    # bayesian-proposal-1 slot has seq_length 96) -- NOT P1's 256 / 60
    facts = report["derived_facts"]
    assert facts["hidden_size"] == 128
    assert facts["seq_length"] == 96
    assert facts["campaign_frozen"]["implied_lstm_input_dim"] == 34


def test_preflight_fails_closed_on_tampered_generated_config_hidden_size(tmp_path):
    manifest, entry, _out = _prepare(tmp_path)
    run_cfg = (tmp_path / "run" / "config.yml")
    cfg = yaml.safe_load(run_cfg.read_text(encoding="utf-8"))
    cfg["hidden_size"] = 999
    run_cfg.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(DevpopAuditPreflightError, match="preflight failed") as exc:
        preflight_devpop_audit_entry(
            selection_manifest=manifest,
            entry_trial_id=entry["trial_id"],
            run_dir=tmp_path / "run",
        )
    assert "hidden_size" in exc.value.report["failed_checks"]


def test_preflight_fails_closed_on_tampered_eval_run_manifest_epoch(tmp_path):
    manifest, entry, _out = _prepare(tmp_path)
    mpath = tmp_path / "run" / "DEVPOP_AUDIT_EVAL_RUN_MANIFEST.json"
    m = json.loads(mpath.read_text(encoding="utf-8"))
    m["checkpoint_epoch"] = m["checkpoint_epoch"] + 1
    mpath.write_text(json.dumps(m), encoding="utf-8")

    with pytest.raises(DevpopAuditPreflightError) as exc:
        preflight_devpop_audit_entry(
            selection_manifest=manifest,
            entry_trial_id=entry["trial_id"],
            run_dir=tmp_path / "run",
        )
    assert "checkpoint_epoch" in exc.value.report["failed_checks"]


def test_preflight_fails_closed_on_staged_checkpoint_byte_tamper(tmp_path):
    manifest, entry, _out = _prepare(tmp_path)
    staged = tmp_path / "run" / entry["checkpoint_filename"]
    staged.write_bytes(b"corrupted-weights")

    with pytest.raises(DevpopAuditPreflightError) as exc:
        preflight_devpop_audit_entry(
            selection_manifest=manifest,
            entry_trial_id=entry["trial_id"],
            run_dir=tmp_path / "run",
        )
    assert "staged_checkpoint_sha256" in exc.value.report["failed_checks"]


def test_preflight_non_strict_returns_report_without_raising(tmp_path):
    manifest, entry, _out = _prepare(tmp_path)
    run_cfg = (tmp_path / "run" / "config.yml")
    cfg = yaml.safe_load(run_cfg.read_text(encoding="utf-8"))
    cfg["seq_length"] = 12
    run_cfg.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    report = preflight_devpop_audit_entry(
        selection_manifest=manifest,
        entry_trial_id=entry["trial_id"],
        run_dir=tmp_path / "run",
        strict=False,
    )
    assert report["ok"] is False
    assert "seq_length" in report["failed_checks"]

"""Local, no-W&B tests for the v2 six-axis production Bayesian proposal
adapter (Section D, additive six-axis campaign foundation)."""
from __future__ import annotations
import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.baseline import sweep_v1_campaign as sweep
from src.baseline.fixed_support_contract_v2 import build_fixed_support_contract, write_fixed_support_contract
from src.baseline.sweep_v2_six_axis_campaign import (
    CAMPAIGN_ID_V2, DOMAIN_VERSION_V2, FORBIDDEN_V1_SWEEP_ID, OBJECTIVE_ID_V2, SweepV2CampaignError,
)
from src.baseline.sweep_v2_six_axis_production_adapter import (
    PreparationPathsV2, SweepV2PreparationError, prepare_bayesian_proposal_v2,
)
from tests._pilot_support import (
    BASELINE_POLICY_PATH, REAL_DEVELOPMENT, SPLITS_DIR, build_full_union_package, write_screening_basin_ids_file,
)

_OVERLAY_PATH = Path(__file__).parents[1] / "config" / "stage1_scientific_baseline_v2_six_axis_overlay_v001.yaml"


def _build_fixed_support_contract(tmp_path) -> Path:
    n = 10
    per_basin_date = {"01234567": np.arange(n)}
    per_basin_admitted = {"01234567": np.zeros(n, dtype=bool)}
    per_basin_admitted["01234567"][2:8] = True
    contract = build_fixed_support_contract(
        contract_id=OBJECTIVE_ID_V2, lead_hours=6, target_variable="qobs_mm_per_h_lead06",
        period="test_period", date_start="2024-01-01", date_end="2024-01-01",
        source_gap_policy_identity="test_gap_policy_v001", screening_basin_ids_sha256="0" * 64,
        per_basin_date=per_basin_date, per_basin_admitted=per_basin_admitted,
    )
    path = write_fixed_support_contract(contract, tmp_path / "fixed_support_contract.json")
    return path


def _paths(tmp_path, monkeypatch):
    package = build_full_union_package(tmp_path / "package")
    manifests = package / "manifests"
    (manifests / "file_checksums.csv").write_text("relative_path,sha256,size_bytes,artifact_role\n", encoding="utf-8")
    (package / "run_provenance.json").write_text('{"fixture":true}\n', encoding="utf-8")
    import src.baseline.sweep_v1_production_adapter as adapter
    monkeypatch.setattr(adapter, "PACKAGE_MANIFEST_SHA256", hashlib.sha256((manifests / "package_manifest.json").read_bytes()).hexdigest())
    monkeypatch.setattr(adapter, "PACKAGE_FILE_CHECKSUMS_SHA256", hashlib.sha256((manifests / "file_checksums.csv").read_bytes()).hexdigest())
    monkeypatch.setattr(adapter, "PACKAGE_RUN_PROVENANCE_SHA256", hashlib.sha256((package / "run_provenance.json").read_bytes()).hexdigest())
    splits = tmp_path / "canonical_splits"; splits.mkdir()
    for source in Path(SPLITS_DIR).glob("*.txt"):
        (splits / source.name).write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
    screening = write_screening_basin_ids_file(tmp_path / "screening.txt", REAL_DEVELOPMENT[:400])
    monkeypatch.setattr("src.baseline.pilot_lead06_config.sha256_of", lambda _: sweep.SCREENING_ARTIFACT_SHA256)
    contract_path = _build_fixed_support_contract(tmp_path)
    return PreparationPathsV2(BASELINE_POLICY_PATH, _OVERLAY_PATH, package, splits, screening, contract_path)


def _proposal(**changes):
    value = {"learning_rate": 3e-4, "hidden_size": 128, "embedding_dropout": 0.10,
             "output_dropout": 0.25, "batch_size": 256, "seq_length": 96, "proposal_order": 7,
             "wandb_sweep_id": "v2-prod-sweep", "wandb_run_id": "run-7"}
    value.update(changes)
    return value


def test_prepares_six_axis_config_with_variable_seq_length(tmp_path, monkeypatch):
    prepared = prepare_bayesian_proposal_v2(proposal=_proposal(), paths=_paths(tmp_path, monkeypatch))
    cfg = prepared.bundle.config_mapping
    assert cfg["seq_length"] == 96
    assert prepared.configuration_id.startswith("sweep_v2_cfg_")
    assert prepared.proposal_id == f"{CAMPAIGN_ID_V2}__bayesian__proposal007"
    assert prepared.evidence["seq_length_raw"] == 96 and prepared.evidence["seq_length_normalized"] == 96
    assert prepared.evidence["campaign_id"] == CAMPAIGN_ID_V2 and prepared.evidence["domain_version"] == DOMAIN_VERSION_V2
    assert prepared.evidence["objective_id"] == OBJECTIVE_ID_V2 and prepared.evidence["objective_score"] is None
    assert prepared.evidence["support_contract_version"] == OBJECTIVE_ID_V2


def test_accepts_integral_float_seq_length_from_q_uniform(tmp_path, monkeypatch):
    prepared = prepare_bayesian_proposal_v2(proposal=_proposal(seq_length=72.0), paths=_paths(tmp_path, monkeypatch))
    assert prepared.bundle.config_mapping["seq_length"] == 72
    assert isinstance(prepared.evidence["seq_length_normalized"], int)


@pytest.mark.parametrize("seq_length", [72, 48, 120])
def test_different_legal_seq_lengths_change_configuration_identity(tmp_path, monkeypatch, seq_length):
    a = prepare_bayesian_proposal_v2(proposal=_proposal(seq_length=72), paths=_paths(tmp_path / "a", monkeypatch))
    b = prepare_bayesian_proposal_v2(proposal=_proposal(seq_length=seq_length), paths=_paths(tmp_path / "b", monkeypatch))
    if seq_length == 72:
        assert a.configuration_id == b.configuration_id
    else:
        assert a.configuration_id != b.configuration_id


@pytest.mark.parametrize("bad_seq_length", [True, "72", float("nan"), float("inf"), 72.5, 47, 121, 50])
def test_rejects_illegal_seq_length(tmp_path, monkeypatch, bad_seq_length):
    with pytest.raises(SweepV2CampaignError):
        prepare_bayesian_proposal_v2(proposal=_proposal(seq_length=bad_seq_length), paths=_paths(tmp_path, monkeypatch))


def test_rejects_missing_seq_length_axis(tmp_path, monkeypatch):
    proposal = _proposal()
    del proposal["seq_length"]
    with pytest.raises(SweepV2CampaignError):
        prepare_bayesian_proposal_v2(proposal=proposal, paths=_paths(tmp_path, monkeypatch))


def test_two_controller_proposals_at_identical_coordinates_do_not_collide(tmp_path, monkeypatch):
    first = prepare_bayesian_proposal_v2(proposal=_proposal(proposal_order=1), paths=_paths(tmp_path / "one", monkeypatch))
    second = prepare_bayesian_proposal_v2(proposal=_proposal(proposal_order=2), paths=_paths(tmp_path / "two", monkeypatch))
    assert first.configuration_id == second.configuration_id
    assert first.proposal_id != second.proposal_id
    assert first.trial_id != second.trial_id


def test_refuses_v1_production_sweep_id(tmp_path, monkeypatch):
    with pytest.raises(SweepV2CampaignError):
        prepare_bayesian_proposal_v2(
            proposal=_proposal(wandb_sweep_id=FORBIDDEN_V1_SWEEP_ID), paths=_paths(tmp_path, monkeypatch)
        )


def test_refuses_v1_campaign_and_domain_identity(tmp_path, monkeypatch):
    with pytest.raises(SweepV2CampaignError):
        prepare_bayesian_proposal_v2(
            proposal=_proposal(campaign_id=sweep.CAMPAIGN_ID), paths=_paths(tmp_path / "a", monkeypatch)
        )
    with pytest.raises(SweepV2CampaignError):
        prepare_bayesian_proposal_v2(
            proposal=_proposal(domain_version=sweep.DOMAIN_VERSION), paths=_paths(tmp_path / "b", monkeypatch)
        )


def test_rejects_forbidden_scientific_override_keys(tmp_path, monkeypatch):
    with pytest.raises(SweepV2CampaignError):
        prepare_bayesian_proposal_v2(proposal=_proposal(seed=1), paths=_paths(tmp_path / "a", monkeypatch))
    with pytest.raises(SweepV2CampaignError):
        prepare_bayesian_proposal_v2(proposal=_proposal(support_contract_sha256="x" * 64), paths=_paths(tmp_path / "b", monkeypatch))


def test_rejects_unknown_seventh_axis(tmp_path, monkeypatch):
    with pytest.raises(SweepV2CampaignError):
        prepare_bayesian_proposal_v2(proposal=_proposal(extra_axis=1), paths=_paths(tmp_path, monkeypatch))


def test_generated_config_preserves_frozen_six_axis_contract(tmp_path, monkeypatch):
    prepared = prepare_bayesian_proposal_v2(proposal=_proposal(seq_length=84), paths=_paths(tmp_path, monkeypatch))
    cfg = prepared.bundle.config_mapping
    assert (cfg["epochs"], cfg["max_updates_per_epoch"], cfg["optimizer"], cfg["seed"]) == (12, 50_000, "Adam", 967139)
    assert cfg["dynamic_inputs"] == ["mrms_qpe_1h_mm", "rtma_2t_K"]
    assert cfg["statics_embedding"] == {"type": "fc", "hiddens": [128, 32], "activation": "tanh", "dropout": 0.1}
    assert (cfg["validation_start_date"], cfg["validation_end_date"]) == ("01/01/2024", "31/12/2024")
    assert prepared.bundle.population_role is not None and len(prepared.bundle.validation_basin_ids) == 400

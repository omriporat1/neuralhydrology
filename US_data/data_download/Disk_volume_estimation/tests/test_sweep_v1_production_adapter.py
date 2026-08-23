"""Local, no-W&B tests for the production Bayesian proposal adapter."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import pytest

from src.baseline import sweep_v1_campaign as sweep
from src.baseline.sweep_v1_production_adapter import (
    PreparationPaths, SweepV1PreparationError, canonicalize_wandb_proposal,
    prepare_bayesian_proposal, write_prepared_proposal,
)
from tests._pilot_support import BASELINE_POLICY_PATH, REAL_DEVELOPMENT, SPLITS_DIR, build_full_union_package, write_screening_basin_ids_file


def _paths(tmp_path, monkeypatch):
    package = build_full_union_package(tmp_path / "package")
    manifests = package / "manifests"
    # Test-only fixture identity injection: the public CLI has no equivalent
    # bypass.  The production constants remain the certified v002 pins.
    (manifests / "file_checksums.csv").write_text("relative_path,sha256,size_bytes,artifact_role\n", encoding="utf-8")
    (package / "run_provenance.json").write_text('{"fixture":true}\n', encoding="utf-8")
    import src.baseline.sweep_v1_production_adapter as adapter
    monkeypatch.setattr(adapter, "PACKAGE_MANIFEST_SHA256", hashlib.sha256((manifests / "package_manifest.json").read_bytes()).hexdigest())
    monkeypatch.setattr(adapter, "PACKAGE_FILE_CHECKSUMS_SHA256", hashlib.sha256((manifests / "file_checksums.csv").read_bytes()).hexdigest())
    monkeypatch.setattr(adapter, "PACKAGE_RUN_PROVENANCE_SHA256", hashlib.sha256((package / "run_provenance.json").read_bytes()).hexdigest())
    # Git's Windows checkout uses CRLF, while the frozen Moriah/runtime bytes
    # are the committed LF blobs named by the production SHA pins.
    splits = tmp_path / "canonical_splits"; splits.mkdir()
    for source in Path(SPLITS_DIR).glob("*.txt"):
        (splits / source.name).write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
    screening = write_screening_basin_ids_file(tmp_path / "screening.txt", REAL_DEVELOPMENT[:400])
    # Synthetic local packages cannot have the production subset's bytes;
    # production code still pins the literal checksum and validates it.
    monkeypatch.setattr("src.baseline.pilot_lead06_config.sha256_of", lambda _: sweep.SCREENING_ARTIFACT_SHA256)
    return PreparationPaths(BASELINE_POLICY_PATH, package, splits, screening)


def _proposal(**changes):
    value = {"learning_rate": 3e-4, "hidden_size": 128, "embedding_dropout": 0.10,
             "output_dropout": 0.25, "batch_size": 256, "proposal_order": 7,
             "wandb_sweep_id": "prod-sweep", "wandb_run_id": "run-7"}
    value.update(changes)
    return value


def test_prepares_exact_medium_fidelity_contract_and_writes_evidence(tmp_path, monkeypatch):
    prepared = prepare_bayesian_proposal(proposal=_proposal(), paths=_paths(tmp_path, monkeypatch))
    cfg = prepared.bundle.config_mapping
    assert prepared.configuration_id == sweep.configuration_id({key: _proposal()[key] for key in ("learning_rate", "hidden_size", "embedding_dropout", "output_dropout", "batch_size")})
    assert (prepared.proposal_id, prepared.trial_id) == (sweep.proposal_id("bayesian", 7), sweep.trial_id(prepared.configuration_id))
    assert (cfg["epochs"], cfg["max_updates_per_epoch"], cfg["save_weights_every"], cfg["optimizer"], cfg["seed"]) == (12, 50_000, 1, "Adam", 967139)
    assert cfg["dynamic_inputs"] == ["mrms_qpe_1h_mm", "rtma_2t_K"]
    assert cfg["statics_embedding"] == {"type": "fc", "hiddens": [128, 32], "activation": "tanh", "dropout": 0.1}
    assert (cfg["validation_start_date"], cfg["validation_end_date"]) == ("01/01/2024", "31/12/2024")
    record = write_prepared_proposal(prepared, tmp_path / "prepared")
    assert record["prepare_status"] == "PASS" and record["objective_score"] is None and record["sealed_scope"] is False
    assert record["artifact_identity_status"] == "PASS"
    assert record["generated_nh_config_sha256"] == hashlib.sha256(Path(record["generated_nh_config_path"]).read_bytes()).hexdigest()


@pytest.mark.parametrize("change", [
    {"learning_rate": 9e-5}, {"learning_rate": 1.1e-3}, {"learning_rate": float("nan")},
    {"hidden_size": 32}, {"embedding_dropout": -0.01}, {"output_dropout": 0.41}, {"batch_size": 1024},
    {"seed": 1}, {"epochs": 6}, {"max_updates_per_epoch": 1}, {"package_identity": "other"},
    {"screening_artifact_sha256": "other"}, {"sealed_scope": True}, {"qualification_kind": "toy"},
])
def test_rejects_illegal_domain_and_scientific_overrides(tmp_path, monkeypatch, change):
    with pytest.raises(SweepV1PreparationError):
        prepare_bayesian_proposal(proposal=_proposal(**change), paths=_paths(tmp_path, monkeypatch))


@pytest.mark.parametrize("proposal", [
    {"hidden_size": 128, "embedding_dropout": 0.1, "output_dropout": 0.2, "batch_size": 256, "proposal_order": 1},
    {**_proposal(), "extra_axis": 1}, {**_proposal(), "search_arm": "random_control"}, {**_proposal(), "proposal_order": 0},
])
def test_rejects_missing_extra_and_nonproduction_provenance(tmp_path, monkeypatch, proposal):
    with pytest.raises(SweepV1PreparationError):
        prepare_bayesian_proposal(proposal=proposal, paths=_paths(tmp_path, monkeypatch))


def test_wandb_like_mapping_is_identity_equivalent_and_ids_are_telemetry_only(tmp_path, monkeypatch):
    direct = _proposal(wandb_sweep_id="first", wandb_run_id="one")
    external = canonicalize_wandb_proposal(
        {"batch_size": 256, "output_dropout": 0.25, "hidden_size": 128, "learning_rate": 3e-4, "embedding_dropout": 0.1},
        {"proposal_order": 7, "wandb_sweep_id": "second", "wandb_run_id": "two"},
    )
    first = prepare_bayesian_proposal(proposal=direct, paths=_paths(tmp_path / "one", monkeypatch))
    second = prepare_bayesian_proposal(proposal=external, paths=_paths(tmp_path / "two", monkeypatch))
    assert first.configuration_id == second.configuration_id
    assert first.bundle.config_mapping == second.bundle.config_mapping
    assert first.proposal_id == second.proposal_id


def test_retries_keep_proposal_and_configuration_but_change_trial_attempt(tmp_path, monkeypatch):
    first = prepare_bayesian_proposal(proposal=_proposal(execution_generation=1), paths=_paths(tmp_path / "one", monkeypatch))
    retry = prepare_bayesian_proposal(proposal=_proposal(execution_generation=2), paths=_paths(tmp_path / "two", monkeypatch))
    assert (first.configuration_id, first.proposal_id) == (retry.configuration_id, retry.proposal_id)
    assert first.trial_id != retry.trial_id


def test_random_manifest_remains_pinned():
    path = Path(__file__).parents[1] / "config/stage1_phase_b_sweep_v1_original_domain_v001_random_control_manifest.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sweep.RANDOM_CONTROL_MANIFEST_SHA256


def test_structurally_plausible_fake_package_fails_the_production_v002_pin(tmp_path, monkeypatch):
    package = build_full_union_package(tmp_path / "fabricated_v002")
    screening = write_screening_basin_ids_file(tmp_path / "screening.txt", REAL_DEVELOPMENT[:400])
    monkeypatch.setattr("src.baseline.pilot_lead06_config.sha256_of", lambda _: sweep.SCREENING_ARTIFACT_SHA256)
    with pytest.raises(SweepV1PreparationError, match="package_manifest_sha256"):
        prepare_bayesian_proposal(proposal=_proposal(), paths=PreparationPaths(BASELINE_POLICY_PATH, package, SPLITS_DIR, screening))


@pytest.mark.parametrize("split_name", ["development_train.txt", "spatial_holdout_nonca.txt"])
def test_modified_split_bytes_fail_even_when_membership_is_unchanged(tmp_path, monkeypatch, split_name):
    paths = _paths(tmp_path, monkeypatch)
    alternate = tmp_path / "alternate_splits"; alternate.mkdir()
    for source in Path(paths.splits_dir).glob("*.txt"):
        target = alternate / source.name
        target.write_bytes(source.read_bytes())
    # A newline preserves line membership but must not preserve identity.
    target = alternate / split_name
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(SweepV1PreparationError, match="split_sha256"):
        prepare_bayesian_proposal(proposal=_proposal(), paths=PreparationPaths(BASELINE_POLICY_PATH, paths.package_root, alternate, paths.screening_basin_ids_path))


def test_production_cli_has_no_split_override():
    source = (Path(__file__).parents[1] / "scripts" / "prepare_sweep_v1_bayesian_proposal.py").read_text(encoding="utf-8")
    assert 'add_argument("--splits-dir"' not in source
    assert 'canonical_splits = ROOT / "config" / "stage1_baseline_splits_v001"' in source

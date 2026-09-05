"""SHARED-A4 vertical integration test: a frozen selection-manifest entry ->
the audit-specific full-development-population eval-run producer -> a
producer-built provenance receipt -> a synthetic (never NH-executed)
period-results pickle -> the existing SHARED-A2 consumer
(:func:`evaluate_devpop_common120_audit_row`) -> one canonical audit row.

Uses the SAME real committed development-population basin-membership /
baseline-policy / v2 six-axis overlay fixtures already exercised by
``tests/test_sweep_v2_six_axis_production_adapter.py`` for config generation
(so the producer's ``build_pilot_bundle_with_validation_scope`` call runs its
real basin-membership/target/lead validation against the real 2,307-basin
development split) -- but the NH result / evaluator population/contract are
small, fully synthetic fixtures local to this file, exactly mirroring
``tests/test_devpop_common120_audit_evaluator.py``. Nothing here contacts
Moriah/Slurm/W&B, runs NeuralHydrology, or builds the real seven-checkpoint
production manifest.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.baseline.devpop_audit_eval_run_producer import (
    AUDIT_EVAL_RUN_MANIFEST_FILENAME,
    AUDIT_EVAL_RUN_MARKER_FILENAME,
    DevpopAuditEvalRunProducerError,
    prepare_devpop_audit_eval_run_dir,
)
from src.baseline.devpop_audit_selection_manifest import (
    build_devpop_audit_selection_manifest_entry,
    compute_devpop_audit_selection_manifest_sha256,
    selection_manifest_entry_to_checkpoint_identity,
    validate_devpop_audit_selection_manifest,
)
from src.baseline.devpop_common120_audit_contract import (
    CANONICAL_SOURCE_GAP_POLICY_IDENTITY,
    CANONICAL_TARGET_VARIABLE,
    ExpectedPopulationSpec,
    build_devpop_audit_contract,
)
from src.baseline.devpop_common120_audit_evaluator import (
    build_devpop_audit_provenance_receipt,
    evaluate_devpop_common120_audit_row,
)
from src.baseline.fixed_support_contract_v2 import build_fixed_support_contract, write_fixed_support_contract
from src.baseline.sweep_v2_six_axis_campaign import OBJECTIVE_ID_V2
from tests._pilot_support import BASELINE_POLICY_PATH, REAL_DEVELOPMENT, SPLITS_DIR, build_full_union_package

_OVERLAY_PATH = Path(__file__).parents[1] / "config" / "stage1_scientific_baseline_v2_six_axis_overlay_v001.yaml"

TARGET = CANONICAL_TARGET_VARIABLE
LEAD = 6
N_VAL = 60
SUPPORT_IDX = np.arange(20, 40)
EVAL_BASINS = ("00000001", "00000002", "00000003")

_EVAL_IDENT = {
    "package_manifest_sha256": "a" * 64,
    "package_file_checksums_sha256": "b" * 64,
    "package_run_provenance_sha256": "c" * 64,
    "development_split_sha256": "d" * 64,
    "spatial_holdout_split_sha256": "e" * 64,
}


# --------------------------------------------------------------------------- #
# producer-side fixtures (real config-generation path)
# --------------------------------------------------------------------------- #

def _fixed_support_contract_path(tmp_path, *, admitted_slice=slice(2, 8)) -> Path:
    n = 10
    per_basin_date = {"01234567": np.arange(n)}
    per_basin_admitted = {"01234567": np.zeros(n, dtype=bool)}
    per_basin_admitted["01234567"][admitted_slice] = True
    contract = build_fixed_support_contract(
        contract_id=OBJECTIVE_ID_V2, lead_hours=LEAD, target_variable=TARGET,
        period="test_period", date_start="2024-01-01", date_end="2024-01-01",
        source_gap_policy_identity="test_gap_policy_v001", screening_basin_ids_sha256="0" * 64,
        package_manifest_sha256="a" * 64, package_file_checksums_sha256="b" * 64,
        package_run_provenance_sha256="c" * 64, development_split_sha256="d" * 64,
        spatial_holdout_split_sha256="e" * 64,
        per_basin_date=per_basin_date, per_basin_admitted=per_basin_admitted,
    )
    return write_fixed_support_contract(contract, tmp_path / "fixed_support_contract.json")


def _seven_entry_manifest(tmp_path, *, ckpt_bytes: bytes, epoch: int = 3):
    """Build + validate a synthetic FROZEN seven-entry selection manifest
    (Bayesian proposals 1-3 + random-control Wave-1 1-4, distinct seq_length
    per slot so no two configuration_ids collide). Its Bayesian proposal-1
    entry names the real synthetic checkpoint bytes under test.

    Returns ``(validated_manifest, selected_entry, fixed_support_contract_path)``.
    """
    contract = _fixed_support_contract_path(tmp_path)
    cd = json.loads(contract.read_text(encoding="utf-8"))
    support_version, support_sha256 = cd["contract_id"], cd["checksum_sha256"]

    def _mk(search_arm: str, order: int, seq_length: int, *, selected: bool = False) -> dict:
        ep = epoch if selected else 3
        ckpt_sha256 = (
            hashlib.sha256(ckpt_bytes).hexdigest() if selected
            else hashlib.sha256(f"ckpt-{search_arm}-{order}".encode()).hexdigest()
        )
        return build_devpop_audit_selection_manifest_entry(
            search_arm=search_arm,
            proposal_order=order,
            hyperparameters={
                "learning_rate": 3e-4, "hidden_size": 128, "embedding_dropout": 0.10,
                "output_dropout": 0.25, "batch_size": 256, "seq_length": seq_length,
            },
            screening_score=0.71,
            screening_best_epoch=ep,
            screening_evidence_path=f"reports/screening/{search_arm}_{order}.json",
            source_run_dir=f"/scratch/runs/{search_arm}_{order}",
            checkpoint_filename=f"model_epoch{ep:03d}.pt",
            checkpoint_sha256=ckpt_sha256,
            selection_policy="frozen_screening_best_epoch_v001",
            support_contract_version=support_version,
            support_contract_sha256=support_sha256,
        )

    entries = [
        _mk("bayesian", 1, 96, selected=True),
        _mk("bayesian", 2, 84),
        _mk("bayesian", 3, 72),
        _mk("random_control", 1, 48),
        _mk("random_control", 2, 60),
        _mk("random_control", 3, 108),
        _mk("random_control", 4, 120),
    ]
    manifest = validate_devpop_audit_selection_manifest(entries)
    return manifest, entries[0], contract


def _producer_paths(tmp_path):
    package = build_full_union_package(tmp_path / "package")
    return dict(
        baseline_policy_path=BASELINE_POLICY_PATH,
        policy_overlay_path=_OVERLAY_PATH,
        package_root=package,
        splits_dir=SPLITS_DIR,
        run_profile_name="pilot_lead06_emb128x32_seedA_v001",
    )


# --------------------------------------------------------------------------- #
# evaluator-side synthetic fixtures (mirrors test_devpop_common120_audit_evaluator.py)
# --------------------------------------------------------------------------- #

def _val_dates(n: int = N_VAL, start: str = "2024-06-01") -> np.ndarray:
    return pd.date_range(start, periods=n, freq="h").to_numpy()


def _basin_arrays(seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    obs = rng.uniform(1.0, 50.0, size=N_VAL)
    sim = obs.copy()
    off = np.ones(N_VAL, dtype=bool)
    off[SUPPORT_IDX] = False
    sim[off] = obs[off] + 500.0
    return obs, sim


def _write_run_pickle(run_dir, basin_ids, *, epoch: int) -> None:
    dates = _val_dates()
    results = {}
    for i, b in enumerate(basin_ids):
        obs, sim = _basin_arrays(100 + i)
        ds = xr.Dataset(
            {
                f"{TARGET}_obs": ("date", np.asarray(obs, dtype=float)),
                f"{TARGET}_sim": ("date", np.asarray(sim, dtype=float)),
            },
            coords={"date": dates},
        )
        results[b] = {"1h": {"xr": ds}}
    pdir = Path(run_dir) / "validation" / f"model_epoch{epoch:03d}"
    pdir.mkdir(parents=True, exist_ok=True)
    with open(pdir / "validation_results.p", "wb") as fh:
        pickle.dump(results, fh)


def _write_eval_package(package_root, basin_ids, *, n: int = 300) -> None:
    ts_dir = Path(package_root) / "time_series"
    ts_dir.mkdir(parents=True, exist_ok=True)
    for i, b in enumerate(basin_ids):
        area = 150.0 + 100.0 * i
        rng = np.random.default_rng(7000 + i)
        qobs_m3s = rng.uniform(2.0, 300.0, size=n)
        target = np.full(n, np.nan)
        target[: n - LEAD] = 3.6 * qobs_m3s[LEAD:] / area
        xr.Dataset(
            {"qobs_m3s": ("date", qobs_m3s), TARGET: ("date", target)},
            coords={"date": np.arange(n)},
        ).to_netcdf(ts_dir / f"{b}.nc")


def _eval_population_and_contract() -> tuple:
    population = ExpectedPopulationSpec.for_synthetic_fixture(role="development_train", basin_ids=list(EVAL_BASINS))
    dates = _val_dates()
    admitted = np.zeros(len(dates), dtype=bool)
    admitted[SUPPORT_IDX] = True
    contract = build_devpop_audit_contract(
        population=population,
        target_variable=TARGET,
        source_gap_policy_identity=CANONICAL_SOURCE_GAP_POLICY_IDENTITY,
        per_basin_date={b: dates for b in EVAL_BASINS},
        per_basin_admitted={b: admitted for b in EVAL_BASINS},
        **_EVAL_IDENT,
    )
    return population, contract


# --------------------------------------------------------------------------- #
# the vertical test
# --------------------------------------------------------------------------- #

def test_manifest_entry_to_producer_to_a2_evaluator_vertical(tmp_path):
    ckpt_bytes = b"synthetic-screened-checkpoint-weights"
    scaler_bytes = b"synthetic-scaler-yaml-bytes"

    manifest, entry, _contract_path = _seven_entry_manifest(tmp_path, ckpt_bytes=ckpt_bytes)
    ckpt_src = tmp_path / "source_ckpt" / entry["checkpoint_filename"]
    ckpt_src.parent.mkdir(parents=True)
    ckpt_src.write_bytes(ckpt_bytes)
    scaler_src = tmp_path / "source_ckpt" / "train_data_scaler.yml"
    scaler_src.write_bytes(scaler_bytes)

    producer_manifest = prepare_devpop_audit_eval_run_dir(
        selection_manifest=manifest,
        entry_trial_id=entry["trial_id"],
        fixed_support_contract_path=_contract_path,
        checkpoint_src_path=ckpt_src,
        scaler_src_path=scaler_src,
        out_generated_dir=tmp_path / "generated",
        out_run_dir=tmp_path / "run",
        **_producer_paths(tmp_path),
    )

    # (0) the producer-persisted run manifest carries the EXACT validated
    # seven-entry selection-manifest SHA -- so every staged run can later
    # prove "I came from entry X of this frozen seven-entry manifest".
    assert producer_manifest["selection_manifest_sha256"] == manifest["manifest_sha256"]
    assert producer_manifest["selection_manifest_sha256"] == (
        compute_devpop_audit_selection_manifest_sha256(manifest["entries"])
    )

    # (1) producer output is exactly what the SHARED-A4 design doc requires
    assert producer_manifest["trial_id"] == entry["trial_id"]
    assert producer_manifest["configuration_id"] == entry["configuration_id"]
    assert producer_manifest["target_variable"] == CANONICAL_TARGET_VARIABLE
    assert producer_manifest["lead_hours"] == 6
    assert producer_manifest["validation_basin_count"] == len(REAL_DEVELOPMENT)
    assert producer_manifest["checkpoint_sha256"] == hashlib.sha256(ckpt_bytes).hexdigest()

    run_dir = tmp_path / "run"
    assert (run_dir / AUDIT_EVAL_RUN_MARKER_FILENAME).is_file()
    persisted = json.loads((run_dir / AUDIT_EVAL_RUN_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert persisted["selection_manifest_sha256"] == manifest["manifest_sha256"]
    assert (run_dir / f"model_epoch{entry['screening_best_epoch']:03d}.pt").read_bytes() == ckpt_bytes
    assert (run_dir / "train_data" / "train_data_scaler.yml").read_bytes() == scaler_bytes
    assert (run_dir / "config.yml").is_file()

    # (2) NH is never executed here -- fabricate its expected output shape
    # directly, exactly as the SHARED-A2 fixture does.
    _write_run_pickle(run_dir, EVAL_BASINS, epoch=entry["screening_best_epoch"])

    # (3) provenance receipt built from the REAL on-disk bytes the producer
    # just wrote (checkpoint + the just-written result pickle) -- this is the
    # authoritative producer -> consumer binding, not a caller assertion.
    checkpoint_path = run_dir / f"model_epoch{entry['screening_best_epoch']:03d}.pt"
    receipt = build_devpop_audit_provenance_receipt(
        trial_id=entry["trial_id"],
        configuration_id=entry["configuration_id"],
        run_dir=run_dir,
        period="validation",
        checkpoint_epoch=entry["screening_best_epoch"],
        checkpoint_path=checkpoint_path,
    )

    checkpoint_identity = selection_manifest_entry_to_checkpoint_identity(entry)

    eval_package_root = tmp_path / "eval_pkg"
    _write_eval_package(eval_package_root, EVAL_BASINS)
    population, contract = _eval_population_and_contract()

    # (4) the existing, unmodified SHARED-A2 consumer -- it obtains every
    # identity fact through checkpoint_identity/provenance_receipt, never by
    # reconstructing them from the manifest entry or the producer's files.
    row = evaluate_devpop_common120_audit_row(
        checkpoint_identity=checkpoint_identity,
        run_dir=run_dir,
        package_root=eval_package_root,
        population=population,
        contract=contract,
        provenance_receipt=receipt,
        checkpoint_path=checkpoint_path,
    )

    assert row["trial_id"] == entry["trial_id"]
    assert row["configuration_id"] == entry["configuration_id"]
    assert row["checkpoint_epoch"] == entry["screening_best_epoch"]
    assert row["checkpoint_sha256"] == hashlib.sha256(ckpt_bytes).hexdigest()
    assert row["provenance_verified"] is True
    result = row["result"]
    assert result["n_basins_evaluated"] == len(EVAL_BASINS)
    assert result["evaluated_basin_ids"] == sorted(EVAL_BASINS)
    # Support alignment genuinely consumed: NSE == 1.0 is only reachable if
    # scoring was restricted to the frozen 20-hour Common-120 subset (sim is
    # deliberately broken on every off-support hour) -- so this is real
    # evidence the producer's run_dir flowed into the real raw-space path,
    # not a caller-asserted shortcut.
    for basin_row in result["per_basin"]:
        assert basin_row["nse"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# fail-closed producer tests
# --------------------------------------------------------------------------- #

def test_producer_rejects_checkpoint_hash_mismatch(tmp_path):
    manifest, entry, contract_path = _seven_entry_manifest(tmp_path, ckpt_bytes=b"expected-bytes")
    ckpt_src = tmp_path / "source_ckpt" / entry["checkpoint_filename"]
    ckpt_src.parent.mkdir(parents=True)
    ckpt_src.write_bytes(b"WRONG-bytes-do-not-match-frozen-hash")
    scaler_src = tmp_path / "source_ckpt" / "train_data_scaler.yml"
    scaler_src.write_bytes(b"scaler")

    with pytest.raises(DevpopAuditEvalRunProducerError, match="sha256"):
        prepare_devpop_audit_eval_run_dir(
            selection_manifest=manifest,
            entry_trial_id=entry["trial_id"],
            fixed_support_contract_path=contract_path,
            checkpoint_src_path=ckpt_src,
            scaler_src_path=scaler_src,
            out_generated_dir=tmp_path / "generated",
            out_run_dir=tmp_path / "run",
            **_producer_paths(tmp_path),
        )
    assert not (tmp_path / "run").exists()


def test_producer_rejects_checkpoint_filename_mismatch(tmp_path):
    manifest, entry, contract_path = _seven_entry_manifest(tmp_path, ckpt_bytes=b"expected-bytes")
    ckpt_src = tmp_path / "source_ckpt" / "model_epochXYZ.pt"
    ckpt_src.parent.mkdir(parents=True)
    ckpt_src.write_bytes(b"expected-bytes")
    scaler_src = tmp_path / "source_ckpt" / "train_data_scaler.yml"
    scaler_src.write_bytes(b"scaler")

    with pytest.raises(DevpopAuditEvalRunProducerError, match="filename"):
        prepare_devpop_audit_eval_run_dir(
            selection_manifest=manifest,
            entry_trial_id=entry["trial_id"],
            fixed_support_contract_path=contract_path,
            checkpoint_src_path=ckpt_src,
            scaler_src_path=scaler_src,
            out_generated_dir=tmp_path / "generated",
            out_run_dir=tmp_path / "run",
            **_producer_paths(tmp_path),
        )


def test_producer_rejects_configuration_id_under_different_support_contract(tmp_path):
    manifest, entry, _contract_path = _seven_entry_manifest(tmp_path, ckpt_bytes=b"expected-bytes")
    other_contract_path = _fixed_support_contract_path(tmp_path / "other", admitted_slice=slice(1, 4))
    ckpt_src = tmp_path / "source_ckpt" / entry["checkpoint_filename"]
    ckpt_src.parent.mkdir(parents=True)
    ckpt_src.write_bytes(b"expected-bytes")
    scaler_src = tmp_path / "source_ckpt" / "train_data_scaler.yml"
    scaler_src.write_bytes(b"scaler")

    with pytest.raises(DevpopAuditEvalRunProducerError, match="configuration_id"):
        prepare_devpop_audit_eval_run_dir(
            selection_manifest=manifest,
            entry_trial_id=entry["trial_id"],
            fixed_support_contract_path=other_contract_path,
            checkpoint_src_path=ckpt_src,
            scaler_src_path=scaler_src,
            out_generated_dir=tmp_path / "generated",
            out_run_dir=tmp_path / "run",
            **_producer_paths(tmp_path),
        )


def test_producer_rejects_trial_id_not_in_seven_entry_manifest(tmp_path):
    # Defect 1: an individually-valid standalone entry is no longer a legal
    # input -- the producer only accepts a member of the validated,
    # hash-pinned seven-entry manifest.
    manifest, entry, contract_path = _seven_entry_manifest(tmp_path, ckpt_bytes=b"expected-bytes")
    ckpt_src = tmp_path / "source_ckpt" / entry["checkpoint_filename"]
    ckpt_src.parent.mkdir(parents=True)
    ckpt_src.write_bytes(b"expected-bytes")
    scaler_src = tmp_path / "source_ckpt" / "train_data_scaler.yml"
    scaler_src.write_bytes(b"scaler")

    with pytest.raises(DevpopAuditEvalRunProducerError, match="does not identify exactly one entry"):
        prepare_devpop_audit_eval_run_dir(
            selection_manifest=manifest,
            entry_trial_id="sweep_v2_campaign__bayesian__proposal001__not_a_real_trial",
            fixed_support_contract_path=contract_path,
            checkpoint_src_path=ckpt_src,
            scaler_src_path=scaler_src,
            out_generated_dir=tmp_path / "generated",
            out_run_dir=tmp_path / "run",
            **_producer_paths(tmp_path),
        )
    assert not (tmp_path / "run").exists()


def test_producer_rejects_tampered_selection_manifest_sha256(tmp_path):
    # Defect 1: the producer carries the VALIDATED manifest SHA, and refuses
    # a manifest whose recorded manifest_sha256 no longer matches its entries.
    manifest, entry, contract_path = _seven_entry_manifest(tmp_path, ckpt_bytes=b"expected-bytes")
    ckpt_src = tmp_path / "source_ckpt" / entry["checkpoint_filename"]
    ckpt_src.parent.mkdir(parents=True)
    ckpt_src.write_bytes(b"expected-bytes")
    scaler_src = tmp_path / "source_ckpt" / "train_data_scaler.yml"
    scaler_src.write_bytes(b"scaler")

    tampered = dict(manifest)
    tampered["manifest_sha256"] = "0" * 64

    with pytest.raises(DevpopAuditEvalRunProducerError, match="manifest_sha256"):
        prepare_devpop_audit_eval_run_dir(
            selection_manifest=tampered,
            entry_trial_id=entry["trial_id"],
            fixed_support_contract_path=contract_path,
            checkpoint_src_path=ckpt_src,
            scaler_src_path=scaler_src,
            out_generated_dir=tmp_path / "generated",
            out_run_dir=tmp_path / "run",
            **_producer_paths(tmp_path),
        )

"""Vertical synthetic test for the development-population Common-120 audit
consumer/evaluator seam (milestone SHARED-A2).

One genuine small end-to-end fixture exercises the whole consumer boundary --
explicit checkpoint/trial identity -> an already-produced NH period-results
pickle -> frozen Common-120 support alignment from the SHARED-A1 contract ->
the REAL qualified raw-space path (self-derived basin area, mm/h->m^3/s
conversion, per-basin + aggregate metrics) -> the SHARED-A1 strict completeness
gate -> one fixture-labelled audit row.

Nothing here contacts Moriah / Slurm / W&B, runs checkpoint inference, selects a
checkpoint, or builds the real 2,307-basin support artifact.  Small synthetic
NH-style results and a small synthetic package are written to ``tmp_path`` so
the real raw-space helpers actually execute (they are not monkeypatched).

The support-alignment assertion is the point of the happy path: each basin's
simulation equals its observation only on the frozen 20-hour Common-120
support and is deliberately broken everywhere else, so a per-basin NSE of
exactly 1.0 is reachable *only* if the evaluator restricted scoring to the
frozen support subset.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import src.baseline.devpop_common120_audit_evaluator as evaluator_module
from src.baseline.devpop_common120_audit_contract import (
    CANONICAL_SOURCE_GAP_POLICY_IDENTITY,
    CANONICAL_TARGET_VARIABLE,
    DevpopAuditCompletenessError,
    DevpopAuditContractError,
    ExpectedPopulationSpec,
    build_devpop_audit_contract,
)
from src.baseline.devpop_common120_audit_evaluator import (
    PROVENANCE_RECEIPT_SCHEMA,
    DevpopAuditCheckpointIdentity,
    DevpopAuditEvaluatorError,
    DevpopAuditProvenanceReceipt,
    build_devpop_audit_provenance_receipt,
    evaluate_devpop_common120_audit_row,
    load_devpop_audit_provenance_receipt,
    write_devpop_audit_provenance_receipt,
)

CKPT_BYTES = b"synthetic-checkpoint-weights-0000"

TARGET = CANONICAL_TARGET_VARIABLE
LEAD = 6
N_VAL = 60
SUPPORT_IDX = np.arange(20, 40)  # 20 frozen Common-120 admitted timestamps
BASINS = ("00000001", "00000002", "00000003")

IDENT = {
    "package_manifest_sha256": "a" * 64,
    "package_file_checksums_sha256": "b" * 64,
    "package_run_provenance_sha256": "c" * 64,
    "development_split_sha256": "d" * 64,
    "spatial_holdout_split_sha256": "e" * 64,
}


# --------------------------------------------------------------------------- #
# fixture builders
# --------------------------------------------------------------------------- #

def _val_dates(n: int = N_VAL, start: str = "2024-06-01") -> np.ndarray:
    return pd.date_range(start, periods=n, freq="h").to_numpy()


def _basin_arrays(seed: int) -> tuple:
    """obs finite everywhere; sim == obs ONLY on the frozen support, broken by
    a large additive error everywhere else."""
    rng = np.random.default_rng(seed)
    obs = rng.uniform(1.0, 50.0, size=N_VAL)
    sim = obs.copy()
    off = np.ones(N_VAL, dtype=bool)
    off[SUPPORT_IDX] = False
    sim[off] = obs[off] + 500.0
    return obs, sim


def _population(ids=BASINS) -> ExpectedPopulationSpec:
    return ExpectedPopulationSpec.for_synthetic_fixture(role="development_train", basin_ids=list(ids))


def _contract(population: ExpectedPopulationSpec, *, dates=None, support_idx=SUPPORT_IDX) -> dict:
    dates = _val_dates() if dates is None else dates
    admitted = np.zeros(len(dates), dtype=bool)
    admitted[support_idx] = True
    return build_devpop_audit_contract(
        population=population,
        target_variable=CANONICAL_TARGET_VARIABLE,
        source_gap_policy_identity=CANONICAL_SOURCE_GAP_POLICY_IDENTITY,
        per_basin_date={b: dates for b in population.basin_ids},
        per_basin_admitted={b: admitted for b in population.basin_ids},
        **IDENT,
    )


def _write_run_pickle(run_dir, basin_ids, *, epoch: int = 1, dates=None, obs_sim=None) -> dict:
    dates = _val_dates() if dates is None else dates
    obs_sim = obs_sim or {}
    results = {}
    for i, b in enumerate(basin_ids):
        obs, sim = obs_sim.get(b, _basin_arrays(100 + i))
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
    return results


def _write_package(package_root, basin_ids, *, n: int = 300) -> dict:
    ts_dir = Path(package_root) / "time_series"
    ts_dir.mkdir(parents=True, exist_ok=True)
    areas = {}
    for i, b in enumerate(basin_ids):
        area = 150.0 + 100.0 * i
        areas[b] = area
        rng = np.random.default_rng(7000 + i)
        qobs_m3s = rng.uniform(2.0, 300.0, size=n)
        target = np.full(n, np.nan)
        target[: n - LEAD] = 3.6 * qobs_m3s[LEAD:] / area
        xr.Dataset(
            {"qobs_m3s": ("date", qobs_m3s), TARGET: ("date", target)},
            coords={"date": np.arange(n)},
        ).to_netcdf(ts_dir / f"{b}.nc")
    return areas


def _identity(**kw) -> DevpopAuditCheckpointIdentity:
    base = dict(
        trial_id="trial-A",
        configuration_id="cfg-A",
        checkpoint_epoch=1,
        checkpoint_sha256=hashlib.sha256(CKPT_BYTES).hexdigest(),
    )
    base.update(kw)
    return DevpopAuditCheckpointIdentity(**base)


def _write_checkpoint(tmp_path, *, epoch: int = 1, data: bytes = CKPT_BYTES) -> Path:
    ckpt = Path(tmp_path) / f"model_epoch{epoch:03d}.pt"
    ckpt.write_bytes(data)
    return ckpt


def _setup(
    tmp_path,
    *,
    run_basins=BASINS,
    pop_basins=BASINS,
    dates=None,
    obs_sim=None,
    trial_id="trial-A",
    configuration_id="cfg-A",
    epoch=1,
    ckpt_bytes=CKPT_BYTES,
):
    """Build a complete valid synthetic environment (package, NH result pickle,
    checkpoint file, and a producer provenance receipt bound to the ACTUAL
    checkpoint + result bytes); return the evaluator kwargs."""
    population = _population(pop_basins)
    contract = _contract(_population(BASINS))  # contract always covers the full BASINS set
    _write_package(tmp_path / "pkg", BASINS)
    _write_run_pickle(tmp_path / "run", run_basins, epoch=epoch, dates=dates, obs_sim=obs_sim)
    ckpt = _write_checkpoint(tmp_path, epoch=epoch, data=ckpt_bytes)
    receipt = build_devpop_audit_provenance_receipt(
        trial_id=trial_id,
        configuration_id=configuration_id,
        run_dir=tmp_path / "run",
        period="validation",
        checkpoint_epoch=epoch,
        checkpoint_path=ckpt,
    )
    return dict(
        checkpoint_identity=_identity(
            trial_id=trial_id,
            configuration_id=configuration_id,
            checkpoint_epoch=epoch,
            checkpoint_sha256=hashlib.sha256(ckpt_bytes).hexdigest(),
        ),
        run_dir=tmp_path / "run",
        package_root=tmp_path / "pkg",
        population=population,
        contract=contract,
        provenance_receipt=receipt,
        checkpoint_path=ckpt,
    )


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #

def test_vertical_synthetic_audit_row_happy_path(tmp_path):
    kwargs = _setup(tmp_path)
    row = evaluate_devpop_common120_audit_row(**kwargs)

    # (1) explicit checkpoint / trial identity survives verbatim onto the row
    ckpt_digest = hashlib.sha256(CKPT_BYTES).hexdigest()
    assert row["trial_id"] == "trial-A"
    assert row["configuration_id"] == "cfg-A"
    assert row["checkpoint_epoch"] == 1
    assert row["checkpoint_sha256"] == ckpt_digest
    assert row["checkpoint_identity"] == {
        "trial_id": "trial-A",
        "configuration_id": "cfg-A",
        "checkpoint_epoch": 1,
        "checkpoint_sha256": ckpt_digest,
    }

    # (2) the already-produced NH result was consumed; diagnostic scope only
    assert row["objective_scope"] == "devpop_audit"
    assert row["contract_id"] == "common120_raw_space_nse_devpop_audit_v001"

    # (3) exact frozen contract timestamps applied -> real raw-space path ran
    result = row["result"]
    assert result["n_basins_requested"] == 3
    assert result["n_basins_evaluated"] == 3
    assert result["n_basins_excluded"] == 0
    assert result["evaluated_basin_ids"] == sorted(BASINS)
    assert [r["n_admitted"] for r in result["per_basin"]] == [20, 20, 20]
    assert [r["n_common120_support_eligible"] for r in result["per_basin"]] == [20, 20, 20]

    # (4) aggregate / count accounting generated and reconciled
    assert result["aggregate"]["n_admitted_total"] == 60
    assert result["aggregate"]["n_sim_nonfinite_at_admitted_total"] == 0
    assert result["aggregate"]["metrics"]["nse"]["n_finite_basins"] == 3

    # (5) support alignment is actually CONSUMED: NSE == 1.0 is only reachable
    #     if scoring was restricted to the frozen 20-hour Common-120 subset,
    #     because sim is deliberately broken on every off-support hour.
    for r in result["per_basin"]:
        assert r["nse"] == pytest.approx(1.0)
        assert r["n_sim_nonfinite_at_admitted"] == 0
    assert row["nse_median"] == pytest.approx(1.0)

    #     ... and the "unmasked" NSE (all 60 hours) is materially different,
    #     so NSE == 1.0 genuinely corresponds to the support subset.
    obs, sim = _basin_arrays(100)
    err = sim - obs
    unmasked_nse = 1.0 - np.sum(err ** 2) / np.sum((obs - obs.mean()) ** 2)
    assert unmasked_nse < 0.0

    # (6) synthetic A1 completeness passed -> fixture-labelled row, never canonical
    assert row["fixture_completeness"] is True
    assert row["completeness"]["fixture_completeness"] is True
    assert row["canonical_completeness"] is False
    assert row["canonical_population_verified"] is False
    assert "canonical_completeness" not in row["completeness"]

    # (7) the provenance receipt was verified against the ACTUAL consumed files
    assert row["provenance_verified"] is True
    prov = row["provenance"]
    assert prov["schema"] == PROVENANCE_RECEIPT_SCHEMA
    assert prov["provenance_verified"] is True
    assert prov["trial_id"] == "trial-A"
    assert prov["configuration_id"] == "cfg-A"
    assert prov["checkpoint_epoch"] == 1
    assert prov["period"] == "validation"
    assert prov["period_results_relpath"] == "validation/model_epoch001/validation_results.p"
    ckpt = kwargs["checkpoint_path"]
    results_p = kwargs["run_dir"] / "validation" / "model_epoch001" / "validation_results.p"
    assert prov["checkpoint_sha256"] == hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert prov["period_results_sha256"] == hashlib.sha256(results_p.read_bytes()).hexdigest()
    assert row["checkpoint_sha256"] == prov["checkpoint_sha256"]


def test_provenance_receipt_accepted_as_written_json_path(tmp_path):
    kwargs = _setup(tmp_path)
    receipt_path = tmp_path / "provenance_receipt.json"
    write_devpop_audit_provenance_receipt(kwargs["provenance_receipt"], receipt_path)
    reloaded = load_devpop_audit_provenance_receipt(receipt_path)
    assert reloaded == kwargs["provenance_receipt"]

    kwargs["provenance_receipt"] = receipt_path  # consumer accepts a path
    row = evaluate_devpop_common120_audit_row(**kwargs)
    assert row["provenance_verified"] is True


# --------------------------------------------------------------------------- #
# focused fail-closed tests at THIS consumer boundary
# --------------------------------------------------------------------------- #

def test_frozen_support_timestamp_absent_from_run_coordinate_fails_closed(tmp_path):
    # run pickle carries a disjoint date coordinate -> no frozen support
    # timestamp is present; the evaluator must refuse to realign/truncate.
    kwargs = _setup(tmp_path, dates=_val_dates(start="2024-09-01"))
    with pytest.raises(DevpopAuditEvaluatorError, match="refusing to silently realign"):
        evaluate_devpop_common120_audit_row(**kwargs)


def test_expected_basin_missing_from_nh_result_fails_closed(tmp_path):
    kwargs = _setup(tmp_path, run_basins=BASINS[:2])  # third population basin absent
    with pytest.raises(DevpopAuditEvaluatorError, match="missing from period_results"):
        evaluate_devpop_common120_audit_row(**kwargs)


def test_nonfinite_simulation_at_admitted_timestamp_fails_closed(tmp_path):
    obs, sim = _basin_arrays(100)
    sim = sim.copy()
    sim[SUPPORT_IDX[0]] = np.nan
    kwargs = _setup(tmp_path, obs_sim={BASINS[0]: (obs, sim)})
    with pytest.raises(DevpopAuditEvaluatorError, match="non-finite simulation at an admitted Common-120 timestamp"):
        evaluate_devpop_common120_audit_row(**kwargs)


def test_frozen_admitted_observation_not_naturally_finite_fails_closed(tmp_path):
    obs, sim = _basin_arrays(100)
    obs = obs.copy()
    obs[SUPPORT_IDX[0]] = np.nan
    kwargs = _setup(tmp_path, obs_sim={BASINS[0]: (obs, sim)})
    with pytest.raises(DevpopAuditEvaluatorError, match="not a naturally finite observation"):
        evaluate_devpop_common120_audit_row(**kwargs)


def test_declared_checkpoint_sha_contradicting_the_receipt_fails_closed(tmp_path):
    kwargs = _setup(tmp_path)
    # a well-formed but wrong declared checkpoint SHA -> disagrees with the receipt
    kwargs["checkpoint_identity"] = _identity(checkpoint_sha256="0" * 64)
    with pytest.raises(DevpopAuditEvaluatorError, match="contradicts the declared checkpoint identity"):
        evaluate_devpop_common120_audit_row(**kwargs)


def test_population_contract_basin_mismatch_fails_closed(tmp_path):
    kwargs = _setup(tmp_path, pop_basins=BASINS)
    kwargs["population"] = _population(("00000001", "00000002", "00000009"))
    with pytest.raises(DevpopAuditEvaluatorError, match="do not equal the expected population"):
        evaluate_devpop_common120_audit_row(**kwargs)


def test_fixture_population_cannot_satisfy_the_canonical_path(tmp_path):
    kwargs = _setup(tmp_path)
    with pytest.raises((DevpopAuditEvaluatorError, DevpopAuditContractError, DevpopAuditCompletenessError)):
        evaluate_devpop_common120_audit_row(require_canonical=True, **kwargs)


# --------------------------------------------------------------------------- #
# provenance boundary: result <-> checkpoint/trial binding (repair 1)
# --------------------------------------------------------------------------- #

def test_result_pickle_modified_after_receipt_fails_closed(tmp_path):
    kwargs = _setup(tmp_path)
    # rewrite the consumed {period}_results.p AFTER the receipt was created
    _write_run_pickle(kwargs["run_dir"], BASINS, obs_sim={BASINS[0]: _basin_arrays(999)})
    with pytest.raises(
        DevpopAuditEvaluatorError,
        match="period-results artifact consumed does not match the provenance receipt",
    ):
        evaluate_devpop_common120_audit_row(**kwargs)


def test_checkpoint_bytes_modified_after_receipt_fails_closed(tmp_path):
    kwargs = _setup(tmp_path)
    # tamper the checkpoint file; declared identity still matches the receipt,
    # so the failure is specifically the on-disk bytes vs the receipt.
    kwargs["checkpoint_path"].write_bytes(CKPT_BYTES + b"-tampered")
    with pytest.raises(
        DevpopAuditEvaluatorError,
        match="checkpoint artifact consumed does not match the provenance receipt",
    ):
        evaluate_devpop_common120_audit_row(**kwargs)


def test_receipt_epoch_contradiction_fails_closed(tmp_path):
    kwargs = _setup(tmp_path)
    tampered = dict(kwargs["provenance_receipt"].to_mapping())
    tampered["checkpoint_epoch"] = 2
    kwargs["provenance_receipt"] = tampered
    with pytest.raises(DevpopAuditEvaluatorError, match="checkpoint_epoch.*contradicts the declared checkpoint identity"):
        evaluate_devpop_common120_audit_row(**kwargs)


def test_receipt_trial_contradiction_fails_closed(tmp_path):
    kwargs = _setup(tmp_path)
    tampered = dict(kwargs["provenance_receipt"].to_mapping())
    tampered["configuration_id"] = "cfg-OTHER"
    kwargs["provenance_receipt"] = tampered
    with pytest.raises(DevpopAuditEvaluatorError, match="configuration_id.*contradicts the declared checkpoint identity"):
        evaluate_devpop_common120_audit_row(**kwargs)


def test_receipt_period_path_contradiction_fails_closed(tmp_path):
    kwargs = _setup(tmp_path)
    tampered = dict(kwargs["provenance_receipt"].to_mapping())
    tampered["period_results_relpath"] = "validation/model_epoch002/validation_results.p"
    kwargs["provenance_receipt"] = tampered
    with pytest.raises(DevpopAuditEvaluatorError, match="not the canonical period/epoch result path"):
        evaluate_devpop_common120_audit_row(**kwargs)


def test_missing_provenance_evidence_fails_closed(tmp_path):
    # (a) a receipt mapping missing a required field
    kwargs = _setup(tmp_path)
    incomplete = dict(kwargs["provenance_receipt"].to_mapping())
    incomplete.pop("period_results_sha256")
    kwargs["provenance_receipt"] = incomplete
    with pytest.raises(DevpopAuditEvaluatorError, match="missing="):
        evaluate_devpop_common120_audit_row(**kwargs)

    # (b) the checkpoint artifact named by the receipt is absent from disk
    kwargs = _setup(tmp_path)
    kwargs["checkpoint_path"].unlink()
    with pytest.raises(DevpopAuditEvaluatorError, match="checkpoint artifact does not exist"):
        evaluate_devpop_common120_audit_row(**kwargs)


def test_receipt_cannot_be_built_without_the_real_artifacts(tmp_path):
    # producer-side builder also refuses to fabricate a receipt for absent files
    with pytest.raises(DevpopAuditEvaluatorError, match="does not exist"):
        build_devpop_audit_provenance_receipt(
            trial_id="t",
            configuration_id="c",
            run_dir=tmp_path / "nope",
            period="validation",
            checkpoint_epoch=1,
            checkpoint_path=tmp_path / "absent.pt",
        )


# --------------------------------------------------------------------------- #
# canonical authority is not forgeable (repair 2)
# --------------------------------------------------------------------------- #

def test_no_forgeable_row_level_canonical_assertion_is_exposed():
    # the removed helper trusted caller-set booleans on a row; it is gone.
    assert not hasattr(evaluator_module, "assert_canonical_devpop_audit_row")
    assert "assert_canonical_devpop_audit_row" not in evaluator_module.__all__


def test_fabricated_canonical_booleans_do_not_establish_canonical_authority(tmp_path):
    # A hand-built row carrying every canonical-looking True flag is inert:
    # nothing in the module consumes row-level booleans as canonical evidence,
    # and the genuine canonical path still fails closed on a synthetic fixture.
    forged_row = {
        "schema": "flashnh_stage1_devpop_common120_audit_row_v001",
        "canonical_completeness": True,
        "canonical_population_verified": True,
        "completeness": {
            "canonical_completeness": True,
            "canonical_population_verified": True,
        },
    }
    canonical_asserters = [
        name
        for name in evaluator_module.__all__
        if "canonical" in name and name.startswith(("assert", "require", "validate"))
    ]
    assert canonical_asserters == []

    kwargs = _setup(tmp_path)
    with pytest.raises((DevpopAuditEvaluatorError, DevpopAuditContractError, DevpopAuditCompletenessError)):
        evaluate_devpop_common120_audit_row(require_canonical=True, **kwargs)
    # the forged row never became canonical evidence anywhere
    assert forged_row["canonical_completeness"] is True  # still just a dict the caller made

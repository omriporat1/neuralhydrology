"""Focused tests for the Common-336 full-development comparison preparation
(src/baseline/seqlen_common336_full_development.py) and its additive
``contract_schema`` seam in the fixed-support scorer.

Synthetic inputs only. The package uses the real experiment constants (floor
336, L in {120, 240, 336}, lead 6, a calendar 2024 validation start with
December-2023 history) on a short horizon. Every test except the vertical one
is non-training (fake NH pickles). The vertical test trains two tiny
NeuralHydrology models on synthetic data and is skipped unless explicitly
opted in with FLASHNH_RUN_NH_TRAINING_TESTS=1.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.baseline import fixed_support_contract_v2 as fixed  # noqa: E402
from src.baseline import seqlen_common336_full_development as fd  # noqa: E402
from src.baseline.devpop_common120_audit_contract import (  # noqa: E402
    DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256,
    DEVELOPMENT_TRAIN_ROLE,
    canonical_membership_sha256,
)

# --------------------------------------------------------------------------- #
# Synthetic package
# --------------------------------------------------------------------------- #
START = pd.Timestamp("2023-11-01")
N_HOURS = 2016                            # through 2024-01-23T23
VAL_START, VAL_END = "2024-01-01", "2024-01-21"
VAL_FIRST = pd.Timestamp("2024-01-01T00")
LAST_ISSUE = pd.Timestamp("2024-01-21T17")  # 2024-01-21T23 - lead
CUTOFF = "2024-01-21T23:00:00"
LEAD = 6
TARGET = "qobs_mm_per_h_lead06"
DYNAMIC = ["mrms_qpe_1h_mm", "rtma_2t_K"]
AREA_KM2 = 100.0
BASINS = ["B01", "B02", "B03"]
GAPS = ["2023-11-20T05", "2024-01-05T12", "2024-01-23T00"]   # documented, every basin
INTERNAL_GAP = pd.Timestamp("2024-01-05T12")
INPUT_NAN = {"B02": ["2023-12-20T00"]}    # in the 336 h window of early-January issues only; not in gap file
TARGET_NAN = {"B03": ["2024-01-03T02"]}
IDENTITIES = {"gap_artifact_sha256": "1" * 64, "package_manifest_sha256": "a" * 64,
              "package_file_checksums_sha256": "b" * 64, "package_run_provenance_sha256": "c" * 64,
              "development_split_sha256": "d" * 64, "spatial_holdout_split_sha256": "e" * 64}
SCREENING_SHA = "9" * 64


def _hour(ts) -> int:
    return int((pd.Timestamp(ts) - START) / pd.Timedelta(hours=1))


def _build_package(root: Path, *, input_nan=None, target_nan=None, qobs_nan_basins=(), missing=()) -> Path:
    input_nan = INPUT_NAN if input_nan is None else input_nan
    target_nan = TARGET_NAN if target_nan is None else target_nan
    dates = pd.date_range(START, periods=N_HOURS, freq="h")
    for sub in ("time_series", "basins", "masks"):
        (root / sub).mkdir(parents=True)
    rng = np.random.default_rng(11)
    for k, basin in enumerate(BASINS):
        mrms = rng.gamma(0.5, 1.0, N_HOURS)
        rtma = 270.0 + 10.0 * np.sin(np.arange(N_HOURS) / 24.0) + k
        for ts in GAPS:
            mrms[_hour(ts)] = rtma[_hour(ts)] = np.nan
        for ts in input_nan.get(basin, []):
            rtma[_hour(ts)] = np.nan
        qobs = 5.0 + np.convolve(np.nan_to_num(mrms), np.ones(12), mode="same")
        target = np.full(N_HOURS, np.nan)
        target[:-LEAD] = 3.6 * qobs[LEAD:] / AREA_KM2
        for ts in target_nan.get(basin, []):
            target[_hour(ts)] = np.nan
        if basin in qobs_nan_basins:
            qobs[:] = np.nan
        if basin in missing:
            continue
        xr.Dataset({DYNAMIC[0]: ("date", mrms), DYNAMIC[1]: ("date", rtma), "qobs_m3s": ("date", qobs),
                    TARGET: ("date", target)}, coords={"date": dates}).to_netcdf(root / "time_series" / f"{basin}.nc")
    for period in ("train", "validation", "test"):
        (root / "basins" / f"{period}.txt").write_text("\n".join(BASINS) + "\n")
    (root / "masks" / "gap_timestamps.json").write_text(json.dumps([str(pd.Timestamp(t)) + "Z" for t in GAPS]))
    return root


def _compute(root: Path, *, basins=BASINS, role=fd.FIXTURE_ONLY_ROLE) -> dict:
    from src.baseline.gap_mask_io import load_gap_timestamps_json

    return fd.compute_full_development_support(
        package_root=root, basin_ids=basins, population_role=role, date_start=VAL_START, date_end=VAL_END,
        gap_timestamps=load_gap_timestamps_json(root / "masks" / "gap_timestamps.json"), dynamic_inputs=DYNAMIC,
        target_variable=TARGET, lead_hours=LEAD, identities=IDENTITIES, screening_subset_artifact_sha256=SCREENING_SHA)


def _support(contract, basin) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(fd.full_development_support_dates(contract, basin))


def _rows(result) -> dict:
    return {r["basin_id"]: r for r in result["accounting"]["rows"]}


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    root = _build_package(tmp_path_factory.mktemp("c336fd_pkg"))
    result = _compute(root)
    contract_path = fd.write_full_development_support_contract(result["contract"], root / "support" / "contract.json")
    return {"root": root, "result": result, "contract": result["contract"], "contract_path": contract_path}


# --------------------------------------------------------------------------- #
# 1. Support contract
# --------------------------------------------------------------------------- #

def test_expected_support_and_loss_causes(package):
    contract = package["contract"]
    first_block_end = INTERNAL_GAP - pd.Timedelta(hours=1)
    after_gap = INTERNAL_GAP + pd.Timedelta(hours=336)          # first issue whose 336 h window clears the gap
    b01 = pd.date_range(VAL_FIRST, first_block_end, freq="h").append(pd.date_range(after_gap, LAST_ISSUE, freq="h"))
    assert _support(contract, "B01").equals(b01) and len(b01) == 162
    # B02: its own 2023-12-20T00 rtma NaN blocks 2024-01-01T00..2024-01-02T23 at L336 only.
    assert _support(contract, "B02").equals(b01[b01 > pd.Timestamp("2024-01-02T23")])
    assert _support(contract, "B03").equals(b01[b01 != pd.Timestamp("2024-01-03T02")])
    assert contract["eligible_counts"] == {"B01": 162, "B02": 114, "B03": 161}

    rows = _rows(package["result"])
    # Internal gap: L120 natural issues 2024-01-10T12..2024-01-19T11 (216 h) lack a continuous 336 h history.
    assert rows["B01"]["n_natural_L120"] == 108 + (LAST_ISSUE - (INTERNAL_GAP + pd.Timedelta(hours=120))) // pd.Timedelta(hours=1) + 1
    assert rows["B01"]["lost_documented_gap_L120"] == 216
    assert rows["B01"]["lost_documented_gap_L240"] == 96
    assert rows["B02"]["lost_nonfinite_input_L120"] == rows["B02"]["lost_nonfinite_input_L240"] == 48
    for row in rows.values():
        assert row["lost_timeline_start_L120"] == row["lost_timeline_start_L336"] == 0
        assert row["n_natural_L336"] == row["n_admitted"]
        for L in fd.ACCOUNTED_SEQ_LENGTHS:
            assert row[f"n_natural_L{L}"] - row["n_admitted"] == sum(
                row[f"{c}_L{L}"] for c in ("lost_timeline_start", "lost_documented_gap", "lost_nonfinite_input"))
    assert contract["per_basin_accounting"]["B01"]["n_natural_L120"] == rows["B01"]["n_natural_L120"]
    # Recorded identity: development population key, never the screening contract's key.
    assert contract["development_basin_ids_sha256"] == canonical_membership_sha256(BASINS)
    assert "screening_basin_ids_sha256" not in contract
    assert contract["screening_subset_artifact_sha256"] == SCREENING_SHA
    assert contract["observation_cutoff"] == CUTOFF and contract["seq_length_floor"] == 336
    assert fd.load_full_development_support_contract(package["contract_path"]) == contract


def test_year_start_admitted_from_december_history_and_one_missing_hour_removes_it(tmp_path):
    jan1 = VAL_FIRST
    window_start = jan1 - pd.Timedelta(hours=335)               # 2023-12-18T01
    assert window_start == pd.Timestamp("2023-12-18T01")
    outside = _compute(_build_package(tmp_path / "outside", input_nan={"B01": ["2023-12-18T00"]}))
    assert _support(outside["contract"], "B01")[0] == jan1
    inside = _compute(_build_package(tmp_path / "inside", input_nan={"B01": [str(window_start)]}))
    support = _support(inside["contract"], "B01")
    assert jan1 not in support and support[0] == jan1 + pd.Timedelta(hours=1)
    row = _rows(inside)["B01"]
    # Still a natural L120/L240 candidate; excluded from the common support used by every length.
    assert row["n_natural_L120"] == _rows(outside)["B01"]["n_natural_L120"]
    assert row["lost_nonfinite_input_L120"] == row["lost_nonfinite_input_L240"] == 1


def test_zero_support_and_unreadable_basin_fail_with_accounting(tmp_path):
    zero = _build_package(tmp_path / "zero", target_nan={"B03": [str(t) for t in pd.date_range(VAL_FIRST, LAST_ISSUE, freq="h")]})
    with pytest.raises(fd.FullDevelopmentSupportBuildError, match="zero Common-336 support") as exc:
        _compute(zero)
    assert exc.value.accounting["failures"] == [{"basin_id": "B03", "reason": "zero Common-336 support"}]
    assert {r["basin_id"] for r in exc.value.accounting["rows"]} == set(BASINS)
    missing = _build_package(tmp_path / "missing", missing=("B02",))
    with pytest.raises(fd.FullDevelopmentSupportBuildError, match="missing basin NetCDF"):
        _compute(missing)


def test_development_role_requires_pinned_membership(package):
    with pytest.raises(fd.FullDevelopmentError, match="pinned 2307-basin membership"):
        _compute(package["root"], role=DEVELOPMENT_TRAIN_ROLE)
    with pytest.raises(fd.FullDevelopmentError, match="population_role"):
        _compute(package["root"], role="screening")
    with pytest.raises(fd.FullDevelopmentError, match="unique"):
        _compute(package["root"], basins=["B01", "B01"])


def _rechecksum(contract):
    contract = json.loads(json.dumps(contract))
    contract.pop("checksum_sha256")
    from src.baseline.seqlen_common336_support import _checksum
    contract["checksum_sha256"] = _checksum(contract)
    return contract


def test_contract_tamper_and_malformed_support_rejected(package):
    contract = package["contract"]
    silent = json.loads(json.dumps(contract))
    silent["per_basin_accounting"]["B02"]["n_nonfinite_input_hours_not_gap"] += 1
    with pytest.raises(fd.FullDevelopmentError, match="checksum mismatch"):
        fd.validate_full_development_support_contract(silent)
    tampered = json.loads(json.dumps(contract))
    tampered["eligible_counts"]["B01"] -= 1
    tampered["per_basin_intervals"]["B01"][-1][1] = str(LAST_ISSUE - pd.Timedelta(hours=1))
    tampered["per_basin_accounting"]["B01"]["n_admitted"] -= 1
    tampered["per_basin_accounting"]["B01"]["n_natural_L336"] -= 1
    with pytest.raises(fd.FullDevelopmentError):          # accounting/loss partition broken or checksum
        fd.validate_full_development_support_contract(tampered)
    stale = json.loads(json.dumps(contract))
    stale["date_end"] = "2024-01-20"
    stale["observation_cutoff"] = "2024-01-20T23:00:00"
    with pytest.raises(fd.FullDevelopmentError, match="outside admissible issue times"):
        fd.validate_full_development_support_contract(_rechecksum(stale))
    for key, value in (("observation_cutoff", "2024-12-31T23:00:00"), ("seq_length_floor", 120),
                       ("development_basin_ids_sha256", "0" * 64)):
        bad = json.loads(json.dumps(contract))
        bad[key] = value
        with pytest.raises(fd.FullDevelopmentError):
            fd.validate_full_development_support_contract(_rechecksum(bad))
    renamed = json.loads(json.dumps(contract))
    renamed["screening_basin_ids_sha256"] = renamed.pop("development_basin_ids_sha256")
    with pytest.raises(fd.FullDevelopmentError, match="contract keys"):
        fd.validate_full_development_support_contract(renamed)
    assert fd.validate_full_development_support_contract(json.loads(json.dumps(contract))) == contract
    with pytest.raises(fd.FullDevelopmentError, match="refusing to overwrite"):
        fd.write_full_development_support_contract(contract, package["contract_path"])


def test_contract_schemas_are_not_interchangeable(package, tmp_path):
    contract = package["contract"]
    with pytest.raises(fixed.FixedSupportContractError):
        fixed.validate_fixed_support_contract(contract, expected_contract_id=fd.CONTRACT_ID,
                                              expected_seq_length_floor=336)
    common = dict(run_dir=tmp_path, epoch=1, package_root=package["root"], contract=contract,
                  expected_contract_id=fd.CONTRACT_ID, expected_seq_length_floor=336)
    with pytest.raises(fixed.FixedSupportContractError):          # default v2 schema refuses it
        fixed.evaluate_fixed_support_raw_space_metrics(**common)
    with pytest.raises(fixed.FixedSupportContractError, match="unknown fixed-support contract schema"):
        fixed.evaluate_fixed_support_raw_space_metrics(**common, contract_schema="something_else")
    with pytest.raises(fixed.FixedSupportContractError, match="400-basin screening gate"):
        fixed.evaluate_fixed_support_raw_space_metrics(**common, contract_schema=fd.CONTRACT_SCHEMA_NAME,
                                                       require_full_screening_population=True)
    with pytest.raises(fixed.FixedSupportContractError, match="expected"):
        fixed.evaluate_fixed_support_raw_space_metrics(**dict(common, expected_contract_id="common336_screening"),
                                                       contract_schema=fd.CONTRACT_SCHEMA_NAME)
    with pytest.raises(fd.FullDevelopmentError):
        fd.require_production_contract_identity(contract)


# --------------------------------------------------------------------------- #
# 2. Completeness around the scorer (fake NH pickles)
# --------------------------------------------------------------------------- #

def _fake_run(root: Path, package_root: Path, name: str, *, nan_sim_at=None, drop=()) -> Path:
    from src.baseline.nh_seed_evaluation import weight_stem

    run_dir = root / name
    period_dir = run_dir / "validation" / weight_stem(1)
    period_dir.mkdir(parents=True)
    (run_dir / "config.yml").write_text(f'experiment_name: {name}\nflashnh_observation_cutoff: "{CUTOFF}"\n')
    dates = pd.date_range(VAL_FIRST, pd.Timestamp(f"{VAL_END}T23"), freq="h")
    results = {}
    for basin in BASINS:
        if basin in drop:
            continue
        with xr.open_dataset(package_root / "time_series" / f"{basin}.nc") as ds:
            obs = ds[TARGET].sel(date=dates).values.astype(np.float64)
        sim = 0.9 * np.nan_to_num(obs) + 0.01
        if nan_sim_at and nan_sim_at[0] == basin:
            sim[dates.get_loc(pd.Timestamp(nan_sim_at[1]))] = np.nan
        results[basin] = {"1h": {"xr": xr.Dataset({f"{TARGET}_obs": ("date", obs), f"{TARGET}_sim": ("date", sim)},
                                                  coords={"date": dates})}}
    with open(period_dir / "validation_results.p", "wb") as fh:
        pickle.dump(results, fh)
    return run_dir


def _canary(run_dir, package_root, contract, basins=BASINS):
    return fd.evaluate_full_development_canary(run_dir=run_dir, epoch=1, package_root=package_root, contract=contract,
                                               canary_basin_ids=basins, fixture_only_allow_unqualified_contract=True)


def test_canary_passes_and_records_completeness(package, tmp_path):
    result = _canary(_fake_run(tmp_path, package["root"], "ok"), package["root"], package["contract"])
    assert result["n_basins_evaluated"] == 3 and result["n_basins_excluded"] == 0
    assert result["observation_cutoff"] == str(pd.Timestamp(CUTOFF))
    assert {r["basin_id"]: r["n_admitted"] for r in result["per_basin"]} == package["contract"]["eligible_counts"]
    assert result["completeness"]["scope"] == "canary"
    # The canary still requires a fixture contract to declare itself and production contracts otherwise.
    with pytest.raises(fd.FullDevelopmentError):
        fd.evaluate_full_development_canary(run_dir=tmp_path / "ok", epoch=1, package_root=package["root"],
                                            contract=package["contract"], canary_basin_ids=BASINS)


def test_nonfinite_prediction_at_admitted_hour_fails_closed(package, tmp_path):
    run_dir = _fake_run(tmp_path, package["root"], "nan", nan_sim_at=("B02", "2024-01-20T00"))
    with pytest.raises(fd.FullDevelopmentError, match="non-finite prediction"):
        _canary(run_dir, package["root"], package["contract"])
    # A NaN prediction OUTSIDE the support is irrelevant (2024-01-07T00 is inside the internal-gap exclusion).
    _canary(_fake_run(tmp_path, package["root"], "nan_out", nan_sim_at=("B02", "2024-01-07T00")),
            package["root"], package["contract"])


def test_area_derivation_failure_fails_whole_evaluation(package, tmp_path):
    bad_pkg = _build_package(tmp_path / "pkg_area", qobs_nan_basins=("B03",))
    run_dir = _fake_run(tmp_path, bad_pkg, "area")
    raw = fixed.evaluate_fixed_support_raw_space_metrics(
        run_dir=run_dir, epoch=1, package_root=bad_pkg, contract=package["contract"],
        contract_schema=fd.CONTRACT_SCHEMA_NAME, expected_contract_id=fd.CONTRACT_ID, expected_seq_length_floor=336)
    assert raw["n_basins_excluded"] == 1  # the unchanged scorer soft-excludes...
    with pytest.raises(fd.FullDevelopmentError, match="exclusions"):  # ...the gate fails the evaluation
        _canary(run_dir, bad_pkg, package["contract"])


def test_omitted_canary_basin_and_wrong_cutoff_fail(package, tmp_path):
    run_dir = _fake_run(tmp_path, package["root"], "drop", drop=("B02",))
    with pytest.raises(fixed.FixedSupportContractError, match="missing from this run"):
        _canary(run_dir, package["root"], package["contract"])
    ok = _fake_run(tmp_path, package["root"], "ok2")
    result = _canary(ok, package["root"], package["contract"], basins=["B01", "B03"])
    # The declared subset is enforced exactly: a result lacking one declared basin is rejected.
    result["per_basin"] = [r for r in result["per_basin"] if r["basin_id"] != "B03"]
    with pytest.raises(fd.FullDevelopmentError, match="missing \\['B03'\\]"):
        fd.require_complete_scored_population(result, contract=package["contract"], expected_basin_ids=["B01", "B03"])
    with pytest.raises(fd.FullDevelopmentError, match="outside the contract"):
        _canary(ok, package["root"], package["contract"], basins=["B01", "B99"])
    (ok / "config.yml").write_text('flashnh_observation_cutoff: "2024-01-20T23:00:00"\n')
    with pytest.raises(fd.FullDevelopmentError, match="observation cutoff"):
        _canary(ok, package["root"], package["contract"])


def _dev_ids() -> list:
    path = REPO_ROOT / "config" / "stage1_baseline_splits_v001" / "development_train.txt"
    ids = sorted(line.strip() for line in path.read_text().splitlines() if line.strip())
    assert len(ids) == 2307 and canonical_membership_sha256(ids) == DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256
    return ids


def _fabricated_production_contract(ids) -> dict:
    record = {key: 0 for key in fd._ACCOUNTING_KEYS}
    record.update({"n_admitted": 1, **{f"n_natural_L{L}": 1 for L in fd.ACCOUNTED_SEQ_LENGTHS}})
    return fd.assemble_full_development_contract(
        basin_ids=ids, population_role=DEVELOPMENT_TRAIN_ROLE, date_start=fd.PRODUCTION_DATE_START,
        date_end=fd.PRODUCTION_DATE_END, dynamic_inputs=fd.PRODUCTION_DYNAMIC_INPUTS,
        target_variable=fd.PRODUCTION_TARGET_VARIABLE, lead_hours=fd.PRODUCTION_LEAD_HOURS,
        identities=fd._production_identities(),
        screening_subset_artifact_sha256=fd._production_identities()["screening_subset_artifact_sha256"],
        per_basin_intervals={b: [["2024-03-01T00:00:00", "2024-03-01T00:00:00"]] for b in ids},
        eligible_counts={b: 1 for b in ids}, per_basin_accounting={b: dict(record) for b in ids})


def _fabricated_result(contract, ids, *, excluded=()):
    digest = fd.admitted_hours_sha256(np.array(["2024-03-01T00"], dtype="datetime64[ns]"))
    rows = [{"basin_id": b, "n_admitted": 1, "n_fixed_support_eligible": 1, "n_sim_nonfinite_at_admitted": 0,
             "admitted_hours_sha256": digest, **{m: 0.5 for m in fd._REQUIRED_FINITE_METRICS}} for b in ids]
    return {"contract_id": contract["contract_id"], "contract_checksum_sha256": contract["checksum_sha256"],
            "seq_length_floor": 336, "observation_cutoff": str(pd.Timestamp(fd.PRODUCTION_OBSERVATION_CUTOFF)),
            "n_basins_requested": len(ids) + len(excluded), "n_basins_evaluated": len(ids),
            "n_basins_excluded": len(excluded), "basins_excluded": [{"basin_id": b, "reason": "x"} for b in excluded],
            "per_basin": rows,
            "aggregate": {"n_basins": len(ids), "n_admitted_total": len(ids), "n_sim_nonfinite_at_admitted_total": 0}}


def test_full_development_gate_requires_exactly_2307_basins():
    ids = _dev_ids()
    contract = _fabricated_production_contract(ids)
    fd.require_production_contract_identity(contract)
    fd.require_full_development_scored_population(_fabricated_result(contract, ids), contract=contract)
    for subset, excluded in ((ids[:-1], ()), (ids[:1000], ()), (ids[:-1], ids[-1:])):
        with pytest.raises(fd.FullDevelopmentError):
            fd.require_full_development_scored_population(_fabricated_result(contract, subset, excluded=excluded),
                                                          contract=contract)
    nonfinite = _fabricated_result(contract, ids)
    nonfinite["per_basin"][7]["nse"] = float("nan")
    with pytest.raises(fd.FullDevelopmentError, match="not finite"):
        fd.require_full_development_scored_population(nonfinite, contract=contract)
    # A 1000-basin subsample (NH validate_n_random_basins) is not a canary either unless declared.
    with pytest.raises(fd.FullDevelopmentError):
        fd.require_complete_scored_population(_fabricated_result(contract, ids[:1000]), contract=contract,
                                              expected_basin_ids=ids)


def test_cross_run_support_identity(package, tmp_path):
    a = _canary(_fake_run(tmp_path, package["root"], "a"), package["root"], package["contract"])
    b = _canary(_fake_run(tmp_path, package["root"], "b"), package["root"], package["contract"])
    summary = fd.require_identical_support_across_runs({"a": a, "b": b}, contract=package["contract"],
                                                       expected_run_ids=["a", "b"])
    assert summary == {"n_runs": 2, "contract_checksum_sha256": package["contract"]["checksum_sha256"],
                       "n_basins": 3, "n_admitted_total": 437}
    with pytest.raises(fd.FullDevelopmentError, match="run set mismatch"):
        fd.require_identical_support_across_runs({"a": a}, contract=package["contract"], expected_run_ids=["a", "b"])
    b["per_basin"][0]["n_admitted"] -= 1
    with pytest.raises(fd.FullDevelopmentError, match="admitted"):
        fd.require_identical_support_across_runs({"a": a, "b": b}, contract=package["contract"])


def test_equal_counts_but_different_hours_fail(package, tmp_path, monkeypatch):
    contract = package["contract"]
    a = _canary(_fake_run(tmp_path, package["root"], "a"), package["root"], contract)
    b = _canary(_fake_run(tmp_path, package["root"], "b"), package["root"], contract)
    assert a["per_basin"][0]["admitted_hours_sha256"] == fd.admitted_hours_sha256(_support(contract, "B01"))
    shifted = fd.admitted_hours_sha256(_support(contract, "B01") + pd.Timedelta(hours=1))   # same count
    b["per_basin"][0]["admitted_hours_sha256"] = shifted
    assert b["per_basin"][0]["n_admitted"] == a["per_basin"][0]["n_admitted"]
    with pytest.raises(fd.FullDevelopmentError, match="admitted hours differ"):
        fd.require_identical_support_across_runs({"a": a, "b": b}, contract=contract)
    with pytest.raises(fd.FullDevelopmentError, match="not the contract support hours"):
        fd.require_complete_scored_population(b, contract=contract, expected_basin_ids=BASINS)

    # The digest is taken from the scorer's own admitted series: a scorer that
    # reported B01 on hours shifted by one (same count) fails the canary gate.
    real = fd.evaluate_fixed_support_raw_space_metrics

    def shifted_scorer(**kwargs):
        result = real(**kwargs)
        s = result["admitted_series_by_basin"]["B01"]
        result["admitted_series_by_basin"]["B01"] = type(s)(basin_id=s.basin_id, date=s.date + np.timedelta64(1, "h"),
                                                            obs_m3s=s.obs_m3s, sim_m3s=s.sim_m3s)
        return result

    monkeypatch.setattr(fd, "evaluate_fixed_support_raw_space_metrics", shifted_scorer)
    with pytest.raises(fd.FullDevelopmentError, match="'B01': scored admitted hours"):
        _canary(_fake_run(tmp_path, package["root"], "c"), package["root"], contract)


def _screening_contract(root: Path, screening=("B01", "B03")) -> dict:
    """Built by the producer of the qualified screening contract itself."""
    from src.baseline import seqlen_common336_support as c336
    from src.baseline.gap_mask_io import load_gap_timestamps_json

    out = c336.compute_common336_support(
        package_root=root, train_basins=BASINS, screening_basins=list(screening),
        train_period=("2023-11-01T00", "2023-12-31T23"), validation_period=(VAL_FIRST, f"{VAL_END}T23"),
        gap_timestamps=load_gap_timestamps_json(root / "masks" / "gap_timestamps.json"), dynamic_inputs=DYNAMIC,
        target_variable=TARGET, lead_hours=LEAD, seq_lengths=c336.SEQ_LENGTHS, identities=IDENTITIES,
        train_basin_ids_sha256="8" * 64, screening_basin_ids_sha256=SCREENING_SHA)
    return out["screening_contract"]


def test_screening_basins_reproduce_the_qualified_screening_support(package):
    screening = _screening_contract(package["root"])
    summary = fd.require_screening_support_reproduced(package["contract"], screening)
    assert summary["n_screening_basins"] == 2 and summary["n_admitted_total"] == 162 + 161

    # One hour removed from one screening basin's support: rejected, nothing reconciled.
    bad = json.loads(json.dumps(screening))
    bad["per_basin_support"]["B03"] = bad["per_basin_support"]["B03"][1:]
    bad["eligible_counts"]["B03"] -= 1
    bad["checksum_sha256"] = hashlib.sha256(fixed._canonical_payload_for_checksum(bad)).hexdigest()
    with pytest.raises(fd.FullDevelopmentError, match="1 of 2 screening basins do not reproduce"):
        fd.require_screening_support_reproduced(package["contract"], bad)

    # An identity disagreement is rejected before any hour comparison.
    other = json.loads(json.dumps(screening))
    other["development_split_sha256"] = "f" * 64
    other["checksum_sha256"] = hashlib.sha256(fixed._canonical_payload_for_checksum(other)).hexdigest()
    with pytest.raises(fd.FullDevelopmentError, match="development_split_sha256"):
        fd.require_screening_support_reproduced(package["contract"], other)
    # A production contract accepts only the pinned qualified screening contract.
    ids = _dev_ids()
    with pytest.raises(fd.FullDevelopmentError, match="not the qualified"):
        fd.require_screening_support_reproduced(_fabricated_production_contract(ids), screening)


def test_selected_checkpoints_are_the_frozen_nine():
    rows = fd.SELECTED_CHECKPOINTS
    assert [(c.seq_length, c.seed, c.epoch) for c in rows] == [
        (120, 967139, 4), (240, 967139, 9), (336, 967139, 11), (120, 329911494, 7), (240, 329911494, 8),
        (336, 329911494, 9), (120, 3023892610, 5), (240, 3023892610, 5), (336, 3023892610, 12)]
    assert len({c.model_sha256 for c in rows}) == 9 and len({c.config_sha256 for c in rows}) == 9
    assert all(c.nh_run_dir_name.startswith(c.run_id + "_1610_") for c in rows)
    assert rows[0].run_id == "seqlen_common336_seed3_v001__L120__seedA967139__mf12x50000__attempt001"


# --------------------------------------------------------------------------- #
# 3. Staging
# --------------------------------------------------------------------------- #

def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _realistic_source(tmp_path: Path, package_root: Path, *, name="srcrun", stamp="1610_143831") -> tuple:
    """A source NH run directory shaped like the real ones (config keys and
    values of the qualified Common-336 runs, synthetic paths)."""
    from src.baseline.nh_seed_evaluation import weight_stem

    run_dir = tmp_path / "runs" / f"{name}_{stamp}"
    (run_dir / "train_data").mkdir(parents=True)
    (run_dir / "validation" / weight_stem(4)).mkdir(parents=True)
    (run_dir / "validation" / weight_stem(4) / "validation_results.p").write_bytes(b"qualified-400-results")
    (run_dir / f"{weight_stem(4)}.pt").write_bytes(b"model-bytes-epoch4")
    (run_dir / f"{weight_stem(5)}.pt").write_bytes(b"model-bytes-epoch5")
    (run_dir / "train_data" / "train_data_scaler.yml").write_text("xarray_feature_center: {}\n")
    p19 = tmp_path / "p19" / "screening_400.txt"
    p19.parent.mkdir(exist_ok=True)
    p19.write_text("B01\nB02\n")
    dyn = "\n".join(f"- {d}" for d in DYNAMIC)
    (run_dir / "config.yml").write_text(f"""batch_size: 256
data_dir: {package_root.as_posix()}
dataset: flashnh
device: cuda:0
dynamic_inputs:
{dyn}
epochs: 12
experiment_name: {name}
flashnh_common_support_file: {(tmp_path / 'support' / 'train_support.json').as_posix()}
flashnh_observation_cutoff: '{CUTOFF}'
hidden_size: 128
img_log_dir: {(run_dir / 'img_log').as_posix()}
learning_rate: 0.001
loss: NSE
max_updates_per_epoch: 50000
model: cudalstm
number_of_basins: 3
output_dropout: 0.4
predict_last_n: 1
run_dir: {run_dir.as_posix()}
seed: 967139
seq_length: 120
static_attributes:
- area_km2
statics_embedding:
  type: fc
  hiddens:
  - 64
target_variables:
- {TARGET}
test_basin_file: {(package_root / 'basins' / 'test.txt').as_posix()}
test_end_date: 23/01/2024
test_start_date: 22/01/2024
train_basin_file: {(package_root / 'basins' / 'train.txt').as_posix()}
train_dir: {(run_dir / 'train_data').as_posix()}
train_end_date: 31/12/2023
train_start_date: 01/11/2023
validate_every: 3
validate_n_random_basins: 1000
validation_basin_file: {p19.as_posix()}
validation_end_date: 21/01/2024
validation_start_date: 01/01/2024
""")
    checkpoint = fd.SelectedCheckpoint(
        qualification_index=1, run_id=name, nh_run_dir_name=run_dir.name, seq_length=120, seed=967139, epoch=4,
        model_sha256=_sha(run_dir / f"{weight_stem(4)}.pt"),
        scaler_sha256=_sha(run_dir / "train_data" / "train_data_scaler.yml"), config_sha256=_sha(run_dir / "config.yml"))
    dev_file = tmp_path / "development_train.txt"
    dev_file.write_text("\n".join(BASINS) + "\n")
    return run_dir, checkpoint, dev_file


def _tree_bytes(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_staging_copies_verified_artifacts_and_changes_only_whitelisted_keys(package, tmp_path):
    from neuralhydrology.utils.config import Config

    from src.baseline.nh_register import register_flashnh_dataset
    from src.baseline.nh_seed_evaluation import NHSeedEvaluationError, raise_if_evaluation_only_bundle

    source, checkpoint, dev_file = _realistic_source(tmp_path, package["root"])
    before = _tree_bytes(source)
    out = tmp_path / "staged" / checkpoint.run_id
    manifest = fd.stage_full_development_eval_run(
        source_run_dir=source, out_run_dir=out, checkpoint=checkpoint, contract=package["contract"],
        development_basin_file=dev_file, fixture_only_allow_unqualified_contract=True)
    assert _tree_bytes(source) == before                     # source (incl. qualified results) untouched
    assert manifest["source_tree_unchanged"] and manifest["scope"] == "full_development"
    assert {c["key"] for c in manifest["config_changes"]} == {
        "validation_basin_file", "validate_n_random_basins", "run_dir", "train_dir", "img_log_dir"}
    staged_cfg = fd._read_yaml(out / "config.yml")
    source_cfg = fd._read_yaml(source / "config.yml")
    assert staged_cfg["validate_n_random_basins"] == 3 and source_cfg["validate_n_random_basins"] == 1000
    assert (out / fd.STAGED_VALIDATION_BASIN_FILENAME).read_text() == "B01\nB02\nB03\n"
    assert staged_cfg["flashnh_observation_cutoff"] == source_cfg["flashnh_observation_cutoff"] == CUTOFF
    for key in set(source_cfg) - set(fd.ALLOWED_STAGED_CONFIG_CHANGES):
        assert staged_cfg[key] == source_cfg[key], key
    assert _sha(out / "model_epoch004.pt") == checkpoint.model_sha256
    assert _sha(out / "train_data" / "train_data_scaler.yml") == checkpoint.scaler_sha256
    assert not (out / "model_epoch005.pt").exists() and not (out / "validation").exists()
    with pytest.raises(NHSeedEvaluationError):
        raise_if_evaluation_only_bundle(out)
    recorded = json.loads((out / fd.STAGING_MANIFEST_FILENAME).read_text())
    assert recorded["staged_sha256"]["config.yml"] == _sha(out / "config.yml")
    assert recorded["contract_checksum_sha256"] == package["contract"]["checksum_sha256"]
    register_flashnh_dataset()
    cfg = Config(out / "config.yml")
    assert cfg.validate_n_random_basins == 3 and Path(cfg.validation_basin_file) == (out / fd.STAGED_VALIDATION_BASIN_FILENAME).resolve()

    # Refusals: existing output, collision with the source, SHA mismatch, membership mismatch, production pin.
    with pytest.raises(fd.FullDevelopmentError, match="existing staging directory"):
        fd.stage_full_development_eval_run(source_run_dir=source, out_run_dir=out, checkpoint=checkpoint,
                                           contract=package["contract"], development_basin_file=dev_file,
                                           fixture_only_allow_unqualified_contract=True)
    with pytest.raises(fd.FullDevelopmentError):
        fd.stage_full_development_eval_run(source_run_dir=source, out_run_dir=source / "eval", checkpoint=checkpoint,
                                           contract=package["contract"], development_basin_file=dev_file,
                                           fixture_only_allow_unqualified_contract=True)
    from dataclasses import replace
    for bad, match in ((replace(checkpoint, scaler_sha256="0" * 64), "scaler sha256"),
                       (replace(checkpoint, config_sha256="0" * 64), "config.yml sha256"),
                       (replace(checkpoint, seq_length=336), "seq_length"),
                       (replace(checkpoint, nh_run_dir_name="other"), "source run dir")):
        with pytest.raises(fd.FullDevelopmentError, match=match):
            fd.stage_full_development_eval_run(source_run_dir=source, out_run_dir=tmp_path / "x" / match,
                                               checkpoint=bad, contract=package["contract"],
                                               development_basin_file=dev_file,
                                               fixture_only_allow_unqualified_contract=True)
    short = tmp_path / "short.txt"
    short.write_text("B01\nB02\n")
    with pytest.raises(fd.FullDevelopmentError, match="membership"):
        fd.stage_full_development_eval_run(source_run_dir=source, out_run_dir=tmp_path / "y", checkpoint=checkpoint,
                                           contract=package["contract"], development_basin_file=short,
                                           fixture_only_allow_unqualified_contract=True)
    with pytest.raises(fd.FullDevelopmentError):
        fd.stage_full_development_eval_run(source_run_dir=source, out_run_dir=tmp_path / "z", checkpoint=checkpoint,
                                           contract=package["contract"], development_basin_file=dev_file)
    assert _tree_bytes(source) == before


def test_staging_refuses_a_changed_cutoff_in_the_source(package, tmp_path):
    source, checkpoint, dev_file = _realistic_source(tmp_path, package["root"])
    text = (source / "config.yml").read_text().replace(CUTOFF, "2024-01-20T23:00:00")
    (source / "config.yml").write_text(text)
    from dataclasses import replace
    checkpoint = replace(checkpoint, config_sha256=_sha(source / "config.yml"))
    with pytest.raises(fd.FullDevelopmentError, match="flashnh_observation_cutoff"):
        fd.stage_full_development_eval_run(source_run_dir=source, out_run_dir=tmp_path / "o", checkpoint=checkpoint,
                                           contract=package["contract"], development_basin_file=dev_file,
                                           fixture_only_allow_unqualified_contract=True)


# --------------------------------------------------------------------------- #
# Vertical: real NH train -> source eval (subsampled) -> stage -> staged NH eval -> scorer + gates
# --------------------------------------------------------------------------- #

def _train_cfg_text(root: Path, *, seq_length: int, name: str) -> str:
    dyn = "\n".join(f"  - {d}" for d in DYNAMIC)
    return f"""experiment_name: {name}
train_basin_file: {(root / 'basins' / 'train.txt').as_posix()}
validation_basin_file: {(root / 'basins' / 'validation.txt').as_posix()}
test_basin_file: {(root / 'basins' / 'test.txt').as_posix()}
run_dir: {(root / 'runs').as_posix()}
data_dir: {root.as_posix()}
dataset: flashnh
train_start_date: "01/11/2023"
train_end_date: "31/12/2023"
validation_start_date: "01/01/2024"
validation_end_date: "21/01/2024"
test_start_date: "22/01/2024"
test_end_date: "23/01/2024"
target_variables:
  - {TARGET}
dynamic_inputs:
{dyn}
model: cudalstm
hidden_size: 4
head: regression
output_activation: linear
predict_last_n: 1
batch_size: 64
optimizer: Adam
learning_rate: 0.001
loss: NSE
epochs: 1
save_weights_every: 1
log_interval: 50
validate_n_random_basins: 2
log_tensorboard: False
num_workers: 0
seq_length: {seq_length}
seed: 967139
device: cpu
verbose: 0
flashnh_observation_cutoff: "{CUTOFF}"
"""


@pytest.mark.skipif(os.environ.get("FLASHNH_RUN_NH_TRAINING_TESTS") != "1",
                    reason="trains tiny NH models; opt in with FLASHNH_RUN_NH_TRAINING_TESTS=1")
def test_vertical_stage_then_score_full_population_on_identical_support(package, tmp_path):
    from neuralhydrology.evaluation.evaluate import start_evaluation
    from neuralhydrology.training.train import start_training
    from neuralhydrology.utils.config import Config

    from src.baseline.nh_register import register_flashnh_dataset

    register_flashnh_dataset()
    root = tmp_path / "pkg"
    shutil.copytree(package["root"], root)
    contract = package["contract"]
    dev_file = tmp_path / "development_train.txt"
    dev_file.write_text("\n".join(BASINS) + "\n")
    results = {}
    for L in (120, 336):
        name = f"vfd_L{L}"
        cfg_path = root / f"{name}.yml"
        cfg_path.write_text(_train_cfg_text(root, seq_length=L, name=name))
        start_training(Config(cfg_path))
        (source,) = [p for p in (root / "runs").iterdir() if p.name.startswith(name + "_")]
        # Source evaluation as NH would do it: subsampled to validate_n_random_basins=2.
        start_evaluation(Config(source / "config.yml"), source, epoch=1, period="validation")
        with open(source / "validation" / "model_epoch001" / "validation_results.p", "rb") as fh:
            assert len(pickle.load(fh)) == 2
        source_bytes = _tree_bytes(source)
        checkpoint = fd.SelectedCheckpoint(
            qualification_index=0, run_id=name, nh_run_dir_name=source.name, seq_length=L, seed=967139, epoch=1,
            model_sha256=_sha(source / "model_epoch001.pt"),
            scaler_sha256=_sha(source / "train_data" / "train_data_scaler.yml"), config_sha256=_sha(source / "config.yml"))
        staged = tmp_path / "staged" / name
        fd.stage_full_development_eval_run(source_run_dir=source, out_run_dir=staged, checkpoint=checkpoint,
                                           contract=contract, development_basin_file=dev_file,
                                           fixture_only_allow_unqualified_contract=True)
        start_evaluation(Config(staged / "config.yml"), staged, epoch=1, period="validation")
        with open(staged / "validation" / "model_epoch001" / "validation_results.p", "rb") as fh:
            assert sorted(pickle.load(fh)) == BASINS          # no 1000/2-basin subsampling in the staged run
        results[name] = _canary(staged, root, contract)
        assert _tree_bytes(source) == source_bytes            # original run folder never written
    summary = fd.require_identical_support_across_runs(results, contract=contract, expected_run_ids=sorted(results))
    assert summary["n_basins"] == 3 and summary["n_admitted_total"] == sum(contract["eligible_counts"].values())
    for result in results.values():
        assert result["observation_cutoff"] == str(pd.Timestamp(CUTOFF))
        assert all(row["n_sim_nonfinite_at_admitted"] == 0 for row in result["per_basin"])

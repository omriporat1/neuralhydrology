"""Population-completeness tests for the v2 fixed-support evaluator.

These use explicitly synthetic in-memory epoch results.  The production gate
is enabled only for the 400-basin cases; the small-fixture case proves that
generic evaluator use remains available only when that gate is disabled.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.baseline import fixed_support_contract_v2 as fixed
from src.baseline.nh_raw_space_evaluation import RawSpaceEvaluationError
from src.baseline.sweep_v2_six_axis_campaign import OBJECTIVE_ID_V2


class _Array:
    def __init__(self, values):
        self.values = np.asarray(values)


class _Dataset:
    def __init__(self):
        self.coords = {"date": _Array([0, 1, 2])}
        self.data_vars = {"qobs_mm_per_h_lead06_obs", "qobs_mm_per_h_lead06_sim"}
        self._values = {
            "qobs_mm_per_h_lead06_obs": _Array([1.0, 2.0, 3.0]),
            "qobs_mm_per_h_lead06_sim": _Array([1.0, 2.0, 3.0]),
        }

    def __getitem__(self, key):
        return self._values[key]


def _contract(n_basins: int) -> dict:
    ids = [f"{value:08d}" for value in range(n_basins)]
    return fixed.build_fixed_support_contract(
        contract_id=OBJECTIVE_ID_V2, lead_hours=6, target_variable="qobs_mm_per_h_lead06",
        period="fixture", date_start="2024-01-01", date_end="2024-01-01",
        source_gap_policy_identity="fixture_gap_v001", screening_basin_ids_sha256="0" * 64,
        package_manifest_sha256="a"*64, package_file_checksums_sha256="b"*64, package_run_provenance_sha256="c"*64,
        development_split_sha256="d"*64, spatial_holdout_split_sha256="e"*64,
        per_basin_date={basin_id: np.array([0, 1, 2]) for basin_id in ids},
        per_basin_admitted={basin_id: np.array([True, True, True]) for basin_id in ids},
    )


def _wire_synthetic_epoch(monkeypatch, contract, *, area_failure=None, metric_id=None, metric_nse=0.5):
    dataset = _Dataset()
    period = {basin_id: {"1h": {"xr": dataset}} for basin_id in contract["basin_ids"]}
    monkeypatch.setattr(fixed, "load_period_results", lambda *_: period)
    monkeypatch.setattr(fixed, "basin_netcdf_path", lambda *_: "fixture.nc")

    def area(*_, basin_id, **__):
        if basin_id == area_failure:
            raise RawSpaceEvaluationError("synthetic area failure")
        return SimpleNamespace(area_km2=100.0, consistent=True, relative_mad=0.0)

    def metric(*, basin_id, **_):
        return {
            "basin_id": metric_id(basin_id) if metric_id else basin_id,
            "nse": metric_nse, "n_sim_nonfinite_at_admitted": 0, "n_admitted": 3,
        }

    def aggregate(rows):
        finite = sum(np.isfinite(row["nse"]) for row in rows)
        return {"n_basins": len(rows), "metrics": {"nse": {"n_finite_basins": finite, "median": 0.5}}}

    monkeypatch.setattr(fixed, "derive_basin_area_km2_from_netcdf", area)
    monkeypatch.setattr(fixed, "evaluate_basin_raw_space", metric)
    monkeypatch.setattr(fixed, "aggregate_raw_space_metrics", aggregate)
    return period


def test_production_fixed_support_requires_all_400_evaluated_basins(monkeypatch):
    contract = _contract(400)
    _wire_synthetic_epoch(monkeypatch, contract)
    result = fixed.evaluate_fixed_support_raw_space_metrics(
        run_dir="fixture", epoch=1, package_root="fixture", contract=contract,
        basin_ids=contract["basin_ids"], require_full_screening_population=True,
    )
    assert (result["n_basins_requested"], result["n_basins_evaluated"], result["n_basins_excluded"]) == (400, 400, 0)
    assert {row["basin_id"] for row in result["per_basin"]} == set(contract["basin_ids"])


def test_production_fixed_support_refuses_area_exclusion_and_finite_399_basin_aggregate(monkeypatch):
    contract = _contract(400)
    _wire_synthetic_epoch(monkeypatch, contract, area_failure=contract["basin_ids"][-1])
    with pytest.raises(fixed.FixedSupportContractError, match="400 requested, 400 evaluated, zero excluded"):
        fixed.evaluate_fixed_support_raw_space_metrics(
            run_dir="fixture", epoch=1, package_root="fixture", contract=contract,
            basin_ids=contract["basin_ids"], require_full_screening_population=True,
        )


def test_production_fixed_support_refuses_missing_or_extra_or_duplicate_evaluated_identity(monkeypatch):
    contract = _contract(400)
    _wire_synthetic_epoch(monkeypatch, contract, metric_id=lambda basin_id: "unexpected" if basin_id == "00000000" else basin_id)
    with pytest.raises(fixed.FixedSupportContractError, match="evaluated basin IDs"):
        fixed.evaluate_fixed_support_raw_space_metrics(
            run_dir="fixture", epoch=1, package_root="fixture", contract=contract,
            basin_ids=contract["basin_ids"], require_full_screening_population=True,
        )

    _wire_synthetic_epoch(monkeypatch, contract, metric_id=lambda _: "00000000")
    with pytest.raises(fixed.FixedSupportContractError, match="evaluated basin IDs"):
        fixed.evaluate_fixed_support_raw_space_metrics(
            run_dir="fixture", epoch=1, package_root="fixture", contract=contract,
            basin_ids=contract["basin_ids"], require_full_screening_population=True,
        )


def test_production_fixed_support_refuses_missing_basin_result_and_nonfinite_basin_metric(monkeypatch):
    contract = _contract(400)
    period = _wire_synthetic_epoch(monkeypatch, contract)
    del period[contract["basin_ids"][-1]]
    with pytest.raises(fixed.FixedSupportContractError, match="missing from this run"):
        fixed.evaluate_fixed_support_raw_space_metrics(
            run_dir="fixture", epoch=1, package_root="fixture", contract=contract,
            basin_ids=contract["basin_ids"], require_full_screening_population=True,
        )

    _wire_synthetic_epoch(monkeypatch, contract, metric_nse=float("nan"))
    with pytest.raises(fixed.FixedSupportContractError, match="finite NSE"):
        fixed.evaluate_fixed_support_raw_space_metrics(
            run_dir="fixture", epoch=1, package_root="fixture", contract=contract,
            basin_ids=contract["basin_ids"], require_full_screening_population=True,
        )


def test_small_synthetic_fixture_is_supported_only_when_production_completeness_is_disabled(monkeypatch):
    contract = _contract(1)
    _wire_synthetic_epoch(monkeypatch, contract)
    result = fixed.evaluate_fixed_support_raw_space_metrics(
        run_dir="fixture", epoch=1, package_root="fixture", contract=contract,
    )
    assert result["n_basins_evaluated"] == 1
    with pytest.raises(fixed.FixedSupportContractError, match="exactly 400 unique"):
        fixed.evaluate_fixed_support_raw_space_metrics(
            run_dir="fixture", epoch=1, package_root="fixture", contract=contract,
            require_full_screening_population=True,
        )

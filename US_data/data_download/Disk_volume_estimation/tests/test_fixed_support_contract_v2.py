"""Population-completeness tests for the v2 fixed-support evaluator.

These use explicitly synthetic in-memory epoch results.  The production gate
is enabled only for the 400-basin cases; the small-fixture case proves that
generic evaluator use remains available only when that gate is disabled.
"""
from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from src.baseline import fixed_support_contract_v2 as fixed
from src.baseline.nh_raw_space_evaluation import RawSpaceEvaluationError
from src.baseline.nh_seed_evaluation import weight_stem
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


# --------------------------------------------------------------------------- #
# RD1-C4-A: admitted-series result contract (return_admitted_series)
# --------------------------------------------------------------------------- #

def test_return_admitted_series_disabled_leaves_legacy_shape_unchanged(monkeypatch):
    contract = _contract(1)
    _wire_synthetic_epoch(monkeypatch, contract)
    result = fixed.evaluate_fixed_support_raw_space_metrics(
        run_dir="fixture", epoch=1, package_root="fixture", contract=contract,
    )
    assert "admitted_series_by_basin" not in result
    assert set(result) == {
        "objective_scope", "contract_id", "contract_checksum_sha256", "seq_length_floor",
        "n_basins_requested", "n_basins_evaluated", "n_basins_excluded", "basins_excluded",
        "per_basin", "aggregate",
    }


def test_admitted_series_arrays_are_read_only():
    # Defensive immutability: public scientific results must not be
    # mutable by a caller after construction.
    series = fixed.AdmittedSeries(
        basin_id="b1", date=np.arange(3), obs_m3s=np.array([1.0, 2.0, 3.0]), sim_m3s=np.array([1.1, 2.1, 3.1]),
    )
    with pytest.raises(ValueError):
        series.obs_m3s[0] = 999.0
    with pytest.raises(ValueError):
        series.sim_m3s[0] = 999.0
    with pytest.raises(ValueError):
        series.date[0] = 999


def test_admitted_series_copies_defensively_and_leaves_caller_arrays_untouched():
    # RD1-C4 review follow-on: __post_init__ must store an independent copy
    # of each caller-supplied array, never a view/alias of it -- a caller's
    # own original array must remain writeable and independent of the
    # stored AdmittedSeries value, while the stored copy itself rejects
    # mutation.
    date = np.arange(3)
    obs = np.array([1.0, 2.0, 3.0])
    sim = np.array([1.1, 2.1, 3.1])
    series = fixed.AdmittedSeries(basin_id="b1", date=date, obs_m3s=obs, sim_m3s=sim)

    assert not np.shares_memory(date, series.date)
    assert not np.shares_memory(obs, series.obs_m3s)
    assert not np.shares_memory(sim, series.sim_m3s)
    assert date.flags.writeable and obs.flags.writeable and sim.flags.writeable

    obs[0] = 999.0
    assert series.obs_m3s[0] == 1.0  # mutating the caller's original has no effect on the stored copy

    with pytest.raises(ValueError):
        series.obs_m3s[0] = 999.0


def _write_package_basin_netcdf(package_root, basin_id, *, area_km2, lead_hours, n, seed=1):
    rng = np.random.default_rng(seed)
    qobs_m3s = rng.uniform(1.0, 200.0, size=n)
    usable_n = n - lead_hours
    target_mm_per_h = np.full(n, np.nan)
    target_mm_per_h[:usable_n] = 3.6 * qobs_m3s[lead_hours:lead_hours + usable_n] / area_km2
    ts_dir = package_root / "time_series"
    ts_dir.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {
            "qobs_m3s": ("date", qobs_m3s),
            "qobs_mm_per_h_lead06": ("date", target_mm_per_h),
        },
        coords={"date": np.arange(n)},
    ).to_netcdf(ts_dir / f"{basin_id}.nc")


def _write_validation_pickle(run_dir, epoch, basin_results):
    period_dir = run_dir / "validation" / weight_stem(epoch)
    period_dir.mkdir(parents=True, exist_ok=True)
    with open(period_dir / "validation_results.p", "wb") as fh:
        pickle.dump(basin_results, fh)


def test_return_admitted_series_vertical_real_load_support_alignment_and_conversion(tmp_path):
    """One vertical synthetic test using a real validation-result pickle and a
    real package NetCDF (no mocking of loading/support/alignment/conversion),
    proving the RD1-C4-A producer-consumer interface end to end (Interface /
    Consumer Contract Gate item 7)."""
    basin_id = "00000001"
    area_km2 = 100.0
    lead_hours = 6
    target_variable = "qobs_mm_per_h_lead06"
    n = 300

    package_root = tmp_path / "package"
    _write_package_basin_netcdf(package_root, basin_id, area_km2=area_km2, lead_hours=lead_hours, n=n)

    # Frozen contract support: 20 timestamps, deliberately serialized in a
    # different order than the run's own ascending date coordinate, so the
    # test proves alignment by timestamp identity rather than row position.
    rng = np.random.default_rng(7)
    support_indices = np.sort(rng.choice(n - lead_hours, size=20, replace=False))
    shuffled_support = rng.permutation(support_indices).astype(np.int64)

    contract = fixed.build_fixed_support_contract(
        contract_id=OBJECTIVE_ID_V2, lead_hours=lead_hours, target_variable=target_variable,
        period="validation", date_start="2024-01-01", date_end="2024-01-01",
        source_gap_policy_identity="fixture_gap_v001", screening_basin_ids_sha256="0" * 64,
        package_manifest_sha256="a" * 64, package_file_checksums_sha256="b" * 64,
        package_run_provenance_sha256="c" * 64, development_split_sha256="d" * 64,
        spatial_holdout_split_sha256="e" * 64,
        per_basin_date={basin_id: shuffled_support},
        per_basin_admitted={basin_id: np.ones(len(shuffled_support), dtype=bool)},
    )
    assert list(contract["per_basin_support"][basin_id]) == [int(v) for v in shuffled_support]
    assert not np.array_equal(shuffled_support, support_indices)  # genuinely reordered

    obs_mm_per_h = 0.01 * (np.arange(n) + 1.0)
    sim_mm_per_h = 2.0 * obs_mm_per_h
    nonfinite_index = int(support_indices[0])
    sim_mm_per_h[nonfinite_index] = np.nan

    period_dataset = xr.Dataset(
        {
            f"{target_variable}_obs": ("date", obs_mm_per_h),
            f"{target_variable}_sim": ("date", sim_mm_per_h),
        },
        coords={"date": np.arange(n)},
    )
    run_dir = tmp_path / "run"
    _write_validation_pickle(run_dir, 1, {basin_id: {"1h": {"xr": period_dataset}}})

    result = fixed.evaluate_fixed_support_raw_space_metrics(
        run_dir=run_dir, epoch=1, package_root=package_root, contract=contract,
        return_admitted_series=True,
    )

    assert result["n_basins_evaluated"] == 1
    assert {row["basin_id"] for row in result["per_basin"]} == {basin_id}
    metric_row = result["per_basin"][0]
    assert metric_row["n_sim_nonfinite_at_admitted"] == 1

    series_by_basin = result["admitted_series_by_basin"]
    assert set(series_by_basin) == {row["basin_id"] for row in result["per_basin"]}
    series = series_by_basin[basin_id]
    assert isinstance(series, fixed.AdmittedSeries)
    assert series.basin_id == basin_id

    # Canonical order is the frozen contract's per_basin_support order, not
    # the run's ascending date order.
    np.testing.assert_array_equal(series.date, shuffled_support)
    assert len(series.date) == len(series.obs_m3s) == len(series.sim_m3s) == len(shuffled_support)

    expected_obs_m3s = obs_mm_per_h[shuffled_support] * area_km2 / 3.6
    expected_sim_m3s = sim_mm_per_h[shuffled_support] * area_km2 / 3.6  # NaN preserved at nonfinite_index
    np.testing.assert_allclose(series.obs_m3s, expected_obs_m3s)
    nonfinite_position = int(np.where(shuffled_support == nonfinite_index)[0][0])
    assert np.isnan(series.sim_m3s[nonfinite_position])
    finite_positions = np.arange(len(shuffled_support)) != nonfinite_position
    np.testing.assert_allclose(series.sim_m3s[finite_positions], expected_sim_m3s[finite_positions])
    assert np.isnan(expected_sim_m3s[nonfinite_position])


# --------------------------------------------------------------------------- #
# RD1-C4 review Finding 4: canonicalize datetime units before admitted-series
# identity indexing (frozen contract dates always deserialize to
# ``datetime64[ns]``; a run's own date coordinate may be a different,
# equally-valid ``datetime64`` unit).
# --------------------------------------------------------------------------- #

def _datetime_contract(basin_id: str, dates_ns: np.ndarray) -> dict:
    return fixed.build_fixed_support_contract(
        contract_id=OBJECTIVE_ID_V2, lead_hours=6, target_variable="qobs_mm_per_h_lead06",
        period="fixture", date_start="2024-01-01", date_end="2024-01-01",
        source_gap_policy_identity="fixture_gap_v001", screening_basin_ids_sha256="0" * 64,
        package_manifest_sha256="a" * 64, package_file_checksums_sha256="b" * 64,
        package_run_provenance_sha256="c" * 64, development_split_sha256="d" * 64,
        spatial_holdout_split_sha256="e" * 64,
        per_basin_date={basin_id: dates_ns},
        per_basin_admitted={basin_id: np.ones(len(dates_ns), dtype=bool)},
    )


def _wire_datetime_epoch(monkeypatch, *, basin_id, run_dates):
    dataset = _Dataset()
    dataset.coords["date"] = _Array(run_dates)
    dataset._values = {
        "qobs_mm_per_h_lead06_obs": _Array(np.arange(len(run_dates), dtype=np.float64) + 1.0),
        "qobs_mm_per_h_lead06_sim": _Array(np.arange(len(run_dates), dtype=np.float64) + 1.0),
    }
    period = {basin_id: {"1h": {"xr": dataset}}}
    monkeypatch.setattr(fixed, "load_period_results", lambda *_: period)
    monkeypatch.setattr(fixed, "basin_netcdf_path", lambda *_: "fixture.nc")
    monkeypatch.setattr(
        fixed, "derive_basin_area_km2_from_netcdf",
        lambda *_, **__: SimpleNamespace(area_km2=100.0, consistent=True, relative_mad=0.0),
    )

    def metric(*, basin_id, obs_mm_per_h, sim_mm_per_h, area_km2, return_admitted_arrays):
        admitted_mask = np.isfinite(obs_mm_per_h)
        row = {
            "basin_id": basin_id, "nse": 0.5,
            "n_sim_nonfinite_at_admitted": 0, "n_admitted": int(admitted_mask.sum()),
        }
        if return_admitted_arrays:
            row["_admitted_obs_m3s"] = obs_mm_per_h[admitted_mask] * area_km2 / 3.6
            row["_admitted_sim_m3s"] = sim_mm_per_h[admitted_mask] * area_km2 / 3.6
        return row

    monkeypatch.setattr(fixed, "evaluate_basin_raw_space", metric)
    monkeypatch.setattr(fixed, "aggregate_raw_space_metrics", lambda rows: {"n_basins": len(rows), "metrics": {}})
    return period


def test_datetime_unit_mismatch_between_run_and_contract_reorders_successfully(monkeypatch):
    basin_id = "00000001"
    # Day-aligned timestamps so a day-resolution run coordinate loses no
    # information relative to the contract's always-datetime64[ns] support.
    dates_ns = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[ns]")
    contract = _datetime_contract(basin_id, dates_ns)
    assert contract["date_dtype"] == "datetime64"

    # Run's own date coordinate is day-resolution -- coarser than the
    # contract's always-datetime64[ns] serialized support. Equal by value,
    # but a naive dict-keyed reindex would previously KeyError or silently
    # misalign because the two arrays' scalar elements hash differently.
    run_dates = dates_ns.astype("datetime64[D]")
    _wire_datetime_epoch(monkeypatch, basin_id=basin_id, run_dates=run_dates)

    result = fixed.evaluate_fixed_support_raw_space_metrics(
        run_dir="fixture", epoch=1, package_root="fixture", contract=contract,
        return_admitted_series=True,
    )
    assert result["n_basins_evaluated"] == 1
    series = result["admitted_series_by_basin"][basin_id]
    np.testing.assert_array_equal(series.date, dates_ns)
    np.testing.assert_allclose(series.obs_m3s, np.array([1.0, 2.0, 3.0]) * 100.0 / 3.6)


def test_datetime_unit_mismatch_another_compatible_unit_succeeds(monkeypatch):
    basin_id = "00000001"
    dates_ns = np.array(["2024-01-01T00", "2024-01-01T01", "2024-01-01T02"], dtype="datetime64[ns]")
    contract = _datetime_contract(basin_id, dates_ns)

    run_dates = dates_ns.astype("datetime64[s]")
    _wire_datetime_epoch(monkeypatch, basin_id=basin_id, run_dates=run_dates)

    result = fixed.evaluate_fixed_support_raw_space_metrics(
        run_dir="fixture", epoch=1, package_root="fixture", contract=contract,
        return_admitted_series=True,
    )
    series = result["admitted_series_by_basin"][basin_id]
    np.testing.assert_array_equal(series.date, dates_ns)


def test_datetime_duplicate_run_timestamps_fail_after_normalization(monkeypatch):
    basin_id = "00000001"
    dates_ns = np.array(["2024-01-01T00", "2024-01-01T01"], dtype="datetime64[ns]")
    contract = _datetime_contract(basin_id, dates_ns)

    # Duplicate at the run's own (non-ns) unit; the duplicate check must
    # fire on the canonicalized array, not be skipped or corrupted by the
    # unit-normalization step.
    run_dates = np.array(["2024-01-01T00", "2024-01-01T00"], dtype="datetime64[s]")
    _wire_datetime_epoch(monkeypatch, basin_id=basin_id, run_dates=run_dates)

    with pytest.raises(fixed.FixedSupportContractError, match="duplicates"):
        fixed.evaluate_fixed_support_raw_space_metrics(
            run_dir="fixture", epoch=1, package_root="fixture", contract=contract,
        )


def test_datetime_genuinely_absent_support_timestamp_fails_closed(monkeypatch):
    basin_id = "00000001"
    dates_ns = np.array(["2024-01-01T00", "2024-01-01T01", "2024-01-01T02"], dtype="datetime64[ns]")
    contract = _datetime_contract(basin_id, dates_ns)

    # The run's own date coordinate is missing the contract's third
    # admitted timestamp entirely (not merely a different unit).
    run_dates = dates_ns[:2].astype("datetime64[s]")
    _wire_datetime_epoch(monkeypatch, basin_id=basin_id, run_dates=run_dates)

    with pytest.raises(fixed.FixedSupportContractError, match="date/period contradiction"):
        fixed.evaluate_fixed_support_raw_space_metrics(
            run_dir="fixture", epoch=1, package_root="fixture", contract=contract,
        )


def test_datetime_unsupported_dtype_raises_clear_error(monkeypatch):
    basin_id = "00000001"
    dates_ns = np.array(["2024-01-01T00", "2024-01-01T01"], dtype="datetime64[ns]")
    contract = _datetime_contract(basin_id, dates_ns)

    # A run date coordinate that is neither datetime64 nor integer for a
    # 'datetime64' contract must fail closed with a clear identity message,
    # not a raw numpy/TypeError deep inside dict construction.
    run_dates = np.array([1.0, 2.0])
    _wire_datetime_epoch(monkeypatch, basin_id=basin_id, run_dates=run_dates)

    with pytest.raises(fixed.FixedSupportContractError, match="expected a datetime64"):
        fixed.evaluate_fixed_support_raw_space_metrics(
            run_dir="fixture", epoch=1, package_root="fixture", contract=contract,
        )


def test_integer_timestamp_canonicalization_remains_correct(monkeypatch):
    contract = _contract(1)
    basin_id = contract["basin_ids"][0]
    _wire_datetime_epoch(monkeypatch, basin_id=basin_id, run_dates=np.array([0, 1, 2], dtype=np.int32))
    result = fixed.evaluate_fixed_support_raw_space_metrics(
        run_dir="fixture", epoch=1, package_root="fixture", contract=contract,
        return_admitted_series=True,
    )
    assert result["n_basins_evaluated"] == 1
    series = result["admitted_series_by_basin"][basin_id]
    assert series.date.dtype == np.dtype("int64")
    np.testing.assert_array_equal(series.date, np.array([0, 1, 2], dtype="int64"))


# --------------------------------------------------------------------------- #
# RD1-C4 review Finding 6: immutable support-contract provenance
# --------------------------------------------------------------------------- #

def test_build_support_contract_provenance_matches_validated_contract():
    contract = _contract(1)
    provenance = fixed.build_support_contract_provenance(contract)
    assert isinstance(provenance, fixed.SupportContractProvenance)
    for field in (
        "schema_name", "schema_version", "contract_id", "checksum_sha256", "seq_length_floor",
        "period", "date_start", "date_end", "target_variable", "lead_hours",
        "screening_basin_ids_sha256", "source_gap_policy_identity", "package_manifest_sha256",
        "package_file_checksums_sha256", "package_run_provenance_sha256",
        "development_split_sha256", "spatial_holdout_split_sha256",
    ):
        assert getattr(provenance, field) == contract[field]


def test_build_support_contract_provenance_rejects_malformed_contract():
    contract = dict(_contract(1))
    del contract["checksum_sha256"]
    with pytest.raises(fixed.FixedSupportContractError):
        fixed.build_support_contract_provenance(contract)


def test_support_contract_provenance_is_frozen():
    provenance = fixed.build_support_contract_provenance(_contract(1))
    with pytest.raises(Exception):
        provenance.contract_id = "tampered"

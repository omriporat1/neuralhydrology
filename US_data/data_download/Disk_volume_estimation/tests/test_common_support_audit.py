import numpy as np
import pytest
import xarray as xr

from src.baseline.common_support_audit import (
    CommonSupportAuditError,
    basin_date_and_admitted,
    common_support_admitted_mask,
    common_support_metrics_for_run_period,
)


def _period_results(basin_id, obs, sim, *, dates=None):
    n = len(obs)
    dates = np.arange(n) if dates is None else dates
    xr_ds = xr.Dataset(
        {
            "qobs_mm_per_h_lead06_obs": ("date", np.asarray(obs, dtype=float)),
            "qobs_mm_per_h_lead06_sim": ("date", np.asarray(sim, dtype=float)),
        },
        coords={"date": dates},
    )
    return {basin_id: {"1h": {"xr": xr_ds}}}


def _write_package_basin_netcdf(package_root, basin_id, *, area_km2, lead_hours, n=300, seed=1):
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


# ---------------------------------------------------------------------------
# basin_date_and_admitted
# ---------------------------------------------------------------------------

def test_basin_date_and_admitted_returns_date_obs_sim():
    results = _period_results("01234567", obs=[1.0, np.nan, 3.0], sim=[1.1, 2.1, 3.1])
    date_values, obs, sim = basin_date_and_admitted(results, "01234567", "qobs_mm_per_h_lead06")
    assert list(date_values) == [0, 1, 2]
    assert np.isnan(obs[1])
    assert sim[0] == pytest.approx(1.1)


def test_basin_date_and_admitted_missing_basin_raises():
    results = _period_results("01234567", obs=[1.0], sim=[1.0])
    with pytest.raises(CommonSupportAuditError):
        basin_date_and_admitted(results, "not_present", "qobs_mm_per_h_lead06")


def test_basin_date_and_admitted_missing_target_vars_raises():
    xr_ds = xr.Dataset({"unrelated": ("date", np.ones(3))}, coords={"date": np.arange(3)})
    results = {"01234567": {"1h": {"xr": xr_ds}}}
    with pytest.raises(CommonSupportAuditError):
        basin_date_and_admitted(results, "01234567", "qobs_mm_per_h_lead06")


# ---------------------------------------------------------------------------
# common_support_admitted_mask
# ---------------------------------------------------------------------------

def test_common_support_admitted_mask_intersects_across_candidates():
    # seq12 admits positions {0,1,2,3}; seq24 admits {1,2,3}; seq48 admits {2,3}.
    date_values = np.arange(5)
    obs_seq12 = np.array([1.0, 1.0, 1.0, 1.0, np.nan])
    obs_seq24 = np.array([np.nan, 1.0, 1.0, 1.0, np.nan])
    obs_seq48 = np.array([np.nan, np.nan, 1.0, 1.0, np.nan])
    mask = common_support_admitted_mask(
        {
            "seq12": (date_values, obs_seq12),
            "seq24": (date_values, obs_seq24),
            "seq48": (date_values, obs_seq48),
        }
    )
    assert list(mask) == [False, False, True, True, False]


def test_common_support_admitted_mask_raises_on_date_mismatch():
    obs = np.array([1.0, 1.0, 1.0])
    with pytest.raises(CommonSupportAuditError):
        common_support_admitted_mask(
            {
                "seq12": (np.arange(3), obs),
                "seq24": (np.arange(1, 4), obs),  # shifted dates -- must not be silently paired
            }
        )


def test_common_support_admitted_mask_raises_on_empty_input():
    with pytest.raises(CommonSupportAuditError):
        common_support_admitted_mask({})


# ---------------------------------------------------------------------------
# common_support_metrics_for_run_period (end-to-end with fabricated NH pickles)
# ---------------------------------------------------------------------------

def test_common_support_metrics_restricts_to_shared_admitted_positions(tmp_path):
    package_root = tmp_path / "package"
    basin_id = "01234567"
    _write_package_basin_netcdf(package_root, basin_id, area_km2=50.0, lead_hours=6, n=300)

    # seq24 loses one extra early sample (index 0) relative to seq12's admitted set.
    obs_seq12 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    obs_seq24 = np.array([np.nan, 2.0, 3.0, 4.0, 5.0])
    sim = obs_seq12.copy()

    period_results_by_candidate = {
        "seq12": _period_results(basin_id, obs=obs_seq12, sim=sim),
        "seq24": _period_results(basin_id, obs=obs_seq24, sim=sim),
    }

    result = common_support_metrics_for_run_period(
        period_results_by_candidate=period_results_by_candidate,
        package_root=package_root,
        target_variable="qobs_mm_per_h_lead06",
        lead_hours=6,
        basin_ids=[basin_id],
    )

    assert result["n_basins_evaluated"] == 1
    assert result["n_basins_excluded"] == 0
    row = result["per_basin_common_support"][0]
    assert row["n_common_admitted"] == 4  # index 0 excluded by seq24
    assert row["n_natural_admitted__seq12"] == 5
    assert row["n_natural_admitted__seq24"] == 4

    for cid in ("seq12", "seq24"):
        per_basin = result["by_candidate"][cid]["per_basin"]
        assert len(per_basin) == 1
        # perfect sim==obs at admitted positions except where obs was
        # withheld by the common-support mask -- NSE must be 1.0 for both
        # candidates once restricted to the shared 4-sample support.
        assert per_basin[0]["n_admitted"] == 4
        assert per_basin[0]["nse"] == pytest.approx(1.0)


def test_common_support_metrics_excludes_basin_missing_from_one_candidate(tmp_path):
    package_root = tmp_path / "package"
    basin_id_present = "01111111"
    basin_id_missing = "02222222"
    _write_package_basin_netcdf(package_root, basin_id_present, area_km2=50.0, lead_hours=6, n=300)
    _write_package_basin_netcdf(package_root, basin_id_missing, area_km2=50.0, lead_hours=6, n=300)

    obs = np.array([1.0, 2.0, 3.0])
    period_results_by_candidate = {
        "seq12": {
            **_period_results(basin_id_present, obs=obs, sim=obs.copy()),
            **_period_results(basin_id_missing, obs=obs, sim=obs.copy()),
        },
        "seq24": _period_results(basin_id_present, obs=obs, sim=obs.copy()),  # missing basin_id_missing
    }

    result = common_support_metrics_for_run_period(
        period_results_by_candidate=period_results_by_candidate,
        package_root=package_root,
        target_variable="qobs_mm_per_h_lead06",
        lead_hours=6,
        basin_ids=[basin_id_present, basin_id_missing],
    )

    assert result["n_basins_evaluated"] == 1
    assert result["n_basins_excluded"] == 1
    assert result["basins_excluded"][0]["basin_id"] == basin_id_missing


def test_common_support_metrics_is_deterministic(tmp_path):
    package_root = tmp_path / "package"
    basin_id = "03333333"
    _write_package_basin_netcdf(package_root, basin_id, area_km2=75.0, lead_hours=6, n=300)

    rng = np.random.default_rng(7)
    obs = rng.uniform(0.1, 5.0, size=20)
    sim = obs + rng.normal(0, 0.1, size=20)
    period_results_by_candidate = {
        "seq12": _period_results(basin_id, obs=obs, sim=sim),
        "seq24": _period_results(basin_id, obs=obs, sim=sim),
    }

    kwargs = dict(
        period_results_by_candidate=period_results_by_candidate,
        package_root=package_root,
        target_variable="qobs_mm_per_h_lead06",
        lead_hours=6,
        basin_ids=[basin_id],
    )
    result_a = common_support_metrics_for_run_period(**kwargs)
    result_b = common_support_metrics_for_run_period(**kwargs)
    assert result_a["by_candidate"]["seq12"]["per_basin"][0]["nse"] == pytest.approx(
        result_b["by_candidate"]["seq12"]["per_basin"][0]["nse"]
    )
    assert result_a["per_basin_common_support"] == result_b["per_basin_common_support"]

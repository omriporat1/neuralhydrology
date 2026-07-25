import numpy as np
import pytest
import xarray as xr

from src.baseline.nh_raw_space_evaluation import (
    RawSpaceEvaluationError,
    aggregate_raw_space_metrics,
    convert_period_to_raw_space,
    derive_basin_area_km2,
    derive_basin_area_km2_from_netcdf,
    evaluate_basin_raw_space,
    raw_space_metrics,
)


def _synthetic_series(area_km2: float, lead_hours: int, n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    qobs_m3s = rng.uniform(1.0, 500.0, size=n)
    usable_n = n - lead_hours
    target_mm_per_h = np.full(n, np.nan)
    target_mm_per_h[:usable_n] = 3.6 * qobs_m3s[lead_hours:lead_hours + usable_n] / area_km2
    return qobs_m3s, target_mm_per_h


# ---------------------------------------------------------------------------
# derive_basin_area_km2
# ---------------------------------------------------------------------------

def test_derive_basin_area_km2_recovers_known_area():
    qobs_m3s, target_mm_per_h = _synthetic_series(area_km2=123.45, lead_hours=6, n=500)
    result = derive_basin_area_km2(qobs_m3s, target_mm_per_h, lead_hours=6, basin_id="basinA")
    assert result.area_km2 == pytest.approx(123.45, rel=1e-9)
    assert result.n_samples == 500 - 6
    assert result.consistent is True
    assert result.relative_mad < 1e-9


def test_derive_basin_area_km2_wrong_lead_hours_breaks_alignment_and_is_flagged_inconsistent():
    qobs_m3s, target_mm_per_h = _synthetic_series(area_km2=200.0, lead_hours=6, n=500)
    # Deliberately query with the WRONG lead (5 instead of the 6 used to build
    # the series) -- uncorrelated random qobs values mean the mis-aligned
    # per-sample area estimates are essentially random, so the derivation
    # must NOT silently recover a clean, consistent area.
    result = derive_basin_area_km2(qobs_m3s, target_mm_per_h, lead_hours=5, basin_id="basinA")
    assert result.consistent is False
    assert result.relative_mad > 1e-3


def test_derive_basin_area_km2_raises_on_insufficient_samples():
    qobs_m3s, target_mm_per_h = _synthetic_series(area_km2=50.0, lead_hours=6, n=500)
    with pytest.raises(RawSpaceEvaluationError):
        derive_basin_area_km2(qobs_m3s, target_mm_per_h, lead_hours=6, min_samples=1000)


def test_derive_basin_area_km2_raises_on_too_short_series_for_lead():
    qobs_m3s = np.array([1.0, 2.0, 3.0])
    target_mm_per_h = np.array([1.0, 2.0, 3.0])
    with pytest.raises(RawSpaceEvaluationError):
        derive_basin_area_km2(qobs_m3s, target_mm_per_h, lead_hours=6)


def test_derive_basin_area_km2_raises_on_mismatched_lengths():
    with pytest.raises(RawSpaceEvaluationError):
        derive_basin_area_km2(np.ones(10), np.ones(9), lead_hours=1)


def test_derive_basin_area_km2_flags_bimodal_area_as_inconsistent_without_raising():
    n = 400
    lead = 6
    rng = np.random.default_rng(7)
    qobs_m3s = rng.uniform(1.0, 100.0, size=n)
    usable_n = n - lead
    target_mm_per_h = np.full(n, np.nan)
    # Half the samples behave as if area == 10, half as if area == 1000:
    # a real single-basin series should never show this, so it must be
    # flagged inconsistent rather than averaged away.
    half = usable_n // 2
    target_mm_per_h[:half] = 3.6 * qobs_m3s[lead:lead + half] / 10.0
    target_mm_per_h[half:usable_n] = 3.6 * qobs_m3s[lead + half:lead + usable_n] / 1000.0
    result = derive_basin_area_km2(qobs_m3s, target_mm_per_h, lead_hours=lead)
    assert result.consistent is False


def test_derive_basin_area_km2_from_netcdf_matches_direct_call(tmp_path):
    qobs_m3s, target_mm_per_h = _synthetic_series(area_km2=77.0, lead_hours=6, n=300)
    nc_path = tmp_path / "basinA.nc"
    xr.Dataset(
        {
            "qobs_m3s": ("date", qobs_m3s),
            "qobs_mm_per_h_lead06": ("date", target_mm_per_h),
        },
        coords={"date": np.arange(300)},
    ).to_netcdf(nc_path)

    result = derive_basin_area_km2_from_netcdf(
        nc_path, basin_id="basinA", target_variable="qobs_mm_per_h_lead06", lead_hours=6
    )
    direct = derive_basin_area_km2(qobs_m3s, target_mm_per_h, lead_hours=6)
    assert result.area_km2 == pytest.approx(direct.area_km2, rel=1e-9)


def test_derive_basin_area_km2_from_netcdf_missing_variable_raises(tmp_path):
    nc_path = tmp_path / "basinB.nc"
    xr.Dataset({"qobs_m3s": ("date", np.ones(10))}, coords={"date": np.arange(10)}).to_netcdf(nc_path)
    with pytest.raises(RawSpaceEvaluationError):
        derive_basin_area_km2_from_netcdf(
            nc_path, basin_id="basinB", target_variable="qobs_mm_per_h_lead06", lead_hours=6, min_samples=1
        )


def test_derive_basin_area_km2_from_netcdf_missing_file_raises(tmp_path):
    with pytest.raises(RawSpaceEvaluationError):
        derive_basin_area_km2_from_netcdf(
            tmp_path / "does_not_exist.nc", basin_id="x", target_variable="y", lead_hours=6
        )


# ---------------------------------------------------------------------------
# convert_period_to_raw_space -- consistent NaN masking
# ---------------------------------------------------------------------------

def test_convert_period_to_raw_space_masks_nan_obs_and_converts_admitted_values():
    obs_mm_per_h = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
    sim_mm_per_h = np.array([1.1, 5.0, 2.2, np.nan, 3.3])
    area_km2 = 100.0

    result = convert_period_to_raw_space(obs_mm_per_h, sim_mm_per_h, area_km2)

    assert result.n_total == 5
    assert result.n_admitted == 3
    assert list(result.admitted_mask) == [True, False, True, False, True]
    # Non-admitted (obs-NaN) positions are NaN in both raw-space outputs.
    assert np.isnan(result.obs_m3s[1]) and np.isnan(result.sim_m3s[1])
    assert np.isnan(result.obs_m3s[3]) and np.isnan(result.sim_m3s[3])
    # Admitted positions are converted via the same units contract.
    expected_obs_0 = 1.0 * area_km2 / 3.6
    assert result.obs_m3s[0] == pytest.approx(expected_obs_0)
    expected_sim_0 = 1.1 * area_km2 / 3.6
    assert result.sim_m3s[0] == pytest.approx(expected_sim_0)


def test_convert_period_to_raw_space_counts_nonfinite_sim_at_admitted_positions():
    obs_mm_per_h = np.array([1.0, 2.0, 3.0])
    sim_mm_per_h = np.array([1.0, np.inf, np.nan])
    result = convert_period_to_raw_space(obs_mm_per_h, sim_mm_per_h, area_km2=50.0)
    assert result.n_admitted == 3
    assert result.n_sim_nonfinite_at_admitted == 2
    # Non-finite sim at an admitted position becomes NaN in raw space, not an
    # exception and not silently dropped from n_admitted.
    assert np.isnan(result.sim_m3s[1])
    assert np.isnan(result.sim_m3s[2])
    assert np.isfinite(result.sim_m3s[0])


def test_convert_period_to_raw_space_rejects_shape_mismatch():
    with pytest.raises(RawSpaceEvaluationError):
        convert_period_to_raw_space(np.ones(5), np.ones(4), area_km2=10.0)


# ---------------------------------------------------------------------------
# raw_space_metrics
# ---------------------------------------------------------------------------

def test_raw_space_metrics_perfect_prediction():
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = raw_space_metrics(obs, obs.copy())
    assert metrics["n_samples"] == 5
    assert metrics["nse"] == pytest.approx(1.0)
    assert metrics["kge"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["bias"] == pytest.approx(0.0)
    assert metrics["pbias"] == pytest.approx(0.0)
    assert metrics["pearson_r"] == pytest.approx(1.0)


def test_raw_space_metrics_insufficient_samples_returns_nan_not_error():
    metrics = raw_space_metrics(np.array([1.0]), np.array([1.0]))
    assert metrics["n_samples"] == 1
    assert np.isnan(metrics["nse"])
    assert np.isnan(metrics["kge"])


def test_raw_space_metrics_zero_variance_obs_gives_nan_nse_and_correlation():
    obs = np.array([5.0, 5.0, 5.0, 5.0])
    sim = np.array([4.0, 5.0, 6.0, 5.0])
    metrics = raw_space_metrics(obs, sim)
    assert np.isnan(metrics["nse"])
    assert np.isnan(metrics["pearson_r"])
    # rmse/mae/bias are still well-defined even when NSE/correlation are not.
    assert np.isfinite(metrics["rmse"])


def test_raw_space_metrics_filters_nonfinite_pairs():
    obs = np.array([1.0, 2.0, np.nan, 4.0])
    sim = np.array([1.0, np.inf, 3.0, 4.0])
    metrics = raw_space_metrics(obs, sim)
    assert metrics["n_samples"] == 2


# ---------------------------------------------------------------------------
# evaluate_basin_raw_space + aggregate_raw_space_metrics
# ---------------------------------------------------------------------------

def test_evaluate_basin_raw_space_end_to_end():
    obs_mm_per_h = np.array([1.0, 2.0, np.nan, 4.0])
    sim_mm_per_h = np.array([1.0, 2.0, 5.0, 4.0])
    result = evaluate_basin_raw_space(
        basin_id="basinZ", obs_mm_per_h=obs_mm_per_h, sim_mm_per_h=sim_mm_per_h, area_km2=10.0
    )
    assert result["basin_id"] == "basinZ"
    assert result["n_total"] == 4
    assert result["n_admitted"] == 3
    assert result["nse"] == pytest.approx(1.0)


def test_aggregate_raw_space_metrics_basic():
    rows = [
        {"n_admitted": 10, "n_sim_nonfinite_at_admitted": 0, "nse": 0.8, "kge": 0.7, "rmse": 1.0, "mae": 0.5,
         "pearson_r": 0.9, "bias": 0.1, "pbias": 1.0},
        {"n_admitted": 20, "n_sim_nonfinite_at_admitted": 1, "nse": 0.6, "kge": 0.5, "rmse": 2.0, "mae": 1.0,
         "pearson_r": 0.7, "bias": -0.2, "pbias": -1.0},
        {"n_admitted": 5, "n_sim_nonfinite_at_admitted": 0, "nse": float("nan"), "kge": float("nan"),
         "rmse": float("nan"), "mae": float("nan"), "pearson_r": float("nan"), "bias": float("nan"),
         "pbias": float("nan")},
    ]
    aggregate = aggregate_raw_space_metrics(rows)
    assert aggregate["n_basins"] == 3
    assert aggregate["n_admitted_total"] == 35
    assert aggregate["n_sim_nonfinite_at_admitted_total"] == 1
    assert aggregate["metrics"]["nse"]["n_finite_basins"] == 2
    assert aggregate["metrics"]["nse"]["median"] == pytest.approx(0.7)


def test_aggregate_raw_space_metrics_all_nan_metric_reports_zero_finite():
    rows = [
        {"n_admitted": 1, "n_sim_nonfinite_at_admitted": 0, "nse": float("nan"), "kge": float("nan"),
         "rmse": float("nan"), "mae": float("nan"), "pearson_r": float("nan"), "bias": float("nan"),
         "pbias": float("nan")},
    ]
    aggregate = aggregate_raw_space_metrics(rows)
    assert aggregate["metrics"]["nse"]["n_finite_basins"] == 0
    assert np.isnan(aggregate["metrics"]["nse"]["median"])

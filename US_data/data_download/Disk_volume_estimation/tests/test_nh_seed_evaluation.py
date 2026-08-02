import pickle

import numpy as np
import pytest
import xarray as xr

from src.baseline.nh_config_generation import (
    HOLDOUT_MARKER_FILENAME,
    HoldoutBundleTrainingRejected,
    raise_if_holdout_bundle,
)
from src.baseline.nh_seed_evaluation import (
    EVALUATION_ONLY_MARKER_FILENAME,
    NHSeedEvaluationError,
    basin_netcdf_path,
    load_period_results,
    prepare_development_population_eval_run_dir,
    prepare_external_scaler_eval_run_dir,
    raise_if_evaluation_only_bundle,
    raw_space_metrics_for_run_period,
    require_holdout_bundle,
    weight_stem,
)


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


def _write_validation_pickle(run_dir, epoch, basin_results):
    period_dir = run_dir / "validation" / weight_stem(epoch)
    period_dir.mkdir(parents=True, exist_ok=True)
    with open(period_dir / "validation_results.p", "wb") as fh:
        pickle.dump(basin_results, fh)


# ---------------------------------------------------------------------------
# weight_stem / basin_netcdf_path / load_period_results
# ---------------------------------------------------------------------------

def test_weight_stem_zero_pads_epoch():
    assert weight_stem(5) == "model_epoch005"
    assert weight_stem(123) == "model_epoch123"


def test_basin_netcdf_path_uses_time_series_subdir(tmp_path):
    path = basin_netcdf_path(tmp_path, "01234567")
    assert path == tmp_path / "time_series" / "01234567.nc"


def test_load_period_results_missing_pickle_raises(tmp_path):
    with pytest.raises(NHSeedEvaluationError):
        load_period_results(tmp_path, "validation", 5)


def test_load_period_results_round_trips(tmp_path):
    payload = {"01234567": {"1h": {"xr": "placeholder"}}}
    _write_validation_pickle(tmp_path, 5, payload)
    loaded = load_period_results(tmp_path, "validation", 5)
    assert loaded == payload


# ---------------------------------------------------------------------------
# require_holdout_bundle / raise_if_holdout_bundle (guard pair)
# ---------------------------------------------------------------------------

def test_require_holdout_bundle_raises_without_marker(tmp_path):
    with pytest.raises(NHSeedEvaluationError):
        require_holdout_bundle(tmp_path)


def test_require_holdout_bundle_passes_with_marker(tmp_path):
    (tmp_path / HOLDOUT_MARKER_FILENAME).write_text("TEST ONLY\n", encoding="utf-8")
    require_holdout_bundle(tmp_path)  # must not raise


def test_raise_if_holdout_bundle_passes_for_development_bundle(tmp_path):
    raise_if_holdout_bundle(tmp_path)  # no marker -> must not raise


def test_raise_if_holdout_bundle_blocks_training_launcher_on_holdout_bundle(tmp_path):
    (tmp_path / HOLDOUT_MARKER_FILENAME).write_text("TEST ONLY\n", encoding="utf-8")
    with pytest.raises(HoldoutBundleTrainingRejected):
        raise_if_holdout_bundle(tmp_path)


# ---------------------------------------------------------------------------
# raise_if_evaluation_only_bundle (guard for train/continue entrypoints)
# ---------------------------------------------------------------------------

def test_raise_if_evaluation_only_bundle_passes_for_ordinary_directory(tmp_path):
    raise_if_evaluation_only_bundle(tmp_path)  # no marker -> must not raise


def test_raise_if_evaluation_only_bundle_blocks_training_launcher_on_eval_only_dir(tmp_path):
    (tmp_path / EVALUATION_ONLY_MARKER_FILENAME).write_text("EVAL ONLY\n", encoding="utf-8")
    with pytest.raises(NHSeedEvaluationError):
        raise_if_evaluation_only_bundle(tmp_path)


# ---------------------------------------------------------------------------
# prepare_external_scaler_eval_run_dir
# ---------------------------------------------------------------------------

def _make_holdout_bundle(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text("holdout: true\n", encoding="utf-8")
    (path / HOLDOUT_MARKER_FILENAME).write_text("TEST ONLY\n", encoding="utf-8")


def _make_development_run_dir(path, epoch, *, scaler_bytes=b"scaler-bytes-v1", checkpoint_bytes=b"checkpoint-bytes-v1"):
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{weight_stem(epoch)}.pt").write_bytes(checkpoint_bytes)
    (path / "train_data").mkdir(exist_ok=True)
    (path / "train_data" / "train_data_scaler.yml").write_bytes(scaler_bytes)


def test_prepare_external_scaler_eval_run_dir_reuses_scaler_byte_identical(tmp_path):
    holdout_dir = tmp_path / "spatial_holdout"
    dev_dir = tmp_path / "development_run"
    out_dir = tmp_path / "holdout_eval_epoch005"
    _make_holdout_bundle(holdout_dir)
    _make_development_run_dir(dev_dir, epoch=5)

    manifest = prepare_external_scaler_eval_run_dir(
        development_run_dir=dev_dir, epoch=5, holdout_generated_dir=holdout_dir, out_run_dir=out_dir
    )

    assert manifest["scaler_reused_unchanged"] is True
    scaler_dst = out_dir / "train_data" / "train_data_scaler.yml"
    checkpoint_dst = out_dir / f"{weight_stem(5)}.pt"
    config_dst = out_dir / "config.yml"
    assert scaler_dst.read_bytes() == (dev_dir / "train_data" / "train_data_scaler.yml").read_bytes()
    assert checkpoint_dst.read_bytes() == (dev_dir / f"{weight_stem(5)}.pt").read_bytes()
    assert config_dst.read_bytes() == (holdout_dir / "config.yaml").read_bytes()
    assert (out_dir / "EXTERNAL_SCALER_EVAL_MANIFEST.json").is_file()


def test_prepare_external_scaler_eval_run_dir_requires_source_scaler(tmp_path):
    holdout_dir = tmp_path / "spatial_holdout"
    dev_dir = tmp_path / "development_run"
    _make_holdout_bundle(holdout_dir)
    dev_dir.mkdir(parents=True)
    (dev_dir / f"{weight_stem(5)}.pt").write_bytes(b"checkpoint-only-no-scaler")
    # deliberately no train_data/train_data_scaler.yml

    with pytest.raises(NHSeedEvaluationError):
        prepare_external_scaler_eval_run_dir(
            development_run_dir=dev_dir, epoch=5, holdout_generated_dir=holdout_dir, out_run_dir=tmp_path / "out"
        )


def test_prepare_external_scaler_eval_run_dir_requires_source_checkpoint(tmp_path):
    holdout_dir = tmp_path / "spatial_holdout"
    dev_dir = tmp_path / "development_run"
    _make_holdout_bundle(holdout_dir)
    dev_dir.mkdir(parents=True)
    (dev_dir / "train_data").mkdir()
    (dev_dir / "train_data" / "train_data_scaler.yml").write_bytes(b"scaler-only-no-checkpoint")
    # deliberately no model_epoch005.pt

    with pytest.raises(NHSeedEvaluationError):
        prepare_external_scaler_eval_run_dir(
            development_run_dir=dev_dir, epoch=5, holdout_generated_dir=holdout_dir, out_run_dir=tmp_path / "out"
        )


def test_prepare_external_scaler_eval_run_dir_rejects_non_holdout_source(tmp_path):
    non_holdout_dir = tmp_path / "development_generated"
    non_holdout_dir.mkdir(parents=True)
    (non_holdout_dir / "config.yaml").write_text("dev: true\n", encoding="utf-8")
    # deliberately no TEST_ONLY_DO_NOT_TRAIN.txt marker
    dev_dir = tmp_path / "development_run"
    _make_development_run_dir(dev_dir, epoch=5)

    with pytest.raises(NHSeedEvaluationError):
        prepare_external_scaler_eval_run_dir(
            development_run_dir=dev_dir,
            epoch=5,
            holdout_generated_dir=non_holdout_dir,
            out_run_dir=tmp_path / "out",
        )


def test_prepare_external_scaler_eval_run_dir_rejects_out_dir_equal_to_development_dir(tmp_path):
    holdout_dir = tmp_path / "spatial_holdout"
    dev_dir = tmp_path / "development_run"
    _make_holdout_bundle(holdout_dir)
    _make_development_run_dir(dev_dir, epoch=5)

    # force=True must NOT bypass this check -- a collision here would rmtree
    # the original, already-trained development run directory.
    with pytest.raises(NHSeedEvaluationError):
        prepare_external_scaler_eval_run_dir(
            development_run_dir=dev_dir, epoch=5, holdout_generated_dir=holdout_dir,
            out_run_dir=dev_dir, force=True,
        )
    assert (dev_dir / f"{weight_stem(5)}.pt").is_file()  # original untouched


def test_prepare_external_scaler_eval_run_dir_rejects_out_dir_nested_in_development_dir(tmp_path):
    holdout_dir = tmp_path / "spatial_holdout"
    dev_dir = tmp_path / "development_run"
    _make_holdout_bundle(holdout_dir)
    _make_development_run_dir(dev_dir, epoch=5)

    with pytest.raises(NHSeedEvaluationError):
        prepare_external_scaler_eval_run_dir(
            development_run_dir=dev_dir, epoch=5, holdout_generated_dir=holdout_dir,
            out_run_dir=dev_dir / "nested_out", force=True,
        )


def test_prepare_external_scaler_eval_run_dir_refuses_existing_dir_without_force(tmp_path):
    holdout_dir = tmp_path / "spatial_holdout"
    dev_dir = tmp_path / "development_run"
    out_dir = tmp_path / "out"
    _make_holdout_bundle(holdout_dir)
    _make_development_run_dir(dev_dir, epoch=5)
    out_dir.mkdir()
    (out_dir / "stale_marker.txt").write_text("stale\n", encoding="utf-8")

    with pytest.raises(NHSeedEvaluationError):
        prepare_external_scaler_eval_run_dir(
            development_run_dir=dev_dir, epoch=5, holdout_generated_dir=holdout_dir, out_run_dir=out_dir
        )

    # force=True overwrites cleanly instead of erroring.
    manifest = prepare_external_scaler_eval_run_dir(
        development_run_dir=dev_dir, epoch=5, holdout_generated_dir=holdout_dir, out_run_dir=out_dir, force=True
    )
    assert manifest["scaler_reused_unchanged"] is True
    assert not (out_dir / "stale_marker.txt").exists()


# ---------------------------------------------------------------------------
# prepare_development_population_eval_run_dir
# ---------------------------------------------------------------------------

def _make_eval_bundle(path, *, holdout=False):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text("atlas24_eval: true\n", encoding="utf-8")
    if holdout:
        (path / HOLDOUT_MARKER_FILENAME).write_text("TEST ONLY\n", encoding="utf-8")


def test_prepare_development_population_eval_run_dir_reuses_scaler_byte_identical(tmp_path):
    eval_dir = tmp_path / "atlas24_eval_generated"
    dev_dir = tmp_path / "development_run"
    out_dir = tmp_path / "atlas24_eval_run"
    _make_eval_bundle(eval_dir)
    _make_development_run_dir(dev_dir, epoch=6)

    manifest = prepare_development_population_eval_run_dir(
        development_run_dir=dev_dir, epoch=6, eval_generated_dir=eval_dir, out_run_dir=out_dir
    )

    assert manifest["scaler_reused_unchanged"] is True
    scaler_dst = out_dir / "train_data" / "train_data_scaler.yml"
    checkpoint_dst = out_dir / f"{weight_stem(6)}.pt"
    config_dst = out_dir / "config.yml"
    assert scaler_dst.read_bytes() == (dev_dir / "train_data" / "train_data_scaler.yml").read_bytes()
    assert checkpoint_dst.read_bytes() == (dev_dir / f"{weight_stem(6)}.pt").read_bytes()
    assert config_dst.read_bytes() == (eval_dir / "config.yaml").read_bytes()
    assert (out_dir / "DEVELOPMENT_POPULATION_EVAL_MANIFEST.json").is_file()
    marker_text = (out_dir / EVALUATION_ONLY_MARKER_FILENAME).read_text(encoding="utf-8")
    assert "Do NOT run a trainer" in marker_text


def test_prepare_development_population_eval_run_dir_rejects_holdout_bundle(tmp_path):
    eval_dir = tmp_path / "holdout_generated"
    dev_dir = tmp_path / "development_run"
    _make_eval_bundle(eval_dir, holdout=True)
    _make_development_run_dir(dev_dir, epoch=6)

    with pytest.raises(HoldoutBundleTrainingRejected):
        prepare_development_population_eval_run_dir(
            development_run_dir=dev_dir, epoch=6, eval_generated_dir=eval_dir, out_run_dir=tmp_path / "out"
        )


def test_prepare_development_population_eval_run_dir_requires_source_scaler(tmp_path):
    eval_dir = tmp_path / "atlas24_eval_generated"
    dev_dir = tmp_path / "development_run"
    _make_eval_bundle(eval_dir)
    dev_dir.mkdir(parents=True)
    (dev_dir / f"{weight_stem(6)}.pt").write_bytes(b"checkpoint-only-no-scaler")
    # deliberately no train_data/train_data_scaler.yml

    with pytest.raises(NHSeedEvaluationError):
        prepare_development_population_eval_run_dir(
            development_run_dir=dev_dir, epoch=6, eval_generated_dir=eval_dir, out_run_dir=tmp_path / "out"
        )


def test_prepare_development_population_eval_run_dir_requires_source_checkpoint(tmp_path):
    eval_dir = tmp_path / "atlas24_eval_generated"
    dev_dir = tmp_path / "development_run"
    _make_eval_bundle(eval_dir)
    dev_dir.mkdir(parents=True)
    (dev_dir / "train_data").mkdir()
    (dev_dir / "train_data" / "train_data_scaler.yml").write_bytes(b"scaler-only-no-checkpoint")
    # deliberately no model_epoch006.pt

    with pytest.raises(NHSeedEvaluationError):
        prepare_development_population_eval_run_dir(
            development_run_dir=dev_dir, epoch=6, eval_generated_dir=eval_dir, out_run_dir=tmp_path / "out"
        )


def test_prepare_development_population_eval_run_dir_rejects_out_dir_equal_to_development_dir(tmp_path):
    eval_dir = tmp_path / "atlas24_eval_generated"
    dev_dir = tmp_path / "development_run"
    _make_eval_bundle(eval_dir)
    _make_development_run_dir(dev_dir, epoch=6)

    # force=True must NOT bypass this check -- a collision here would rmtree
    # the original, already-trained development run directory (Part L's
    # atlas24 derivative always passes force=True for idempotent reruns).
    with pytest.raises(NHSeedEvaluationError):
        prepare_development_population_eval_run_dir(
            development_run_dir=dev_dir, epoch=6, eval_generated_dir=eval_dir,
            out_run_dir=dev_dir, force=True,
        )
    assert (dev_dir / f"{weight_stem(6)}.pt").is_file()  # original untouched


def test_prepare_development_population_eval_run_dir_rejects_out_dir_nested_in_development_dir(tmp_path):
    eval_dir = tmp_path / "atlas24_eval_generated"
    dev_dir = tmp_path / "development_run"
    _make_eval_bundle(eval_dir)
    _make_development_run_dir(dev_dir, epoch=6)

    with pytest.raises(NHSeedEvaluationError):
        prepare_development_population_eval_run_dir(
            development_run_dir=dev_dir, epoch=6, eval_generated_dir=eval_dir,
            out_run_dir=dev_dir / "nested_out", force=True,
        )


def test_prepare_development_population_eval_run_dir_refuses_existing_dir_without_force(tmp_path):
    eval_dir = tmp_path / "atlas24_eval_generated"
    dev_dir = tmp_path / "development_run"
    out_dir = tmp_path / "out"
    _make_eval_bundle(eval_dir)
    _make_development_run_dir(dev_dir, epoch=6)
    out_dir.mkdir()
    (out_dir / "stale_marker.txt").write_text("stale\n", encoding="utf-8")

    with pytest.raises(NHSeedEvaluationError):
        prepare_development_population_eval_run_dir(
            development_run_dir=dev_dir, epoch=6, eval_generated_dir=eval_dir, out_run_dir=out_dir
        )

    # force=True overwrites cleanly instead of erroring.
    manifest = prepare_development_population_eval_run_dir(
        development_run_dir=dev_dir, epoch=6, eval_generated_dir=eval_dir, out_run_dir=out_dir, force=True
    )
    assert manifest["scaler_reused_unchanged"] is True
    assert not (out_dir / "stale_marker.txt").exists()


# ---------------------------------------------------------------------------
# raw_space_metrics_for_run_period (end-to-end with fabricated NH pickle)
# ---------------------------------------------------------------------------

def test_raw_space_metrics_for_run_period_end_to_end(tmp_path):
    run_dir = tmp_path / "run"
    package_root = tmp_path / "package"
    basin_id = "01234567"
    _write_package_basin_netcdf(package_root, basin_id, area_km2=50.0, lead_hours=6)

    obs = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    sim = obs.copy()
    xr_ds = xr.Dataset(
        {
            "qobs_mm_per_h_lead06_obs": ("date", obs),
            "qobs_mm_per_h_lead06_sim": ("date", sim),
        },
        coords={"date": np.arange(5)},
    )
    _write_validation_pickle(run_dir, 5, {basin_id: {"1h": {"xr": xr_ds}}})

    result = raw_space_metrics_for_run_period(
        run_dir=run_dir,
        period="validation",
        epoch=5,
        package_root=package_root,
        target_variable="qobs_mm_per_h_lead06",
        lead_hours=6,
    )

    assert result["n_basins_requested"] == 1
    assert result["n_basins_evaluated"] == 1
    assert result["n_basins_area_excluded"] == 0
    assert result["per_basin"][0]["basin_id"] == basin_id
    assert result["per_basin"][0]["nse"] == pytest.approx(1.0)
    assert result["aggregate"]["n_basins"] == 1


def test_raw_space_metrics_for_run_period_excludes_basin_with_insufficient_area_samples(tmp_path):
    run_dir = tmp_path / "run"
    package_root = tmp_path / "package"
    basin_id = "09999999"
    # Too short a package series to derive a reliable area (default min_samples=100).
    _write_package_basin_netcdf(package_root, basin_id, area_km2=50.0, lead_hours=6, n=20)

    obs = np.array([1.0, 2.0, 3.0])
    xr_ds = xr.Dataset(
        {
            "qobs_mm_per_h_lead06_obs": ("date", obs),
            "qobs_mm_per_h_lead06_sim": ("date", obs.copy()),
        },
        coords={"date": np.arange(3)},
    )
    _write_validation_pickle(run_dir, 5, {basin_id: {"1h": {"xr": xr_ds}}})

    result = raw_space_metrics_for_run_period(
        run_dir=run_dir,
        period="validation",
        epoch=5,
        package_root=package_root,
        target_variable="qobs_mm_per_h_lead06",
        lead_hours=6,
    )

    assert result["n_basins_evaluated"] == 0
    assert result["n_basins_area_excluded"] == 1
    assert result["area_derivation_excluded"][0]["basin_id"] == basin_id


def test_raw_space_metrics_for_run_period_excludes_basin_missing_target_data_vars(tmp_path):
    run_dir = tmp_path / "run"
    package_root = tmp_path / "package"
    basin_id = "01111111"
    _write_package_basin_netcdf(package_root, basin_id, area_km2=50.0, lead_hours=6)

    xr_ds = xr.Dataset({"unrelated_var": ("date", np.ones(5))}, coords={"date": np.arange(5)})
    _write_validation_pickle(run_dir, 5, {basin_id: {"1h": {"xr": xr_ds}}})

    result = raw_space_metrics_for_run_period(
        run_dir=run_dir,
        period="validation",
        epoch=5,
        package_root=package_root,
        target_variable="qobs_mm_per_h_lead06",
        lead_hours=6,
    )

    assert result["n_basins_evaluated"] == 0
    assert result["n_basins_area_excluded"] == 1
    assert "missing xr result" in result["area_derivation_excluded"][0]["reason"]


def test_raw_space_metrics_for_run_period_missing_requested_basin_raises(tmp_path):
    run_dir = tmp_path / "run"
    package_root = tmp_path / "package"
    basin_id = "02222222"
    _write_package_basin_netcdf(package_root, basin_id, area_km2=50.0, lead_hours=6)
    xr_ds = xr.Dataset(
        {
            "qobs_mm_per_h_lead06_obs": ("date", np.array([1.0, 2.0])),
            "qobs_mm_per_h_lead06_sim": ("date", np.array([1.0, 2.0])),
        },
        coords={"date": np.arange(2)},
    )
    _write_validation_pickle(run_dir, 5, {basin_id: {"1h": {"xr": xr_ds}}})

    with pytest.raises(NHSeedEvaluationError):
        raw_space_metrics_for_run_period(
            run_dir=run_dir,
            period="validation",
            epoch=5,
            package_root=package_root,
            target_variable="qobs_mm_per_h_lead06",
            lead_hours=6,
            basin_ids=["some_other_basin_not_in_results"],
        )

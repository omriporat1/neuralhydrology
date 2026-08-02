"""Tests for src/baseline/hydrograph_rendering.py.

Synthetic fixtures only -- no NeuralHydrology/torch import, no h2o/Moriah
dependency, no real emb128x64_seedA evidence. Mirrors the fixture/testing
conventions already used by tests/test_hydrograph_atlas_selection.py and
tests/test_hydrograph_atlas_events.py.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.baseline import nh_raw_space_evaluation as raw_space_mod
from src.baseline.hydrograph_atlas_events import select_atlas_events
from src.baseline import hydrograph_rendering as rendering_mod
from src.baseline.hydrograph_rendering import (
    COMPACT_SELECTION_DIMENSIONS,
    HydrographRenderingError,
    BasinSeries,
    extract_basin_xr,
    load_atlas_selection_csv,
    load_basin_series,
    observed_series_for_events,
    render_stage1_hydrographs,
    select_compact_basins,
    sha256_of,
)
from src.baseline.units import discharge_m3s_to_runoff_mm_per_h

TARGET_VARIABLE = "qobs_mm_per_h_lead06"
LEAD_HOURS = 6
AREA_KM2 = 500.0
MIN_AREA_SAMPLES = 50
REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------

def _write_package_netcdf(package_root, basin_id, *, area_km2=AREA_KM2, n=200, seed=0):
    rng = np.random.default_rng(seed)
    qobs_m3s = rng.uniform(10.0, 100.0, size=n)
    target_mm_per_h = np.full(n, np.nan)
    usable_n = n - LEAD_HOURS
    target_mm_per_h[:usable_n] = discharge_m3s_to_runoff_mm_per_h(qobs_m3s[LEAD_HOURS:LEAD_HOURS + usable_n], area_km2)
    ds = xr.Dataset(
        {
            "qobs_m3s": ("date", qobs_m3s),
            TARGET_VARIABLE: ("date", target_mm_per_h),
        },
        coords={"date": np.arange(n)},
    )
    ts_dir = package_root / "time_series"
    ts_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(ts_dir / f"{basin_id}.nc")


def _basin_result_entry(*, n=300, seed=0, freq="1H", nan_positions=(5, 42)):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    obs = rng.uniform(0.01, 5.0, size=n)
    sim = obs + rng.normal(0.0, 0.2, size=n)
    for i in nan_positions:
        obs[i] = np.nan
    ds = xr.Dataset(
        {
            f"{TARGET_VARIABLE}_obs": ("date", obs),
            f"{TARGET_VARIABLE}_sim": ("date", sim),
        },
        coords={"date": dates},
    )
    return {freq: {"xr": ds, "NSE": 0.5}}


def _atlas_rows():
    return [
        {"gauge_id": "10000001", "canonical_basin_role": "development_train", "huc02": "01",
         "macro_region": "northeast", "geo_side": "east", "area_class": "low", "hydro_class": "wet",
         "nse": 0.8, "skill_stratum": "strong", "selection_reason": "x"},
        {"gauge_id": "10000002", "canonical_basin_role": "development_train", "huc02": "01",
         "macro_region": "northeast", "geo_side": "east", "area_class": "middle", "hydro_class": "wet",
         "nse": 0.3, "skill_stratum": "typical", "selection_reason": "x"},
        {"gauge_id": "10000003", "canonical_basin_role": "development_train", "huc02": "14",
         "macro_region": "colorado", "geo_side": "west", "area_class": "high", "hydro_class": "dry",
         "nse": 0.1, "skill_stratum": "weak", "selection_reason": "x"},
        {"gauge_id": "10000004", "canonical_basin_role": "development_train", "huc02": "14",
         "macro_region": "colorado", "geo_side": "west", "area_class": "low", "hydro_class": "dry",
         "nse": -0.3, "skill_stratum": "severe_failure_lower_tail", "selection_reason": "x"},
        {"gauge_id": "10000005", "canonical_basin_role": "development_train", "huc02": "01",
         "macro_region": "northeast", "geo_side": "east", "area_class": "high", "hydro_class": "wet",
         "nse": 0.6, "skill_stratum": "strong", "selection_reason": "x"},
        {"gauge_id": "10000006", "canonical_basin_role": "development_train", "huc02": "14",
         "macro_region": "colorado", "geo_side": "west", "area_class": "middle", "hydro_class": "dry",
         "nse": 0.2, "skill_stratum": "typical", "selection_reason": "x"},
    ]


def _build_fixture(tmp_path, basin_ids, *, n_result=300, n_package=200, freq="1H"):
    package_root = tmp_path / "package"
    results = {}
    for i, basin_id in enumerate(basin_ids):
        _write_package_netcdf(package_root, basin_id, n=n_package, seed=i)
        results[basin_id] = _basin_result_entry(n=n_result, seed=100 + i, freq=freq)

    result_pickle_path = tmp_path / "validation_results.p"
    with open(result_pickle_path, "wb") as fh:
        pickle.dump(results, fh)

    atlas_csv_path = tmp_path / "hydrograph_atlas_basin_selection.csv"
    pd.DataFrame(_atlas_rows()).to_csv(atlas_csv_path, index=False)

    return results, result_pickle_path, package_root, atlas_csv_path


_ALL_BASIN_IDS = [row["gauge_id"] for row in _atlas_rows()]


# ---------------------------------------------------------------------------
# load_atlas_selection_csv
# ---------------------------------------------------------------------------

def test_load_atlas_selection_csv_basic(tmp_path):
    _, _, _, atlas_csv_path = _build_fixture(tmp_path, _ALL_BASIN_IDS[:2])
    df = load_atlas_selection_csv(atlas_csv_path)
    assert len(df) == 6
    assert set(COMPACT_SELECTION_DIMENSIONS).issubset(df.columns)


def test_load_atlas_selection_csv_missing_file(tmp_path):
    with pytest.raises(HydrographRenderingError):
        load_atlas_selection_csv(tmp_path / "does_not_exist.csv")


def test_load_atlas_selection_csv_missing_column(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"gauge_id": ["1"], "nse": [0.1]}).to_csv(path, index=False)
    with pytest.raises(HydrographRenderingError):
        load_atlas_selection_csv(path)


def test_load_atlas_selection_csv_rejects_duplicate_gauge_id(tmp_path):
    rows = _atlas_rows()
    rows.append(dict(rows[0]))
    path = tmp_path / "dup.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(HydrographRenderingError):
        load_atlas_selection_csv(path)


# ---------------------------------------------------------------------------
# select_compact_basins -- items 1-4
# ---------------------------------------------------------------------------

def test_select_compact_basins_deterministic_repeat(tmp_path):
    atlas_df = pd.DataFrame(_atlas_rows())
    sel1, manifest1 = select_compact_basins(atlas_df, target_count=4)
    sel2, manifest2 = select_compact_basins(atlas_df, target_count=4)
    pd.testing.assert_frame_equal(sel1, sel2)
    assert manifest1["selection_order"] == manifest2["selection_order"]


def test_select_compact_basins_no_duplicates(tmp_path):
    atlas_df = pd.DataFrame(_atlas_rows())
    sel, _ = select_compact_basins(atlas_df, target_count=5)
    assert not sel["gauge_id"].duplicated().any()


def test_select_compact_basins_subset_of_atlas_input(tmp_path):
    atlas_df = pd.DataFrame(_atlas_rows())
    sel, _ = select_compact_basins(atlas_df, target_count=4)
    assert set(sel["gauge_id"]).issubset(set(atlas_df["gauge_id"]))


def test_select_compact_basins_stable_under_row_reordering(tmp_path):
    atlas_df = pd.DataFrame(_atlas_rows())
    shuffled = atlas_df.sample(frac=1.0, random_state=123).reset_index(drop=True)
    sel1, _ = select_compact_basins(atlas_df, target_count=4)
    sel2, _ = select_compact_basins(shuffled, target_count=4)
    assert list(sel1["gauge_id"]) == list(sel2["gauge_id"])


def test_select_compact_basins_maximizes_dimension_coverage(tmp_path):
    atlas_df = pd.DataFrame(_atlas_rows())
    sel, manifest = select_compact_basins(atlas_df, target_count=4)
    # 4 picks are enough to cover both geo_side values and >1 skill_stratum/area_class value.
    assert set(sel["geo_side"]) == {"east", "west"}
    assert manifest["dimension_omitted"]["name"] == "flashy_vs_smoother"


def test_select_compact_basins_rejects_out_of_range_target_count(tmp_path):
    atlas_df = pd.DataFrame(_atlas_rows())
    with pytest.raises(HydrographRenderingError):
        select_compact_basins(atlas_df, target_count=0)
    with pytest.raises(HydrographRenderingError):
        select_compact_basins(atlas_df, target_count=999)


def test_select_compact_basins_explicit_tie_break_ascending_gauge_id():
    # Identical stratification on both rows -> equal marginal coverage gain;
    # the tie must resolve to the ascending gauge_id, not incidental
    # set/dict/row order.
    rows = [
        {"gauge_id": "20000002", "skill_stratum": "strong", "area_class": "low", "geo_side": "east", "nse": 0.5},
        {"gauge_id": "20000001", "skill_stratum": "strong", "area_class": "low", "geo_side": "east", "nse": 0.5},
    ]
    atlas_df = pd.DataFrame(rows)
    sel, manifest = select_compact_basins(atlas_df, target_count=1)
    assert list(sel["gauge_id"]) == ["20000001"]
    assert manifest["selection_order"][0]["gauge_id"] == "20000001"


def test_select_compact_basins_full_coverage_reporting():
    atlas_df = pd.DataFrame(_atlas_rows())
    _, manifest_full = select_compact_basins(atlas_df, target_count=6)
    assert manifest_full["full_coverage_achieved"] is True
    for dim in COMPACT_SELECTION_DIMENSIONS:
        assert manifest_full["dimension_coverage"][dim]["fully_covered"] is True

    _, manifest_partial = select_compact_basins(atlas_df, target_count=1)
    assert manifest_partial["full_coverage_achieved"] is False


# ---------------------------------------------------------------------------
# extract_basin_xr / load_basin_series -- items 6,7,8,9,10,11
# ---------------------------------------------------------------------------

def test_raw_space_functions_are_reused_not_reimplemented():
    import src.baseline.hydrograph_rendering as mod
    assert mod.convert_period_to_raw_space is raw_space_mod.convert_period_to_raw_space
    assert mod.raw_space_metrics is raw_space_mod.raw_space_metrics
    assert mod.derive_basin_area_km2_from_netcdf is raw_space_mod.derive_basin_area_km2_from_netcdf


def test_extract_basin_xr_correctly_reads_synthetic_pickle(tmp_path):
    results, _, _, _ = _build_fixture(tmp_path, _ALL_BASIN_IDS[:1])
    basin_id = _ALL_BASIN_IDS[0]
    xr_ds = extract_basin_xr(results, basin_id, TARGET_VARIABLE)
    assert f"{TARGET_VARIABLE}_obs" in xr_ds.data_vars
    assert f"{TARGET_VARIABLE}_sim" in xr_ds.data_vars


def test_extract_basin_xr_missing_basin_rejected(tmp_path):
    results, _, _, _ = _build_fixture(tmp_path, _ALL_BASIN_IDS[:1])
    with pytest.raises(HydrographRenderingError, match="not found"):
        extract_basin_xr(results, "99999999", TARGET_VARIABLE)


def test_extract_basin_xr_missing_target_variable_rejected(tmp_path):
    results, _, _, _ = _build_fixture(tmp_path, _ALL_BASIN_IDS[:1])
    basin_id = _ALL_BASIN_IDS[0]
    with pytest.raises(HydrographRenderingError, match="target variable"):
        extract_basin_xr(results, basin_id, "not_a_real_target")


def test_extract_basin_xr_multi_frequency_requires_explicit_freq(tmp_path):
    results, _, _, _ = _build_fixture(tmp_path, _ALL_BASIN_IDS[:1])
    basin_id = _ALL_BASIN_IDS[0]
    entry = results[basin_id]
    entry["3H"] = entry["1H"]  # a second frequency key -> ambiguous without freq=
    with pytest.raises(HydrographRenderingError, match="frequencies"):
        extract_basin_xr(results, basin_id, TARGET_VARIABLE)
    # explicit freq resolves deterministically to the requested frequency's Dataset
    xr_ds = extract_basin_xr(results, basin_id, TARGET_VARIABLE, freq="3H")
    assert xr_ds is entry["3H"]["xr"]


@pytest.mark.parametrize("malformed_results", [
    {"10000001": "not_a_dict"},
    {"10000001": {}},
    {"10000001": {"1H": {"no_xr_key": 1}}},
    "not_a_dict_at_all",
])
def test_extract_basin_xr_rejects_malformed_pickle_structure(malformed_results):
    with pytest.raises(HydrographRenderingError):
        extract_basin_xr(malformed_results, "10000001", TARGET_VARIABLE)


def test_load_basin_series_nan_handling_and_admitted_mask(tmp_path):
    results, _, package_root, _ = _build_fixture(tmp_path, _ALL_BASIN_IDS[:1])
    basin_id = _ALL_BASIN_IDS[0]
    bs = load_basin_series(
        results=results, basin_id=basin_id, target_variable=TARGET_VARIABLE,
        package_root=package_root, lead_hours=LEAD_HOURS, min_area_samples=MIN_AREA_SAMPLES,
    )
    assert isinstance(bs, BasinSeries)
    assert bs.n_admitted == bs.admitted_mask.sum()
    assert bs.n_admitted < bs.n_total  # the 2 injected NaNs were excluded
    assert np.isnan(bs.obs_m3s[~bs.admitted_mask]).all()
    assert np.isfinite(bs.metrics["nse"])


# ---------------------------------------------------------------------------
# Validation-only period guard -- item 12
# ---------------------------------------------------------------------------

def test_render_stage1_hydrographs_rejects_non_validation_period(tmp_path):
    _, result_pickle_path, package_root, atlas_csv_path = _build_fixture(tmp_path, _ALL_BASIN_IDS)
    with pytest.raises(HydrographRenderingError, match="not permitted"):
        render_stage1_hydrographs(
            result_pickle=result_pickle_path, period="test", epoch=15,
            package_root=package_root, target_variable=TARGET_VARIABLE, lead_hours=LEAD_HOURS,
            atlas_csv=atlas_csv_path, out_dir=tmp_path / "out", mode="compact",
        )


# ---------------------------------------------------------------------------
# Event selection is observed-only -- item 5
# ---------------------------------------------------------------------------

def test_event_windows_depend_only_on_observations(tmp_path):
    results, _, package_root, _ = _build_fixture(tmp_path, _ALL_BASIN_IDS[:1])
    basin_id = _ALL_BASIN_IDS[0]
    bs = load_basin_series(
        results=results, basin_id=basin_id, target_variable=TARGET_VARIABLE,
        package_root=package_root, lead_hours=LEAD_HOURS, min_area_samples=MIN_AREA_SAMPLES,
    )
    obs_only_series = observed_series_for_events(bs)
    events_a = select_atlas_events(obs_only_series, min_separation_hours=72, pre_hours=24, post_hours=48)

    # A BasinSeries that differs only in predicted discharge must yield identical events.
    tampered = BasinSeries(
        basin_id=bs.basin_id, dates=bs.dates, obs_m3s=bs.obs_m3s,
        sim_m3s=bs.sim_m3s * 0.0 + 999.0, admitted_mask=bs.admitted_mask,
        area_km2=bs.area_km2, n_admitted=bs.n_admitted, n_total=bs.n_total, metrics=bs.metrics,
    )
    events_b = select_atlas_events(
        observed_series_for_events(tampered), min_separation_hours=72, pre_hours=24, post_hours=48
    )
    assert events_a.keys() == events_b.keys()
    for key in events_a:
        assert events_a[key] == events_b[key]


# ---------------------------------------------------------------------------
# Full orchestration -- items 13,14,15,16,17,18
# ---------------------------------------------------------------------------

def test_render_stage1_hydrographs_compact_dry_run(tmp_path):
    _, result_pickle_path, package_root, atlas_csv_path = _build_fixture(tmp_path, _ALL_BASIN_IDS)
    out_dir = tmp_path / "out"
    summary = render_stage1_hydrographs(
        result_pickle=result_pickle_path, period="validation", epoch=15,
        package_root=package_root, target_variable=TARGET_VARIABLE, lead_hours=LEAD_HOURS,
        atlas_csv=atlas_csv_path, out_dir=out_dir, mode="compact", compact_target_count=4,
        min_area_samples=MIN_AREA_SAMPLES, write_outputs=False,
    )
    assert summary["dry_run"] is True
    assert len(summary["compact_basin_ids"]) == 4
    assert not out_dir.exists()  # nothing written outside caller-supplied out_dir, and nothing at all in dry-run mode


def test_render_stage1_hydrographs_compact_writes_outputs(tmp_path):
    _, result_pickle_path, package_root, atlas_csv_path = _build_fixture(tmp_path, _ALL_BASIN_IDS)
    out_dir = tmp_path / "out"
    summary = render_stage1_hydrographs(
        result_pickle=result_pickle_path, period="validation", epoch=15,
        package_root=package_root, target_variable=TARGET_VARIABLE, lead_hours=LEAD_HOURS,
        atlas_csv=atlas_csv_path, out_dir=out_dir, mode="compact", compact_target_count=4,
        min_area_samples=MIN_AREA_SAMPLES,
    )
    assert (out_dir / "compact_panel.png").is_file()
    assert (out_dir / "compact_basin_membership.json").is_file()
    assert (out_dir / "per_basin_metrics.csv").is_file()
    assert (out_dir / "event_window_table.csv").is_file()
    assert (out_dir / "rendering_manifest.json").is_file()
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "atlas").exists() is False  # compact-only mode renders no atlas dir
    assert summary["n_basins_rendered"] == 4


def test_render_stage1_hydrographs_full_atlas_dry_run_small_synthetic_atlas(tmp_path):
    _, result_pickle_path, package_root, atlas_csv_path = _build_fixture(tmp_path, _ALL_BASIN_IDS)
    out_dir = tmp_path / "out"
    summary = render_stage1_hydrographs(
        result_pickle=result_pickle_path, period="validation", epoch=15,
        package_root=package_root, target_variable=TARGET_VARIABLE, lead_hours=LEAD_HOURS,
        atlas_csv=atlas_csv_path, out_dir=out_dir, mode="full",
        min_area_samples=MIN_AREA_SAMPLES,
    )
    assert (out_dir / "atlas").is_dir()
    for basin_id in _ALL_BASIN_IDS:
        assert (out_dir / "atlas" / f"{basin_id}.png").is_file()
    assert summary["n_basins_rendered"] == len(_ALL_BASIN_IDS)


def test_render_stage1_hydrographs_manifest_content(tmp_path):
    _, result_pickle_path, package_root, atlas_csv_path = _build_fixture(tmp_path, _ALL_BASIN_IDS)
    out_dir = tmp_path / "out"
    render_stage1_hydrographs(
        result_pickle=result_pickle_path, period="validation", epoch=15,
        package_root=package_root, target_variable=TARGET_VARIABLE, lead_hours=LEAD_HOURS,
        atlas_csv=atlas_csv_path, out_dir=out_dir, mode="compact", compact_target_count=4,
        min_area_samples=MIN_AREA_SAMPLES,
    )
    manifest = json.loads((out_dir / "rendering_manifest.json").read_text(encoding="utf-8"))
    for key in [
        "result_pickle_path", "result_pickle_sha256", "epoch", "period", "target_variable",
        "package_root", "atlas_csv_path", "atlas_csv_sha256", "compact_selection",
        "event_selection_basis", "raw_space_conversion_source", "output_files", "generated_at_utc",
    ]:
        assert key in manifest
    assert manifest["event_selection_basis"] == "observed_discharge_only"
    assert manifest["period"] == "validation"


def test_render_stage1_hydrographs_output_checksums_match(tmp_path):
    _, result_pickle_path, package_root, atlas_csv_path = _build_fixture(tmp_path, _ALL_BASIN_IDS)
    out_dir = tmp_path / "out"
    render_stage1_hydrographs(
        result_pickle=result_pickle_path, period="validation", epoch=15,
        package_root=package_root, target_variable=TARGET_VARIABLE, lead_hours=LEAD_HOURS,
        atlas_csv=atlas_csv_path, out_dir=out_dir, mode="compact", compact_target_count=4,
        min_area_samples=MIN_AREA_SAMPLES,
    )
    manifest = json.loads((out_dir / "rendering_manifest.json").read_text(encoding="utf-8"))
    for name, record in manifest["output_files"].items():
        assert sha256_of(out_dir / name) == record["sha256"]


def test_render_stage1_hydrographs_outputs_confined_to_out_dir(tmp_path):
    _, result_pickle_path, package_root, atlas_csv_path = _build_fixture(tmp_path, _ALL_BASIN_IDS)
    out_dir = tmp_path / "nested" / "out"
    summary = render_stage1_hydrographs(
        result_pickle=result_pickle_path, period="validation", epoch=15,
        package_root=package_root, target_variable=TARGET_VARIABLE, lead_hours=LEAD_HOURS,
        atlas_csv=atlas_csv_path, out_dir=out_dir, mode="compact", compact_target_count=4,
        min_area_samples=MIN_AREA_SAMPLES,
    )
    for rel_name, path_str in summary["output_files"].items():
        assert str(out_dir) in path_str


def test_render_stage1_hydrographs_repeated_run_equivalent_outputs(tmp_path):
    _, result_pickle_path, package_root, atlas_csv_path = _build_fixture(tmp_path, _ALL_BASIN_IDS)
    out_dir_a = tmp_path / "out_a"
    out_dir_b = tmp_path / "out_b"
    render_stage1_hydrographs(
        result_pickle=result_pickle_path, period="validation", epoch=15,
        package_root=package_root, target_variable=TARGET_VARIABLE, lead_hours=LEAD_HOURS,
        atlas_csv=atlas_csv_path, out_dir=out_dir_a, mode="compact", compact_target_count=4,
        min_area_samples=MIN_AREA_SAMPLES,
    )
    render_stage1_hydrographs(
        result_pickle=result_pickle_path, period="validation", epoch=15,
        package_root=package_root, target_variable=TARGET_VARIABLE, lead_hours=LEAD_HOURS,
        atlas_csv=atlas_csv_path, out_dir=out_dir_b, mode="compact", compact_target_count=4,
        min_area_samples=MIN_AREA_SAMPLES,
    )
    metrics_a = pd.read_csv(out_dir_a / "per_basin_metrics.csv")
    metrics_b = pd.read_csv(out_dir_b / "per_basin_metrics.csv")
    pd.testing.assert_frame_equal(metrics_a, metrics_b)

    manifest_a = json.loads((out_dir_a / "rendering_manifest.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((out_dir_b / "rendering_manifest.json").read_text(encoding="utf-8"))
    for key in manifest_a:
        if key in ("generated_at_utc", "output_files"):
            continue
        assert manifest_a[key] == manifest_b[key]


# ---------------------------------------------------------------------------
# Explicit-pickle vs. run-dir conflict; output-directory safety; figure
# closure; partial-failure manifest safety; selected-checkpoint epoch
# ---------------------------------------------------------------------------

def test_render_stage1_hydrographs_rejects_conflicting_pickle_and_rundir(tmp_path):
    _, result_pickle_path, package_root, atlas_csv_path = _build_fixture(tmp_path, _ALL_BASIN_IDS[:1])
    with pytest.raises(HydrographRenderingError, match="both supplied"):
        render_stage1_hydrographs(
            run_dir=tmp_path / "some_run_dir", result_pickle=result_pickle_path,
            period="validation", epoch=6, package_root=package_root,
            target_variable=TARGET_VARIABLE, lead_hours=LEAD_HOURS,
            atlas_csv=atlas_csv_path, out_dir=tmp_path / "out", mode="compact",
            write_outputs=False,
        )


def test_render_stage1_hydrographs_rejects_out_dir_inside_tracked_src(tmp_path):
    _, result_pickle_path, package_root, atlas_csv_path = _build_fixture(tmp_path, _ALL_BASIN_IDS[:1])
    with pytest.raises(HydrographRenderingError, match="tracked directory"):
        render_stage1_hydrographs(
            result_pickle=result_pickle_path, period="validation", epoch=6,
            package_root=package_root, target_variable=TARGET_VARIABLE, lead_hours=LEAD_HOURS,
            atlas_csv=atlas_csv_path,
            out_dir=REPO_ROOT / "src" / "baseline" / "tmp_should_never_be_created",
            mode="compact", compact_target_count=1, repo_root=REPO_ROOT,
        )
    assert not (REPO_ROOT / "src" / "baseline" / "tmp_should_never_be_created").exists()


def test_render_stage1_hydrographs_nonempty_out_dir_requires_force(tmp_path):
    _, result_pickle_path, package_root, atlas_csv_path = _build_fixture(tmp_path, _ALL_BASIN_IDS)
    out_dir = tmp_path / "out"
    render_stage1_hydrographs(
        result_pickle=result_pickle_path, period="validation", epoch=6,
        package_root=package_root, target_variable=TARGET_VARIABLE, lead_hours=LEAD_HOURS,
        atlas_csv=atlas_csv_path, out_dir=out_dir, mode="compact", compact_target_count=4,
        min_area_samples=MIN_AREA_SAMPLES,
    )
    with pytest.raises(HydrographRenderingError, match="non-empty"):
        render_stage1_hydrographs(
            result_pickle=result_pickle_path, period="validation", epoch=6,
            package_root=package_root, target_variable=TARGET_VARIABLE, lead_hours=LEAD_HOURS,
            atlas_csv=atlas_csv_path, out_dir=out_dir, mode="compact", compact_target_count=4,
            min_area_samples=MIN_AREA_SAMPLES,
        )
    # force=True makes the overwrite explicit rather than silent
    render_stage1_hydrographs(
        result_pickle=result_pickle_path, period="validation", epoch=6,
        package_root=package_root, target_variable=TARGET_VARIABLE, lead_hours=LEAD_HOURS,
        atlas_csv=atlas_csv_path, out_dir=out_dir, mode="compact", compact_target_count=4,
        min_area_samples=MIN_AREA_SAMPLES, force=True,
    )
    assert (out_dir / "rendering_manifest.json").is_file()


def test_render_stage1_hydrographs_closes_all_figures(tmp_path):
    _, result_pickle_path, package_root, atlas_csv_path = _build_fixture(tmp_path, _ALL_BASIN_IDS)
    out_dir = tmp_path / "out"
    assert plt.get_fignums() == []
    render_stage1_hydrographs(
        result_pickle=result_pickle_path, period="validation", epoch=6,
        package_root=package_root, target_variable=TARGET_VARIABLE, lead_hours=LEAD_HOURS,
        atlas_csv=atlas_csv_path, out_dir=out_dir, mode="both", compact_target_count=4,
        min_area_samples=MIN_AREA_SAMPLES,
    )
    assert plt.get_fignums() == []  # every figure (compact + 6 atlas basins) was closed after saving


def test_render_stage1_hydrographs_partial_rendering_failure_leaves_no_manifest(tmp_path, monkeypatch):
    _, result_pickle_path, package_root, atlas_csv_path = _build_fixture(tmp_path, _ALL_BASIN_IDS)
    out_dir = tmp_path / "out"

    original_render_basin_panel = rendering_mod.render_basin_panel
    call_count = {"n": 0}

    def _flaky_render_basin_panel(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 3:
            raise RuntimeError("synthetic rendering failure")
        return original_render_basin_panel(*args, **kwargs)

    monkeypatch.setattr(rendering_mod, "render_basin_panel", _flaky_render_basin_panel)
    with pytest.raises(RuntimeError, match="synthetic rendering failure"):
        rendering_mod.render_stage1_hydrographs(
            result_pickle=result_pickle_path, period="validation", epoch=6,
            package_root=package_root, target_variable=TARGET_VARIABLE, lead_hours=LEAD_HOURS,
            atlas_csv=atlas_csv_path, out_dir=out_dir, mode="full",
            min_area_samples=MIN_AREA_SAMPLES,
        )
    # A crash partway through atlas rendering must never leave a manifest/summary
    # behind that would misrepresent the run as a complete, successful atlas.
    assert not (out_dir / "rendering_manifest.json").exists()
    assert not (out_dir / "summary.json").exists()


def test_render_stage1_hydrographs_preserves_explicit_selected_epoch(tmp_path):
    # epoch=6 here stands in for a "selected checkpoint" that is explicitly
    # different from an arbitrary "stop epoch" (e.g. 15 used elsewhere in
    # this file) -- the renderer must store exactly the caller-supplied
    # value with no internal notion of which epoch is "the" epoch.
    _, result_pickle_path, package_root, atlas_csv_path = _build_fixture(tmp_path, _ALL_BASIN_IDS)
    out_dir = tmp_path / "out"
    summary = render_stage1_hydrographs(
        result_pickle=result_pickle_path, period="validation", epoch=6,
        package_root=package_root, target_variable=TARGET_VARIABLE, lead_hours=LEAD_HOURS,
        atlas_csv=atlas_csv_path, out_dir=out_dir, mode="compact", compact_target_count=4,
        min_area_samples=MIN_AREA_SAMPLES,
    )
    assert summary["epoch"] == 6
    manifest = json.loads((out_dir / "rendering_manifest.json").read_text(encoding="utf-8"))
    assert manifest["epoch"] == 6

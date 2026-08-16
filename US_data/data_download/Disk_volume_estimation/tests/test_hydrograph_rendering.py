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
from src.baseline.hydrograph_atlas_events import EventWindow
from src.baseline.hydrograph_rendering import (
    COMPACT_SELECTION_DIMENSIONS,
    MRMS_QPE_VARIABLE,
    HydrographRenderingError,
    BasinSeries,
    ScaleSpec,
    compute_compact_event_metrics,
    compute_target_valid_dates,
    derive_comparison_scale,
    DisplayWindow,
    derive_display_window,
    extract_basin_xr,
    format_basin_area_title,
    load_atlas_selection_csv,
    load_basin_series,
    load_mrms_series,
    observed_series_for_events,
    render_basin_panel,
    render_compact_panel,
    render_multi_candidate_basin_panel,
    render_interpretation_template,
    render_stage1_compact_comparison_package,
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

def _write_package_netcdf(package_root, basin_id, *, area_km2=AREA_KM2, n=200, seed=0, with_mrms=True,
                           start="2023-01-01"):
    rng = np.random.default_rng(seed)
    qobs_m3s = rng.uniform(10.0, 100.0, size=n)
    target_mm_per_h = np.full(n, np.nan)
    usable_n = n - LEAD_HOURS
    target_mm_per_h[:usable_n] = discharge_m3s_to_runoff_mm_per_h(qobs_m3s[LEAD_HOURS:LEAD_HOURS + usable_n], area_km2)
    data_vars = {
        "qobs_m3s": ("date", qobs_m3s),
        TARGET_VARIABLE: ("date", target_mm_per_h),
    }
    if with_mrms:
        data_vars[MRMS_QPE_VARIABLE] = ("date", rng.uniform(0.0, 5.0, size=n))
    ds = xr.Dataset(
        data_vars,
        coords={"date": pd.date_range(start, periods=n, freq="h")},
    )
    ts_dir = package_root / "time_series"
    ts_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(ts_dir / f"{basin_id}.nc")


def _basin_result_entry(*, n=300, seed=0, freq="1H", nan_positions=(5, 42), start="2023-01-01"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="h")
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


def _build_fixture(tmp_path, basin_ids, *, n_result=300, n_package=200, freq="1H", with_mrms=True,
                    start="2023-01-01"):
    package_root = tmp_path / "package"
    results = {}
    for i, basin_id in enumerate(basin_ids):
        _write_package_netcdf(package_root, basin_id, n=n_package, seed=i, with_mrms=with_mrms, start=start)
        results[basin_id] = _basin_result_entry(n=n_result, seed=100 + i, freq=freq, start=start)

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
        basin_id=bs.basin_id, dates=bs.dates, issue_dates=bs.issue_dates, obs_m3s=bs.obs_m3s,
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



# ---------------------------------------------------------------------------
# L.3d -- basin area in title (missing/invalid area handled explicitly)
# ---------------------------------------------------------------------------

def test_format_basin_area_title_formats_km2():
    title = format_basin_area_title("01074520", 1234.0)
    assert title == "01074520 — Area: 1,234 km²"
    assert "km" in title


@pytest.mark.parametrize("bad_area", [None, float("nan"), 0.0, -5.0])
def test_format_basin_area_title_rejects_missing_invalid_area(bad_area):
    with pytest.raises(HydrographRenderingError):
        format_basin_area_title("01074520", bad_area)


# ---------------------------------------------------------------------------
# L.3d -- time alignment: target-valid dates, single shift, no double shift
# ---------------------------------------------------------------------------

def test_compute_target_valid_dates_shifts_by_lead_hours():
    issue = pd.date_range("2023-01-01", periods=3, freq="h")
    shifted = compute_target_valid_dates(issue, 6)
    assert list(shifted) == [t + pd.Timedelta(hours=6) for t in issue]


def test_compute_target_valid_dates_rejects_negative_lead_hours():
    issue = pd.date_range("2023-01-01", periods=3, freq="h")
    with pytest.raises(HydrographRenderingError):
        compute_target_valid_dates(issue, -1)


def test_load_basin_series_dates_are_target_valid_time_shifted_exactly_once(tmp_path):
    results, _, package_root, _ = _build_fixture(tmp_path, _ALL_BASIN_IDS[:1])
    basin_id = _ALL_BASIN_IDS[0]
    bs = load_basin_series(
        results=results, basin_id=basin_id, target_variable=TARGET_VARIABLE,
        package_root=package_root, lead_hours=LEAD_HOURS, min_area_samples=MIN_AREA_SAMPLES,
    )
    expected = compute_target_valid_dates(bs.issue_dates, LEAD_HOURS)
    pd.testing.assert_index_equal(bs.dates, expected)
    # Applying the shift a second time on top of bs.dates would NOT reproduce
    # bs.issue_dates -- proving the shift baked into bs.dates was applied once.
    assert bs.issue_dates[0] == bs.dates[0] - pd.Timedelta(hours=LEAD_HOURS)
    assert bs.issue_dates[0] != bs.dates[0]


# ---------------------------------------------------------------------------
# L.3d -- MRMS precipitation: unshifted physical valid time, explicit units
# ---------------------------------------------------------------------------

def test_load_mrms_series_dates_are_unshifted_physical_valid_time(tmp_path):
    _, _, package_root, _ = _build_fixture(tmp_path, _ALL_BASIN_IDS[:1])
    basin_id = _ALL_BASIN_IDS[0]
    expected_dates = pd.date_range("2023-01-01", periods=200, freq="h")
    mrms = load_mrms_series(package_root, basin_id)
    assert mrms.name == MRMS_QPE_VARIABLE
    assert (pd.DatetimeIndex(mrms.index) == expected_dates).all()


def test_load_mrms_series_missing_variable_raises_explicitly(tmp_path):
    _, _, package_root, _ = _build_fixture(tmp_path, _ALL_BASIN_IDS[:1], with_mrms=False)
    basin_id = _ALL_BASIN_IDS[0]
    with pytest.raises(HydrographRenderingError, match=MRMS_QPE_VARIABLE):
        load_mrms_series(package_root, basin_id)


def test_precip_axis_inverted_zero_at_top_of_secondary_axis():
    fig, ax = plt.subplots()
    dates = pd.date_range("2023-01-01", periods=5, freq="h")
    series = pd.Series([0.0, 2.0, 5.0, 1.0, 0.0], index=dates, name=MRMS_QPE_VARIABLE)
    rendering_mod._add_precip_axis(ax, series, None)
    ax2 = fig.axes[-1]
    bottom, top = ax2.get_ylim()
    assert top == pytest.approx(0.0)
    assert bottom > top  # zero at the top; precipitation increases downward
    assert "mm" in ax2.get_ylabel()
    plt.close(fig)


# ---------------------------------------------------------------------------
# L.3d -- cross-series simultaneous trace + event-window timestamps
# (coverage gaps identified by the read-only L.3d timestamp-semantics audit;
#  see reports/stage1_validation_optimization_foundation_v001/
#  part_l3d_timestamp_audit_v001/.../timestamp_test_inventory.md sections 2/5/6)
# ---------------------------------------------------------------------------

def test_cross_series_known_lead6_timestamp_trace_all_three_series(tmp_path):
    """Traces one known lead-6 example through all three series at once via
    the real public loaders (load_basin_series, load_mrms_series) rather than
    only restating compute_target_valid_dates arithmetic in isolation:
    NH issue time 2024-01-01 00:00 -> obs AND sim target-valid time
    2024-01-01 06:00 (one shared date axis) -> MRMS QPE stays at its own
    unshifted package date 2024-01-01 00:00.
    """
    results, _, package_root, _ = _build_fixture(tmp_path, _ALL_BASIN_IDS[:1], start="2024-01-01")
    basin_id = _ALL_BASIN_IDS[0]
    bs = load_basin_series(
        results=results, basin_id=basin_id, target_variable=TARGET_VARIABLE,
        package_root=package_root, lead_hours=LEAD_HOURS, min_area_samples=MIN_AREA_SAMPLES,
    )
    mrms = load_mrms_series(package_root, basin_id)

    known_issue_time = pd.Timestamp("2024-01-01 00:00")
    known_target_valid_time = pd.Timestamp("2024-01-01 06:00")
    assert known_target_valid_time == known_issue_time + pd.Timedelta(hours=LEAD_HOURS)

    assert bs.issue_dates[0] == known_issue_time
    assert bs.dates[0] == known_target_valid_time

    # obs and sim are read off the identical shared date axis: both are
    # target-valid, and both are the *same* axis, not two independently
    # shifted copies that merely happen to agree.
    obs_dates = pd.Series(bs.obs_m3s, index=bs.dates).index
    sim_dates = pd.Series(bs.sim_m3s, index=bs.dates).index
    assert (obs_dates == sim_dates).all()
    assert obs_dates[0] == known_target_valid_time
    assert sim_dates[0] == known_target_valid_time

    # Their date is issue time plus lead_hours, elementwise (not just at index 0).
    assert (bs.dates == bs.issue_dates + pd.Timedelta(hours=LEAD_HOURS)).all()

    # Rainfall retains its own original package date: it is never re-indexed
    # onto bs.dates and never shifted by lead_hours.
    assert mrms.index[0] == known_issue_time
    assert mrms.index[0] != known_target_valid_time

    # No second six-hour shift anywhere: bs.dates is exactly one lead_hours
    # step away from bs.issue_dates -- not two (12h) and not zero (0h).
    assert bs.dates[0] - bs.issue_dates[0] == pd.Timedelta(hours=LEAD_HOURS)


def test_event_window_timestamps_are_target_valid_not_issue_time(tmp_path):
    """Proves select_atlas_events (fed via observed_series_for_events) reports
    peak_time/window_start/window_end in target-valid time, because it is
    given BasinSeries.dates directly, not the original NH issue-time
    coordinate. Does not change the event-selection algorithm itself; only
    checks its output timestamps against the known lead_hours shift.
    """
    n_result = 200
    basin_id = "01074520"
    package_root = tmp_path / "package"
    _write_package_netcdf(package_root, basin_id, n=n_result, seed=0, start="2024-01-01")

    rng = np.random.default_rng(7)
    obs = np.clip(1.0 + rng.normal(0.0, 0.05, size=n_result), 0.1, None)
    spike_idx = 100
    obs[spike_idx] = 500.0  # single unambiguous peak, far from series edges
    sim = obs.copy()
    issue_dates = pd.date_range("2024-01-01", periods=n_result, freq="h")
    ds_result = xr.Dataset(
        {
            f"{TARGET_VARIABLE}_obs": ("date", obs),
            f"{TARGET_VARIABLE}_sim": ("date", sim),
        },
        coords={"date": issue_dates},
    )
    results = {basin_id: {"1H": {"xr": ds_result, "NSE": 0.5}}}

    bs = load_basin_series(
        results=results, basin_id=basin_id, target_variable=TARGET_VARIABLE,
        package_root=package_root, lead_hours=LEAD_HOURS, min_area_samples=MIN_AREA_SAMPLES,
    )
    events = select_atlas_events(
        observed_series_for_events(bs), min_separation_hours=72, pre_hours=24, post_hours=48
    )
    # The spike is the global-max observation, so exactly one selected event
    # (whichever magnitude_class the greedy quantile-nearest assignment gives
    # it -- not asserted here, since that assignment is the pre-existing
    # event-selection algorithm and is out of scope for this test) is centered
    # on it. Identify that event by its peak_time rather than assuming a class
    # label, then check only its timestamps.
    matches = [window for window in events.values() if window.peak_time == bs.dates[spike_idx]]
    assert len(matches) == 1
    window = matches[0]

    known_issue_peak_time = issue_dates[spike_idx]
    known_target_valid_peak_time = known_issue_peak_time + pd.Timedelta(hours=LEAD_HOURS)

    assert window.peak_time == bs.dates[spike_idx]
    assert window.peak_time == known_target_valid_peak_time
    assert window.peak_time != known_issue_peak_time
    assert window.window_start == known_target_valid_peak_time - pd.Timedelta(hours=24)
    assert window.window_end == known_target_valid_peak_time + pd.Timedelta(hours=48)
    assert window.window_clipped is False


# ---------------------------------------------------------------------------
# L.3d -- shared comparison scale (does not clip either candidate/observed)
# ---------------------------------------------------------------------------

def _synthetic_basin_series(*, obs, sim, dates=None):
    if dates is None:
        dates = pd.date_range("2023-01-01", periods=len(obs), freq="h")
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.ones(len(obs), dtype=bool)
    return BasinSeries(
        basin_id="X", dates=pd.DatetimeIndex(dates),
        issue_dates=pd.DatetimeIndex(dates) - pd.Timedelta(hours=LEAD_HOURS),
        obs_m3s=obs, sim_m3s=sim, admitted_mask=mask,
        area_km2=100.0, n_admitted=len(obs), n_total=len(obs), metrics={},
    )


def test_derive_comparison_scale_shared_across_candidates_does_not_clip():
    dates = pd.date_range("2023-01-01", periods=10, freq="h")
    bs_a = _synthetic_basin_series(obs=np.linspace(10, 20, 10), sim=np.linspace(5, 15, 10), dates=dates)
    bs_b = _synthetic_basin_series(obs=np.linspace(10, 20, 10), sim=np.linspace(30, 40, 10), dates=dates)
    precip = pd.Series(np.linspace(0, 3, 10), index=dates, name=MRMS_QPE_VARIABLE)

    scale = derive_comparison_scale([bs_a, bs_b], precip_series_list=[precip])
    assert scale.discharge_min <= 5.0  # bs_a's sim minimum is not clipped
    assert scale.discharge_max >= 40.0  # bs_b's sim maximum is not clipped
    assert scale.precip_max >= 3.0
    assert scale.x_min == dates.min()
    assert scale.x_max == dates.max()

    # Deriving again from the same inputs reproduces an identical spec --
    # the same ScaleSpec instance can be handed to every candidate's render
    # call to guarantee identical x/discharge-y/precip-y limits.
    scale_again = derive_comparison_scale([bs_a, bs_b], precip_series_list=[precip])
    assert scale == scale_again


def test_derive_comparison_scale_restricts_to_window():
    dates = pd.date_range("2023-01-01", periods=10, freq="h")
    bs = _synthetic_basin_series(
        obs=[1, 1, 1, 1, 100, 1, 1, 1, 1, 1], sim=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dates=dates,
    )
    window = EventWindow(
        magnitude_class="largest", peak_time=dates[1], peak_value=1.0,
        window_start=dates[0], window_end=dates[2], window_clipped=False, n_missing_in_window=0,
    )
    scale = derive_comparison_scale([bs], window=window)
    assert scale.discharge_max < 100.0  # the out-of-window spike must not affect the window-scoped scale
    assert scale.x_min == dates[0]
    assert scale.x_max == dates[2]


# ---------------------------------------------------------------------------
# Sequence-Length-A hydrograph sanity check -- derive_display_window
# ---------------------------------------------------------------------------

def _make_event_window(*, peak_time, pre_hours=24, post_hours=48):
    return EventWindow(
        magnitude_class="high", peak_time=peak_time, peak_value=42.0,
        window_start=peak_time - pd.Timedelta(hours=pre_hours),
        window_end=peak_time + pd.Timedelta(hours=post_hours),
        window_clipped=False, n_missing_in_window=0,
    )


def test_derive_display_window_extends_start_preserves_peak_and_end():
    peak_time = pd.Timestamp("2023-03-10 12:00")
    window = _make_event_window(peak_time=peak_time)  # default 24h pre / 48h post
    display = derive_display_window(window, min_pre_hours=72)

    assert display.peak_time == window.peak_time
    assert display.window_end == window.window_end
    assert display.window_start == peak_time - pd.Timedelta(hours=72)
    assert display.actual_pre_hours == pytest.approx(72.0)
    assert display.requested_pre_hours == 72
    assert display.clipped_to_series_start is False
    assert display.magnitude_class == window.magnitude_class


def test_derive_display_window_does_not_narrow_an_already_wider_window():
    peak_time = pd.Timestamp("2023-03-10 12:00")
    # Original event window already has 96h of pre-peak context -- wider
    # than the requested 72h minimum, so it must be left alone, not shrunk.
    window = _make_event_window(peak_time=peak_time, pre_hours=96)
    display = derive_display_window(window, min_pre_hours=72)

    assert display.window_start == window.window_start
    assert display.actual_pre_hours == pytest.approx(96.0)


def test_derive_display_window_does_not_mutate_input_window():
    peak_time = pd.Timestamp("2023-03-10 12:00")
    window = _make_event_window(peak_time=peak_time)
    original_start = window.window_start
    derive_display_window(window, min_pre_hours=72)
    assert window.window_start == original_start  # frozen EventWindow untouched


def test_derive_display_window_clips_to_series_start():
    peak_time = pd.Timestamp("2023-03-10 12:00")
    window = _make_event_window(peak_time=peak_time)
    series_start = peak_time - pd.Timedelta(hours=30)  # less than the requested 72h
    display = derive_display_window(window, min_pre_hours=72, series_start=series_start)

    assert display.window_start == series_start
    assert display.clipped_to_series_start is True
    assert display.actual_pre_hours == pytest.approx(30.0)


def test_derive_display_window_rejects_negative_min_pre_hours():
    window = _make_event_window(peak_time=pd.Timestamp("2023-03-10 12:00"))
    with pytest.raises(HydrographRenderingError):
        derive_display_window(window, min_pre_hours=-1)


def test_derive_display_window_is_drop_in_compatible_with_derive_comparison_scale():
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    peak_time = dates[80]
    obs = np.full(100, 1.0)
    obs[80] = 100.0
    bs = _synthetic_basin_series(obs=obs, sim=np.full(100, 1.0), dates=dates)
    window = _make_event_window(peak_time=peak_time)
    display = derive_display_window(window, min_pre_hours=72)

    scale = derive_comparison_scale([bs], window=display)
    assert scale.x_min == display.window_start
    assert scale.x_max == display.window_end


# ---------------------------------------------------------------------------
# L.3d -- compact per-window metrics table (reuses raw_space_metrics)
# ---------------------------------------------------------------------------

def test_compute_compact_event_metrics_columns_and_deterministic_ordering(tmp_path):
    basin_ids = _ALL_BASIN_IDS[:2]
    results, _, package_root, _ = _build_fixture(tmp_path, basin_ids)
    basin_series_by_id = {}
    events_by_basin = {}
    for basin_id in basin_ids:
        bs = load_basin_series(
            results=results, basin_id=basin_id, target_variable=TARGET_VARIABLE,
            package_root=package_root, lead_hours=LEAD_HOURS, min_area_samples=MIN_AREA_SAMPLES,
        )
        basin_series_by_id[basin_id] = bs
        events_by_basin[basin_id] = select_atlas_events(observed_series_for_events(bs))

    table = compute_compact_event_metrics(basin_series_by_id, events_by_basin, candidate_id="cand_a")
    expected_columns = [
        "candidate_id", "basin_id", "area_km2", "window_id", "window_start", "window_end", "n_admitted",
        "nse", "kge", "rmse", "mae", "bias", "pbias",
        "obs_peak_value", "obs_peak_time", "sim_peak_value", "sim_peak_time",
        "peak_magnitude_error", "peak_timing_error_hours",
    ]
    assert list(table.columns) == expected_columns
    assert (table["candidate_id"] == "cand_a").all()
    basin_window_pairs = list(zip(table["basin_id"], table["window_id"]))
    assert basin_window_pairs == sorted(basin_window_pairs)


def test_compute_compact_event_metrics_peak_magnitude_and_timing_error_signs():
    dates = pd.date_range("2023-01-01", periods=6, freq="h")
    obs = np.array([1.0, 1.0, 10.0, 1.0, 1.0, 1.0])
    sim = np.array([1.0, 1.0, 1.0, 1.0, 20.0, 1.0])
    bs = _synthetic_basin_series(obs=obs, sim=sim, dates=dates)
    window = EventWindow(
        magnitude_class="largest", peak_time=dates[2], peak_value=10.0,
        window_start=dates[0], window_end=dates[-1], window_clipped=False, n_missing_in_window=0,
    )
    table = compute_compact_event_metrics({"X": bs}, {"X": {"largest": window}}, candidate_id="c")
    row = table.iloc[0]
    assert row["obs_peak_value"] == 10.0 and row["obs_peak_time"] == dates[2]
    assert row["sim_peak_value"] == 20.0 and row["sim_peak_time"] == dates[4]
    assert row["peak_magnitude_error"] == pytest.approx(10.0)
    assert row["peak_timing_error_hours"] == pytest.approx(2.0)  # sim peak is 2h after obs peak


# ---------------------------------------------------------------------------
# L.3d -- interpretation template (structured prompts, no auto-conclusions)
# ---------------------------------------------------------------------------

def test_render_interpretation_template_distinguishes_obs_pred_and_rainfall_timing(tmp_path):
    out_path = tmp_path / "interp.md"
    render_interpretation_template(["01074520", "06910800"], out_path=out_path, candidate_id="cand_a")
    text = out_path.read_text(encoding="utf-8")
    assert "cand_a" in text
    assert "01074520" in text and "06910800" in text
    for prompt in (
        "Peak magnitude", "Peak timing", "False peaks", "Recession", "Baseflow",
        "over-/under-prediction bias", "Rainfall-runoff timing", "Rainfall/discharge mismatches",
    ):
        assert prompt in text
    assert "Observations (black)" in text and "predictions (orange)" in text
    assert text.count("- **") == 8 * 2  # exactly the 8 structured prompts per basin, for 2 basins


# ---------------------------------------------------------------------------
# L.3d -- standard compact comparison package (additive orchestration entrypoint)
# ---------------------------------------------------------------------------

def test_render_stage1_compact_comparison_package_exposes_no_period_or_scope_argument():
    import inspect
    sig = inspect.signature(render_stage1_compact_comparison_package)
    forbidden = {"period", "scope", "split", "run_dir"}
    assert forbidden.isdisjoint(sig.parameters.keys())


def test_render_stage1_compact_comparison_package_smoke(tmp_path):
    _, result_pickle_path, package_root, atlas_csv_path = _build_fixture(tmp_path, _ALL_BASIN_IDS)
    out_dir = tmp_path / "compact_package"
    result = render_stage1_compact_comparison_package(
        result_pickle=result_pickle_path, epoch=6, package_root=package_root,
        target_variable=TARGET_VARIABLE, lead_hours=LEAD_HOURS, atlas_csv=atlas_csv_path,
        out_dir=out_dir, candidate_id="cand_a", compact_target_count=4,
        min_area_samples=MIN_AREA_SAMPLES,
    )
    assert (out_dir / "compact_panel.png").is_file()
    assert (out_dir / "compact_basin_membership.json").is_file()
    assert (out_dir / "compact_event_metrics.csv").is_file()
    assert (out_dir / "event_window_table.csv").is_file()
    assert (out_dir / "interpretation_template.md").is_file()
    assert (out_dir / "rendering_manifest.json").is_file()
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "basin_panels").is_dir()
    assert len(result["compact_basin_ids"]) == 4
    for basin_id in result["compact_basin_ids"]:
        assert (out_dir / "basin_panels" / f"{basin_id}.png").is_file()

    manifest = result["manifest"]
    assert manifest["mrms_qpe_variable"] == MRMS_QPE_VARIABLE
    assert manifest["candidate_id"] == "cand_a"
    assert manifest["time_alignment_convention"]["prediction_valid_time"]
    assert manifest["time_alignment_convention"]["precipitation_valid_time"]

    metrics_df = pd.read_csv(out_dir / "compact_event_metrics.csv")
    assert "peak_timing_error_hours" in metrics_df.columns
    assert "peak_magnitude_error" in metrics_df.columns


def test_render_stage1_compact_comparison_package_closes_all_figures(tmp_path):
    _, result_pickle_path, package_root, atlas_csv_path = _build_fixture(tmp_path, _ALL_BASIN_IDS)
    out_dir = tmp_path / "compact_package"
    assert plt.get_fignums() == []
    render_stage1_compact_comparison_package(
        result_pickle=result_pickle_path, epoch=6, package_root=package_root,
        target_variable=TARGET_VARIABLE, lead_hours=LEAD_HOURS, atlas_csv=atlas_csv_path,
        out_dir=out_dir, candidate_id="cand_a", compact_target_count=4,
        min_area_samples=MIN_AREA_SAMPLES,
    )
    assert plt.get_fignums() == []


def test_render_stage1_compact_comparison_package_shared_scale_wiring(tmp_path):
    _, result_pickle_path, package_root, atlas_csv_path = _build_fixture(tmp_path, _ALL_BASIN_IDS)
    with open(result_pickle_path, "rb") as fh:
        results = pickle.load(fh)
    basin_id = _ALL_BASIN_IDS[0]
    bs = load_basin_series(
        results=results, basin_id=basin_id, target_variable=TARGET_VARIABLE,
        package_root=package_root, lead_hours=LEAD_HOURS, min_area_samples=MIN_AREA_SAMPLES,
    )
    scale = derive_comparison_scale([bs])
    atlas_df = load_atlas_selection_csv(atlas_csv_path)
    compact_df, _ = select_compact_basins(atlas_df, target_count=4)
    scale_by_basin = {gid: scale for gid in compact_df["gauge_id"]}

    out_dir_a = tmp_path / "cand_a"
    out_dir_b = tmp_path / "cand_b"
    for out_dir, candidate_id in ((out_dir_a, "cand_a"), (out_dir_b, "cand_b")):
        render_stage1_compact_comparison_package(
            result_pickle=result_pickle_path, epoch=6, package_root=package_root,
            target_variable=TARGET_VARIABLE, lead_hours=LEAD_HOURS, atlas_csv=atlas_csv_path,
            out_dir=out_dir, candidate_id=candidate_id, compact_target_count=4,
            min_area_samples=MIN_AREA_SAMPLES, scale_by_basin=scale_by_basin,
            render_individual_basin_panels=False,
        )
    manifest_a = json.loads((out_dir_a / "rendering_manifest.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((out_dir_b / "rendering_manifest.json").read_text(encoding="utf-8"))
    assert manifest_a["shared_scale_supplied"] is True
    assert manifest_b["shared_scale_supplied"] is True


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


# ---------------------------------------------------------------------------
# render_multi_candidate_basin_panel -- true N-candidate overlay
# ---------------------------------------------------------------------------

def _overlay_candidates(*, obs, sim_by_candidate, dates=None):
    """Build one BasinSeries per candidate sharing the same obs/basin_id."""
    if dates is None:
        dates = pd.date_range("2023-01-01", periods=len(obs), freq="h")
    return {
        cand_id: _synthetic_basin_series(obs=obs, sim=sim, dates=dates)
        for cand_id, sim in sim_by_candidate.items()
    }


def test_render_multi_candidate_basin_panel_writes_output_file(tmp_path):
    dates = pd.date_range("2023-01-01", periods=10, freq="h")
    series_by_cand = _overlay_candidates(
        obs=np.linspace(10, 20, 10),
        sim_by_candidate={
            "P": np.linspace(5, 15, 10), "PT": np.linspace(8, 18, 10),
            "PTM": np.linspace(7, 17, 10), "PTMW": np.linspace(9, 19, 10),
        },
        dates=dates,
    )
    window = EventWindow(
        magnitude_class="largest", peak_time=dates[5], peak_value=20.0,
        window_start=dates[0], window_end=dates[-1], window_clipped=False, n_missing_in_window=0,
    )
    out_path = tmp_path / "overlay.png"
    result = render_multi_candidate_basin_panel(
        series_by_cand, window=window,
        candidate_labels={"P": "P (ep5)", "PT": "PT (ep3)", "PTM": "PTM (ep5)", "PTMW": "PTMW (ep3)"},
        out_path=out_path,
    )
    assert result == out_path
    assert out_path.is_file()


def test_render_multi_candidate_basin_panel_closes_figure(tmp_path):
    dates = pd.date_range("2023-01-01", periods=10, freq="h")
    series_by_cand = _overlay_candidates(
        obs=np.linspace(10, 20, 10),
        sim_by_candidate={"P": np.linspace(5, 15, 10), "PT": np.linspace(8, 18, 10)},
        dates=dates,
    )
    window = EventWindow(
        magnitude_class="largest", peak_time=dates[5], peak_value=20.0,
        window_start=dates[0], window_end=dates[-1], window_clipped=False, n_missing_in_window=0,
    )
    assert plt.get_fignums() == []
    render_multi_candidate_basin_panel(
        series_by_cand, window=window, candidate_labels={"P": "P (ep5)", "PT": "PT (ep3)"},
        out_path=tmp_path / "overlay.png",
    )
    assert plt.get_fignums() == []


def test_render_multi_candidate_basin_panel_rejects_empty_input(tmp_path):
    dates = pd.date_range("2023-01-01", periods=5, freq="h")
    window = EventWindow(
        magnitude_class="largest", peak_time=dates[2], peak_value=1.0,
        window_start=dates[0], window_end=dates[-1], window_clipped=False, n_missing_in_window=0,
    )
    with pytest.raises(HydrographRenderingError):
        render_multi_candidate_basin_panel(
            {}, window=window, candidate_labels={}, out_path=tmp_path / "overlay.png",
        )


def test_render_multi_candidate_basin_panel_rejects_mismatched_basin_ids(tmp_path):
    dates = pd.date_range("2023-01-01", periods=5, freq="h")
    obs = np.linspace(1, 5, 5)
    bs_a = _synthetic_basin_series(obs=obs, sim=obs, dates=dates)
    bs_b = BasinSeries(
        basin_id="OTHER", dates=pd.DatetimeIndex(dates),
        issue_dates=pd.DatetimeIndex(dates) - pd.Timedelta(hours=LEAD_HOURS),
        obs_m3s=obs, sim_m3s=obs, admitted_mask=np.ones(5, dtype=bool),
        area_km2=100.0, n_admitted=5, n_total=5, metrics={},
    )
    window = EventWindow(
        magnitude_class="largest", peak_time=dates[2], peak_value=1.0,
        window_start=dates[0], window_end=dates[-1], window_clipped=False, n_missing_in_window=0,
    )
    with pytest.raises(HydrographRenderingError):
        render_multi_candidate_basin_panel(
            {"A": bs_a, "B": bs_b}, window=window, candidate_labels={"A": "A", "B": "B"},
            out_path=tmp_path / "overlay.png",
        )


def test_render_multi_candidate_basin_panel_rejects_mismatched_observed_series(tmp_path):
    dates = pd.date_range("2023-01-01", periods=5, freq="h")
    series_by_cand = {
        "A": _synthetic_basin_series(obs=[1, 2, 3, 4, 5], sim=[1, 1, 1, 1, 1], dates=dates),
        "B": _synthetic_basin_series(obs=[1, 2, 3, 4, 999], sim=[1, 1, 1, 1, 1], dates=dates),
    }
    window = EventWindow(
        magnitude_class="largest", peak_time=dates[2], peak_value=3.0,
        window_start=dates[0], window_end=dates[-1], window_clipped=False, n_missing_in_window=0,
    )
    with pytest.raises(HydrographRenderingError):
        render_multi_candidate_basin_panel(
            series_by_cand, window=window, candidate_labels={"A": "A", "B": "B"},
            out_path=tmp_path / "overlay.png",
        )


def test_render_multi_candidate_basin_panel_rejects_unknown_candidate_order(tmp_path):
    dates = pd.date_range("2023-01-01", periods=5, freq="h")
    series_by_cand = _overlay_candidates(obs=[1, 2, 3, 4, 5], sim_by_candidate={"A": [1, 1, 1, 1, 1]}, dates=dates)
    window = EventWindow(
        magnitude_class="largest", peak_time=dates[2], peak_value=3.0,
        window_start=dates[0], window_end=dates[-1], window_clipped=False, n_missing_in_window=0,
    )
    with pytest.raises(HydrographRenderingError):
        render_multi_candidate_basin_panel(
            series_by_cand, window=window, candidate_labels={"A": "A"},
            candidate_order=["A", "NOT_A_CANDIDATE"], out_path=tmp_path / "overlay.png",
        )


def test_render_multi_candidate_basin_panel_legend_has_observed_plus_all_candidates(tmp_path):
    dates = pd.date_range("2023-01-01", periods=10, freq="h")
    series_by_cand = _overlay_candidates(
        obs=np.linspace(10, 20, 10),
        sim_by_candidate={"P": np.linspace(5, 15, 10), "PT": np.linspace(8, 18, 10), "PTM": np.linspace(7, 17, 10)},
        dates=dates,
    )
    window = EventWindow(
        magnitude_class="largest", peak_time=dates[5], peak_value=20.0,
        window_start=dates[0], window_end=dates[-1], window_clipped=False, n_missing_in_window=0,
    )
    captured = {}
    real_subplots = plt.subplots

    def _spy_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    orig = rendering_mod.plt.subplots
    rendering_mod.plt.subplots = _spy_subplots
    try:
        render_multi_candidate_basin_panel(
            series_by_cand, window=window,
            candidate_labels={"P": "P (ep5)", "PT": "PT (ep3)", "PTM": "PTM (ep5)"},
            candidate_order=["P", "PT", "PTM"], out_path=tmp_path / "overlay.png",
        )
    finally:
        rendering_mod.plt.subplots = orig
    labels = [line.get_label() for line in captured["ax"].get_lines()]
    assert labels == ["observed", "P (ep5)", "PT (ep3)", "PTM (ep5)"]

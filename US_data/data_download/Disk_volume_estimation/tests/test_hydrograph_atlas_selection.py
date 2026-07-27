"""Tests for src/baseline/hydrograph_atlas_selection.py.

Synthetic fixtures only -- no h2o/Moriah dependency, mirroring
tests/test_compact_package_selection.py's conventions. One test additionally
loads the real, committed canonical split_assignment.csv and the real (local,
untracked-evidence) epoch009 per-basin NSE table to smoke-test the full
24-basin selection end to end.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.baseline.compact_selection import SelectionError as CompactSelectionError
from src.baseline.compact_selection import select_universe
from src.baseline.hydrograph_atlas_selection import (
    SelectionError,
    assign_skill_stratum,
    build_hydrograph_atlas_selection,
    classify_geo_side,
    compute_skill_quartile_edges,
    load_atlas_policy,
    load_per_basin_nse,
    write_selection_artifacts,
)

AREA_CLASSES = ["low", "middle", "high"]
_TEST_MACRO_REGIONS = {
    "northeast_mid_atlantic": ["01", "02"],
    "southeast": ["03"],
    "great_lakes_ohio_tennessee": ["04", "05", "06"],
    "mississippi": ["07", "08"],
    "plains_missouri_south_central": ["09", "10U", "10L", "11", "12", "13"],
    "colorado_great_basin": ["14", "15", "16"],
    "pacific_northwest_california": ["17", "18"],
    "alaska_hawaii_other": ["19", "20", "21"],
}
_TEST_EAST_MACRO_REGIONS = [
    "northeast_mid_atlantic", "southeast", "great_lakes_ohio_tennessee", "mississippi",
]
_TEST_WEST_MACRO_REGIONS = [
    "plains_missouri_south_central", "colorado_great_basin", "pacific_northwest_california",
]
# Deliberately spans both east (01, 03) and west (14, 16) macro regions.
HUC02S = ["01", "03", "14", "16"]


def _base_policy(**overrides):
    policy = {
        "selection_name": "test_hydrograph_atlas_selection_policy",
        "algorithm_id": "stage1_hydrograph_atlas_skill_area_geo_quota_selection_v1",
        "algorithm_version": 1,
        "seed": 42,
        "target_count": 16,
        "min_target_count": 8,
        "max_target_count": 30,
        "selection_universe": {
            "required_split_role": "development_train",
            "forbidden_state": "CA",
            "forbidden_split_roles": [
                "spatial_holdout_nonca", "california_finetune_train", "california_holdout",
            ],
            "required_columns": ["STAID", "split_role", "STATE", "HUC02", "area_class", "hydro_class"],
        },
        "stratification": {
            "cell_dims": ["skill_stratum", "area_class"],
            "area_class_valid_values": ["low", "middle", "high"],
            "skill_stratum_valid_values": [
                "severe_failure_lower_tail", "weak", "typical", "strong",
            ],
            "min_quota_per_nonempty_cell": 1,
        },
        "skill_stratum": {
            "quantile_edges": [0.25, 0.5, 0.75],
            "labels": {
                "below_p25": "severe_failure_lower_tail",
                "p25_to_p50": "weak",
                "p50_to_p75": "typical",
                "at_or_above_p75": "strong",
            },
        },
        "geography": {
            "distinct_huc02_soft_minimum": 2,
            "macro_region_map_version": 1,
            "macro_regions": _TEST_MACRO_REGIONS,
            "east_macro_regions": _TEST_EAST_MACRO_REGIONS,
            "west_macro_regions": _TEST_WEST_MACRO_REGIONS,
        },
        "output": {"default_subdir_name": "test"},
    }
    policy.update(overrides)
    return policy


def _grid_rows():
    rows = []
    counter = 100001
    for huc in HUC02S:
        for area in AREA_CLASSES:
            for _ in range(6):
                rows.append({
                    "STAID": f"00{counter:06d}", "split_role": "development_train", "STATE": "ME",
                    "HUC02": huc, "area_class": area, "hydro_class": "low",
                })
                counter += 1
    return rows


def _assignment_df():
    return pd.DataFrame(_grid_rows())


def _universe():
    return select_universe(_assignment_df(), _base_policy())


def _nse_series_for(universe, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    values = rng.uniform(-1.0, 1.0, size=len(universe))
    return pd.Series(values, index=universe.index, name="nse")


# ---------------------------------------------------------------------------
# load_atlas_policy
# ---------------------------------------------------------------------------

def test_load_atlas_policy_loads_real_config():
    real_config = Path(__file__).resolve().parents[1] / "config" / "stage1_hydrograph_atlas_selection_v001.yaml"
    policy = load_atlas_policy(real_config)
    assert policy["selection_name"] == "stage1_hydrograph_atlas_selection_v001"
    assert policy["min_target_count"] <= policy["target_count"] <= policy["max_target_count"]


def test_load_atlas_policy_rejects_missing_key(tmp_path):
    import yaml
    policy = _base_policy()
    del policy["geography"]
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(policy), encoding="utf-8")
    with pytest.raises(SelectionError):
        load_atlas_policy(path)


def test_load_atlas_policy_rejects_out_of_range_target_count(tmp_path):
    import yaml
    policy = _base_policy(target_count=999)
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(policy), encoding="utf-8")
    with pytest.raises(SelectionError):
        load_atlas_policy(path)


def test_load_atlas_policy_missing_file_raises(tmp_path):
    with pytest.raises(SelectionError):
        load_atlas_policy(tmp_path / "does_not_exist.yaml")


# ---------------------------------------------------------------------------
# load_per_basin_nse
# ---------------------------------------------------------------------------

def test_load_per_basin_nse_basic(tmp_path):
    df = pd.DataFrame({"basin_id": ["00100001", "00100002"], "nse": [0.5, -0.2]})
    path = tmp_path / "per_basin.csv"
    df.to_csv(path, index=False)
    series = load_per_basin_nse(path)
    assert series.loc["00100001"] == 0.5
    assert series.loc["00100002"] == -0.2


def test_load_per_basin_nse_drops_nonfinite(tmp_path):
    df = pd.DataFrame({"basin_id": ["00100001", "00100002"], "nse": [0.5, np.nan]})
    path = tmp_path / "per_basin.csv"
    df.to_csv(path, index=False)
    series = load_per_basin_nse(path)
    assert "00100002" not in series.index
    assert len(series) == 1


def test_load_per_basin_nse_rejects_duplicate_staid(tmp_path):
    df = pd.DataFrame({"basin_id": ["00100001", "00100001"], "nse": [0.5, 0.6]})
    path = tmp_path / "per_basin.csv"
    df.to_csv(path, index=False)
    with pytest.raises(SelectionError):
        load_per_basin_nse(path)


def test_load_per_basin_nse_rejects_missing_column(tmp_path):
    df = pd.DataFrame({"basin_id": ["00100001"], "other_col": [0.5]})
    path = tmp_path / "per_basin.csv"
    df.to_csv(path, index=False)
    with pytest.raises(SelectionError):
        load_per_basin_nse(path)


# ---------------------------------------------------------------------------
# compute_skill_quartile_edges / assign_skill_stratum
# ---------------------------------------------------------------------------

def test_compute_skill_quartile_edges_basic():
    values = pd.Series(np.arange(1, 101, dtype=float))
    p25, p50, p75 = compute_skill_quartile_edges(values)
    assert p25 == pytest.approx(25.75)
    assert p50 == pytest.approx(50.5)
    assert p75 == pytest.approx(75.25)


def test_compute_skill_quartile_edges_rejects_nulls():
    values = pd.Series([1.0, np.nan, 3.0])
    with pytest.raises(SelectionError):
        compute_skill_quartile_edges(values)


def test_assign_skill_stratum_boundaries():
    edges = (10.0, 20.0, 30.0)
    values = pd.Series([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0])
    out = assign_skill_stratum(values, edges)
    assert list(out) == [
        "severe_failure_lower_tail",  # 5 < 10
        "weak",                        # 10 -> not < 10, < 20
        "weak",                        # 15
        "typical",                     # 20 -> not < 20, < 30
        "typical",                     # 25
        "strong",                      # 30 -> not < 30
        "strong",                      # 35
    ]


# ---------------------------------------------------------------------------
# classify_geo_side
# ---------------------------------------------------------------------------

def test_classify_geo_side_east_west():
    policy = _base_policy()
    huc02 = pd.Series(["01", "14", "03", "16"], index=["a", "b", "c", "d"])
    sides = classify_geo_side(huc02, policy)
    assert sides.loc["a"] == "east"
    assert sides.loc["b"] == "west"
    assert sides.loc["c"] == "east"
    assert sides.loc["d"] == "west"


def test_classify_geo_side_rejects_unmapped_huc02():
    policy = _base_policy()
    huc02 = pd.Series(["99"], index=["a"])
    with pytest.raises(CompactSelectionError):
        classify_geo_side(huc02, policy)


# ---------------------------------------------------------------------------
# build_hydrograph_atlas_selection -- full-pipeline acceptance checks
# ---------------------------------------------------------------------------

def test_build_hydrograph_atlas_selection_deterministic_repeat():
    universe = _universe()
    nse = _nse_series_for(universe)
    policy = _base_policy()
    sel1, manifest1 = build_hydrograph_atlas_selection(universe, nse, policy)
    sel2, manifest2 = build_hydrograph_atlas_selection(universe, nse, policy)
    pd.testing.assert_frame_equal(sel1, sel2)
    assert manifest1["cell_quota"] == manifest2["cell_quota"]


def test_build_hydrograph_atlas_selection_exact_target_count():
    universe = _universe()
    nse = _nse_series_for(universe)
    policy = _base_policy(target_count=12)
    sel, manifest = build_hydrograph_atlas_selection(universe, nse, policy)
    assert len(sel) == 12
    assert manifest["counts"]["n_selected"] == 12


def test_build_hydrograph_atlas_selection_no_leakage_outside_universe():
    universe = _universe()
    nse = _nse_series_for(universe)
    sel, _ = build_hydrograph_atlas_selection(universe, nse, _base_policy())
    assert set(sel["gauge_id"]).issubset(set(universe.index))


def test_build_hydrograph_atlas_selection_no_duplicate_gauge_ids():
    universe = _universe()
    nse = _nse_series_for(universe)
    sel, _ = build_hydrograph_atlas_selection(universe, nse, _base_policy())
    assert not sel["gauge_id"].duplicated().any()


def test_build_hydrograph_atlas_selection_east_and_west_represented():
    universe = _universe()
    nse = _nse_series_for(universe)
    sel, manifest = build_hydrograph_atlas_selection(universe, nse, _base_policy())
    assert manifest["east_west_breadth"]["n_east"] > 0
    assert manifest["east_west_breadth"]["n_west"] > 0
    assert set(sel["geo_side"]) == {"east", "west"}


def test_build_hydrograph_atlas_selection_raises_without_east_west_spread():
    universe = _universe()
    universe = universe.loc[universe["HUC02"].isin(["01", "03"])]  # east only
    nse = _nse_series_for(universe)
    with pytest.raises(SelectionError, match="east/west"):
        build_hydrograph_atlas_selection(universe, nse, _base_policy(target_count=8))


def test_build_hydrograph_atlas_selection_skill_stratum_representation():
    universe = _universe()
    nse = _nse_series_for(universe)
    sel, manifest = build_hydrograph_atlas_selection(universe, nse, _base_policy(target_count=16))
    assert "skill_quartile_edges" in manifest
    assert set(sel["skill_stratum"]).issubset({
        "severe_failure_lower_tail", "weak", "typical", "strong",
    })


def test_build_hydrograph_atlas_selection_missing_nse_basins_excluded():
    universe = _universe()
    nse = _nse_series_for(universe)
    dropped = sorted(universe.index)[:5]
    nse = nse.drop(index=dropped)
    sel, manifest = build_hydrograph_atlas_selection(universe, nse, _base_policy(target_count=12))
    assert set(sel["gauge_id"]).isdisjoint(set(dropped))
    assert set(manifest["universe_basins_missing_nse"]) == set(dropped)


def test_build_hydrograph_atlas_selection_raises_when_population_too_small():
    universe = _universe().iloc[:5]
    nse = _nse_series_for(universe)
    with pytest.raises(SelectionError):
        build_hydrograph_atlas_selection(universe, nse, _base_policy(target_count=16))


def test_build_hydrograph_atlas_selection_stable_ordering():
    universe = _universe()
    nse = _nse_series_for(universe)
    sel, _ = build_hydrograph_atlas_selection(universe, nse, _base_policy())
    assert list(sel["gauge_id"]) == sorted(sel["gauge_id"])


def test_build_hydrograph_atlas_selection_no_cherry_picking_columns_only():
    # Guard against accidentally wiring in a prediction-error or test-set
    # signal: the selection frame must only ever contain columns derivable
    # from canonical split metadata + the supplied validation-only NSE.
    universe = _universe()
    nse = _nse_series_for(universe)
    sel, _ = build_hydrograph_atlas_selection(universe, nse, _base_policy())
    allowed = {
        "gauge_id", "canonical_basin_role", "huc02", "macro_region", "geo_side",
        "area_class", "hydro_class", "nse", "skill_stratum", "selection_reason",
    }
    assert set(sel.columns) == allowed


# ---------------------------------------------------------------------------
# Real-data smoke test (canonical split + real seed-run per-basin evidence)
# ---------------------------------------------------------------------------

_REAL_SPLIT_ASSIGNMENT = Path(__file__).resolve().parents[1] / "config" / "stage1_baseline_splits_v001" / "split_assignment.csv"
_REAL_PER_BASIN_NSE = (
    Path(__file__).resolve().parents[1] / "reports" / "seed_validation_review_v001"
    / "per_basin" / "epoch009" / "epoch009_per_basin_metrics.csv"
)


@pytest.mark.skipif(
    not (_REAL_SPLIT_ASSIGNMENT.is_file() and _REAL_PER_BASIN_NSE.is_file()),
    reason="real canonical split assignment or real per-basin NSE evidence not present locally",
)
def test_build_hydrograph_atlas_selection_real_data_smoke():
    from src.baseline.compact_selection import load_split_assignment
    from src.baseline.hydrograph_atlas_selection import load_atlas_policy

    real_policy_path = Path(__file__).resolve().parents[1] / "config" / "stage1_hydrograph_atlas_selection_v001.yaml"
    policy = load_atlas_policy(real_policy_path)
    assignment_df = load_split_assignment(
        _REAL_SPLIT_ASSIGNMENT,
        ["STAID", "split_role", "STATE", "HUC02", "area_class", "hydro_class"],
    )
    universe = select_universe(assignment_df, policy)
    nse = load_per_basin_nse(_REAL_PER_BASIN_NSE)

    sel, manifest = build_hydrograph_atlas_selection(universe, nse, policy)

    assert len(sel) == policy["target_count"] == 24
    assert not sel["gauge_id"].duplicated().any()
    assert set(sel["gauge_id"]).issubset(set(universe.index))
    assert manifest["east_west_breadth"]["n_east"] > 0
    assert manifest["east_west_breadth"]["n_west"] > 0
    assert manifest["empty_cells"] == []  # all 24 cells populated, verified against real data


# ---------------------------------------------------------------------------
# write_selection_artifacts -- manifest/checksum consistency
# ---------------------------------------------------------------------------

def _sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_write_selection_artifacts_manifest_checksums_consistent(tmp_path):
    universe = _universe()
    nse = _nse_series_for(universe)
    sel, manifest_pieces = build_hydrograph_atlas_selection(universe, nse, _base_policy())
    manifest = {"created_by": "test", "status": "candidate", **manifest_pieces}
    out_dir = tmp_path / "out"
    paths = write_selection_artifacts(out_dir, sel, manifest)

    expected_files = {
        "hydrograph_atlas_basin_selection.csv", "hydrograph_atlas_basin_ids.txt",
        "selection_summary.md", "selection_summary.json", "selection_manifest.json",
    }
    assert expected_files.issubset(paths.keys())

    written_manifest = json.loads(paths["selection_manifest.json"].read_text(encoding="utf-8"))
    for name, path in paths.items():
        if name == "selection_manifest.json":
            continue
        assert written_manifest["artifact_sha256"][name] == _sha256_of(path)

    ids_text = paths["hydrograph_atlas_basin_ids.txt"].read_text(encoding="utf-8").splitlines()
    assert ids_text == sorted(sel["gauge_id"].tolist())


def test_write_selection_artifacts_refuses_nonempty_dir_without_force(tmp_path):
    universe = _universe()
    nse = _nse_series_for(universe)
    sel, manifest_pieces = build_hydrograph_atlas_selection(universe, nse, _base_policy())
    manifest = {"created_by": "test", "status": "candidate", **manifest_pieces}
    out_dir = tmp_path / "out"
    write_selection_artifacts(out_dir, sel, manifest)
    with pytest.raises(SelectionError):
        write_selection_artifacts(out_dir, sel, manifest)
    write_selection_artifacts(out_dir, sel, manifest, force=True)

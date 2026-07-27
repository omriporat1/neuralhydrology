"""Tests for src/baseline/screening_subset_selection.py.

Synthetic fixtures only -- no h2o/Moriah dependency, mirroring
tests/test_hydrograph_atlas_selection.py's conventions. One test additionally
loads the real, committed canonical split_assignment.csv plus the real
(local, untracked-evidence) epoch009 per-basin NSE table and the real WY2024
RBI screening results to smoke-test the full ~400-basin selection end to end.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.baseline.compact_selection import select_universe
from src.baseline.screening_subset_selection import (
    SelectionError,
    build_screening_subset_selection,
    load_flow_variability,
    load_per_basin_nse,
    load_screening_subset_policy,
    write_selection_artifacts,
)

AREA_CLASSES = ["low", "middle", "high"]
HYDRO_CLASSES = ["low", "middle", "high"]
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
# Deliberately spans 3 distinct macro regions (01 -> northeast_mid_atlantic,
# 03 -> southeast, 14 & 16 -> colorado_great_basin).
HUC02S = ["01", "03", "14", "16"]


def _base_policy(**overrides):
    policy = {
        "selection_name": "test_screening_subset_policy",
        "algorithm_id": "stage1_screening_subset_proportional_composite_stratum_selection_v1",
        "algorithm_version": 1,
        "seed": 42,
        "target_count": 80,
        "min_target_count": 40,
        "max_target_count": 200,
        "selection_universe": {
            "required_split_role": "development_train",
            "forbidden_state": "CA",
            "forbidden_split_roles": [
                "spatial_holdout_nonca", "california_finetune_train", "california_holdout",
            ],
            "required_columns": ["STAID", "split_role", "STATE", "HUC02", "area_class", "hydro_class"],
        },
        "stratification": {
            "cell_dims": ["macro_region", "area_class", "hydro_class", "flow_var_class", "skill_stratum"],
            "area_class_valid_values": ["low", "middle", "high"],
            "hydro_class_valid_values": ["low", "middle", "high", "missing"],
            "flow_var_class_valid_values": ["low", "middle", "high"],
            "skill_stratum_valid_values": [
                "severe_failure_lower_tail", "weak", "typical", "strong",
            ],
            "min_quota_per_nonempty_cell": 0,
        },
        "flow_variability": {
            "source_value_column": "rbi",
            "source_staid_column": "STAID",
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
            "macro_region_map_version": 1,
            "macro_regions": _TEST_MACRO_REGIONS,
        },
        "output": {"default_subdir_name": "test"},
    }
    policy.update(overrides)
    return policy


def _grid_rows(n_per_cell: int = 6):
    rows = []
    counter = 100001
    for huc in HUC02S:
        for area in AREA_CLASSES:
            for hydro in HYDRO_CLASSES:
                for _ in range(n_per_cell):
                    rows.append({
                        "STAID": f"00{counter:06d}", "split_role": "development_train", "STATE": "ME",
                        "HUC02": huc, "area_class": area, "hydro_class": hydro,
                    })
                    counter += 1
    return rows


def _assignment_df(n_per_cell: int = 6):
    return pd.DataFrame(_grid_rows(n_per_cell))


def _universe(n_per_cell: int = 6):
    return select_universe(_assignment_df(n_per_cell), _base_policy())


def _nse_series_for(universe, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    values = rng.uniform(-1.0, 1.0, size=len(universe))
    return pd.Series(values, index=universe.index, name="nse")


def _flow_var_series_for(universe, rng_seed=1):
    rng = np.random.default_rng(rng_seed)
    values = rng.uniform(0.0, 1.0, size=len(universe))
    return pd.Series(values, index=universe.index, name="rbi")


# ---------------------------------------------------------------------------
# load_screening_subset_policy
# ---------------------------------------------------------------------------

def test_load_screening_subset_policy_loads_real_config():
    real_config = Path(__file__).resolve().parents[1] / "config" / "stage1_screening_subset_v001.yaml"
    policy = load_screening_subset_policy(real_config)
    assert policy["selection_name"] == "stage1_screening_subset_v001"
    assert policy["min_target_count"] <= policy["target_count"] <= policy["max_target_count"]


def test_load_screening_subset_policy_rejects_missing_key(tmp_path):
    import yaml
    policy = _base_policy()
    del policy["flow_variability"]
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(policy), encoding="utf-8")
    with pytest.raises(SelectionError):
        load_screening_subset_policy(path)


def test_load_screening_subset_policy_rejects_out_of_range_target_count(tmp_path):
    import yaml
    policy = _base_policy(target_count=999999)
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(policy), encoding="utf-8")
    with pytest.raises(SelectionError):
        load_screening_subset_policy(path)


def test_load_screening_subset_policy_missing_file_raises(tmp_path):
    with pytest.raises(SelectionError):
        load_screening_subset_policy(tmp_path / "does_not_exist.yaml")


# ---------------------------------------------------------------------------
# load_flow_variability
# ---------------------------------------------------------------------------

def test_load_flow_variability_basic(tmp_path):
    df = pd.DataFrame({"STAID": ["00100001", "00100002"], "rbi": [0.1, 0.5]})
    path = tmp_path / "rbi.csv"
    df.to_csv(path, index=False)
    series = load_flow_variability(path)
    assert series.loc["00100001"] == 0.1
    assert series.loc["00100002"] == 0.5


def test_load_flow_variability_drops_nonfinite(tmp_path):
    df = pd.DataFrame({"STAID": ["00100001", "00100002"], "rbi": [0.1, np.nan]})
    path = tmp_path / "rbi.csv"
    df.to_csv(path, index=False)
    series = load_flow_variability(path)
    assert "00100002" not in series.index
    assert len(series) == 1


def test_load_flow_variability_rejects_duplicate_staid(tmp_path):
    df = pd.DataFrame({"STAID": ["00100001", "00100001"], "rbi": [0.1, 0.2]})
    path = tmp_path / "rbi.csv"
    df.to_csv(path, index=False)
    with pytest.raises(SelectionError):
        load_flow_variability(path)


def test_load_flow_variability_rejects_missing_column(tmp_path):
    df = pd.DataFrame({"STAID": ["00100001"], "other_col": [0.1]})
    path = tmp_path / "rbi.csv"
    df.to_csv(path, index=False)
    with pytest.raises(SelectionError):
        load_flow_variability(path)


# ---------------------------------------------------------------------------
# build_screening_subset_selection -- full-pipeline acceptance checks
# ---------------------------------------------------------------------------

def test_build_screening_subset_selection_deterministic_repeat():
    universe = _universe()
    nse = _nse_series_for(universe)
    flow_var = _flow_var_series_for(universe)
    policy = _base_policy()
    sel1, manifest1 = build_screening_subset_selection(universe, nse, flow_var, policy)
    sel2, manifest2 = build_screening_subset_selection(universe, nse, flow_var, policy)
    pd.testing.assert_frame_equal(sel1, sel2)
    assert manifest1["counts"] == manifest2["counts"]


def test_build_screening_subset_selection_exact_target_count():
    universe = _universe()
    nse = _nse_series_for(universe)
    flow_var = _flow_var_series_for(universe)
    policy = _base_policy(target_count=50)
    sel, manifest = build_screening_subset_selection(universe, nse, flow_var, policy)
    assert len(sel) == 50
    assert manifest["counts"]["n_selected"] == 50


def test_build_screening_subset_selection_no_leakage_outside_universe():
    universe = _universe()
    nse = _nse_series_for(universe)
    flow_var = _flow_var_series_for(universe)
    sel, _ = build_screening_subset_selection(universe, nse, flow_var, _base_policy())
    assert set(sel["gauge_id"]).issubset(set(universe.index))


def test_build_screening_subset_selection_no_duplicate_gauge_ids():
    universe = _universe()
    nse = _nse_series_for(universe)
    flow_var = _flow_var_series_for(universe)
    sel, _ = build_screening_subset_selection(universe, nse, flow_var, _base_policy())
    assert not sel["gauge_id"].duplicated().any()


def test_build_screening_subset_selection_stable_ordering():
    universe = _universe()
    nse = _nse_series_for(universe)
    flow_var = _flow_var_series_for(universe)
    sel, _ = build_screening_subset_selection(universe, nse, flow_var, _base_policy())
    assert list(sel["gauge_id"]) == sorted(sel["gauge_id"])


def test_build_screening_subset_selection_different_seed_changes_membership():
    universe = _universe()
    nse = _nse_series_for(universe)
    flow_var = _flow_var_series_for(universe)
    sel1, _ = build_screening_subset_selection(universe, nse, flow_var, _base_policy(seed=42))
    sel2, _ = build_screening_subset_selection(universe, nse, flow_var, _base_policy(seed=43))
    assert set(sel1["gauge_id"]) != set(sel2["gauge_id"])
    assert len(sel1) == len(sel2)


def test_build_screening_subset_selection_multiple_macro_regions_represented():
    universe = _universe()
    nse = _nse_series_for(universe)
    flow_var = _flow_var_series_for(universe)
    sel, manifest = build_screening_subset_selection(universe, nse, flow_var, _base_policy())
    assert sel["macro_region"].nunique() >= 2
    assert manifest["counts"]["distinct_macro_region"] >= 2


def test_build_screening_subset_selection_skill_and_flow_var_strata_present():
    universe = _universe()
    nse = _nse_series_for(universe)
    flow_var = _flow_var_series_for(universe)
    sel, manifest = build_screening_subset_selection(universe, nse, flow_var, _base_policy())
    assert set(sel["skill_stratum"]).issubset({
        "severe_failure_lower_tail", "weak", "typical", "strong",
    })
    assert set(sel["flow_var_class"]).issubset({"low", "middle", "high"})
    assert "skill_quartile_edges" in manifest
    assert "flow_var_tercile_edges" in manifest


def test_build_screening_subset_selection_missing_nse_basins_excluded():
    universe = _universe()
    nse = _nse_series_for(universe)
    flow_var = _flow_var_series_for(universe)
    dropped = sorted(universe.index)[:5]
    nse = nse.drop(index=dropped)
    sel, manifest = build_screening_subset_selection(universe, nse, flow_var, _base_policy(target_count=50))
    assert set(sel["gauge_id"]).isdisjoint(set(dropped))
    assert set(manifest["universe_basins_missing_nse"]) == set(dropped)


def test_build_screening_subset_selection_missing_flow_var_basins_excluded():
    universe = _universe()
    nse = _nse_series_for(universe)
    flow_var = _flow_var_series_for(universe)
    dropped = sorted(universe.index)[:5]
    flow_var = flow_var.drop(index=dropped)
    sel, manifest = build_screening_subset_selection(universe, nse, flow_var, _base_policy(target_count=50))
    assert set(sel["gauge_id"]).isdisjoint(set(dropped))
    assert set(manifest["universe_basins_missing_flow_variability"]) == set(dropped)


def test_build_screening_subset_selection_raises_when_population_too_small():
    universe = _universe().iloc[:10]
    nse = _nse_series_for(universe)
    flow_var = _flow_var_series_for(universe)
    with pytest.raises(SelectionError):
        build_screening_subset_selection(universe, nse, flow_var, _base_policy(target_count=80))


def test_build_screening_subset_selection_proportional_not_one_per_cell():
    # With min_quota_per_nonempty_cell=0 and a small target relative to the
    # number of joint composite cells, some non-empty cells must legitimately
    # receive zero quota -- this is the defining behavior distinguishing
    # Part D's proportional allocation from Part C's "one guaranteed basin
    # per cell" rule.
    universe = _universe(n_per_cell=6)
    nse = _nse_series_for(universe)
    flow_var = _flow_var_series_for(universe)
    sel, manifest = build_screening_subset_selection(universe, nse, flow_var, _base_policy(target_count=50))
    assert manifest["n_cells_with_zero_quota"] > 0
    assert manifest["n_nonempty_cells"] > len(sel)


def test_build_screening_subset_selection_reasonable_marginal_balance():
    # A large-population synthetic fixture should reproduce area_class
    # marginals reasonably closely (proportional sampling, not exact -- a
    # few percentage points of slack is expected from rounding).
    universe = _universe(n_per_cell=20)
    nse = _nse_series_for(universe)
    flow_var = _flow_var_series_for(universe)
    sel, manifest = build_screening_subset_selection(universe, nse, flow_var, _base_policy(target_count=200))
    assert manifest["max_abs_marginal_frac_diff"] < 0.10


def test_build_screening_subset_selection_hydro_class_missing_bucket_included():
    rows = _grid_rows(n_per_cell=6)
    # Add a small population of basins with a "missing" hydro_class (mirrors
    # the 5 real development_train basins lacking an aridity value) -- these
    # must not be silently dropped from the grid.
    counter = 900001
    for _ in range(10):
        rows.append({
            "STAID": f"00{counter:06d}", "split_role": "development_train", "STATE": "ME",
            "HUC02": "01", "area_class": "low", "hydro_class": "missing",
        })
        counter += 1
    assignment_df = pd.DataFrame(rows)
    universe = select_universe(assignment_df, _base_policy())
    nse = _nse_series_for(universe)
    flow_var = _flow_var_series_for(universe)
    sel, manifest = build_screening_subset_selection(universe, nse, flow_var, _base_policy(target_count=80))
    # Not asserting "missing" is necessarily drawn (proportional allocation
    # may legitimately assign it zero quota given its tiny population share)
    # -- only that it was not structurally excluded from the eligible grid.
    assert "missing" not in set(manifest["basins_excluded_from_grid"])


# ---------------------------------------------------------------------------
# Real-data smoke test (canonical split + real seed-run evidence + real RBI)
# ---------------------------------------------------------------------------

_REAL_SPLIT_ASSIGNMENT = Path(__file__).resolve().parents[1] / "config" / "stage1_baseline_splits_v001" / "split_assignment.csv"
_REAL_PER_BASIN_NSE = (
    Path(__file__).resolve().parents[1] / "reports" / "seed_validation_review_v001"
    / "per_basin" / "epoch009" / "epoch009_per_basin_metrics.csv"
)
_REAL_RBI_TABLE = (
    Path(__file__).resolve().parents[1] / "reports" / "flashnh_usgs_rbi_screening_wy2024_v001"
    / "usgs_rbi_screening_results.csv"
)


@pytest.mark.skipif(
    not (_REAL_SPLIT_ASSIGNMENT.is_file() and _REAL_PER_BASIN_NSE.is_file() and _REAL_RBI_TABLE.is_file()),
    reason="real canonical split assignment, per-basin NSE, or RBI evidence not present locally",
)
def test_build_screening_subset_selection_real_data_smoke():
    from src.baseline.compact_selection import load_split_assignment

    real_policy_path = Path(__file__).resolve().parents[1] / "config" / "stage1_screening_subset_v001.yaml"
    policy = load_screening_subset_policy(real_policy_path)
    assignment_df = load_split_assignment(
        _REAL_SPLIT_ASSIGNMENT,
        ["STAID", "split_role", "STATE", "HUC02", "area_class", "hydro_class"],
    )
    universe = select_universe(assignment_df, policy)
    nse = load_per_basin_nse(_REAL_PER_BASIN_NSE)
    flow_var = load_flow_variability(
        _REAL_RBI_TABLE,
        staid_column=policy["flow_variability"]["source_staid_column"],
        value_column=policy["flow_variability"]["source_value_column"],
    )

    sel, manifest = build_screening_subset_selection(universe, nse, flow_var, policy)

    assert len(sel) == policy["target_count"] == 400
    assert not sel["gauge_id"].duplicated().any()
    assert set(sel["gauge_id"]).issubset(set(universe.index))
    assert manifest["max_abs_marginal_frac_diff"] < 0.05


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
    flow_var = _flow_var_series_for(universe)
    sel, manifest_pieces = build_screening_subset_selection(universe, nse, flow_var, _base_policy())
    manifest = {"created_by": "test", "status": "candidate", **manifest_pieces}
    out_dir = tmp_path / "out"
    paths = write_selection_artifacts(out_dir, sel, manifest)

    expected_files = {
        "screening_subset_basin_selection.csv", "screening_subset_basin_ids.txt",
        "selection_summary.md", "selection_summary.json", "selection_manifest.json",
    }
    assert expected_files.issubset(paths.keys())

    written_manifest = json.loads(paths["selection_manifest.json"].read_text(encoding="utf-8"))
    for name, path in paths.items():
        if name == "selection_manifest.json":
            continue
        assert written_manifest["artifact_sha256"][name] == _sha256_of(path)

    ids_text = paths["screening_subset_basin_ids.txt"].read_text(encoding="utf-8").splitlines()
    assert ids_text == sorted(sel["gauge_id"].tolist())


def test_write_selection_artifacts_refuses_nonempty_dir_without_force(tmp_path):
    universe = _universe()
    nse = _nse_series_for(universe)
    flow_var = _flow_var_series_for(universe)
    sel, manifest_pieces = build_screening_subset_selection(universe, nse, flow_var, _base_policy())
    manifest = {"created_by": "test", "status": "candidate", **manifest_pieces}
    out_dir = tmp_path / "out"
    write_selection_artifacts(out_dir, sel, manifest)
    with pytest.raises(SelectionError):
        write_selection_artifacts(out_dir, sel, manifest)
    write_selection_artifacts(out_dir, sel, manifest, force=True)

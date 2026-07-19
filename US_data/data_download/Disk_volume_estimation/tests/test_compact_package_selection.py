"""Tests for src/baseline/compact_selection.py.

Synthetic fixtures only -- no h2o/Moriah dependency. The synthetic
development pool below deliberately mirrors the real Stage 1 split's known
edge cases (a 9-char and a 15-char STAID, one hydro_class == "missing"
basin) plus an explicit holdout and a California row that must never be
selected, so a single fixture exercises every acceptance check in
docs/stage1_compact_package_selection.md.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.baseline.compact_selection import (
    SelectionError,
    allocate_cell_quota,
    bin_qobs_completeness,
    bin_static_missing,
    build_compact_selection,
    build_macro_region_map,
    build_reserved_selection,
    flag_unusual_identifier,
    load_qobs_status,
    load_selection_policy,
    load_split_assignment,
    macro_region_for_huc02,
    macro_region_side,
    make_cell_rngs,
    select_universe,
    select_within_cell,
    write_selection_artifacts,
)

AREA_CLASSES = ["low", "middle", "high"]
HYDRO_CLASSES = ["low", "middle", "high"]
# Deliberately spans both east (01, 02, 03) and west (14, 16, 17) macro
# regions -- a HUC01-06-only fixture cannot exercise the east/west breadth
# checks below (see docs/stage1_compact_package_selection.md, "Geographic
# diagnostics").
HUC02S = ["01", "02", "03", "14", "16", "17"]

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


def _base_policy(geography_overrides=None, **overrides):
    policy = {
        "selection_name": "test_compact_selection_policy",
        "algorithm_id": "stage1_compact_diversity_quota_selection_v1",
        "algorithm_version": 2,
        "seed": 42,
        "target_count": 12,
        "min_target_count": 4,
        "max_target_count": 50,
        "selection_universe": {
            "required_split_role": "development_train",
            "forbidden_state": "CA",
            "forbidden_split_roles": [
                "spatial_holdout_nonca", "california_finetune_train", "california_holdout",
            ],
            "required_columns": ["STAID", "split_role", "STATE", "HUC02", "area_class", "hydro_class"],
        },
        "stratification": {
            "cell_dims": ["area_class", "hydro_class"],
            "valid_class_values": ["low", "middle", "high"],
            "min_quota_per_nonempty_cell": 1,
        },
        "geography": {
            "distinct_huc02_soft_minimum": 2,
            "macro_region_map_version": 1,
            "macro_regions": _TEST_MACRO_REGIONS,
            "east_macro_regions": _TEST_EAST_MACRO_REGIONS,
            "west_macro_regions": _TEST_WEST_MACRO_REGIONS,
            "distinct_macro_region_soft_minimum": 2,
            # Off by default so pre-existing tests (written before the
            # east/west breadth check existed) aren't incidentally
            # sensitive to which side of CONUS seed=42 happens to draw at
            # a given target_count. Tests that specifically exercise the
            # check turn it on via geography_overrides.
            "require_east_west_spread": False,
        },
        "qobs_completeness": {
            "coverage_column_candidates": ["coverage_fraction", "qobs_coverage"],
            "staid_column_candidates": ["STAID", "gauge_id"],
            "bin_edges": {"low_max": 0.90, "mid_max": 0.99},
        },
        "static_missingness": {
            "model_input_role": "model_input",
            "bin_edges": {"none_max": 0, "some_max": 5},
        },
        "reserved_categories": {
            "unusual_identifier": {"standard_length": 8, "forced_include_cap": 1},
            "hydro_stratifier_gap": {"forced_include_cap": 1},
            "static_missing_value_case": {"forced_include_cap": 1},
        },
        "output": {"default_subdir_name": "test"},
    }
    if geography_overrides:
        policy["geography"] = {**policy["geography"], **geography_overrides}
    policy.update(overrides)
    return policy


def _grid_rows():
    rows = []
    counter = 100001
    for huc in HUC02S:
        for area in AREA_CLASSES:
            for hydro in HYDRO_CLASSES:
                for _ in range(3):
                    rows.append({
                        "STAID": f"00{counter:06d}", "split_role": "development_train", "STATE": "ME",
                        "HUC02": huc, "area_class": area, "hydro_class": hydro,
                    })
                    counter += 1
    return rows


def _edge_case_rows():
    return [
        {"STAID": "123456789", "split_role": "development_train", "STATE": "ME",
         "HUC02": "01", "area_class": "low", "hydro_class": "low"},
        {"STAID": "123456789012345", "split_role": "development_train", "STATE": "CO",
         "HUC02": "10L", "area_class": "high", "hydro_class": "low"},
        {"STAID": "00999001", "split_role": "development_train", "STATE": "CO",
         "HUC02": "10L", "area_class": "high", "hydro_class": "missing"},
    ]


def _overlap_edge_case_row():
    # Deliberately satisfies BOTH the unusual_identifier predicate (15-char
    # STAID) AND the hydro_stratifier_gap predicate (hydro_class == "missing")
    # -- this is the real-world pattern in the canonical 2,307-basin pool,
    # where all 5 real 15-char STAIDs also have hydro_class == "missing".
    # Kept separate from _edge_case_rows() (opt-in via include_overlap_case)
    # so the many pre-existing tests that assume 00999001 is reserved purely
    # for hydro_stratifier_gap are unaffected unless a test opts in.
    return {"STAID": "223456789012345", "split_role": "development_train", "STATE": "CO",
            "HUC02": "10L", "area_class": "middle", "hydro_class": "missing"}


def _leakage_rows():
    return [
        {"STAID": "00777001", "split_role": "spatial_holdout_nonca", "STATE": "ME",
         "HUC02": "01", "area_class": "low", "hydro_class": "low"},
        {"STAID": "00666001", "split_role": "california_finetune_train", "STATE": "CA",
         "HUC02": "18", "area_class": "low", "hydro_class": "low"},
    ]


def _all_rows(include_edge_cases=True, include_leakage=True, include_overlap_case=False):
    rows = _grid_rows()
    if include_edge_cases:
        rows += _edge_case_rows()
    if include_overlap_case:
        rows += [_overlap_edge_case_row()]
    if include_leakage:
        rows += _leakage_rows()
    return rows


def _assignment_df(**kwargs):
    return pd.DataFrame(_all_rows(**kwargs))


def _write_assignment_csv(tmp_path, rows=None, name="split_assignment.csv"):
    df = pd.DataFrame(rows if rows is not None else _all_rows())
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def _universe(**kwargs):
    return select_universe(_assignment_df(**kwargs), _base_policy())


# ---------------------------------------------------------------------------
# load_split_assignment / select_universe
# ---------------------------------------------------------------------------

def test_load_split_assignment_preserves_leading_zeros(tmp_path):
    path = _write_assignment_csv(tmp_path)
    df = load_split_assignment(path, ["STAID", "split_role", "STATE", "HUC02", "area_class", "hydro_class"])
    assert (df["STAID"].str.len() >= 8).all()
    assert "00100001" in set(df["STAID"])


def test_load_split_assignment_rejects_duplicate_staid(tmp_path):
    rows = _all_rows()
    rows.append(dict(rows[0]))  # exact duplicate row
    path = _write_assignment_csv(tmp_path, rows=rows)
    with pytest.raises(SelectionError):
        load_split_assignment(path, ["STAID", "split_role", "STATE", "HUC02", "area_class", "hydro_class"])


def test_load_split_assignment_rejects_missing_column(tmp_path):
    df = _assignment_df().drop(columns=["HUC02"])
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(SelectionError):
        load_split_assignment(path, ["STAID", "split_role", "STATE", "HUC02", "area_class", "hydro_class"])


def test_select_universe_excludes_california_and_holdout():
    universe = _universe()
    assert "00777001" not in universe.index  # spatial_holdout_nonca
    assert "00666001" not in universe.index  # california_finetune_train
    assert (universe["STATE"] != "CA").all()


def test_select_universe_rejects_missing_required_role():
    df = _assignment_df()
    df["split_role"] = "spatial_holdout_nonca"
    with pytest.raises(SelectionError):
        select_universe(df, _base_policy())


def test_select_universe_rejects_california_under_development_role():
    df = _assignment_df()
    df.loc[df["STAID"] == "00666001", "split_role"] = "development_train"
    with pytest.raises(SelectionError):
        select_universe(df, _base_policy())


# ---------------------------------------------------------------------------
# flag_unusual_identifier / bin_* helpers
# ---------------------------------------------------------------------------

def test_flag_unusual_identifier():
    assert flag_unusual_identifier("00100001", 8) is False
    assert flag_unusual_identifier("123456789", 8) is True
    assert flag_unusual_identifier("123456789012345", 8) is True


def test_bin_qobs_completeness():
    policy = _base_policy()
    assert bin_qobs_completeness(np.nan, policy) == "not_evaluated"
    assert bin_qobs_completeness(0.5, policy) == "low"
    assert bin_qobs_completeness(0.95, policy) == "mid"
    assert bin_qobs_completeness(0.999, policy) == "high"


def test_bin_static_missing():
    policy = _base_policy()
    assert bin_static_missing(np.nan, policy) == "not_evaluated"
    assert bin_static_missing(0, policy) == "none"
    assert bin_static_missing(3, policy) == "some"
    assert bin_static_missing(50, policy) == "high"


# ---------------------------------------------------------------------------
# build_reserved_selection
# ---------------------------------------------------------------------------

def test_build_reserved_selection_picks_unusual_and_hydro_gap():
    universe = _universe()
    reserved, reasons, category_log = build_reserved_selection(universe, _base_policy())
    assert "123456789" in reserved or "123456789012345" in reserved
    assert "00999001" in reserved  # the only hydro_class == "missing" basin
    assert reasons["00999001"] == ["hydro_stratifier_gap"]
    statuses = {e["category"]: e["status"] for e in category_log}
    assert statuses["unusual_identifier"] == "ok"
    assert statuses["hydro_stratifier_gap"] == "ok"
    assert statuses["static_missing_value_case"] == "not_evaluated_no_attributes_input"


def test_build_reserved_selection_graceful_when_category_absent():
    universe = _universe(include_edge_cases=False)
    reserved, reasons, category_log = build_reserved_selection(universe, _base_policy())
    assert reserved == []
    statuses = {e["category"]: e["status"] for e in category_log}
    assert statuses["unusual_identifier"] == "no_eligible_candidates"
    assert statuses["hydro_stratifier_gap"] == "no_eligible_candidates"


def test_build_reserved_selection_overlapping_predicates_covered_by_one_basin():
    # 223456789012345 satisfies BOTH unusual_identifier (15-char) and
    # hydro_stratifier_gap (hydro_class == "missing") -- mirrors the real
    # 2,307-basin pool, where all 5 real 15-char STAIDs are also
    # hydro_class == "missing". A parsimonious selector should reserve this
    # ONE basin for both categories, not spend a second pick on
    # 00999001 (hydro_stratifier_gap only) just because it also matches.
    universe = _universe(include_overlap_case=True)
    reserved, reasons, category_log = build_reserved_selection(universe, _base_policy())
    assert "223456789012345" in reserved
    assert reasons["223456789012345"] == ["hydro_stratifier_gap", "unusual_identifier"]
    log_by_category = {e["category"]: e for e in category_log}
    assert log_by_category["unusual_identifier"]["status"] == "ok"
    assert log_by_category["unusual_identifier"]["picked"] == ["223456789012345"]
    assert log_by_category["hydro_stratifier_gap"]["status"] == "covered_by_overlap"
    assert log_by_category["hydro_stratifier_gap"]["covered_by"] == "223456789012345"
    # 00999001 only matches hydro_stratifier_gap, which is already covered by
    # overlap -- it must not be spent as a redundant second reserved pick.
    assert "00999001" not in reserved


def test_build_reserved_selection_static_missing_gets_own_slot_when_not_covered():
    universe = _universe(include_overlap_case=True)
    missing_counts = pd.Series(0, index=universe.index, name="static_missing_model_input_count")
    ordinary_basin = sorted(universe.index)[0]  # an ordinary grid basin, not an edge case
    missing_counts.loc[ordinary_basin] = 3
    universe = universe.join(missing_counts)
    reserved, reasons, category_log = build_reserved_selection(universe, _base_policy())
    # static_missing_value_case is not satisfied by either already-reserved
    # overlap basin, so it must get its own reserved slot.
    assert ordinary_basin in reserved
    log_by_category = {e["category"]: e for e in category_log}
    assert log_by_category["static_missing_value_case"]["status"] == "ok"
    assert log_by_category["static_missing_value_case"]["covered_by"] is None
    assert log_by_category["static_missing_value_case"]["picked"] == [ordinary_basin]


# ---------------------------------------------------------------------------
# allocate_cell_quota
# ---------------------------------------------------------------------------

def test_allocate_cell_quota_proportional_sums_to_quota():
    cell_sizes = {("low", "low"): 10, ("low", "high"): 10, ("high", "high"): 10}
    quota, repair_log = allocate_cell_quota(cell_sizes, remaining_quota=9, min_per_nonempty_cell=1)
    assert sum(quota.values()) == 9
    assert all(v >= 1 for v in quota.values())


def test_allocate_cell_quota_trims_when_minimums_exceed_budget():
    cell_sizes = {("low", "low"): 5, ("low", "high"): 5, ("high", "high"): 5, ("high", "low"): 5}
    quota, repair_log = allocate_cell_quota(cell_sizes, remaining_quota=2, min_per_nonempty_cell=1)
    assert sum(quota.values()) == 2
    assert repair_log  # trimming step recorded


def test_allocate_cell_quota_rejects_over_capacity():
    with pytest.raises(SelectionError):
        allocate_cell_quota({("low", "low"): 3}, remaining_quota=5, min_per_nonempty_cell=1)


def test_allocate_cell_quota_zero_quota_zero_cells():
    quota, repair_log = allocate_cell_quota({}, remaining_quota=0, min_per_nonempty_cell=1)
    assert quota == {}
    assert repair_log == []


# ---------------------------------------------------------------------------
# select_within_cell / make_cell_rngs
# ---------------------------------------------------------------------------

def test_select_within_cell_deterministic_and_huc_breadth():
    cell_df = pd.DataFrame({
        "HUC02": ["01"] * 5 + ["02"] * 5,
    }, index=[f"S{i:02d}" for i in range(10)])
    rng1 = make_cell_rngs(42, [("low", "low")])[("low", "low")]
    rng2 = make_cell_rngs(42, [("low", "low")])[("low", "low")]
    picks1 = select_within_cell(cell_df, 4, rng1)
    picks2 = select_within_cell(cell_df, 4, rng2)
    assert picks1 == picks2  # deterministic given the same seed
    assert len(picks1) == 4
    hucs = {cell_df.loc[p, "HUC02"] for p in picks1}
    assert hucs == {"01", "02"}  # round-robin visits both HUC02 groups


def _five_huc_cell_df():
    hucs = ["01", "02", "03", "04", "05"]
    return pd.DataFrame(
        {"HUC02": [h for h in hucs for _ in range(3)]},
        index=[f"S{i:02d}" for i in range(15)],
    )


def test_select_within_cell_no_systematic_preference_for_early_huc_codes():
    # 5 HUC02 groups, quota 2 (< number of groups) -- the pathological case
    # from the original bug: fixed ascending-order visitation would ALWAYS
    # pick from {"01", "02"} regardless of seed. A seeded permutation must
    # surface other HUC02 pairs across a range of seeds.
    cell_df = _five_huc_cell_df()
    picked_huc_pairs = set()
    for seed in range(30):
        rng = make_cell_rngs(seed, [("low", "low")])[("low", "low")]
        picks = select_within_cell(cell_df, 2, rng)
        picked_huc_pairs.add(frozenset(cell_df.loc[p, "HUC02"] for p in picks))
    assert picked_huc_pairs != {frozenset({"01", "02"})}


def test_select_within_cell_identical_seed_is_byte_deterministic():
    cell_df = _five_huc_cell_df()
    for seed in (0, 1, 42, 12345):
        rng_a = make_cell_rngs(seed, [("low", "low")])[("low", "low")]
        rng_b = make_cell_rngs(seed, [("low", "low")])[("low", "low")]
        assert select_within_cell(cell_df, 2, rng_a) == select_within_cell(cell_df, 2, rng_b)


def test_select_within_cell_different_seed_changes_deterministic_output():
    cell_df = _five_huc_cell_df()
    outcomes = set()
    for seed in range(10):
        rng = make_cell_rngs(seed, [("low", "low")])[("low", "low")]
        outcomes.add(tuple(select_within_cell(cell_df, 2, rng)))
    # a different seed must be able to change the (still deterministic) output
    assert len(outcomes) > 1


# ---------------------------------------------------------------------------
# macro-region geographic diagnostics
# ---------------------------------------------------------------------------

def test_build_macro_region_map_covers_test_huc_codes():
    region_map = build_macro_region_map(_base_policy())
    for huc in HUC02S:
        assert huc in region_map


def test_build_macro_region_map_rejects_duplicate_huc_assignment():
    policy = _base_policy(geography_overrides={
        "macro_regions": {
            "region_a": ["01", "02"],
            "region_b": ["02", "03"],  # 02 listed twice -- contradictory policy
        },
    })
    with pytest.raises(SelectionError):
        build_macro_region_map(policy)


def test_macro_region_for_huc02_unmapped_code_raises():
    region_map = build_macro_region_map(_base_policy())
    with pytest.raises(SelectionError):
        macro_region_for_huc02("99", region_map)


def test_macro_region_side_east_west_other():
    policy = _base_policy()
    assert macro_region_side("northeast_mid_atlantic", policy) == "east"
    assert macro_region_side("colorado_great_basin", policy) == "west"
    assert macro_region_side("alaska_hawaii_other", policy) == "other"


# ---------------------------------------------------------------------------
# build_compact_selection -- full-pipeline acceptance checks
# ---------------------------------------------------------------------------

def test_build_compact_selection_deterministic_repeat():
    universe = _universe()
    policy = _base_policy()
    sel1, manifest1 = build_compact_selection(universe, None, policy)
    sel2, manifest2 = build_compact_selection(universe, None, policy)
    pd.testing.assert_frame_equal(sel1, sel2)
    assert manifest1["cell_quota"] == manifest2["cell_quota"]


def test_build_compact_selection_exact_target_count():
    universe = _universe()
    policy = _base_policy(target_count=15)
    sel, manifest = build_compact_selection(universe, None, policy)
    assert len(sel) == 15
    assert manifest["counts"]["n_selected"] == 15


def test_build_compact_selection_all_in_development_pool_no_leakage():
    assignment_df = _assignment_df()
    universe = select_universe(assignment_df, _base_policy())
    sel, _ = build_compact_selection(universe, None, _base_policy())
    dev_ids = set(universe.index)
    holdout_and_ca_ids = set(assignment_df.loc[
        assignment_df["split_role"] != "development_train", "STAID"
    ])
    assert set(sel["gauge_id"]).issubset(dev_ids)
    assert set(sel["gauge_id"]).isdisjoint(holdout_and_ca_ids)
    assert (sel["canonical_basin_role"] == "development_train").all()


def test_build_compact_selection_no_duplicate_gauge_ids():
    universe = _universe()
    sel, _ = build_compact_selection(universe, None, _base_policy())
    assert not sel["gauge_id"].duplicated().any()


def test_build_compact_selection_preserves_leading_zeros():
    universe = _universe()
    sel, _ = build_compact_selection(universe, None, _base_policy(target_count=40))
    zero_padded = [g for g in sel["gauge_id"] if g.startswith("00")]
    assert zero_padded  # at least one selected basin keeps its leading zeros


def test_build_compact_selection_area_and_hydro_class_representation():
    universe = _universe()
    sel, _ = build_compact_selection(universe, None, _base_policy(target_count=27))
    assert set(sel["area_class"]) >= {"low", "middle", "high"} - {"missing"}
    assert set(sel["hydro_class"]) & {"low", "middle", "high"} == {"low", "middle", "high"}


def test_build_compact_selection_huc_breadth_meets_soft_minimum():
    universe = _universe()
    policy = _base_policy(target_count=15)
    sel, manifest = build_compact_selection(universe, None, policy)
    assert manifest["counts"]["distinct_huc02"] >= policy["geography"]["distinct_huc02_soft_minimum"]
    assert manifest["distinct_huc02_soft_minimum_met"] is True


def test_build_compact_selection_east_and_west_represented_when_available():
    universe = _universe(include_edge_cases=False, include_leakage=False)
    policy = _base_policy(
        target_count=27, geography_overrides={"require_east_west_spread": True}
    )
    sel, manifest = build_compact_selection(universe, None, policy)
    assert manifest["east_west_breadth"]["n_east"] > 0
    assert manifest["east_west_breadth"]["n_west"] > 0
    assert set(sel["macro_region_side"]) >= {"east", "west"}


def test_build_compact_selection_raises_without_required_east_west_spread():
    universe = _universe(include_edge_cases=False, include_leakage=False)
    universe = universe.loc[universe["HUC02"].isin(["01", "02", "03"])]  # east only
    policy = _base_policy(
        target_count=9, geography_overrides={"require_east_west_spread": True}
    )
    with pytest.raises(SelectionError, match="east/west"):
        build_compact_selection(universe, None, policy)


def test_build_compact_selection_macro_region_counts_in_manifest():
    universe = _universe(include_edge_cases=False, include_leakage=False)
    policy = _base_policy(target_count=18)
    sel, manifest = build_compact_selection(universe, None, policy)
    assert sum(manifest["macro_region_counts"].values()) == len(sel)
    assert sum(manifest["macro_region_side_counts"].values()) == len(sel)
    assert manifest["distinct_macro_regions"] == sel["macro_region"].nunique()
    assert "east_west_breadth" in manifest
    assert set(manifest["east_west_breadth"].keys()) == {"n_east", "n_west", "required"}


def test_build_compact_selection_stable_ordering():
    universe = _universe()
    sel, _ = build_compact_selection(universe, None, _base_policy())
    assert list(sel["gauge_id"]) == sorted(sel["gauge_id"])


def test_build_compact_selection_includes_static_missing_case_when_present():
    universe = _universe()
    missing_counts = pd.Series(0, index=universe.index, name="static_missing_model_input_count")
    a_basin = sorted(universe.index)[0]
    missing_counts.loc[a_basin] = 4
    universe = universe.join(missing_counts)
    sel, manifest = build_compact_selection(universe, None, _base_policy())
    row = sel.loc[sel["gauge_id"] == a_basin]
    assert not row.empty
    assert "static_missing_value_case" in row.iloc[0]["selection_reason"]
    assert row.iloc[0]["static_missing_bin"] == "some"
    statuses = {e["category"]: e["status"] for e in manifest["reserved_category_log"]}
    assert statuses["static_missing_value_case"] == "ok"


def test_build_compact_selection_raises_when_population_too_small():
    universe = _universe().iloc[:5]
    with pytest.raises(SelectionError):
        build_compact_selection(universe, None, _base_policy(target_count=12))


def test_build_compact_selection_qobs_annotation_when_supplied():
    universe = _universe()
    qobs_df = pd.DataFrame(
        {"qobs_coverage_fraction": [0.5, 0.97]},
        index=pd.Index(sorted(universe.index)[:2], name="STAID"),
    )
    sel, _ = build_compact_selection(universe, qobs_df, _base_policy(target_count=30))
    annotated = sel.loc[sel["gauge_id"].isin(qobs_df.index)]
    assert (annotated["qobs_completeness_bin"] != "not_evaluated").all()
    not_annotated = sel.loc[~sel["gauge_id"].isin(qobs_df.index)]
    assert (not_annotated["qobs_completeness_bin"] == "not_evaluated").all()


# ---------------------------------------------------------------------------
# load_qobs_status
# ---------------------------------------------------------------------------

def test_load_qobs_status_accepts_coverage_fraction_column(tmp_path):
    df = pd.DataFrame({"STAID": ["00100001", "00100002"], "coverage_fraction": [0.5, 0.9], "target_status": ["A", "B"]})
    path = tmp_path / "target_status.csv"
    df.to_csv(path, index=False)
    out = load_qobs_status(path, _base_policy())
    assert out.loc["00100001", "qobs_coverage_fraction"] == 0.5
    assert out.loc["00100002", "target_status"] == "B"


def test_load_qobs_status_accepts_qobs_coverage_alias_column(tmp_path):
    df = pd.DataFrame({"STAID": ["00100001"], "qobs_coverage": [0.75]})
    path = tmp_path / "coverage.csv"
    df.to_csv(path, index=False)
    out = load_qobs_status(path, _base_policy())
    assert out.loc["00100001", "qobs_coverage_fraction"] == 0.75


def test_load_qobs_status_rejects_missing_coverage_column(tmp_path):
    df = pd.DataFrame({"STAID": ["00100001"], "some_other_column": [1]})
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(SelectionError):
        load_qobs_status(path, _base_policy())


# ---------------------------------------------------------------------------
# load_selection_policy portability checks
# ---------------------------------------------------------------------------

def test_load_selection_policy_rejects_absolute_path(tmp_path):
    policy = _base_policy()
    policy["some_path"] = "/data42/omrip/Flash-NH/tmp"
    path = tmp_path / "policy.yaml"
    import yaml
    path.write_text(yaml.safe_dump(policy), encoding="utf-8")
    with pytest.raises(SelectionError):
        load_selection_policy(path)


def test_load_selection_policy_rejects_credential_like_key(tmp_path):
    policy = _base_policy()
    policy["api_key"] = "not-a-real-secret"
    path = tmp_path / "policy.yaml"
    import yaml
    path.write_text(yaml.safe_dump(policy), encoding="utf-8")
    with pytest.raises(SelectionError):
        load_selection_policy(path)


def test_load_selection_policy_rejects_out_of_range_target_count(tmp_path):
    policy = _base_policy(target_count=999)
    path = tmp_path / "policy.yaml"
    import yaml
    path.write_text(yaml.safe_dump(policy), encoding="utf-8")
    with pytest.raises(SelectionError):
        load_selection_policy(path)


def test_load_selection_policy_loads_real_config():
    from pathlib import Path
    real_config = Path(__file__).resolve().parents[1] / "config" / "stage1_compact_package_selection_v001.yaml"
    policy = load_selection_policy(real_config)
    assert policy["selection_name"] == "stage1_compact_package_selection_v001"
    assert policy["min_target_count"] <= policy["target_count"] <= policy["max_target_count"]


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
    sel, manifest_pieces = build_compact_selection(universe, None, _base_policy())
    manifest = {"created_by": "test", "status": "candidate", **manifest_pieces}
    out_dir = tmp_path / "out"
    paths = write_selection_artifacts(out_dir, sel, manifest)

    expected_files = {
        "compact_basin_selection.csv", "compact_basin_ids.txt",
        "selection_summary.md", "selection_summary.json", "selection_manifest.json",
    }
    assert expected_files.issubset(paths.keys())

    written_manifest = json.loads(paths["selection_manifest.json"].read_text(encoding="utf-8"))
    for name, path in paths.items():
        if name == "selection_manifest.json":
            continue
        assert written_manifest["artifact_sha256"][name] == _sha256_of(path)

    ids_text = paths["compact_basin_ids.txt"].read_text(encoding="utf-8").splitlines()
    assert ids_text == sorted(sel["gauge_id"].tolist())

    csv_df = pd.read_csv(paths["compact_basin_selection.csv"], dtype={"gauge_id": str})
    assert set(csv_df["gauge_id"]) == set(sel["gauge_id"])
    assert (csv_df["gauge_id"].str.len() >= 8).all()


def test_write_selection_artifacts_refuses_nonempty_dir_without_force(tmp_path):
    universe = _universe()
    sel, manifest_pieces = build_compact_selection(universe, None, _base_policy())
    manifest = {"created_by": "test", "status": "candidate", **manifest_pieces}
    out_dir = tmp_path / "out"
    write_selection_artifacts(out_dir, sel, manifest)
    with pytest.raises(SelectionError):
        write_selection_artifacts(out_dir, sel, manifest)
    # force=True is allowed to overwrite
    write_selection_artifacts(out_dir, sel, manifest, force=True)

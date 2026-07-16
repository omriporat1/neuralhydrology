"""Tests for src/baseline/splits.py (Milestone 2K-G-I I-A2).

Structural/pooling tests exercise build_allocation_groups + select_holdout
directly with hand-crafted area_class/hydro_class columns -- this sidesteps
having to hand-predict real quantile edges for a synthetic population.
compute_tercile_edges/assign_tercile_class are tested separately with plain
numeric arrays. Full-pipeline (build_split_assignment) tests use a modest
synthetic population and assert invariants (coverage, no-overlap, CA
separation, missing-aridity handling, determinism, ID preservation,
holdout-fraction-in-a-loose-tolerance) rather than exact predicted group
membership, since real tercile edges depend on the whole population.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.baseline.splits import (
    REASON_DIRECT_STRATUM_SAMPLE,
    REASON_MISSING_STRATIFIER,
    REASON_SPARSE_POOL_FORCED_TRAINING,
    REASON_SPARSE_POOL_SAMPLE,
    AllocationGroup,
    SplitGenerationError,
    _assign_population_roles,
    assign_tercile_class,
    build_allocation_groups,
    build_split_assignment,
    compute_tercile_edges,
    join_eligible_with_matrix,
    load_eligible_basins,
    load_matrix_for_splits,
    make_split_rngs,
    select_holdout,
    sha256_of,
    write_split_artifacts,
)

# ---------------------------------------------------------------------------
# compute_tercile_edges / assign_tercile_class
# ---------------------------------------------------------------------------


def test_compute_tercile_edges_simple_1_to_9():
    edges = compute_tercile_edges(pd.Series(range(1, 10), dtype=float))
    assert edges == pytest.approx((3.6667, 6.3333), abs=1e-3)


def test_compute_tercile_edges_rejects_nulls():
    with pytest.raises(SplitGenerationError):
        compute_tercile_edges(pd.Series([1.0, np.nan, 3.0]))


def test_compute_tercile_edges_rejects_empty():
    with pytest.raises(SplitGenerationError):
        compute_tercile_edges(pd.Series([], dtype=float))


def test_assign_tercile_class_boundaries():
    edges = (3.0, 6.0)
    values = pd.Series([1.0, 3.0, 3.0001, 6.0, 6.0001, 9.0])
    out = assign_tercile_class(values, edges).tolist()
    assert out == ["low", "low", "middle", "middle", "high", "high"]


def test_assign_tercile_class_duplicate_edges_collapses_middle():
    # A mass point covering the whole population: e1 == e2.
    values = pd.Series([5.0, 5.0, 5.0, 5.0])
    edges = compute_tercile_edges(values)
    assert edges[0] == edges[1]
    out = assign_tercile_class(values, edges).unique().tolist()
    assert out == ["low"]  # every value <= e1 -> low; middle/high structurally empty


def test_assign_tercile_class_rejects_nulls():
    with pytest.raises(SplitGenerationError):
        assign_tercile_class(pd.Series([1.0, np.nan]), (1.0, 2.0))


# ---------------------------------------------------------------------------
# build_allocation_groups / select_holdout (structural, hand-crafted classes)
# ---------------------------------------------------------------------------


def _frame(rows):
    return pd.DataFrame(rows)


def test_sufficient_stratum_stays_its_own_group():
    rows = [
        {"STAID": f"S{i:03d}", "HUC02": "02", "area_class": "low", "hydro_class": "low"}
        for i in range(12)
    ]
    groups, fallback = build_allocation_groups(
        _frame(rows), staid_col="STAID",
        stratum_cols=["HUC02", "area_class", "hydro_class"], pool_key_col="HUC02",
        min_stratum_size=10,
    )
    assert fallback == []
    assert len(groups) == 1
    assert groups[0].kind == "stratum"
    assert groups[0].group_id == "stratum:02:low:low"
    assert set(groups[0].staids) == {r["STAID"] for r in rows}


def test_sparse_strata_pooled_within_huc02_when_pool_sufficient():
    rows = []
    for cls, n in [("low", 5), ("middle", 5), ("high", 5)]:
        rows += [
            {"STAID": f"{cls}{i}", "HUC02": "06", "area_class": cls, "hydro_class": cls}
            for i in range(n)
        ]
    groups, fallback = build_allocation_groups(
        _frame(rows), staid_col="STAID",
        stratum_cols=["HUC02", "area_class", "hydro_class"], pool_key_col="HUC02",
        min_stratum_size=10,
    )
    assert fallback == []
    assert len(groups) == 1
    assert groups[0].kind == "sparse_pool"
    assert groups[0].group_id == "sparse_pool:06"
    assert len(groups[0].staids) == 15


def test_sparse_pool_below_min_size_forces_fallback_no_group():
    rows = [
        {"STAID": f"S{i}", "HUC02": "07", "area_class": "low", "hydro_class": "low"}
        for i in range(7)
    ]
    groups, fallback = build_allocation_groups(
        _frame(rows), staid_col="STAID",
        stratum_cols=["HUC02", "area_class", "hydro_class"], pool_key_col="HUC02",
        min_stratum_size=10,
    )
    assert groups == []
    assert len(fallback) == 7
    assert {e["reason"] for e in fallback} == {"sparse_pool_below_min_stratum_size"}
    assert {e["pool_key"] for e in fallback} == {"07"}


def test_huc02_singleton_forced_to_fallback():
    rows = [{"STAID": "SINGLE", "HUC02": "09", "area_class": "low", "hydro_class": "low"}]
    groups, fallback = build_allocation_groups(
        _frame(rows), staid_col="STAID",
        stratum_cols=["HUC02", "area_class", "hydro_class"], pool_key_col="HUC02",
        min_stratum_size=10,
    )
    assert groups == []
    assert len(fallback) == 1
    assert fallback[0]["staid"] == "SINGLE"
    assert fallback[0]["pool_size"] == 1


def test_huc02_strings_10L_10U_preserved_in_group_id():
    rows = [
        {"STAID": f"L{i}", "HUC02": "10L", "area_class": "low", "hydro_class": "low"}
        for i in range(12)
    ] + [
        {"STAID": f"U{i}", "HUC02": "10U", "area_class": "low", "hydro_class": "low"}
        for i in range(12)
    ]
    groups, fallback = build_allocation_groups(
        _frame(rows), staid_col="STAID",
        stratum_cols=["HUC02", "area_class", "hydro_class"], pool_key_col="HUC02",
        min_stratum_size=10,
    )
    assert fallback == []
    group_ids = sorted(g.group_id for g in groups)
    assert group_ids == ["stratum:10L:low:low", "stratum:10U:low:low"]


def test_sufficient_stratum_not_downgraded_by_sibling_sparse_stratum():
    rows = [
        {"STAID": f"suff{i}", "HUC02": "04", "area_class": "low", "hydro_class": "low"}
        for i in range(12)
    ] + [
        {"STAID": f"sp{i}", "HUC02": "04", "area_class": "high", "hydro_class": "high"}
        for i in range(3)
    ]
    groups, fallback = build_allocation_groups(
        _frame(rows), staid_col="STAID",
        stratum_cols=["HUC02", "area_class", "hydro_class"], pool_key_col="HUC02",
        min_stratum_size=10,
    )
    kinds = {g.group_id: g.kind for g in groups}
    assert kinds["stratum:04:low:low"] == "stratum"
    assert len(next(g for g in groups if g.group_id == "stratum:04:low:low").staids) == 12
    # the 3 sparse basins fell below min size even after pooling (only 3 in the HUC02's sparse set)
    assert len(fallback) == 3
    assert {e["staid"] for e in fallback} == {f"sp{i}" for i in range(3)}


def test_global_pool_when_pool_key_col_is_none():
    # Each basin its own singleton cell -> all sparse -> pooled into one group.
    rows = [
        {"STAID": f"C{i}", "area_class": f"a{i}", "hydro_class": f"h{i}"}
        for i in range(12)
    ]
    groups, fallback = build_allocation_groups(
        _frame(rows), staid_col="STAID",
        stratum_cols=["area_class", "hydro_class"], pool_key_col=None,
        min_stratum_size=10,
    )
    assert fallback == []
    assert len(groups) == 1
    assert groups[0].group_id == "sparse_pool:ALL"
    assert len(groups[0].staids) == 12


def test_global_pool_below_min_size_forces_fallback():
    rows = [{"STAID": f"C{i}", "area_class": f"a{i}", "hydro_class": f"h{i}"} for i in range(6)]
    groups, fallback = build_allocation_groups(
        _frame(rows), staid_col="STAID",
        stratum_cols=["area_class", "hydro_class"], pool_key_col=None,
        min_stratum_size=10,
    )
    assert groups == []
    assert len(fallback) == 6


def test_select_holdout_deterministic_same_seed():
    group = AllocationGroup(group_id="g1", kind="stratum", staids=tuple(f"S{i}" for i in range(40)))
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    a1 = select_holdout([group], rng1, 0.10)
    a2 = select_holdout([group], rng2, 0.10)
    assert a1 == a2
    holdout_count = sum(1 for v in a1.values() if v["is_holdout"])
    assert holdout_count == round(0.10 * 40)


def test_select_holdout_different_seed_changes_selection():
    group = AllocationGroup(group_id="g1", kind="stratum", staids=tuple(f"S{i}" for i in range(40)))
    a1 = select_holdout([group], np.random.default_rng(42), 0.10)
    a2 = select_holdout([group], np.random.default_rng(43), 0.10)
    holdout1 = {s for s, v in a1.items() if v["is_holdout"]}
    holdout2 = {s for s, v in a2.items() if v["is_holdout"]}
    assert holdout1 != holdout2


def test_select_holdout_rounding_matches_python_round():
    group = AllocationGroup(group_id="g1", kind="sparse_pool", staids=tuple(f"S{i}" for i in range(15)))
    a = select_holdout([group], np.random.default_rng(1), 0.10)
    holdout_count = sum(1 for v in a.values() if v["is_holdout"])
    assert holdout_count == round(0.10 * 15)


def test_make_split_rngs_deterministic_and_independent():
    n1, c1 = make_split_rngs(42)
    n2, c2 = make_split_rngs(42)
    assert n1.integers(0, 1_000_000, size=5).tolist() == n2.integers(0, 1_000_000, size=5).tolist()
    n3, c3 = make_split_rngs(42)
    n4, c4 = make_split_rngs(43)
    assert n3.integers(0, 1_000_000, size=5).tolist() != n4.integers(0, 1_000_000, size=5).tolist()
    n5, c5 = make_split_rngs(42)
    assert n5.integers(0, 1_000_000, size=5).tolist() != c5.integers(0, 1_000_000, size=5).tolist()


def test_sha256_of_matches_manual_hash(tmp_path):
    import hashlib
    p = tmp_path / "f.txt"
    p.write_text("hello world", encoding="utf-8")
    assert sha256_of(p) == hashlib.sha256(b"hello world").hexdigest()


# ---------------------------------------------------------------------------
# load_eligible_basins
# ---------------------------------------------------------------------------


def test_load_eligible_basins_normalizes_and_sorts(tmp_path):
    p = tmp_path / "eligible.txt"
    p.write_text("1019000\n00000123\n9484000\n", encoding="utf-8")
    result = load_eligible_basins(p)
    assert result == sorted(["01019000", "00000123", "09484000"])


def test_load_eligible_basins_preserves_nonstandard_lengths(tmp_path):
    p = tmp_path / "eligible.txt"
    p.write_text("103366092\n393109104464500\n01019000\n", encoding="utf-8")
    result = load_eligible_basins(p)
    assert "103366092" in result
    assert "393109104464500" in result


def test_load_eligible_basins_rejects_malformed_staid(tmp_path):
    p = tmp_path / "eligible.txt"
    p.write_text("1019000\nABCDEFGH\n", encoding="utf-8")
    with pytest.raises(SplitGenerationError):
        load_eligible_basins(p)


def test_load_eligible_basins_rejects_duplicate_after_normalization(tmp_path):
    p = tmp_path / "eligible.txt"
    p.write_text("1\n00000001\n", encoding="utf-8")
    with pytest.raises(SplitGenerationError, match="duplicate"):
        load_eligible_basins(p)


def test_load_eligible_basins_rejects_empty_file(tmp_path):
    p = tmp_path / "eligible.txt"
    p.write_text("   \n\n", encoding="utf-8")
    with pytest.raises(SplitGenerationError, match="empty"):
        load_eligible_basins(p)


def test_load_eligible_basins_rejects_missing_file(tmp_path):
    with pytest.raises(SplitGenerationError, match="not found"):
        load_eligible_basins(tmp_path / "nope.txt")


# ---------------------------------------------------------------------------
# load_matrix_for_splits / join_eligible_with_matrix
# ---------------------------------------------------------------------------


def _write_parquet(tmp_path, df, name="matrix.parquet"):
    p = tmp_path / name
    df.to_parquet(p)
    return p


def test_load_matrix_for_splits_happy_path(tmp_path):
    df = pd.DataFrame({
        "gauge_id": ["01019000", "01019001"],
        "STATE": ["ME", "ME"],
        "HUC02": ["01", "01"],
        "DRAIN_SQKM": [10.0, 20.0],
        "ari_ix_uav": [0.5, 0.6],
    }).set_index("gauge_id")
    p = _write_parquet(tmp_path, df)
    loaded = load_matrix_for_splits(p, ["STATE", "HUC02", "DRAIN_SQKM", "ari_ix_uav"])
    assert loaded.index.name == "gauge_id"
    assert list(loaded.index) == ["01019000", "01019001"]


def test_load_matrix_for_splits_rejects_missing_required_column(tmp_path):
    df = pd.DataFrame({
        "gauge_id": ["01019000"], "STATE": ["ME"], "HUC02": ["01"], "DRAIN_SQKM": [10.0],
    }).set_index("gauge_id")
    p = _write_parquet(tmp_path, df)
    with pytest.raises(SplitGenerationError, match="missing required columns"):
        load_matrix_for_splits(p, ["STATE", "HUC02", "DRAIN_SQKM", "ari_ix_uav"])


def test_load_matrix_for_splits_rejects_numeric_gauge_id(tmp_path):
    df = pd.DataFrame({
        "gauge_id": [1019000, 1019001], "STATE": ["ME", "ME"],
    }).set_index("gauge_id")
    p = _write_parquet(tmp_path, df)
    with pytest.raises(SplitGenerationError, match="numeric"):
        load_matrix_for_splits(p, ["STATE"])


def test_load_matrix_for_splits_rejects_duplicate_gauge_id(tmp_path):
    df = pd.DataFrame({
        "gauge_id": ["01019000", "01019000"], "STATE": ["ME", "ME"],
    }).set_index("gauge_id")
    p = _write_parquet(tmp_path, df)
    with pytest.raises(SplitGenerationError, match="duplicate"):
        load_matrix_for_splits(p, ["STATE"])


def test_join_eligible_with_matrix_happy_path():
    matrix = pd.DataFrame(
        {"STATE": ["ME", "ME", "CA"]},
        index=pd.Index(["01019000", "01019001", "11000000"], name="gauge_id"),
    )
    joined = join_eligible_with_matrix(matrix, ["01019000", "11000000"])
    assert list(joined.index) == ["01019000", "11000000"]


def test_join_eligible_with_matrix_rejects_failed_join():
    matrix = pd.DataFrame({"STATE": ["ME"]}, index=pd.Index(["01019000"], name="gauge_id"))
    with pytest.raises(SplitGenerationError, match="missing from matrix"):
        join_eligible_with_matrix(matrix, ["01019000", "99999999"])


def test_join_eligible_with_matrix_rejects_duplicate_input():
    matrix = pd.DataFrame({"STATE": ["ME"]}, index=pd.Index(["01019000"], name="gauge_id"))
    with pytest.raises(SplitGenerationError, match="duplicate"):
        join_eligible_with_matrix(matrix, ["01019000", "01019000"])


# ---------------------------------------------------------------------------
# _assign_population_roles: missing-aridity handling (Option B), in isolation
# ---------------------------------------------------------------------------


def _small_population_frame(n_with_aridity=9, n_missing=2, state="ME", huc02="01"):
    rows = {}
    for i in range(n_with_aridity):
        rows[f"K{i:02d}"] = {
            "STATE": state, "HUC02": huc02,
            "DRAIN_SQKM": float(i + 1), "ari_ix_uav": float(i + 1),
        }
    for i in range(n_missing):
        rows[f"M{i:02d}"] = {
            "STATE": state, "HUC02": huc02, "DRAIN_SQKM": 100.0 + i, "ari_ix_uav": np.nan,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def test_missing_aridity_basins_forced_to_training_never_holdout():
    frame = _small_population_frame()
    rng = np.random.default_rng(42)
    records, edges, fallback = _assign_population_roles(
        frame, rng, holdout_fraction=0.10,
        holdout_role="spatial_holdout_nonca", training_role="development_train",
        use_geography=True, min_stratum_size=10,
    )
    by_staid = {r["STAID"]: r for r in records}
    for i in range(2):
        rec = by_staid[f"M{i:02d}"]
        assert rec["split_role"] == "development_train"
        assert rec["assignment_reason"] == REASON_MISSING_STRATIFIER
        assert rec["stratum_id"] == REASON_MISSING_STRATIFIER
        assert rec["hydro_class"] == "missing"


def test_missing_aridity_never_selected_across_many_seeds():
    frame = _small_population_frame(n_with_aridity=9, n_missing=3)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        records, _, _ = _assign_population_roles(
            frame, rng, holdout_fraction=0.10,
            holdout_role="spatial_holdout_nonca", training_role="development_train",
            use_geography=True, min_stratum_size=10,
        )
        for r in records:
            if r["assignment_reason"] == REASON_MISSING_STRATIFIER:
                assert r["split_role"] != "spatial_holdout_nonca"


def test_assign_population_roles_returns_tercile_edges():
    frame = _small_population_frame(n_with_aridity=9, n_missing=0)
    rng = np.random.default_rng(42)
    _, edges, _ = _assign_population_roles(
        frame, rng, holdout_fraction=0.10,
        holdout_role="spatial_holdout_nonca", training_role="development_train",
        use_geography=True, min_stratum_size=10,
    )
    assert edges["area_edges"] is not None
    assert edges["aridity_edges"] is not None


# ---------------------------------------------------------------------------
# Full-pipeline integration tests (build_split_assignment)
# ---------------------------------------------------------------------------

_NONCA_HUC02S = ["01", "02", "03", "04", "05"]


def _build_integration_population():
    rows = {}
    rng = np.random.default_rng(123456)

    for huc02 in _NONCA_HUC02S:
        n = 30
        areas = rng.uniform(10, 1000, n)
        aridities = rng.uniform(0.0, 1.0, n)
        for i in range(n):
            staid = f"{huc02}{i:06d}"
            rows[staid] = {
                "STATE": "TX" if huc02 != "05" else "OK",
                "HUC02": huc02,
                "DRAIN_SQKM": float(areas[i]),
                "ari_ix_uav": float(aridities[i]),
            }

    # 5 non-CA missing-aridity basins.
    for i in range(5):
        staid = f"99{i:06d}"
        rows[staid] = {"STATE": "TX", "HUC02": "01", "DRAIN_SQKM": 250.0 + i, "ari_ix_uav": np.nan}

    # HUC02 with non-standard-length STAIDs mixed into the population.
    rows["103366092"] = {"STATE": "TX", "HUC02": "02", "DRAIN_SQKM": 333.0, "ari_ix_uav": 0.33}
    rows["393109104464500"] = {"STATE": "TX", "HUC02": "03", "DRAIN_SQKM": 444.0, "ari_ix_uav": 0.44}

    # HUC02 strings with L/U suffixes, deliberately small (exercise fallback,
    # but the assertion of interest is that "10L"/"10U" survive verbatim).
    for i in range(4):
        rows[f"10L{i:05d}"] = {"STATE": "TX", "HUC02": "10L", "DRAIN_SQKM": 50.0 + i, "ari_ix_uav": 0.1 + 0.01 * i}
    for i in range(4):
        rows[f"10U{i:05d}"] = {"STATE": "TX", "HUC02": "10U", "DRAIN_SQKM": 60.0 + i, "ari_ix_uav": 0.2 + 0.01 * i}

    # California population.
    ca_rng = np.random.default_rng(654321)
    n_ca = 40
    ca_areas = ca_rng.uniform(10, 1000, n_ca)
    ca_aridities = ca_rng.uniform(0.0, 1.0, n_ca)
    for i in range(n_ca):
        staid = f"CA{i:06d}"
        rows[staid] = {"STATE": "CA", "HUC02": "18", "DRAIN_SQKM": float(ca_areas[i]), "ari_ix_uav": float(ca_aridities[i])}
    # 1 CA missing-aridity basin.
    rows["CA999999"] = {"STATE": "CA", "HUC02": "18", "DRAIN_SQKM": 500.0, "ari_ix_uav": np.nan}

    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "gauge_id"
    return frame


@pytest.fixture(scope="module")
def integration_population():
    return _build_integration_population()


def test_every_basin_assigned_exactly_once(integration_population):
    assignment_df, _ = build_split_assignment(
        integration_population, seed=42, nonca_holdout_fraction=0.10, ca_holdout_fraction=0.10,
    )
    assert len(assignment_df) == len(integration_population)
    assert assignment_df["STAID"].duplicated().sum() == 0
    assert set(assignment_df["STAID"]) == set(integration_population.index)


def test_no_overlap_training_and_holdout(integration_population):
    assignment_df, _ = build_split_assignment(
        integration_population, seed=42, nonca_holdout_fraction=0.10, ca_holdout_fraction=0.10,
    )
    dev = set(assignment_df.loc[assignment_df["split_role"] == "development_train", "STAID"])
    holdout = set(assignment_df.loc[assignment_df["split_role"] == "spatial_holdout_nonca", "STAID"])
    assert dev.isdisjoint(holdout)


def test_california_separation(integration_population):
    assignment_df, _ = build_split_assignment(
        integration_population, seed=42, nonca_holdout_fraction=0.10, ca_holdout_fraction=0.10,
    )
    ca_rows = assignment_df.loc[assignment_df["STATE"] == "CA"]
    nonca_rows = assignment_df.loc[assignment_df["STATE"] != "CA"]
    assert set(ca_rows["split_role"]) <= {"california_finetune_train", "california_holdout"}
    assert set(nonca_rows["split_role"]) <= {"development_train", "spatial_holdout_nonca"}


def test_huc02_10L_10U_preserved_in_assignment_output(integration_population):
    assignment_df, _ = build_split_assignment(
        integration_population, seed=42, nonca_holdout_fraction=0.10, ca_holdout_fraction=0.10,
    )
    huc02_values = set(assignment_df["HUC02"].astype(str))
    assert "10L" in huc02_values
    assert "10U" in huc02_values
    assert "10" not in {v for v in huc02_values if v in ("10L", "10U")}  # sanity: distinct strings


def test_nonstandard_staids_preserved(integration_population):
    assignment_df, _ = build_split_assignment(
        integration_population, seed=42, nonca_holdout_fraction=0.10, ca_holdout_fraction=0.10,
    )
    staids = set(assignment_df["STAID"])
    assert "103366092" in staids
    assert "393109104464500" in staids


def test_missing_aridity_basins_forced_training_full_pipeline(integration_population):
    assignment_df, _ = build_split_assignment(
        integration_population, seed=42, nonca_holdout_fraction=0.10, ca_holdout_fraction=0.10,
    )
    missing = assignment_df.loc[assignment_df["assignment_reason"] == REASON_MISSING_STRATIFIER]
    assert len(missing) == 6  # 5 non-CA + 1 CA
    assert set(missing["split_role"]) == {"development_train", "california_finetune_train"}
    assert "spatial_holdout_nonca" not in set(missing["split_role"])
    assert "california_holdout" not in set(missing["split_role"])


def test_assignment_reason_values_are_the_four_concise_codes(integration_population):
    assignment_df, _ = build_split_assignment(
        integration_population, seed=42, nonca_holdout_fraction=0.10, ca_holdout_fraction=0.10,
    )
    observed = set(assignment_df["assignment_reason"])
    assert observed <= {
        REASON_MISSING_STRATIFIER,
        REASON_DIRECT_STRATUM_SAMPLE,
        REASON_SPARSE_POOL_SAMPLE,
        REASON_SPARSE_POOL_FORCED_TRAINING,
    }
    assert REASON_MISSING_STRATIFIER in observed


def test_manifest_pieces_include_huc02_role_counts_and_missing_stratifier_basins(integration_population):
    assignment_df, manifest = build_split_assignment(
        integration_population, seed=42, nonca_holdout_fraction=0.10, ca_holdout_fraction=0.10,
    )
    assert manifest["algorithm_version"] == 1

    missing_info = manifest["missing_stratifier_basins"]
    assert missing_info["reason"] == REASON_MISSING_STRATIFIER
    assert missing_info["count"] == 6
    assert set(missing_info["staids"]) == set(
        assignment_df.loc[assignment_df["assignment_reason"] == REASON_MISSING_STRATIFIER, "STAID"]
    )

    huc02_counts = manifest["huc02_role_counts"]
    huc02_01_dev = assignment_df.loc[
        (assignment_df["HUC02"] == "01") & (assignment_df["split_role"] == "development_train")
    ]
    assert huc02_counts["01"]["development_train"] == len(huc02_01_dev)


def test_determinism_same_seed_byte_identical(integration_population):
    a1, m1 = build_split_assignment(
        integration_population, seed=42, nonca_holdout_fraction=0.10, ca_holdout_fraction=0.10,
    )
    a2, m2 = build_split_assignment(
        integration_population, seed=42, nonca_holdout_fraction=0.10, ca_holdout_fraction=0.10,
    )
    assert a1.sort_values("STAID").reset_index(drop=True).equals(
        a2.sort_values("STAID").reset_index(drop=True)
    )
    assert m1["tercile_edges"] == m2["tercile_edges"]
    assert m1["counts"] == m2["counts"]


def test_different_seed_changes_holdout_selection(integration_population):
    a1, _ = build_split_assignment(
        integration_population, seed=42, nonca_holdout_fraction=0.10, ca_holdout_fraction=0.10,
    )
    a2, _ = build_split_assignment(
        integration_population, seed=43, nonca_holdout_fraction=0.10, ca_holdout_fraction=0.10,
    )
    holdout1 = set(a1.loc[a1["split_role"].isin(["spatial_holdout_nonca", "california_holdout"]), "STAID"])
    holdout2 = set(a2.loc[a2["split_role"].isin(["spatial_holdout_nonca", "california_holdout"]), "STAID"])
    assert holdout1 != holdout2


def test_holdout_fraction_within_loose_tolerance(integration_population):
    _, manifest = build_split_assignment(
        integration_population, seed=42, nonca_holdout_fraction=0.10, ca_holdout_fraction=0.10,
    )
    nonca_frac = manifest["resulting_fractions"]["nonca_holdout_of_nonca"]
    ca_frac = manifest["resulting_fractions"]["ca_holdout_of_ca"]
    assert 0.03 <= nonca_frac <= 0.20
    assert 0.03 <= ca_frac <= 0.25


def test_rejects_when_state_malformed(integration_population):
    bad = integration_population.copy()
    bad.loc[bad.index[0], "STATE"] = "ZZ"
    with pytest.raises(SplitGenerationError, match="STATE"):
        build_split_assignment(bad, seed=42, nonca_holdout_fraction=0.10, ca_holdout_fraction=0.10)


def test_rejects_nonpositive_area(integration_population):
    bad = integration_population.copy()
    bad.loc[bad.index[0], "DRAIN_SQKM"] = 0.0
    with pytest.raises(SplitGenerationError, match="DRAIN_SQKM"):
        build_split_assignment(bad, seed=42, nonca_holdout_fraction=0.10, ca_holdout_fraction=0.10)


def test_rejects_nonfinite_area(integration_population):
    bad = integration_population.copy()
    bad.loc[bad.index[0], "DRAIN_SQKM"] = np.inf
    with pytest.raises(SplitGenerationError, match="DRAIN_SQKM"):
        build_split_assignment(bad, seed=42, nonca_holdout_fraction=0.10, ca_holdout_fraction=0.10)


def test_rejects_nonfinite_aridity(integration_population):
    bad = integration_population.copy()
    bad.loc[bad.index[0], "ari_ix_uav"] = np.inf
    with pytest.raises(SplitGenerationError, match="ari_ix_uav"):
        build_split_assignment(bad, seed=42, nonca_holdout_fraction=0.10, ca_holdout_fraction=0.10)


# ---------------------------------------------------------------------------
# write_split_artifacts
# ---------------------------------------------------------------------------


def test_write_split_artifacts_deterministic_checksums(tmp_path, integration_population):
    assignment_df, manifest_pieces = build_split_assignment(
        integration_population, seed=42, nonca_holdout_fraction=0.10, ca_holdout_fraction=0.10,
    )
    eligible_all = sorted(integration_population.index.tolist())
    manifest = {"seed": 42, **manifest_pieces}

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    paths1 = write_split_artifacts(out1, eligible_all, assignment_df, manifest)
    paths2 = write_split_artifacts(out2, eligible_all, assignment_df, manifest)

    assert set(paths1.keys()) == set(paths2.keys())
    for name in paths1:
        assert sha256_of(paths1[name]) == sha256_of(paths2[name])


def test_write_split_artifacts_temporal_lists_byte_identical(tmp_path, integration_population):
    assignment_df, manifest_pieces = build_split_assignment(
        integration_population, seed=42, nonca_holdout_fraction=0.10, ca_holdout_fraction=0.10,
    )
    eligible_all = sorted(integration_population.index.tolist())
    paths = write_split_artifacts(tmp_path / "run", eligible_all, assignment_df, manifest_pieces)
    sha_dev = sha256_of(paths["development_train.txt"])
    sha_val = sha256_of(paths["validation.txt"])
    sha_test = sha256_of(paths["temporal_test.txt"])
    assert sha_dev == sha_val == sha_test


def test_write_split_artifacts_refuses_nonempty_dir_without_force(tmp_path, integration_population):
    assignment_df, manifest_pieces = build_split_assignment(
        integration_population, seed=42, nonca_holdout_fraction=0.10, ca_holdout_fraction=0.10,
    )
    eligible_all = sorted(integration_population.index.tolist())
    out_dir = tmp_path / "run"
    write_split_artifacts(out_dir, eligible_all, assignment_df, manifest_pieces)
    with pytest.raises(SplitGenerationError, match="force"):
        write_split_artifacts(out_dir, eligible_all, assignment_df, manifest_pieces)
    # succeeds with force
    write_split_artifacts(out_dir, eligible_all, assignment_df, manifest_pieces, force=True)


def test_write_split_artifacts_manifest_records_artifact_checksums_noncircularly(
    tmp_path, integration_population
):
    assignment_df, manifest_pieces = build_split_assignment(
        integration_population, seed=42, nonca_holdout_fraction=0.10, ca_holdout_fraction=0.10,
    )
    eligible_all = sorted(integration_population.index.tolist())
    paths = write_split_artifacts(tmp_path / "run", eligible_all, assignment_df, manifest_pieces)

    written_manifest = json.loads(paths["split_manifest.json"].read_text(encoding="utf-8"))
    artifact_sha256 = written_manifest["artifact_sha256"]

    assert "split_manifest.json" not in artifact_sha256
    assert set(artifact_sha256) == {name for name in paths if name != "split_manifest.json"}
    for name, expected_sha in artifact_sha256.items():
        assert sha256_of(paths[name]) == expected_sha


def test_write_split_artifacts_eligible_list_matches_input(tmp_path, integration_population):
    assignment_df, manifest_pieces = build_split_assignment(
        integration_population, seed=42, nonca_holdout_fraction=0.10, ca_holdout_fraction=0.10,
    )
    eligible_all = sorted(integration_population.index.tolist())
    paths = write_split_artifacts(tmp_path / "run", eligible_all, assignment_df, manifest_pieces)
    written = paths["eligible_basins_v001.txt"].read_text(encoding="utf-8").splitlines()
    assert written == eligible_all

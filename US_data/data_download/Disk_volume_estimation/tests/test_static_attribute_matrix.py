"""Tests for the 2026-07-20 static-attribute semantic correction: sentinel
decoding, role reclassification, and the strengthened independent auditor
(scripts/build_stage1_static_attribute_matrix.py,
scripts/audit_stage1_static_attribute_matrix.py). Synthetic fixtures only --
no h2o/real data required. See docs/decision_log.md (2026-07-20 entry) for
the binding decisions this codifies.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import scripts.audit_stage1_static_attribute_matrix as auditor
import scripts.build_stage1_static_attribute_matrix as builder


# ---------------------------------------------------------------------------
# Part 1 -- sentinel decoding (builder._decode_column_sentinels)
# ---------------------------------------------------------------------------

def test_mapped_sentinel_converted_to_nan():
    counts: dict[str, int] = {}
    series = pd.Series(["1.5", "-999", "3.2", "-999"], index=["a", "b", "c", "d"])
    out = builder._decode_column_sentinels(series, "RAW_DIS_NEAREST_DAM", counts)
    assert pd.isna(out["b"]) and pd.isna(out["d"])
    assert out["a"] == "1.5" and out["c"] == "3.2"
    assert counts["RAW_DIS_NEAREST_DAM"] == 2


def test_non_sentinel_values_unchanged():
    counts: dict[str, int] = {}
    series = pd.Series(["-998", "-999.0", "0", ""])
    out = builder._decode_column_sentinels(series, "STRAHLER_MAX", counts)
    # -99 is the mapped sentinel for STRAHLER_MAX, not -998 or -999.
    assert out[0] == "-998"
    assert out[1] == "-999.0"
    assert out[2] == "0"
    assert counts["STRAHLER_MAX"] == 0


def test_unmapped_column_untouched_even_with_sentinel_looking_values():
    counts: dict[str, int] = {}
    series = pd.Series(["-999", "-9999", "-99"])
    out = builder._decode_column_sentinels(series, "SOME_UNRELATED_COLUMN", counts)
    pd.testing.assert_series_equal(out, series)
    assert "SOME_UNRELATED_COLUMN" not in counts


def test_zero_replacement_count_recorded_not_omitted():
    counts: dict[str, int] = {}
    series = pd.Series(["1.0", "2.0", "3.0"])
    builder._decode_column_sentinels(series, "PERHOR", counts)
    assert counts == {"PERHOR": 0}


def test_each_mapped_column_decodes_its_own_sentinel():
    for col, sentinels in builder._SENTINEL_VALUES_BY_COLUMN.items():
        counts: dict[str, int] = {}
        sentinel_val = sorted(sentinels)[0]
        series = pd.Series([str(sentinel_val), "1.0"])
        out = builder._decode_column_sentinels(series, col, counts)
        assert pd.isna(out[0]), f"{col}: sentinel {sentinel_val} not decoded"
        assert out[1] == "1.0"
        assert counts[col] == 1


def test_nonnumeric_value_in_mapped_column_fails_loud():
    counts: dict[str, int] = {}
    series = pd.Series(["1.5", "not_a_number", "-999"])
    with pytest.raises(SystemExit):
        builder._decode_column_sentinels(series, "RAW_DIS_NEAREST_DAM", counts)


def test_blank_values_in_mapped_column_pass_through_as_missing():
    counts: dict[str, int] = {}
    series = pd.Series(["1.5", "", "-999"])
    out = builder._decode_column_sentinels(series, "RAW_DIS_NEAREST_DAM", counts)
    assert out[1] == ""  # untouched by sentinel decode; downstream to_numeric -> NaN
    assert counts["RAW_DIS_NEAREST_DAM"] == 1


# ---------------------------------------------------------------------------
# Part 2 -- role classification (builder._classify_columns)
# ---------------------------------------------------------------------------

def test_coordinates_classified_diagnostic_latlon():
    roles = builder._classify_columns("attributes_gageii_BasinID.csv", ["LAT_GAGE", "LNG_GAGE"])
    assert roles == {"LAT_GAGE": "diagnostic_latlon", "LNG_GAGE": "diagnostic_latlon"}
    roles2 = builder._classify_columns("attributes_gageii_Bas_Morph.csv", ["LAT_CENT", "LONG_CENT"])
    assert roles2 == {"LAT_CENT": "diagnostic_latlon", "LONG_CENT": "diagnostic_latlon"}


def test_record_network_qa_fields_not_model_input():
    cols = sorted(builder._DIAGNOSTIC_RECORD_NETWORK_QA)
    roles = builder._classify_columns("attributes_gageii_FlowRec.csv", cols)
    for c in cols:
        assert roles[c] == "diagnostic_record_network_qa", f"{c} classified as {roles[c]!r}"


def test_former_binary_flags_route_to_record_network_qa_not_binary_flag():
    # HCDN_2009/HBN36/OLD_HCDN/NSIP_SENTINEL/ACTIVE09 are members of both
    # _BINARY_FLAGS (encoding-style hint) and _DIAGNOSTIC_RECORD_NETWORK_QA
    # (role). The role must win: they must NOT classify as "binary_flag"
    # (which would make them model_input).
    roles = builder._classify_columns("attributes_gageii_BasinID.csv", sorted(builder._BINARY_FLAGS))
    for c in builder._BINARY_FLAGS:
        assert roles[c] == "diagnostic_record_network_qa", f"{c} classified as {roles[c]!r}"


def test_lka_pc_use_deferred_ambiguous():
    roles = builder._classify_columns("attributes_hydroATLAS.csv", ["lka_pc_use"])
    assert roles == {"lka_pc_use": "deferred_ambiguous"}


def test_retained_fields_remain_candidate_model_input():
    roles = builder._classify_columns(
        "attributes_gageii_Hydro.csv", ["PERHOR", "STRAHLER_MAX"]
    )
    assert roles["PERHOR"] == "candidate_model_input"
    assert roles["STRAHLER_MAX"] == "candidate_model_input"
    roles2 = builder._classify_columns(
        "attributes_hydroATLAS.csv", ["dor_pc_pva", "dis_m3_pyr", "run_mm_syr"]
    )
    assert roles2["dor_pc_pva"] == "candidate_model_input"
    assert roles2["dis_m3_pyr"] == "candidate_model_input"
    assert roles2["run_mm_syr"] == "candidate_model_input"


# ---------------------------------------------------------------------------
# Part 3 -- end-to-end fixture build: RAW_* columns fall through the ordinary
# high-missingness mechanism after sentinel decoding, not by name.
# ---------------------------------------------------------------------------

def _write_gageii_csv(path, staids, columns: dict[str, list]):
    df = pd.DataFrame({"STAID": staids, **columns})
    df.to_csv(path, index=False)


def test_raw_infra_columns_excluded_via_high_missingness_after_decode(tmp_path):
    # 25 basins; RAW_DIS_NEAREST_DAM has -999 (sentinel) for 6/25 = 24% -> excluded.
    # PERHOR/STRAHLER_MAX have a single -9999/-99 each (4%) -> stay eligible.
    n = 25
    staids = [f"{i:08d}" for i in range(1, n + 1)]
    n_sentinel = 6
    raw_dam = ["-999"] * n_sentinel + [str(10.0 + i) for i in range(n - n_sentinel)]
    perhor = ["-9999"] + [str(1.0 + i) for i in range(n - 1)]
    strahler = ["-99"] + [str(2 + (i % 5)) for i in range(n - 1)]

    src_dir = tmp_path / "source"
    src_dir.mkdir()
    _write_gageii_csv(src_dir / "attributes_gageii_HydroMod_Dams.csv", staids,
                       {"RAW_DIS_NEAREST_DAM": raw_dam})
    _write_gageii_csv(src_dir / "attributes_gageii_Hydro.csv", staids,
                       {"PERHOR": perhor, "STRAHLER_MAX": strahler})
    _write_gageii_csv(src_dir / "attributes_hydroATLAS.csv", staids,
                       {"dor_pc_pva": [str(0.1 * i) for i in range(n)]})
    _write_gageii_csv(src_dir / "attributes_nldas2_climate.csv", staids,
                       {"nldas2_dummy": [str(1.0 + i) for i in range(n)]})

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({"STAID": staids, "final_training_status": ["TRAIN_CORE"] * n}).to_csv(
        manifest_path, index=False
    )

    out_dir = tmp_path / "out"
    ns = _Namespace(
        source_dir=str(src_dir), manifest=str(manifest_path), out_dir=str(out_dir),
        matrix_name="test_matrix", require_checksums=False, force=True, dry_run=False,
    )
    orig_gap = builder._EXPECTED_HYDROATLAS_GAP_STAIDS
    builder._EXPECTED_HYDROATLAS_GAP_STAIDS = frozenset()  # fixture has full HydroATLAS coverage
    try:
        builder.build(ns)
    finally:
        builder._EXPECTED_HYDROATLAS_GAP_STAIDS = orig_gap

    matrix = pd.read_parquet(out_dir / "test_matrix.parquet")
    with open(out_dir / "test_matrix_column_manifest.json") as f:
        manifest_json = json.load(f)
    roles = {c: v["role"] for c, v in manifest_json["columns"].items()}

    assert roles.get("RAW_DIS_NEAREST_DAM") is None or roles.get("RAW_DIS_NEAREST_DAM") != "model_input"
    assert "RAW_DIS_NEAREST_DAM" not in matrix.columns
    assert roles["PERHOR"] == "model_input"
    assert roles["STRAHLER_MAX"] == "model_input"
    assert not (matrix["PERHOR"] == -9999.0).any()
    assert not (matrix["STRAHLER_MAX"] == -99.0).any()

    with open(out_dir / "test_matrix_provenance.json") as f:
        prov = json.load(f)
    assert "RAW_DIS_NEAREST_DAM" in prov["high_missing_excluded_model_input"]
    assert prov["sentinel_decoding"]["replacement_counts_by_source_qualified_column"][
        "attributes_gageii_HydroMod_Dams.csv:RAW_DIS_NEAREST_DAM"
    ] == n_sentinel


class _Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# ---------------------------------------------------------------------------
# Part 4 -- auditor regressions
# ---------------------------------------------------------------------------

def _base_matrix_and_manifest(tmp_path, model_input_extra_roles=None):
    """Minimal valid matrix + column manifest + provenance the auditor accepts,
    with one basin so per-column checks have something to operate on. Callers
    override specific column roles/values to trigger individual hard-fail checks.
    """
    staids = [f"{i:08d}" for i in range(1, 6)]
    df = pd.DataFrame(index=pd.Index(staids, name="gauge_id"))
    df["good_model_input"] = [1.0, 2.0, 3.0, 4.0, 5.0]
    df["PERHOR"] = [10.0, 11.0, 12.0, 13.0, 14.0]
    df["STRAHLER_MAX"] = [2.0, 3.0, 4.0, 5.0, 6.0]
    df["dor_pc_pva"] = [0.1, 0.2, 0.3, 0.4, 0.5]
    df["dis_m3_pyr"] = [15.0, 25.0, 35.0, 45.0, 55.0]
    df["run_mm_syr"] = [100.0, 200.0, 300.0, 400.0, 500.0]
    df["STATE"] = ["CA"] * 5
    df["HUC02"] = ["18"] * 5
    df["LAT_GAGE"] = [37.0] * 5
    df["LNG_GAGE"] = [-120.0] * 5
    df["LAT_CENT"] = [37.1] * 5
    df["LONG_CENT"] = [-120.1] * 5
    for c in sorted(auditor._DIAGNOSTIC_RECORD_NETWORK_QA):
        df[c] = [0, 1, 0, 1, 0]
    df["lka_pc_use"] = [0.0] * 5
    df["hydroatlas_coverage_flag"] = [1] * 5
    df["final_training_status"] = ["TRAIN_CORE"] * 5

    roles = {
        "good_model_input": "model_input", "PERHOR": "model_input",
        "STRAHLER_MAX": "model_input", "dor_pc_pva": "model_input",
        "dis_m3_pyr": "model_input", "run_mm_syr": "model_input",
        "STATE": "split_support", "HUC02": "split_support",
        "LAT_GAGE": "diagnostic_latlon", "LNG_GAGE": "diagnostic_latlon",
        "LAT_CENT": "diagnostic_latlon", "LONG_CENT": "diagnostic_latlon",
        "lka_pc_use": "deferred_ambiguous",
        "hydroatlas_coverage_flag": "flag", "final_training_status": "flag",
    }
    for c in auditor._DIAGNOSTIC_RECORD_NETWORK_QA:
        roles[c] = "diagnostic_record_network_qa"
    if model_input_extra_roles:
        roles.update(model_input_extra_roles)

    matrix_dir = tmp_path
    matrix_path = matrix_dir / "test_matrix.parquet"
    df.to_parquet(matrix_path)
    matrix_sha256 = auditor._sha256(matrix_path)

    manifest_json = {
        "matrix_name": "test_matrix", "n_rows": len(df), "n_columns": len(df.columns),
        "columns": {c: {"role": roles.get(c, "model_input"), "source_file": "dummy.csv"} for c in df.columns},
    }
    with open(matrix_dir / "test_matrix_column_manifest.json", "w") as f:
        json.dump(manifest_json, f)

    provenance = {
        "matrix_sha256": matrix_sha256,
        "hydroatlas_gap": {"basins_flagged_missing": 0},
    }
    with open(matrix_dir / "test_matrix_provenance.json", "w") as f:
        json.dump(provenance, f)

    basin_manifest_path = matrix_dir / "basin_manifest.csv"
    pd.DataFrame({"STAID": staids}).to_csv(basin_manifest_path, index=False)

    return matrix_dir, basin_manifest_path


def _run_auditor(matrix_dir, basin_manifest_path):
    auditor._issues.clear()
    auditor._warns.clear()
    auditor._oks.clear()
    ns = _Namespace(
        matrix_dir=str(matrix_dir), matrix_name="test_matrix", manifest=str(basin_manifest_path),
    )
    orig_parse_args = auditor._parse_args
    auditor._parse_args = lambda: ns
    try:
        with pytest.raises(SystemExit) as exc_info:
            auditor.main()
    finally:
        auditor._parse_args = orig_parse_args
    return exc_info.value.code, list(auditor._issues)


def test_auditor_passes_corrected_fixture(tmp_path):
    # HydroATLAS gap check requires an exact-5-basin gap OR flag column all-1s
    # matching zero flagged; use a fixture with hydroatlas_coverage_flag all 1
    # and an empty expected gap override so check 9 doesn't need real STAIDs.
    matrix_dir, basin_manifest_path = _base_matrix_and_manifest(tmp_path)
    orig_gap = auditor._EXPECTED_HYDROATLAS_GAP_STAIDS
    auditor._EXPECTED_HYDROATLAS_GAP_STAIDS = frozenset()
    try:
        code, issues = _run_auditor(matrix_dir, basin_manifest_path)
    finally:
        auditor._EXPECTED_HYDROATLAS_GAP_STAIDS = orig_gap
    assert code == 0, f"expected PASS, got issues: {issues}"


def test_auditor_fails_on_surviving_sentinel_in_model_input(tmp_path):
    matrix_dir, basin_manifest_path = _base_matrix_and_manifest(tmp_path)
    df = pd.read_parquet(matrix_dir / "test_matrix.parquet")
    df.loc[df.index[0], "PERHOR"] = -9999.0
    df.to_parquet(matrix_dir / "test_matrix.parquet")
    # refresh checksum record so this failure isn't masked by a checksum mismatch
    prov_path = matrix_dir / "test_matrix_provenance.json"
    prov = json.loads(prov_path.read_text())
    prov["matrix_sha256"] = auditor._sha256(matrix_dir / "test_matrix.parquet")
    prov_path.write_text(json.dumps(prov))

    orig_gap = auditor._EXPECTED_HYDROATLAS_GAP_STAIDS
    auditor._EXPECTED_HYDROATLAS_GAP_STAIDS = frozenset()
    try:
        code, issues = _run_auditor(matrix_dir, basin_manifest_path)
    finally:
        auditor._EXPECTED_HYDROATLAS_GAP_STAIDS = orig_gap
    assert code == 1
    assert any("sentinel" in i.lower() for i in issues)


def test_auditor_fails_on_lat_cent_as_model_input(tmp_path):
    matrix_dir, basin_manifest_path = _base_matrix_and_manifest(
        tmp_path, model_input_extra_roles={"LAT_CENT": "model_input"}
    )
    orig_gap = auditor._EXPECTED_HYDROATLAS_GAP_STAIDS
    auditor._EXPECTED_HYDROATLAS_GAP_STAIDS = frozenset()
    try:
        code, issues = _run_auditor(matrix_dir, basin_manifest_path)
    finally:
        auditor._EXPECTED_HYDROATLAS_GAP_STAIDS = orig_gap
    assert code == 1
    assert any("Direct coordinate fields excluded from model_input" in i for i in issues)


def test_auditor_fails_on_record_network_qa_field_as_model_input(tmp_path):
    matrix_dir, basin_manifest_path = _base_matrix_and_manifest(
        tmp_path, model_input_extra_roles={"ACTIVE09": "model_input"}
    )
    orig_gap = auditor._EXPECTED_HYDROATLAS_GAP_STAIDS
    auditor._EXPECTED_HYDROATLAS_GAP_STAIDS = frozenset()
    try:
        code, issues = _run_auditor(matrix_dir, basin_manifest_path)
    finally:
        auditor._EXPECTED_HYDROATLAS_GAP_STAIDS = orig_gap
    assert code == 1
    assert any("record/network/QA" in i for i in issues)


def test_auditor_fails_on_lka_pc_use_as_model_input(tmp_path):
    matrix_dir, basin_manifest_path = _base_matrix_and_manifest(
        tmp_path, model_input_extra_roles={"lka_pc_use": "model_input"}
    )
    orig_gap = auditor._EXPECTED_HYDROATLAS_GAP_STAIDS
    auditor._EXPECTED_HYDROATLAS_GAP_STAIDS = frozenset()
    try:
        code, issues = _run_auditor(matrix_dir, basin_manifest_path)
    finally:
        auditor._EXPECTED_HYDROATLAS_GAP_STAIDS = orig_gap
    assert code == 1
    assert any("lka_pc_use excluded from model_input" in i for i in issues)


def test_auditor_fails_on_raw_infra_column_surviving_as_model_input(tmp_path):
    matrix_dir, basin_manifest_path = _base_matrix_and_manifest(tmp_path)
    df = pd.read_parquet(matrix_dir / "test_matrix.parquet")
    df["RAW_DIS_NEAREST_DAM"] = [1.0, 2.0, 3.0, 4.0, 5.0]
    df.to_parquet(matrix_dir / "test_matrix.parquet")

    manifest_json_path = matrix_dir / "test_matrix_column_manifest.json"
    manifest_json = json.loads(manifest_json_path.read_text())
    manifest_json["columns"]["RAW_DIS_NEAREST_DAM"] = {"role": "model_input", "source_file": "dummy.csv"}
    manifest_json_path.write_text(json.dumps(manifest_json))

    prov_path = matrix_dir / "test_matrix_provenance.json"
    prov = json.loads(prov_path.read_text())
    prov["matrix_sha256"] = auditor._sha256(matrix_dir / "test_matrix.parquet")
    prov_path.write_text(json.dumps(prov))

    orig_gap = auditor._EXPECTED_HYDROATLAS_GAP_STAIDS
    auditor._EXPECTED_HYDROATLAS_GAP_STAIDS = frozenset()
    try:
        code, issues = _run_auditor(matrix_dir, basin_manifest_path)
    finally:
        auditor._EXPECTED_HYDROATLAS_GAP_STAIDS = orig_gap
    assert code == 1
    assert any("Infrastructure-distance RAW_* columns excluded from model_input" in i for i in issues)

"""Tests for scripts/prepare_stage1_full_static_attributes.py: the thin
production orchestration script that fits development-training-only median
imputation and development-training-only zero-variance trainability
projection, then applies both, frozen, to the full 2,557-basin non-California
package population (2,307 development-training + 250 spatial-holdout).
Synthetic fixtures only -- no h2o/real data required.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import scripts.prepare_stage1_full_static_attributes as script
from src.baseline.static_preparation import StaticPreparationError

MODEL_INPUT_COLS = ["attr_a", "attr_b", "attr_c"]


def _write_manifest(tmp_path, model_input_cols=MODEL_INPUT_COLS, name="manifest.json"):
    columns = {c: {"role": "model_input", "source_file": "gagesii"} for c in model_input_cols}
    columns["STATE"] = {"role": "split_support", "source_file": "gagesii"}
    p = tmp_path / name
    p.write_text(json.dumps({"columns": columns}), encoding="utf-8")
    return p


def _write_basin_list(tmp_path, basin_ids, name):
    p = tmp_path / name
    p.write_text("\n".join(basin_ids) + "\n", encoding="utf-8")
    return p


def _dev_id(i):
    return f"0100{i:04d}"


def _holdout_id(i):
    return f"0900{i:04d}"


def _write_matrix(tmp_path, dev_ids, holdout_ids, *, const_dev_col=False, name="matrix.parquet"):
    """Build a synthetic matrix: attr_a/b vary across dev-train; attr_c is
    constant across dev-train when const_dev_col=True (must be excluded by
    the zero-variance fit), varying at holdout (must not leak in)."""
    rows = {}
    for i, sid in enumerate(dev_ids):
        rows[sid] = {
            "attr_a": float(i + 1),
            "attr_b": float(10 * (i + 1)),
            "attr_c": 5.0 if const_dev_col else float(i + 1),
            "STATE": "ME",
        }
    for j, sid in enumerate(holdout_ids):
        rows[sid] = {
            "attr_a": float(100 + j),
            "attr_b": float(1000 + j),
            "attr_c": float(200 + j),  # varies at holdout even if const at dev
            "STATE": "NC",
        }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "gauge_id"
    p = tmp_path / name
    df.to_parquet(p)
    return p


_DEV_IDS = [_dev_id(i) for i in range(5)]
_HOLDOUT_IDS = [_holdout_id(i) for i in range(3)]


def _run_synthetic(tmp_path, *, const_dev_col=True, force=False, out_name="out"):
    matrix_p = _write_matrix(tmp_path, _DEV_IDS, _HOLDOUT_IDS, const_dev_col=const_dev_col)
    manifest_p = _write_manifest(tmp_path)
    dev_list_p = _write_basin_list(tmp_path, _DEV_IDS, "development_train.txt")
    holdout_list_p = _write_basin_list(tmp_path, _HOLDOUT_IDS, "spatial_holdout_nonca.txt")
    out_dir = tmp_path / out_name
    return script.prepare_full_static_attributes(
        static_matrix_path=matrix_p,
        column_manifest_path=manifest_p,
        development_basin_list_path=dev_list_p,
        spatial_holdout_basin_list_path=holdout_list_p,
        output_dir=out_dir,
        force=force,
        expected_development_count=len(_DEV_IDS),
        expected_holdout_count=len(_HOLDOUT_IDS),
        expected_total_count=len(_DEV_IDS) + len(_HOLDOUT_IDS),
        expected_model_input_count=len(MODEL_INPUT_COLS),
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_happy_path_summary_counts(tmp_path):
    summary = _run_synthetic(tmp_path)
    assert summary["development_basin_count"] == 5
    assert summary["spatial_holdout_basin_count"] == 3
    assert summary["package_population_count"] == 8
    assert summary["candidate_column_count"] == 3
    assert summary["retained_column_count"] == 2
    assert summary["excluded_column_count"] == 1
    assert summary["excluded_columns"] == ["attr_c"]
    assert summary["fit_basin_scope"] == "development_training_only"


def test_canonical_imputed_table_retains_all_candidate_columns(tmp_path):
    summary = _run_synthetic(tmp_path)
    imputed = pd.read_parquet(summary["output_paths"]["imputed_static_attributes.parquet"])
    assert list(imputed.columns) == MODEL_INPUT_COLS
    assert len(imputed) == 8
    assert np.all(np.isfinite(imputed.to_numpy(dtype=float)))


def test_retained_table_uses_only_frozen_retained_columns(tmp_path):
    summary = _run_synthetic(tmp_path)
    retained = pd.read_parquet(summary["output_paths"]["retained_static_attributes.parquet"])
    assert list(retained.columns) == summary["retained_columns"]
    assert "attr_c" not in retained.columns
    assert len(retained) == 8


def test_imputation_fit_excludes_holdout_rows(tmp_path):
    # attr_a is NaN for one dev-train basin; holdout carries wildly different
    # values (100+) that must never leak into the fitted median.
    matrix_p = tmp_path / "matrix.parquet"
    rows = {}
    for i, sid in enumerate(_DEV_IDS):
        rows[sid] = {"attr_a": np.nan if i == 0 else float(i), "attr_b": float(i + 1), "attr_c": float(i + 1)}
    for j, sid in enumerate(_HOLDOUT_IDS):
        rows[sid] = {"attr_a": 999.0, "attr_b": 999.0, "attr_c": float(200 + j)}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "gauge_id"
    df.to_parquet(matrix_p)

    manifest_p = _write_manifest(tmp_path)
    dev_list_p = _write_basin_list(tmp_path, _DEV_IDS, "development_train.txt")
    holdout_list_p = _write_basin_list(tmp_path, _HOLDOUT_IDS, "spatial_holdout_nonca.txt")
    summary = script.prepare_full_static_attributes(
        static_matrix_path=matrix_p,
        column_manifest_path=manifest_p,
        development_basin_list_path=dev_list_p,
        spatial_holdout_basin_list_path=holdout_list_p,
        output_dir=tmp_path / "out",
        expected_development_count=len(_DEV_IDS),
        expected_holdout_count=len(_HOLDOUT_IDS),
        expected_total_count=len(_DEV_IDS) + len(_HOLDOUT_IDS),
        expected_model_input_count=len(MODEL_INPUT_COLS),
    )
    imputed = pd.read_parquet(summary["output_paths"]["imputed_static_attributes.parquet"])
    # Median of dev-train attr_a values [1, 2, 3, 4] (index 0 is NaN) is 2.5;
    # if holdout (999.0) had leaked in, this would be far larger.
    assert imputed.loc[_DEV_IDS[0], "attr_a"] == pytest.approx(2.5)


def test_zero_variance_fit_excludes_holdout_rows(tmp_path):
    # attr_c constant (5.0) at dev-train but varies at holdout -- must be
    # excluded (proves the zero-variance fit ignores holdout rows).
    summary = _run_synthetic(tmp_path, const_dev_col=True)
    assert "attr_c" in summary["excluded_columns"]


def test_zero_variance_fit_not_excluded_when_dev_varies(tmp_path):
    summary = _run_synthetic(tmp_path, const_dev_col=False)
    assert summary["excluded_columns"] == []
    assert summary["retained_column_count"] == 3


def test_ordering_is_deterministic(tmp_path):
    summary_a = _run_synthetic(tmp_path, out_name="out_a")
    summary_b = _run_synthetic(tmp_path, out_name="out_b")
    imputed_a = pd.read_parquet(summary_a["output_paths"]["imputed_static_attributes.parquet"])
    imputed_b = pd.read_parquet(summary_b["output_paths"]["imputed_static_attributes.parquet"])
    assert list(imputed_a.index) == list(imputed_b.index)
    assert list(imputed_a.index) == sorted(imputed_a.index)
    assert summary_a["retained_columns"] == summary_b["retained_columns"]


def test_retained_columns_txt_matches_summary(tmp_path):
    summary = _run_synthetic(tmp_path)
    text = (tmp_path / "out" / "retained_static_columns.txt").read_text(encoding="utf-8")
    assert text.splitlines() == summary["retained_columns"]
    excluded_text = (tmp_path / "out" / "excluded_zero_variance_columns.txt").read_text(encoding="utf-8")
    assert excluded_text.splitlines() == summary["excluded_columns"]


def test_summary_checksums_present_and_consistent(tmp_path):
    summary = _run_synthetic(tmp_path)
    assert len(summary["static_matrix_sha256"]) == 64
    assert len(summary["column_manifest_sha256"]) == 64
    assert len(summary["development_basin_list_sha256"]) == 64
    assert len(summary["spatial_holdout_basin_list_sha256"]) == 64
    assert set(summary["output_sha256"]) == set(summary["output_paths"])
    for name, path in summary["output_paths"].items():
        from src.baseline.splits import sha256_of
        assert summary["output_sha256"][name] == sha256_of(path)

    written = json.loads((tmp_path / "out" / "run_summary.json").read_text(encoding="utf-8"))
    assert written["package_population_count"] == summary["package_population_count"]


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------

def test_overlapping_lists_fails(tmp_path):
    matrix_p = _write_matrix(tmp_path, _DEV_IDS, _HOLDOUT_IDS)
    manifest_p = _write_manifest(tmp_path)
    dev_list_p = _write_basin_list(tmp_path, _DEV_IDS, "development_train.txt")
    # Overlap: holdout list includes one dev-train basin.
    holdout_list_p = _write_basin_list(tmp_path, _HOLDOUT_IDS + [_DEV_IDS[0]], "spatial_holdout_nonca.txt")
    with pytest.raises(StaticPreparationError, match="overlap"):
        script.prepare_full_static_attributes(
            static_matrix_path=matrix_p,
            column_manifest_path=manifest_p,
            development_basin_list_path=dev_list_p,
            spatial_holdout_basin_list_path=holdout_list_p,
            output_dir=tmp_path / "out",
            expected_development_count=len(_DEV_IDS),
            expected_holdout_count=len(_HOLDOUT_IDS) + 1,
            expected_total_count=len(_DEV_IDS) + len(_HOLDOUT_IDS),
            expected_model_input_count=len(MODEL_INPUT_COLS),
        )


def test_wrong_population_count_fails(tmp_path):
    matrix_p = _write_matrix(tmp_path, _DEV_IDS, _HOLDOUT_IDS)
    manifest_p = _write_manifest(tmp_path)
    dev_list_p = _write_basin_list(tmp_path, _DEV_IDS, "development_train.txt")
    holdout_list_p = _write_basin_list(tmp_path, _HOLDOUT_IDS, "spatial_holdout_nonca.txt")
    with pytest.raises(StaticPreparationError, match="development basin count"):
        script.prepare_full_static_attributes(
            static_matrix_path=matrix_p,
            column_manifest_path=manifest_p,
            development_basin_list_path=dev_list_p,
            spatial_holdout_basin_list_path=holdout_list_p,
            output_dir=tmp_path / "out",
            expected_development_count=len(_DEV_IDS) + 1,  # mismatch: actual is len(_DEV_IDS)
            expected_holdout_count=len(_HOLDOUT_IDS),
            expected_total_count=len(_DEV_IDS) + len(_HOLDOUT_IDS),
            expected_model_input_count=len(MODEL_INPUT_COLS),
        )


def test_wrong_model_input_count_fails(tmp_path):
    matrix_p = _write_matrix(tmp_path, _DEV_IDS, _HOLDOUT_IDS)
    manifest_p = _write_manifest(tmp_path)
    dev_list_p = _write_basin_list(tmp_path, _DEV_IDS, "development_train.txt")
    holdout_list_p = _write_basin_list(tmp_path, _HOLDOUT_IDS, "spatial_holdout_nonca.txt")
    with pytest.raises(StaticPreparationError, match="candidate model_input column count"):
        script.prepare_full_static_attributes(
            static_matrix_path=matrix_p,
            column_manifest_path=manifest_p,
            development_basin_list_path=dev_list_p,
            spatial_holdout_basin_list_path=holdout_list_p,
            output_dir=tmp_path / "out",
            expected_development_count=len(_DEV_IDS),
            expected_holdout_count=len(_HOLDOUT_IDS),
            expected_total_count=len(_DEV_IDS) + len(_HOLDOUT_IDS),
            expected_model_input_count=99,
        )


def test_existing_output_without_force_fails(tmp_path):
    _run_synthetic(tmp_path)  # writes to tmp_path/out with correct expected_* overrides
    with pytest.raises(StaticPreparationError, match="force"):
        _run_synthetic(tmp_path, force=False)


def test_existing_output_with_force_succeeds(tmp_path):
    _run_synthetic(tmp_path)
    summary = _run_synthetic(tmp_path, force=True)
    assert summary["package_population_count"] == 8


def test_missing_basin_in_matrix_fails(tmp_path):
    matrix_p = _write_matrix(tmp_path, _DEV_IDS, _HOLDOUT_IDS)
    manifest_p = _write_manifest(tmp_path)
    dev_list_p = _write_basin_list(tmp_path, _DEV_IDS + ["01999999"], "development_train.txt")
    holdout_list_p = _write_basin_list(tmp_path, _HOLDOUT_IDS, "spatial_holdout_nonca.txt")
    with pytest.raises(StaticPreparationError, match="missing from matrix"):
        script.prepare_full_static_attributes(
            static_matrix_path=matrix_p,
            column_manifest_path=manifest_p,
            development_basin_list_path=dev_list_p,
            spatial_holdout_basin_list_path=holdout_list_p,
            output_dir=tmp_path / "out",
            expected_development_count=len(_DEV_IDS) + 1,
            expected_holdout_count=len(_HOLDOUT_IDS),
            expected_total_count=len(_DEV_IDS) + 1 + len(_HOLDOUT_IDS),
            expected_model_input_count=len(MODEL_INPUT_COLS),
        )

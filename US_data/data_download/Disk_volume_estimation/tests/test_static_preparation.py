"""Tests for src/baseline/static_preparation.py (Milestone 2K-G-I primitives
increment): canonical static-matrix loading, development-train-only median
imputation fit, frozen-value apply to a target basin subset, and manifest
writing. Synthetic fixtures only -- no h2o/real data required.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.baseline.static_preparation import (
    ImputationFit,
    StaticPreparationError,
    apply_imputation,
    build_imputation_manifest,
    fit_development_median_imputation,
    load_column_manifest,
    load_development_train_basin_ids,
    load_static_matrix,
    model_input_columns_from_manifest,
    select_basin_rows,
    split_model_input_and_metadata,
    validate_unique_normalized_basin_ids,
    write_imputation_artifacts,
)

MODEL_INPUT_COLS = ["attr_a", "attr_b", "attr_c"]
META_COLS = ["HUC02", "STATE"]


def _write_parquet(tmp_path, df, name="matrix.parquet"):
    p = tmp_path / name
    df.to_parquet(p)
    return p


def _write_manifest(tmp_path, model_input_cols=MODEL_INPUT_COLS, other_role_cols=("HUC02", "STATE"), name="manifest.json"):
    columns = {c: {"role": "model_input", "source_file": "gagesii"} for c in model_input_cols}
    for c in other_role_cols:
        columns[c] = {"role": "split_support", "source_file": "gagesii"}
    p = tmp_path / name
    p.write_text(json.dumps({"columns": columns}), encoding="utf-8")
    return p


def _matrix_df(rows):
    """rows: dict[staid] -> dict of column values (any missing MODEL_INPUT_COLS default to a value)."""
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "gauge_id"
    return frame


# ---------------------------------------------------------------------------
# load_column_manifest / model_input_columns_from_manifest
# ---------------------------------------------------------------------------


def test_load_column_manifest_happy_path(tmp_path):
    p = _write_manifest(tmp_path)
    manifest = load_column_manifest(p)
    assert set(manifest["columns"]) == set(MODEL_INPUT_COLS) | {"HUC02", "STATE"}


def test_load_column_manifest_missing_file(tmp_path):
    with pytest.raises(StaticPreparationError, match="not found"):
        load_column_manifest(tmp_path / "nope.json")


def test_load_column_manifest_malformed(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"not_columns": {}}), encoding="utf-8")
    with pytest.raises(StaticPreparationError, match="columns"):
        load_column_manifest(p)


def test_model_input_columns_from_manifest_sorted(tmp_path):
    manifest = load_column_manifest(_write_manifest(tmp_path))
    assert model_input_columns_from_manifest(manifest) == sorted(MODEL_INPUT_COLS)


def test_model_input_columns_from_manifest_empty_role_raises(tmp_path):
    manifest = load_column_manifest(_write_manifest(tmp_path))
    with pytest.raises(StaticPreparationError, match="role"):
        model_input_columns_from_manifest(manifest, role="nonexistent_role")


# ---------------------------------------------------------------------------
# load_static_matrix
# ---------------------------------------------------------------------------


def test_load_static_matrix_happy_path(tmp_path):
    df = _matrix_df({
        "01019000": {"attr_a": 1.0, "attr_b": 2.0, "attr_c": 3.0, "HUC02": "01", "STATE": "ME"},
        "01019001": {"attr_a": 4.0, "attr_b": 5.0, "attr_c": 6.0, "HUC02": "01", "STATE": "ME"},
    })
    matrix_p = _write_parquet(tmp_path, df)
    manifest_p = _write_manifest(tmp_path)
    matrix_df, model_input_cols, manifest = load_static_matrix(matrix_p, manifest_p)
    assert model_input_cols == sorted(MODEL_INPUT_COLS)
    assert list(matrix_df.index) == ["01019000", "01019001"]
    assert "HUC02" in matrix_df.columns  # non-model metadata preserved too


def test_load_static_matrix_retains_all_parquet_columns_not_just_model_input(tmp_path):
    # Proves the documented contract: load_static_matrix() -> load_matrix_for_splits()
    # returns the full, unsubsetted parquet (see src/baseline/splits.py), not just
    # model_input_cols. "extra_unlisted_col" is not declared in the column manifest
    # at all (neither model_input nor any other role), so this also proves the
    # retention is driven by the parquet's own columns, not by the manifest's schema.
    df = _matrix_df({
        "01019000": {
            "attr_a": 1.0, "attr_b": 2.0, "attr_c": 3.0,
            "HUC02": "01", "STATE": "ME", "extra_unlisted_col": "unmodeled_value",
        },
    })
    matrix_p = _write_parquet(tmp_path, df)
    manifest_p = _write_manifest(tmp_path)  # only declares attr_a/b/c + HUC02/STATE
    matrix_df, model_input_cols, _manifest = load_static_matrix(matrix_p, manifest_p)
    assert model_input_cols == sorted(MODEL_INPUT_COLS)
    assert set(matrix_df.columns) == set(df.columns)
    assert matrix_df.loc["01019000", "extra_unlisted_col"] == "unmodeled_value"


def test_load_static_matrix_wraps_split_generation_error(tmp_path):
    df = _matrix_df({"01019000": {"attr_a": 1.0, "attr_b": 2.0, "HUC02": "01", "STATE": "ME"}})
    matrix_p = _write_parquet(tmp_path, df)  # missing attr_c
    manifest_p = _write_manifest(tmp_path)
    with pytest.raises(StaticPreparationError, match="missing required columns"):
        load_static_matrix(matrix_p, manifest_p)


# ---------------------------------------------------------------------------
# validate_unique_normalized_basin_ids / select_basin_rows / split
# ---------------------------------------------------------------------------


def test_validate_unique_normalized_basin_ids_normalizes():
    assert validate_unique_normalized_basin_ids(["1019000", "393109104464500"]) == [
        "01019000",
        "393109104464500",
    ]


def test_validate_unique_normalized_basin_ids_rejects_duplicate():
    with pytest.raises(StaticPreparationError, match="duplicate"):
        validate_unique_normalized_basin_ids(["1019000", "01019000"])


def test_validate_unique_normalized_basin_ids_rejects_malformed():
    with pytest.raises(StaticPreparationError, match="malformed"):
        validate_unique_normalized_basin_ids(["ABCDEFGH"])


def test_select_basin_rows_happy_path():
    matrix = pd.DataFrame(
        {"attr_a": [1.0, 2.0]}, index=pd.Index(["01019000", "01019001"], name="gauge_id")
    )
    out = select_basin_rows(matrix, ["1019000"])
    assert list(out.index) == ["01019000"]


def test_select_basin_rows_missing_from_matrix_raises():
    matrix = pd.DataFrame({"attr_a": [1.0]}, index=pd.Index(["01019000"], name="gauge_id"))
    with pytest.raises(StaticPreparationError, match="missing from matrix"):
        select_basin_rows(matrix, ["99999999"])


def test_split_model_input_and_metadata():
    df = pd.DataFrame(
        {"attr_a": [1.0], "attr_b": [2.0], "attr_c": [3.0], "HUC02": ["01"], "STATE": ["ME"]},
        index=pd.Index(["01019000"], name="gauge_id"),
    )
    model_input_df, metadata_df = split_model_input_and_metadata(df, MODEL_INPUT_COLS)
    assert list(model_input_df.columns) == MODEL_INPUT_COLS
    assert list(metadata_df.columns) == META_COLS


# ---------------------------------------------------------------------------
# load_development_train_basin_ids
# ---------------------------------------------------------------------------


def _write_split_assignment(tmp_path, rows, name="split_assignment.csv"):
    df = pd.DataFrame(rows)
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_load_development_train_basin_ids_happy_path(tmp_path):
    p = _write_split_assignment(tmp_path, [
        {"STAID": "01019000", "STATE": "ME", "split_role": "development_train"},
        {"STAID": "01019001", "STATE": "ME", "split_role": "development_train"},
        {"STAID": "02000000", "STATE": "NC", "split_role": "spatial_holdout_nonca"},
        {"STAID": "11000000", "STATE": "CA", "split_role": "california_finetune_train"},
    ])
    ids = load_development_train_basin_ids(p)
    assert ids == ["01019000", "01019001"]


def test_load_development_train_basin_ids_rejects_ca_under_dev_role(tmp_path):
    p = _write_split_assignment(tmp_path, [
        {"STAID": "01019000", "STATE": "ME", "split_role": "development_train"},
        {"STAID": "11000000", "STATE": "CA", "split_role": "development_train"},
    ])
    with pytest.raises(StaticPreparationError, match="CA"):
        load_development_train_basin_ids(p)


def test_load_development_train_basin_ids_rejects_missing_role(tmp_path):
    p = _write_split_assignment(tmp_path, [
        {"STAID": "01019000", "STATE": "ME", "split_role": "spatial_holdout_nonca"},
    ])
    with pytest.raises(StaticPreparationError, match="development_train"):
        load_development_train_basin_ids(p)


def test_load_development_train_basin_ids_rejects_missing_column(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"STAID": ["01019000"], "split_role": ["development_train"]}).to_csv(p, index=False)
    with pytest.raises(StaticPreparationError, match="STATE"):
        load_development_train_basin_ids(p)


def test_load_development_train_basin_ids_missing_file(tmp_path):
    with pytest.raises(StaticPreparationError, match="not found"):
        load_development_train_basin_ids(tmp_path / "nope.csv")


# ---------------------------------------------------------------------------
# fit_development_median_imputation
# ---------------------------------------------------------------------------


_D001, _D002, _D003 = "01000001", "01000002", "01000003"
_H001, _H002 = "09000001", "09000002"
_T001 = "01000009"


def _dev_holdout_matrix():
    # dev-train basins: attr_a medians to 2.0; holdout basins carry wildly
    # different values that must NEVER influence the fit (no-leakage check).
    rows = {
        _D001: {"attr_a": 1.0, "attr_b": 10.0, "attr_c": np.nan},
        _D002: {"attr_a": 2.0, "attr_b": 20.0, "attr_c": 100.0},
        _D003: {"attr_a": 3.0, "attr_b": np.nan, "attr_c": 200.0},
        _H001: {"attr_a": 999.0, "attr_b": 999.0, "attr_c": 999.0},
        _H002: {"attr_a": -999.0, "attr_b": -999.0, "attr_c": -999.0},
    }
    return _matrix_df(rows)


def test_fit_development_median_imputation_uses_only_dev_train_rows():
    matrix = _dev_holdout_matrix()
    fit = fit_development_median_imputation(matrix, MODEL_INPUT_COLS, [_D001, _D002, _D003])
    assert fit.fitted_values["attr_a"] == pytest.approx(2.0)
    assert fit.fit_population_size == 3
    assert fit.fit_basin_scope == "development_training_only"


def test_fit_development_median_imputation_no_leakage_from_holdout():
    matrix = _dev_holdout_matrix()
    fit = fit_development_median_imputation(matrix, MODEL_INPUT_COLS, [_D001, _D002, _D003])
    # If holdout rows (999 / -999) had leaked in, the median would be very different.
    for col in MODEL_INPUT_COLS:
        assert abs(fit.fitted_values[col]) < 900


def test_fit_development_median_imputation_records_missing_before_fit():
    matrix = _dev_holdout_matrix()
    fit = fit_development_median_imputation(matrix, MODEL_INPUT_COLS, [_D001, _D002, _D003])
    assert fit.n_missing_before_fit["attr_c"] == 1
    assert fit.n_missing_before_fit["attr_b"] == 1
    assert fit.n_missing_before_fit["attr_a"] == 0


def test_fit_development_median_imputation_rejects_non_median_strategy():
    matrix = _dev_holdout_matrix()
    with pytest.raises(StaticPreparationError, match="median"):
        fit_development_median_imputation(matrix, MODEL_INPUT_COLS, [_D001], strategy="mean")


def test_fit_development_median_imputation_all_nan_column_raises_by_default():
    rows = {
        _D001: {"attr_a": 1.0, "attr_b": np.nan, "attr_c": 1.0},
        _D002: {"attr_a": 2.0, "attr_b": np.nan, "attr_c": 2.0},
    }
    matrix = _matrix_df(rows)
    with pytest.raises(StaticPreparationError, match="all-NaN"):
        fit_development_median_imputation(matrix, MODEL_INPUT_COLS, [_D001, _D002])


def test_fit_development_median_imputation_all_nan_column_allowed_when_flagged_off():
    rows = {
        _D001: {"attr_a": 1.0, "attr_b": np.nan, "attr_c": 1.0},
        _D002: {"attr_a": 2.0, "attr_b": np.nan, "attr_c": 2.0},
    }
    matrix = _matrix_df(rows)
    fit = fit_development_median_imputation(
        matrix, MODEL_INPUT_COLS, [_D001, _D002], fail_if_all_nan=False
    )
    assert fit.columns_all_nan_in_fit_population == ("attr_b",)


# ---------------------------------------------------------------------------
# apply_imputation (frozen values applied to a target subset, e.g. compact package)
# ---------------------------------------------------------------------------


def test_apply_imputation_fills_nans_with_frozen_values():
    matrix = _dev_holdout_matrix()
    fit = fit_development_median_imputation(matrix, MODEL_INPUT_COLS, [_D001, _D002, _D003])
    imputed_df, mask_df, counts = apply_imputation(matrix, [_D001], fit)
    assert imputed_df.loc[_D001, "attr_c"] == pytest.approx(fit.fitted_values["attr_c"])
    assert bool(mask_df.loc[_D001, "attr_c"]) is True
    assert bool(mask_df.loc[_D001, "attr_a"]) is False
    assert counts["attr_c"] == {"n_missing_before": 1, "n_missing_after": 0}


def test_apply_imputation_never_touches_non_nan_values():
    matrix = _dev_holdout_matrix()
    fit = fit_development_median_imputation(matrix, MODEL_INPUT_COLS, [_D001, _D002, _D003])
    imputed_df, _, _ = apply_imputation(matrix, [_D002], fit)
    assert imputed_df.loc[_D002, "attr_a"] == pytest.approx(2.0)
    assert imputed_df.loc[_D002, "attr_c"] == pytest.approx(100.0)


def test_apply_imputation_compound_edge_case_many_missing_columns():
    # Mimics real basin 393109104464500: one target basin missing nearly
    # every model_input column.
    cols = [f"attr_{i:03d}" for i in range(20)]
    dev_ids = [f"0100{i:04d}" for i in range(5)]
    dev_rows = {sid: {c: float(i + 1) for c in cols} for i, sid in enumerate(dev_ids)}
    edge_row = {c: np.nan for c in cols}
    edge_row[cols[0]] = 7.0  # one observed value survives
    matrix = _matrix_df({**dev_rows, "393109104464500": edge_row})

    fit = fit_development_median_imputation(matrix, cols, dev_ids)
    imputed_df, mask_df, counts = apply_imputation(matrix, ["393109104464500"], fit)

    assert mask_df.loc["393109104464500"].sum() == len(cols) - 1
    assert not mask_df.loc["393109104464500", cols[0]]
    assert imputed_df.loc["393109104464500"].isna().sum() == 0
    assert counts[cols[1]]["n_missing_before"] == 1
    assert counts[cols[1]]["n_missing_after"] == 0


def test_apply_imputation_raises_when_no_fitted_value_available():
    rows = {
        _D001: {"attr_a": 1.0, "attr_b": np.nan, "attr_c": 1.0},
        _D002: {"attr_a": 2.0, "attr_b": np.nan, "attr_c": 2.0},
        _T001: {"attr_a": 3.0, "attr_b": np.nan, "attr_c": 3.0},
    }
    matrix = _matrix_df(rows)
    fit = fit_development_median_imputation(
        matrix, MODEL_INPUT_COLS, [_D001, _D002], fail_if_all_nan=False
    )
    with pytest.raises(StaticPreparationError, match="no valid fitted"):
        apply_imputation(matrix, [_T001], fit)


# ---------------------------------------------------------------------------
# build_imputation_manifest / write_imputation_artifacts
# ---------------------------------------------------------------------------


def test_build_imputation_manifest_fields(tmp_path):
    df = _matrix_df({
        _D001: {"attr_a": 1.0, "attr_b": 2.0, "attr_c": 3.0},
        _D002: {"attr_a": 2.0, "attr_b": 3.0, "attr_c": 4.0},
    })
    matrix_p = _write_parquet(tmp_path, df)
    manifest_p = _write_manifest(tmp_path, other_role_cols=())
    fit = fit_development_median_imputation(df, MODEL_INPUT_COLS, [_D001, _D002])
    imputed_df, mask_df, counts = apply_imputation(df, [_D001], fit)

    manifest = build_imputation_manifest(
        attributes_parquet_path=matrix_p,
        column_manifest_path=manifest_p,
        fit=fit,
        applied_basin_ids=[_D001],
        counts=counts,
    )
    assert manifest["algorithm_id"] == "stage1_static_median_imputation_v1"
    assert manifest["fit_basin_scope"] == "development_training_only"
    assert manifest["fit_population_size"] == 2
    assert manifest["n_applied_basins"] == 1
    assert set(manifest["per_column"]) == set(MODEL_INPUT_COLS)
    assert manifest["per_column"]["attr_a"]["fitted_value"] == pytest.approx(1.5)
    assert len(manifest["input_matrix_sha256"]) == 64
    assert len(manifest["column_manifest_sha256"]) == 64


def test_write_imputation_artifacts_roundtrip_and_checksums(tmp_path):
    df = _matrix_df({
        _D001: {"attr_a": 1.0, "attr_b": 2.0, "attr_c": np.nan},
        _D002: {"attr_a": 2.0, "attr_b": 3.0, "attr_c": 4.0},
    })
    matrix_p = _write_parquet(tmp_path, df)
    manifest_p = _write_manifest(tmp_path, other_role_cols=())
    fit = fit_development_median_imputation(df, MODEL_INPUT_COLS, [_D001, _D002])
    imputed_df, mask_df, counts = apply_imputation(df, [_D001], fit)
    manifest = build_imputation_manifest(
        attributes_parquet_path=matrix_p,
        column_manifest_path=manifest_p,
        fit=fit,
        applied_basin_ids=[_D001],
        counts=counts,
    )

    out_dir = tmp_path / "out"
    paths = write_imputation_artifacts(out_dir, imputed_df, mask_df, manifest)
    assert set(paths) == {
        "imputed_static_attributes.parquet",
        "imputed_value_mask.parquet",
        "imputation_manifest.json",
    }
    written_manifest = json.loads(paths["imputation_manifest.json"].read_text(encoding="utf-8"))
    assert set(written_manifest["artifact_sha256"]) == {
        "imputed_static_attributes.parquet",
        "imputed_value_mask.parquet",
    }

    roundtrip_df = pd.read_parquet(paths["imputed_static_attributes.parquet"])
    assert roundtrip_df.loc[_D001, "attr_c"] == pytest.approx(fit.fitted_values["attr_c"])


def test_write_imputation_artifacts_refuses_nonempty_dir_without_force(tmp_path):
    df = _matrix_df({_D001: {"attr_a": 1.0, "attr_b": 2.0, "attr_c": 3.0}})
    fit = ImputationFit(
        strategy="median", fit_basin_scope="development_training_only", fit_population_size=1,
        model_input_columns=tuple(MODEL_INPUT_COLS),
        fitted_values={c: 1.0 for c in MODEL_INPUT_COLS},
        n_missing_before_fit={c: 0 for c in MODEL_INPUT_COLS},
        columns_all_nan_in_fit_population=(),
    )
    imputed_df, mask_df, counts = apply_imputation(df, [_D001], fit)
    manifest = {"algorithm_id": "x"}
    out_dir = tmp_path / "out"
    write_imputation_artifacts(out_dir, imputed_df, mask_df, manifest)
    with pytest.raises(StaticPreparationError, match="force"):
        write_imputation_artifacts(out_dir, imputed_df, mask_df, manifest)
    write_imputation_artifacts(out_dir, imputed_df, mask_df, manifest, force=True)

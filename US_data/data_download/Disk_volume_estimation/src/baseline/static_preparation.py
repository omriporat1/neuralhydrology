"""Stage 1 canonical static-attribute preparation + development-only median
imputation (Milestone 2K-G-I primitives increment).

Implements the binding static-attribute imputation policy
(``config/stage1_scientific_baseline_v001.yaml::static_attributes.imputation``,
signed off 2026-07-13; detail in
``docs/stage1_baseline_package_implementation_plan.md`` sec 15): impute
``model_input`` NaNs with per-column medians fitted **only** on the canonical
``development_train`` population, then apply those frozen values unchanged to
any other Stage 1-3 basin subset (validation, temporal test, spatial holdout,
or a subset drawn from them, e.g. the accepted Compact Scientific Package).
Never fits from validation, temporal-test, spatial-holdout, or California
basins.

Small, deterministic, testable functions -- not a class hierarchy, mirroring
the style of :mod:`src.baseline.splits` and
:mod:`src.baseline.compact_selection`. Reuses
:func:`src.baseline.splits.load_matrix_for_splits`,
:func:`src.baseline.splits.join_eligible_with_matrix`, and
:func:`src.baseline.staid.normalize_staid` rather than reimplementing matrix
loading or basin-ID validation.

Does not build NeuralHydrology packages, does not touch ``FlashNHDataset``,
does not train models, and does not invent a new imputation method beyond
the signed-off per-column median.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .splits import SplitGenerationError, join_eligible_with_matrix, load_matrix_for_splits, sha256_of
from .staid import normalize_staid

__all__ = [
    "StaticPreparationError",
    "sha256_of",
    "load_column_manifest",
    "model_input_columns_from_manifest",
    "load_static_matrix",
    "validate_unique_normalized_basin_ids",
    "select_basin_rows",
    "split_model_input_and_metadata",
    "load_development_train_basin_ids",
    "ImputationFit",
    "fit_development_median_imputation",
    "apply_imputation",
    "build_imputation_manifest",
    "write_imputation_artifacts",
]

_ALGORITHM_ID = "stage1_static_median_imputation_v1"
_ALGORITHM_VERSION = 1
_DEFAULT_MODEL_INPUT_ROLE = "model_input"


class StaticPreparationError(ValueError):
    """Raised for an invalid static matrix, manifest, basin list, or fit/apply state."""


# ---------------------------------------------------------------------------
# Column-role manifest + matrix loading
# ---------------------------------------------------------------------------

def load_column_manifest(path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise StaticPreparationError(f"column-role manifest not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if "columns" not in data or not isinstance(data["columns"], dict):
        raise StaticPreparationError(f"{p}: missing/malformed top-level 'columns' mapping")
    return data


def model_input_columns_from_manifest(manifest: dict, role: str = _DEFAULT_MODEL_INPUT_ROLE) -> list:
    """Sorted column names whose manifest role equals ``role`` (default model_input)."""
    cols = sorted(c for c, meta in manifest["columns"].items() if meta.get("role") == role)
    if not cols:
        raise StaticPreparationError(f"no columns with role={role!r} found in column manifest")
    return cols


def load_static_matrix(attributes_parquet, column_manifest_path, role: str = _DEFAULT_MODEL_INPUT_ROLE):
    """Load the canonical static matrix, its manifest, and the model_input column list.

    Returns ``(matrix_df, model_input_cols, manifest)``. ``matrix_df`` is
    indexed by ``gauge_id`` and retains every column present in the parquet
    (not just ``model_input``) so callers can separately preserve non-model
    metadata via :func:`split_model_input_and_metadata`.
    """
    manifest = load_column_manifest(column_manifest_path)
    model_input_cols = model_input_columns_from_manifest(manifest, role=role)
    try:
        matrix_df = load_matrix_for_splits(attributes_parquet, model_input_cols)
    except SplitGenerationError as exc:
        raise StaticPreparationError(str(exc)) from exc
    return matrix_df, model_input_cols, manifest


# ---------------------------------------------------------------------------
# Basin-ID normalization / membership
# ---------------------------------------------------------------------------

def validate_unique_normalized_basin_ids(basin_ids) -> list:
    """Normalize every ID via ``staid.normalize_staid``; fail loud on duplicates."""
    normalized = []
    for raw in basin_ids:
        try:
            normalized.append(normalize_staid(raw))
        except (TypeError, ValueError) as exc:
            raise StaticPreparationError(f"malformed basin id {raw!r}: {exc}") from exc
    if len(normalized) != len(set(normalized)):
        dupes = sorted({s for s in normalized if normalized.count(s) > 1})
        raise StaticPreparationError(f"duplicate normalized basin id(s): {dupes}")
    return normalized


def select_basin_rows(matrix_df: pd.DataFrame, basin_ids) -> pd.DataFrame:
    """Exact one-to-one basin-membership selection (normalize, then join)."""
    normalized = validate_unique_normalized_basin_ids(basin_ids)
    try:
        return join_eligible_with_matrix(matrix_df, normalized)
    except SplitGenerationError as exc:
        raise StaticPreparationError(str(exc)) from exc


def split_model_input_and_metadata(matrix_df: pd.DataFrame, model_input_cols):
    """Return ``(model_input_df, metadata_df)``; metadata_df holds every other column."""
    model_input_cols = list(model_input_cols)
    metadata_cols = [c for c in matrix_df.columns if c not in model_input_cols]
    return matrix_df[model_input_cols].copy(), matrix_df[metadata_cols].copy()


def load_development_train_basin_ids(
    split_assignment_path,
    *,
    split_role_column: str = "split_role",
    staid_column: str = "STAID",
    state_column: str = "STATE",
    required_role: str = "development_train",
    forbidden_state: str = "CA",
) -> list:
    """Load only the canonical development-training basin IDs from a split
    assignment CSV -- the sole permitted imputation-fit population. Fails
    loud if the role is absent, if any matching row is California, or on
    duplicate normalized STAIDs. Self-contained (does not depend on
    :mod:`src.baseline.compact_selection`'s selection-policy schema).
    """
    p = Path(split_assignment_path)
    if not p.is_file():
        raise StaticPreparationError(f"split assignment file not found: {p}")
    df = pd.read_csv(p, dtype={staid_column: str})
    for col in (split_role_column, staid_column, state_column):
        if col not in df.columns:
            raise StaticPreparationError(f"{p}: missing required column {col!r}")

    if required_role not in set(df[split_role_column]):
        raise StaticPreparationError(f"{p}: no rows with {split_role_column}={required_role!r}")

    rows = df.loc[df[split_role_column] == required_role]
    bad_state = rows.loc[rows[state_column].astype(str) == forbidden_state]
    if len(bad_state):
        raise StaticPreparationError(
            f"{len(bad_state)} {forbidden_state} basin(s) found under "
            f"{split_role_column}={required_role!r}: "
            f"{sorted(bad_state[staid_column])[:10]}"
        )
    return validate_unique_normalized_basin_ids(rows[staid_column].tolist())


# ---------------------------------------------------------------------------
# Development-only median-imputation fit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImputationFit:
    """Frozen per-column imputation statistics fit on one basin population."""

    strategy: str
    fit_basin_scope: str
    fit_population_size: int
    model_input_columns: tuple
    fitted_values: dict
    n_missing_before_fit: dict
    columns_all_nan_in_fit_population: tuple


def fit_development_median_imputation(
    matrix_df: pd.DataFrame,
    model_input_cols,
    development_train_basin_ids,
    *,
    strategy: str = "median",
    fail_if_all_nan: bool = True,
) -> ImputationFit:
    """Fit per-column medians using **only** the supplied development-training
    basin subset of ``matrix_df``. Per binding policy, raises
    :class:`StaticPreparationError` if any ``model_input`` column is entirely
    NaN over that fit population and ``fail_if_all_nan`` is True (the
    default, matching
    ``static_attributes.imputation.fail_if_all_nan_in_development_training``).
    """
    if strategy != "median":
        raise StaticPreparationError(f"only strategy='median' is implemented, got {strategy!r}")

    fit_rows = select_basin_rows(matrix_df, development_train_basin_ids)
    model_input_cols = list(model_input_cols)

    fitted_values = {}
    n_missing_before = {}
    all_nan_cols = []
    for col in model_input_cols:
        series = fit_rows[col]
        n_missing_before[col] = int(series.isna().sum())
        if series.isna().all():
            all_nan_cols.append(col)
            fitted_values[col] = float("nan")
            continue
        fitted_values[col] = float(series.median())

    if all_nan_cols and fail_if_all_nan:
        raise StaticPreparationError(
            f"{len(all_nan_cols)} model_input column(s) are all-NaN over the "
            f"development-training fit population ({len(fit_rows)} basins): "
            f"{all_nan_cols[:10]}"
        )

    return ImputationFit(
        strategy=strategy,
        fit_basin_scope="development_training_only",
        fit_population_size=len(fit_rows),
        model_input_columns=tuple(model_input_cols),
        fitted_values=fitted_values,
        n_missing_before_fit=n_missing_before,
        columns_all_nan_in_fit_population=tuple(all_nan_cols),
    )


# ---------------------------------------------------------------------------
# Apply frozen imputation to a target basin subset (never refit)
# ---------------------------------------------------------------------------

def apply_imputation(matrix_df: pd.DataFrame, basin_ids, fit: ImputationFit):
    """Apply ``fit``'s frozen per-column values, unchanged, to the selected
    basin subset (e.g. the 32 Compact Scientific Package basins).

    Returns ``(imputed_df, imputed_mask_df, counts)``:
      - ``imputed_df``: model_input columns only, NaNs filled.
      - ``imputed_mask_df``: same shape, boolean, True where a value was
        imputed (i.e. was NaN before this call) -- the audit mask.
      - ``counts``: ``{column: {"n_missing_before": int, "n_missing_after": int}}``,
        counted over this subset only (independent of the fit population's
        own missing counts, which live on ``fit.n_missing_before_fit``).
    """
    rows = select_basin_rows(matrix_df, basin_ids)
    cols = list(fit.model_input_columns)
    subset = rows[cols].copy()
    imputed_mask_df = subset.isna()

    imputed_df = subset.copy()
    for col in cols:
        if not imputed_mask_df[col].any():
            continue
        value = fit.fitted_values[col]
        if pd.isna(value):
            raise StaticPreparationError(
                f"column {col!r} has missing value(s) in the target subset but no "
                "valid fitted imputation value (all-NaN in the development-training "
                "fit population)"
            )
        imputed_df[col] = imputed_df[col].fillna(value)

    n_missing_after = imputed_df.isna().sum()
    counts = {
        col: {
            "n_missing_before": int(imputed_mask_df[col].sum()),
            "n_missing_after": int(n_missing_after[col]),
        }
        for col in cols
    }
    return imputed_df, imputed_mask_df, counts


# ---------------------------------------------------------------------------
# Manifest / artifact writing
# ---------------------------------------------------------------------------

def build_imputation_manifest(
    *,
    attributes_parquet_path,
    column_manifest_path,
    fit: ImputationFit,
    applied_basin_ids,
    counts: dict,
) -> dict:
    applied_normalized = validate_unique_normalized_basin_ids(applied_basin_ids)
    return {
        "algorithm_id": _ALGORITHM_ID,
        "algorithm_version": _ALGORITHM_VERSION,
        "input_matrix_path": str(attributes_parquet_path),
        "input_matrix_sha256": sha256_of(attributes_parquet_path),
        "column_manifest_path": str(column_manifest_path),
        "column_manifest_sha256": sha256_of(column_manifest_path),
        "fit_strategy": fit.strategy,
        "fit_basin_scope": fit.fit_basin_scope,
        "fit_population_size": fit.fit_population_size,
        "n_applied_basins": len(applied_normalized),
        "columns_all_nan_in_fit_population": list(fit.columns_all_nan_in_fit_population),
        "per_column": {
            col: {
                "method": fit.strategy,
                "fitted_value": fit.fitted_values[col],
                "n_missing_before_fit_in_development_training": fit.n_missing_before_fit[col],
                "n_missing_before_apply": counts[col]["n_missing_before"],
                "n_missing_after_apply": counts[col]["n_missing_after"],
            }
            for col in fit.model_input_columns
        },
    }


def write_imputation_artifacts(out_dir, imputed_df: pd.DataFrame, imputed_mask_df: pd.DataFrame, manifest: dict, force: bool = False) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise StaticPreparationError(
            f"output directory already exists and is non-empty: {out_dir} (use --force)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict = {}

    matrix_path = out_dir / "imputed_static_attributes.parquet"
    imputed_df.to_parquet(matrix_path)
    paths["imputed_static_attributes.parquet"] = matrix_path

    mask_path = out_dir / "imputed_value_mask.parquet"
    imputed_mask_df.to_parquet(mask_path)
    paths["imputed_value_mask.parquet"] = mask_path

    artifact_sha256 = {name: sha256_of(p) for name, p in sorted(paths.items())}
    manifest_to_write = {**manifest, "artifact_sha256": artifact_sha256}
    manifest_path = out_dir / "imputation_manifest.json"
    manifest_path.write_text(json.dumps(manifest_to_write, indent=2, default=str), encoding="utf-8")
    paths["imputation_manifest.json"] = manifest_path

    return paths

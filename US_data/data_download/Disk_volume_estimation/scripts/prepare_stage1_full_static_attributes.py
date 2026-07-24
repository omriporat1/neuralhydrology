#!/usr/bin/env python3
"""Stage 1 full non-California static-attribute preparation.

Fits development-training-only median imputation and development-training-
only zero-variance trainability projection (both implemented in
:mod:`src.baseline.static_preparation`) and applies each frozen fit,
unchanged, to the full 2,557-basin non-California package population
(2,307 development-training + 250 spatial-holdout basins). Never fits from
the spatial-holdout basins. See docs/decision_log.md and
docs/stage1_baseline_package_implementation_plan.md for the binding
imputation and trainability-projection policy this orchestrates.

This is a thin orchestration script: it calls existing library functions
from src.baseline.static_preparation and src.baseline.splits and does not
reimplement their logic. It does not build a NeuralHydrology package, does
not touch FlashNHDataset, and does not train or evaluate a model.

The canonical 473-column imputed table remains the package's static-input
contract; the retained-only table is a downstream run/config convenience
artifact, not a schema change.

Usage (h2o):
  cd /data42/omrip/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation

  source /opt/conda/etc/profile.d/conda.sh
  conda activate /data42/omrip/Flash-NH/envs/flashnh-stage1

  python scripts/prepare_stage1_full_static_attributes.py \\
    --static-matrix              /data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v002/stage1_static_attributes_v002.parquet \\
    --column-manifest            /data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v002/stage1_static_attributes_v002_column_manifest.json \\
    --development-basin-list     config/stage1_baseline_splits_v001/development_train.txt \\
    --spatial-holdout-basin-list config/stage1_baseline_splits_v001/spatial_holdout_nonca.txt \\
    --output-dir                 /data42/omrip/Flash-NH/tmp/stage1_full_static_attributes_v001 \\
    --force
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.baseline.splits import load_eligible_basins, sha256_of  # noqa: E402
from src.baseline.static_preparation import (  # noqa: E402
    StaticPreparationError,
    apply_imputation,
    apply_zero_variance_projection,
    build_imputation_manifest,
    build_zero_variance_manifest,
    fit_development_median_imputation,
    fit_zero_variance_projection,
    load_static_matrix,
    write_imputation_artifacts,
    write_zero_variance_manifest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Pinned Stage 1 production counts (binding scientific contract). Never
# exposed as CLI arguments -- see the internal helpers below for the only
# place these are overridable, and only for tests against synthetic data.
EXPECTED_DEVELOPMENT_COUNT = 2307
EXPECTED_SPATIAL_HOLDOUT_COUNT = 250
EXPECTED_TOTAL_COUNT = 2557
EXPECTED_MODEL_INPUT_COUNT = 473


# ---------------------------------------------------------------------------
# Internal validation helpers (expected counts overridable here only, for
# tests against small synthetic populations -- never wired to the CLI)
# ---------------------------------------------------------------------------

def _validate_candidate_column_count(
    n_candidate_columns: int,
    *,
    expected_model_input_count: int = EXPECTED_MODEL_INPUT_COUNT,
) -> None:
    if n_candidate_columns != expected_model_input_count:
        raise StaticPreparationError(
            f"candidate model_input column count {n_candidate_columns} != "
            f"expected {expected_model_input_count}"
        )


def _validate_population(
    development_ids: list,
    holdout_ids: list,
    *,
    expected_development_count: int = EXPECTED_DEVELOPMENT_COUNT,
    expected_holdout_count: int = EXPECTED_SPATIAL_HOLDOUT_COUNT,
    expected_total_count: int = EXPECTED_TOTAL_COUNT,
) -> list:
    """Validate development/holdout counts, reject overlap, and return the
    ordered (sorted) package population -- the union of both lists."""
    if len(development_ids) != expected_development_count:
        raise StaticPreparationError(
            f"development basin count {len(development_ids)} != "
            f"expected {expected_development_count}"
        )
    if len(holdout_ids) != expected_holdout_count:
        raise StaticPreparationError(
            f"spatial-holdout basin count {len(holdout_ids)} != "
            f"expected {expected_holdout_count}"
        )
    overlap = sorted(set(development_ids) & set(holdout_ids))
    if overlap:
        raise StaticPreparationError(
            f"development and spatial-holdout basin lists overlap "
            f"({len(overlap)} basin(s)): {overlap[:10]}"
        )
    package_ids = sorted(set(development_ids) | set(holdout_ids))
    if len(package_ids) != expected_total_count:
        raise StaticPreparationError(
            f"package population count {len(package_ids)} != "
            f"expected {expected_total_count}"
        )
    return package_ids


def _check_output_dir(out_dir: Path, force: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise StaticPreparationError(
            f"output directory already exists and is non-empty: {out_dir} (use --force)"
        )


def _write_column_list(path: Path, columns) -> None:
    path.write_text("\n".join(columns) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def prepare_full_static_attributes(
    *,
    static_matrix_path,
    column_manifest_path,
    development_basin_list_path,
    spatial_holdout_basin_list_path,
    output_dir,
    force: bool = False,
    expected_development_count: int = EXPECTED_DEVELOPMENT_COUNT,
    expected_holdout_count: int = EXPECTED_SPATIAL_HOLDOUT_COUNT,
    expected_total_count: int = EXPECTED_TOTAL_COUNT,
    expected_model_input_count: int = EXPECTED_MODEL_INPUT_COUNT,
) -> dict:
    """Fit development-only imputation + development-only zero-variance
    projection and apply both, frozen, to the full non-California package
    population. Writes deterministic output artifacts to ``output_dir`` and
    returns the run-summary dict (also written as run_summary.json).

    ``expected_*`` are overridable only for tests against small synthetic
    populations; the CLI (main()) always calls this with the pinned Stage 1
    production defaults.
    """
    out_dir = Path(output_dir)
    _check_output_dir(out_dir, force)

    log.info("Loading canonical static matrix + column manifest ...")
    matrix_df, model_input_cols, _manifest = load_static_matrix(
        static_matrix_path, column_manifest_path
    )
    _validate_candidate_column_count(
        len(model_input_cols), expected_model_input_count=expected_model_input_count
    )
    log.info(
        "Matrix: %d basins, %d candidate model_input columns",
        len(matrix_df), len(model_input_cols),
    )

    log.info("Loading development-training and spatial-holdout basin lists ...")
    development_ids = load_eligible_basins(development_basin_list_path)
    holdout_ids = load_eligible_basins(spatial_holdout_basin_list_path)
    package_ids = _validate_population(
        development_ids,
        holdout_ids,
        expected_development_count=expected_development_count,
        expected_holdout_count=expected_holdout_count,
        expected_total_count=expected_total_count,
    )
    log.info(
        "development_train=%d spatial_holdout_nonca=%d package_population=%d",
        len(development_ids), len(holdout_ids), len(package_ids),
    )

    log.info("Fitting development-training-only median imputation ...")
    imputation_fit = fit_development_median_imputation(
        matrix_df, model_input_cols, development_ids
    )
    log.info(
        "Imputation fit: %d columns, fit_population_size=%d",
        len(imputation_fit.model_input_columns), imputation_fit.fit_population_size,
    )

    log.info("Applying frozen imputation to the full %d-basin package ...", len(package_ids))
    imputed_df, imputed_mask_df, impute_counts = apply_imputation(
        matrix_df, package_ids, imputation_fit
    )

    if not np.all(np.isfinite(imputed_df.to_numpy(dtype=float))):
        raise StaticPreparationError(
            "imputed static matrix contains non-finite value(s) after applying "
            "the frozen development-training imputation -- refusing to proceed"
        )

    imputation_manifest = build_imputation_manifest(
        attributes_parquet_path=static_matrix_path,
        column_manifest_path=column_manifest_path,
        fit=imputation_fit,
        applied_basin_ids=package_ids,
        counts=impute_counts,
    )

    log.info("Fitting development-training-only zero-variance trainability projection ...")
    zero_variance_fit = fit_zero_variance_projection(
        imputed_df, development_ids, model_input_cols
    )
    log.info(
        "Zero-variance fit: %d retained, %d excluded (of %d candidates)",
        len(zero_variance_fit.retained_columns),
        len(zero_variance_fit.excluded_columns),
        len(zero_variance_fit.candidate_columns),
    )

    retained_df = apply_zero_variance_projection(imputed_df, zero_variance_fit)

    zero_variance_manifest = build_zero_variance_manifest(
        column_manifest_path=column_manifest_path,
        fit=zero_variance_fit,
        development_train_basin_list_path=development_basin_list_path,
        imputation_manifest=imputation_manifest,
    )

    log.info("Writing output artifacts to %s ...", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imputation_paths = write_imputation_artifacts(
        out_dir, imputed_df, imputed_mask_df, imputation_manifest, force=True
    )

    retained_path = out_dir / "retained_static_attributes.parquet"
    retained_df.to_parquet(retained_path)

    zero_variance_manifest_path = write_zero_variance_manifest(
        out_dir / "zero_variance_manifest.json", zero_variance_manifest, force=True
    )

    retained_columns_path = out_dir / "retained_static_columns.txt"
    _write_column_list(retained_columns_path, zero_variance_fit.retained_columns)

    excluded_columns_path = out_dir / "excluded_zero_variance_columns.txt"
    _write_column_list(excluded_columns_path, zero_variance_fit.excluded_columns)

    output_paths = {
        "imputed_static_attributes.parquet": imputation_paths["imputed_static_attributes.parquet"],
        "imputed_value_mask.parquet": imputation_paths["imputed_value_mask.parquet"],
        "imputation_manifest.json": imputation_paths["imputation_manifest.json"],
        "retained_static_attributes.parquet": retained_path,
        "zero_variance_manifest.json": zero_variance_manifest_path,
        "retained_static_columns.txt": retained_columns_path,
        "excluded_zero_variance_columns.txt": excluded_columns_path,
    }
    output_sha256 = {name: sha256_of(p) for name, p in sorted(output_paths.items())}

    fit_basin_scopes = {imputation_fit.fit_basin_scope, zero_variance_fit.fit_basin_scope}
    if fit_basin_scopes != {"development_training_only"}:
        raise StaticPreparationError(
            f"unexpected fit_basin_scope(s), expected only development_training_only: "
            f"{fit_basin_scopes}"
        )

    summary = {
        "static_matrix_path": str(static_matrix_path),
        "static_matrix_sha256": sha256_of(static_matrix_path),
        "column_manifest_path": str(column_manifest_path),
        "column_manifest_sha256": sha256_of(column_manifest_path),
        "development_basin_list_path": str(development_basin_list_path),
        "development_basin_list_sha256": sha256_of(development_basin_list_path),
        "spatial_holdout_basin_list_path": str(spatial_holdout_basin_list_path),
        "spatial_holdout_basin_list_sha256": sha256_of(spatial_holdout_basin_list_path),
        "development_basin_count": len(development_ids),
        "spatial_holdout_basin_count": len(holdout_ids),
        "package_population_count": len(package_ids),
        "candidate_column_count": len(model_input_cols),
        "retained_column_count": len(zero_variance_fit.retained_columns),
        "excluded_column_count": len(zero_variance_fit.excluded_columns),
        "retained_columns": list(zero_variance_fit.retained_columns),
        "excluded_columns": list(zero_variance_fit.excluded_columns),
        "fit_basin_scope": "development_training_only",
        "output_dir": str(out_dir),
        "output_paths": {name: str(p) for name, p in sorted(output_paths.items())},
        "output_sha256": output_sha256,
    }

    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["run_summary_path"] = str(summary_path)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--static-matrix", required=True,
                   help="Canonical stage1_static_attributes_v002 parquet.")
    p.add_argument("--column-manifest", required=True,
                   help="Canonical static column-role manifest JSON.")
    p.add_argument("--development-basin-list", required=True,
                   help="Newline-delimited development_train basin-ID file (2,307 basins).")
    p.add_argument("--spatial-holdout-basin-list", required=True,
                   help="Newline-delimited spatial_holdout_nonca basin-ID file (250 basins).")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for all generated artifacts.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite output-dir contents if already populated.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        summary = prepare_full_static_attributes(
            static_matrix_path=args.static_matrix,
            column_manifest_path=args.column_manifest,
            development_basin_list_path=args.development_basin_list,
            spatial_holdout_basin_list_path=args.spatial_holdout_basin_list,
            output_dir=args.output_dir,
            force=args.force,
        )
    except StaticPreparationError as exc:
        log.error("Static preparation failed: %s", exc)
        sys.exit(1)

    log.info(
        "PASS: development=%d holdout=%d package=%d candidate=%d retained=%d excluded=%d",
        summary["development_basin_count"],
        summary["spatial_holdout_basin_count"],
        summary["package_population_count"],
        summary["candidate_column_count"],
        summary["retained_column_count"],
        summary["excluded_column_count"],
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

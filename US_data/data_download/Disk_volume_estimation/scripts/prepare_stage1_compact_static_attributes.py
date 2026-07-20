#!/usr/bin/env python3
"""Real-data audit CLI for Stage 1 compact-package static-attribute
imputation (Milestone 2K-G-I primitives increment).

Fits per-column median imputation on the canonical development-training
basin population, then applies those frozen values to a target basin
subset (e.g. the accepted 32-basin Compact Scientific Package). Never
fits from validation, temporal-test, spatial-holdout, or California basins.
See docs/decision_log.md (2026-07-20 entry) and
config/stage1_scientific_baseline_v001.yaml::static_attributes.imputation
for the signed-off policy this CLI implements.

This script performs no package build, no NeuralHydrology step, and no
training. It is an audit/preparation utility only, intended to be run
against canonical h2o paths -- NOT run from a local session.

Usage (h2o):
  cd /data42/omrip/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation

  source /opt/conda/etc/profile.d/conda.sh
  conda activate /data42/omrip/Flash-NH/envs/flashnh-stage1

  python scripts/prepare_stage1_compact_static_attributes.py \\
    --attributes-parquet /data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v002/stage1_static_attributes_v002.parquet \\
    --column-manifest    /data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v002/stage1_static_attributes_v002_column_manifest.json \\
    --split-assignment   config/stage1_baseline_splits_v001/split_assignment.csv \\
    --target-basins       /data42/omrip/Flash-NH/tmp/stage1_compact_package_selection_v001_evidence/compact_basin_ids.txt \\
    --out-dir             /data42/omrip/Flash-NH/tmp/stage1_compact_static_imputation_v002/ \\
    --force
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.baseline.splits import load_eligible_basins  # noqa: E402
from src.baseline.static_preparation import (  # noqa: E402
    StaticPreparationError,
    apply_imputation,
    build_imputation_manifest,
    fit_development_median_imputation,
    load_development_train_basin_ids,
    load_static_matrix,
    write_imputation_artifacts,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--attributes-parquet", required=True,
        help="Canonical static-attribute matrix parquet (gauge_id-indexed).",
    )
    p.add_argument(
        "--column-manifest", required=True,
        help="Canonical static column-role manifest JSON.",
    )
    p.add_argument(
        "--split-assignment", required=True,
        help="split_assignment.csv (columns: STAID, STATE, split_role); "
             "development_train rows define the imputation fit population.",
    )
    p.add_argument(
        "--target-basins", required=True,
        help="Newline-delimited basin-ID file the frozen imputation is applied "
             "to (e.g. compact_basin_ids.txt).",
    )
    p.add_argument(
        "--out-dir", required=True,
        help="Output directory for imputed_static_attributes.parquet, "
             "imputed_value_mask.parquet, imputation_manifest.json.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite out-dir if it already exists and is non-empty.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate inputs and fit imputation; print a summary; write nothing.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    log.info("Loading static matrix + column manifest ...")
    matrix_df, model_input_cols, _manifest = load_static_matrix(
        args.attributes_parquet, args.column_manifest
    )
    log.info("Matrix: %d basins, %d model_input columns", len(matrix_df), len(model_input_cols))

    log.info("Loading development-training fit population ...")
    dev_train_ids = load_development_train_basin_ids(args.split_assignment)
    log.info("development_train population: %d basins", len(dev_train_ids))

    log.info("Loading target basin list ...")
    target_ids = load_eligible_basins(args.target_basins)
    log.info("target basins: %d", len(target_ids))

    try:
        fit = fit_development_median_imputation(matrix_df, model_input_cols, dev_train_ids)
    except StaticPreparationError as exc:
        log.error("Imputation fit failed: %s", exc)
        sys.exit(1)

    total_missing_before_fit = sum(fit.n_missing_before_fit.values())
    log.info(
        "Fit complete: %d columns, %d total missing values in fit population",
        len(fit.model_input_columns), total_missing_before_fit,
    )

    try:
        imputed_df, imputed_mask_df, counts = apply_imputation(matrix_df, target_ids, fit)
    except StaticPreparationError as exc:
        log.error("Applying imputation to target basins failed: %s", exc)
        sys.exit(1)

    n_imputed_values = int(imputed_mask_df.to_numpy().sum())
    n_basins_with_any_imputation = int((imputed_mask_df.sum(axis=1) > 0).sum())
    log.info(
        "Applied to %d target basins: %d values imputed across %d basins",
        len(target_ids), n_imputed_values, n_basins_with_any_imputation,
    )

    manifest = build_imputation_manifest(
        attributes_parquet_path=args.attributes_parquet,
        column_manifest_path=args.column_manifest,
        fit=fit,
        applied_basin_ids=target_ids,
        counts=counts,
    )

    if args.dry_run:
        log.info("--dry-run set: not writing artifacts.")
        return

    paths = write_imputation_artifacts(
        args.out_dir, imputed_df, imputed_mask_df, manifest, force=args.force
    )
    for name, path in sorted(paths.items()):
        log.info("Wrote %s -> %s", name, path)


if __name__ == "__main__":
    main()

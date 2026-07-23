#!/usr/bin/env python
"""Thin CLI wiring for the Gate 3A Compact Scientific Package builder.

Argument parsing and I/O wiring only -- all scientific logic lives in
:mod:`src.baseline.package_builder` (Gate 3A), :mod:`src.baseline.package_assembly`
(Gate 1), and :mod:`src.baseline.package_netcdf` (Gate 2). This script does not
duplicate any of that logic.

This is the local-orchestration CLI (Gate 3A). It has no h2o/Moriah-specific
behavior; it only assumes local filesystem paths supplied via arguments.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline.gap_mask_io import load_missing_hour_products, select_gap_timestamps  # noqa: E402
from src.baseline.package_builder import (  # noqa: E402
    default_local_basin_source_loader,
    derive_expected_index_from_policy,
    read_area_csv,
    read_basin_ids_file,
    resolve_gap_product_scope,
    build_compact_scientific_package,
)
from src.baseline.package_netcdf import (  # noqa: E402
    REGISTERED_PACKAGE_NETCDF_SCHEMAS,
    resolve_package_netcdf_schema,
)
from src.baseline.policy import load_stage1_baseline_policy, validate_stage1_baseline_policy  # noqa: E402
from src.baseline.splits import sha256_of  # noqa: E402
from src.baseline.static_preparation import load_static_matrix  # noqa: E402

_REGISTERED_PACKAGE_SCHEMA_NAMES = tuple(s.name for s in REGISTERED_PACKAGE_NETCDF_SCHEMAS)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-yaml", required=True, help="Path to the validated Stage 1 policy YAML.")
    parser.add_argument(
        "--package-schema",
        required=True,
        choices=_REGISTERED_PACKAGE_SCHEMA_NAMES,
        help=(
            "Registered on-disk NetCDF package schema to build every basin file with "
            "(see src.baseline.package_netcdf). Must be selected explicitly -- there is "
            "no default -- so a production build can never silently produce a legacy "
            f"'time'-coordinate package. One of: {', '.join(_REGISTERED_PACKAGE_SCHEMA_NAMES)}."
        ),
    )
    parser.add_argument(
        "--basin-ids-file", required=True, help="Text file with one basin ID per line (exact package membership)."
    )
    parser.add_argument(
        "--static-attributes-parquet", required=True, help="Prepared compact static matrix (indexed by gauge_id)."
    )
    parser.add_argument("--static-column-manifest", required=True, help="Column-role manifest JSON.")
    parser.add_argument(
        "--static-preparation-manifest",
        default=None,
        help=(
            "Optional imputation_manifest.json for the prepared compact static artifact "
            "(as written by static_preparation.write_imputation_artifacts). When supplied, "
            "its declared artifact_sha256['imputed_static_attributes.parquet'] checksum is "
            "cross-checked against the actual --static-attributes-parquet file (computed by "
            "this CLI, never trusted from the manifest)."
        ),
    )
    parser.add_argument(
        "--area-csv", required=True, help="CSV with columns gauge_id,DRAIN_SQKM (basin drainage areas)."
    )
    parser.add_argument("--forcing-root", required=True, help="Root containing time_series/<id>.parquet forcing files.")
    parser.add_argument("--qobs-root", required=True, help="Root containing time_series/<id>.nc streamflow files.")
    parser.add_argument("--gap-inventory-csv", required=True, help="Missing-hour-products inventory CSV.")
    parser.add_argument("--output-package-root", required=True, help="Destination package directory.")
    parser.add_argument("--evidence-root", default=None, help="Destination QC-evidence directory (non-authoritative).")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing package/evidence dir.")
    parser.add_argument("--write-qc-csv", action="store_true", help="Also export per-basin QC CSVs to --evidence-root.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and build in a temp dir but do not promote.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    package_netcdf_schema = resolve_package_netcdf_schema(args.package_schema)

    policy = validate_stage1_baseline_policy(load_stage1_baseline_policy(args.policy_yaml))
    expected_index = derive_expected_index_from_policy(policy)

    basin_ids = read_basin_ids_file(args.basin_ids_file)

    static_matrix, model_input_columns, static_column_manifest = load_static_matrix(
        args.static_attributes_parquet, args.static_column_manifest
    )
    static_attributes = static_matrix[model_input_columns]

    area_by_basin = read_area_csv(args.area_csv, basin_ids=basin_ids)

    gap_product_scope = resolve_gap_product_scope(policy)
    gap_df = load_missing_hour_products(args.gap_inventory_csv)
    gap_timestamps = select_gap_timestamps(gap_df, products=gap_product_scope)

    loader = default_local_basin_source_loader(
        args.forcing_root, args.qobs_root, area_by_basin, dynamic_inputs=policy["dynamic_inputs"]
    )

    # Canonical population-matrix identity, as declared by the validated
    # policy -- independent of the prepared/compact artifact actually
    # supplied via --static-attributes-parquet below (see
    # package_builder._validate_prepared_static_artifact).
    canonical_static_source_provenance = {
        "matrix_name": policy["static_attributes"]["matrix_name"],
        "sha256": policy["static_attributes"]["sha256"],
    }

    # Prepared artifact identity: always computed by this CLI from the
    # actual file bytes, never trusted from a caller-supplied value.
    prepared_static_attributes_provenance = {
        "static_attributes_parquet": str(Path(args.static_attributes_parquet)),
        "sha256": sha256_of(Path(args.static_attributes_parquet)),
    }

    static_preparation_manifest = None
    if args.static_preparation_manifest is not None:
        manifest_path = Path(args.static_preparation_manifest)
        if not manifest_path.is_file():
            raise SystemExit(f"--static-preparation-manifest not found: {manifest_path}")
        static_preparation_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prepared_static_attributes_provenance["static_preparation_manifest"] = str(manifest_path)

    result = build_compact_scientific_package(
        basin_ids=basin_ids,
        load_basin_source=loader,
        static_attributes=static_attributes,
        static_model_input_columns=model_input_columns,
        static_column_manifest=static_column_manifest,
        gap_timestamps=gap_timestamps,
        gap_product_scope=gap_product_scope,
        expected_index=expected_index,
        output_package_root=args.output_package_root,
        evidence_root=args.evidence_root,
        policy=policy,
        write_qc_csv=args.write_qc_csv,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        policy_provenance={"policy_yaml": str(Path(args.policy_yaml))},
        static_attributes_provenance=canonical_static_source_provenance,
        prepared_static_attributes_provenance=prepared_static_attributes_provenance,
        static_preparation_manifest=static_preparation_manifest,
        basin_selection_provenance={"basin_ids_file": str(Path(args.basin_ids_file))},
        gap_inventory_provenance={"gap_inventory_csv": str(Path(args.gap_inventory_csv))},
        package_netcdf_schema=package_netcdf_schema,
    )

    print(
        f"Package build {'validated (dry-run, not promoted)' if result.dry_run else 'complete'}: "
        f"{len(result.basin_ids)} basins -> {result.package_root}"
    )
    if result.evidence_root is not None:
        print(f"QC evidence written to: {result.evidence_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Gate 4 CLI: independent audit of the Stage 1 Compact Scientific Package.

Argument parsing and I/O wiring only. All independent verification logic
lives in :mod:`src.baseline.package_audit`; this script does not duplicate
any of it and does not import any package-construction module
(``package_builder``, ``package_assembly``, ``package_netcdf``, ``units``,
``lead_targets``). See ``docs/stage1_compact_package_independent_audit.md``
for the full independence boundary and command templates.

This is a local-orchestration CLI: it has no h2o/Moriah-specific behavior and
only assumes local filesystem paths supplied via arguments. It never modifies
or rebuilds the package it audits.

Two modes:
  --mode preflight  Existence/readability check of the package layout and
                     every supplied path. No NetCDF/parquet content is read
                     and no values are compared. Fast; safe to run first.
  --mode full       The complete Gate 4 independent audit (see the 14
                     objectives in the design doc).

Exit status is 0 iff the requested checks all pass (PASS); nonzero (1) if
any check fails (FAIL); 2 on a setup/usage error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline.package_audit import (  # noqa: E402
    PackageAuditError,
    run_audit,
    run_preflight,
    write_audit_outputs,
)
from src.baseline.policy import load_stage1_baseline_policy, validate_stage1_baseline_policy  # noqa: E402


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=["preflight", "full"], help="Which audit mode to run.")
    parser.add_argument("--package-root", required=True, help="Path to the built Compact Scientific Package root.")
    parser.add_argument("--output-dir", required=True, help="Destination directory for generated audit outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty output dir.")

    parser.add_argument("--policy-yaml", required=True, help="Path to the validated Stage 1 scientific policy YAML.")
    parser.add_argument("--basin-selection-file", default=None, help="Accepted basin-ids file (one id per line).")
    parser.add_argument("--prepared-static-parquet", default=None, help="Prepared compact static matrix used to build the package.")
    parser.add_argument("--static-column-manifest", default=None, help="Column-role manifest JSON for the static matrix.")
    parser.add_argument(
        "--imputation-manifest",
        default=None,
        help="imputation_manifest.json (as written by static_preparation.write_imputation_artifacts). "
        "Required for --mode full.",
    )
    parser.add_argument(
        "--imputed-value-mask",
        default=None,
        help="imputed_value_mask.parquet, used with --imputation-manifest to verify imputation placement. "
        "Required for --mode full.",
    )
    parser.add_argument("--forcing-root", default=None, help="Root containing time_series/<id>.parquet forcing files.")
    parser.add_argument("--qobs-root", default=None, help="Root containing time_series/<id>.nc streamflow files.")
    parser.add_argument("--area-csv", default=None, help="CSV with columns gauge_id,DRAIN_SQKM.")
    parser.add_argument("--gap-inventory-csv", default=None, help="Missing-hour-products inventory CSV.")
    parser.add_argument(
        "--qc-evidence-root",
        default=None,
        help="Non-authoritative QC evidence root; cross-checked, never trusted. Required for --mode full.",
    )
    parser.add_argument(
        "--build-git-commit",
        default=None,
        help="Git commit of the package-builder code that produced --package-root (bound into the audit manifest).",
    )
    return parser.parse_args(argv)


_FULL_MODE_REQUIRED = (
    "basin_selection_file",
    "prepared_static_parquet",
    "static_column_manifest",
    "imputation_manifest",
    "imputed_value_mask",
    "forcing_root",
    "qobs_root",
    "area_csv",
    "gap_inventory_csv",
    "qc_evidence_root",
    "build_git_commit",
)


def main(argv=None) -> int:
    args = _parse_args(argv)

    if args.mode == "full":
        missing = [name for name in _FULL_MODE_REQUIRED if getattr(args, name) is None]
        if missing:
            flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            print(f"--mode full requires: {flags}", file=sys.stderr)
            return 2

    try:
        policy = validate_stage1_baseline_policy(load_stage1_baseline_policy(args.policy_yaml))
    except Exception as exc:
        print(f"failed to load/validate --policy-yaml: {exc}", file=sys.stderr)
        return 2

    log_lines = [f"mode={args.mode}", f"package_root={args.package_root}"]

    try:
        if args.mode == "preflight":
            report, diagnostics = run_preflight(
                package_root=args.package_root,
                policy_path=args.policy_yaml,
                basin_selection_path=args.basin_selection_file,
                prepared_static_parquet_path=args.prepared_static_parquet,
                static_column_manifest_path=args.static_column_manifest,
                imputation_manifest_path=args.imputation_manifest,
                imputed_value_mask_path=args.imputed_value_mask,
                forcing_root=args.forcing_root,
                qobs_root=args.qobs_root,
                area_csv_path=args.area_csv,
                gap_inventory_csv_path=args.gap_inventory_csv,
                qc_evidence_root=args.qc_evidence_root,
            )
            diagnostics["audit_manifest"] = None
        else:
            audit_command = " ".join(["audit_stage1_compact_scientific_package.py"] + (argv if argv is not None else sys.argv[1:]))
            report, diagnostics = run_audit(
                package_root=args.package_root,
                policy=policy,
                policy_path=args.policy_yaml,
                basin_selection_path=args.basin_selection_file,
                prepared_static_parquet_path=args.prepared_static_parquet,
                static_column_manifest_path=args.static_column_manifest,
                forcing_root=args.forcing_root,
                qobs_root=args.qobs_root,
                area_csv_path=args.area_csv,
                gap_inventory_csv_path=args.gap_inventory_csv,
                imputation_manifest_path=args.imputation_manifest,
                imputed_value_mask_path=args.imputed_value_mask,
                qc_evidence_root=args.qc_evidence_root,
                build_git_commit=args.build_git_commit,
                audit_command=audit_command,
            )
    except PackageAuditError as exc:
        print(f"audit could not run: {exc}", file=sys.stderr)
        return 2

    log_lines.append(f"status={report.status}")
    log_lines.append(f"errors={report.error_count} warnings={report.warning_count} ok={report.ok_count}")

    written = write_audit_outputs(args.output_dir, report, diagnostics, overwrite=args.overwrite, log_lines=log_lines)

    print(f"Audit ({args.mode}) status: {report.status}")
    print(f"errors={report.error_count} warnings={report.warning_count} ok={report.ok_count}")
    print(f"Outputs written to: {written['audit_results.json'].parent}")
    if report.error_count:
        print("Failed checks:", file=sys.stderr)
        for msg in report.failed_messages():
            print(f"  - {msg}", file=sys.stderr)

    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

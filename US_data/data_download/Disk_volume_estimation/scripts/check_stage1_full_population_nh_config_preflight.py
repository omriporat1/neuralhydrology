#!/usr/bin/env python
"""Structural preflight for the full-population Stage 1 dual config bundle
(development + spatial-holdout, local implementation increment, no training).

Argument parsing and I/O wiring only. All check logic lives in
:func:`src.baseline.nh_structural_preflight.run_full_population_structural_preflight`,
which composes the existing single-bundle structural checks
(:func:`check_generated_config_structure`) for BOTH bundles plus, when
dataset construction is not skipped:

  1. Real ``FlashNHDataset`` train/validation/test construction against the
     development bundle (fitting the scaler once, on development-train only).
  2. Real ``FlashNHDataset`` test-only construction against the
     spatial-holdout bundle, reusing the development scaler unchanged and
     never fitting a new one and never touching the holdout config's
     ``train_dir``.

Both bundles are rendered from the same policy/common_kwargs, so they share
one set of expected train/validation/test start/end dates (--expected-*-date
below) -- there is no separate holdout date set.

The expected development/spatial-holdout basin counts (2,307 / 250) are
pinned constants inside ``nh_structural_preflight.py`` -- mirroring the
never-CLI-exposed convention used by ``prepare_stage1_full_static_attributes.py``
-- and are not accepted as CLI flags here.

Exit status is 0 iff every requested check passes (PASS); 1 if any check
fails (FAIL); 2 on a setup/usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline.nh_structural_preflight import (  # noqa: E402
    NHStructuralPreflightError,
    run_full_population_structural_preflight,
)


def _fail(message: str) -> int:
    print(f"FATAL: {message}", file=sys.stderr)
    return 2


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--development-generated-dir", required=True,
                   help="development/ output directory from generate_stage1_full_population_nh_config.py")
    p.add_argument("--spatial-holdout-generated-dir", required=True,
                   help="spatial_holdout/ output directory from generate_stage1_full_population_nh_config.py")
    p.add_argument("--package-root", required=True, help="Package root both bundles were rendered against")
    p.add_argument("--expected-seq-length", type=int, required=True)
    p.add_argument("--expected-target-variable", required=True)
    p.add_argument("--expected-dynamic-inputs", nargs="+", required=True)
    p.add_argument("--expected-static-column-count", type=int, required=True)
    p.add_argument("--expected-predict-last-n", type=int, required=True)
    p.add_argument("--expected-train-start-date", required=True)
    p.add_argument("--expected-train-end-date", required=True)
    p.add_argument("--expected-validation-start-date", required=True)
    p.add_argument("--expected-validation-end-date", required=True)
    p.add_argument("--expected-test-start-date", required=True)
    p.add_argument("--expected-test-end-date", required=True)
    p.add_argument("--repo-root", default=str(REPO_ROOT), help="Repo root, used for the tracked-output-path safety check")
    p.add_argument(
        "--skip-dataset-construction",
        action="store_true",
        help="Skip the real FlashNHDataset construction layers even if the structural checks pass",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    expected_dates = {
        "train_start_date": args.expected_train_start_date,
        "train_end_date": args.expected_train_end_date,
        "validation_start_date": args.expected_validation_start_date,
        "validation_end_date": args.expected_validation_end_date,
        "test_start_date": args.expected_test_start_date,
        "test_end_date": args.expected_test_end_date,
    }

    try:
        report = run_full_population_structural_preflight(
            development_generated_dir=args.development_generated_dir,
            spatial_holdout_generated_dir=args.spatial_holdout_generated_dir,
            package_root=args.package_root,
            expected_seq_length=args.expected_seq_length,
            expected_target_variable=args.expected_target_variable,
            expected_dynamic_inputs=args.expected_dynamic_inputs,
            expected_static_column_count=args.expected_static_column_count,
            expected_predict_last_n=args.expected_predict_last_n,
            expected_dates=expected_dates,
            repo_root=args.repo_root,
            run_dataset_construction=not args.skip_dataset_construction,
        )
    except NHStructuralPreflightError as exc:
        return _fail(str(exc))

    summary = {
        "status": report.status,
        "error_count": report.error_count,
        "warning_count": report.warning_count,
        "ok_count": report.ok_count,
        "checks": [
            {"severity": r.severity, "check_id": r.check_id, "message": r.message} for r in report.records
        ],
    }
    print(json.dumps(summary, indent=2))
    print(f"Preflight status: {report.status}")
    print(f"errors={report.error_count} warnings={report.warning_count} ok={report.ok_count}")
    if report.error_count:
        print("Failed checks:", file=sys.stderr)
        for msg in report.failed_messages():
            print(f"  - {msg}", file=sys.stderr)

    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Evidence checks for a completed epoch's validation+test evaluation of a
Stage 1 NeuralHydrology run (produced by ``scripts/run_stage1_nh.py eval``).

Argument parsing, sha256-of-protected-files bookkeeping, and JSON/exit-code
output only. All check logic lives in
:mod:`src.baseline.nh_evaluation_check`, reusing
:class:`src.baseline.package_audit.AuditReport` for reporting. Never runs
training or evaluation itself -- both must already have completed (see
``scripts/run_stage1_nh_lead06_seq24_evaluate_moriah.sbatch``).

Exit status is 0 iff every check passes (PASS); 1 if any check fails (FAIL);
2 on a setup/usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline.nh_evaluation_check import run_evaluation_check  # noqa: E402


def _fail(message: str) -> int:
    print(f"FATAL: {message}", file=sys.stderr)
    return 2


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="Completed NeuralHydrology run directory")
    p.add_argument("--epoch", type=int, required=True, help="Epoch evaluated, e.g. 2 for model_epoch002.pt")
    p.add_argument("--expected-target-variable", required=True)
    p.add_argument("--expected-basin-count", type=int, required=True)
    p.add_argument("--expected-validation-year", type=int, required=True)
    p.add_argument("--expected-test-year", type=int, required=True)
    p.add_argument("--expected-metrics", nargs="+", required=True)
    p.add_argument(
        "--protected-relpath",
        action="append",
        default=[],
        dest="protected_relpaths",
        metavar="RUN_DIR_RELATIVE_PATH",
        help=(
            "run_dir-relative path of a training artifact that evaluation must not modify "
            "(e.g. config.yml, model_epoch002.pt, train_data/train_data_scaler.yml). "
            "Repeat for each file. Its sha256 is recomputed now and compared against "
            "--pre-eval-sha256-json's captured value for the same key."
        ),
    )
    p.add_argument(
        "--pre-eval-sha256-json",
        required=True,
        help=(
            "Path to a JSON file mapping each --protected-relpath to the sha256 hex digest "
            "captured for it BEFORE any evaluation invocation ran (produced by the calling "
            "launcher, not by this script, since this script only runs after the fact)."
        ),
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        return _fail(f"--run-dir does not exist or is not a directory: {run_dir}")

    pre_eval_sha256_path = Path(args.pre_eval_sha256_json)
    if not pre_eval_sha256_path.exists():
        return _fail(f"--pre-eval-sha256-json does not exist: {pre_eval_sha256_path}")
    with open(pre_eval_sha256_path) as fh:
        pre_eval_sha256 = json.load(fh)

    missing_keys = sorted(set(args.protected_relpaths) - set(pre_eval_sha256.keys()))
    if missing_keys:
        return _fail(f"--pre-eval-sha256-json is missing entries for: {missing_keys}")
    pre_eval_sha256 = {k: pre_eval_sha256[k] for k in args.protected_relpaths}

    report, period_stats = run_evaluation_check(
        run_dir=run_dir,
        epoch=args.epoch,
        expected_target_variable=args.expected_target_variable,
        expected_basin_count=args.expected_basin_count,
        expected_validation_year=args.expected_validation_year,
        expected_test_year=args.expected_test_year,
        expected_metric_names=args.expected_metrics,
        pre_eval_sha256=pre_eval_sha256,
    )

    summary = {
        "status": report.status,
        "run_dir": str(run_dir),
        "epoch": args.epoch,
        "error_count": report.error_count,
        "warning_count": report.warning_count,
        "ok_count": report.ok_count,
        "periods": period_stats,
        "checks": [{"severity": r.severity, "check_id": r.check_id, "message": r.message} for r in report.records],
    }
    print(json.dumps(summary, indent=2))
    print(f"Evaluation-check status: {report.status}")
    print(f"errors={report.error_count} warnings={report.warning_count} ok={report.ok_count}")
    if report.error_count:
        print("Failed checks:", file=sys.stderr)
        for msg in report.failed_messages():
            print(f"  - {msg}", file=sys.stderr)

    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

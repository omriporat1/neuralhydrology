#!/usr/bin/env python
"""Independently audit a Stage 1 candidate spatial split (Milestone 2K-G-I I-A3).

This CLI does NOT regenerate, correct, or promote a split. It re-derives every
fact it checks from the canonical policy and inputs (see
src/baseline/split_audit.py for the independence boundary) and compares them
against an existing candidate artifact bundle (and, optionally, a repeat
directory produced by re-running the I-A2 generator with the same inputs).

Usage:
    python scripts/audit_stage1_baseline_splits.py \\
        --policy config/stage1_scientific_baseline_v001.yaml \\
        --attributes-parquet <path to stage1_static_attributes_v001.parquet> \\
        --eligible-basins <path to eligible_basins_v001.txt> \\
        --candidate-dir tmp/stage1_baseline_splits_v001_candidate \\
        --repeat-dir tmp/stage1_baseline_splits_v001_candidate_repeat \\
        --out-dir tmp/stage1_baseline_splits_v001_audit

Exit code is 0 only if the audit found zero errors (warnings do not affect
the exit code). Generated outputs are written only under --out-dir and are
not committed to source control.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline.policy import Stage1BaselinePolicyError, load_stage1_baseline_policy
from src.baseline.split_audit import SplitAuditError, run_audit, write_audit_outputs


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default=str(REPO_ROOT / "config" / "stage1_scientific_baseline_v001.yaml"),
                    help="Path to the Stage 1 scientific baseline policy YAML")
    p.add_argument("--attributes-parquet", required=True,
                    help="Path to stage1_static_attributes_v001.parquet (or checksum-verified local copy)")
    p.add_argument("--eligible-basins", required=True,
                    help="Path to eligible_basins_v001.txt")
    p.add_argument("--forcing-basins", default=None,
                    help="Optional path to forcing_basins_v001.txt, to verify target/forcing agreement")
    p.add_argument("--candidate-dir", required=True,
                    help="Path to the candidate split artifact directory to audit")
    p.add_argument("--repeat-dir", default=None,
                    help="Optional path to a repeat-generation directory for byte-identity comparison")
    p.add_argument("--out-dir", required=True,
                    help="Directory to write audit_summary.json / audit_summary.md / audit_checks.csv (untracked)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        policy = load_stage1_baseline_policy(args.policy)
    except Stage1BaselinePolicyError as exc:
        print(f"FATAL: policy load/validate failed: {exc}", file=sys.stderr)
        return 1

    try:
        report, diagnostics = run_audit(
            policy=policy,
            policy_path=args.policy,
            candidate_dir=args.candidate_dir,
            repeat_dir=args.repeat_dir,
            attributes_parquet=args.attributes_parquet,
            eligible_basins=args.eligible_basins,
            forcing_basins=args.forcing_basins,
        )
    except SplitAuditError as exc:
        print(f"FATAL: audit could not run: {exc}", file=sys.stderr)
        return 1

    paths = write_audit_outputs(args.out_dir, report, diagnostics)

    print(f"status: {report.status}")
    print(f"errors: {report.error_count}  warnings: {report.warning_count}  ok: {report.ok_count}")
    print(f"outputs written under: {args.out_dir}")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    if report.error_count:
        print("\nfailed checks:", file=sys.stderr)
        for msg in report.failed_messages():
            print(f"  - {msg}", file=sys.stderr)

    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

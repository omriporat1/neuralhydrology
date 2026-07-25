#!/usr/bin/env python
"""Raw-space (m^3/s) evaluation for a completed Stage 1 seed run (Parts E/F).

Reads an already-completed NH evaluation pickle (``<period>_results.p``,
produced beforehand by ``scripts/run_stage1_nh.py eval`` -- or, for
"validation", produced automatically during training itself since the seed
profile sets ``validate_every: 1``) and computes raw-space metrics via
``src.baseline.nh_seed_evaluation.raw_space_metrics_for_run_period``. Does
not invoke NeuralHydrology itself and does not import torch, so it runs on a
CPU-only allocation.

Two subcommands:

  validation-sweep   Part E. Computes raw-space metrics for every epoch that
                      already has a validation/model_epochNNN/
                      validation_results.p under run_dir (produced by
                      training's own per-epoch validation, never re-run
                      here), ranks them by median per-basin raw-space NSE,
                      and writes a machine-readable checkpoint-selection
                      artifact. This performs the ranking computation only --
                      the actual freeze/selection decision and its written
                      record are Part E's own separate step, reviewed before
                      touching temporal-test or spatial-holdout data.

  single              Part F. Computes raw-space metrics for one already-
                      evaluated (period, epoch) pair against one run_dir --
                      used for both the development temporal-test (period
                      "test" against the development run_dir) and the
                      spatial-holdout test (period "test" against the
                      prepared external-scaler eval run_dir from
                      prepare_stage1_seed_holdout_eval_run_dir.py).

Never uses temporal-test or spatial-holdout performance for checkpoint
selection -- validation-sweep only ever reads the "validation" period.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline.nh_seed_evaluation import NHSeedEvaluationError, raw_space_metrics_for_run_period


def _fail(message: str) -> None:
    print(f"FATAL: {message}", file=sys.stderr)
    sys.exit(1)


def _discover_validation_epochs(run_dir: Path) -> list:
    validation_dir = run_dir / "validation"
    if not validation_dir.is_dir():
        return []
    epochs = []
    for child in sorted(validation_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("model_epoch"):
            continue
        if not (child / "validation_results.p").is_file():
            continue
        try:
            epochs.append(int(child.name[len("model_epoch"):]))
        except ValueError:
            continue
    return sorted(epochs)


def _common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--package-root", required=True, help="Certified stage1_scientific_package_v002 root")
    p.add_argument("--target-variable", default="qobs_mm_per_h_lead06")
    p.add_argument("--lead-hours", type=int, default=6)
    p.add_argument("--out-json", required=True)


def cmd_validation_sweep(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    epochs = _discover_validation_epochs(run_dir)
    if not epochs:
        _fail(f"no validation/model_epochNNN/validation_results.p found under {run_dir}")
        return 1

    per_epoch = []
    for epoch in epochs:
        try:
            result = raw_space_metrics_for_run_period(
                run_dir=run_dir,
                period="validation",
                epoch=epoch,
                package_root=args.package_root,
                target_variable=args.target_variable,
                lead_hours=args.lead_hours,
            )
        except NHSeedEvaluationError as exc:
            print(f"WARNING: epoch {epoch} skipped: {exc}", file=sys.stderr)
            continue
        median_nse = result["aggregate"]["metrics"].get("nse", {}).get("median")
        per_epoch.append({
            "epoch": epoch,
            "median_raw_nse": median_nse,
            "n_basins_evaluated": result["n_basins_evaluated"],
            "n_basins_area_excluded": result["n_basins_area_excluded"],
            "aggregate": result["aggregate"],
        })

    eligible = [e for e in per_epoch if e["median_raw_nse"] is not None]
    ranked = sorted(eligible, key=lambda e: e["median_raw_nse"], reverse=True)
    recommended_epoch = ranked[0]["epoch"] if ranked else None

    artifact = {
        "schema_name": "stage1_seed_checkpoint_selection_sweep",
        "schema_version": 1,
        "run_dir": str(run_dir),
        "selection_basis": "development_validation_only_raw_space_median_nse",
        "note": (
            "This artifact ranks candidate checkpoints by development-validation-only "
            "raw-space NSE. It does NOT itself constitute the frozen selection decision -- "
            "Part E's freeze step records the reviewed choice separately, before any "
            "temporal-test or spatial-holdout data is touched."
        ),
        "package_root": str(args.package_root),
        "target_variable": args.target_variable,
        "lead_hours": args.lead_hours,
        "epochs_evaluated": [e["epoch"] for e in per_epoch],
        "recommended_epoch": recommended_epoch,
        "per_epoch": per_epoch,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)
    print(json.dumps(artifact, indent=2))
    return 0


def cmd_single(args: argparse.Namespace) -> int:
    try:
        result = raw_space_metrics_for_run_period(
            run_dir=args.run_dir,
            period=args.period,
            epoch=args.epoch,
            package_root=args.package_root,
            target_variable=args.target_variable,
            lead_hours=args.lead_hours,
        )
    except NHSeedEvaluationError as exc:
        _fail(str(exc))
        return 1
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sweep_p = sub.add_parser("validation-sweep", help="Part E: rank candidate checkpoints by raw-space validation NSE")
    sweep_p.add_argument("--run-dir", required=True, help="Development training run directory")
    _common_args(sweep_p)

    single_p = sub.add_parser("single", help="Part F: raw-space metrics for one already-evaluated (period, epoch)")
    single_p.add_argument("--run-dir", required=True)
    single_p.add_argument("--period", required=True, choices=["train", "validation", "test"])
    single_p.add_argument("--epoch", type=int, required=True)
    _common_args(single_p)

    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "validation-sweep":
        return cmd_validation_sweep(args)
    return cmd_single(args)


if __name__ == "__main__":
    sys.exit(main())

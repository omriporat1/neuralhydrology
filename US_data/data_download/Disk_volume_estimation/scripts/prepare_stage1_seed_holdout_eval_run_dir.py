#!/usr/bin/env python
"""Prepare a minimal NH-Tester-compatible run directory for evaluating the
Stage 1 spatial-holdout population (Part F) with an already-trained
development seed checkpoint, reusing the development run's fitted scaler
byte-for-byte (never refit).

Thin CLI wrapper around
``src.baseline.nh_seed_evaluation.prepare_external_scaler_eval_run_dir``.
Pure file I/O -- does not import torch/NeuralHydrology, does not train or
evaluate anything itself. After this script succeeds, evaluate the holdout
test period against the resulting run directory with the ordinary:

    python scripts/run_stage1_nh.py eval <out-run-dir> --period test --epoch <epoch>

Usage:
    python scripts/prepare_stage1_seed_holdout_eval_run_dir.py \\
        --development-run-dir /path/to/development/runs/stage1_seed_..._HHMMSS \\
        --epoch 17 \\
        --holdout-generated-dir /path/to/stage1_full_population_nh_config_lead06_seq24_seed_v001/spatial_holdout \\
        --out-run-dir /path/to/stage1_holdout_eval_epoch017
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline.nh_seed_evaluation import NHSeedEvaluationError, prepare_external_scaler_eval_run_dir


def _fail(message: str) -> None:
    print(f"FATAL: {message}", file=sys.stderr)
    sys.exit(1)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--development-run-dir", required=True, help="Completed development training run directory")
    p.add_argument("--epoch", type=int, required=True, help="Frozen checkpoint epoch (see Part E selection artifact)")
    p.add_argument("--holdout-generated-dir", required=True,
                   help="The spatial_holdout/ generated config bundle directory (must contain "
                        "TEST_ONLY_DO_NOT_TRAIN.txt)")
    p.add_argument("--out-run-dir", required=True, help="Destination run directory to create")
    p.add_argument("--force", action="store_true", help="Overwrite out-run-dir if it already exists")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        manifest = prepare_external_scaler_eval_run_dir(
            development_run_dir=args.development_run_dir,
            epoch=args.epoch,
            holdout_generated_dir=args.holdout_generated_dir,
            out_run_dir=args.out_run_dir,
            force=args.force,
        )
    except NHSeedEvaluationError as exc:
        _fail(str(exc))
        return 1
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

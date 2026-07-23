"""Thin Flash-NH entrypoint for NeuralHydrology 1.13 train/eval runs.

Registers FlashNHDataset under the "flashnh" dataset key, then delegates
directly to neuralhydrology.nh_run.start_run / eval_run. Contains no
modeling or data-transformation logic of its own, and does not submit any
Slurm job -- it is meant to be invoked inside an already-allocated
interactive or batch Python process.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Direct execution (`python scripts/run_stage1_nh.py ...`) puts scripts/ --
# not the repository work directory -- at the front of sys.path, so the
# sibling top-level package `src` is otherwise unimportable regardless of the
# caller's current working directory. Insert the repo work directory
# (this file's parent's parent) before importing src.baseline.nh_register.
_REPO_WORKDIR = Path(__file__).resolve().parent.parent
if str(_REPO_WORKDIR) not in sys.path:
    sys.path.insert(0, str(_REPO_WORKDIR))

from src.baseline.nh_register import register_flashnh_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run neuralhydrology.nh_run.start_run")
    train_parser.add_argument("config_file", type=Path)

    eval_parser = subparsers.add_parser("eval", help="Run neuralhydrology.nh_run.eval_run")
    eval_parser.add_argument("run_dir", type=Path)
    eval_parser.add_argument("--period", default="test", choices=["train", "validation", "test"])
    eval_parser.add_argument("--epoch", type=int, default=None)

    args = parser.parse_args()

    register_flashnh_dataset()

    from neuralhydrology.nh_run import eval_run, start_run

    if args.command == "train":
        start_run(config_file=args.config_file)
    elif args.command == "eval":
        eval_run(run_dir=args.run_dir, period=args.period, epoch=args.epoch)


if __name__ == "__main__":
    main()

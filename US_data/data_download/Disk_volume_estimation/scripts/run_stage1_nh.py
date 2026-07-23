"""Thin Flash-NH entrypoint for NeuralHydrology 1.13 train/eval runs.

Registers FlashNHDataset under the "flashnh" dataset key, then delegates to
neuralhydrology.nh_run.start_run (train) or
neuralhydrology.evaluation.evaluate.start_evaluation (eval, built inline so an
optional --metrics list can be applied to the loaded Config in memory without
writing back to the run's config.yml). Contains no modeling or
data-transformation logic of its own, and does not submit any Slurm job -- it
is meant to be invoked inside an already-allocated interactive or batch
Python process.
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

    eval_parser = subparsers.add_parser(
        "eval", help="Run neuralhydrology.evaluation.evaluate.start_evaluation on a completed run"
    )
    eval_parser.add_argument("run_dir", type=Path)
    eval_parser.add_argument("--period", default="test", choices=["train", "validation", "test"])
    eval_parser.add_argument("--epoch", type=int, default=None)
    eval_parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        metavar="METRIC",
        help=(
            "Metric names to compute for this evaluation invocation only (see "
            "neuralhydrology.evaluation.metrics.get_available_metrics for the "
            "supported names, e.g. NSE RMSE KGE Pearson-r Beta-KGE). Applied "
            "in memory to the run's loaded Config object and never written "
            "back to run_dir/config.yml, so the frozen training config on "
            "disk is untouched. If omitted, whatever 'metrics' the run's own "
            "config.yml already specifies is used (empty by default -- "
            "identical to this command's behavior before --metrics existed)."
        ),
    )

    args = parser.parse_args()

    register_flashnh_dataset()

    if args.command == "train":
        from neuralhydrology.nh_run import start_run

        start_run(config_file=args.config_file)
    elif args.command == "eval":
        # Built directly (instead of neuralhydrology.nh_run.eval_run) only so
        # --metrics can be applied to the in-memory Config before evaluation;
        # otherwise identical to what eval_run does internally. The saved
        # training scaler (train_data/train_data_scaler.yml) is always
        # reloaded from run_dir by NeuralHydrology's Tester itself and is
        # never refit here, regardless of --metrics.
        from neuralhydrology.evaluation.evaluate import start_evaluation
        from neuralhydrology.utils.config import Config
        from neuralhydrology.utils.logging_utils import setup_logging

        config = Config(args.run_dir / "config.yml")
        if args.metrics is not None:
            config.metrics = args.metrics
        # Matches neuralhydrology.nh_run._main()'s own behavior for mode
        # "evaluate" (appends to the run's existing output.log via a
        # logging.FileHandler opened in its default "a" mode, so no training
        # log content is lost). This makes BaseTester's own
        # "Using the model weights from <path>" line observable afterwards,
        # which is the objective evidence that the requested --epoch was the
        # checkpoint actually loaded.
        setup_logging(str(args.run_dir / "output.log"))
        start_evaluation(cfg=config, run_dir=args.run_dir, epoch=args.epoch, period=args.period)


if __name__ == "__main__":
    main()

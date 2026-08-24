#!/bin/bash
# One Slurm allocation runs exactly one random-control row.  This launcher is
# never invoked by local tests and performs no loop over scientific
# candidates -- mirrors run_sweep_v1_wandb_agent_moriah.sbatch's
# one-allocation-per-trial rule for the Bayesian arm. No wandb agent/sweep is
# involved: random-control rows are a fixed committed manifest, not a live
# W&B search.
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=flashnh-sweep-v1-random-control
set -euo pipefail
: "${ROW_INDEX:?set the 0-based random-control manifest row index}"
: "${PACKAGE_ROOT:?set the production package root}"
: "${SCREENING_BASIN_IDS:?set the screening basin ids path}"
: "${OUTPUT_ROOT:?set the output root for this trial}"
python scripts/run_sweep_v1_random_control_trial.py \
    --row-index "$ROW_INDEX" --package-root "$PACKAGE_ROOT" \
    --screening-basin-ids "$SCREENING_BASIN_IDS" --output-root "$OUTPUT_ROOT"

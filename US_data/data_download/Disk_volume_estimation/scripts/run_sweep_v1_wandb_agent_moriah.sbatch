#!/bin/bash
# One Slurm allocation runs exactly one W&B proposal.  This launcher is never
# invoked by local tests and performs no loop over scientific candidates.
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=flashnh-sweep-v1-agent
set -euo pipefail
: "${WANDB_SWEEP_ID:?set the already-created production sweep id}"
# The bridge is the sole sweep program; --count 1 is the bounded proposal rule.
wandb agent --count 1 "$WANDB_SWEEP_ID"

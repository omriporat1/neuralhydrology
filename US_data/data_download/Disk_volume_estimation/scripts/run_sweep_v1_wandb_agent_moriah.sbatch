#!/bin/bash
# One Slurm allocation runs exactly one W&B proposal.  This launcher is never
# invoked by local tests and performs no loop over scientific candidates.
#
# The production sweep config's command is explicit
# (["${interpreter}", "${program}"], no ${args}) so wandb agent -- not this
# script -- constructs run_sweep_v1_wandb_bridge.py's argv directly.  The
# four FLASHNH_SWEEP_V1_* variables below are this launcher's only channel
# for handing the bridge its operational inputs: they are exported into this
# shell's environment, which wandb agent and every child process it spawns
# inherit by standard OS subprocess semantics.
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=flashnh-sweep-v1-agent
set -euo pipefail
: "${WANDB_SWEEP_ID:?set the already-created production sweep id}"
: "${FLASHNH_SWEEP_V1_PACKAGE_ROOT:?set the production package root}"
: "${FLASHNH_SWEEP_V1_SCREENING_BASIN_IDS:?set the screening basin ids path}"
: "${FLASHNH_SWEEP_V1_OUTPUT_ROOT:?set the output root for this trial}"
: "${FLASHNH_SWEEP_V1_PROPOSAL_ORDER:?set the explicit, caller-tracked Bayesian proposal order (1 for the first real attempt); never inferred}"
case "$FLASHNH_SWEEP_V1_PROPOSAL_ORDER" in
    ''|*[!0-9]*) echo "FLASHNH_SWEEP_V1_PROPOSAL_ORDER must be a positive integer, got '$FLASHNH_SWEEP_V1_PROPOSAL_ORDER'" >&2; exit 1 ;;
    0) echo "FLASHNH_SWEEP_V1_PROPOSAL_ORDER must be a positive integer, got 0" >&2; exit 1 ;;
esac
export FLASHNH_SWEEP_V1_PACKAGE_ROOT FLASHNH_SWEEP_V1_SCREENING_BASIN_IDS FLASHNH_SWEEP_V1_OUTPUT_ROOT FLASHNH_SWEEP_V1_PROPOSAL_ORDER
# The bridge is the sole sweep program; --count 1 is the bounded proposal rule.
wandb agent --count 1 "$WANDB_SWEEP_ID"

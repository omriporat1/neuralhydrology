#!/bin/bash
#SBATCH --job-name=flashnh-sweep-v2-wandb-agent
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/sweep-v2-wandb-agent-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/sweep-v2-wandb-agent-%j.err
set -euo pipefail

: "${WANDB_SWEEP_ID:?set a future v2 production sweep id}"
: "${WANDB_PROJECT:?set the v2 production project}"
: "${WANDB_ENTITY:?set the v2 production entity}"
FORBIDDEN_PRODUCTION_SWEEP_ID="4x3btz2s"
if [ "${WANDB_SWEEP_ID}" = "${FORBIDDEN_PRODUCTION_SWEEP_ID}" ]; then
    echo "REFUSING: v1 production sweep" >&2
    exit 1
fi
export WANDB_PROJECT WANDB_ENTITY

FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
REPO_WORKDIR="${REPO_WORKDIR:-${FLASHNH_BASE}/repos/flash-nh/US_data/data_download/Disk_volume_estimation}"
CANONICAL_PYTHON="${CANONICAL_PYTHON:-${FLASHNH_BASE}/envs/flashnh-moriah/bin/python}"
test -d "${REPO_WORKDIR}" || { echo "MISSING repo work directory" >&2; exit 1; }
cd "${REPO_WORKDIR}"
test -x "${CANONICAL_PYTHON}" || { echo "MISSING canonical interpreter" >&2; exit 1; }
export PATH="$(dirname "${CANONICAL_PYTHON}"):${PATH}"

# The production sweep's static command supplies the single checked manifest
# path to the bridge; this launcher consumes exactly one controller proposal.
wandb agent --count 1 "${WANDB_SWEEP_ID}"

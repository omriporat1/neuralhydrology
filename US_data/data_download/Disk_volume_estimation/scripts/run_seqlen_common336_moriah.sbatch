#!/usr/bin/env bash
#SBATCH --job-name=flashnh-seqlen-c336
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --array=0-14%5
#SBATCH --chdir=/sci/labs/efratmorin/omripo/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation/.scratch_local/seqlen_common336_seed3_v001/logs/%x-%A_%a.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation/.scratch_local/seqlen_common336_seed3_v001/logs/%x-%A_%a.err
#
# seqlen_common336_seed3_v001: 15 fresh trials (L in {72,120,168,240,336} x
# seeds A/B/C), one array task per trial, task index = campaign_manifest.json
# row index (seed-major: 0-4 seed A, 5-9 seed B, 10-14 seed C). The %5 limit
# caps simultaneously running tasks at five; it does NOT guarantee that one
# seed's five lengths run together (Slurm starts the next pending index as
# soon as any task ends). Never contacts W&B or a sweep controller; every
# config declares flashnh_observation_cutoff, so no observation after the
# validation period is read and the test period is refused.
#
# PREPARED FOR REVIEW ONLY -- not submitted by the preparation milestone.
# Submission requires the user's approval of the batch and resource estimate,
# and a reviewed commit (EXPECTED_COMMIT) whose clean tracked tree is pinned by
# src.baseline.sweep_v1_runtime_contract inside the runner.
#
# Required environment:
#   EXPECTED_COMMIT  full 40-hex commit this launch is pinned to
# Fixed (not overridable): checkout, interpreter, campaign manifest.
set -euo pipefail

: "${EXPECTED_COMMIT:?set the full 40-hex git commit this launch is pinned to}"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo "EXPECTED_COMMIT is not a 40-hex commit" >&2; exit 1; }
: "${SLURM_ARRAY_TASK_ID:?this launcher must run as a Slurm array task}"

FLASHNH_BASE=/sci/labs/efratmorin/omripo/Flash-NH
REPO_WORKDIR="${FLASHNH_BASE}/repos/flash-nh/US_data/data_download/Disk_volume_estimation"
CANONICAL_PYTHON="${FLASHNH_BASE}/envs/flashnh-moriah/bin/python"
CAMPAIGN_ROOT="${REPO_WORKDIR}/.scratch_local/seqlen_common336_seed3_v001"
# v002 manifest (schema 2: derived seed B, observation cutoff). The v001
# training_output/ (seed B 1729, no cutoff) is superseded and never launched.
CAMPAIGN_MANIFEST="${CAMPAIGN_ROOT}/training_output_v002/campaign_manifest.json"

# No W&B; all temp/cache plumbing stays under the project root.
export WANDB_MODE=disabled WANDB_DISABLED=true
export TMPDIR="${CAMPAIGN_ROOT}/tmp/${SLURM_ARRAY_JOB_ID:-x}_${SLURM_ARRAY_TASK_ID}"
export XDG_CACHE_HOME="${CAMPAIGN_ROOT}/cache" MPLCONFIGDIR="${CAMPAIGN_ROOT}/cache/matplotlib"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

echo "=== seqlen_common336 trial ${SLURM_ARRAY_TASK_ID} ==="; hostname; date -u
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-} ARRAY=${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID} EXPECTED_COMMIT=${EXPECTED_COMMIT}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
cd "${REPO_WORKDIR}"
test -x "${CANONICAL_PYTHON}" && test -f "${CAMPAIGN_MANIFEST}"

exec "${CANONICAL_PYTHON}" scripts/run_seqlen_common336_trial.py \
    --campaign-manifest "${CAMPAIGN_MANIFEST}" \
    --index "${SLURM_ARRAY_TASK_ID}" \
    --expected-commit "${EXPECTED_COMMIT}" \
    --expected-runtime-python "${CANONICAL_PYTHON}"

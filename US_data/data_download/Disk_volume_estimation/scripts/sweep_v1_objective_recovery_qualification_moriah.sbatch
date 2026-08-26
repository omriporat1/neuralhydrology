#!/usr/bin/env bash
#SBATCH --job-name=flashnh-objrecov-qual
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/sweep-v1-objective-recovery-qual-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/sweep-v1-objective-recovery-qual-%j.err
#
# ATTEMPT005 closure task, Section F: operationally qualify
# src.baseline.sweep_v1_objective_recovery.recover_and_publish_objective
# against ONE disposable, explicitly non-scientific real W&B sweep/run.
#
# CPU-only, no GRES, no NeuralHydrology/torch import, no scientific data.
# NEVER calls wandb.agent(). NEVER references production sweep 4x3btz2s or
# run ardib08c -- scripts/sweep_v1_objective_recovery_qualification.py
# refuses outright if its own newly-created disposable identity ever
# collided with either of those (astronomically unlikely, checked anyway).
#
# Usage:
#   EXPECTED_COMMIT=<commit> \
#       sbatch --export=ALL,EXPECTED_COMMIT \
#       scripts/sweep_v1_objective_recovery_qualification_moriah.sbatch
set -uo pipefail

EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"
if [ -z "${EXPECTED_COMMIT}" ]; then
    echo "EXPECTED_COMMIT is required (the exact commit hash you reviewed and are authorizing)." >&2
    echo "Set it explicitly, e.g.: EXPECTED_COMMIT=<hash> sbatch --export=ALL,EXPECTED_COMMIT $0" >&2
    exit 2
fi

FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
REPO_WORKDIR="${REPO_WORKDIR:-${FLASHNH_BASE}/repos/flash-nh/US_data/data_download/Disk_volume_estimation}"
CANONICAL_PYTHON="${CANONICAL_PYTHON:-${FLASHNH_BASE}/envs/flashnh-moriah/bin/python}"

RESULT_DIR="${FLASHNH_BASE}/evidence/sweep_v1_objective_recovery_qualification_v001/run_${SLURM_JOB_ID}"
mkdir -p "${RESULT_DIR}"
RESULT_LOG="${RESULT_DIR}/run_human.log"
exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== Sweep-v1 objective-recovery qualification (Section F) ==="
hostname
date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-unknown}"
echo "EXPECTED_COMMIT (authorized by submitter): ${EXPECTED_COMMIT}"
echo "REPO_WORKDIR: ${REPO_WORKDIR}"
echo "CANONICAL_PYTHON: ${CANONICAL_PYTHON}"

test -d "${REPO_WORKDIR}" || { echo "MISSING expected repo work directory: ${REPO_WORKDIR}"; exit 1; }
cd "${REPO_WORKDIR}"
git log --oneline -3
SOURCE_COMMIT=$(git rev-parse HEAD)
GIT_STATUS_PORCELAIN=$(git status --porcelain)
echo "SOURCE_COMMIT: ${SOURCE_COMMIT}"
TRACKED_CHANGES=$(echo "${GIT_STATUS_PORCELAIN}" | grep -v '^??' || true)
if [ -n "${TRACKED_CHANGES}" ]; then
    echo "REFUSING to run: tracked files are locally modified in ${REPO_WORKDIR}:" >&2
    echo "${TRACKED_CHANGES}" >&2
    exit 1
fi
if [ "${SOURCE_COMMIT}" != "${EXPECTED_COMMIT}" ]; then
    echo "REFUSING to run: repo HEAD (${SOURCE_COMMIT}) does not match the explicitly" >&2
    echo "authorized EXPECTED_COMMIT (${EXPECTED_COMMIT})." >&2
    exit 1
fi
echo "Commit pin OK: HEAD matches EXPECTED_COMMIT (${EXPECTED_COMMIT})."

test -x "${CANONICAL_PYTHON}" || { echo "MISSING canonical runtime interpreter: ${CANONICAL_PYTHON}"; exit 1; }

echo "=== launching scripts/sweep_v1_objective_recovery_qualification.py ==="
"${CANONICAL_PYTHON}" scripts/sweep_v1_objective_recovery_qualification.py \
    --expected-commit "${EXPECTED_COMMIT}" \
    --expected-runtime-python "${CANONICAL_PYTHON}" \
    --out-dir "${RESULT_DIR}"
QUAL_STATUS=$?

echo "=== done (exit ${QUAL_STATUS}) ==="
exit "${QUAL_STATUS}"

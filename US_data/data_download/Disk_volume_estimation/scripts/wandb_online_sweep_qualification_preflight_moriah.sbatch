#!/usr/bin/env bash
#SBATCH --job-name=flashnh-wandb-online-sweep-qual-preflight
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/wandb-online-sweep-qual-preflight-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/wandb-online-sweep-qual-preflight-%j.err
#
# Phase-B online W&B sweep qualification -- STEP A (CPU PREFLIGHT). Runs
# scripts/wandb_online_sweep_qualification_preflight.py from the exact
# canonical Moriah runtime (never an ephemeral venv -- unlike the sibling
# offline launch-contract qualification, wandb 0.28.1 is already installed
# in the shared flashnh-moriah conda env, so this job must use it directly).
# CPU-only, no GRES, no NH/torch import, no scientific data.
#
# This job intentionally does NOT export WANDB_MODE=offline -- the whole
# point of this qualification wave is the ONLINE path. It never prints,
# logs, or stores the WANDB_API_KEY value itself; only a boolean
# credential-present result is recorded (see the preflight script's
# docstring).
#
# Usage:
#   EXPECTED_COMMIT=<commit> sbatch --export=ALL,EXPECTED_COMMIT \
#       scripts/wandb_online_sweep_qualification_preflight_moriah.sbatch
set -uo pipefail

EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"
if [ -z "${EXPECTED_COMMIT}" ]; then
    echo "EXPECTED_COMMIT is required (the exact commit hash you reviewed and are authorizing)." >&2
    echo "Set it explicitly, e.g.: EXPECTED_COMMIT=<hash> sbatch --export=ALL,EXPECTED_COMMIT $0" >&2
    exit 2
fi

FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
REPO_WORKDIR="${REPO_WORKDIR:-${FLASHNH_BASE}/repos/flash-nh}"
CANONICAL_PYTHON="${CANONICAL_PYTHON:-${FLASHNH_BASE}/envs/flashnh-moriah/bin/python}"

RESULT_DIR="${FLASHNH_BASE}/evidence/phase_b_wandb_online_sweep_qualification_v001/preflight_${SLURM_JOB_ID}"
mkdir -p "${RESULT_DIR}"
RESULT_LOG="${RESULT_DIR}/preflight_human.log"
exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== Phase-B online W&B sweep qualification: CPU preflight ==="
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

echo "=== launching scripts/wandb_online_sweep_qualification_preflight.py ==="
"${CANONICAL_PYTHON}" scripts/wandb_online_sweep_qualification_preflight.py \
    --expected-commit "${EXPECTED_COMMIT}" \
    --expected-runtime-python "${CANONICAL_PYTHON}" \
    --out-dir "${RESULT_DIR}"
QUAL_STATUS=$?

echo "=== done (exit ${QUAL_STATUS}) ==="
exit "${QUAL_STATUS}"

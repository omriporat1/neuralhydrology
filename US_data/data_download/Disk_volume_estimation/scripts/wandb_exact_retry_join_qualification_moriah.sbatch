#!/usr/bin/env bash
#SBATCH --job-name=flashnh-wandb-exact-retry-join-qual
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/wandb-exact-retry-join-qual-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/wandb-exact-retry-join-qual-%j.err
#
# Exact-retry W&B join qualification -- ONE bounded Slurm allocation runs the
# entire disposable-sweep scenario in scripts/wandb_exact_retry_join_qualification.py
# (one real controller proposal, two manual sweep joins via
# wandb.Settings(sweep_id=...) with no wandb.agent() call around them, a
# second real controller proposal, and wandb.Api() inspection of all four
# runs). This is a single self-contained script, not a multi-job sequence
# like the wandb_online_sweep_qualification_* family, so one allocation is
# sufficient.
#
# CPU-only, no GRES, no NH/torch import, no scientific data, no production
# sweep/run identity anywhere in this file or the script it launches. Does
# NOT export WANDB_MODE=offline -- online is the entire point; the
# underlying script itself verifies every run is really online and fails
# loudly (no silent offline fallback) otherwise.
#
# Usage:
#   EXPECTED_COMMIT=<commit> \
#       sbatch --export=ALL,EXPECTED_COMMIT \
#       scripts/wandb_exact_retry_join_qualification_moriah.sbatch
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

RESULT_DIR="${FLASHNH_BASE}/evidence/wandb_exact_retry_join_qualification_v001/run_${SLURM_JOB_ID}"
mkdir -p "${RESULT_DIR}"
RESULT_LOG="${RESULT_DIR}/run_human.log"
exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== Exact-retry W&B join qualification (non-scientific, disposable sweep) ==="
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

echo "=== launching scripts/wandb_exact_retry_join_qualification.py ==="
"${CANONICAL_PYTHON}" scripts/wandb_exact_retry_join_qualification.py \
    --expected-commit "${EXPECTED_COMMIT}" \
    --expected-runtime-python "${CANONICAL_PYTHON}" \
    --out-dir "${RESULT_DIR}"
QUAL_STATUS=$?

echo "=== done (exit ${QUAL_STATUS}) ==="
exit "${QUAL_STATUS}"

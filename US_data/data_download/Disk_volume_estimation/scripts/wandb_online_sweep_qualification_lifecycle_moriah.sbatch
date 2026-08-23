#!/usr/bin/env bash
#SBATCH --job-name=flashnh-wandb-online-sweep-qual-lifecycle
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/wandb-online-sweep-qual-lifecycle-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/wandb-online-sweep-qual-lifecycle-%j.err
#
# Phase-B online W&B sweep qualification -- STEPS C and E (pause / resume /
# stop / status). Reused across all four lifecycle calls via ACTION.
# Qualifies the underlying W&B lifecycle primitive that a future production
# boundary-review controller will need (docs/
# stage1_phase_b_sweep_v1_launch_contract.md, "Boundary reviews and
# immutable waves") -- this job does not implement that controller.
#
# CPU-only, no GRES. A single read/status call still counts as an online API
# request, so this stays off the login node like every other step here.
#
# Usage:
#   SWEEP_ID=<entity/project/sweep_id> ACTION=pause \
#       sbatch --export=ALL,SWEEP_ID,ACTION \
#       scripts/wandb_online_sweep_qualification_lifecycle_moriah.sbatch
set -uo pipefail

SWEEP_ID="${SWEEP_ID:-}"
if [ -z "${SWEEP_ID}" ]; then
    echo "SWEEP_ID is required (entity/project/sweep_id)." >&2
    exit 2
fi
ACTION="${ACTION:-}"
case "${ACTION}" in
    status|pause|resume|stop) ;;
    *)
        echo "ACTION must be one of: status pause resume stop (got '${ACTION}')." >&2
        exit 2
        ;;
esac

FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
REPO_WORKDIR="${REPO_WORKDIR:-${FLASHNH_BASE}/repos/flash-nh}"
CANONICAL_PYTHON="${CANONICAL_PYTHON:-${FLASHNH_BASE}/envs/flashnh-moriah/bin/python}"

RESULT_DIR="${FLASHNH_BASE}/evidence/phase_b_wandb_online_sweep_qualification_v001/lifecycle_${ACTION}_${SLURM_JOB_ID}"
mkdir -p "${RESULT_DIR}"
RESULT_LOG="${RESULT_DIR}/lifecycle_human.log"
exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== Phase-B online W&B sweep qualification: lifecycle (${ACTION}) ==="
hostname
date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-unknown}"
echo "SWEEP_ID: ${SWEEP_ID}"
echo "ACTION: ${ACTION}"
echo "REPO_WORKDIR: ${REPO_WORKDIR}"
echo "CANONICAL_PYTHON: ${CANONICAL_PYTHON}"

test -d "${REPO_WORKDIR}" || { echo "MISSING expected repo work directory: ${REPO_WORKDIR}"; exit 1; }
cd "${REPO_WORKDIR}"
test -x "${CANONICAL_PYTHON}" || { echo "MISSING canonical runtime interpreter: ${CANONICAL_PYTHON}"; exit 1; }

echo "=== launching scripts/wandb_online_sweep_qualification_lifecycle.py ==="
"${CANONICAL_PYTHON}" scripts/wandb_online_sweep_qualification_lifecycle.py \
    --sweep-id "${SWEEP_ID}" \
    --action "${ACTION}" \
    --out-dir "${RESULT_DIR}"
QUAL_STATUS=$?

echo "=== done (exit ${QUAL_STATUS}) ==="
exit "${QUAL_STATUS}"

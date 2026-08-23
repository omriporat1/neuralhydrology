#!/usr/bin/env bash
#SBATCH --job-name=flashnh-wandb-online-sweep-qual-run
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/wandb-online-sweep-qual-run-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/wandb-online-sweep-qual-run-%j.err
#
# Phase-B online W&B sweep qualification -- STEPS B and D (ONE agent
# proposal per allocation). Reused for BOTH the first job (creates the toy
# sweep, SWEEP_ID unset) and the second job (SWEEP_ID set to the sweep the
# first job created) -- the same launcher, invoked twice, is the smallest
# reusable way to qualify "one bounded Slurm allocation -> one W&B agent ->
# exactly one proposal" without a second near-duplicate file.
#
# Runs scripts/wandb_online_sweep_qualification_run.py from the exact
# canonical Moriah runtime. CPU-only, no GRES, no NH/torch import, no
# scientific data. Does NOT export WANDB_MODE=offline -- online is the
# entire point of this qualification wave; the underlying script itself
# verifies the resulting run is really online and fails loudly otherwise.
#
# Usage (first job -- creates the sweep):
#   EXPECTED_COMMIT=<commit> PROPOSAL_LABEL=first \
#       sbatch --export=ALL,EXPECTED_COMMIT,PROPOSAL_LABEL \
#       scripts/wandb_online_sweep_qualification_run_moriah.sbatch
#
# Usage (second job -- reuses the sweep from the first job's evidence):
#   EXPECTED_COMMIT=<commit> PROPOSAL_LABEL=second SWEEP_ID=<entity/project/sweep_id> \
#       sbatch --export=ALL,EXPECTED_COMMIT,PROPOSAL_LABEL,SWEEP_ID \
#       scripts/wandb_online_sweep_qualification_run_moriah.sbatch
set -uo pipefail

EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"
if [ -z "${EXPECTED_COMMIT}" ]; then
    echo "EXPECTED_COMMIT is required (the exact commit hash you reviewed and are authorizing)." >&2
    echo "Set it explicitly, e.g.: EXPECTED_COMMIT=<hash> sbatch --export=ALL,EXPECTED_COMMIT $0" >&2
    exit 2
fi
PROPOSAL_LABEL="${PROPOSAL_LABEL:-}"
if [ -z "${PROPOSAL_LABEL}" ]; then
    echo "PROPOSAL_LABEL is required (e.g. 'first' or 'second')." >&2
    exit 2
fi
SWEEP_ID="${SWEEP_ID:-}"

FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
REPO_WORKDIR="${REPO_WORKDIR:-${FLASHNH_BASE}/repos/flash-nh}"
CANONICAL_PYTHON="${CANONICAL_PYTHON:-${FLASHNH_BASE}/envs/flashnh-moriah/bin/python}"

RESULT_DIR="${FLASHNH_BASE}/evidence/phase_b_wandb_online_sweep_qualification_v001/run_${PROPOSAL_LABEL}_${SLURM_JOB_ID}"
mkdir -p "${RESULT_DIR}"
RESULT_LOG="${RESULT_DIR}/run_human.log"
exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== Phase-B online W&B sweep qualification: agent run (${PROPOSAL_LABEL}) ==="
hostname
date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-unknown}"
echo "EXPECTED_COMMIT (authorized by submitter): ${EXPECTED_COMMIT}"
echo "PROPOSAL_LABEL: ${PROPOSAL_LABEL}"
echo "SWEEP_ID (empty means: create a new toy sweep): ${SWEEP_ID}"
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

SWEEP_ID_ARGS=()
if [ -n "${SWEEP_ID}" ]; then
    SWEEP_ID_ARGS=(--sweep-id "${SWEEP_ID}")
fi

echo "=== launching scripts/wandb_online_sweep_qualification_run.py ==="
"${CANONICAL_PYTHON}" scripts/wandb_online_sweep_qualification_run.py \
    --expected-commit "${EXPECTED_COMMIT}" \
    --expected-runtime-python "${CANONICAL_PYTHON}" \
    --proposal-label "${PROPOSAL_LABEL}" \
    --out-dir "${RESULT_DIR}" \
    "${SWEEP_ID_ARGS[@]}"
QUAL_STATUS=$?

echo "=== done (exit ${QUAL_STATUS}) ==="
exit "${QUAL_STATUS}"

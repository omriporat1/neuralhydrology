#!/usr/bin/env bash
#SBATCH --job-name=flashnh-rd1-c4-results
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/rd1-c4-results-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/rd1-c4-results-%j.err
#
# RD1-C4-E production run: the formal 24-trial hydrological review
# (assemble_rd1_hydrological_review via src.baseline.rd1_c4_results_runner),
# producing the frozen full-support per-basin NSE/KGE table, the Q98
# diagnostic table, and the RD1-C3 basin-distribution summary that RD1-C4-D1
# explicitly does not produce.
#
# CPU-only, single process, all 24 trials sequentially in one job (no array):
# assemble_rd1_hydrological_review is a batch entry point over the roster and
# is not meant to be split per-trial. Resources are sized from RD1-C4-D1's own
# single-trial benchmark (job 46176003: ~10 min wall, ~780 MB MaxRSS for one
# trial's diagnostic-only pass over 400 basins); this consumer does strictly
# more per-trial work (raw-space NSE/KGE fixed-support evaluation plus basin
# distribution plus Q98 diagnostics), so time and memory are given generous
# headroom rather than a tight extrapolation.
#
# Performs no scientific classification, selects no threshold/winner, and
# closes nothing: it only authenticates the same frozen roster D1 used,
# calls the already-qualified batch consumer, and serializes its output.
#
# No GPU, no training, no inference, no W&B, no controller, no sealed scope.

set -uo pipefail

FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
ENV_PREFIX="${ENV_PREFIX:-${FLASHNH_BASE}/envs/flashnh-moriah}"
REPO_CLONE_DIR="${REPO_CLONE_DIR:-${FLASHNH_BASE}/repos/flash-nh-rd1c4-d1}"
REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"

RD1_C4_TRIAL_LIST="${RD1_C4_TRIAL_LIST:?export RD1_C4_TRIAL_LIST=<frozen 24-trial roster JSON>}"
RD1_C4_CONTRACT="${RD1_C4_CONTRACT:?export RD1_C4_CONTRACT=<frozen fixed-support contract JSON>}"
RD1_C4_PACKAGE_ROOT="${RD1_C4_PACKAGE_ROOT:?export RD1_C4_PACKAGE_ROOT=<qualified package root>}"
RD1_C4_OUT_DIR="${RD1_C4_OUT_DIR:?export RD1_C4_OUT_DIR=<fresh, not-yet-existing output dir>}"

for _path in "${RD1_C4_TRIAL_LIST}" "${RD1_C4_CONTRACT}"; do
    if [ ! -f "${_path}" ]; then
        echo "REFUSING: required input file does not exist: ${_path}" >&2
        exit 2
    fi
done
if [ ! -d "${RD1_C4_PACKAGE_ROOT}" ]; then
    echo "REFUSING: package root is not a directory: ${RD1_C4_PACKAGE_ROOT}" >&2
    exit 2
fi
for _path in "${RD1_C4_PACKAGE_ROOT}" "${RD1_C4_TRIAL_LIST}" "${RD1_C4_CONTRACT}" "${RD1_C4_OUT_DIR}"; do
    case "${_path}" in
        *temporal_test*|*spatial_holdout*|*california*|*sealed*)
            echo "REFUSING: input path names a sealed/protected scope: ${_path}" >&2
            exit 2
            ;;
    esac
done
if [ -e "${RD1_C4_OUT_DIR}" ] || [ -L "${RD1_C4_OUT_DIR}" ]; then
    echo "REFUSING: RD1_C4_OUT_DIR already exists: ${RD1_C4_OUT_DIR}" >&2
    exit 2
fi

mkdir -p "$(dirname "${RD1_C4_OUT_DIR}")"
RESULT_LOG="${FLASHNH_BASE}/tmp/rd1_c4_results/rd1_c4_results_${SLURM_JOB_ID:-local}.log"
mkdir -p "$(dirname "${RESULT_LOG}")"
exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== RD1-C4-E results production run ==="
hostname
date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"

if ! type module &>/dev/null; then
    for _modinit in \
        /etc/profile.d/huji-lmod.sh \
        /etc/profile.d/modules.sh \
        /usr/share/lmod/lmod/init/bash \
        /etc/profile.d/lmod.sh
    do
        [ -f "${_modinit}" ] && { source "${_modinit}"; break; }
    done
    unset _modinit
fi

module load spack/all
module load miniconda3/24.3.0-gcc-iqeknet

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
which python

cd "${REPO_WORKDIR}" || exit 2
echo "--- isolated worktree git state ---"
git -C "${REPO_CLONE_DIR}" status --short
git -C "${REPO_CLONE_DIR}" rev-parse HEAD
echo "--- canonical repo unchanged check ---"
git -C "${FLASHNH_BASE}/repos/flash-nh" status --short

export PYTHONUNBUFFERED=1

echo "--- input identities ---"
sha256sum "${RD1_C4_TRIAL_LIST}"
sha256sum "${RD1_C4_CONTRACT}"

/usr/bin/time -v python -m src.baseline.rd1_c4_results_runner \
    --trial-list "${RD1_C4_TRIAL_LIST}" \
    --contract "${RD1_C4_CONTRACT}" \
    --package-root "${RD1_C4_PACKAGE_ROOT}" \
    --out-dir "${RD1_C4_OUT_DIR}"
STATUS=$?

echo "=== RD1-C4-E results production done (exit ${STATUS}) ==="
echo "Evidence bundle (if successful): ${RD1_C4_OUT_DIR}"
exit ${STATUS}

#!/usr/bin/env bash
#SBATCH --job-name=flashnh-rd1-c4-f-hydrograph
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/rd1-c4-f-hydrograph-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/rd1-c4-f-hydrograph-%j.err
#
# RD1-C4-F descriptive hydrograph supplement: renders same-panel incumbent
# overlay figures for the 3 already-selected basins
# (src.baseline.rd1_c4_f_hydrograph_runner), reusing the same frozen 24-trial
# roster and fixed-support contract RD1-C4-E authenticated, and the same
# qualified extraction/conversion helpers RD1-C4-E used. Only 6 unique
# incumbent trials x 3 basins are extracted (not all 24 trials x 400 basins),
# so this is a small fraction of RD1-C4-E's own cost; time/memory are sized
# with generous headroom off RD1-C4-E's own real measurement (job 46183269:
# ~2h/~2.7GB peak RSS for the full 24 x 400 raw-space pass) rather than a
# tight extrapolation.
#
# Performs no scientific classification, selects no threshold/winner, and
# closes nothing: it only authenticates the same frozen roster RD1-C4-E used,
# calls the already-qualified extraction/rendering helpers, and serializes
# compact figures + a hash-verified manifest. Every figure is explicitly
# labeled as a descriptive run-progress example, never winner/promotion
# evidence.
#
# No GPU, no training, no inference, no W&B, no controller, no sealed scope.

set -uo pipefail

FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
ENV_PREFIX="${ENV_PREFIX:-${FLASHNH_BASE}/envs/flashnh-moriah}"
REPO_CLONE_DIR="${REPO_CLONE_DIR:-${FLASHNH_BASE}/repos/flash-nh-rd1c4-d1}"
REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"

RD1_C4_F_TRIAL_LIST="${RD1_C4_F_TRIAL_LIST:?export RD1_C4_F_TRIAL_LIST=<frozen 24-trial roster JSON>}"
RD1_C4_F_CONTRACT="${RD1_C4_F_CONTRACT:?export RD1_C4_F_CONTRACT=<frozen fixed-support contract JSON>}"
RD1_C4_F_PACKAGE_ROOT="${RD1_C4_F_PACKAGE_ROOT:?export RD1_C4_F_PACKAGE_ROOT=<qualified package root>}"
RD1_C4_F_SELECTION_MANIFEST="${RD1_C4_F_SELECTION_MANIFEST:?export RD1_C4_F_SELECTION_MANIFEST=<hydrograph_basin_selection_manifest.json>}"
RD1_C4_F_CONFIG_TABLE="${RD1_C4_F_CONFIG_TABLE:?export RD1_C4_F_CONFIG_TABLE=<configuration_level_table.csv>}"
RD1_C4_F_OUT_DIR="${RD1_C4_F_OUT_DIR:?export RD1_C4_F_OUT_DIR=<fresh, not-yet-existing output dir>}"

for _path in "${RD1_C4_F_TRIAL_LIST}" "${RD1_C4_F_CONTRACT}" "${RD1_C4_F_SELECTION_MANIFEST}" "${RD1_C4_F_CONFIG_TABLE}"; do
    if [ ! -f "${_path}" ]; then
        echo "REFUSING: required input file does not exist: ${_path}" >&2
        exit 2
    fi
done
if [ ! -d "${RD1_C4_F_PACKAGE_ROOT}" ]; then
    echo "REFUSING: package root is not a directory: ${RD1_C4_F_PACKAGE_ROOT}" >&2
    exit 2
fi
for _path in "${RD1_C4_F_PACKAGE_ROOT}" "${RD1_C4_F_TRIAL_LIST}" "${RD1_C4_F_CONTRACT}" "${RD1_C4_F_OUT_DIR}"; do
    case "${_path}" in
        *temporal_test*|*spatial_holdout*|*california*|*sealed*)
            echo "REFUSING: input path names a sealed/protected scope: ${_path}" >&2
            exit 2
            ;;
    esac
done
if [ -e "${RD1_C4_F_OUT_DIR}" ] || [ -L "${RD1_C4_F_OUT_DIR}" ]; then
    echo "REFUSING: RD1_C4_F_OUT_DIR already exists: ${RD1_C4_F_OUT_DIR}" >&2
    exit 2
fi

mkdir -p "$(dirname "${RD1_C4_F_OUT_DIR}")"
RESULT_LOG="${FLASHNH_BASE}/tmp/rd1_c4_f/hydrograph_logs/rd1_c4_f_hydrograph_${SLURM_JOB_ID:-local}.log"
mkdir -p "$(dirname "${RESULT_LOG}")"
exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== RD1-C4-F descriptive hydrograph supplement run ==="
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
export MPLBACKEND=Agg

echo "--- input identities ---"
sha256sum "${RD1_C4_F_TRIAL_LIST}"
sha256sum "${RD1_C4_F_CONTRACT}"
sha256sum "${RD1_C4_F_SELECTION_MANIFEST}"
sha256sum "${RD1_C4_F_CONFIG_TABLE}"

/usr/bin/time -v python -m src.baseline.rd1_c4_f_hydrograph_runner \
    --trial-list-path "${RD1_C4_F_TRIAL_LIST}" \
    --contract-path "${RD1_C4_F_CONTRACT}" \
    --package-root "${RD1_C4_F_PACKAGE_ROOT}" \
    --selection-manifest-path "${RD1_C4_F_SELECTION_MANIFEST}" \
    --configuration-table-path "${RD1_C4_F_CONFIG_TABLE}" \
    --out-dir "${RD1_C4_F_OUT_DIR}"
STATUS=$?

echo "=== RD1-C4-F descriptive hydrograph supplement done (exit ${STATUS}) ==="
echo "Evidence bundle (if successful): ${RD1_C4_F_OUT_DIR}"
exit ${STATUS}

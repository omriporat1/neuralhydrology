#!/usr/bin/env bash
#SBATCH --job-name=flashnh-render-hydrographs
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/render-hydrographs-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/render-hydrographs-%j.err
#
# Stage 1 hydrograph rendering (Part L.3 / L.3b): renders the compact
# comparison panel + full 24-basin atlas for one already-evaluated NH
# checkpoint, using the committed scripts/render_stage1_hydrographs.py CLI
# unchanged. Reads only: an existing validation_results.p (never re-runs
# training or evaluation), the certified package's per-basin NetCDFs (for
# basin-area self-derivation), and the existing Part C atlas-selection CSV.
# CPU-only (glacier, no GRES): the renderer imports numpy/pandas/matplotlib
# only, never torch/neuralhydrology (see hydrograph_rendering.py), so this
# follows the same no-GPU resource profile already used by
# run_stage1_seed_percentile_diagnostics_moriah.sbatch for the same class of
# pickle-reading + figure-generation workload.
#
# Usage:
#   sbatch scripts/render_stage1_hydrographs_moriah.sbatch \
#       <run_dir> <epoch> <atlas_csv_repo_relative_path> <out_dir_name>
# All four are required positional args -- no hard-coded run/epoch, so this
# script is reusable for future candidates, not emb128x64_seedA-specific.
# Package root, target variable, and lead hours default to the current
# Stage 1 scientific baseline and can be overridden via env vars.

set -uo pipefail

RUN_DIR="${1:?USAGE: sbatch $0 <run_dir> <epoch> <atlas_csv_repo_relative_path> <out_dir_name>}"
EPOCH="${2:?missing epoch}"
ATLAS_CSV_REL="${3:?missing atlas_csv_repo_relative_path}"
OUT_DIR_NAME="${4:?missing out_dir_name}"

FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
ENV_PREFIX="${FLASHNH_BASE}/envs/flashnh-moriah"

# Same authoritative clean clone as the pilot launcher -- NOT the other,
# dirty secondary clone elsewhere under omripo's home directory.
PILOT_REPO_CLONE_DIR="${PILOT_REPO_CLONE_DIR:-/sci/labs/efratmorin/omripo/PhD/Python/neuralhydrology}"
REPO_CLONE_DIR="${PILOT_REPO_CLONE_DIR}"
REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"

PACKAGE_ROOT="${RENDER_PACKAGE_ROOT:-${FLASHNH_BASE}/data/stage1_scientific_package_v002}"
TARGET_VARIABLE="${RENDER_TARGET_VARIABLE:-qobs_mm_per_h_lead06}"
LEAD_HOURS="${RENDER_LEAD_HOURS:-6}"

ATLAS_CSV="${REPO_WORKDIR}/${ATLAS_CSV_REL}"
OUT_DIR="${FLASHNH_BASE}/tmp/${OUT_DIR_NAME}"

RESULT_DIR="${FLASHNH_BASE}/evidence/render_hydrographs_${SLURM_JOB_ID}"
mkdir -p "${RESULT_DIR}"
RESULT_LOG="${RESULT_DIR}/render_human.log"
exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== Stage 1 hydrograph rendering: run_dir=${RUN_DIR} epoch=${EPOCH} ==="
hostname
date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-unknown}"
scontrol show job "${SLURM_JOB_ID}" || true

test -d "${RUN_DIR}" || { echo "MISSING run dir: ${RUN_DIR}"; exit 1; }
test -d "${PACKAGE_ROOT}" || { echo "MISSING transferred package: ${PACKAGE_ROOT}"; exit 1; }
test -f "${ATLAS_CSV}" || { echo "MISSING atlas selection csv: ${ATLAS_CSV}"; exit 1; }

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

# Pre-launch repository guard: refuse to render against a repo work
# directory whose tracked files have been locally modified relative to the
# reviewed commit -- mirrors the pilot launcher's own guard. Untracked
# files (e.g. this task's own generated evidence/reports) are tolerated.
test -d "${REPO_WORKDIR}" || { echo "MISSING expected repo work directory: ${REPO_WORKDIR}"; exit 1; }
cd "${REPO_WORKDIR}"
git log --oneline -3
SOURCE_COMMIT=$(git rev-parse HEAD)
GIT_STATUS_PORCELAIN=$(git status --porcelain)
echo "SOURCE_COMMIT: ${SOURCE_COMMIT}"
echo "--- git status --porcelain (${REPO_WORKDIR}) ---"
echo "${GIT_STATUS_PORCELAIN}"
echo "-------------------------------------------------"
TRACKED_CHANGES=$(echo "${GIT_STATUS_PORCELAIN}" | grep -v '^??' || true)
if [ -n "${TRACKED_CHANGES}" ]; then
    echo "REFUSING to render: tracked files are locally modified in ${REPO_WORKDIR}:" >&2
    echo "${TRACKED_CHANGES}" >&2
    exit 1
fi

echo "=== input checksums ==="
RESULT_PICKLE="${RUN_DIR}/validation/model_epoch$(printf '%03d' "${EPOCH}")/validation_results.p"
CHECKPOINT="${RUN_DIR}/model_epoch$(printf '%03d' "${EPOCH}").pt"
for f in "${RESULT_PICKLE}" "${CHECKPOINT}" "${ATLAS_CSV}" "${PACKAGE_ROOT}/manifests/package_manifest.json"; do
    if [ -f "${f}" ]; then
        sha256sum "${f}"
    else
        echo "NOTE: not present at expected flat path: ${f}"
    fi
done

echo "=== launching scripts/render_stage1_hydrographs.py ==="
python scripts/render_stage1_hydrographs.py \
    --run-dir "${RUN_DIR}" \
    --period validation \
    --epoch "${EPOCH}" \
    --package-root "${PACKAGE_ROOT}" \
    --target-variable "${TARGET_VARIABLE}" \
    --lead-hours "${LEAD_HOURS}" \
    --atlas-csv "${ATLAS_CSV}" \
    --out-dir "${OUT_DIR}" \
    --mode both
RENDER_STATUS=$?

echo "=== render exit code: ${RENDER_STATUS} ==="

if [ "${RENDER_STATUS}" -eq 0 ] && [ -d "${OUT_DIR}" ]; then
    echo "=== output inventory ==="
    find "${OUT_DIR}" -type f -exec ls -la {} \;
    echo "=== output checksums ==="
    find "${OUT_DIR}" -type f -exec sha256sum {} \;
fi

echo "=== sacct accounting (may be incomplete until job fully finalizes) ==="
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS,NNodes,NodeList,ReqCPUS,ReqMem 2>&1 || true

echo "=== done (exit ${RENDER_STATUS}) ==="
exit ${RENDER_STATUS}

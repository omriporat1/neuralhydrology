#!/usr/bin/env bash
#SBATCH --job-name=flashnh-fullpop-lead06-seq24-seed-percentile-diagnostics
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/fullpop-lead06-seq24-seed-percentile-diagnostics-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/fullpop-lead06-seq24-seed-percentile-diagnostics-%j.err
#
# Stage 1 validation and optimization foundation, Part A: full-percentile
# diagnostic closure over the existing 11-epoch seed run's development-
# validation-only per-basin raw-space NSE. Reads only the validation_results.p
# pickles NeuralHydrology already wrote during seed training -- never re-runs
# evaluation, never touches temporal-test or spatial-holdout data, never
# selects/freezes a checkpoint. Runs on `glacier` (CPU, no GRES) since
# scripts/compute_seed_percentile_diagnostics.py imports numpy only, never
# torch/neuralhydrology. Same resource profile as the existing checkpoint-
# ranking sweep job (mem=16G was sufficient there for the same per-epoch
# pickle-reading workload); the interactive login-node invocation of this
# exact script was OOM-killed (exit 137), which is why this runs under Slurm.

set -uo pipefail

FLASHNH_BASE="/sci/labs/efratmorin/omripo/Flash-NH"
ENV_PREFIX="${FLASHNH_BASE}/envs/flashnh-moriah"
REPO_CLONE_DIR="${FLASHNH_BASE}/repos/flash-nh"
REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"

TRANSFERRED_PACKAGE="${FLASHNH_BASE}/data/stage1_scientific_package_v002"
BASE_RUN_DIR="${FLASHNH_BASE}/runs/stage1_full_population_nh_config_lead06_seq24_seed_v001/development/runs/stage1_compact_lead06_seq24_v001_2507_154104"
CONTINUE_RUN_DIR="${BASE_RUN_DIR}/continue_training_from_epoch003"

OUT_DIR="${FLASHNH_BASE}/tmp/stage1_seed_percentile_diagnostics_v001"
mkdir -p "${OUT_DIR}"

RESULT_LOG="${OUT_DIR}/percentile_diagnostics_human.log"
exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== part A: seed percentile diagnostic closure ==="
hostname
date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "BASE_RUN_DIR: ${BASE_RUN_DIR}"
echo "CONTINUE_RUN_DIR: ${CONTINUE_RUN_DIR}"

test -d "${BASE_RUN_DIR}" || { echo "MISSING run dir: ${BASE_RUN_DIR}"; exit 1; }
test -d "${CONTINUE_RUN_DIR}" || { echo "MISSING run dir: ${CONTINUE_RUN_DIR}"; exit 1; }
test -d "${TRANSFERRED_PACKAGE}" || { echo "MISSING transferred package: ${TRANSFERRED_PACKAGE}"; exit 1; }

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

cd "${REPO_WORKDIR}"
git log --oneline -3

python scripts/compute_seed_percentile_diagnostics.py \
    --epoch-run-dir 1="${BASE_RUN_DIR}" \
    --epoch-run-dir 2="${BASE_RUN_DIR}" \
    --epoch-run-dir 3="${BASE_RUN_DIR}" \
    --epoch-run-dir 4="${CONTINUE_RUN_DIR}" \
    --epoch-run-dir 5="${CONTINUE_RUN_DIR}" \
    --epoch-run-dir 6="${CONTINUE_RUN_DIR}" \
    --epoch-run-dir 7="${CONTINUE_RUN_DIR}" \
    --epoch-run-dir 8="${CONTINUE_RUN_DIR}" \
    --epoch-run-dir 9="${CONTINUE_RUN_DIR}" \
    --epoch-run-dir 10="${CONTINUE_RUN_DIR}" \
    --epoch-run-dir 11="${CONTINUE_RUN_DIR}" \
    --package-root "${TRANSFERRED_PACKAGE}" \
    --target-variable qobs_mm_per_h_lead06 \
    --lead-hours 6 \
    --out-dir "${OUT_DIR}"
DIAG_STATUS=$?

echo "=== done (exit ${DIAG_STATUS}) ==="
exit ${DIAG_STATUS}

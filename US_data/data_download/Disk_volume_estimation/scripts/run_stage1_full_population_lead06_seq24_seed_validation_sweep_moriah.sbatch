#!/usr/bin/env bash
#SBATCH --job-name=flashnh-fullpop-lead06-seq24-seed-validation-sweep
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/fullpop-lead06-seq24-seed-validation-sweep-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/fullpop-lead06-seq24-seed-validation-sweep-%j.err
#
# Part E: rank the seed run's candidate checkpoints by development-
# validation-only raw-space (m^3/s) NSE. Reads the per-epoch validation
# results NeuralHydrology already wrote DURING training itself (seed profile
# sets validate_every: 1, so every saved checkpoint already has a
# validation/model_epochNNN/validation_results.p) -- never re-runs
# evaluation, never touches temporal-test or spatial-holdout data. Runs on
# `glacier` (CPU, no GRES) since scripts/evaluate_stage1_seed_raw_space.py
# imports numpy/xarray/pandas only, never torch/neuralhydrology.
#
# Output is a RANKING artifact for review, not an automatic freeze: Part E's
# actual checkpoint-selection decision is a separate, explicitly recorded
# step taken after a human/agent reviews this table -- see
# docs/decision_log.md's "initial full-population seed training profile"
# entry and scripts/evaluate_stage1_seed_raw_space.py's module docstring.

set -uo pipefail

FLASHNH_BASE="/sci/labs/efratmorin/omripo/Flash-NH"
ENV_PREFIX="${FLASHNH_BASE}/envs/flashnh-moriah"
REPO_CLONE_DIR="${FLASHNH_BASE}/repos/flash-nh"
REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"

TRANSFERRED_PACKAGE="${FLASHNH_BASE}/data/stage1_scientific_package_v002"
GENERATED_OUT_DIR="${FLASHNH_BASE}/runs/stage1_full_population_nh_config_lead06_seq24_seed_v001"
RUN_DIR_MARKER_FILE="${GENERATED_OUT_DIR}/SEED_TRAIN_ACTIVE_RUN_DIR.txt"

# Override with `sbatch --export=RUN_DIR=/explicit/path ...` to sweep a
# specific run directory; defaults to whatever the training launcher last
# recorded as the active seed run.
if [ -z "${RUN_DIR:-}" ]; then
    test -f "${RUN_DIR_MARKER_FILE}" || { echo "MISSING run-dir marker: ${RUN_DIR_MARKER_FILE} (set RUN_DIR explicitly, or run training first)"; exit 1; }
    RUN_DIR="$(cat "${RUN_DIR_MARKER_FILE}")"
fi

RESULT_DIR="${FLASHNH_BASE}/evidence/fullpop_lead06_seq24_seed_validation_sweep_${SLURM_JOB_ID}"
mkdir -p "${RESULT_DIR}"
RESULT_LOG="${RESULT_DIR}/validation_sweep_human.log"
exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== part E: development-validation-only checkpoint ranking sweep ==="
hostname
date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "RUN_DIR: ${RUN_DIR}"

test -d "${RUN_DIR}" || { echo "MISSING run dir: ${RUN_DIR}"; exit 1; }
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

python scripts/evaluate_stage1_seed_raw_space.py validation-sweep \
    --run-dir "${RUN_DIR}" \
    --package-root "${TRANSFERRED_PACKAGE}" \
    --target-variable qobs_mm_per_h_lead06 \
    --lead-hours 6 \
    --out-json "${RESULT_DIR}/checkpoint_selection_sweep.json"
SWEEP_STATUS=$?

echo "=== done (exit ${SWEEP_STATUS}) ==="
exit ${SWEEP_STATUS}

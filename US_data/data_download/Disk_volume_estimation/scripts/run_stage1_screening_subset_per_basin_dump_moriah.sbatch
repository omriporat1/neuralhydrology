#!/usr/bin/env bash
#SBATCH --job-name=flashnh-fullpop-lead06-seq24-screening-subset-per-basin-dump
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/fullpop-lead06-seq24-screening-subset-per-basin-dump-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/fullpop-lead06-seq24-screening-subset-per-basin-dump-%j.err
#
# Stage 1 validation and optimization foundation, Part D support: fill in
# per-basin raw-space validation metric tables for the seed run's remaining
# epochs (1,2,3,4,5,8,10,11 -- epochs 6,7,9 already have persisted per-basin
# CSVs from the earlier seed_validation_review_v001 bundle). Reuses
# scripts/dump_per_basin_table.py unmodified, which itself reuses
# src.baseline.nh_seed_evaluation.raw_space_metrics_for_run_period and reads
# only the already-computed validation_results.p pickles NeuralHydrology
# wrote during seed training -- never reruns NH inference, never touches
# temporal-test/spatial-holdout data. Same CPU-only resource profile as the
# existing percentile-diagnostics job (glacier partition, no GRES, mem=16G).

set -uo pipefail

FLASHNH_BASE="/sci/labs/efratmorin/omripo/Flash-NH"
ENV_PREFIX="${FLASHNH_BASE}/envs/flashnh-moriah"
REPO_CLONE_DIR="${FLASHNH_BASE}/repos/flash-nh"
REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"

TRANSFERRED_PACKAGE="${FLASHNH_BASE}/data/stage1_scientific_package_v002"
BASE_RUN_DIR="${FLASHNH_BASE}/runs/stage1_full_population_nh_config_lead06_seq24_seed_v001/development/runs/stage1_compact_lead06_seq24_v001_2507_154104"
CONTINUE_RUN_DIR="${BASE_RUN_DIR}/continue_training_from_epoch003"

OUT_DIR="${FLASHNH_BASE}/tmp/stage1_screening_subset_per_basin_dump_v001"
mkdir -p "${OUT_DIR}"

RESULT_LOG="${OUT_DIR}/per_basin_dump_human.log"
exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== Part D support: per-basin dump for epochs 1,2,3,4,5,8,10,11 ==="
hostname
date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"

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

STATUS=0
for epoch_spec in "1=${BASE_RUN_DIR}" "2=${BASE_RUN_DIR}" "3=${BASE_RUN_DIR}" \
                   "4=${CONTINUE_RUN_DIR}" "5=${CONTINUE_RUN_DIR}" "8=${CONTINUE_RUN_DIR}" \
                   "10=${CONTINUE_RUN_DIR}" "11=${CONTINUE_RUN_DIR}"; do
    epoch="${epoch_spec%%=*}"
    run_dir="${epoch_spec#*=}"
    epoch_padded=$(printf "%03d" "${epoch}")
    echo "--- epoch ${epoch} (run_dir=${run_dir}) ---"
    python scripts/dump_per_basin_table.py \
        --run-dir "${run_dir}" \
        --epoch "${epoch}" \
        --package-root "${TRANSFERRED_PACKAGE}" \
        --target-variable qobs_mm_per_h_lead06 \
        --lead-hours 6 \
        --out-csv "${OUT_DIR}/epoch${epoch_padded}/epoch${epoch_padded}_per_basin_metrics.csv"
    rc=$?
    if [ ${rc} -ne 0 ]; then
        echo "FAILED epoch ${epoch} (exit ${rc})"
        STATUS=1
    fi
done

echo "=== done (exit ${STATUS}) ==="
exit ${STATUS}

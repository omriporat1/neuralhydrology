#!/usr/bin/env bash
#SBATCH --job-name=flashnh-fullpop-lead06-seq24-seed-frozen-eval
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/fullpop-lead06-seq24-seed-frozen-eval-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/fullpop-lead06-seq24-seed-frozen-eval-%j.err
#
# Part F: evaluate the ONE frozen checkpoint (selected by Part E from
# development-validation-only raw-space NSE -- see
# run_stage1_full_population_lead06_seq24_seed_validation_sweep_moriah.sbatch)
# on (1) the development temporal test (2025) and (2) the spatial-holdout
# test (2025, 250 non-CA basins), reusing the development run's fitted
# scaler unchanged for the holdout evaluation. Must be run AFTER EPOCH has
# been set to the reviewed, frozen Part E selection -- this script does not
# choose the epoch itself.
#
# Partition/GRES matches the training/GPU-smoke convention (--partition=catfish
# --gres=gpu:l4:1); override at submission time only for the one documented
# resource correction the directive allows, never silently.

set -uo pipefail

FLASHNH_BASE="/sci/labs/efratmorin/omripo/Flash-NH"
ENV_PREFIX="${FLASHNH_BASE}/envs/flashnh-moriah"
REPO_CLONE_DIR="${FLASHNH_BASE}/repos/flash-nh"
REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"

TRANSFERRED_PACKAGE="${FLASHNH_BASE}/data/stage1_scientific_package_v002"
GENERATED_OUT_DIR="${FLASHNH_BASE}/runs/stage1_full_population_nh_config_lead06_seq24_seed_v001"
DEVELOPMENT_DIR="${GENERATED_OUT_DIR}/development"
SPATIAL_HOLDOUT_DIR="${GENERATED_OUT_DIR}/spatial_holdout"
RUN_DIR_MARKER_FILE="${GENERATED_OUT_DIR}/SEED_TRAIN_ACTIVE_RUN_DIR.txt"

# EPOCH must be set explicitly to Part E's frozen, reviewed selection --
# refuse to guess a default.
EPOCH="${EPOCH:?set EPOCH to the frozen Part E checkpoint selection, e.g. --export=EPOCH=17}"
METRICS=(NSE RMSE KGE Pearson-r Beta-KGE)

if [ -z "${RUN_DIR:-}" ]; then
    test -f "${RUN_DIR_MARKER_FILE}" || { echo "MISSING run-dir marker: ${RUN_DIR_MARKER_FILE} (set RUN_DIR explicitly)"; exit 1; }
    RUN_DIR="$(cat "${RUN_DIR_MARKER_FILE}")"
fi
HOLDOUT_EVAL_RUN_DIR="${FLASHNH_BASE}/runs/stage1_seed_holdout_eval_epoch$(printf '%03d' "${EPOCH}")"

RESULT_DIR="${FLASHNH_BASE}/evidence/fullpop_lead06_seq24_seed_frozen_eval_${SLURM_JOB_ID}"
mkdir -p "${RESULT_DIR}"
RESULT_LOG="${RESULT_DIR}/frozen_eval_human.log"
exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== part F: frozen-checkpoint temporal-test + spatial-holdout evaluation ==="
hostname
date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "RUN_DIR (development, already trained): ${RUN_DIR}"
echo "EPOCH (frozen, Part E selection): ${EPOCH}"
echo "HOLDOUT_EVAL_RUN_DIR (to be prepared): ${HOLDOUT_EVAL_RUN_DIR}"

test -d "${RUN_DIR}" || { echo "MISSING development run dir: ${RUN_DIR}"; exit 1; }
test -f "${RUN_DIR}/model_epoch$(printf '%03d' "${EPOCH}").pt" || { echo "MISSING frozen checkpoint for epoch ${EPOCH}"; exit 1; }
test -d "${TRANSFERRED_PACKAGE}" || { echo "MISSING transferred package: ${TRANSFERRED_PACKAGE}"; exit 1; }
test -f "${SPATIAL_HOLDOUT_DIR}/TEST_ONLY_DO_NOT_TRAIN.txt" || { echo "MISSING holdout marker in ${SPATIAL_HOLDOUT_DIR}"; exit 1; }

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
module load nvidia/580.95.05
module load cuda/12.8.1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
which python

nvidia-smi
python -u -c "
import torch
print('PyTorch:', torch.__version__)
if not torch.cuda.is_available():
    raise SystemExit('CUDA not available -- check module loads and GPU allocation')
print('device:', torch.cuda.get_device_name(0))
"

cd "${REPO_WORKDIR}"
git log --oneline -3
SOURCE_COMMIT=$(git rev-parse HEAD)

echo "=== capture pre-eval sha256 of protected development training artifacts ==="
WEIGHT_STEM="model_epoch$(printf '%03d' "${EPOCH}")"
PRE_EVAL_SHA256_JSON="${RESULT_DIR}/pre_eval_sha256.json"
{
    echo "{"
    echo "  \"config.yml\": \"$(sha256sum "${RUN_DIR}/config.yml" | cut -d' ' -f1)\","
    echo "  \"${WEIGHT_STEM}.pt\": \"$(sha256sum "${RUN_DIR}/${WEIGHT_STEM}.pt" | cut -d' ' -f1)\","
    echo "  \"train_data/train_data_scaler.yml\": \"$(sha256sum "${RUN_DIR}/train_data/train_data_scaler.yml" | cut -d' ' -f1)\""
    echo "}"
} > "${PRE_EVAL_SHA256_JSON}"
cat "${PRE_EVAL_SHA256_JSON}"

echo "=== (1) development temporal test (2025) ==="
python scripts/run_stage1_nh.py eval "${RUN_DIR}" --period test --epoch "${EPOCH}" --metrics "${METRICS[@]}"
DEV_TEST_STATUS=$?
echo "development test-period evaluation exit code: ${DEV_TEST_STATUS}"

echo "=== (2a) prepare external-scaler spatial-holdout eval run dir ==="
python scripts/prepare_stage1_seed_holdout_eval_run_dir.py \
    --development-run-dir "${RUN_DIR}" \
    --epoch "${EPOCH}" \
    --holdout-generated-dir "${SPATIAL_HOLDOUT_DIR}" \
    --out-run-dir "${HOLDOUT_EVAL_RUN_DIR}" \
    --force
PREP_STATUS=$?
echo "holdout eval run_dir preparation exit code: ${PREP_STATUS}"

HOLDOUT_TEST_STATUS=1
if [ "${PREP_STATUS}" -eq 0 ]; then
    echo "=== (2b) spatial-holdout test (2025, 250 non-CA basins, reused scaler) ==="
    python scripts/run_stage1_nh.py eval "${HOLDOUT_EVAL_RUN_DIR}" --period test --epoch "${EPOCH}" --metrics "${METRICS[@]}"
    HOLDOUT_TEST_STATUS=$?
    echo "spatial-holdout test-period evaluation exit code: ${HOLDOUT_TEST_STATUS}"
fi

echo "=== raw-space (m^3/s) metrics: development temporal test ==="
python scripts/evaluate_stage1_seed_raw_space.py single \
    --run-dir "${RUN_DIR}" --period test --epoch "${EPOCH}" \
    --package-root "${TRANSFERRED_PACKAGE}" --target-variable qobs_mm_per_h_lead06 --lead-hours 6 \
    --out-json "${RESULT_DIR}/development_temporal_test_raw_space.json"
DEV_RAW_STATUS=$?

HOLDOUT_RAW_STATUS=1
if [ "${HOLDOUT_TEST_STATUS}" -eq 0 ]; then
    echo "=== raw-space (m^3/s) metrics: spatial holdout test ==="
    python scripts/evaluate_stage1_seed_raw_space.py single \
        --run-dir "${HOLDOUT_EVAL_RUN_DIR}" --period test --epoch "${EPOCH}" \
        --package-root "${TRANSFERRED_PACKAGE}" --target-variable qobs_mm_per_h_lead06 --lead-hours 6 \
        --out-json "${RESULT_DIR}/spatial_holdout_test_raw_space.json"
    HOLDOUT_RAW_STATUS=$?
fi

echo "=== verify development training artifacts were not modified by evaluation ==="
POST_EVAL_SHA256_JSON="${RESULT_DIR}/post_eval_sha256.json"
{
    echo "{"
    echo "  \"config.yml\": \"$(sha256sum "${RUN_DIR}/config.yml" | cut -d' ' -f1)\","
    echo "  \"${WEIGHT_STEM}.pt\": \"$(sha256sum "${RUN_DIR}/${WEIGHT_STEM}.pt" | cut -d' ' -f1)\","
    echo "  \"train_data/train_data_scaler.yml\": \"$(sha256sum "${RUN_DIR}/train_data/train_data_scaler.yml" | cut -d' ' -f1)\""
    echo "}"
} > "${POST_EVAL_SHA256_JSON}"
ARTIFACT_UNCHANGED_STATUS=0
diff "${PRE_EVAL_SHA256_JSON}" "${POST_EVAL_SHA256_JSON}" || ARTIFACT_UNCHANGED_STATUS=1

OVERALL_STATUS=0
[ "${DEV_TEST_STATUS}" -eq 0 ] || OVERALL_STATUS=1
[ "${PREP_STATUS}" -eq 0 ] || OVERALL_STATUS=1
[ "${HOLDOUT_TEST_STATUS}" -eq 0 ] || OVERALL_STATUS=1
[ "${DEV_RAW_STATUS}" -eq 0 ] || OVERALL_STATUS=1
[ "${HOLDOUT_RAW_STATUS}" -eq 0 ] || OVERALL_STATUS=1
[ "${ARTIFACT_UNCHANGED_STATUS}" -eq 0 ] || OVERALL_STATUS=1

python -c "
import json
out = {
    'status': 'PASS' if ${OVERALL_STATUS} == 0 else 'FAIL',
    'run_dir': '${RUN_DIR}',
    'holdout_eval_run_dir': '${HOLDOUT_EVAL_RUN_DIR}',
    'epoch': ${EPOCH},
    'development_test_exit_code': ${DEV_TEST_STATUS},
    'holdout_prep_exit_code': ${PREP_STATUS},
    'holdout_test_exit_code': ${HOLDOUT_TEST_STATUS},
    'development_raw_space_exit_code': ${DEV_RAW_STATUS},
    'holdout_raw_space_exit_code': ${HOLDOUT_RAW_STATUS},
    'training_artifacts_unchanged': ${ARTIFACT_UNCHANGED_STATUS} == 0,
    'development_temporal_test_raw_space_json': '${RESULT_DIR}/development_temporal_test_raw_space.json',
    'spatial_holdout_test_raw_space_json': '${RESULT_DIR}/spatial_holdout_test_raw_space.json',
    'pre_eval_sha256_json': '${PRE_EVAL_SHA256_JSON}',
    'post_eval_sha256_json': '${POST_EVAL_SHA256_JSON}',
    'slurm_job_id': '${SLURM_JOB_ID}',
    'source_commit': '${SOURCE_COMMIT}',
    'hostname': __import__('socket').gethostname(),
}
json.dump(out, open('${RESULT_DIR}/frozen_eval_result.json', 'w'), indent=2)
print(json.dumps(out, indent=2))
"

echo "=== done (exit ${OVERALL_STATUS}) ==="
exit ${OVERALL_STATUS}

#!/usr/bin/env bash
#SBATCH --job-name=flashnh-nh-lead06-seq24-evaluate
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/nh-lead06-seq24-evaluate-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/nh-lead06-seq24-evaluate-%j.err
#
# Explicit validation (2024) + test (2025) evaluation of the epoch-2 checkpoint
# from the successful compact NH training smoke (job 45625002). Does not
# retrain: only invokes `scripts/run_stage1_nh.py eval ...` against the
# already-completed run directory below, then runs a companion evidence check
# (check_stage1_nh_evaluation.py) proving the checkpoint/scaler/basin/date/
# finiteness expectations hold and that no training artifact was modified.
# This is integration verification and artifact production, not scientific
# model assessment -- do not over-interpret the metric values it produces.
#
# Partition/GRES reused verbatim from the paired GPU training smoke launcher
# (run_stage1_nh_lead06_seq24_gpu_smoke_moriah.sbatch); resources here are
# reduced (2 CPUs, 16G, 15 min) since evaluating 32 basins twice is far
# lighter than 2 epochs of training.

set -uo pipefail

FLASHNH_BASE="/sci/labs/efratmorin/omripo/Flash-NH"
ENV_PREFIX="${FLASHNH_BASE}/envs/flashnh-moriah"
REPO_CLONE_DIR="${FLASHNH_BASE}/repos/flash-nh"
REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"

# Completed training run being evaluated (job 45625002, CPU-preflight
# provenance 45624926, epoch 2 of 2, PASS). Override via
# --export=RUN_DIR=...,TRAIN_JOB_ID=... only to evaluate a different
# already-completed compact run with the same schema.
RUN_DIR="${RUN_DIR:-${FLASHNH_BASE}/runs/stage1_nh_config_lead06_seq24_v001/runs/stage1_compact_lead06_seq24_v001_2307_135829}"
TRAIN_JOB_ID="${TRAIN_JOB_ID:-45625002}"
EPOCH="${EPOCH:-2}"
EXPECTED_TARGET_VARIABLE="qobs_mm_per_h_lead06"
EXPECTED_BASIN_COUNT=32
EXPECTED_VALIDATION_YEAR=2024
EXPECTED_TEST_YEAR=2025
METRICS=(NSE RMSE KGE Pearson-r Beta-KGE)
WEIGHT_STEM=$(printf "model_epoch%03d" "${EPOCH}")

RESULT_DIR="${FLASHNH_BASE}/evidence/nh_lead06_seq24_evaluate_${SLURM_JOB_ID}"
mkdir -p "${RESULT_DIR}"
RESULT_LOG="${RESULT_DIR}/evaluate_human.log"
exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== preflight ==="
hostname
date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "RUN_DIR: ${RUN_DIR}"
echo "TRAIN_JOB_ID (provenance): ${TRAIN_JOB_ID}"
echo "EPOCH: ${EPOCH} (${WEIGHT_STEM}.pt)"

echo "=== gate: required run artifacts must already exist (no retraining here) ==="
CONFIG_YML="${RUN_DIR}/config.yml"
CHECKPOINT_PT="${RUN_DIR}/${WEIGHT_STEM}.pt"
SCALER_YML="${RUN_DIR}/train_data/train_data_scaler.yml"
for f in "${CONFIG_YML}" "${CHECKPOINT_PT}" "${SCALER_YML}"; do
    test -f "${f}" || { echo "MISSING required run artifact: ${f}"; exit 1; }
    echo "found: ${f}"
done

echo "=== capture pre-evaluation sha256 of protected training artifacts ==="
PRE_EVAL_SHA256_JSON="${RESULT_DIR}/pre_eval_sha256.json"
PROTECTED_RELPATHS=(config.yml "${WEIGHT_STEM}.pt" train_data/train_data_scaler.yml)
{
    echo "{"
    n=${#PROTECTED_RELPATHS[@]}
    for i in "${!PROTECTED_RELPATHS[@]}"; do
        rel="${PROTECTED_RELPATHS[$i]}"
        hash=$(sha256sum "${RUN_DIR}/${rel}" | cut -d' ' -f1)
        sep=$([ "$i" -lt "$((n - 1))" ] && echo "," || echo "")
        printf '  "%s": "%s"%s\n' "${rel}" "${hash}" "${sep}"
    done
    echo "}"
} > "${PRE_EVAL_SHA256_JSON}"
cat "${PRE_EVAL_SHA256_JSON}"

if ! type module &>/dev/null; then
    for _modinit in \
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
cuda_ok = torch.cuda.is_available()
print('cuda available:', cuda_ok)
if not cuda_ok:
    raise SystemExit('CUDA not available -- check module loads and GPU allocation')
print('device:', torch.cuda.get_device_name(0))
"

cd "${REPO_WORKDIR}"
git log --oneline -3
SOURCE_COMMIT=$(git rev-parse HEAD)

echo "=== evaluate validation period (2024) ==="
python scripts/run_stage1_nh.py eval "${RUN_DIR}" --period validation --epoch "${EPOCH}" --metrics "${METRICS[@]}"
VALIDATION_STATUS=$?
echo "validation evaluation exit code: ${VALIDATION_STATUS}"

echo "=== evaluate test period (2025) ==="
python scripts/run_stage1_nh.py eval "${RUN_DIR}" --period test --epoch "${EPOCH}" --metrics "${METRICS[@]}"
TEST_STATUS=$?
echo "test evaluation exit code: ${TEST_STATUS}"

echo "=== evidence check (checkpoint/scaler/basins/dates/finiteness/no-overwrite) ==="
python scripts/check_stage1_nh_evaluation.py \
    --run-dir "${RUN_DIR}" \
    --epoch "${EPOCH}" \
    --expected-target-variable "${EXPECTED_TARGET_VARIABLE}" \
    --expected-basin-count "${EXPECTED_BASIN_COUNT}" \
    --expected-validation-year "${EXPECTED_VALIDATION_YEAR}" \
    --expected-test-year "${EXPECTED_TEST_YEAR}" \
    --expected-metrics "${METRICS[@]}" \
    --protected-relpath config.yml \
    --protected-relpath "${WEIGHT_STEM}.pt" \
    --protected-relpath train_data/train_data_scaler.yml \
    --pre-eval-sha256-json "${PRE_EVAL_SHA256_JSON}" \
    > "${RESULT_DIR}/evaluation_check_result.json"
CHECK_STATUS=$?
echo "evidence check exit code: ${CHECK_STATUS}"

OVERALL_STATUS=0
[ "${VALIDATION_STATUS}" -eq 0 ] || OVERALL_STATUS=1
[ "${TEST_STATUS}" -eq 0 ] || OVERALL_STATUS=1
[ "${CHECK_STATUS}" -eq 0 ] || OVERALL_STATUS=1

VALIDATION_OUTPUT_DIR="${RUN_DIR}/validation/${WEIGHT_STEM}"
TEST_OUTPUT_DIR="${RUN_DIR}/test/${WEIGHT_STEM}"

python -c "
import json
out = {
    'status': 'PASS' if ${OVERALL_STATUS} == 0 else 'FAIL',
    'run_dir': '${RUN_DIR}',
    'epoch': ${EPOCH},
    'validation_exit_code': ${VALIDATION_STATUS},
    'test_exit_code': ${TEST_STATUS},
    'evidence_check_exit_code': ${CHECK_STATUS},
    'validation_output_dir': '${VALIDATION_OUTPUT_DIR}',
    'test_output_dir': '${TEST_OUTPUT_DIR}',
    'metrics_requested': ['NSE', 'RMSE', 'KGE', 'Pearson-r', 'Beta-KGE'],
    'pre_eval_sha256_json': '${PRE_EVAL_SHA256_JSON}',
    'evidence_check_result_json': '${RESULT_DIR}/evaluation_check_result.json',
    'slurm_job_id': '${SLURM_JOB_ID}',
    'train_job_id': '${TRAIN_JOB_ID}',
    'source_commit': '${SOURCE_COMMIT}',
    'hostname': __import__('socket').gethostname(),
}
json.dump(out, open('${RESULT_DIR}/evaluate_result.json', 'w'), indent=2)
print(json.dumps(out, indent=2))
"

echo "=== done (exit ${OVERALL_STATUS}) ==="
exit ${OVERALL_STATUS}

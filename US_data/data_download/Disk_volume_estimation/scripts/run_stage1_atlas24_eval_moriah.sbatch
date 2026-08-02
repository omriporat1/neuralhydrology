#!/usr/bin/env bash
#SBATCH --job-name=flashnh-emb128x64-seedA-atlas24-eval
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/emb128x64-seedA-atlas24-eval-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/emb128x64-seedA-atlas24-eval-%j.err
#
# Evaluation-only derivative of the completed emb128x64_seedA run (epoch 6,
# frozen checkpoint): re-evaluates the SAME checkpoint + SAME fitted scaler
# against the fixed 24-basin hydrograph atlas instead of the original run's
# ~400-basin screening-subset validation scope. Diagnostic/visualization
# purposes only -- NOT authoritative for full-population validation, NOT
# usable for checkpoint/architecture selection, NOT a replacement for the
# original screening-subset validation result, NOT temporal-test or
# spatial-holdout evidence. Never retrains, never refits a scaler, never
# touches the original run directory, never touches temporal-test /
# spatial-holdout / California data.

set -uo pipefail

FLASHNH_BASE="/sci/labs/efratmorin/omripo/Flash-NH"
ENV_PREFIX="${FLASHNH_BASE}/envs/flashnh-moriah"
REPO_CLONE_DIR="${FLASHNH_BASE}/repos/flash-nh"
REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"

DEVELOPMENT_RUN_DIR="${FLASHNH_BASE}/runs/stage1_lead06_pilot_emb128x64_seedA_v001/runs/stage1_lead06_pilot_emb128x64_seedA_v001_2807_113156"
EPOCH=6
TRANSFERRED_PACKAGE="${FLASHNH_BASE}/data/stage1_scientific_package_v002"
ATLAS_CSV="${REPO_WORKDIR}/reports/stage1_validation_optimization_foundation_v001/part_c_hydrograph_atlas/selection_v001/hydrograph_atlas_basin_selection.csv"

DERIVATIVE_BASE="${FLASHNH_BASE}/runs/stage1_lead06_pilot_emb128x64_seedA_atlas24_eval_v001"
GENERATED_BUNDLE_DIR="${DERIVATIVE_BASE}/generated"
OUT_RUN_DIR="${DERIVATIVE_BASE}/run"

ORIGINAL_CONFIG_YML="${DEVELOPMENT_RUN_DIR}/config.yml"

METRICS=(NSE RMSE KGE Pearson-r Beta-KGE)

RESULT_DIR="${FLASHNH_BASE}/evidence/emb128x64_seedA_atlas24_eval_${SLURM_JOB_ID}"
mkdir -p "${RESULT_DIR}"
RESULT_LOG="${RESULT_DIR}/atlas24_eval_human.log"
exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== emb128x64_seedA epoch 6: 24-basin hydrograph-atlas evaluation-only derivative ==="
hostname
date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "DEVELOPMENT_RUN_DIR (original, must remain unmodified): ${DEVELOPMENT_RUN_DIR}"
echo "EPOCH (fixed, must stay 6): ${EPOCH}"
echo "GENERATED_BUNDLE_DIR: ${GENERATED_BUNDLE_DIR}"
echo "OUT_RUN_DIR: ${OUT_RUN_DIR}"

test -d "${DEVELOPMENT_RUN_DIR}" || { echo "MISSING development run dir: ${DEVELOPMENT_RUN_DIR}"; exit 1; }
test -f "${DEVELOPMENT_RUN_DIR}/model_epoch$(printf '%03d' "${EPOCH}").pt" || { echo "MISSING checkpoint for epoch ${EPOCH}"; exit 1; }
test -f "${ORIGINAL_CONFIG_YML}" || { echo "MISSING original config.yml: ${ORIGINAL_CONFIG_YML}"; exit 1; }
test -d "${TRANSFERRED_PACKAGE}" || { echo "MISSING transferred package: ${TRANSFERRED_PACKAGE}"; exit 1; }
test -f "${ATLAS_CSV}" || { echo "MISSING atlas selection CSV: ${ATLAS_CSV}"; exit 1; }

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

echo "=== capture pre-run sha256 of protected original training artifacts ==="
WEIGHT_STEM="model_epoch$(printf '%03d' "${EPOCH}")"
PRE_RUN_SHA256_JSON="${RESULT_DIR}/pre_run_original_sha256.json"
{
    echo "{"
    echo "  \"config.yml\": \"$(sha256sum "${ORIGINAL_CONFIG_YML}" | cut -d' ' -f1)\","
    echo "  \"${WEIGHT_STEM}.pt\": \"$(sha256sum "${DEVELOPMENT_RUN_DIR}/${WEIGHT_STEM}.pt" | cut -d' ' -f1)\","
    echo "  \"train_data/train_data_scaler.yml\": \"$(sha256sum "${DEVELOPMENT_RUN_DIR}/train_data/train_data_scaler.yml" | cut -d' ' -f1)\""
    echo "}"
} > "${PRE_RUN_SHA256_JSON}"
cat "${PRE_RUN_SHA256_JSON}"

echo "=== also snapshot the original ~400-basin screening validation results pickle (must survive unchanged) ==="
ORIGINAL_VALIDATION_PICKLE="${DEVELOPMENT_RUN_DIR}/validation/${WEIGHT_STEM}/validation_results.p"
ORIGINAL_VALIDATION_PICKLE_SHA256="MISSING"
if [ -f "${ORIGINAL_VALIDATION_PICKLE}" ]; then
    ORIGINAL_VALIDATION_PICKLE_SHA256="$(sha256sum "${ORIGINAL_VALIDATION_PICKLE}" | cut -d' ' -f1)"
fi
echo "original validation pickle sha256 (pre-run): ${ORIGINAL_VALIDATION_PICKLE_SHA256}"

echo "=== (1) build atlas24 derivative config bundle + evaluation-only run dir ==="
python scripts/build_stage1_atlas24_eval_run_dir.py \
    --development-run-dir "${DEVELOPMENT_RUN_DIR}" \
    --epoch "${EPOCH}" \
    --package-root "${TRANSFERRED_PACKAGE}" \
    --atlas-csv "${ATLAS_CSV}" \
    --generated-bundle-dir "${GENERATED_BUNDLE_DIR}" \
    --out-run-dir "${OUT_RUN_DIR}" \
    --force
BUILD_STATUS=$?
echo "build exit code: ${BUILD_STATUS}"

test -f "${OUT_RUN_DIR}/EVALUATION_ONLY_DO_NOT_TRAIN.txt" || { echo "MISSING training-prevention marker in ${OUT_RUN_DIR}"; exit 1; }

echo "=== config-diff check: derivative config.yml vs original frozen config.yml ==="
CONFIG_DIFF_JSON="${RESULT_DIR}/config_diff.json"
python -c "
import json
import sys
import yaml

original_path = '${ORIGINAL_CONFIG_YML}'
derivative_path = '${OUT_RUN_DIR}/config.yml'

with open(original_path) as fh:
    original = yaml.safe_load(fh)
with open(derivative_path) as fh:
    derivative = yaml.safe_load(fh)

expected_differing_keys = {
    'train_basin_file', 'validation_basin_file', 'test_basin_file',
    'run_dir', 'train_dir', 'img_log_dir', 'experiment_name', 'commit_hash',
}
# NH stamps these into config.yml itself as run bookkeeping (git commit, resolved
# subdirectory paths, basin-file line count, package version) only once a run
# actually executes. This derivative is evaluation-only by design -- it must
# never train (see EVALUATION_ONLY_DO_NOT_TRAIN.txt) -- so its pre-run generated
# config.yml legitimately never gains these keys. Their absence here is expected,
# not a sign the derivative diverges from the original in any way that matters.
allowed_missing_in_derivative = {
    'commit_hash', 'img_log_dir', 'number_of_basins', 'package_version', 'train_dir',
}

all_keys = sorted(set(original) | set(derivative))
unexpected_diffs = {}
expected_diffs = {}
missing_keys = sorted(set(original) - set(derivative) - allowed_missing_in_derivative)
extra_keys = sorted(set(derivative) - set(original))
for key in all_keys:
    if key not in original or key not in derivative:
        continue
    if original[key] != derivative[key]:
        if key in expected_differing_keys:
            expected_diffs[key] = {'original': original[key], 'derivative': derivative[key]}
        else:
            unexpected_diffs[key] = {'original': original[key], 'derivative': derivative[key]}

def read_basin_file(p):
    with open(p) as fh:
        return sorted(line.strip() for line in fh if line.strip())

train_content_match = read_basin_file(original['train_basin_file'].strip() if isinstance(original['train_basin_file'], str) else original['train_basin_file']) == \
    read_basin_file(derivative['train_basin_file'].strip() if isinstance(derivative['train_basin_file'], str) else derivative['train_basin_file'])
test_content_match = read_basin_file(original['test_basin_file'].strip() if isinstance(original['test_basin_file'], str) else original['test_basin_file']) == \
    read_basin_file(derivative['test_basin_file'].strip() if isinstance(derivative['test_basin_file'], str) else derivative['test_basin_file'])
validation_content = read_basin_file(derivative['validation_basin_file'].strip() if isinstance(derivative['validation_basin_file'], str) else derivative['validation_basin_file'])
validation_content_is_24 = len(validation_content) == 24

result = {
    'missing_keys_in_derivative': missing_keys,
    'allowed_missing_keys_in_derivative': sorted((set(original) - set(derivative)) & allowed_missing_in_derivative),
    'extra_keys_in_derivative': extra_keys,
    'expected_diffs': expected_diffs,
    'unexpected_diffs': unexpected_diffs,
    'train_basin_file_content_identical': train_content_match,
    'test_basin_file_content_identical': test_content_match,
    'validation_basin_file_content_is_exactly_24': validation_content_is_24,
    'validation_basin_file_content_count': len(validation_content),
}
with open('${CONFIG_DIFF_JSON}', 'w') as fh:
    json.dump(result, fh, indent=2, default=str)
print(json.dumps(result, indent=2, default=str))

ok = (
    not missing_keys and not extra_keys and not unexpected_diffs
    and train_content_match and test_content_match and validation_content_is_24
)
sys.exit(0 if ok else 1)
"
CONFIG_DIFF_STATUS=$?
echo "config-diff check exit code: ${CONFIG_DIFF_STATUS}"

echo "=== (2) evaluate the 24-basin atlas validation period ==="
EVAL_STATUS=1
if [ "${BUILD_STATUS}" -eq 0 ] && [ "${CONFIG_DIFF_STATUS}" -eq 0 ]; then
    python scripts/run_stage1_nh.py eval "${OUT_RUN_DIR}" --period validation --epoch "${EPOCH}" --metrics "${METRICS[@]}"
    EVAL_STATUS=$?
fi
echo "evaluation exit code: ${EVAL_STATUS}"

echo "=== verify original training artifacts were not modified ==="
POST_RUN_SHA256_JSON="${RESULT_DIR}/post_run_original_sha256.json"
{
    echo "{"
    echo "  \"config.yml\": \"$(sha256sum "${ORIGINAL_CONFIG_YML}" | cut -d' ' -f1)\","
    echo "  \"${WEIGHT_STEM}.pt\": \"$(sha256sum "${DEVELOPMENT_RUN_DIR}/${WEIGHT_STEM}.pt" | cut -d' ' -f1)\","
    echo "  \"train_data/train_data_scaler.yml\": \"$(sha256sum "${DEVELOPMENT_RUN_DIR}/train_data/train_data_scaler.yml" | cut -d' ' -f1)\""
    echo "}"
} > "${POST_RUN_SHA256_JSON}"
ARTIFACT_UNCHANGED_STATUS=0
diff "${PRE_RUN_SHA256_JSON}" "${POST_RUN_SHA256_JSON}" || ARTIFACT_UNCHANGED_STATUS=1

ORIGINAL_VALIDATION_PICKLE_POST_SHA256="MISSING"
if [ -f "${ORIGINAL_VALIDATION_PICKLE}" ]; then
    ORIGINAL_VALIDATION_PICKLE_POST_SHA256="$(sha256sum "${ORIGINAL_VALIDATION_PICKLE}" | cut -d' ' -f1)"
fi
ORIGINAL_VALIDATION_PICKLE_UNCHANGED=0
[ "${ORIGINAL_VALIDATION_PICKLE_SHA256}" = "${ORIGINAL_VALIDATION_PICKLE_POST_SHA256}" ] || ORIGINAL_VALIDATION_PICKLE_UNCHANGED=1
echo "original validation pickle sha256 (post-run): ${ORIGINAL_VALIDATION_PICKLE_POST_SHA256} (unchanged: $([ ${ORIGINAL_VALIDATION_PICKLE_UNCHANGED} -eq 0 ] && echo true || echo false))"

echo "=== verify the derivative result pickle: exactly the 24 atlas basins, saved under the derivative run dir (not the original) ==="
DERIVATIVE_RESULTS_PICKLE="${OUT_RUN_DIR}/validation/${WEIGHT_STEM}/validation_results.p"
PICKLE_CHECK_JSON="${RESULT_DIR}/derivative_pickle_check.json"
PICKLE_CHECK_STATUS=1
if [ "${EVAL_STATUS}" -eq 0 ]; then
    python -c "
import json
import pickle
import sys

atlas_basin_ids = sorted([
    '01074520', '02097314', '02455000', '03115400', '03165000', '03204500',
    '03302050', '03529500', '03592000', '03597590', '06204050', '06909500',
    '06910800', '06918740', '07218000', '07263295', '07362587', '07375800',
    '08271000', '09292000', '09408000', '12094000', '14169000', '14180300',
])

with open('${DERIVATIVE_RESULTS_PICKLE}', 'rb') as fh:
    results = pickle.load(fh)

actual_basin_ids = sorted(results.keys())
target_variable = 'qobs_mm_per_h_lead06'
obs_key = f'{target_variable}_obs'
sim_key = f'{target_variable}_sim'
missing_vars = []
for basin_id, freq_results in results.items():
    for freq, freq_result in freq_results.items():
        xr_ds = freq_result.get('xr')
        if xr_ds is None or obs_key not in xr_ds.data_vars or sim_key not in xr_ds.data_vars:
            missing_vars.append({'basin_id': basin_id, 'freq': freq})

result = {
    'n_basins_in_pickle': len(actual_basin_ids),
    'basin_ids_match_atlas24_exactly': actual_basin_ids == atlas_basin_ids,
    'actual_basin_ids': actual_basin_ids,
    'missing_obs_sim_vars': missing_vars,
    'pickle_path': '${DERIVATIVE_RESULTS_PICKLE}',
    'pickle_sha256': None,
}
with open('${PICKLE_CHECK_JSON}', 'w') as fh:
    json.dump(result, fh, indent=2, default=str)
print(json.dumps(result, indent=2, default=str))
ok = result['basin_ids_match_atlas24_exactly'] and not missing_vars
sys.exit(0 if ok else 1)
"
    PICKLE_CHECK_STATUS=$?
fi
echo "derivative pickle check exit code: ${PICKLE_CHECK_STATUS}"

if [ -f "${DERIVATIVE_RESULTS_PICKLE}" ]; then
    DERIVATIVE_RESULTS_PICKLE_SHA256="$(sha256sum "${DERIVATIVE_RESULTS_PICKLE}" | cut -d' ' -f1)"
else
    DERIVATIVE_RESULTS_PICKLE_SHA256="MISSING"
fi

OVERALL_STATUS=0
[ "${BUILD_STATUS}" -eq 0 ] || OVERALL_STATUS=1
[ "${CONFIG_DIFF_STATUS}" -eq 0 ] || OVERALL_STATUS=1
[ "${EVAL_STATUS}" -eq 0 ] || OVERALL_STATUS=1
[ "${ARTIFACT_UNCHANGED_STATUS}" -eq 0 ] || OVERALL_STATUS=1
[ "${ORIGINAL_VALIDATION_PICKLE_UNCHANGED}" -eq 0 ] || OVERALL_STATUS=1
[ "${PICKLE_CHECK_STATUS}" -eq 0 ] || OVERALL_STATUS=1

python -c "
import json
out = {
    'status': 'PASS' if ${OVERALL_STATUS} == 0 else 'FAIL',
    'development_run_dir': '${DEVELOPMENT_RUN_DIR}',
    'out_run_dir': '${OUT_RUN_DIR}',
    'epoch': ${EPOCH},
    'build_exit_code': ${BUILD_STATUS},
    'config_diff_exit_code': ${CONFIG_DIFF_STATUS},
    'evaluation_exit_code': ${EVAL_STATUS},
    'original_training_artifacts_unchanged': ${ARTIFACT_UNCHANGED_STATUS} == 0,
    'original_validation_pickle_unchanged': ${ORIGINAL_VALIDATION_PICKLE_UNCHANGED} == 0,
    'derivative_pickle_check_exit_code': ${PICKLE_CHECK_STATUS},
    'derivative_results_pickle': '${DERIVATIVE_RESULTS_PICKLE}',
    'derivative_results_pickle_sha256': '${DERIVATIVE_RESULTS_PICKLE_SHA256}',
    'config_diff_json': '${CONFIG_DIFF_JSON}',
    'pickle_check_json': '${PICKLE_CHECK_JSON}',
    'pre_run_original_sha256_json': '${PRE_RUN_SHA256_JSON}',
    'post_run_original_sha256_json': '${POST_RUN_SHA256_JSON}',
    'slurm_job_id': '${SLURM_JOB_ID}',
    'source_commit': '${SOURCE_COMMIT}',
    'hostname': __import__('socket').gethostname(),
}
json.dump(out, open('${RESULT_DIR}/atlas24_eval_result.json', 'w'), indent=2)
print(json.dumps(out, indent=2))
"

echo "=== done (exit ${OVERALL_STATUS}) ==="
exit ${OVERALL_STATUS}

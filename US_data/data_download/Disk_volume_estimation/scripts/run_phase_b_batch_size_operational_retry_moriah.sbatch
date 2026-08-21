#!/usr/bin/env bash
#SBATCH --job-name=flashnh-bs-opqual-a2
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/batch-size-opqual-attempt2-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/batch-size-opqual-attempt2-%j.err
#
# Attempt 2: OPERATIONAL QUALIFICATION ONLY -- NOT SCIENTIFIC PERFORMANCE
# EVIDENCE. This is one allocation containing three independent, fresh
# Python/CUDA processes. It does not submit nested Slurm jobs.
set -uo pipefail

readonly PARENT_ATTEMPT_ID="${OPQUAL_PARENT_ATTEMPT_ID:-attempt2_combined_sequential_operational_retry_v001}"
readonly BATCH_SIZES=(128 256 512)
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"
if [ -z "${EXPECTED_COMMIT}" ]; then
    echo "EXPECTED_COMMIT is required." >&2
    exit 2
fi

FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
REPO_CLONE_DIR="${FLASHNH_REPO_CLONE_DIR:-${FLASHNH_BASE}/repos/flash-nh}"
REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"
ENV_PREFIX="${FLASHNH_BASE}/envs/flashnh-moriah"
MORIAH_PYTHON="${FLASHNH_MORIAH_PYTHON:-${ENV_PREFIX}/bin/python}"
EVIDENCE_ROOT="${OPQUAL_COMBINED_EVIDENCE_ROOT:-${FLASHNH_BASE}/evidence/phase_b_batch_size_operational_qualification_attempt2_combined_v001}"
ALLOCATION_DIR="${EVIDENCE_ROOT}/allocation_job${SLURM_JOB_ID:-unknown}"
SINGLE_BATCH_LAUNCHER="${REPO_WORKDIR}/scripts/run_phase_b_batch_size_operational_qualification_moriah.sbatch"

mkdir -p "${ALLOCATION_DIR}"
exec > >(tee -a "${ALLOCATION_DIR}/launcher.log") 2>&1
echo "OPERATIONAL QUALIFICATION ONLY -- NOT SCIENTIFIC PERFORMANCE EVIDENCE"
echo "attempt_id=${PARENT_ATTEMPT_ID} batches=${BATCH_SIZES[*]}"
echo "job_id=${SLURM_JOB_ID:-unknown} partition=${SLURM_JOB_PARTITION:-unknown}"
SUBMIT_WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd -P)}"
echo "SUBMIT_WORKDIR: ${SUBMIT_WORKDIR}"
echo "CONFIGURED_REPO_WORKDIR: ${REPO_WORKDIR}"
echo "MORIAH_PYTHON: ${MORIAH_PYTHON}"

# Keep the repaired exact-commit guard authoritative at allocation preflight.
test -d "${REPO_WORKDIR}" || { echo "REFUSING: missing canonical repo workdir: ${REPO_WORKDIR}" >&2; exit 1; }
RESOLVED_REPOSITORY=$(git -C "${REPO_WORKDIR}" rev-parse --show-toplevel) || {
    echo "REFUSING: unable to resolve Git repository from ${REPO_WORKDIR}" >&2
    exit 1
}
ACTUAL_COMMIT=$(git -C "${RESOLVED_REPOSITORY}" rev-parse HEAD)
TRACKED_CHANGES=$(git -C "${RESOLVED_REPOSITORY}" status --porcelain | grep -v '^??' || true)
echo "RESOLVED_REPOSITORY: ${RESOLVED_REPOSITORY}"
echo "EXPECTED_COMMIT: ${EXPECTED_COMMIT}"
echo "ACTUAL_COMMIT: ${ACTUAL_COMMIT}"
test -z "${TRACKED_CHANGES}" || { echo "REFUSING: tracked working-tree changes" >&2; exit 1; }
test "${ACTUAL_COMMIT}" = "${EXPECTED_COMMIT}" || {
    echo "REFUSING: ACTUAL_COMMIT does not match EXPECTED_COMMIT" >&2
    exit 1
}

cd "${REPO_WORKDIR}"
test -x "${MORIAH_PYTHON}" || { echo "REFUSING: MORIAH_PYTHON is not executable: ${MORIAH_PYTHON}" >&2; exit 1; }
"${MORIAH_PYTHON}" --version
"${MORIAH_PYTHON}" - <<'PY'
import yaml, torch, neuralhydrology
print("MORIAH_PYTHON_IMPORT_PREFLIGHT: yaml torch neuralhydrology PASS")
PY

# The child launcher is invoked with bash, never sbatch. Each invocation is a
# fresh shell and starts its own fresh Python process for the real NH smoke.
export FLASHNH_REPO_CLONE_DIR="${REPO_CLONE_DIR}"
export OPQUAL_EVIDENCE_ROOT="${EVIDENCE_ROOT}"
export FLASHNH_MORIAH_PYTHON="${MORIAH_PYTHON}"
export OPQUAL_ATTEMPT_ID="${PARENT_ATTEMPT_ID}"
declare -A BATCH_EXIT_STATUS
for batch_size in "${BATCH_SIZES[@]}"; do
    echo "BEGIN_BATCH batch_size=${batch_size} attempt_id=${PARENT_ATTEMPT_ID}"
    set +e
    bash "${SINGLE_BATCH_LAUNCHER}" "${batch_size}"
    BATCH_EXIT_STATUS["${batch_size}"]=$?
    set -e
    echo "END_BATCH batch_size=${batch_size} exit_status=${BATCH_EXIT_STATUS[${batch_size}]}"
done

RESULT_DIR="${ALLOCATION_DIR}" ATTEMPT_ID="${PARENT_ATTEMPT_ID}" EVIDENCE_ROOT="${EVIDENCE_ROOT}" \
EXPECTED_COMMIT="${EXPECTED_COMMIT}" BATCH_EXIT_128="${BATCH_EXIT_STATUS[128]}" \
BATCH_EXIT_256="${BATCH_EXIT_STATUS[256]}" BATCH_EXIT_512="${BATCH_EXIT_STATUS[512]}" "${MORIAH_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

batch_sizes = (128, 256, 512)
evidence_root = Path(os.environ['EVIDENCE_ROOT'])
records = []
all_passed = True
for batch_size in batch_sizes:
    result_path = evidence_root / f"bs{batch_size}_job{os.environ.get('SLURM_JOB_ID', 'unknown')}" / 'qualification_result.json'
    process_exit_status = int(os.environ[f'BATCH_EXIT_{batch_size}'])
    record = {
        'batch_size': batch_size,
        'attempt_id': os.environ['ATTEMPT_ID'],
        'subprocess_exit_status': process_exit_status,
        'result_path': str(result_path),
    }
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding='utf-8'))
        preparation_path = result_path.parent / 'prepared' / 'qualification_preparation.json'
        record.update({
            'workflow_status': result.get('status'),
            'source_commit': result.get('source_commit'),
            'result_attempt_id': result.get('attempt_id'),
            'hidden_size': result.get('hidden_size'),
            'operational_updates': result.get('operational_updates'),
            'forward_backward_optimizer_completed': result.get('forward_backward_optimizer_completed'),
            'device': result.get('device'),
            'elapsed_seconds': result.get('elapsed_seconds'),
            'peak_gpu_memory_bytes': result.get('peak_gpu_memory_bytes'),
            'error_excerpt': result.get('error_excerpt'),
        })
        if preparation_path.is_file():
            preparation = json.loads(preparation_path.read_text(encoding='utf-8'))
            record.update({
                'intended_sweep_v1_updates_per_epoch': preparation.get('intended_sweep_v1_updates_per_epoch'),
                'intended_config_sha256': preparation.get('config_sha256', {}).get('intended'),
                'operational_config_sha256': preparation.get('config_sha256', {}).get('operational_smoke'),
                'preparation_path': str(preparation_path),
            })
    else:
        record['workflow_status'] = 'NO_QUALIFICATION_RESULT'
        record['failure_excerpt'] = 'child launcher exited before qualification_result.json was created'
    if process_exit_status != 0 or record['workflow_status'] != 'PASS':
        all_passed = False
    records.append(record)

summary = {
    'schema_name': 'phase_b_batch_size_operational_qualification_attempt2_summary',
    'schema_version': 1,
    'classification': 'OPERATIONAL QUALIFICATION ONLY -- NOT SCIENTIFIC PERFORMANCE EVIDENCE',
    'attempt_id': os.environ['ATTEMPT_ID'],
    'source_commit_expected': os.environ['EXPECTED_COMMIT'],
    'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
    'batch_execution_order': list(batch_sizes),
    'overall_status': 'PASS' if all_passed else 'PARTIAL_FAILURE_OR_FAIL',
    'batches': records,
}
path = Path(os.environ['RESULT_DIR']) / 'attempt2_combined_summary.json'
path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
raise SystemExit(0 if all_passed else 1)
PY

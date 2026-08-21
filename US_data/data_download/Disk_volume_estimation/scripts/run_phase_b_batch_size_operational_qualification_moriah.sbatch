#!/usr/bin/env bash
#SBATCH --job-name=flashnh-bs-opqual
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/batch-size-opqual-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/batch-size-opqual-%j.err
#
# Phase-B batch-size OPERATIONAL QUALIFICATION ONLY.  This launcher runs one
# eight-update H256 smoke for exactly one of {128,256,512}; it does not run
# Sweep-v1, scientific validation, W&B, or a performance comparison.
set -uo pipefail

BATCH_SIZE="${BATCH_SIZE:-${1:-}}"
case "${BATCH_SIZE}" in 128|256|512) ;; *) echo "USAGE: sbatch $0 <128|256|512>" >&2; exit 2;; esac
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"
if [ -z "${EXPECTED_COMMIT}" ]; then echo "EXPECTED_COMMIT is required." >&2; exit 2; fi

FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
# Resolve the reviewed clone deterministically from the canonical Flash-NH
# runtime base.  Do not inherit PILOT_REPO_CLONE_DIR: that historical default
# names a separate, stale clone and caused Attempt 1's preflight refusal.
REPO_CLONE_DIR="${FLASHNH_REPO_CLONE_DIR:-${FLASHNH_BASE}/repos/flash-nh}"
REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"
ENV_PREFIX="${FLASHNH_BASE}/envs/flashnh-moriah"
PACKAGE_ROOT="${PILOT_PACKAGE_ROOT:-${FLASHNH_BASE}/data/stage1_scientific_package_v002}"
EVIDENCE_ROOT="${OPQUAL_EVIDENCE_ROOT:-${FLASHNH_BASE}/evidence/phase_b_batch_size_operational_qualification_only}"
RESULT_DIR="${EVIDENCE_ROOT}/bs${BATCH_SIZE}_job${SLURM_JOB_ID}"

mkdir -p "${RESULT_DIR}"
exec > >(tee -a "${RESULT_DIR}/launcher.log") 2>&1
echo "OPERATIONAL QUALIFICATION ONLY — NOT SCIENTIFIC PERFORMANCE EVIDENCE"
echo "batch_size=${BATCH_SIZE} hidden_size=256 operational_updates=8"
echo "job_id=${SLURM_JOB_ID:-unknown} partition=${SLURM_JOB_PARTITION:-unknown}"
SUBMIT_WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd -P)}"
echo "SUBMIT_WORKDIR: ${SUBMIT_WORKDIR}"
echo "CONFIGURED_REPO_WORKDIR: ${REPO_WORKDIR}"

test -d "${REPO_WORKDIR}" || { echo "missing canonical repo workdir: ${REPO_WORKDIR}" >&2; exit 1; }
test -d "${PACKAGE_ROOT}" || { echo "missing package root" >&2; exit 1; }
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
SOURCE_COMMIT="${ACTUAL_COMMIT}"

module load spack/all
module load miniconda3/24.3.0-gcc-iqeknet
module load nvidia/580.95.05
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
nvidia-smi | tee "${RESULT_DIR}/nvidia_smi_before.txt"

python -u scripts/prepare_phase_b_batch_size_operational_qualification.py \
  --batch-size "${BATCH_SIZE}" --package-root "${PACKAGE_ROOT}" \
  --out-dir "${RESULT_DIR}/prepared"

set +e
RESULT_DIR="${RESULT_DIR}" BATCH_SIZE="${BATCH_SIZE}" SOURCE_COMMIT="${SOURCE_COMMIT}" python -u - <<'PY'
import json, os, time, traceback
from pathlib import Path
import torch
from neuralhydrology.nh_run import start_run
from src.baseline.nh_register import register_flashnh_dataset

root = Path(os.environ['RESULT_DIR'])
config = root / 'prepared' / 'operational_smoke_config' / 'config.yaml'
result = {'classification': 'OPERATIONAL QUALIFICATION ONLY — NOT SCIENTIFIC PERFORMANCE EVIDENCE',
          'batch_size': int(os.environ['BATCH_SIZE']), 'hidden_size': 256,
          'operational_updates': 8, 'source_commit': os.environ['SOURCE_COMMIT'],
          'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'device': torch.cuda.get_device_name(0),
          'cuda_device_count': torch.cuda.device_count(), 'forward_backward_optimizer_completed': False}
started = time.monotonic()
try:
    torch.cuda.reset_peak_memory_stats(0)
    register_flashnh_dataset()
    start_run(config_file=config)
    result['forward_backward_optimizer_completed'] = True
    result['status'] = 'PASS'
except Exception as exc:
    result['status'] = 'FAIL'
    result['error_type'] = type(exc).__name__
    result['error_excerpt'] = ''.join(traceback.format_exception_only(type(exc), exc)).strip()[:2000]
finally:
    result['elapsed_seconds'] = time.monotonic() - started
    result['peak_gpu_memory_bytes'] = int(torch.cuda.max_memory_allocated(0))
    (root / 'qualification_result.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
raise SystemExit(0 if result['status'] == 'PASS' else 1)
PY
PYTHON_STATUS=$?
set -e
RESULT_DIR="${RESULT_DIR}" PYTHON_STATUS="${PYTHON_STATUS}" python - <<'PY'
import json, os
from pathlib import Path
path = Path(os.environ['RESULT_DIR']) / 'qualification_result.json'
if path.is_file():
    result = json.loads(path.read_text())
    result['launcher_python_exit_status'] = int(os.environ['PYTHON_STATUS'])
    path.write_text(json.dumps(result, indent=2))
PY
nvidia-smi | tee "${RESULT_DIR}/nvidia_smi_after.txt"
exit "${PYTHON_STATUS}"

#!/usr/bin/env bash
#SBATCH --job-name=flashnh-bs-opqual-cpu
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/batch-size-opqual-cpu-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/batch-size-opqual-cpu-%j.err
# Attempt 3 CPU preflight only: no CUDA requirement, start_run, or training.
set -euo pipefail

readonly PREFLIGHT_ATTEMPT_ID="attempt3_environment_qualified_combined_operational_retry_v001"
readonly BATCH_SIZES=(128 256 512)
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"
[ -n "${EXPECTED_COMMIT}" ] || { echo "EXPECTED_COMMIT is required." >&2; exit 2; }
FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
REPO_CLONE_DIR="${FLASHNH_REPO_CLONE_DIR:-${FLASHNH_BASE}/repos/flash-nh}"
REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"
MORIAH_PYTHON="${FLASHNH_MORIAH_PYTHON:-${FLASHNH_BASE}/envs/flashnh-moriah/bin/python}"
PACKAGE_ROOT="${PILOT_PACKAGE_ROOT:-${FLASHNH_BASE}/data/stage1_scientific_package_v002}"
EVIDENCE_ROOT="${OPQUAL_ATTEMPT3_CPU_PREFLIGHT_EVIDENCE_ROOT:-${FLASHNH_BASE}/evidence/phase_b_batch_size_operational_qualification_attempt3_cpu_preflight_v001}"
RESULT_DIR="${EVIDENCE_ROOT}/job${SLURM_JOB_ID:-unknown}"
mkdir -p "${RESULT_DIR}"
exec > >(tee -a "${RESULT_DIR}/launcher.log") 2>&1
echo "CPU PREFLIGHT ONLY -- NOT TRAINING OR BATCH-SIZE QUALIFICATION"
echo "attempt_id=${PREFLIGHT_ATTEMPT_ID} python=${MORIAH_PYTHON}"

test -d "${REPO_WORKDIR}" || { echo "REFUSING: missing canonical repo workdir" >&2; exit 1; }
RESOLVED_REPOSITORY=$(git -C "${REPO_WORKDIR}" rev-parse --show-toplevel)
ACTUAL_COMMIT=$(git -C "${RESOLVED_REPOSITORY}" rev-parse HEAD)
echo "RESOLVED_REPOSITORY: ${RESOLVED_REPOSITORY}"
echo "EXPECTED_COMMIT: ${EXPECTED_COMMIT}"
echo "ACTUAL_COMMIT: ${ACTUAL_COMMIT}"
test "${ACTUAL_COMMIT}" = "${EXPECTED_COMMIT}" || { echo "REFUSING: ACTUAL_COMMIT does not match EXPECTED_COMMIT" >&2; exit 1; }
test -x "${MORIAH_PYTHON}" || { echo "REFUSING: MORIAH_PYTHON is not executable: ${MORIAH_PYTHON}" >&2; exit 1; }
"${MORIAH_PYTHON}" --version
"${MORIAH_PYTHON}" - <<'PY'
import yaml, torch, neuralhydrology
print("MORIAH_PYTHON_IMPORT_PREFLIGHT: yaml torch neuralhydrology PASS")
PY

cd "${REPO_WORKDIR}"
declare -A PREP_EXIT
for batch_size in "${BATCH_SIZES[@]}"; do
    set +e
    "${MORIAH_PYTHON}" -u scripts/prepare_phase_b_batch_size_operational_qualification.py --batch-size "${batch_size}" --package-root "${PACKAGE_ROOT}" --out-dir "${RESULT_DIR}/bs${batch_size}/prepared"
    PREP_EXIT["${batch_size}"]=$?
    set -e
done
RESULT_DIR="${RESULT_DIR}" ATTEMPT_ID="${PREFLIGHT_ATTEMPT_ID}" B128="${PREP_EXIT[128]}" B256="${PREP_EXIT[256]}" B512="${PREP_EXIT[512]}" "${MORIAH_PYTHON}" - <<'PY'
import json, os
from pathlib import Path
import yaml
root = Path(os.environ['RESULT_DIR'])
records, passed = [], True
for batch in (128, 256, 512):
    prep = root / f'bs{batch}' / 'prepared' / 'qualification_preparation.json'
    record = {'batch_size': batch, 'preparation_exit_status': int(os.environ[f'B{batch}']), 'preparation_path': str(prep)}
    if prep.is_file():
        data = json.loads(prep.read_text())
        intended = yaml.safe_load(Path(data['intended_config']['config.yaml']).read_text())
        smoke = yaml.safe_load(Path(data['operational_smoke_config']['config.yaml']).read_text())
        record.update({'intended_updates': intended['max_updates_per_epoch'], 'operational_updates': smoke['max_updates_per_epoch'], 'checksums': data['config_sha256']})
    if record['preparation_exit_status'] != 0 or record.get('intended_updates') != 50000 or record.get('operational_updates') != 8:
        passed = False
    records.append(record)
summary = {'classification': 'CPU PREFLIGHT ONLY -- NOT TRAINING OR BATCH-SIZE QUALIFICATION', 'attempt_id': os.environ['ATTEMPT_ID'], 'batch_order': [128,256,512], 'status': 'PASS' if passed else 'FAIL', 'batches': records}
(root / 'attempt3_cpu_preflight_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
raise SystemExit(0 if passed else 1)
PY

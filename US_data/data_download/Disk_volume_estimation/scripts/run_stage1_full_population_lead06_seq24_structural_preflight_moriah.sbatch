#!/usr/bin/env bash
#SBATCH --job-name=flashnh-fullpop-lead06-seq24-structural-preflight
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/fullpop-lead06-seq24-structural-preflight-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/fullpop-lead06-seq24-structural-preflight-%j.err
#
# Full-population Moriah launch-readiness gate -- Part 5 (file-only structural
# preflight, no real FlashNHDataset construction). Generates exactly one real
# (lead=6h, seq_length=24) full-population NH config bundle pair
# (development/ + spatial_holdout/) against the Moriah-transferred certified
# stage1_scientific_package_v002 package, via the committed, unmodified
# scripts/generate_stage1_full_population_nh_config.py, then runs
# scripts/check_stage1_full_population_nh_config_preflight.py WITH
# --skip-dataset-construction (Layer 1 structural checks only). This is
# orchestration only -- no new check logic is added here.
#
# Runs on `glacier` (CPU, no GRES, confirmed via `sinfo`) since this job never
# touches CUDA/torch.cuda. Must PASS with 0 errors before the real
# dataset-construction preflight (Part 6,
# run_stage1_full_population_lead06_seq24_dataset_construction_moriah.sbatch)
# is submitted.

set -uo pipefail

FLASHNH_BASE="/sci/labs/efratmorin/omripo/Flash-NH"
ENV_PREFIX="${FLASHNH_BASE}/envs/flashnh-moriah"
REPO_CLONE_DIR="${FLASHNH_BASE}/repos/flash-nh"
REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"

TRANSFERRED_PACKAGE="${FLASHNH_BASE}/data/stage1_scientific_package_v002"
GENERATED_OUT_DIR="${FLASHNH_BASE}/runs/stage1_full_population_nh_config_lead06_seq24_v001"
RESULT_DIR="${FLASHNH_BASE}/evidence/fullpop_lead06_seq24_structural_preflight_${SLURM_JOB_ID}"

mkdir -p "${RESULT_DIR}"
RESULT_JSON="${RESULT_DIR}/structural_preflight_result.json"
RESULT_LOG="${RESULT_DIR}/structural_preflight_human.log"

exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== part 5: file-only structural preflight ==="
hostname
date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "partition: ${SLURM_JOB_PARTITION:-unknown}"

if ! command -v module &>/dev/null; then
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
python --version

echo "=== verify transferred inputs ==="
FAIL=0
test -d "${REPO_WORKDIR}" && echo "repo workdir OK: ${REPO_WORKDIR}" || { echo "MISSING repo workdir: ${REPO_WORKDIR}"; FAIL=1; }
test -d "${TRANSFERRED_PACKAGE}" && echo "transferred package OK: ${TRANSFERRED_PACKAGE}" || { echo "MISSING transferred package: ${TRANSFERRED_PACKAGE}"; FAIL=1; }
if [ "${FAIL}" -ne 0 ]; then
    echo '{"status": "FAIL", "reason": "missing transferred inputs"}' > "${RESULT_JSON}"
    exit 1
fi

cd "${REPO_WORKDIR}"

echo "=== git state ==="
git log --oneline -3
git status --short

echo "=== generate development + spatial_holdout config bundles (lead=6h, seq_length=24) ==="
python scripts/generate_stage1_full_population_nh_config.py \
    --package-root "${TRANSFERRED_PACKAGE}" \
    --lead-hours 6 \
    --seq-length 24 \
    --out-dir "${GENERATED_OUT_DIR}" \
    --force
GEN_STATUS=$?
if [ "${GEN_STATUS}" -ne 0 ]; then
    echo '{"status": "FAIL", "reason": "config generation failed"}' > "${RESULT_JSON}"
    exit 1
fi

echo "=== verify TEST_ONLY_DO_NOT_TRAIN.txt marker present only in spatial_holdout bundle ==="
MARKER_FAIL=0
if [ -f "${GENERATED_OUT_DIR}/development/TEST_ONLY_DO_NOT_TRAIN.txt" ]; then
    echo "UNEXPECTED: marker present in development/"
    MARKER_FAIL=1
else
    echo "OK: no marker in development/"
fi
if [ -f "${GENERATED_OUT_DIR}/spatial_holdout/TEST_ONLY_DO_NOT_TRAIN.txt" ]; then
    echo "OK: marker present in spatial_holdout/"
else
    echo "MISSING: marker not found in spatial_holdout/"
    MARKER_FAIL=1
fi
if [ "${MARKER_FAIL}" -ne 0 ]; then
    echo '{"status": "FAIL", "reason": "TEST_ONLY_DO_NOT_TRAIN.txt marker check failed"}' > "${RESULT_JSON}"
    exit 1
fi

echo "=== run file-only structural preflight (Layer 1 only, --skip-dataset-construction) ==="
python scripts/check_stage1_full_population_nh_config_preflight.py \
    --development-generated-dir "${GENERATED_OUT_DIR}/development" \
    --spatial-holdout-generated-dir "${GENERATED_OUT_DIR}/spatial_holdout" \
    --package-root "${TRANSFERRED_PACKAGE}" \
    --expected-seq-length 24 \
    --expected-target-variable qobs_mm_per_h_lead06 \
    --expected-dynamic-inputs mrms_qpe_1h_mm rtma_2t_K rtma_2d_K rtma_2sh_kgkg rtma_10u_ms rtma_10v_ms mrms_qpe_1h_mm_gap rtma_gap \
    --expected-static-column-count 473 \
    --expected-predict-last-n 1 \
    --expected-train-start-date 14/10/2020 \
    --expected-train-end-date 31/12/2023 \
    --expected-validation-start-date 01/01/2024 \
    --expected-validation-end-date 31/12/2024 \
    --expected-test-start-date 01/01/2025 \
    --expected-test-end-date 31/12/2025 \
    --repo-root "${REPO_WORKDIR}" \
    --skip-dataset-construction \
    > "${RESULT_DIR}/structural_preflight_stdout.json" 2> "${RESULT_DIR}/structural_preflight_stderr.log"
PREFLIGHT_STATUS=$?

cat "${RESULT_DIR}/structural_preflight_stdout.json"
echo "=== preflight exit code: ${PREFLIGHT_STATUS} ==="

python -u -c "
import json
stdout_path = '${RESULT_DIR}/structural_preflight_stdout.json'
try:
    with open(stdout_path, 'r', encoding='utf-8') as fh:
        text = fh.read()
    d, _ = json.JSONDecoder().raw_decode(text)
    status = d.get('status', 'FAIL')
    out = {
        'status': status,
        'error_count': d.get('error_count'),
        'warning_count': d.get('warning_count'),
        'ok_count': d.get('ok_count'),
        'generated_out_dir': '${GENERATED_OUT_DIR}',
        'slurm_job_id': '${SLURM_JOB_ID}',
        'hostname': __import__('socket').gethostname(),
    }
    exit_ok = (status == 'PASS')
except Exception as exc:
    out = {
        'status': 'FAIL',
        'reason': f'could not parse structural preflight output: {exc}',
        'generated_out_dir': '${GENERATED_OUT_DIR}',
        'slurm_job_id': '${SLURM_JOB_ID}',
        'hostname': __import__('socket').gethostname(),
    }
    exit_ok = False

import sys
json.dump(out, open('${RESULT_JSON}', 'w'), indent=2)
print(json.dumps(out, indent=2))
sys.exit(0 if exit_ok else 1)
"
FINAL_STATUS=$?

echo "=== done (exit ${FINAL_STATUS}) ==="
exit ${FINAL_STATUS}

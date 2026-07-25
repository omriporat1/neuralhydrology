#!/usr/bin/env bash
#SBATCH --job-name=flashnh-fullpop-lead06-seq24-dataset-construction
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/fullpop-lead06-seq24-dataset-construction-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/fullpop-lead06-seq24-dataset-construction-%j.err
#
# Full-population Moriah launch-readiness gate -- Part 6 (real FlashNHDataset
# construction preflight, CPU only, no training). Repeats the Part 5
# structural checks, then constructs the real development train/validation/
# temporal-test FlashNHDataset (scaler fit only on development-train,
# scaler={} explicit) and the spatial-holdout test dataset reusing that exact
# external scaler unchanged. Reuses the existing, unmodified
# scripts/check_stage1_full_population_nh_config_preflight.py and
# src/baseline/nh_structural_preflight.py's
# run_full_population_structural_preflight -- no new scientific-check logic
# is added here, this script is orchestration only.
#
# Resource sizing: the full development population (2,307 basins) is ~72x the
# 32-basin compact-package smoke that needed 8G/30min (job 45624926, 851,339 /
# 274,347 / 263,637 admitted train/validation/test samples via a plain
# per-sample Python loop in check_flashnh_dataset_construction -- not
# vectorized/batched). Sized generously (128G, 48h, glacier nodes confirmed at
# 128 CPU / ~1TB RAM each) since the per-sample finiteness-check loop is
# single-threaded and CPU count will not materially shorten it; this is a
# real, previously-untested runtime regime and may itself surface a
# runtime/memory-impracticality stop condition rather than a code defect.
#
# Runs on `glacier` (CPU, no GRES) -- never touches CUDA/torch.cuda.

set -uo pipefail

FLASHNH_BASE="/sci/labs/efratmorin/omripo/Flash-NH"
ENV_PREFIX="${FLASHNH_BASE}/envs/flashnh-moriah"
REPO_CLONE_DIR="${FLASHNH_BASE}/repos/flash-nh"
REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"

TRANSFERRED_PACKAGE="${FLASHNH_BASE}/data/stage1_scientific_package_v002"
GENERATED_OUT_DIR="${FLASHNH_BASE}/runs/stage1_full_population_nh_config_lead06_seq24_v001"
RESULT_DIR="${FLASHNH_BASE}/evidence/fullpop_lead06_seq24_dataset_construction_${SLURM_JOB_ID}"

mkdir -p "${RESULT_DIR}"
RESULT_JSON="${RESULT_DIR}/dataset_construction_result.json"
RESULT_LOG="${RESULT_DIR}/dataset_construction_human.log"

exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== part 6: real FlashNHDataset construction preflight (CPU) ==="
hostname
date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "partition: ${SLURM_JOB_PARTITION:-unknown}"
nproc

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

echo "=== environment info ==="
python -u -c "
import sys
print('Python:', sys.version.split()[0])
import torch
print('PyTorch:', torch.__version__)
print('cuda available (expected False on this CPU-only partition):', torch.cuda.is_available())
import neuralhydrology
print('neuralhydrology import OK (no __version__ attribute is expected on 1.13)')
"

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

echo "=== regenerate development + spatial_holdout config bundles (lead=6h, seq_length=24) ==="
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

echo "=== run full structural + real dataset-construction preflight (Layer 1 + Layer 2) ==="
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
        'checks': d.get('checks'),
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

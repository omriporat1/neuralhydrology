#!/usr/bin/env bash
#SBATCH --job-name=flashnh-nh-lead06-seq24-cpu-preflight
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/nh-lead06-seq24-cpu-preflight-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/nh-lead06-seq24-cpu-preflight-%j.err
#
# First real compact NH smoke run -- Job 1 of 2 (CPU-only construction preflight,
# no training). Runs on the `glacier` partition: confirmed CPU partition, no GRES,
# default partition (docs/stage1_neuralhydrology_preflight.md Sec 10.6 sinfo table).
# `catfish`/`gpu:l4:1` (the only partition/GRES ever used by run_stage1_smoke0/1_moriah.sbatch
# and setup_flashnh_moriah_env.sbatch) is a GPU-only convention and intentionally NOT reused
# here -- this job never touches CUDA/torch.cuda, only real FlashNHDataset construction.
#
# Regenerates one lead06/seq24 NH config bundle against the Moriah-transferred certified
# package (the h2o-generated bundle from Deliverable 2 has h2o-absolute paths baked into
# config.yaml -- data_dir/train_basin_file/etc. -- and is not usable as-is on Moriah), then
# runs the existing, unmodified scripts/check_stage1_nh_config_preflight.py WITHOUT
# --skip-dataset-construction, so Layer 2 (real FlashNHDataset train/validation/test
# construction, fresh scaler={} for train + reused for validation/test, finite-value checks,
# filter stats, single-target check) executes for real. No new check logic is added here;
# this script is orchestration only, per "do not refactor the config generator or preflight."
#
# Precondition: the certified compact package + this repo (checked out at the commit
# documented in the transfer marker) must already be transferred/cloned onto Moriah
# (Deliverable 3 / repo clone step). This job does not train -- see the paired GPU-smoke
# script (run_stage1_nh_lead06_seq24_gpu_smoke_moriah.sbatch) for that.

set -uo pipefail

FLASHNH_BASE="/sci/labs/efratmorin/omripo/Flash-NH"
ENV_PREFIX="${FLASHNH_BASE}/envs/flashnh-moriah"
# Repo clone location: not yet confirmed as executed on Moriah (no prior job ever needed
# repo source there -- Smoke 0/1 only used the pip-installed `nh-run` entry point). Chosen
# for consistency with the one CONFIRMED clone-of-this-repo layout (h2o,
# docs/stage1_h2o_operations_preflight.md: /data42/omrip/Flash-NH/repos/flash-nh/US_data/...).
# If you already cloned this repo elsewhere on Moriah, edit REPO_WORKDIR below instead of
# re-cloning.
REPO_CLONE_DIR="${FLASHNH_BASE}/repos/flash-nh"
REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"

TRANSFERRED_PACKAGE="${FLASHNH_BASE}/data/stage1_compact_scientific_package_v001"
GENERATED_OUT_DIR="${FLASHNH_BASE}/runs/stage1_nh_config_lead06_seq24_v001"
RESULT_DIR="${FLASHNH_BASE}/evidence/nh_lead06_seq24_cpu_preflight_${SLURM_JOB_ID}"

mkdir -p "${RESULT_DIR}"
RESULT_JSON="${RESULT_DIR}/cpu_preflight_result.json"
RESULT_LOG="${RESULT_DIR}/cpu_preflight_human.log"

exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== preflight ==="
hostname
date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "partition: ${SLURM_JOB_PARTITION:-unknown}"

# Initialize module system for non-interactive Slurm shells (same fix applied in
# setup_flashnh_moriah_env.sbatch / run_stage1_smoke0_moriah.sbatch).
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
# No nvidia/cuda modules: this job is CPU-only and never calls torch.cuda.
# (nh_dataset.py / nh_structural_preflight.py never read cfg.device.)

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

echo "=== regenerate lead06/seq24 config against the Moriah-local package copy ==="
python scripts/generate_stage1_nh_config.py \
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

CONFIG_YAML="${GENERATED_OUT_DIR}/config.yaml"

echo "=== run structural preflight (Layer 1 + Layer 2 real FlashNHDataset construction) ==="
python scripts/check_stage1_nh_config_preflight.py \
    --generated-dir "${GENERATED_OUT_DIR}" \
    --package-root "${TRANSFERRED_PACKAGE}" \
    --expected-basin-count 32 \
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
import json, sys
d = json.load(open('${RESULT_DIR}/structural_preflight_stdout.json'))
out = {
    'status': d['status'],
    'error_count': d['error_count'],
    'warning_count': d['warning_count'],
    'ok_count': d['ok_count'],
    'config_yaml': '${CONFIG_YAML}',
    'generated_out_dir': '${GENERATED_OUT_DIR}',
    'slurm_job_id': '${SLURM_JOB_ID}',
    'hostname': __import__('socket').gethostname(),
}
json.dump(out, open('${RESULT_JSON}', 'w'), indent=2)
print(json.dumps(out, indent=2))
sys.exit(0 if d['status'] == 'PASS' else 1)
"
FINAL_STATUS=$?

echo "=== done (exit ${FINAL_STATUS}) ==="
exit ${FINAL_STATUS}

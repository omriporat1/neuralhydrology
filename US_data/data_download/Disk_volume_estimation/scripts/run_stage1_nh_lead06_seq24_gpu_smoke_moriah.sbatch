#!/usr/bin/env bash
#SBATCH --job-name=flashnh-nh-lead06-seq24-gpu-smoke
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/nh-lead06-seq24-gpu-smoke-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/nh-lead06-seq24-gpu-smoke-%j.err
#
# First real compact NH smoke run -- Job 2 of 2 (GPU training smoke, 2-epoch compact
# profile only). Partition/GRES reused verbatim from the only confirmed GPU convention
# in this repo (run_stage1_smoke0/1_moriah.sbatch, setup_flashnh_moriah_env.sbatch):
# --partition=catfish --gres=gpu:l4:1.
#
# MUST NOT be submitted before the paired CPU-preflight job
# (run_stage1_nh_lead06_seq24_cpu_preflight_moriah.sbatch) has been reviewed and its
# result JSON shows status == PASS. This script re-checks that JSON defensively before
# training, but the real gate is Omri reviewing the CPU-preflight evidence -- this is not
# a substitute for that review.
#
# Uses scripts/run_stage1_nh.py (not bare `nh-run`) because the generated config sets
# dataset: flashnh, which only run_stage1_nh.py registers before delegating to
# neuralhydrology.nh_run.start_run. Trains the frozen 2-epoch compact smoke profile only
# (hidden_size 64, batch_size 64, Adam lr 1e-3, MSE, validate_every 1,
# validate_n_random_basins 32) -- no other lead/seq_length combination, no W&B.

set -uo pipefail

FLASHNH_BASE="/sci/labs/efratmorin/omripo/Flash-NH"
ENV_PREFIX="${FLASHNH_BASE}/envs/flashnh-moriah"
REPO_CLONE_DIR="${FLASHNH_BASE}/repos/flash-nh"
REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"

GENERATED_OUT_DIR="${FLASHNH_BASE}/runs/stage1_nh_config_lead06_seq24_v001"
CONFIG_YAML="${GENERATED_OUT_DIR}/config.yaml"

# Must match the *latest* CPU-preflight job's evidence dir (edit CPU_PREFLIGHT_JOB_ID to
# the job ID reviewed as PASS before submitting this job).
CPU_PREFLIGHT_JOB_ID="${CPU_PREFLIGHT_JOB_ID:?set CPU_PREFLIGHT_JOB_ID to the reviewed-PASS CPU-preflight Slurm job ID before submitting, e.g. --export=CPU_PREFLIGHT_JOB_ID=12345}"
CPU_PREFLIGHT_RESULT="${FLASHNH_BASE}/evidence/nh_lead06_seq24_cpu_preflight_${CPU_PREFLIGHT_JOB_ID}/cpu_preflight_result.json"

RESULT_DIR="${FLASHNH_BASE}/evidence/nh_lead06_seq24_gpu_smoke_${SLURM_JOB_ID}"
mkdir -p "${RESULT_DIR}"
RESULT_LOG="${RESULT_DIR}/gpu_smoke_human.log"
exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== preflight ==="
hostname
date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

echo "=== gate: CPU preflight must be PASS ==="
test -f "${CPU_PREFLIGHT_RESULT}" || { echo "MISSING CPU preflight result: ${CPU_PREFLIGHT_RESULT}"; exit 1; }
"${ENV_PREFIX}/bin/python" -c "
import json, sys
d = json.load(open('${CPU_PREFLIGHT_RESULT}'))
print('CPU preflight status:', d.get('status'))
sys.exit(0 if d.get('status') == 'PASS' else 1)
" || { echo "CPU preflight was not PASS -- refusing to train"; exit 1; }

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

test -f "${CONFIG_YAML}" && echo "config OK: ${CONFIG_YAML}" || { echo "MISSING config: ${CONFIG_YAML}"; exit 1; }

cd "${REPO_WORKDIR}"
git log --oneline -3

echo "=== run compact smoke training (2 epochs, lead06/seq24) ==="
BEFORE_RUN_MARKER="${RESULT_DIR}/.before_run"
touch "${BEFORE_RUN_MARKER}"

python scripts/run_stage1_nh.py train "${CONFIG_YAML}"
TRAIN_STATUS=$?

RUN_BASE="${GENERATED_OUT_DIR}/runs"
NEW_RUN_DIR=$(find "${RUN_BASE}" -mindepth 1 -maxdepth 1 -type d -newer "${BEFORE_RUN_MARKER}" | sort | tail -1)
rm -f "${BEFORE_RUN_MARKER}"

echo "=== training exit code: ${TRAIN_STATUS} ==="
echo "run output directory: ${NEW_RUN_DIR:-UNKNOWN}"

python -c "
import json
out = {
    'status': 'PASS' if ${TRAIN_STATUS} == 0 else 'FAIL',
    'train_exit_code': ${TRAIN_STATUS},
    'config_yaml': '${CONFIG_YAML}',
    'run_dir': '${NEW_RUN_DIR}',
    'slurm_job_id': '${SLURM_JOB_ID}',
    'cpu_preflight_job_id': '${CPU_PREFLIGHT_JOB_ID}',
    'hostname': __import__('socket').gethostname(),
}
json.dump(out, open('${RESULT_DIR}/gpu_smoke_result.json', 'w'), indent=2)
print(json.dumps(out, indent=2))
"

echo "=== done (exit ${TRAIN_STATUS}) ==="
exit ${TRAIN_STATUS}

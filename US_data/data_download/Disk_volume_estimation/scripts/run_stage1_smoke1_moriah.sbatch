#!/usr/bin/env bash
#SBATCH --job-name=flashnh-smoke1
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/smoke1-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/smoke1-%j.err
#
# Smoke 1 — Minimal meteorology smoke.
# PURPOSE: Verify 6 core RTMA forcing variables (2t, 2d, 2sh, 10u, 10v) load and
# train correctly alongside MRMS QPE.  seq_length=24 (same as Smoke 0) to isolate
# input-variable expansion from any lookback-window change.  3 epochs.
# NOT a scientific baseline.
#
# Pre-conditions before submitting:
#   - Smoke 0 PASS (job 45370683 — confirmed 2026-07-02)
#   - Smoke 1 preflight PASS (72 OK / 0 FAIL — confirmed 2026-07-02)
#   - Package regenerated with seq_length=24 Smoke 1 config (re-transfer required if
#     prior package had seq_length=72)
#   - flashnh-moriah env installed (job 45365952 PASS)
#   - Package at: /sci/labs/efratmorin/omripo/Flash-NH/data/stage1_pilot_v001
# NeuralHydrology 1.13 compat: dataset=generic, DD/MM/YYYY dates, epochs key,
# head=regression, output_activation=linear, attributes/ canonical subdir layout.
# Module fix from env-install job applied (non-interactive Slurm shell issue).

set -euo pipefail

FLASHNH_BASE="/sci/labs/efratmorin/omripo/Flash-NH"
ENV_PREFIX="${FLASHNH_BASE}/envs/flashnh-moriah"
DATA="${FLASHNH_BASE}/data/stage1_pilot_v001"
CONFIG="${DATA}/configs/stage1_smoke1_nh.yml"

cd "${FLASHNH_BASE}"

# Initialize module system for non-interactive Slurm shells (same fix as env-install job).
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

echo "=== preflight ==="
hostname
date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
which nh-run
nvidia-smi
python -u -c "
import sys, torch
print('Python:', sys.version.split()[0], flush=True)
print('PyTorch:', torch.__version__, flush=True)
cuda_ok = torch.cuda.is_available()
print('cuda available:', cuda_ok, flush=True)
if cuda_ok:
    print('cuda device:', torch.cuda.get_device_name(0), flush=True)
else:
    raise SystemExit('CUDA not available — check module loads and GPU allocation')
"
test -f "${CONFIG}" && echo "config OK: ${CONFIG}" || { echo "MISSING config: ${CONFIG}"; exit 1; }
echo "NC files in package: $(find "${DATA}/time_series" -name '*.nc' | wc -l)"
echo "attributes/attributes.csv: $(test -f "${DATA}/attributes/attributes.csv" && echo OK || echo MISSING)"

echo "=== run Smoke 1 ==="
# nh-run confirmed present after env install (job 45365952):
#   /sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah/bin/nh-run
#   nh-run --help prints valid usage with train/continue_training/finetune/evaluate.
# Fallback if nh-run is somehow not on PATH after activation:
#   python -m neuralhydrology.nh_run train --config-file "${CONFIG}"
nh-run train --config-file "${CONFIG}"

echo "=== done ==="

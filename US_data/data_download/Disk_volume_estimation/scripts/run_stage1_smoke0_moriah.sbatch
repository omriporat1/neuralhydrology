#!/usr/bin/env bash
#SBATCH --job-name=flashnh-smoke0
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/smoke0-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/smoke0-%j.err
#
# 2K-G-C-A preflight template — NOT YET RUN.
# Runs Smoke 0 (rain-only plumbing test) against the NH pilot package once it has
# been transferred to Moriah and the flashnh-moriah env has been installed.
# Do not submit until both of those steps are confirmed done.

set -euo pipefail

FLASHNH_BASE="/sci/labs/efratmorin/omripo/Flash-NH"
ENV_PREFIX="${FLASHNH_BASE}/envs/flashnh-moriah"
DATA="${FLASHNH_BASE}/data/stage1_pilot_v001"
CONFIG="${DATA}/configs/stage1_smoke0_nh.yml"

cd "${FLASHNH_BASE}"

module load miniconda3/24.3.0
module load nvidia/580.95.05
module load cuda/12.8.1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
which python

echo "=== preflight ==="
hostname
nvidia-smi
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
test -f "${CONFIG}" && echo "config OK: ${CONFIG}" || { echo "MISSING config: ${CONFIG}"; exit 1; }
echo "NC files in package: $(find "${DATA}/time_series" -name '*.nc' | wc -l)"

echo "=== run Smoke 0 ==="
# NH invocation — NOT YET VERIFIED on Moriah (env has not been installed/tested).
# Candidate A (primary choice for the first test): console-script entry point,
# installed automatically by `pip install neuralhydrology` per upstream docs
# (https://neuralhydrology.readthedocs.io quick-start: `nh-run train --config-file ...`).
#   nh-run train --config-file "${CONFIG}"
#
# Candidate B (fallback if `nh-run` is not on PATH after activation):
#   python -m neuralhydrology.nh_run train --config-file "${CONFIG}"
#
# Do NOT use `python -m neuralhydrology.training` — this was the previously
# generated invocation (scripts/build_stage1_nh_package.py _write_slurm) and is
# unverified/likely wrong; neuralhydrology.training is not documented as a CLI
# entry module upstream.
#
# Chosen for first test: Candidate A. Fall back to Candidate B only if `nh-run`
# is confirmed absent after env install.
nh-run train --config-file "${CONFIG}"

echo "=== done ==="

#!/bin/bash
# Disposable, non-scientific, ZERO-TRAINING rehearsal of the real
# fresh-proposal W&B-agent bridge (scripts/run_sweep_v1_wandb_bridge.py).
#
# This is the fresh-bridge sibling of
# run_sweep_v1_exact_retry_rehearsal_moriah.sbatch, but it cannot reuse that
# launcher's pattern of invoking main_from_manifest directly: the whole
# point of qualifying the fresh bridge is to exercise a REAL
# `wandb agent` -> `wandb.init()` -> `run.config` controller round trip,
# which the retry bridge (retrying an already-frozen record) never needs.
# So this launcher calls `wandb agent --count 1` exactly like the
# production launcher (run_sweep_v1_wandb_agent_moriah.sbatch) does. The
# only difference is WANDB_SWEEP_ID must be a disposable rehearsal sweep
# whose own W&B sweep config (built by
# build_sweep_v1_wandb_bridge_rehearsal_sweep_config.py) carries one static
# extra positional argument in its `command` field: an absolute rehearsal
# launch-manifest path. That third argv token is what makes `wandb agent`'s
# constructed argv route to run_sweep_v1_wandb_bridge.main_from_manifest
# (mode="rehearsal") instead of main() (hardcoded to mode="production"
# because the production sweep config's own `command` is argless).
#
# The manifest's own mode="rehearsal" + stop_before_training=True (both
# schema-enforced by src.baseline.sweep_v1_wandb_bridge_manifest, which also
# refuses the real production sweep id under mode="rehearsal") is what
# actually stops execution before training -- this script performs no
# training-avoidance logic of its own beyond the belt-and-suspenders guard
# below. CPU-only: no GRES requested.
#SBATCH --job-name=flashnh-sweep-v1-wandb-bridge-rehearsal
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/sweep-v1-wandb-bridge-rehearsal-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/sweep-v1-wandb-bridge-rehearsal-%j.err
set -euo pipefail

: "${WANDB_SWEEP_ID:?set the disposable rehearsal sweep id (never the production sweep)}"

FORBIDDEN_PRODUCTION_SWEEP_ID="4x3btz2s"
if [ "${WANDB_SWEEP_ID}" = "${FORBIDDEN_PRODUCTION_SWEEP_ID}" ]; then
    echo "REFUSING: WANDB_SWEEP_ID must never be the production sweep ${FORBIDDEN_PRODUCTION_SWEEP_ID}" >&2
    exit 1
fi

FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
REPO_WORKDIR="${REPO_WORKDIR:-${FLASHNH_BASE}/repos/flash-nh/US_data/data_download/Disk_volume_estimation}"
CANONICAL_PYTHON="${CANONICAL_PYTHON:-${FLASHNH_BASE}/envs/flashnh-moriah/bin/python}"

echo "=== Sweep-v1 fresh-proposal W&B-agent bridge REHEARSAL (disposable, zero-training) ==="
hostname; date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-unknown}"
echo "WANDB_SWEEP_ID: ${WANDB_SWEEP_ID}"
echo "REPO_WORKDIR: ${REPO_WORKDIR}"
echo "CANONICAL_PYTHON: ${CANONICAL_PYTHON}"

test -d "${REPO_WORKDIR}" || { echo "MISSING expected repo work directory: ${REPO_WORKDIR}"; exit 1; }
cd "${REPO_WORKDIR}"
test -x "${CANONICAL_PYTHON}" || { echo "MISSING canonical runtime interpreter: ${CANONICAL_PYTHON}"; exit 1; }

# wandb agent resolves ${interpreter} in the sweep config's `command` from
# whichever `python` is first on PATH when it spawns the child process, so
# the canonical env's bin/ directory must lead PATH here.
export PATH="$(dirname "${CANONICAL_PYTHON}"):${PATH}"

echo "=== starting agent ==="
# The bridge is the sole sweep program; --count 1 is the bounded proposal
# rule (exactly one disposable run consumed by this whole allocation).
wandb agent --count 1 "${WANDB_SWEEP_ID}"

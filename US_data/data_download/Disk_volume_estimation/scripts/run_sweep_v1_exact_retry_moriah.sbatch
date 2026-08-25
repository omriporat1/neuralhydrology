#!/usr/bin/env bash
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=flashnh-sweep-v1-exact-retry
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/sweep-v1-exact-retry-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/sweep-v1-exact-retry-%j.err
#
# One Slurm allocation runs exactly one exact-proposal PRODUCTION retry of an
# already-INVALID Sweep-v1 attempt via the shared, manifest-driven entry
# point (scripts/run_sweep_v1_exact_retry_bridge.py main_from_manifest) --
# never `wandb agent`, never a Bayesian-controller proposal request. The one
# positional argument is a src.baseline.sweep_v1_launch_manifest JSON file --
# no `--export=ALL,VAR=value,...` list, no long CLI flag chain. Production
# and the disposable rehearsal
# (scripts/run_sweep_v1_exact_retry_rehearsal_moriah.sbatch) share this same
# Python entry, manifest loader, runtime contract, identity resolution, W&B
# initialization helper, config preparation, and executor selector; they
# differ only in Slurm resources and in the manifest's own mode field
# (mode="production" here requires wandb_sweep_id==PRODUCTION_WANDB_SWEEP_ID
# and stop_before_training=False -- schema-enforced, so this launcher cannot
# be pointed at a disposable/rehearsal sweep by operator error).
#
# main_from_manifest calls src.baseline.sweep_v1_runtime_contract's
# run_full_runtime_contract (expected commit / interpreter / repo working
# directory / effective HOME / safe on-host credential-store metadata)
# BEFORE any W&B interaction -- this is the single production entry path;
# the legacy multi-flag CLI route cannot reach the real production sweep
# without that same verification (see scripts/run_sweep_v1_exact_retry_bridge.py).
#
# The bridge itself joins the existing production sweep via
# `wandb.init(settings=wandb.Settings(sweep_id=...))`, a mechanism
# empirically qualified against real wandb 0.28.1 in
# scripts/wandb_exact_retry_join_qualification.py (Moriah job 45939203, all
# 16 checks passed): a manually joined run is incorporated into the sweep's
# Bayesian-visible run population without consuming a controller proposal.
# One allocation retries exactly one attempt with its exact original five
# hyperparameters and Seed A; this launcher performs no loop, no array, and
# never resubmits itself.
#
# Usage:
#   sbatch scripts/run_sweep_v1_exact_retry_moriah.sbatch <launch_manifest.json>
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: sbatch run_sweep_v1_exact_retry_moriah.sbatch <launch_manifest.json>" >&2
    exit 1
fi
MANIFEST_PATH="$1"

FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
REPO_WORKDIR="${REPO_WORKDIR:-${FLASHNH_BASE}/repos/flash-nh/US_data/data_download/Disk_volume_estimation}"
CANONICAL_PYTHON="${CANONICAL_PYTHON:-${FLASHNH_BASE}/envs/flashnh-moriah/bin/python}"

echo "=== Sweep-v1 exact-retry PRODUCTION launch ==="
hostname
date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-unknown}"
echo "MANIFEST_PATH: ${MANIFEST_PATH}"
echo "REPO_WORKDIR: ${REPO_WORKDIR}"
echo "CANONICAL_PYTHON: ${CANONICAL_PYTHON}"

# Cheap, fast-failing shell-level existence checks only. The AUTHORITATIVE
# commit/dirty-tree/interpreter-identity pin is performed once, in Python,
# by the shared src.baseline.sweep_v1_runtime_contract module (called from
# inside main_from_manifest against the manifest's own expected_commit /
# expected_runtime_python fields) -- identical code path for production and
# rehearsal, never duplicated or re-implemented here in shell.
test -f "${MANIFEST_PATH}" || { echo "MISSING launch manifest: ${MANIFEST_PATH}"; exit 1; }
test -d "${REPO_WORKDIR}" || { echo "MISSING expected repo work directory: ${REPO_WORKDIR}"; exit 1; }
cd "${REPO_WORKDIR}"
test -x "${CANONICAL_PYTHON}" || { echo "MISSING canonical runtime interpreter: ${CANONICAL_PYTHON}"; exit 1; }

echo "=== launching scripts/run_sweep_v1_exact_retry_bridge.py (manifest mode) ==="
exec "${CANONICAL_PYTHON}" scripts/run_sweep_v1_exact_retry_bridge.py "${MANIFEST_PATH}"

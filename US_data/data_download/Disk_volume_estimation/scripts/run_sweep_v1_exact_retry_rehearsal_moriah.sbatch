#!/usr/bin/env bash
#SBATCH --job-name=flashnh-sweep-v1-exact-retry-rehearsal
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/sweep-v1-exact-retry-rehearsal-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/sweep-v1-exact-retry-rehearsal-%j.err
#
# Disposable, non-scientific, ZERO-TRAINING exact-retry STARTUP REHEARSAL.
#
# Runs the exact same shared, manifest-driven entry point
# (scripts/run_sweep_v1_exact_retry_bridge.py main_from_manifest) that a
# manifest-driven PRODUCTION exact-retry launch would use. The one
# positional argument is a src.baseline.sweep_v1_launch_manifest JSON file
# -- no `--export=ALL,VAR=value,...` list, no long CLI flag chain. The
# manifest's own mode="rehearsal" + stop_before_training=True fields (schema-
# enforced) are what make this run stop before any NH/torch import or
# training call; there is no separate dry-run flag on this launcher or on
# the executor. See docs/decision_log.md for the accepted read-only
# postmortem (attempt002/job 45939764 wandb-tag rejection,
# attempt003/job 45939848 authentication failure) this rehearsal exists to
# close, and CLAUDE.md Sec. 3 for the sealed-scope/authorization boundaries
# this launcher stays inside.
#
# CPU-only, no GRES. The manifest's own schema validation refuses a
# production wandb_sweep_id under mode="rehearsal" (see
# sweep_v1_launch_manifest.PRODUCTION_WANDB_SWEEP_ID), so this launcher
# cannot be pointed at the real production sweep even by operator error.
#
# Usage:
#   sbatch scripts/run_sweep_v1_exact_retry_rehearsal_moriah.sbatch <launch_manifest.json>
set -uo pipefail

MANIFEST_PATH="${1:?usage: sbatch run_sweep_v1_exact_retry_rehearsal_moriah.sbatch <launch_manifest.json>}"

FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
REPO_WORKDIR="${REPO_WORKDIR:-${FLASHNH_BASE}/repos/flash-nh/US_data/data_download/Disk_volume_estimation}"
CANONICAL_PYTHON="${CANONICAL_PYTHON:-${FLASHNH_BASE}/envs/flashnh-moriah/bin/python}"

echo "=== Sweep-v1 exact-retry startup REHEARSAL (disposable, zero-training) ==="
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
"${CANONICAL_PYTHON}" scripts/run_sweep_v1_exact_retry_bridge.py "${MANIFEST_PATH}"
STATUS=$?

echo "=== done (exit ${STATUS}) ==="
exit "${STATUS}"

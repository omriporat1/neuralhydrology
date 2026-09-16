#!/usr/bin/env bash
#SBATCH --job-name=flashnh-rd1-c4-d1-array
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/rd1-c4-d1-array-%A_%a.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/rd1-c4-d1-array-%A_%a.err
#
# RD1-C4-D1 ENTRY POINT 2 of 3: one trial per array task.
#
# NOT LAUNCHED. This file is created as part of the RD1-C4-D1 implementation
# task and has never been submitted.
#
# There is deliberately NO "#SBATCH --array=" directive here, and there is no
# default concurrency cap anywhere in this file. Both must be supplied by the
# operator at submit time, AFTER the benchmark has measured a real per-trial
# cost:
#
#     export D1_ARRAY_CONFIRM="run-all-24-trials"
#     sbatch --array=0-23%<CAP> scripts/rd1_c4_d1/rd1_c4_d1_array.sbatch
#
# Both the confirmation variable and the explicit --array range are required.
# Submitting this file bare produces a task with no SLURM_ARRAY_TASK_ID, which
# the guard below refuses; submitting it without the confirmation variable is
# refused too. Hard-coding a cap before the benchmark would be guessing, and
# the guess would silently become the campaign's concurrency policy.
#
# Resources are the same PROVISIONAL 1 CPU / 8 GB / 4 h as the benchmark, for
# the same reasons, and should be revised from the benchmark's measurements
# rather than from this comment.
#
# Each task runs exactly one trial and publishes exactly one atomic shard.
# There is no retry loop: a failed task leaves an attempt directory and no
# receipt, and re-running that index re-attempts it cleanly. Automatic retry
# would hide a systematic failure behind 24 identical stack traces.
#
# CPU-only. No GPU, no training, no inference, no NeuralHydrology evaluation
# product generation, no W&B, no controller, no sealed scope, and no automatic
# reduction -- reduction is a separate deliberate step that refuses to run
# until all 24 receipts qualify.

set -uo pipefail

if [ -n "${SLURM_JOB_ID:-}" ]; then
    # Under real Slurm, BASH_SOURCE[0] is the per-job spool copy
    # (/var/spool/slurmd/job<id>/slurm_script), not the checked-out script,
    # so the sibling common file must be located from the intended repo
    # clone identity instead (job 46175271).
    D1_RESOLUTION_MODE="slurm"
    FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
    REPO_CLONE_DIR="${REPO_CLONE_DIR:-${FLASHNH_BASE}/repos/flash-nh-rd1c4-d1}"
    D1_COMMON_SCRIPT="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation/scripts/rd1_c4_d1/rd1_c4_d1_common.sh"
else
    D1_RESOLUTION_MODE="local"
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    D1_COMMON_SCRIPT="${SCRIPT_DIR}/rd1_c4_d1_common.sh"
fi

if [ ! -f "${D1_COMMON_SCRIPT}" ]; then
    echo "REFUSING: cannot locate rd1_c4_d1_common.sh (resolution=${D1_RESOLUTION_MODE}, attempted=${D1_COMMON_SCRIPT}, REPO_CLONE_DIR=${REPO_CLONE_DIR:-unset})." >&2
    return 2 2>/dev/null || exit 2
fi
source "${D1_COMMON_SCRIPT}"

if [ "${D1_ARRAY_CONFIRM:-}" != "run-all-24-trials" ]; then
    echo "REFUSING: the full-array entry point requires an explicit confirmation." >&2
    echo "  export D1_ARRAY_CONFIRM=\"run-all-24-trials\"" >&2
    echo "If you only intended to measure one trial, use rd1_c4_d1_benchmark.sbatch." >&2
    exit 2
fi

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "REFUSING: no SLURM_ARRAY_TASK_ID -- this file was submitted without --array." >&2
    echo "Submit as: sbatch --array=0-23%<CAP> $(basename "${BASH_SOURCE[0]}")" >&2
    echo "The concurrency cap is an operator decision and is intentionally not defaulted." >&2
    exit 2
fi

if [ "${SLURM_ARRAY_TASK_ID}" -lt 0 ] || [ "${SLURM_ARRAY_TASK_ID}" -ge 24 ]; then
    echo "REFUSING: array index ${SLURM_ARRAY_TASK_ID} is outside the frozen 24-trial range 0-23." >&2
    exit 2
fi

rd1_c4_d1_require_inputs || exit 2

mkdir -p "${D1_STORE_ROOT}" "${FLASHNH_BASE}/tmp/rd1_c4_d1"
RESULT_LOG="${FLASHNH_BASE}/tmp/rd1_c4_d1/rd1_c4_d1_array_${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID}.log"
exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== RD1-C4-D1 array task ${SLURM_ARRAY_TASK_ID} of 0-23 ==="
rd1_c4_d1_setup_env || exit 2

python -m src.baseline.rd1_c4_observation_diagnostic_cli array-task \
    --trial-list "${D1_TRIAL_LIST}" \
    --contract "${D1_CONTRACT}" \
    --package-root "${D1_PACKAGE_ROOT}" \
    --store-root "${D1_STORE_ROOT}" \
    --repo-root "${REPO_WORKDIR}" \
    --array-index "${SLURM_ARRAY_TASK_ID}"
STATUS=$?

echo "=== array task ${SLURM_ARRAY_TASK_ID} done (exit ${STATUS}) ==="
echo "Reduction is NOT automatic. Run rd1_c4_d1_reduce.sbatch only after all 24"
echo "receipts exist and qualify."
exit ${STATUS}

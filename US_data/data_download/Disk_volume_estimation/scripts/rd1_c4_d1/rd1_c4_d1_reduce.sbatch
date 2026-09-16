#!/usr/bin/env bash
#SBATCH --job-name=flashnh-rd1-c4-d1-reduce
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/rd1-c4-d1-reduce-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/rd1-c4-d1-reduce-%j.err
#
# RD1-C4-D1 ENTRY POINT 3 of 3: final reduction.
#
# NOT LAUNCHED. This file is created as part of the RD1-C4-D1 implementation
# task and has never been submitted.
#
# Reduction is a separate, deliberate step. It is never chained from the array
# and never runs partially: the reducer itself requires exactly 24 receipt-
# qualified shards, exactly 12 Bayesian and 12 random-control, and exactly the
# 9,600-cell cross-product, and writes nothing at all if any gate fails. The
# pre-flight receipt count below is a courtesy that fails fast with a clear
# message -- it is not the gate.
#
# The reduction reads only shard content that is already on this filesystem.
# It performs no scientific classification, selects no threshold, and closes
# nothing: interpreting the resulting evidence is the user's decision.
#
# CPU-only. No GPU, no training, no inference, no W&B, no controller, no
# sealed scope, no submission.

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

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "REFUSING: reduction runs once, not as an array task." >&2
    exit 2
fi

rd1_c4_d1_require_inputs || exit 2

RECEIPT_DIR="${D1_STORE_ROOT}/observation_provenance/receipts"
N_RECEIPTS=$(find "${RECEIPT_DIR}" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
if [ "${N_RECEIPTS}" -ne 24 ]; then
    echo "REFUSING: found ${N_RECEIPTS} trial receipts under ${RECEIPT_DIR}, expected exactly 24." >&2
    echo "Reduction never runs on a partial cross-product. Complete the missing trials first." >&2
    exit 2
fi

mkdir -p "${D1_OUT_DIR}" "${FLASHNH_BASE}/tmp/rd1_c4_d1"
RESULT_LOG="${FLASHNH_BASE}/tmp/rd1_c4_d1/rd1_c4_d1_reduce_${SLURM_JOB_ID:-local}.log"
exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== RD1-C4-D1 reduction (${N_RECEIPTS} receipts present) ==="
rd1_c4_d1_setup_env || exit 2

python -m src.baseline.rd1_c4_observation_diagnostic_cli reduce \
    --trial-list "${D1_TRIAL_LIST}" \
    --contract "${D1_CONTRACT}" \
    --store-root "${D1_STORE_ROOT}" \
    --out-dir "${D1_OUT_DIR}"
STATUS=$?

echo "=== reduction done (exit ${STATUS}) ==="
echo "Compact outputs in ${D1_OUT_DIR} (summaries, statuses, extremes, manifests,"
echo "receipts, hashes). The per-element detail stays on this filesystem and is"
echo "not part of the transferred layer."
exit ${STATUS}

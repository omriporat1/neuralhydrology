#!/usr/bin/env bash
#SBATCH --job-name=flashnh-rd1-c4-d1-benchmark
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/rd1-c4-d1-benchmark-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/rd1-c4-d1-benchmark-%j.err
#
# RD1-C4-D1 ENTRY POINT 1 of 3: single-trial benchmark.
#
# NOT LAUNCHED. This file is created as part of the RD1-C4-D1 implementation
# task and has never been submitted. Submitting it is a separate, explicit
# operator decision.
#
# Purpose: measure the real cost of ONE trial (400 basins, one ~84.5 MB
# validation pickle, ~3.35 M retained detail elements) before any array is
# sized. The 1 CPU / 8 GB / 4 h request below is a PROVISIONAL default, not a
# measured one:
#   * 1 CPU because no within-task parallelism is implemented. Eight CPUs are
#     deliberately NOT reserved -- reserving cores no code can use wastes the
#     allocation and makes the array harder to schedule.
#   * 8 GB from job 46168080's own sacct MaxRSS (~2.4 GB for 24x400 in one
#     process); D1 holds one trial's results plus one basin's detail.
#   * 4 h as a safe upper bound. The benchmark exists precisely to replace
#     these three numbers with measurements.
#
# ACCIDENT PROOFING. There is intentionally NO "#SBATCH --array" directive in
# this file, the script refuses to run if it finds itself inside an array
# allocation, and the trial must be named explicitly -- it cannot be selected
# by index. A benchmark therefore cannot silently become 24 tasks.
#
# STORE ISOLATION. The benchmark never writes into the shared array/default
# store. The operator must export D1_BENCHMARK_STORE_ROOT with an explicit,
# not-yet-existing path under the approved benchmark area; the value is never
# generated for them, so the benchmark attempt's identity is always something
# the operator chose and can audit. See _d1_benchmark_validate_store_root()
# below for the exact refusal conditions.
#
# CPU-only. No GPU, no training, no inference, no NeuralHydrology evaluation
# product generation, no W&B, no controller, no submission of any other job,
# no sealed scope.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/rd1_c4_d1_common.sh"

# Approved project-local root for disposable benchmark shards. Never the
# shared array/default store (${FLASHNH_BASE}/tmp/rd1_c4_d1/store).
D1_BENCHMARK_STORE_AREA="${FLASHNH_BASE}/tmp/rd1_c4_d1/benchmarks"

_d1_benchmark_validate_store_root() {
    # Validates D1_BENCHMARK_STORE_ROOT (and the already-resolved D1_STORE_ROOT
    # from rd1_c4_d1_common.sh, for the shared-equality check) and, on
    # success, prints the resolved canonical benchmark store path on stdout.
    # Refuses (nonzero return, message on stderr) for every unsafe or
    # ambiguous case. This function only reads state -- it never creates,
    # deletes or mutates anything -- so it is safe to call from a test.
    if [ -z "${D1_BENCHMARK_STORE_ROOT:-}" ]; then
        echo "REFUSING: export D1_BENCHMARK_STORE_ROOT=<explicit, not-yet-existing path> before submitting." >&2
        echo "The benchmark never writes into the shared array/default store, and it never invents a" >&2
        echo "path for you: name a dedicated path under ${D1_BENCHMARK_STORE_AREA} yourself so this" >&2
        echo "benchmark attempt's identity can be recorded and audited." >&2
        return 2
    fi

    if [[ -e "${D1_BENCHMARK_STORE_ROOT}" || -L "${D1_BENCHMARK_STORE_ROOT}" ]]; then
        echo "REFUSING: D1_BENCHMARK_STORE_ROOT already exists or is a symbolic link: ${D1_BENCHMARK_STORE_ROOT}" >&2
        return 1
    fi

    local resolved area shared_default
    resolved="$(realpath -m -- "${D1_BENCHMARK_STORE_ROOT}" 2>/dev/null)" || {
        echo "REFUSING: D1_BENCHMARK_STORE_ROOT could not be resolved to a canonical path: ${D1_BENCHMARK_STORE_ROOT}" >&2
        return 2
    }
    area="$(realpath -m -- "${D1_BENCHMARK_STORE_AREA}")"
    shared_default="$(realpath -m -- "${FLASHNH_BASE}/tmp/rd1_c4_d1/store")"

    case "${resolved}" in
        "${area}"/*) ;;
        *)
            echo "REFUSING: D1_BENCHMARK_STORE_ROOT (${resolved}) is not inside the approved benchmark area ${area}." >&2
            return 2
            ;;
    esac

    if [ "${resolved}" = "${shared_default}" ]; then
        echo "REFUSING: D1_BENCHMARK_STORE_ROOT resolves to the shared array/default store (${shared_default})." >&2
        return 2
    fi

    if [ -n "${D1_STORE_ROOT:-}" ]; then
        local shared_supplied
        shared_supplied="$(realpath -m -- "${D1_STORE_ROOT}")"
        if [ "${resolved}" = "${shared_supplied}" ]; then
            echo "REFUSING: D1_BENCHMARK_STORE_ROOT must not equal the shared D1_STORE_ROOT (${shared_supplied})." >&2
            return 2
        fi
    fi

    if [ -e "${resolved}" ]; then
        echo "REFUSING: benchmark store root already exists: ${resolved} -- a benchmark must never reuse an earlier shard." >&2
        return 2
    fi

    printf '%s\n' "${resolved}"
    return 0
}

# Sourcing this file (e.g. from a test) stops here: everything above this
# guard is safe to define and evaluate at source time (no execution, no I/O
# beyond realpath's pure path arithmetic). Everything below only runs when
# the file is actually executed as a script, exactly as Slurm/bash would.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    return 0 2>/dev/null || exit 0
fi

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] || [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    echo "REFUSING: the benchmark was submitted with --array (task ${SLURM_ARRAY_TASK_ID:-?})." >&2
    echo "The benchmark runs exactly one trial. Use rd1_c4_d1_array.sbatch deliberately" >&2
    echo "if the full 24-trial array is genuinely intended." >&2
    exit 2
fi

if [ -z "${D1_BENCHMARK_TRIAL_ID:-}" ]; then
    echo "REFUSING: export D1_BENCHMARK_TRIAL_ID=<exact trial id> before submitting." >&2
    echo "The benchmark never selects a trial by array index." >&2
    exit 2
fi

D1_BENCHMARK_RESOLVED_STORE_ROOT="$(_d1_benchmark_validate_store_root)" || exit 2
export D1_STORE_ROOT="${D1_BENCHMARK_RESOLVED_STORE_ROOT}"

rd1_c4_d1_require_inputs || exit 2

mkdir -p "${D1_STORE_ROOT}" "${FLASHNH_BASE}/tmp/rd1_c4_d1"
RESULT_LOG="${FLASHNH_BASE}/tmp/rd1_c4_d1/rd1_c4_d1_benchmark_${SLURM_JOB_ID:-local}.log"
exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== RD1-C4-D1 single-trial benchmark ==="
rd1_c4_d1_setup_env || exit 2

/usr/bin/time -v python -m src.baseline.rd1_c4_observation_diagnostic_cli benchmark \
    --trial-list "${D1_TRIAL_LIST}" \
    --contract "${D1_CONTRACT}" \
    --package-root "${D1_PACKAGE_ROOT}" \
    --store-root "${D1_STORE_ROOT}" \
    --repo-root "${REPO_WORKDIR}" \
    --trial-id "${D1_BENCHMARK_TRIAL_ID}"
STATUS=$?

echo "=== benchmark done (exit ${STATUS}) ==="
echo "Size the array from the elapsed time and MaxRSS above. Do not hard-code a"
echo "concurrency cap: the array entry point requires an explicit operator cap."
exit ${STATUS}

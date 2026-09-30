#!/bin/bash
# Clean-login-shell submission of the fixed RD1 P20-P24 Slurm chain.
#
# THIS is the only place any job in this chain gets submitted. Every job
# (scripts/rd1_p20_p24_job.sbatch) runs exactly once and never calls
# sbatch itself -- that is the structural fix for the P20 forensic finding
# (a transition job contacting W&B, then sbatch'ing a child job with an
# implicit --export=ALL, so the child inherited a stale WANDB_SERVICE
# socket handle and failed before creating any run).
#
# Submits P20 unconditionally (afterok on nothing), then P21 afterok P20,
# P22 afterok P21, P23 afterok P22, P24 afterok P23. The loop below is
# HARDCODED to exactly {20,21,22,23,24} -- there is no variable upper bound,
# no "submit N more" option, and no code path that can ever produce a P25
# submission from this script.
#
# Every sbatch call below uses an explicit --export= allowlist. There is no
# --export=ALL anywhere in this script.
#
# USAGE (production launch: invoke the DEPLOYED copy of this script, never
# the tracked repo copy -- scripts/deploy_rd1_p20_p24_harness.py must have
# already deployed the harness, including this script and
# rd1_p20_p24_job.sbatch, into ${CHAIN_DIR}/harness/scripts/ before this
# runs. This script's own self-location logic below (JOB_SBATCH derived from
# BASH_SOURCE) then resolves rd1_p20_p24_job.sbatch from that SAME deployed
# directory, never from the caller's cwd and never from the repository
# scripts/ path):
#   bash ${RD1_CHAIN_DIR}/harness/scripts/submit_rd1_p20_p24_chain.sh \
#       --registry-path /sci/home/omripo/bayes_run_registry_v3.json \
#       --chain-dir /sci/labs/efratmorin/omripo/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation/.scratch_local/rd1_p20_p24_<timestamp> \
#       --wandb-sweep-id wta85z3b \
#       --expected-commit dabd2ca851bd2b3a03035886cfa50015f2c864b4 \
#       --execution-generation <N> \
#       --output-root-base <path> \
#       --p20-pinned-manifest-path <path> \
#       --p20-pinned-manifest-sha256 <sha256> \
#       --payload-repository-path <path> \
#       [--dry-run]
#
# --payload-repository-path is required and points at a detached checkout
# whose HEAD must exactly equal --expected-commit and whose working tree
# must be clean; it is what every job's REPO_WORKDIR resolves to (exported
# below), so the bridge process that actually runs training always executes
# out of this exact checkout, never out of whatever the tracked harness repo
# happens to have checked out at submission time.
#
# (For local development/testing of this script's own logic -- never for a
# production launch -- the tracked repo copy at scripts/
# submit_rd1_p20_p24_chain.sh can be invoked directly instead.)
#
# --dry-run is forwarded to every job (RD1_DRY_RUN=--dry-run): each job
# still runs preflight + manifest identity/validation, then exits 0 before
# any W&B contact. This lets the whole fixed-dependency chain topology be
# exercised end-to-end (Slurm afterok wiring, environment allowlist,
# preflight/manifest logic) with zero W&B/registry mutation.
set -euo pipefail

PROPOSAL_ORDERS=(20 21 22 23 24)

REGISTRY_PATH=""
CHAIN_DIR=""
WANDB_SWEEP_ID=""
EXPECTED_COMMIT=""
EXECUTION_GENERATION=""
OUTPUT_ROOT_BASE=""
P20_PINNED_MANIFEST_PATH=""
P20_PINNED_MANIFEST_SHA256=""
PAYLOAD_REPOSITORY_PATH=""
DRY_RUN=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --registry-path) REGISTRY_PATH="$2"; shift 2 ;;
        --chain-dir) CHAIN_DIR="$2"; shift 2 ;;
        --wandb-sweep-id) WANDB_SWEEP_ID="$2"; shift 2 ;;
        --expected-commit) EXPECTED_COMMIT="$2"; shift 2 ;;
        --execution-generation) EXECUTION_GENERATION="$2"; shift 2 ;;
        --output-root-base) OUTPUT_ROOT_BASE="$2"; shift 2 ;;
        --p20-pinned-manifest-path) P20_PINNED_MANIFEST_PATH="$2"; shift 2 ;;
        --p20-pinned-manifest-sha256) P20_PINNED_MANIFEST_SHA256="$2"; shift 2 ;;
        --payload-repository-path) PAYLOAD_REPOSITORY_PATH="$2"; shift 2 ;;
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        *) echo "FATAL: unknown argument $1" >&2; exit 2 ;;
    esac
done

for _required_name in REGISTRY_PATH CHAIN_DIR WANDB_SWEEP_ID EXPECTED_COMMIT EXECUTION_GENERATION OUTPUT_ROOT_BASE P20_PINNED_MANIFEST_PATH P20_PINNED_MANIFEST_SHA256 PAYLOAD_REPOSITORY_PATH; do
    if [ -z "${!_required_name}" ]; then
        echo "FATAL: --$(echo "${_required_name}" | tr '[:upper:]_' '[:lower:]-') is required" >&2
        exit 2
    fi
done

# A production launch must target only the one authorized sweep, never a
# forbidden or disposable-rehearsal id (same static literals the maintained
# agent launcher refuses on; kept in sync by a focused test).
FORBIDDEN_PRODUCTION_SWEEP_ID="4x3btz2s"
FORBIDDEN_DISPOSABLE_REHEARSAL_SWEEP_ID="oz5p4csb"
if [ "${WANDB_SWEEP_ID}" = "${FORBIDDEN_PRODUCTION_SWEEP_ID}" ] || [ "${WANDB_SWEEP_ID}" = "${FORBIDDEN_DISPOSABLE_REHEARSAL_SWEEP_ID}" ]; then
    echo "FATAL: refusing forbidden sweep id ${WANDB_SWEEP_ID}" >&2
    exit 2
fi

# Restore the Moriah Slurm client environment (stable host facts per
# docs/remote_operations.md); guarded/idempotent, matching the convention in
# scripts/submit_sweep_v2_six_axis_wandb_agent_moriah.sh.
_slurm_bin="${FLASHNH_MORIAH_SLURM_BIN:-/vol/slurm/moriah/bindir/bin}"
_slurm_conf="${FLASHNH_MORIAH_SLURM_CONF:-/vol/slurm/moriah/slurm.conf}"
if ! command -v sbatch >/dev/null 2>&1 && [ -x "${_slurm_bin}/sbatch" ]; then
    export PATH="${_slurm_bin}:${PATH}"
fi
if [ -z "${SLURM_CONF:-}" ] && [ -f "${_slurm_conf}" ]; then
    export SLURM_CONF="${_slurm_conf}"
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "FATAL: sbatch is not on PATH and ${_slurm_bin}/sbatch is not executable; cannot submit" >&2
    exit 2
fi

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SBATCH="${_script_dir}/rd1_p20_p24_job.sbatch"
test -f "${JOB_SBATCH}" || { echo "FATAL: job script not found: ${JOB_SBATCH}" >&2; exit 2; }

# Fail-closed project-local artifact boundary check (CLAUDE.md, non-
# negotiable), mirroring the identical check in rd1_p20_p24_job.sbatch: the
# chain's per-job Slurm log directory (created immediately below) must never
# land outside this project's own .scratch_local/, regardless of what
# --chain-dir was passed. `realpath -m` normalizes without requiring the
# path to exist yet -- CHAIN_DIR itself is created right after this check.
FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
REPO_WORKDIR="${REPO_WORKDIR:-${FLASHNH_BASE}/repos/flash-nh/US_data/data_download/Disk_volume_estimation}"
CHAIN_DIR_REAL="$(realpath -m "${CHAIN_DIR}")"
REPO_WORKDIR_REAL="$(realpath -m "${REPO_WORKDIR}")"
case "${CHAIN_DIR_REAL}/" in
    "${REPO_WORKDIR_REAL}/.scratch_local/"*) ;;
    *)
        echo "FATAL: --chain-dir must resolve beneath ${REPO_WORKDIR_REAL}/.scratch_local/ (got ${CHAIN_DIR_REAL})" >&2
        exit 2
        ;;
esac

# --payload-repository-path must be a clean checkout pinned exactly at
# --expected-commit. This is exported below as the job's REPO_WORKDIR (with
# RD1_PROJECT_ROOT separately pinned to REPO_WORKDIR_REAL above, the real
# tracked-repo project root) so every job's bridge process runs out of this
# checkout instead of silently defaulting to the tracked harness repo.
PAYLOAD_REPOSITORY_PATH_REAL="$(realpath -m "${PAYLOAD_REPOSITORY_PATH}")"
if [ ! -e "${PAYLOAD_REPOSITORY_PATH_REAL}/.git" ]; then
    echo "FATAL: --payload-repository-path ${PAYLOAD_REPOSITORY_PATH_REAL} is not a git checkout" >&2
    exit 2
fi
_payload_head="$(git -C "${PAYLOAD_REPOSITORY_PATH_REAL}" rev-parse HEAD)"
if [ "${_payload_head}" != "${EXPECTED_COMMIT}" ]; then
    echo "FATAL: --payload-repository-path HEAD ${_payload_head} does not match --expected-commit ${EXPECTED_COMMIT}" >&2
    exit 2
fi
# Only tracked-content modifications make the checkout scientifically dirty;
# untracked runtime artifacts (e.g. a wandb/ run-log directory left behind by
# an earlier attempt) do not change what code executes and are intentionally
# not flagged here so this check never forces deleting evidence.
_payload_dirty="$(git -C "${PAYLOAD_REPOSITORY_PATH_REAL}" status --porcelain --untracked-files=no)"
if [ -n "${_payload_dirty}" ]; then
    echo "FATAL: --payload-repository-path ${PAYLOAD_REPOSITORY_PATH_REAL} has uncommitted tracked-file changes; refusing to submit against a non-pristine payload checkout" >&2
    exit 2
fi

LOCK_PATH="${CHAIN_DIR}/rd1_p20_p24_registry.lock"
LOG_DIR="${CHAIN_DIR}/logs"
mkdir -p "${CHAIN_DIR}" "${LOG_DIR}"

echo "=== RD1 P20-P24 fixed chain submission (hardcoded orders: ${PROPOSAL_ORDERS[*]}) ===" >&2
echo "registry=${REGISTRY_PATH} chain_dir=${CHAIN_DIR} sweep=${WANDB_SWEEP_ID} dry_run=${DRY_RUN:-<none>}" >&2

PREVIOUS_JOB_ID=""
declare -A SUBMITTED_JOB_IDS=()
for ORDER in "${PROPOSAL_ORDERS[@]}"; do
    EXPORT_LIST="NONE"
    EXPORT_LIST="${EXPORT_LIST},RD1_PROPOSAL_ORDER=${ORDER}"
    EXPORT_LIST="${EXPORT_LIST},RD1_CHAIN_DIR=${CHAIN_DIR}"
    EXPORT_LIST="${EXPORT_LIST},RD1_REGISTRY_PATH=${REGISTRY_PATH}"
    EXPORT_LIST="${EXPORT_LIST},RD1_LOCK_PATH=${LOCK_PATH}"
    EXPORT_LIST="${EXPORT_LIST},RD1_WANDB_SWEEP_ID=${WANDB_SWEEP_ID}"
    EXPORT_LIST="${EXPORT_LIST},RD1_EXPECTED_COMMIT=${EXPECTED_COMMIT}"
    EXPORT_LIST="${EXPORT_LIST},RD1_EXECUTION_GENERATION=${EXECUTION_GENERATION}"
    EXPORT_LIST="${EXPORT_LIST},RD1_OUTPUT_ROOT_BASE=${OUTPUT_ROOT_BASE}"
    EXPORT_LIST="${EXPORT_LIST},REPO_WORKDIR=${PAYLOAD_REPOSITORY_PATH_REAL}"
    EXPORT_LIST="${EXPORT_LIST},RD1_PROJECT_ROOT=${REPO_WORKDIR_REAL}"
    if [ "${ORDER}" = "20" ]; then
        EXPORT_LIST="${EXPORT_LIST},RD1_P20_PINNED_MANIFEST_PATH=${P20_PINNED_MANIFEST_PATH}"
        EXPORT_LIST="${EXPORT_LIST},RD1_P20_PINNED_MANIFEST_SHA256=${P20_PINNED_MANIFEST_SHA256}"
    fi
    if [ -n "${DRY_RUN}" ]; then
        EXPORT_LIST="${EXPORT_LIST},RD1_DRY_RUN=${DRY_RUN}"
    fi

    SBATCH_ARGS=(
        --parsable
        --export="${EXPORT_LIST}"
        --output="${LOG_DIR}/rd1-p${ORDER}-%j.out"
        --error="${LOG_DIR}/rd1-p${ORDER}-%j.err"
    )
    if [ -n "${PREVIOUS_JOB_ID}" ]; then
        SBATCH_ARGS+=(--dependency="afterok:${PREVIOUS_JOB_ID}")
    fi

    echo "submitting P${ORDER} (dependency=${PREVIOUS_JOB_ID:-<none>})" >&2
    JOB_ID="$(sbatch "${SBATCH_ARGS[@]}" "${JOB_SBATCH}")"
    echo "P${ORDER} job_id=${JOB_ID}" >&2
    SUBMITTED_JOB_IDS["${ORDER}"]="${JOB_ID}"
    PREVIOUS_JOB_ID="${JOB_ID}"
done

echo "--- submitted job ids ---"
for ORDER in "${PROPOSAL_ORDERS[@]}"; do
    echo "P${ORDER}=${SUBMITTED_JOB_IDS[${ORDER}]}"
done

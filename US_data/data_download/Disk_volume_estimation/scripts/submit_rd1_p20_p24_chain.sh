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
# P22 afterok P21, P23 afterok P22, P24 afterok P23. The order set is fixed
# to exactly {20,21,22,23,24} (the default --start-order 20), or, for the
# P21 resume path (--start-order 21), exactly {21,22,23,24} with P21 itself
# afterok nothing -- there is no variable upper bound, no "submit N more"
# option, and no code path that can ever produce a P25 submission or
# resubmit P20 from this script.
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
#       [--start-order 20|21] [--dry-run]
#
# --start-order (optional, default 20): 20 submits the full fixed P20-P24
# chain as above. 21 submits the P21-P24 resume chain instead -- P21 with no
# dependency, P22 afterok P21, P23 afterok P22, P24 afterok P23 -- and:
#   * requires the production registry to already hold exactly P1-P20
#     (verified here, before any sbatch call -- see the resume-precondition
#     check below);
#   * requires --p20-pinned-manifest-path/--p20-pinned-manifest-sha256 to be
#     OMITTED (P20 is not being submitted in this mode, so there is nothing
#     for them to pin);
#   * never submits P20, and never submits anything past P24.
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

REGISTRY_PATH=""
CHAIN_DIR=""
WANDB_SWEEP_ID=""
EXPECTED_COMMIT=""
EXECUTION_GENERATION=""
OUTPUT_ROOT_BASE=""
P20_PINNED_MANIFEST_PATH=""
P20_PINNED_MANIFEST_SHA256=""
PAYLOAD_REPOSITORY_PATH=""
START_ORDER=""
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
        --start-order) START_ORDER="$2"; shift 2 ;;
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        *) echo "FATAL: unknown argument $1" >&2; exit 2 ;;
    esac
done

START_ORDER="${START_ORDER:-20}"
case "${START_ORDER}" in
    20|21) ;;
    *) echo "FATAL: --start-order must be 20 or 21 (got ${START_ORDER})" >&2; exit 2 ;;
esac
case "${START_ORDER}" in
    20) PROPOSAL_ORDERS=(20 21 22 23 24) ;;
    21) PROPOSAL_ORDERS=(21 22 23 24) ;;
esac

REQUIRED_ARG_NAMES=(REGISTRY_PATH CHAIN_DIR WANDB_SWEEP_ID EXPECTED_COMMIT EXECUTION_GENERATION OUTPUT_ROOT_BASE PAYLOAD_REPOSITORY_PATH)
if [ "${START_ORDER}" = "20" ]; then
    REQUIRED_ARG_NAMES+=(P20_PINNED_MANIFEST_PATH P20_PINNED_MANIFEST_SHA256)
fi
for _required_name in "${REQUIRED_ARG_NAMES[@]}"; do
    if [ -z "${!_required_name}" ]; then
        echo "FATAL: --$(echo "${_required_name}" | tr '[:upper:]_' '[:lower:]-') is required" >&2
        exit 2
    fi
done

if [ "${START_ORDER}" = "21" ] && { [ -n "${P20_PINNED_MANIFEST_PATH}" ] || [ -n "${P20_PINNED_MANIFEST_SHA256}" ]; }; then
    echo "FATAL: --p20-pinned-manifest-path/--p20-pinned-manifest-sha256 must not be given with --start-order 21 (P20 is not submitted in resume mode)" >&2
    exit 2
fi

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
# --payload-repository-path may legitimately be a subdirectory of a larger
# checkout (e.g. this project nested inside a full repo clone) rather than a
# git root itself, so existence is proved by a working git command (which
# discovers the enclosing repo upward) rather than a literal <path>/.git
# check, which would wrongly refuse a valid nested checkout.
if ! _payload_head="$(git -C "${PAYLOAD_REPOSITORY_PATH_REAL}" rev-parse HEAD 2>&1)"; then
    echo "FATAL: --payload-repository-path ${PAYLOAD_REPOSITORY_PATH_REAL} is not inside a git checkout (git rev-parse HEAD failed: ${_payload_head})" >&2
    exit 2
fi
if [ "${_payload_head}" != "${EXPECTED_COMMIT}" ]; then
    echo "FATAL: --payload-repository-path HEAD ${_payload_head} does not match --expected-commit ${EXPECTED_COMMIT}" >&2
    exit 2
fi
# Only tracked-content modifications make the checkout scientifically dirty;
# untracked runtime artifacts (e.g. a wandb/ run-log directory left behind by
# an earlier attempt) do not change what code executes and are intentionally
# not flagged here so this check never forces deleting evidence. Scoped with
# "-- ." to this project's own subtree, since --payload-repository-path may
# be a subdirectory of a larger checkout whose other paths are irrelevant.
_payload_dirty="$(git -C "${PAYLOAD_REPOSITORY_PATH_REAL}" status --porcelain --untracked-files=no -- .)"
if [ -n "${_payload_dirty}" ]; then
    echo "FATAL: --payload-repository-path ${PAYLOAD_REPOSITORY_PATH_REAL} has uncommitted tracked-file changes; refusing to submit against a non-pristine payload checkout" >&2
    exit 2
fi

# Resume precondition (Part B, 2026-09-30): before submitting the P21-P24
# resume chain, the production registry must already hold exactly P1-P20.
# Reuses the exact qualified check_preflight/compute_expected_prior_orders
# gate that rd1_p20_p24_job.sbatch's own "preflight" subcommand enforces
# (under an exclusive/shared RegistryLock) at job-run time for every order --
# that per-job locked gate remains the sole authoritative, race-safe
# enforcement point. This check is deliberately lock-free (it never touches
# RegistryLock/fcntl, which is POSIX-only and irrelevant to a plain read) so
# it can run here, at submission time, before any job is even queued: a
# fail-fast convenience, not a replacement authority -- any registry drift
# between this check and job dispatch is still caught by the per-job
# preflight, unchanged.
if [ "${START_ORDER}" = "21" ]; then
    CANONICAL_PYTHON="${CANONICAL_PYTHON:-${FLASHNH_BASE}/envs/flashnh-moriah/bin/python}"
    test -x "${CANONICAL_PYTHON}" || { echo "FATAL: MISSING canonical interpreter for resume precondition check: ${CANONICAL_PYTHON}" >&2; exit 2; }
    echo "--- verifying production registry is exactly P1-P20 before any submission (resume precondition) ---" >&2
    "${CANONICAL_PYTHON}" - "${REGISTRY_PATH}" "${_script_dir}/rd1_p20_p24_job.py" <<'PYEOF'
import importlib.util
import sys

registry_path, job_module_path = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("rd1_p20_p24_job_preflight_check", job_module_path)
job = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = job  # dataclasses' frozen=True needs the module registered to resolve annotations
spec.loader.exec_module(job)

registry = job.read_registry(registry_path)
current_orders = job.registry_orders(registry)
try:
    job.check_preflight(21, current_orders)
except job.PreflightError as exc:
    print(f"FATAL: resume precondition refused: {exc.reason}: {exc.detail}", file=sys.stderr)
    sys.exit(1)
print(f"RESUME_PREFLIGHT_OK start_order=21 current_orders={current_orders}")
PYEOF
fi

LOCK_PATH="${CHAIN_DIR}/rd1_p20_p24_registry.lock"
LOG_DIR="${CHAIN_DIR}/logs"
mkdir -p "${CHAIN_DIR}" "${LOG_DIR}"

echo "=== RD1 P20-P24 fixed chain submission (start_order=${START_ORDER} orders: ${PROPOSAL_ORDERS[*]}) ===" >&2
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

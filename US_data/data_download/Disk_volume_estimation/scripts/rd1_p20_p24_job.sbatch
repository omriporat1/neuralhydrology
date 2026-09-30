#!/bin/bash
#SBATCH --job-name=flashnh-rd1-p20p24
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --chdir=/sci/labs/efratmorin/omripo/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation
set -euo pipefail

# Log containment (review fix, 2026-09-27): this script previously carried
# fixed #SBATCH --output/--error directives pointing outside the project
# root (a Flash-NH-base-wide .../Flash-NH/logs/ directory), violating the
# project-local artifact boundary (CLAUDE.md, non-negotiable). --output/
# --error are now supplied per submission by
# scripts/submit_rd1_p20_p24_chain.sh, pointing at
# ${RD1_CHAIN_DIR}/logs/ -- never hardcoded here.

# One self-contained proposal job in the fixed RD1 P20-P24 chain. Replaces
# the retired self-propagating controller/transition-job pair: this job
# runs exactly once, does preflight -> (reuse|build+validate manifest) ->
# one W&B agent attempt (with at most one narrowly-scoped same-order retry,
# decided entirely by the tested Python driver) -> registry append+verify,
# and exits. IT NEVER CALLS sbatch. The fixed afterok dependency across
# P20-P24 is created once, up front, by
# scripts/submit_rd1_p20_p24_chain.sh from a clean login shell -- that is
# the only place any of these jobs gets submitted.
#
# Correction (Phase A.1 review, 2026-09-27): this script previously ran its
# own bash while-loop implementing a second, weaker retry decision that
# never captured a "before" run-ids snapshot (it only audited once, after a
# failure, with no baseline to diff against). That loop is removed. This
# script is now a thin wrapper: it prepares a clean environment and hands
# the entire attempt/retry decision to
# scripts/rd1_p20_p24_job.py run-agent-with-retry, the one execution path
# that uses the already-tested run_agent_with_retry /
# is_socket_failure_retry_eligible functions.
#
# Every operational input below must arrive via the explicit, reviewed
# --export= allowlist that scripts/submit_rd1_p20_p24_chain.sh sets on the
# sbatch call for this script; there is no --export=ALL anywhere in this
# chain. Missing a required one is a hard, immediate, pre-W&B-contact
# refusal (":?" guards), never a silent default.
: "${RD1_PROPOSAL_ORDER:?required: the proposal order for this job (20-24)}"
: "${RD1_CHAIN_DIR:?required: evidence/manifest/log directory for this chain run}"
: "${RD1_REGISTRY_PATH:?required: path to bayes_run_registry_v3.json}"
: "${RD1_LOCK_PATH:?required: dedicated registry lock file path}"
: "${RD1_WANDB_SWEEP_ID:?required: the one authorized production sweep id}"
: "${RD1_EXPECTED_COMMIT:?required: pinned scientific execution commit}"
: "${RD1_EXECUTION_GENERATION:?required: execution-generation for build-manifest}"
: "${RD1_OUTPUT_ROOT_BASE:?required: output-root-base for build-manifest}"
RD1_DRY_RUN="${RD1_DRY_RUN:-}"
# P20-only: reuse the existing validated manifest byte-for-byte instead of
# building a new one. Both must be set together, and only for order 20.
RD1_P20_PINNED_MANIFEST_PATH="${RD1_P20_PINNED_MANIFEST_PATH:-}"
RD1_P20_PINNED_MANIFEST_SHA256="${RD1_P20_PINNED_MANIFEST_SHA256:-}"
# Optional: override the W&B project/entity the freshly-built manifest (the
# non-P20 branch below) targets. Left unset for every production launch, so
# build-manifest keeps using its own hardcoded default (the one production
# project) exactly as before. Only ever set for an authorized disposable
# diagnostic, whose manifest must target a disposable project/entity instead.
RD1_WANDB_PROJECT="${RD1_WANDB_PROJECT:-}"
RD1_WANDB_ENTITY="${RD1_WANDB_ENTITY:-}"

FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
REPO_WORKDIR="${REPO_WORKDIR:-${FLASHNH_BASE}/repos/flash-nh/US_data/data_download/Disk_volume_estimation}"
# RD1_PROJECT_ROOT anchors the project-local artifact boundary check below.
# It defaults to REPO_WORKDIR, preserving today's behavior exactly. It is
# only ever set independently of REPO_WORKDIR for an authorized
# dual-provenance diagnostic, where REPO_WORKDIR must point at a separate
# pinned-commit payload checkout while RD1_CHAIN_DIR must still resolve
# beneath the real project's .scratch_local/ (not the payload checkout's).
RD1_PROJECT_ROOT="${RD1_PROJECT_ROOT:-${REPO_WORKDIR}}"
CANONICAL_PYTHON="${CANONICAL_PYTHON:-${FLASHNH_BASE}/envs/flashnh-moriah/bin/python}"
test -d "${REPO_WORKDIR}" || { echo "MISSING repo work directory" >&2; exit 1; }
cd "${REPO_WORKDIR}"
test -x "${CANONICAL_PYTHON}" || { echo "MISSING canonical interpreter" >&2; exit 1; }
export PATH="$(dirname "${CANONICAL_PYTHON}"):${PATH}"

# Fail-closed project-local artifact boundary check (CLAUDE.md, non-
# negotiable): RD1_CHAIN_DIR is where this job writes every manifest,
# registry backup, and (see RD1_HARNESS_DIR below) reads the deployed
# operational harness from. It must resolve strictly beneath the project
# root's own ".scratch_local/" -- never the broader Flash-NH base directory,
# and never anywhere outside the project root -- checked here, before any
# directory is created and before any Python driver, manifest, or W&B
# action runs. `realpath -m` normalizes without requiring the path to exist
# yet (RD1_CHAIN_DIR itself is created by mkdir -p below). The boundary is
# anchored to RD1_PROJECT_ROOT, not REPO_WORKDIR directly, so that an
# authorized dual-provenance diagnostic can point REPO_WORKDIR at a separate
# pinned-commit payload checkout without weakening this check.
RD1_CHAIN_DIR_REAL="$(realpath -m "${RD1_CHAIN_DIR}")"
RD1_PROJECT_ROOT_REAL="$(realpath -m "${RD1_PROJECT_ROOT}")"
case "${RD1_CHAIN_DIR_REAL}/" in
    "${RD1_PROJECT_ROOT_REAL}/.scratch_local/"*) ;;
    *)
        echo "FATAL: RD1_CHAIN_DIR must resolve beneath ${RD1_PROJECT_ROOT_REAL}/.scratch_local/ (got ${RD1_CHAIN_DIR_REAL})" >&2
        exit 1
        ;;
esac

echo "=== RD1 P20-P24 fixed chain: proposal ${RD1_PROPOSAL_ORDER} ==="
date -u
hostname
echo "job_id=${SLURM_JOB_ID:-unknown} node=${SLURMD_NODENAME:-unknown}"

# Defense in depth (shell level, before the Python driver even starts):
# strip every inherited WANDB_* variable. The structural fix is "no job
# submits another job", so nothing here should be inheriting a stale
# wandb-core WANDB_SERVICE handle in the first place; this guards against a
# future submission path accidentally forwarding one anyway.
for _v in $(compgen -v WANDB_ 2>/dev/null || true) WANDB; do
    unset "${_v}" || true
done

BACKUP_DIR="${RD1_CHAIN_DIR}/registry_backups"
MANIFEST_DIR="${RD1_CHAIN_DIR}/manifests"
mkdir -p "${BACKUP_DIR}" "${MANIFEST_DIR}"

# The operational harness (this driver + its src/baseline modules) is never
# read from the tracked REPO_WORKDIR checkout at runtime -- it is deployed,
# checksum-verified against its tracked git source, into
# ${RD1_CHAIN_DIR}/harness/ by scripts/deploy_rd1_p20_p24_harness.py before
# this job is submitted. Referencing it by this explicit, RD1_CHAIN_DIR-
# derived absolute path (rather than a cwd-relative "scripts/...") means
# this resolution is correct regardless of the caller's current directory,
# and regardless of whether Slurm's job-script spooling preserves this
# file's own on-disk location.
RD1_HARNESS_DIR="${RD1_CHAIN_DIR}/harness"
JOB_DRIVER=("${RD1_HARNESS_DIR}/scripts/rd1_p20_p24_job.py")
test -f "${JOB_DRIVER[0]}" || {
    echo "MISSING deployed harness job driver (run scripts/deploy_rd1_p20_p24_harness.py first): ${JOB_DRIVER[0]}" >&2
    exit 1
}

# Invocation-time deployed-harness integrity verification (review fix,
# 2026-09-27). scripts/deploy_rd1_p20_p24_harness.py already checksums every
# deployed file against its tracked git source AT DEPLOY TIME and writes
# ${RD1_CHAIN_DIR}/deploy_receipt.json (the chain-dir root, not inside
# harness/) -- but nothing previously re-checked
# that the deployed bundle on disk still matches that receipt AT JOB-RUN
# TIME, before this script hands control to it. A deployed file could have
# been overwritten (accidentally or otherwise) any time between deployment
# and this job actually running.
#
# This check must not trust the deployed harness itself, so it is inlined
# here rather than shipped as a deployed/importable module, and it uses only
# ${CANONICAL_PYTHON}'s standard library -- no "import" of anything under
# RD1_HARNESS_DIR. It fails closed (nonzero exit under `set -e` above aborts
# this script) before the JOB_DRIVER is ever executed, on: a missing or
# malformed receipt; a frozen_scientific_commit that isn't exactly the one
# pinned scientific identity; a chain_dir recorded in the receipt that
# doesn't match this job's own resolved RD1_CHAIN_DIR; any required deployed
# file missing on disk; or any required deployed file's on-disk SHA-256
# differing from the receipt. The required-file list mirrors
# scripts/deploy_rd1_p20_p24_harness.py's DEPLOY_MANIFEST exactly (deployed
# submitter, deployed sbatch script, deployed Python driver, and every
# src/baseline module + package marker the driver imports).
echo "--- verifying deployed harness bundle against deploy_receipt.json ---"
"${CANONICAL_PYTHON}" - "${RD1_CHAIN_DIR}/deploy_receipt.json" "${RD1_CHAIN_DIR_REAL}" "${RD1_HARNESS_DIR}" <<'PYEOF'
import hashlib
import json
import sys
from pathlib import Path

REQUIRED_PATHS = (
    "scripts/submit_rd1_p20_p24_chain.sh",
    "scripts/rd1_p20_p24_job.sbatch",
    "scripts/rd1_p20_p24_job.py",
    "src/__init__.py",
    "src/baseline/__init__.py",
    "src/baseline/rd1_p20_p24_env.py",
    "src/baseline/rd1_p20_p24_registry.py",
    "src/baseline/rd1_p20_p24_retry.py",
    "src/baseline/sweep_v2_six_axis_wandb_bridge_manifest.py",
    "src/baseline/sweep_v1_launch_manifest.py",
    "src/baseline/sweep_v2_six_axis_campaign.py",
    "src/baseline/sweep_v1_campaign.py",
)
FROZEN_SCIENTIFIC_COMMIT = "dabd2ca851bd2b3a03035886cfa50015f2c864b4"


def fail(msg: str) -> None:
    print(f"HARNESS_VERIFY_FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


receipt_path_arg, chain_dir_real_arg, harness_dir_arg = sys.argv[1], sys.argv[2], sys.argv[3]

receipt_file = Path(receipt_path_arg)
if not receipt_file.is_file():
    fail(f"receipt missing: {receipt_path_arg}")

try:
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
except Exception as exc:  # noqa: BLE001 - any parse failure is a fail-closed refusal
    fail(f"receipt malformed (not valid JSON): {exc}")

if not isinstance(receipt, dict):
    fail("receipt malformed (top level is not a JSON object)")

if receipt.get("frozen_scientific_commit") != FROZEN_SCIENTIFIC_COMMIT:
    fail(
        "frozen_scientific_commit mismatch: receipt="
        f"{receipt.get('frozen_scientific_commit')!r} expected={FROZEN_SCIENTIFIC_COMMIT!r}"
    )

if receipt.get("chain_dir") != chain_dir_real_arg:
    fail(f"chain_dir mismatch: receipt={receipt.get('chain_dir')!r} resolved={chain_dir_real_arg!r}")

files = receipt.get("files")
if not isinstance(files, list):
    fail("receipt malformed ('files' is not a list)")

by_path = {}
for entry in files:
    if not isinstance(entry, dict) or "path" not in entry or "sha256" not in entry:
        fail(f"receipt malformed (bad file entry): {entry!r}")
    by_path[entry["path"]] = entry

missing_from_receipt = [p for p in REQUIRED_PATHS if p not in by_path]
if missing_from_receipt:
    fail(f"receipt does not cover all required runtime files: {missing_from_receipt}")

harness_root = Path(harness_dir_arg)
for rel_path in REQUIRED_PATHS:
    expected_sha256 = by_path[rel_path]["sha256"]
    deployed_path = harness_root / Path(rel_path)
    if not deployed_path.is_file():
        fail(f"deployed file missing on disk: {rel_path} ({deployed_path})")
    actual_sha256 = hashlib.sha256(deployed_path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        fail(f"checksum mismatch for {rel_path}: receipt={expected_sha256} on_disk={actual_sha256}")

print(f"HARNESS_VERIFY_OK files_verified={len(REQUIRED_PATHS)}")
PYEOF

echo "--- preflight ---"
"${CANONICAL_PYTHON}" "${JOB_DRIVER[@]}" preflight \
    --order "${RD1_PROPOSAL_ORDER}" \
    --registry-path "${RD1_REGISTRY_PATH}" \
    --lock-path "${RD1_LOCK_PATH}"

if [ "${RD1_PROPOSAL_ORDER}" = "20" ]; then
    test -n "${RD1_P20_PINNED_MANIFEST_PATH}" || { echo "FATAL: P20 requires RD1_P20_PINNED_MANIFEST_PATH" >&2; exit 1; }
    test -n "${RD1_P20_PINNED_MANIFEST_SHA256}" || { echo "FATAL: P20 requires RD1_P20_PINNED_MANIFEST_SHA256" >&2; exit 1; }
    echo "--- P20: verifying pinned manifest is reused byte-for-byte ---"
    "${CANONICAL_PYTHON}" "${JOB_DRIVER[@]}" verify-pinned-manifest \
        --manifest-path "${RD1_P20_PINNED_MANIFEST_PATH}" \
        --expected-sha256 "${RD1_P20_PINNED_MANIFEST_SHA256}"
    MANIFEST_PATH="${RD1_P20_PINNED_MANIFEST_PATH}"
else
    echo "--- P${RD1_PROPOSAL_ORDER}: building this proposal's own manifest (predecessor already passed via afterok) ---"
    MANIFEST_PATH="${MANIFEST_DIR}/rd1_p${RD1_PROPOSAL_ORDER}_manifest.json"
    BUILD_MANIFEST_ARGS=(
        --manifest-path "${MANIFEST_PATH}"
        --wandb-sweep-id "${RD1_WANDB_SWEEP_ID}"
        --expected-commit "${RD1_EXPECTED_COMMIT}"
        --proposal-order "${RD1_PROPOSAL_ORDER}"
        --execution-generation "${RD1_EXECUTION_GENERATION}"
        --output-root-base "${RD1_OUTPUT_ROOT_BASE}"
    )
    if [ -n "${RD1_WANDB_PROJECT}" ]; then
        BUILD_MANIFEST_ARGS+=(--wandb-project "${RD1_WANDB_PROJECT}")
    fi
    if [ -n "${RD1_WANDB_ENTITY}" ]; then
        BUILD_MANIFEST_ARGS+=(--wandb-entity "${RD1_WANDB_ENTITY}")
    fi
    "${CANONICAL_PYTHON}" scripts/create_sweep_v2_six_axis_wandb_bridge_production_sweep.py build-manifest \
        "${BUILD_MANIFEST_ARGS[@]}"
fi

echo "--- validating manifest before any W&B contact ---"
"${CANONICAL_PYTHON}" scripts/create_sweep_v2_six_axis_wandb_bridge_production_sweep.py validate-launch \
    --manifest-path "${MANIFEST_PATH}" \
    --expect-sweep-id "${RD1_WANDB_SWEEP_ID}" \
    --emit env >/dev/null

if [ "${RD1_DRY_RUN}" = "--dry-run" ]; then
    echo "DRY_RUN: stopping before any W&B contact (preflight + manifest identity/validation only)"
    exit 0
fi

echo "--- sanitize-env-probe (defense in depth, must find zero WANDB_* vars) ---"
"${CANONICAL_PYTHON}" "${JOB_DRIVER[@]}" sanitize-env-probe

AGENT_LAUNCHER="${REPO_WORKDIR}/scripts/run_sweep_v2_six_axis_wandb_agent_moriah.sbatch"
test -f "${AGENT_LAUNCHER}" || { echo "MISSING agent launcher: ${AGENT_LAUNCHER}" >&2; exit 1; }

echo "--- running the agent attempt(s): retry decision is entirely inside the tested Python driver ---"
# Project-local temporary-file containment (review fix, 2026-09-27): this
# previously used a bare `mktemp`, which falls back to the OS default
# temporary directory (a machine-global path, never validated against the
# project-local artifact boundary above). RD1_CHAIN_DIR was already
# validated to resolve beneath this project's own .scratch_local/ earlier in
# this script -- RD1_TMP_DIR reuses that same validated root, never a
# default temp location.
RD1_TMP_DIR="${RD1_CHAIN_DIR}/tmp"
mkdir -p "${RD1_TMP_DIR}"
# Persistent (never trap-cleaned, unlike RD1_TMP_DIR) per-attempt evidence:
# this is the root-cause-#1 fix -- every agent-launcher attempt's redacted
# stdout/stderr and outcome metadata is written here regardless of exit
# code, so a launcher that exits 0 while the W&B run it dispatched had
# already crashed still leaves inspectable evidence behind.
RD1_ATTEMPTS_DIR="${RD1_CHAIN_DIR}/agent_attempts"
mkdir -p "${RD1_ATTEMPTS_DIR}"
RESULT_FILE="$(mktemp "${RD1_TMP_DIR}/rd1_p${RD1_PROPOSAL_ORDER}_agent_result.XXXXXX")"
trap 'rm -f "${RESULT_FILE}"' EXIT
set +e
"${CANONICAL_PYTHON}" "${JOB_DRIVER[@]}" run-agent-with-retry \
    --agent-launcher-path "${AGENT_LAUNCHER}" \
    --manifest-path "${MANIFEST_PATH}" \
    --wandb-sweep-id "${RD1_WANDB_SWEEP_ID}" \
    --path "${PATH}" \
    --home "${HOME}" \
    --repo-workdir "${REPO_WORKDIR}" \
    --max-attempts 2 \
    --attempts-dir "${RD1_ATTEMPTS_DIR}" \
    > "${RESULT_FILE}" 2>&1
AGENT_RC=$?
cat "${RESULT_FILE}"
set -e

test "${AGENT_RC}" -eq 0 || { echo "FATAL: run-agent-with-retry did not succeed (rc=${AGENT_RC})" >&2; exit 1; }

RUN_ID="$(grep -oE 'AGENT_OK run_id=.*' "${RESULT_FILE}" | sed 's/^AGENT_OK run_id=//')"
test -n "${RUN_ID}" || { echo "FATAL: run-agent-with-retry exited 0 but printed no run_id" >&2; exit 1; }
echo "resolved run_id=${RUN_ID}"

echo "--- verified registry append ---"
"${CANONICAL_PYTHON}" "${JOB_DRIVER[@]}" append-registry-row \
    --order "${RD1_PROPOSAL_ORDER}" \
    --run-id "${RUN_ID}" \
    --recorded-at-utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --registry-path "${RD1_REGISTRY_PATH}" \
    --lock-path "${RD1_LOCK_PATH}" \
    --backup-dir "${BACKUP_DIR}" \
    --manifest-path "${MANIFEST_PATH}" \
    --wandb-sweep-id "${RD1_WANDB_SWEEP_ID}"

echo "=== P${RD1_PROPOSAL_ORDER} complete: run_id=${RUN_ID} ==="

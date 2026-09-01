#!/usr/bin/env bash
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=flashnh-sweep-v2-six-axis-random-control
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/sweep-v2-six-axis-random-control-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/sweep-v2-six-axis-random-control-%j.err
#
# One Slurm allocation runs exactly one Sweep-v2 six-axis random-control row.
# This launcher is never invoked by local tests and performs no loop over
# scientific candidates -- it mirrors run_sweep_v2_six_axis_wandb_agent_moriah.sbatch's
# one-allocation-per-trial rule for the Bayesian arm. This path never runs a
# live W&B search and never contacts the production Bayesian controller:
# random-control rows are a fixed committed manifest, frozen after Bayesian
# observation 1.
#
# It reuses exactly the Sweep-v1 production launcher's runtime-provenance
# conventions (scripts/run_sweep_v1_exact_retry_moriah.sbatch): the canonical
# Moriah Flash-NH checkout and interpreter, an explicit expected commit, and
# delegation of the authoritative HEAD / clean-tracked-tree / interpreter-
# identity pin to the shared src.baseline.sweep_v1_runtime_contract module
# (called inside the trial runner before any execution). No second provenance
# framework is introduced here, and the untracked research tree is never
# required clean.
#
# PRODUCTION PINS (a submitter cannot weaken these):
#   * FLASHNH_BASE / REPO_WORKDIR / CANONICAL_PYTHON are assigned directly to
#     their canonical Moriah values with NO environment override or fallback --
#     the submission environment cannot redirect the checkout or the interpreter.
#   * The runner's offline self-test hook (FLASHNH_SWEEP_V2_RANDOM_CONTROL_SELFTEST)
#     is refused nonzero if inherited, then unset, so a real allocation can
#     never exit 0 without preparation + execution + interpretation. Invoking
#     the runner directly outside this sbatch keeps the self-test available.
#   * The three scientific policy paths (baseline, v2 six-axis overlay, pilot)
#     are resolved to their canonical committed locations and passed
#     explicitly rather than left to runner CLI defaults.
#
# Intended execution shape: three waves of four concurrent independent jobs.
#   wave 1: ROW_INDEX in {0,1,2,3}
#   wave 2: ROW_INDEX in {4,5,6,7}
#   wave 3: ROW_INDEX in {8,9,10,11}
# Every job writes to ${OUTPUT_ROOT}/<trial_id>, and each committed row has a
# distinct configuration_id -> distinct trial_id, so four concurrent jobs
# sharing one OUTPUT_ROOT never collide.
#
# Required environment:
#   ROW_INDEX        0-based committed manifest row index (exactly one)
#   OUTPUT_ROOT      output root for this trial (writes to OUTPUT_ROOT/<trial_id>)
#   EXPECTED_COMMIT  full 40-hex git commit this launch is pinned to
# Optional environment:
#   EXECUTION_GENERATION       infrastructure-retry generation (default 1)
#   RETRY_OF_TRIAL_ID          frozen attempt001 trial_id (required when
#                              EXECUTION_GENERATION > 1)
#   PRIOR_ATTEMPT_GENERATION   one previously reserved generation for this row
#   MANIFEST_PATH / PACKAGE_ROOT / SCREENING_BASIN_IDS / FIXED_SUPPORT_CONTRACT_PATH
#                              operational input locations on the Moriah data tree
set -euo pipefail

: "${ROW_INDEX:?set the 0-based random-control manifest row index}"
: "${OUTPUT_ROOT:?set the output root for this trial}"
: "${EXPECTED_COMMIT:?set the full 40-hex git commit this launch is pinned to}"

if ! [[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "EXPECTED_COMMIT is not a full 40-hex-character commit id: ${EXPECTED_COMMIT}" >&2
    exit 1
fi

# A real production allocation must never inherit the runner's offline
# self-test hook: an inherited value makes
# run_sweep_v2_six_axis_random_control_trial.py print its resolved lineage and
# exit 0 BEFORE the runtime-contract pin, preparation, execution, or
# interpretation. Refuse visibly, then unset for defence in depth. The runner
# keeps the hook available when invoked directly outside this sbatch.
if [ -n "${FLASHNH_SWEEP_V2_RANDOM_CONTROL_SELFTEST:-}" ]; then
    echo "REFUSING: FLASHNH_SWEEP_V2_RANDOM_CONTROL_SELFTEST is set (${FLASHNH_SWEEP_V2_RANDOM_CONTROL_SELFTEST}); a real production launch must not inherit the runner's offline self-test hook" >&2
    exit 1
fi
unset FLASHNH_SWEEP_V2_RANDOM_CONTROL_SELFTEST || true

# Canonical Moriah Flash-NH paths -- pinned directly, identical values and
# formatting to scripts/run_sweep_v1_exact_retry_moriah.sbatch, the v2 Bayesian
# agent launcher, and docs/remote_operations.md. NOT overridable through the
# submission environment: no ${VAR:-...} fallback, no alternate location.
FLASHNH_BASE=/sci/labs/efratmorin/omripo/Flash-NH
REPO_WORKDIR="${FLASHNH_BASE}/repos/flash-nh/US_data/data_download/Disk_volume_estimation"
CANONICAL_PYTHON="${FLASHNH_BASE}/envs/flashnh-moriah/bin/python"

# The three scientific policy paths -- resolved to the canonical committed
# locations that are also the runner CLI defaults and
# src/baseline/sweep_v2_six_axis_wandb_bridge_registration.py's _REAL_* constants
# -- passed explicitly below rather than left to runner defaults.
BASELINE_POLICY_PATH="${REPO_WORKDIR}/config/stage1_scientific_baseline_v001.yaml"
POLICY_OVERLAY_PATH="${REPO_WORKDIR}/config/stage1_scientific_baseline_v2_six_axis_overlay_v001.yaml"
BASE_PILOT_POLICY_PATH="${REPO_WORKDIR}/config/stage1_lead06_pilot_v001.yaml"

# Operational input locations. The manifest is the committed scientific
# artifact; the package / screening / fixed-support locations are Moriah data
# tree paths that may legitimately differ between deployments.
MANIFEST_PATH="${MANIFEST_PATH:-${REPO_WORKDIR}/config/stage1_phase_b_sweep_v2_six_axis_random_control_v001_random_control_manifest.json}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${FLASHNH_BASE}/data/stage1_scientific_package_v002}"
SCREENING_BASIN_IDS="${SCREENING_BASIN_IDS:-${FLASHNH_BASE}/data/screening_subsets/stage1_provisional_operational_screening_subset_v001/screening_subset_basin_ids.txt}"
FIXED_SUPPORT_CONTRACT_PATH="${FIXED_SUPPORT_CONTRACT_PATH:-${FLASHNH_BASE}/data/fixed_support_contracts/stage1_v2_common120_fixed_support_contract_v001.json}"
EXECUTION_GENERATION="${EXECUTION_GENERATION:-1}"

echo "=== Sweep-v2 six-axis random-control launch ==="
hostname
date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "ROW_INDEX: ${ROW_INDEX}"
echo "EXECUTION_GENERATION: ${EXECUTION_GENERATION}"
echo "EXPECTED_COMMIT: ${EXPECTED_COMMIT}"
echo "REPO_WORKDIR: ${REPO_WORKDIR}"
echo "CANONICAL_PYTHON: ${CANONICAL_PYTHON}"
echo "MANIFEST_PATH: ${MANIFEST_PATH}"
echo "BASELINE_POLICY_PATH: ${BASELINE_POLICY_PATH}"
echo "POLICY_OVERLAY_PATH: ${POLICY_OVERLAY_PATH}"
echo "BASE_PILOT_POLICY_PATH: ${BASE_PILOT_POLICY_PATH}"

# Cheap, fast-failing shell-level existence checks only. The AUTHORITATIVE
# commit / dirty-tracked-tree / interpreter-identity pin is performed once, in
# Python, by the shared src.baseline.sweep_v1_runtime_contract module
# (verify_commit_and_interpreter, called from inside the trial runner before
# any trial execution) -- never duplicated or re-implemented here in shell.
# That check inspects TRACKED files only, so unrelated untracked research
# artifacts are preserved and never block the launch.
test -d "${REPO_WORKDIR}" || { echo "MISSING expected repo work directory: ${REPO_WORKDIR}" >&2; exit 1; }
cd "${REPO_WORKDIR}"
test -x "${CANONICAL_PYTHON}" || { echo "MISSING canonical runtime interpreter: ${CANONICAL_PYTHON}" >&2; exit 1; }
test -f "${MANIFEST_PATH}" || { echo "MISSING random-control manifest: ${MANIFEST_PATH}" >&2; exit 1; }
test -f "${BASELINE_POLICY_PATH}" || { echo "MISSING baseline policy: ${BASELINE_POLICY_PATH}" >&2; exit 1; }
test -f "${POLICY_OVERLAY_PATH}" || { echo "MISSING v2 six-axis overlay policy: ${POLICY_OVERLAY_PATH}" >&2; exit 1; }
test -f "${BASE_PILOT_POLICY_PATH}" || { echo "MISSING pilot policy: ${BASE_PILOT_POLICY_PATH}" >&2; exit 1; }

RETRY_ARGS=()
if [ -n "${RETRY_OF_TRIAL_ID:-}" ]; then
    RETRY_ARGS+=(--retry-of-trial-id "${RETRY_OF_TRIAL_ID}")
fi
if [ -n "${PRIOR_ATTEMPT_GENERATION:-}" ]; then
    RETRY_ARGS+=(--prior-attempt-generation "${PRIOR_ATTEMPT_GENERATION}")
fi

echo "=== launching scripts/run_sweep_v2_six_axis_random_control_trial.py ==="
exec "${CANONICAL_PYTHON}" scripts/run_sweep_v2_six_axis_random_control_trial.py \
    --expected-commit "${EXPECTED_COMMIT}" \
    --expected-runtime-python "${CANONICAL_PYTHON}" \
    --manifest-path "${MANIFEST_PATH}" \
    --row-index "${ROW_INDEX}" \
    --package-root "${PACKAGE_ROOT}" \
    --screening-basin-ids "${SCREENING_BASIN_IDS}" \
    --fixed-support-contract-path "${FIXED_SUPPORT_CONTRACT_PATH}" \
    --baseline-policy-path "${BASELINE_POLICY_PATH}" \
    --policy-overlay-path "${POLICY_OVERLAY_PATH}" \
    --base-pilot-policy-path "${BASE_PILOT_POLICY_PATH}" \
    --output-root "${OUTPUT_ROOT}" \
    --execution-generation "${EXECUTION_GENERATION}" \
    ${RETRY_ARGS[@]+"${RETRY_ARGS[@]}"}

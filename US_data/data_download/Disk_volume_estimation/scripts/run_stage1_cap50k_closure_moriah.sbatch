#!/usr/bin/env bash
#SBATCH --job-name=flashnh-cap50k-closure
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/cap50k-closure-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/cap50k-closure-%j.err
#
# Seed-A 50k embedding-structure closure comparison: Slurm launcher for
# EXACTLY ONE of the two approved closure run_ids --
# emb128x64_seedA_cap_low_cal (incumbent continuation, [128,64], already
# trained through epoch 6) or emb128x32_seedA_cap_low_cal (challenger start,
# [128,32], fresh from original Seed-A initialization). Both are
# max_updates_per_epoch=50000 capped-update calibration candidates -- NOT
# members of the closed six-run matrix in
# config/stage1_lead06_pilot_v001.yaml, and this script never broadens the
# two-run_id allowlist below to any other run_id (real pilot run_id,
# out-of-scope raw-pathway calibration candidate, or 25k-tier candidate).
#
# This is a close structural mirror of the canonical, already-reviewed
# scripts/run_stage1_lead06_pilot_moriah.sbatch (same resource defaults,
# restart-safety design, PREPARE_ONLY threading, TERM-forwarding, and
# authoritative-evidence-file status classification -- see that script's
# header comments for the full rationale, not repeated here). The only real
# differences are (1) the RUN_ID allowlist below is the two closure run_ids
# instead of the six pilot run_ids, (2) it delegates to
# scripts/run_stage1_cap50k_closure.py instead of
# scripts/run_stage1_lead06_pilot.py, (3) a required EXPECTED_COMMIT
# commit-pin (see below), (4) an additional STATUS_ONLY read-only mode, (5)
# training is fixed at CLOSURE_MAX_TARGET_EPOCH=12 (this comparison's
# approved horizon, non-overridable -- no MAX_TARGET_EPOCH knob here), and
# (6) WANDB_MODE=offline is exported unconditionally, never gated behind a
# WANDB_POLICY_PATH override.
#
# Before this script existed, this two-candidate phase was only launchable
# via two untracked, near-duplicate Moriah-local scripts under
# tmp/operations/ (run_pilot_cap_calibration.py's stale, uncommitted,
# EXPECTED_HEAD-checking .sbatch, and
# run_prepare_only_emb128x32_cap_low_cal.sbatch), neither reproducible from
# a clean clone.
#
# Usage:
#   EXPECTED_COMMIT=<FINAL_AUTHORIZED_COMMIT> sbatch --export=ALL,EXPECTED_COMMIT \
#       scripts/run_stage1_cap50k_closure_moriah.sbatch emb128x64_seedA_cap_low_cal
#   EXPECTED_COMMIT=<FINAL_AUTHORIZED_COMMIT> sbatch --export=ALL,EXPECTED_COMMIT \
#       scripts/run_stage1_cap50k_closure_moriah.sbatch emb128x32_seedA_cap_low_cal
#
# --- Commit-pin design (why this differs from the canonical pilot script) ---
# The canonical pilot script (and the untracked calibration .sbatch it was
# copied from) only *report* SOURCE_COMMIT and block on a dirty tree; neither
# refuses a clean-but-wrong commit. The untracked calibration script went
# further and hardcoded a literal EXPECTED_HEAD string -- but a hash
# hardcoded inside a TRACKED file is self-referential (the act of updating
# EXPECTED_HEAD to the new HEAD changes the tree, which changes the commit
# hash the new EXPECTED_HEAD would need to name) and, in practice, goes
# stale the moment any later, unrelated commit lands (exactly what happened:
# the discovered copy's EXPECTED_HEAD no longer matched current HEAD).
#
# This script instead requires EXPECTED_COMMIT as an explicit, required
# submission-time input (env var, no default) -- never a value baked into
# any tracked file, never "whatever HEAD happens to be right now". The
# submitter must name the exact commit they reviewed and are authorizing;
# that value is echoed verbatim into both the console log and the evidence
# JSON below, so the authorized commit is visible in the launch command
# itself and in every downstream evidence artifact, not just in the repo's
# own (dirty-tree-only) self-report. Training refuses to start if actual
# repo HEAD (computed fresh from the checked-out clone, after the existing
# dirty-tree guard) does not equal EXPECTED_COMMIT exactly -- a stale,
# mistyped, or accidentally-omitted value fails closed, before any
# module/CUDA/config-generation work happens. No reset/checkout/pull is ever
# performed by this script to "fix" a mismatch -- a mismatch is always the
# submitter's to resolve (fix the clone, or pass the right value), never
# this script's to silently correct.
set -uo pipefail

RUN_ID="${RUN_ID:-${1:-}}"
case "${RUN_ID}" in
    emb128x64_seedA_cap_low_cal|emb128x32_seedA_cap_low_cal) ;;
    *)
        echo "USAGE: sbatch $0 <run_id>   (or set RUN_ID via --export)" >&2
        echo "run_id must be one of: emb128x64_seedA_cap_low_cal emb128x32_seedA_cap_low_cal" >&2
        exit 2
        ;;
esac

# Required, explicit commit-pin -- see design note above. No default: an
# unset/empty value fails fast, before any module load or GPU allocation is
# spent, rather than silently proceeding against whatever HEAD happens to be
# checked out.
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"
if [ -z "${EXPECTED_COMMIT}" ]; then
    echo "EXPECTED_COMMIT is required (the exact commit hash you reviewed and are authorizing)." >&2
    echo "Set it explicitly, e.g.: EXPECTED_COMMIT=<hash> sbatch --export=ALL,EXPECTED_COMMIT $0 ${RUN_ID}" >&2
    exit 2
fi

# Approved closure horizon: fixed at epoch 12, enforced Python-side
# (CLOSURE_MAX_TARGET_EPOCH in run_stage1_cap50k_closure.py) -- there is
# deliberately no MAX_TARGET_EPOCH env var or --max-target-epoch override
# here; a caller cannot bound training at any other epoch.
CLOSURE_MAX_TARGET_EPOCH=12

# W&B must never go online for this closure comparison. Exported
# unconditionally (never `${WANDB_MODE:-offline}`) so a caller-provided
# WANDB_MODE (e.g. inherited via --export=ALL) is overwritten before any
# Python process -- and therefore before wandb itself -- ever sees it.
export WANDB_MODE=offline

FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
ENV_PREFIX="${FLASHNH_BASE}/envs/flashnh-moriah"

PILOT_REPO_CLONE_DIR="${PILOT_REPO_CLONE_DIR:-/sci/labs/efratmorin/omripo/PhD/Python/neuralhydrology}"
REPO_CLONE_DIR="${PILOT_REPO_CLONE_DIR}"
REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"

PACKAGE_ROOT="${PILOT_PACKAGE_ROOT:-${FLASHNH_BASE}/data/stage1_scientific_package_v002}"
CONFIG_OUT_DIR="${PILOT_CONFIG_OUT_DIR:-${FLASHNH_BASE}/runs/stage1_lead06_pilot_${RUN_ID}_v001}"
EVIDENCE_OUT_DIR="${PILOT_EVIDENCE_OUT_DIR:-${FLASHNH_BASE}/evidence/stage1_lead06_pilot_${RUN_ID}_v001}"

# Optional per-run W&B policy override -- identical mechanism to the
# canonical pilot script.
WANDB_POLICY_PATH="${WANDB_POLICY_PATH:-}"
if [ -n "${WANDB_POLICY_PATH}" ]; then
    WANDB_DIR="${WANDB_DIR:-${FLASHNH_BASE}/wandb/offline/${RUN_ID}}"
    mkdir -p "${WANDB_DIR}"
fi
# WANDB_MODE is already forced to offline above, unconditionally -- retained
# here only as the per-run WANDB_DIR override, never a second mode export.
TRACKING_GENERATION="${TRACKING_GENERATION:-g1}"

# PREPARE_ONLY: generates this run_id's NH config + generation manifest and
# exits before any NH training call or W&B backend init (only ever
# appropriate for the challenger's first invocation -- prepare_pilot_run_only
# refuses if a real NH run directory already exists, which the incumbent
# already has). STATUS_ONLY: read-only continuation-status snapshot, no
# config generation, no training, no W&B, appropriate for either run_id at
# any time (this is the dry-run verification mode -- see
# scripts/run_stage1_cap50k_closure.py's --status-only). Both skip the
# nvidia-smi/CUDA check below since neither imports torch/neuralhydrology.
PREPARE_ONLY="${PREPARE_ONLY:-0}"
STATUS_ONLY="${STATUS_ONLY:-0}"
if [ "${PREPARE_ONLY}" = "1" ] && [ "${STATUS_ONLY}" = "1" ]; then
    echo "PREPARE_ONLY=1 and STATUS_ONLY=1 are mutually exclusive." >&2
    exit 2
fi

RESULT_DIR="${FLASHNH_BASE}/evidence/cap50k_closure_${RUN_ID}_${SLURM_JOB_ID}"
mkdir -p "${RESULT_DIR}" "$(dirname "${CONFIG_OUT_DIR}")" "$(dirname "${EVIDENCE_OUT_DIR}")"
RESULT_LOG="${RESULT_DIR}/closure_human.log"
exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== Seed-A 50k embedding-structure closure: run_id=${RUN_ID} ==="
hostname
date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-unknown}"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unknown}"
echo "EXPECTED_COMMIT (authorized by submitter): ${EXPECTED_COMMIT}"
echo "Approved closure horizon (fixed, non-overridable): epoch ${CLOSURE_MAX_TARGET_EPOCH}"
echo "WANDB_MODE (effective, enforced offline, non-overridable): ${WANDB_MODE}"
scontrol show job "${SLURM_JOB_ID}" || true

if [ "${STATUS_ONLY}" != "1" ]; then
    test -d "${PACKAGE_ROOT}" || { echo "MISSING transferred package: ${PACKAGE_ROOT}"; exit 1; }
fi

if ! type module &>/dev/null; then
    for _modinit in \
        /etc/profile.d/huji-lmod.sh \
        /etc/profile.d/modules.sh \
        /usr/share/lmod/lmod/init/bash \
        /etc/profile.d/lmod.sh
    do
        [ -f "${_modinit}" ] && { source "${_modinit}"; break; }
    done
    unset _modinit
fi

module load spack/all
module load miniconda3/24.3.0-gcc-iqeknet
module load nvidia/580.95.05
module load cuda/12.8.1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
which python

if [ "${PREPARE_ONLY}" = "1" ] || [ "${STATUS_ONLY}" = "1" ]; then
    echo "PREPARE_ONLY=${PREPARE_ONLY} STATUS_ONLY=${STATUS_ONLY}: no NH training call, no W&B backend"
    echo "init, and no GPU/CUDA required; skipping the nvidia-smi/CUDA check."
else
    nvidia-smi
    python -u -c "
import torch
print('PyTorch:', torch.__version__)
cuda_ok = torch.cuda.is_available()
print('cuda available:', cuda_ok)
if not cuda_ok:
    raise SystemExit('CUDA not available -- check module loads and GPU allocation')
print('device:', torch.cuda.get_device_name(0))
"
fi

# Pre-launch repository guard (dirty-tree check, identical to the canonical
# pilot script) followed immediately by the commit-pin check (see design
# note above) -- both must pass before any config generation or training.
test -d "${REPO_WORKDIR}" || { echo "MISSING expected pilot repo work directory: ${REPO_WORKDIR}"; exit 1; }
cd "${REPO_WORKDIR}"
git log --oneline -3
SOURCE_COMMIT=$(git rev-parse HEAD)
GIT_STATUS_PORCELAIN=$(git status --porcelain)
echo "SOURCE_COMMIT: ${SOURCE_COMMIT}"
echo "--- git status --porcelain (${REPO_WORKDIR}) ---"
echo "${GIT_STATUS_PORCELAIN}"
echo "-------------------------------------------------"
TRACKED_CHANGES=$(echo "${GIT_STATUS_PORCELAIN}" | grep -v '^??' || true)
if [ -n "${TRACKED_CHANGES}" ]; then
    echo "REFUSING to launch: tracked files are locally modified in ${REPO_WORKDIR}:" >&2
    echo "${TRACKED_CHANGES}" >&2
    exit 1
fi
if [ "${SOURCE_COMMIT}" != "${EXPECTED_COMMIT}" ]; then
    echo "REFUSING to launch: repo HEAD (${SOURCE_COMMIT}) does not match the explicitly" >&2
    echo "authorized EXPECTED_COMMIT (${EXPECTED_COMMIT}). This is never auto-corrected --" >&2
    echo "fix the clone (checkout the authorized commit) or re-submit with the correct" >&2
    echo "EXPECTED_COMMIT for the commit actually checked out." >&2
    exit 1
fi
echo "Commit pin OK: HEAD matches EXPECTED_COMMIT (${EXPECTED_COMMIT})."

CLOSURE_CLI_ARGS=(
    --run-id "${RUN_ID}"
    --config-out-dir "${CONFIG_OUT_DIR}"
    --tracking-generation "${TRACKING_GENERATION}"
)
if [ "${STATUS_ONLY}" != "1" ]; then
    CLOSURE_CLI_ARGS+=(
        --package-root "${PACKAGE_ROOT}"
        --evidence-out-dir "${EVIDENCE_OUT_DIR}"
        --slurm-job-id "${SLURM_JOB_ID:-}"
        --slurm-partition "${SLURM_JOB_PARTITION:-}"
        --slurm-gres "${SLURM_JOB_GRES:-}"
    )
fi
if [ -n "${WANDB_POLICY_PATH}" ]; then
    CLOSURE_CLI_ARGS+=(--wandb-policy-path "${WANDB_POLICY_PATH}")
    export WANDB_DIR
    echo "W&B tracking override active: policy=${WANDB_POLICY_PATH} dir=${WANDB_DIR} mode=${WANDB_MODE}"
fi
if [ "${PREPARE_ONLY}" = "1" ]; then
    CLOSURE_CLI_ARGS+=(--prepare-only)
fi
if [ "${STATUS_ONLY}" = "1" ]; then
    CLOSURE_CLI_ARGS+=(--status-only)
fi

echo "=== launching scripts/run_stage1_cap50k_closure.py --run-id ${RUN_ID} ==="

if [ "${STATUS_ONLY}" = "1" ]; then
    # Status-only is a single, quick, synchronous, read-only call -- no
    # background PID/TERM-forwarding/authoritative-evidence-file
    # classification machinery is needed (nothing is written, nothing can be
    # interrupted mid-write). Exit code is this call's own.
    python scripts/run_stage1_cap50k_closure.py "${CLOSURE_CLI_ARGS[@]}" \
        | tee "${RESULT_DIR}/closure_status_stdout.json.log"
    exit "${PIPESTATUS[0]}"
fi

python scripts/run_stage1_cap50k_closure.py "${CLOSURE_CLI_ARGS[@]}" \
    > "${RESULT_DIR}/closure_stdout.json.log" &
PILOT_PID=$!

TERM_REQUESTED=0
_on_term() {
    if [ "${TERM_REQUESTED}" -eq 0 ]; then
        TERM_REQUESTED=1
        echo "=== termination signal received at $(date -u) -- forwarding TERM to pilot PID ${PILOT_PID} ==="
        if kill -0 "${PILOT_PID}" 2>/dev/null; then
            kill -TERM "${PILOT_PID}" 2>/dev/null || true
        fi
    fi
}
trap _on_term TERM INT

wait "${PILOT_PID}"
PILOT_STATUS=$?

echo "=== pilot process exit code: ${PILOT_STATUS} ==="

NH_RUN_DIR=""
if [ -s "${RESULT_DIR}/closure_stdout.json.log" ]; then
    NH_RUN_DIR=$(python -c "
import json
try:
    print(json.load(open('${RESULT_DIR}/closure_stdout.json.log'))['nh_run_dir'])
except Exception:
    pass
" 2>/dev/null)
fi
if [ -z "${NH_RUN_DIR}" ]; then
    NH_RUN_DIR=$(find "${CONFIG_OUT_DIR}/runs" -mindepth 1 -maxdepth 1 -type d -name "stage1_lead06_pilot_${RUN_ID}_v001_*" 2>/dev/null | sort | tail -1)
fi

LATEST_CKPT_AFTER=""
LATEST_EPOCH_AFTER=""
if [ -n "${NH_RUN_DIR}" ] && [ -d "${NH_RUN_DIR}" ]; then
    LATEST_CKPT_AFTER=$(find "${NH_RUN_DIR}" -name 'model_epoch*.pt' 2>/dev/null | \
        sed -E 's#^(.*model_epoch0*([0-9]+)\.pt)$#\2 \1#' | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "${LATEST_CKPT_AFTER}" ]; then
        LATEST_EPOCH_AFTER=$(basename "${LATEST_CKPT_AFTER}" | sed -E 's/model_epoch0*([0-9]+)\.pt/\1/')
    fi
fi

# Status classification: identical logic/rationale to the canonical pilot
# script's embedded block (see that script for the full postmortem history
# this design responds to) -- only the two comment-adjacent identifiers
# (result file directory names) differ. Authoritative source is always
# pilot_run_evidence.json / pilot_preparation_result.json under
# --evidence-out-dir, never parsed subprocess stdout.
RUN_ID="${RUN_ID}" NH_RUN_DIR="${NH_RUN_DIR}" \
PILOT_STATUS="${PILOT_STATUS}" PACKAGE_ROOT="${PACKAGE_ROOT}" \
CONFIG_OUT_DIR="${CONFIG_OUT_DIR}" EVIDENCE_OUT_DIR="${EVIDENCE_OUT_DIR}" \
LATEST_CKPT_AFTER="${LATEST_CKPT_AFTER}" LATEST_EPOCH_AFTER="${LATEST_EPOCH_AFTER}" \
TERM_REQUESTED="${TERM_REQUESTED}" SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION:-unknown}" SOURCE_COMMIT="${SOURCE_COMMIT}" \
EXPECTED_COMMIT_USED="${EXPECTED_COMMIT}" \
WANDB_POLICY_PATH_USED="${WANDB_POLICY_PATH}" WANDB_DIR_USED="${WANDB_DIR:-}" \
TRACKING_GENERATION_USED="${TRACKING_GENERATION}" \
CLOSURE_MAX_TARGET_EPOCH_USED="${CLOSURE_MAX_TARGET_EPOCH}" \
PREPARE_ONLY_USED="${PREPARE_ONLY}" \
RESULT_JSON_PATH="${RESULT_DIR}/closure_result.json" \
python -c "
import json
import os
import socket
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

env = os.environ
pilot_status = int(env['PILOT_STATUS'])
nh_run_dir = env['NH_RUN_DIR']
prepare_only_flag = env['PREPARE_ONLY_USED'] == '1'
evidence_out_dir = Path(env['EVIDENCE_OUT_DIR'])

authoritative_filename = 'pilot_preparation_result.json' if prepare_only_flag else 'pilot_run_evidence.json'
authoritative_path = evidence_out_dir / authoritative_filename
authoritative_result = {}
if not authoritative_path.is_file():
    authoritative_parse_status = 'absent_pilot_failed_before_evidence_creation' if pilot_status != 0 else 'absent'
else:
    try:
        with open(authoritative_path) as fh:
            authoritative_result = json.load(fh)
        authoritative_parse_status = 'parsed_successfully'
    except Exception as exc:
        authoritative_result = {}
        authoritative_parse_status = 'corrupt'
        print('AUTHORITATIVE_RESULT_PARSE_ERROR: ' + str(authoritative_path) + ': ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)

prepare_only_result = authoritative_result if (prepare_only_flag and authoritative_parse_status == 'parsed_successfully') else None

continuation_status = authoritative_result.get('continuation_status') or {}
status_fields = {
    'highest_physical_checkpoint_epoch': continuation_status.get('highest_physical_checkpoint_epoch'),
    'highest_screened_epoch': continuation_status.get('highest_screened_epoch'),
    'next_intended_screening_epoch': continuation_status.get('next_intended_screening_epoch'),
    'overshoot_epochs': continuation_status.get('overshoot_epochs'),
    'safe_to_continue_automatically': continuation_status.get('safe_to_continue_automatically'),
    'blocked_reason': None,
}
pilot_final_status = None if prepare_only_flag else authoritative_result.get('run_status')

physical_state_fallback_used = False
if authoritative_parse_status != 'parsed_successfully' and nh_run_dir and os.path.isdir(nh_run_dir):
    physical_state_fallback_used = True
    try:
        from src.baseline.pilot_orchestration import compute_pilot_status_fields
        fallback = compute_pilot_status_fields(nh_run_dir)
        status_fields.update({
            'highest_physical_checkpoint_epoch': fallback['highest_physical_checkpoint_epoch'],
            'highest_screened_epoch': fallback['highest_screened_epoch'],
            'next_intended_screening_epoch': fallback['next_intended_screening_epoch'],
            'overshoot_epochs': fallback['overshoot_epochs'],
            'safe_to_continue_automatically': fallback['safe_to_continue_automatically'],
        })
    except Exception as exc:
        status_fields['status_fallback_error'] = str(exc)

if (
    pilot_final_status is None
    and status_fields.get('safe_to_continue_automatically') is False
    and status_fields.get('overshoot_epochs')
):
    pilot_final_status = 'blocked_continuation_overshoot_conflict'

if pilot_final_status == 'blocked_continuation_overshoot_conflict' and status_fields.get('blocked_reason') is None:
    status_fields['blocked_reason'] = (
        'untrusted physical checkpoint(s) at epoch(s) ' + str(status_fields['overshoot_epochs']) + ' '
        'sit beyond the highest screened epoch -- manual review is required before this '
        'run can safely continue automatically'
    )

if prepare_only_result is not None:
    run_status = 'PREPARED_ONLY'
elif pilot_final_status == 'blocked_continuation_overshoot_conflict':
    run_status = 'BLOCKED_MANUAL_REVIEW_REQUIRED'
elif pilot_final_status == 'paused_at_max_target_epoch':
    run_status = 'PAUSED_AT_MAX_TARGET_EPOCH'
elif pilot_status == 0:
    run_status = 'COMPLETED'
elif env['LATEST_CKPT_AFTER']:
    run_status = 'INTERRUPTED_RESUMABLE'
else:
    run_status = 'FAILED_NO_CHECKPOINT'

run_identity = authoritative_result.get('run_identity') or {}
wandb_block = authoritative_result.get('wandb') or {}
wandb_policy_sha256 = authoritative_result.get('wandb_policy_sha256') or run_identity.get('wandb_policy_sha256')
wandb_run_id = wandb_block.get('wandb_run_id') or run_identity.get('wandb_run_id')

out = {
    'status': run_status,
    'run_id': env['RUN_ID'],
    'pilot_exit_code': pilot_status,
    'pilot_final_status': pilot_final_status,
    'package_root': env['PACKAGE_ROOT'],
    'config_out_dir': env['CONFIG_OUT_DIR'],
    'evidence_out_dir': env['EVIDENCE_OUT_DIR'],
    'nh_run_dir': nh_run_dir,
    'latest_checkpoint': env['LATEST_CKPT_AFTER'],
    'latest_epoch': int(env['LATEST_EPOCH_AFTER']) if env['LATEST_EPOCH_AFTER'] else None,
    **status_fields,
    'term_requested': env['TERM_REQUESTED'] == '1',
    'slurm_job_id': env['SLURM_JOB_ID'],
    'slurm_job_partition': env['SLURM_JOB_PARTITION'],
    'source_commit': env['SOURCE_COMMIT'],
    'expected_commit': env['EXPECTED_COMMIT_USED'],
    'hostname': socket.gethostname(),
    'wandb_policy_path': env['WANDB_POLICY_PATH_USED'] or None,
    'wandb_dir': env['WANDB_DIR_USED'] or None,
    'tracking_generation': env['TRACKING_GENERATION_USED'],
    'closure_max_target_epoch': int(env['CLOSURE_MAX_TARGET_EPOCH_USED']),
    'prepare_only': prepare_only_result is not None,
    'generated_config_path': (prepare_only_result or {}).get('generated_config_path'),
    'generation_manifest_path': (prepare_only_result or {}).get('generation_manifest_path'),
    'preparation_result_path': str(authoritative_path) if prepare_only_result is not None else None,
    'wandb_policy_sha256': wandb_policy_sha256,
    'wandb_run_id': wandb_run_id,
    'authoritative_result_source': authoritative_filename,
    'authoritative_result_parse_status': authoritative_parse_status,
    'physical_state_fallback_used': physical_state_fallback_used,
}
with open(env['RESULT_JSON_PATH'], 'w') as fh:
    json.dump(out, fh, indent=2)
print(json.dumps(out, indent=2))
print('RUN_STATUS_FOR_SHELL=' + run_status)
" | tee "${RESULT_DIR}/closure_result_stdout.log"

RUN_STATUS=$(grep '^RUN_STATUS_FOR_SHELL=' "${RESULT_DIR}/closure_result_stdout.log" | tail -1 | cut -d= -f2-)

echo "=== done (run_status ${RUN_STATUS}, exit ${PILOT_STATUS}) ==="
case "${RUN_STATUS}" in
    COMPLETED) exit 0 ;;
    PREPARED_ONLY) exit 0 ;;
    PAUSED_AT_MAX_TARGET_EPOCH) exit 0 ;;
    INTERRUPTED_RESUMABLE) exit 2 ;;
    *) exit 1 ;;
esac

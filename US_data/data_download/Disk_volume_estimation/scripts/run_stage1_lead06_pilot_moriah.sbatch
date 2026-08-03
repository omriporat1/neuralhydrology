#!/usr/bin/env bash
#SBATCH --job-name=flashnh-lead06-pilot
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/lead06-pilot-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/lead06-pilot-%j.err
#
# Stage 1 lead-6 optimization pilot (task item 7): Slurm launcher for ONE of
# the six closed pilot run_ids (raw_seedA, raw_seedB, emb128x64_seedA,
# emb128x64_seedB, emb64_seedA, emb128_seedA -- see
# config/stage1_lead06_pilot_v001.yaml and docs/stage1_lead06_pilot_v001.md).
# This script is prepared but NOT submitted as part of this task -- it is
# left ready for review. The intended first real Moriah job is
# emb128x64_seedA (the workflow-qualification run), but this script is not
# hard-coded to it: pass any of the six run_ids as $1.
#
# Usage:
#   sbatch scripts/run_stage1_lead06_pilot_moriah.sbatch emb128x64_seedA
#   sbatch --partition=... --gres=... --time=... \
#       scripts/run_stage1_lead06_pilot_moriah.sbatch raw_seedA
#   RUN_ID=raw_seedB sbatch --export=ALL,RUN_ID \
#       scripts/run_stage1_lead06_pilot_moriah.sbatch
#
# Delegates ALL config generation, chunked NH start_run/continue_run
# training, screening-subset evaluation, restart-safe early stopping, W&B
# logging, and compact evidence-bundle writing to
# `scripts/run_stage1_lead06_pilot.py` (task item 6), which itself composes
# `src.baseline.pilot_orchestration.run_pilot`. This script's only jobs are:
# resolve run_id/paths, activate the environment, invoke the CLI wrapper, and
# classify + record the Slurm-level outcome. No modeling, screening, or
# stopping decision is made here.
#
# Resource defaults (binding starting point, see task spec): --cpus-per-task=8
# / --mem=128G, matching num_workers=4 already fixed in
# config/stage1_lead06_pilot_v001.yaml's untuned hyperparameters. These are
# DELIBERATELY the historical full-population seed-train job's ORIGINAL
# values (see run_stage1_full_population_lead06_seq24_seed_train_moriah.sbatch's
# --mem/--cpus-per-task correction history), before that script's later
# resource corrections (16 CPUs / num_workers=12 / 224G) -- this pilot script
# must NOT start from those corrected values. If this pilot proves to need a
# correction of its own, that should happen the same way it did there: one
# documented, resource-only change in response to an observed OOM or a
# measured CPU-bound bottleneck, never a silent default change and never a
# model/training hyperparameter change.
#
# Partition/GRES/time/account/output-root are all caller-configurable, not
# hard-coded beyond the defaults above:
#   - Partition/GRES: override at submission time with
#     `sbatch --partition=... --gres=... this_script <run_id>` (defaults
#     above are the only Moriah GPU class this repo has confirmed working).
#   - Time limit: override with `sbatch --time=... this_script <run_id>`.
#   - Account: this repo's existing sbatch scripts do not set one; add
#     `sbatch --account=... this_script <run_id>` if your allocation requires it.
#   - Output root: set FLASHNH_BASE in the submission environment (e.g.
#     `FLASHNH_BASE=/some/other/root sbatch --export=ALL,FLASHNH_BASE ...`) to
#     redirect logs/config/evidence roots; defaults to the same
#     /sci/labs/efratmorin/omripo/Flash-NH root every other committed script
#     uses.
#
# Restart-safety: unlike the full-population seed-train script, this script
# does NOT need a run-directory marker file. `run_pilot()` itself discovers
# an already-started NH run directory from `config_out_dir` on every
# invocation (`discover_nh_run_dir`, keyed off a fixed, run_id-derived
# experiment name -- see pilot_orchestration.py) and skips any chunk/epoch
# already trained, any screening event already recorded, and any early-
# stopping-state transition already applied (idempotent replay -- see
# pilot_early_stopping.py). CONFIG_OUT_DIR and EVIDENCE_OUT_DIR below are
# therefore fixed per run_id, NOT $SLURM_JOB_ID-scoped, so resubmitting this
# exact script after a wall-time kill (or any other interruption) resumes
# automatically with no operator bookkeeping. #SBATCH --signal=B:TERM@180
# delivers SIGTERM ~3 minutes before the wall-time kill so this is logged,
# not silently truncated; NH's own checkpoint-every-epoch + continue_run is
# what makes an actual resume safe, exactly as already relied upon by the
# full-population seed-train job.
#
# This script never passes --force to the CLI wrapper: --force only controls
# whether an already-generated NH config is regenerated (dangerous on a
# resume of a run that has already started training), never whether the
# evidence bundle is rewritten (run_pilot() always rewrites its own evidence
# bundle regardless of --force -- see pilot_orchestration.py's run_pilot
# docstring). A fresh run_id generates its config on first submission because
# none exists yet; every later resubmission reuses it unchanged.

set -uo pipefail

RUN_ID="${RUN_ID:-${1:-}}"
if [ -z "${RUN_ID}" ]; then
    echo "USAGE: sbatch $0 <run_id>   (or set RUN_ID via --export)" >&2
    echo "run_id must be one of: raw_seedA raw_seedB emb128x64_seedA emb128x64_seedB emb64_seedA emb128_seedA" >&2
    exit 2
fi

# Optional bounded-recovery epoch limit (job 45731908 postmortem -- see
# docs/stage1_lead06_pilot_v001.md and run_pilot()'s max_target_epoch
# parameter): forwarded to the CLI wrapper's --max-target-epoch flag only
# when the submitter explicitly sets MAX_TARGET_EPOCH. Left unset, this
# script's behavior is completely unchanged from before this option existed
# -- no flag is appended, and the CLI wrapper's own default (None, unbounded)
# applies. This is a single scoped recovery knob, not a general runtime-
# override framework: no other pilot parameter is threaded through an
# environment variable this way. Validated here, before any GPU/module setup
# or CLI invocation, so a mistyped value fails fast rather than burning a GPU
# allocation.
MAX_TARGET_EPOCH="${MAX_TARGET_EPOCH:-}"
if [ -n "${MAX_TARGET_EPOCH}" ]; then
    if ! [[ "${MAX_TARGET_EPOCH}" =~ ^[0-9]+$ ]] || [ "${MAX_TARGET_EPOCH}" -le 0 ]; then
        echo "MAX_TARGET_EPOCH must be a positive integer, got: '${MAX_TARGET_EPOCH}'" >&2
        exit 2
    fi
fi

FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
ENV_PREFIX="${FLASHNH_BASE}/envs/flashnh-moriah"

# Code location is independent of FLASHNH_BASE (data/run storage): the
# authoritative clean clone for this pilot is
# /sci/labs/efratmorin/omripo/PhD/Python/neuralhydrology -- NOT the other,
# dirty secondary clone elsewhere under omripo's home directory, and NOT
# derived from FLASHNH_BASE. Override with PILOT_REPO_CLONE_DIR if needed.
PILOT_REPO_CLONE_DIR="${PILOT_REPO_CLONE_DIR:-/sci/labs/efratmorin/omripo/PhD/Python/neuralhydrology}"
REPO_CLONE_DIR="${PILOT_REPO_CLONE_DIR}"
REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"

PACKAGE_ROOT="${PILOT_PACKAGE_ROOT:-${FLASHNH_BASE}/data/stage1_scientific_package_v002}"
CONFIG_OUT_DIR="${PILOT_CONFIG_OUT_DIR:-${FLASHNH_BASE}/runs/stage1_lead06_pilot_${RUN_ID}_v001}"
EVIDENCE_OUT_DIR="${PILOT_EVIDENCE_OUT_DIR:-${FLASHNH_BASE}/evidence/stage1_lead06_pilot_${RUN_ID}_v001}"

# Optional per-run W&B policy override (see docs/stage1_wandb_user_guide.md).
# The committed config/stage1_wandb_tracking_policy_v001.yaml (enabled=false/
# mode=disabled) is used, unmodified, unless the submitter explicitly sets
# WANDB_POLICY_PATH to an untracked, machine-local override policy file --
# this script never edits or reads the committed policy itself, it only
# forwards whatever path (if any) it was given to the CLI wrapper below.
# Whenever an override is supplied, a WANDB_DIR outside this tracked repo
# clone is required so W&B never writes beneath REPO_WORKDIR: default it to a
# RUN_ID-scoped subdirectory of FLASHNH_BASE (never inside the repo clone),
# consistent with every other *_OUT_DIR/PACKAGE_ROOT default above, but allow
# the submitter to override it directly.
WANDB_POLICY_PATH="${WANDB_POLICY_PATH:-}"
if [ -n "${WANDB_POLICY_PATH}" ]; then
    WANDB_DIR="${WANDB_DIR:-${FLASHNH_BASE}/wandb/offline/${RUN_ID}}"
    mkdir -p "${WANDB_DIR}"
fi
TRACKING_GENERATION="${TRACKING_GENERATION:-g1}"

# Preparation-only mode (set PREPARE_ONLY=1 in the submission environment):
# threads --prepare-only through to the CLI wrapper (see
# scripts/run_stage1_lead06_pilot.py), which generates only this run_id's NH
# config + generation manifest and exits before any NH training call or W&B
# backend initialization -- never a "completed" run. Since preparation needs
# no GPU (no torch/neuralhydrology import happens on this path -- see
# src.baseline.pilot_orchestration.prepare_pilot_run_only), the CUDA
# availability check below is skipped in this mode; this script's #SBATCH
# resource header above still requests the same GPU class as an ordinary
# training submission (see this script's header comment -- conditionally
# changing #SBATCH directives at submission time is not attempted here, since
# they are parsed by Slurm before this shell runs). Submit a preparation-only
# job against a non-GPU/lighter allocation instead by overriding at `sbatch`
# time, e.g.:
#   PREPARE_ONLY=1 sbatch --partition=<cpu-partition> --gres= \
#       --export=ALL,PREPARE_ONLY --cpus-per-task=2 --mem=16G \
#       scripts/run_stage1_lead06_pilot_moriah.sbatch raw_seedA
PREPARE_ONLY="${PREPARE_ONLY:-0}"

RESULT_DIR="${FLASHNH_BASE}/evidence/lead06_pilot_${RUN_ID}_${SLURM_JOB_ID}"
mkdir -p "${RESULT_DIR}" "$(dirname "${CONFIG_OUT_DIR}")" "$(dirname "${EVIDENCE_OUT_DIR}")"
RESULT_LOG="${RESULT_DIR}/pilot_human.log"
exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== Stage 1 lead-6 optimization pilot: run_id=${RUN_ID} ==="
hostname
date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-unknown}"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unknown}"
scontrol show job "${SLURM_JOB_ID}" || true

test -d "${PACKAGE_ROOT}" || { echo "MISSING transferred package: ${PACKAGE_ROOT}"; exit 1; }

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

if [ "${PREPARE_ONLY}" = "1" ]; then
    echo "PREPARE_ONLY=1: preparation-only job -- no NH training call, no W&B backend init, "
    echo "and no GPU/CUDA required (see comment above); skipping the nvidia-smi/CUDA check."
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

# Pre-launch repository guard: refuse to launch against a repo work
# directory that doesn't exist, or one whose tracked files have been
# locally modified relative to the reviewed commit -- this pilot must only
# ever run the exact reviewed, committed source. Untracked files (e.g.
# generated evidence/config left under the repo root by a prior local run)
# are tolerated since they cannot affect tracked source/config code.
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

PILOT_CLI_ARGS=(
    --run-id "${RUN_ID}"
    --package-root "${PACKAGE_ROOT}"
    --config-out-dir "${CONFIG_OUT_DIR}"
    --evidence-out-dir "${EVIDENCE_OUT_DIR}"
    --slurm-job-id "${SLURM_JOB_ID:-}"
    --slurm-partition "${SLURM_JOB_PARTITION:-}"
    --slurm-gres "${SLURM_JOB_GRES:-}"
    --tracking-generation "${TRACKING_GENERATION}"
)
if [ -n "${WANDB_POLICY_PATH}" ]; then
    PILOT_CLI_ARGS+=(--wandb-policy-path "${WANDB_POLICY_PATH}")
    # Offline-only, never online: no API key, no `wandb sync`, no checkpoint
    # byte uploads (see docs/stage1_wandb_user_guide.md). WANDB_DIR is always
    # set above (default or caller override) whenever a policy override is
    # supplied, so this never falls back to writing beneath REPO_WORKDIR.
    export WANDB_MODE=offline
    export WANDB_DIR
    echo "W&B tracking override active: policy=${WANDB_POLICY_PATH} dir=${WANDB_DIR} mode=offline"
fi
if [ "${PREPARE_ONLY}" = "1" ]; then
    PILOT_CLI_ARGS+=(--prepare-only)
fi
if [ -n "${MAX_TARGET_EPOCH}" ]; then
    PILOT_CLI_ARGS+=(--max-target-epoch "${MAX_TARGET_EPOCH}")
    echo "Bounded recovery active: --max-target-epoch ${MAX_TARGET_EPOCH}"
fi

echo "=== launching scripts/run_stage1_lead06_pilot.py --run-id ${RUN_ID} ==="
python scripts/run_stage1_lead06_pilot.py "${PILOT_CLI_ARGS[@]}" \
    > "${RESULT_DIR}/pilot_stdout.json.log" &
PILOT_PID=$!

# Real handling for the #SBATCH --signal=B:TERM@180 declared above: forward
# TERM to the pilot child so NH's checkpoint-every-epoch state is left
# intact for resume, rather than letting Slurm's later hard kill truncate
# it mid-write. Guarded so a second TERM/INT (or one delivered after the
# child has already exited) is a no-op -- no recursive trap, no cleanup of
# run outputs here (those are never deleted by this script).
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

# run_stage1_lead06_pilot.py prints a single JSON object with nh_run_dir on
# success; if it was killed before printing (e.g. wall-time SIGTERM mid-chunk)
# fall back to the same experiment-name convention pilot_orchestration.py
# itself uses for discovery, so classification still works without needing to
# parse partial/absent stdout.
NH_RUN_DIR=""
if [ -s "${RESULT_DIR}/pilot_stdout.json.log" ]; then
    NH_RUN_DIR=$(python -c "
import json
try:
    print(json.load(open('${RESULT_DIR}/pilot_stdout.json.log'))['nh_run_dir'])
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
    # No -maxdepth: physical checkpoints may live in a nested
    # continue_training_from_epoch###/ continuation directory, not only
    # flat in NH_RUN_DIR itself (see pilot_orchestration.py's
    # continuation-epoch-semantics note). Sort numerically by the epoch
    # number extracted from each match's basename -- plain path-string
    # sorting is not reliable across differently-nested directories.
    LATEST_CKPT_AFTER=$(find "${NH_RUN_DIR}" -name 'model_epoch*.pt' 2>/dev/null | \
        sed -E 's#^(.*model_epoch0*([0-9]+)\.pt)$#\2 \1#' | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "${LATEST_CKPT_AFTER}" ]; then
        LATEST_EPOCH_AFTER=$(basename "${LATEST_CKPT_AFTER}" | sed -E 's/model_epoch0*([0-9]+)\.pt/\1/')
    fi
fi

# Distinct status fields (task item 7): "a checkpoint physically exists at
# epoch N" must never be read as "screening/stopping-eligibility progressed
# to epoch N" -- see pilot_orchestration.py's continuation-epoch-semantics
# note (the second real qualification-run failure this was written to fix:
# checkpoints 10-15 existed as accidental overshoot while screening had only
# actually reached epoch 9). Primary source is the authoritative evidence
# file run_pilot()/prepare_pilot_run_only() themselves write under
# --evidence-out-dir (pilot_run_evidence.json / pilot_preparation_result.json
# -- see job-45734071 postmortem below, this is NOT parsed from
# pilot_stdout.json.log); if that file is missing or incomplete (e.g.
# wall-time SIGTERM before the pilot subprocess could write it), fall back to
# computing these fields directly from on-disk state via
# pilot_orchestration.compute_pilot_status_fields (no next-intended-epoch in
# this fallback path -- it needs the full pilot policy object, not
# conveniently available here). Passed via environment, never interpolated
# into the Python source string, so a free-text blocked_reason containing
# quotes/newlines can never break this script.
RUN_ID="${RUN_ID}" NH_RUN_DIR="${NH_RUN_DIR}" \
PILOT_STATUS="${PILOT_STATUS}" PACKAGE_ROOT="${PACKAGE_ROOT}" \
CONFIG_OUT_DIR="${CONFIG_OUT_DIR}" EVIDENCE_OUT_DIR="${EVIDENCE_OUT_DIR}" \
LATEST_CKPT_AFTER="${LATEST_CKPT_AFTER}" LATEST_EPOCH_AFTER="${LATEST_EPOCH_AFTER}" \
TERM_REQUESTED="${TERM_REQUESTED}" SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION:-unknown}" SOURCE_COMMIT="${SOURCE_COMMIT}" \
WANDB_POLICY_PATH_USED="${WANDB_POLICY_PATH}" WANDB_DIR_USED="${WANDB_DIR:-}" \
TRACKING_GENERATION_USED="${TRACKING_GENERATION}" \
MAX_TARGET_EPOCH_USED="${MAX_TARGET_EPOCH}" \
PREPARE_ONLY_USED="${PREPARE_ONLY}" \
RESULT_JSON_PATH="${RESULT_DIR}/pilot_result.json" \
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

# Authoritative scientific result source (job-45734071 launcher-summary
# postmortem -- see docs/stage1_lead06_pilot_v001.md). pilot_stdout.json.log
# is the pilot subprocess's raw stdout capture: it also carries NH's own
# log/progress-bar text ahead of the CLI's final printed JSON object, so a
# naive json.load() on that whole file fails on every ordinary training
# invocation (only a --prepare-only invocation's stdout is pure JSON, since
# that path never imports neuralhydrology/torch -- see
# scripts/run_stage1_lead06_pilot.py), and the previous exception handler
# silently discarded every scientifically meaningful field as a result. The
# authoritative result now always comes from the same file
# run_pilot()/prepare_pilot_run_only() themselves write under
# --evidence-out-dir -- never from parsing mixed subprocess stdout, and
# never from a heuristic scan of that stdout for an embedded JSON substring.
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

# --prepare-only's authoritative file has its own 'status': 'PREPARED_ONLY'
# schema (see src.baseline.pilot_orchestration.prepare_pilot_run_only) and
# must never be folded into the ordinary
# COMPLETED/INTERRUPTED_RESUMABLE/etc. classification below, since no
# training or W&B backend initialization occurred.
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

# Physical-state fallback (visible via physical_state_fallback_used below,
# never silent): only fills in fields the authoritative file could not
# supply -- never invents a final_status the CLI/evidence bundle did not
# itself record, and never overwrites a known scientific final status with
# generic process success.
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

# Job 45718473: an unavailable/corrupt authoritative source can still leave
# a genuinely blocked run (untrusted continuation overshoot) undetected by
# pilot_final_status alone -- the physical-state fallback above already
# recomputes safe_to_continue_automatically/overshoot_epochs directly from
# on-disk state in that case. Reuse the exact same
# 'blocked_continuation_overshoot_conflict' status string
# pilot_orchestration.run_pilot() itself uses for this condition, rather
# than inventing a second convention.
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
        'pilot can safely continue automatically'
    )

# Non-zero process exit with a checkpoint already on disk (e.g. wall-time
# SIGTERM mid-chunk) is a resumable partial run, not an unqualified failure
# -- resubmitting this exact script resumes automatically (see
# restart-safety note above). A blocked pilot (untrusted continuation
# overshoot, never trained into automatically -- see pilot_orchestration.py)
# and a deliberate bounded pause (--max-target-epoch reached, job-45731908
# recovery -- see docs/stage1_lead06_pilot_v001.md) are each their own
# distinct, truthful status: neither is ever collapsed into generic
# COMPLETED.
if prepare_only_result is not None:
    # Never COMPLETED: this run_id was only prepared (config + generation
    # manifest written), never trained -- see prepare_only_result's own
    # 'training_started'/'evaluation_started'/'wandb_backend_initialized'
    # fields (all False) and its 'statement' field for the explicit
    # non-training confirmation.
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
    'hostname': socket.gethostname(),
    'wandb_policy_path': env['WANDB_POLICY_PATH_USED'] or None,
    'wandb_dir': env['WANDB_DIR_USED'] or None,
    'tracking_generation': env['TRACKING_GENERATION_USED'],
    'max_target_epoch': int(env['MAX_TARGET_EPOCH_USED']) if env['MAX_TARGET_EPOCH_USED'] else None,
    'prepare_only': prepare_only_result is not None,
    'generated_config_path': (prepare_only_result or {}).get('generated_config_path'),
    'generation_manifest_path': (prepare_only_result or {}).get('generation_manifest_path'),
    # not itself a key inside pilot_preparation_result.json (the file cannot
    # know its own eventual path at the moment it is written -- see
    # prepare_pilot_run_only()); this is simply the path we just read it from.
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
" | tee "${RESULT_DIR}/pilot_result_stdout.log"

RUN_STATUS=$(grep '^RUN_STATUS_FOR_SHELL=' "${RESULT_DIR}/pilot_result_stdout.log" | tail -1 | cut -d= -f2-)

echo "=== done (run_status ${RUN_STATUS}, exit ${PILOT_STATUS}) ==="
# COMPLETED -> 0; PREPARED_ONLY -> 0 (a deliberate, successful --prepare-only
# invocation -- never implies training occurred, see pilot_result.json's
# 'training_started'/'wandb_backend_initialized': false fields);
# PAUSED_AT_MAX_TARGET_EPOCH -> 0 (a deliberate, successful bounded pause at
# --max-target-epoch -- see job-45734071 recovery; this is a distinct,
# truthful status and must never be reported as COMPLETED, since screening
# is intentionally not finished); INTERRUPTED_RESUMABLE -> 2 (nonzero but
# distinct from an unqualified failure -- resubmitting this exact script
# resumes automatically, see restart-safety note above);
# BLOCKED_MANUAL_REVIEW_REQUIRED and FAILED_NO_CHECKPOINT -> 1 (both require
# a human before this run_id can proceed further, though for different
# reasons -- see pilot_result.json's distinct 'status'/'blocked_reason'
# fields for which one actually occurred).
case "${RUN_STATUS}" in
    COMPLETED) exit 0 ;;
    PREPARED_ONLY) exit 0 ;;
    PAUSED_AT_MAX_TARGET_EPOCH) exit 0 ;;
    INTERRUPTED_RESUMABLE) exit 2 ;;
    *) exit 1 ;;
esac

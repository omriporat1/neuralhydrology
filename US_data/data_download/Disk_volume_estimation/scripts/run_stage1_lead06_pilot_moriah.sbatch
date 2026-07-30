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

echo "=== launching scripts/run_stage1_lead06_pilot.py --run-id ${RUN_ID} ==="
python scripts/run_stage1_lead06_pilot.py \
    --run-id "${RUN_ID}" \
    --package-root "${PACKAGE_ROOT}" \
    --config-out-dir "${CONFIG_OUT_DIR}" \
    --evidence-out-dir "${EVIDENCE_OUT_DIR}" \
    --slurm-job-id "${SLURM_JOB_ID:-}" \
    --slurm-partition "${SLURM_JOB_PARTITION:-}" \
    --slurm-gres "${SLURM_JOB_GRES:-}" \
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
# actually reached epoch 9). Primary source is run_stage1_lead06_pilot.py's
# own stdout JSON (already carries these fields from run_pilot()'s return
# dict); if that's missing or incomplete (e.g. wall-time SIGTERM before it
# could print), fall back to computing them directly from on-disk state via
# pilot_orchestration.compute_pilot_status_fields (no next-intended-epoch in
# this fallback path -- it needs the full pilot policy object, not
# conveniently available here). Passed via environment, never interpolated
# into the Python source string, so a free-text blocked_reason containing
# quotes/newlines can never break this script.
STDOUT_JSON_PATH="${RESULT_DIR}/pilot_stdout.json.log"
RUN_ID="${RUN_ID}" NH_RUN_DIR="${NH_RUN_DIR}" STDOUT_JSON_PATH="${STDOUT_JSON_PATH}" \
PILOT_STATUS="${PILOT_STATUS}" PACKAGE_ROOT="${PACKAGE_ROOT}" \
CONFIG_OUT_DIR="${CONFIG_OUT_DIR}" EVIDENCE_OUT_DIR="${EVIDENCE_OUT_DIR}" \
LATEST_CKPT_AFTER="${LATEST_CKPT_AFTER}" LATEST_EPOCH_AFTER="${LATEST_EPOCH_AFTER}" \
TERM_REQUESTED="${TERM_REQUESTED}" SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION:-unknown}" SOURCE_COMMIT="${SOURCE_COMMIT}" \
RESULT_JSON_PATH="${RESULT_DIR}/pilot_result.json" \
python -c "
import json
import os
import socket
import sys

sys.path.insert(0, os.getcwd())

env = os.environ
pilot_status = int(env['PILOT_STATUS'])
nh_run_dir = env['NH_RUN_DIR']

stdout_result = {}
stdout_json_path = env['STDOUT_JSON_PATH']
if os.path.getsize(stdout_json_path) > 0 if os.path.exists(stdout_json_path) else False:
    try:
        with open(stdout_json_path) as fh:
            stdout_result = json.load(fh)
    except Exception:
        stdout_result = {}

status_fields = {
    'highest_physical_checkpoint_epoch': stdout_result.get('highest_physical_checkpoint_epoch'),
    'highest_screened_epoch': stdout_result.get('highest_screened_epoch'),
    'next_intended_screening_epoch': stdout_result.get('next_intended_screening_epoch'),
    'overshoot_epochs': stdout_result.get('overshoot_epochs'),
    'safe_to_continue_automatically': stdout_result.get('safe_to_continue_automatically'),
    'blocked_reason': stdout_result.get('blocked_reason'),
}
pilot_final_status = stdout_result.get('final_status')

if not stdout_result and nh_run_dir and os.path.isdir(nh_run_dir):
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

# Job 45718473: the primary pilot_stdout.json.log path can be empty/
# unavailable for reasons unrelated to a genuine crash (see the fallback
# comment above), and when that happens the fallback just above already
# recomputes safe_to_continue_automatically/overshoot_epochs directly from
# on-disk state -- but pilot_final_status/blocked_reason stayed None, so a
# run correctly blocked by an untrusted continuation overshoot was
# misclassified as COMPLETED below (exit 0, status: COMPLETED) even though
# safe_to_continue_automatically was already False. Reuse the exact same
# 'blocked_continuation_overshoot_conflict' status string
# pilot_orchestration.run_pilot() itself uses for this condition, rather
# than inventing a second convention.
if (
    pilot_final_status is None
    and status_fields.get('safe_to_continue_automatically') is False
    and status_fields.get('overshoot_epochs')
):
    pilot_final_status = 'blocked_continuation_overshoot_conflict'
    status_fields['blocked_reason'] = (
        'derived from on-disk state (primary pilot stdout status was unavailable): '
        'untrusted physical checkpoint(s) at epoch(s) ' + str(status_fields['overshoot_epochs']) + ' '
        'sit beyond the highest screened epoch -- manual review is required before this '
        'pilot can safely continue automatically'
    )

# Non-zero process exit with a checkpoint already on disk (e.g. wall-time
# SIGTERM mid-chunk) is a resumable partial run, not an unqualified failure
# -- resubmitting this exact script resumes automatically (see
# restart-safety note above). A blocked pilot (untrusted continuation
# overshoot, never trained into automatically -- see
# pilot_orchestration.py) is its own distinct status: the process exits 0
# (it made a deliberate, non-crashing determination) but is NOT an
# unqualified COMPLETED, since a human must resolve the overshoot before
# this run_id can proceed.
if pilot_final_status == 'blocked_continuation_overshoot_conflict':
    run_status = 'BLOCKED_MANUAL_REVIEW_REQUIRED'
elif pilot_status == 0:
    run_status = 'COMPLETED'
elif env['LATEST_CKPT_AFTER']:
    run_status = 'INTERRUPTED_RESUMABLE'
else:
    run_status = 'FAILED_NO_CHECKPOINT'

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
}
with open(env['RESULT_JSON_PATH'], 'w') as fh:
    json.dump(out, fh, indent=2)
print(json.dumps(out, indent=2))
print('RUN_STATUS_FOR_SHELL=' + run_status)
" | tee "${RESULT_DIR}/pilot_result_stdout.log"

RUN_STATUS=$(grep '^RUN_STATUS_FOR_SHELL=' "${RESULT_DIR}/pilot_result_stdout.log" | tail -1 | cut -d= -f2-)

echo "=== done (run_status ${RUN_STATUS}, exit ${PILOT_STATUS}) ==="
# COMPLETED -> 0; INTERRUPTED_RESUMABLE -> 2 (nonzero but distinct from an
# unqualified failure -- resubmitting this exact script resumes
# automatically, see restart-safety note above); BLOCKED_MANUAL_REVIEW_REQUIRED
# and FAILED_NO_CHECKPOINT -> 1 (both require a human before this run_id can
# proceed further, though for different reasons -- see pilot_result.json's
# distinct 'status'/'blocked_reason' fields for which one actually occurred).
case "${RUN_STATUS}" in
    COMPLETED) exit 0 ;;
    INTERRUPTED_RESUMABLE) exit 2 ;;
    *) exit 1 ;;
esac

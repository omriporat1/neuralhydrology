#!/bin/bash
# Tracked Moriah submission boundary for the v2 six-axis W&B-agent launcher.
#
# WHY THIS FILE EXISTS
# -------------------
# The only tracked artifacts in the unattended P9-P12 chain were the submitted
# job (run_sweep_v2_six_axis_wandb_agent_moriah.sbatch) and the sweep program
# (run_sweep_v2_six_axis_wandb_bridge.py). Nothing tracked invoked `sbatch`
# itself; that was done by an untracked ~/moriah_bayes_launch.sh helper. A
# detached / non-login parent (setsid / nohup) does NOT get Moriah's Slurm
# `profile.d` hook -- it is interactive-only -- so it loses `PATH` and
# `SLURM_CONF` and cannot even reach `sbatch`. Logic inside the .sbatch job
# cannot repair this: it only runs after submission already succeeded. This
# tiny wrapper is the durable, tracked place to restore that client state
# right before submission.
#
# It is deliberately NOT a scheduler abstraction: it is Moriah-specific, does
# exactly one submission of one known job script, and every restoration step
# is guarded and idempotent (a normal login shell where sbatch already works
# is left untouched).
#
# SCOPE -- what this wrapper does NOT do
# -------------------------------------
# It only restores the Slurm client environment and submits. It does NOT:
#   * build or select the production manifest (a separate, user-authorized
#     step per P9-P12 proposal produces the manifest path passed in here);
#   * verify manifest identity distinctness -- the authoritative
#     checksum/schema/full-v2-identity/mode/stop_before_training/max_agents/
#     forbidden-id contract is enforced by the in-job `validate-launch`
#     seam before any W&B contact, so it is not duplicated here;
#   * perform the read-only pre-launch W&B controller-state gate (the old
#     home helper's "Part D": exactly the N-1 expected prior finished+valid
#     runs, objectives matching recorded observations, controller config ==
#     committed authoritative config, no pre-existing proposal-N trial).
#     That check is inherently per-proposal (it pins mutable campaign
#     state) and remains an explicit step owned by whoever authorizes each
#     proposal launch. This wrapper does no controller reconciliation.
#
# USAGE
#   bash scripts/submit_sweep_v2_six_axis_wandb_agent_moriah.sh <production-manifest-path>
# Invoke it through `bash` (the tracked file is a normal 100644 file, like the
# other scripts/*.sh and scripts/*.sbatch in this repo -- it is not relied on
# to be directly executable). The manifest path may instead be provided via
# FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST.
#
# WANDB_PROJECT / WANDB_ENTITY are NOT set here: the submitted job resolves them
# from the strict manifest (authoritative) and refuses a contradicting inherited
# value. This wrapper only forwards the caller's environment unchanged
# (--export=ALL), so an already-set WANDB_API_KEY reaches the job; the
# credential fallback itself lives in the .sbatch.
# On success the sbatch job id is printed to stdout (sbatch --parsable);
# all diagnostics go to stderr so `JOBID=$(bash submit_...sh ...)` stays clean.
set -euo pipefail

MANIFEST="${1:-${FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST:-}}"
if [ -z "${MANIFEST}" ]; then
    echo "FATAL: production manifest path required (positional arg or FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST)" >&2
    exit 2
fi
if [ ! -f "${MANIFEST}" ]; then
    echo "FATAL: production manifest not found: ${MANIFEST}" >&2
    exit 2
fi

# Restore the Moriah Slurm client environment that a detached / non-login
# parent loses. The two default locations are stable Moriah host facts
# recorded in docs/remote_operations.md (SS 2.3); FLASHNH_MORIAH_SLURM_BIN /
# FLASHNH_MORIAH_SLURM_CONF override them (used only by the focused tests).
# Every step is guarded and idempotent. The real `exec sbatch ...` below is
# the authoritative consumer of the Slurm configuration -- there is no
# separate preflight health check here.
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
AGENT_SBATCH="${_script_dir}/run_sweep_v2_six_axis_wandb_agent_moriah.sbatch"
if [ ! -f "${AGENT_SBATCH}" ]; then
    echo "FATAL: agent job script not found: ${AGENT_SBATCH}" >&2
    exit 2
fi

echo "submit: sbatch=$(command -v sbatch) SLURM_CONF=${SLURM_CONF:-<unset>}" >&2
echo "submit: manifest=${MANIFEST}" >&2
echo "submit: WANDB_PROJECT/WANDB_ENTITY resolved by the job from the strict manifest" >&2

exec sbatch --parsable \
    --export="ALL,FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST=${MANIFEST}" \
    "${AGENT_SBATCH}"

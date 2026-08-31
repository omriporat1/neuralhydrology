#!/bin/bash
#SBATCH --job-name=flashnh-sweep-v2-wandb-agent
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/sweep-v2-wandb-agent-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/sweep-v2-wandb-agent-%j.err
set -euo pipefail

# The reusable v2 production W&B controller has a STATIC command
# (["${interpreter}", "${program}"]) that embeds no proposal-specific
# manifest. Each separately launched one-agent production job selects
# exactly one strict, immutable mode=production launch manifest through this
# explicit operational input, which is exported below so the W&B-created
# bridge subprocess inherits it. Because the controller command omits the
# ${args} macro, W&B never appends swept --key=value flags to the bridge.
#
# The manifest path is the sole launch identity. Before any W&B contact this
# launcher runs the maintained pre-agent validator with the canonical
# interpreter: it loader-validates the manifest (checksum, schema, full v2
# identity), requires mode=production / stop_before_training=false /
# max_agents=1, rejects the frozen v1 production sweep and the CLOSED
# disposable rehearsal sweep, and prints the one authoritative sweep id the
# agent may consume. WANDB_SWEEP_ID is optional and, if set, only a
# redundant cross-check that must equal the validated manifest id exactly.
: "${WANDB_PROJECT:?set the v2 production project}"
: "${WANDB_ENTITY:?set the v2 production entity}"
: "${FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST:?set the one immutable strict mode=production launch manifest path}"

# A real production launch must never inherit an active bridge self-test
# hook: an inherited value can make a real agent proposal exit successfully
# without manifest loading, W&B initialization, proposal intake, training,
# or evaluation. Refuse visibly rather than silently clearing an unexpected
# setting, then unset it afterward for defense in depth.
if [ -n "${FLASHNH_SWEEP_V2_BRIDGE_SELFTEST:-}" ]; then
    echo "REFUSING: FLASHNH_SWEEP_V2_BRIDGE_SELFTEST is set (${FLASHNH_SWEEP_V2_BRIDGE_SELFTEST}); a real production launch must not inherit a bridge self-test hook" >&2
    exit 1
fi
unset FLASHNH_SWEEP_V2_BRIDGE_SELFTEST || true

# Static shell-level pre-contact refusal mirroring the authoritative
# FORBIDDEN_PRODUCTION_SWEEP_IDS set (a test guards these literals against drift).
FORBIDDEN_PRODUCTION_SWEEP_ID="4x3btz2s"
FORBIDDEN_DISPOSABLE_REHEARSAL_SWEEP_ID="oz5p4csb"
for _forbidden in "${FORBIDDEN_PRODUCTION_SWEEP_ID}" "${FORBIDDEN_DISPOSABLE_REHEARSAL_SWEEP_ID}"; do
    if [ -n "${WANDB_SWEEP_ID:-}" ] && [ "${WANDB_SWEEP_ID:-}" = "${_forbidden}" ]; then
        echo "REFUSING: supplied sweep id ${WANDB_SWEEP_ID} is forbidden for every v2 production path" >&2
        exit 1
    fi
done

test -f "${FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST}" || { echo "MISSING v2 production launch manifest: ${FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST}" >&2; exit 1; }
export WANDB_PROJECT WANDB_ENTITY
export FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST

FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
REPO_WORKDIR="${REPO_WORKDIR:-${FLASHNH_BASE}/repos/flash-nh/US_data/data_download/Disk_volume_estimation}"
CANONICAL_PYTHON="${CANONICAL_PYTHON:-${FLASHNH_BASE}/envs/flashnh-moriah/bin/python}"
test -d "${REPO_WORKDIR}" || { echo "MISSING repo work directory" >&2; exit 1; }
cd "${REPO_WORKDIR}"
test -x "${CANONICAL_PYTHON}" || { echo "MISSING canonical interpreter" >&2; exit 1; }
export PATH="$(dirname "${CANONICAL_PYTHON}"):${PATH}"

# Authoritative pre-agent manifest/controller join. Fails nonzero -- before
# the agent runs, before wandb.init(), before proposal intake, before any
# W&B mutation -- on a checksum/schema error, a non-production manifest,
# stop_before_training, max_agents!=1, a forbidden sweep id, or a
# WANDB_SWEEP_ID that contradicts the manifest. Prints ONLY the one
# authoritative sweep id. Imports and contacts no W&B.
VALIDATE_ARGS=(--manifest-path "${FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST}")
if [ -n "${WANDB_SWEEP_ID:-}" ]; then
    VALIDATE_ARGS+=(--expect-sweep-id "${WANDB_SWEEP_ID}")
fi
VALIDATED_SWEEP_ID="$("${CANONICAL_PYTHON}" scripts/create_sweep_v2_six_axis_wandb_bridge_production_sweep.py validate-launch "${VALIDATE_ARGS[@]}")"

for _forbidden in "${FORBIDDEN_PRODUCTION_SWEEP_ID}" "${FORBIDDEN_DISPOSABLE_REHEARSAL_SWEEP_ID}"; do
    if [ "${VALIDATED_SWEEP_ID}" = "${_forbidden}" ]; then
        echo "REFUSING: validated sweep id ${VALIDATED_SWEEP_ID} is forbidden for every v2 production path" >&2
        exit 1
    fi
done

echo "=== Flash-NH v2 bridge production agent (one proposal) ==="
date -u
hostname
pwd
echo "MODE: production"
echo "EXPECTED_COMMIT: ${EXPECTED_COMMIT:-validated by bridge manifest}"
echo "CANONICAL_PYTHON: ${CANONICAL_PYTHON}"
# The one immutable strict manifest is selected through the exported
# FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST operational input, not the sweep
# command; do not expose credentials or dump the environment here.
echo "V2_BRIDGE_MANIFEST: ${FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST}"
echo "VALIDATED_SWEEP_ID: ${VALIDATED_SWEEP_ID}"

# This launcher consumes exactly one controller proposal, targeting the
# sweep id derived from the loader-validated mode=production manifest.
wandb agent --count 1 "${VALIDATED_SWEEP_ID}"

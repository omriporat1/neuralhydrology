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
# disposable rehearsal sweep, and prints the one authoritative launch
# identity (sweep id + W&B project/entity) the agent may consume.
# WANDB_SWEEP_ID is optional and, if set, only a redundant cross-check that
# must equal the validated manifest id exactly.
#
# WANDB_PROJECT / WANDB_ENTITY are NOT independent inputs here: the strict
# mode=production manifest is authoritative for them. They are resolved from
# the loader-validated manifest below; a non-empty inherited value that
# disagrees with the manifest is refused before any W&B contact, never used.
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
# WANDB_SWEEP_ID that contradicts the manifest. With --emit env it prints
# ONLY the manifest-authoritative launch identity (sweep id + W&B
# project/entity) as KEY=VALUE lines. Imports and contacts no W&B.
VALIDATE_ARGS=(--manifest-path "${FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST}" --emit env)
if [ -n "${WANDB_SWEEP_ID:-}" ]; then
    VALIDATE_ARGS+=(--expect-sweep-id "${WANDB_SWEEP_ID}")
fi
# The validator emits the authoritative launch identity as KEY=VALUE lines;
# parse them with a plain shell read loop (no eval, no sed/grep).
_LAUNCH_ENV="$("${CANONICAL_PYTHON}" scripts/create_sweep_v2_six_axis_wandb_bridge_production_sweep.py validate-launch "${VALIDATE_ARGS[@]}")"
_MANIFEST_SWEEP_ID=""
_MANIFEST_PROJECT=""
_MANIFEST_ENTITY=""
while IFS='=' read -r _k _v; do
    case "${_k}" in
        WANDB_SWEEP_ID) _MANIFEST_SWEEP_ID="${_v}" ;;
        WANDB_PROJECT) _MANIFEST_PROJECT="${_v}" ;;
        WANDB_ENTITY) _MANIFEST_ENTITY="${_v}" ;;
    esac
done <<< "${_LAUNCH_ENV}"

VALIDATED_SWEEP_ID="${_MANIFEST_SWEEP_ID}"
test -n "${VALIDATED_SWEEP_ID}" || { echo "FATAL: pre-agent validator produced no sweep id" >&2; exit 1; }
test -n "${_MANIFEST_PROJECT}" || { echo "FATAL: validated manifest carries no wandb_project" >&2; exit 1; }

for _forbidden in "${FORBIDDEN_PRODUCTION_SWEEP_ID}" "${FORBIDDEN_DISPOSABLE_REHEARSAL_SWEEP_ID}"; do
    if [ "${VALIDATED_SWEEP_ID}" = "${_forbidden}" ]; then
        echo "REFUSING: validated sweep id ${VALIDATED_SWEEP_ID} is forbidden for every v2 production path" >&2
        exit 1
    fi
done

# >>> flashnh wandb project/entity manifest authority >>>
# The strict mode=production manifest is authoritative for the W&B
# project/entity. A non-empty inherited value that disagrees with the
# loader-validated manifest is refused before any W&B contact; otherwise the
# manifest's values are adopted and exported. This launcher never invents an
# independent default.
if [ -n "${WANDB_PROJECT:-}" ] && [ "${WANDB_PROJECT}" != "${_MANIFEST_PROJECT}" ]; then
    echo "REFUSING: inherited WANDB_PROJECT='${WANDB_PROJECT}' contradicts the validated manifest project '${_MANIFEST_PROJECT}'" >&2
    exit 1
fi
if [ -n "${WANDB_ENTITY:-}" ] && [ -n "${_MANIFEST_ENTITY}" ] && [ "${WANDB_ENTITY}" != "${_MANIFEST_ENTITY}" ]; then
    echo "REFUSING: inherited WANDB_ENTITY='${WANDB_ENTITY}' contradicts the validated manifest entity '${_MANIFEST_ENTITY}'" >&2
    exit 1
fi
if [ -n "${WANDB_ENTITY:-}" ] && [ -z "${_MANIFEST_ENTITY}" ]; then
    echo "REFUSING: inherited WANDB_ENTITY='${WANDB_ENTITY}' but the validated manifest carries no entity" >&2
    exit 1
fi
WANDB_PROJECT="${_MANIFEST_PROJECT}"
export WANDB_PROJECT
if [ -n "${_MANIFEST_ENTITY}" ]; then
    WANDB_ENTITY="${_MANIFEST_ENTITY}"
    export WANDB_ENTITY
else
    unset WANDB_ENTITY || true
fi
# <<< flashnh wandb project/entity manifest authority <<<

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

# >>> flashnh wandb-agent credential availability (observed-failure hardening) >>>
# The `wandb agent` CLI login path does NOT read $HOME/.netrc (only
# wandb.Api() does). A submitted job whose environment lacks WANDB_API_KEY
# previously failed here with "No API key configured" and created no W&B run.
# Resolution order, without ever printing the value or enabling tracing:
#   1. an existing non-empty WANDB_API_KEY in the job environment is used as-is;
#   2. otherwise the already-configured user key is read from $HOME/.netrc
#      (machine api.wandb.ai) via the Python standard-library parser;
#   3. otherwise fail now, before any W&B contact.
# We only assert that a non-empty configured credential was obtained; W&B
# remains authoritative for whether it actually authenticates.
if [ -z "${WANDB_API_KEY:-}" ]; then
    WANDB_API_KEY="$("${CANONICAL_PYTHON}" -c 'import netrc, sys
try:
    a = netrc.netrc().authenticators("api.wandb.ai")
except Exception:
    a = None
sys.stdout.write(a[2] if a and a[2] else "")' 2>/dev/null)" || WANDB_API_KEY=""
    export WANDB_API_KEY
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "FATAL: no W&B API key available for the sweep-agent login (need WANDB_API_KEY in the job environment or an api.wandb.ai entry in \$HOME/.netrc); refusing before any W&B contact" >&2
    exit 1
fi
# <<< flashnh wandb-agent credential availability <<<

# This launcher consumes exactly one controller proposal, targeting the
# sweep id derived from the loader-validated mode=production manifest.
wandb agent --count 1 "${VALIDATED_SWEEP_ID}"

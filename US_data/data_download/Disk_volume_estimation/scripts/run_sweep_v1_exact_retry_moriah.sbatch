#!/bin/bash
# One Slurm allocation runs exactly one exact-proposal retry of an
# already-INVALID Sweep-v1 attempt via scripts/run_sweep_v1_exact_retry_bridge.py
# invoked DIRECTLY -- never `wandb agent`, never a Bayesian-controller
# proposal request. The bridge itself joins the existing production sweep
# via `wandb.init(settings=wandb.Settings(sweep_id=...))`, a mechanism
# empirically qualified against real wandb 0.28.1 in
# scripts/wandb_exact_retry_join_qualification.py (Moriah job 45939203, all
# 16 checks passed): a manually joined run is incorporated into the sweep's
# Bayesian-visible run population without consuming a controller proposal.
# One allocation retries exactly one attempt with its exact original five
# hyperparameters and Seed A; this launcher performs no loop, no array, and
# never resubmits itself.
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=flashnh-sweep-v1-exact-retry
set -euo pipefail
: "${FLASHNH_RETRY_FROZEN_PROPOSAL_RECORD:?set the original attempt's own execution_provenance.json path}"
: "${FLASHNH_RETRY_EXPECTED_IDENTITY:?set the operator-authored pinned-identity JSON path}"
: "${FLASHNH_RETRY_EXECUTION_GENERATION:?set the strictly-later execution_generation (2 for the first retry)}"
: "${FLASHNH_RETRY_PACKAGE_ROOT:?set the production package root}"
: "${FLASHNH_RETRY_SCREENING_BASIN_IDS:?set the screening basin ids path}"
: "${FLASHNH_RETRY_OUTPUT_ROOT:?set the output root for this retry (never the original attempt's own directory)}"
: "${FLASHNH_RETRY_PROJECT:?set the W&B project}"
: "${FLASHNH_RETRY_ENTITY:?set the W&B entity}"
# Optional: operator-authored JSON array of prior recorded attempts for this
# trial family (e.g. a failed attempt that crashed before wandb.init() and
# left no other durable trace, such as attempt002/job 45939764). Left unset
# for a first retry (generation 2); required in practice for any subsequent
# retry of an already-attempted-and-failed generation.
FLASHNH_RETRY_PRIOR_ATTEMPTS="${FLASHNH_RETRY_PRIOR_ATTEMPTS:-}"
case "$FLASHNH_RETRY_EXECUTION_GENERATION" in
    ''|*[!0-9]*)
        echo "FLASHNH_RETRY_EXECUTION_GENERATION must be a positive integer, got '$FLASHNH_RETRY_EXECUTION_GENERATION'" >&2
        exit 1 ;;
    0|1)
        echo "FLASHNH_RETRY_EXECUTION_GENERATION must strictly exceed the original attempt's execution_generation (1); got '$FLASHNH_RETRY_EXECUTION_GENERATION'" >&2
        exit 1 ;;
esac
prior_attempts_args=()
if [ -n "$FLASHNH_RETRY_PRIOR_ATTEMPTS" ]; then
    prior_attempts_args=(--prior-attempts "$FLASHNH_RETRY_PRIOR_ATTEMPTS")
fi
python scripts/run_sweep_v1_exact_retry_bridge.py \
    --frozen-proposal-record "$FLASHNH_RETRY_FROZEN_PROPOSAL_RECORD" \
    --expected-identity "$FLASHNH_RETRY_EXPECTED_IDENTITY" \
    --execution-generation "$FLASHNH_RETRY_EXECUTION_GENERATION" \
    "${prior_attempts_args[@]}" \
    --package-root "$FLASHNH_RETRY_PACKAGE_ROOT" \
    --screening-basin-ids "$FLASHNH_RETRY_SCREENING_BASIN_IDS" \
    --output-root "$FLASHNH_RETRY_OUTPUT_ROOT" \
    --project "$FLASHNH_RETRY_PROJECT" \
    --entity "$FLASHNH_RETRY_ENTITY"

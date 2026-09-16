#!/usr/bin/env bash
#
# RD1-C4-D1 shared Moriah environment setup, sourced by the three entry points.
# Structure follows the already-qualified rd1_c4_targeted_06911000_p2.sbatch
# (job 46169186): lmod init loop, spack/all, miniconda3/24.3.0-gcc-iqeknet,
# conda activate from an absolute env prefix, isolated repo clone, canonical
# repo unchanged check.
#
# Contains no submission of any kind. Sourcing this file never calls sbatch.

set -uo pipefail

FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
ENV_PREFIX="${ENV_PREFIX:-${FLASHNH_BASE}/envs/flashnh-moriah}"
REPO_CLONE_DIR="${REPO_CLONE_DIR:-${FLASHNH_BASE}/repos/flash-nh-rd1c4-d1}"
REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"

# Frozen strict inputs. Every one must be supplied or already exported; there
# is no discovery, no globbing, and no fallback to a "latest" anything.
D1_TRIAL_LIST="${D1_TRIAL_LIST:-${FLASHNH_BASE}/tmp/rd1_c4_d1/trial_list.json}"
D1_CONTRACT="${D1_CONTRACT:-${FLASHNH_BASE}/tmp/rd1_c4_d1/fixed_support_contract_v2.json}"
D1_PACKAGE_ROOT="${D1_PACKAGE_ROOT:-}"
D1_STORE_ROOT="${D1_STORE_ROOT:-${FLASHNH_BASE}/tmp/rd1_c4_d1/store}"
D1_OUT_DIR="${D1_OUT_DIR:-${FLASHNH_BASE}/tmp/rd1_c4_d1/reduction}"

rd1_c4_d1_require_inputs() {
    local missing=0
    for _var in D1_TRIAL_LIST D1_CONTRACT D1_PACKAGE_ROOT D1_STORE_ROOT; do
        if [ -z "${!_var:-}" ]; then
            echo "REFUSING: ${_var} is not set." >&2
            missing=1
        fi
    done
    for _path in "${D1_TRIAL_LIST}" "${D1_CONTRACT}"; do
        if [ -n "${_path}" ] && [ ! -f "${_path}" ]; then
            echo "REFUSING: required input file does not exist: ${_path}" >&2
            missing=1
        fi
    done
    if [ -n "${D1_PACKAGE_ROOT}" ] && [ ! -d "${D1_PACKAGE_ROOT}" ]; then
        echo "REFUSING: package root is not a directory: ${D1_PACKAGE_ROOT}" >&2
        missing=1
    fi
    # Sealed-scope guard: D1 reads the screening-400 validation scope only.
    # A path that names a sealed scope is refused outright rather than
    # depending on the caller having pointed somewhere safe.
    for _path in "${D1_PACKAGE_ROOT}" "${D1_TRIAL_LIST}" "${D1_CONTRACT}"; do
        case "${_path}" in
            *temporal_test*|*spatial_holdout*|*california*|*sealed*)
                echo "REFUSING: input path names a sealed/protected scope: ${_path}" >&2
                missing=1
                ;;
        esac
    done
    return ${missing}
}

rd1_c4_d1_setup_env() {
    hostname
    date -u
    echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
    echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID:-none}"
    echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-none}"

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

    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_PREFIX}"
    which python

    cd "${REPO_WORKDIR}" || return 1
    echo "--- isolated worktree git state ---"
    git -C "${REPO_CLONE_DIR}" status --short
    git -C "${REPO_CLONE_DIR}" rev-parse HEAD
    echo "--- canonical repo unchanged check ---"
    git -C "${FLASHNH_BASE}/repos/flash-nh" status --short

    # Immediate log flushing: an array task killed at the wall clock must
    # still have left its progress behind.
    export PYTHONUNBUFFERED=1
}

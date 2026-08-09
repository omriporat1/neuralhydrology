#!/usr/bin/env bash
#SBATCH --job-name=flashnh-wandb-launch-contract-qual
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/wandb-launch-contract-qual-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/wandb-launch-contract-qual-%j.err
#
# Generic W&B offline tracking LAUNCH-CONTRACT qualification (docs/
# decision_log.md, 2026-08-09 LR-A closure entry, item 12(i)). Runs
# scripts/wandb_offline_launch_contract_qualification.py on Moriah, closing
# the one scope gap the prior real-package smoke
# (scripts/wandb_real_offline_qualification_smoke.py) documented for itself:
# "Windows only" and never exercised through a real launcher's own
# WANDB_POLICY_PATH/--wandb-policy-path env-or-flag contract.
#
# CPU-only, no training, no NH/torch import: runs on `glacier` (confirmed CPU
# partition, no GRES -- see run_stage1_nh_lead06_seq24_cpu_preflight_moriah.sbatch's
# header note), never touches PACKAGE_ROOT, never generates an NH config,
# never accesses a sealed temporal-test/spatial-holdout set. This is not a
# campaign launcher and shares no code with scripts/
# run_stage1_lr_range_seedA_closure_moriah.sbatch beyond the same
# commit-pin/dirty-tree/module-load boilerplate every Stage 1 Slurm job uses.
#
# The real `wandb` package is not part of the shared flashnh-moriah conda
# env (no prior job has ever needed it -- W&B has stayed backend="null" for
# every real run so far). Rather than mutate that shared env, this job
# creates a throwaway venv layered on the module-loaded Python, installs
# `wandb`+`pyyaml` into it, runs the qualification, and removes the venv --
# the same ephemeral-install-then-uninstall pattern already used for the
# local Windows real-package smoke (docs/decision_log.md, 2026-08-02 entry).
#
# Usage:
#   EXPECTED_COMMIT=<FINAL_AUTHORIZED_COMMIT> sbatch --export=ALL,EXPECTED_COMMIT \
#       scripts/wandb_offline_launch_contract_qualification_moriah.sbatch
set -uo pipefail

EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"
if [ -z "${EXPECTED_COMMIT}" ]; then
    echo "EXPECTED_COMMIT is required (the exact commit hash you reviewed and are authorizing)." >&2
    echo "Set it explicitly, e.g.: EXPECTED_COMMIT=<hash> sbatch --export=ALL,EXPECTED_COMMIT $0" >&2
    exit 2
fi

# W&B must never go online for this qualification either. Exported
# unconditionally, same rationale as every other Stage 1 launcher.
export WANDB_MODE=offline

FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
REPO_CLONE_DIR="${PILOT_REPO_CLONE_DIR:-/sci/labs/efratmorin/omripo/PhD/Python/neuralhydrology}"
REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"

# The one thing this job exists to exercise: an explicit offline-ENABLED
# policy path, selected the same way a real launcher selects it. Defaults to
# the reviewed override this milestone adds; still overridable, exactly like
# every other Stage 1 launcher's WANDB_POLICY_PATH.
WANDB_POLICY_PATH="${WANDB_POLICY_PATH:-config/stage1_wandb_tracking_policy_offline_v001.yaml}"

RESULT_DIR="${FLASHNH_BASE}/evidence/wandb_offline_launch_contract_qualification_${SLURM_JOB_ID}"
WANDB_DIR="${WANDB_DIR:-${RESULT_DIR}/wandb_dir}"
mkdir -p "${RESULT_DIR}" "${WANDB_DIR}"
RESULT_LOG="${RESULT_DIR}/qualification_human.log"
exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== W&B offline launch-contract qualification ==="
hostname
date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-unknown}"
echo "EXPECTED_COMMIT (authorized by submitter): ${EXPECTED_COMMIT}"
echo "WANDB_POLICY_PATH: ${WANDB_POLICY_PATH}"
echo "WANDB_MODE (effective, enforced offline, non-overridable): ${WANDB_MODE}"

if ! type module &>/dev/null; then
    for _modinit in \
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
which python3
python3 --version

test -d "${REPO_WORKDIR}" || { echo "MISSING expected repo work directory: ${REPO_WORKDIR}"; exit 1; }
cd "${REPO_WORKDIR}"
git log --oneline -3
SOURCE_COMMIT=$(git rev-parse HEAD)
GIT_STATUS_PORCELAIN=$(git status --porcelain)
echo "SOURCE_COMMIT: ${SOURCE_COMMIT}"
TRACKED_CHANGES=$(echo "${GIT_STATUS_PORCELAIN}" | grep -v '^??' || true)
if [ -n "${TRACKED_CHANGES}" ]; then
    echo "REFUSING to run: tracked files are locally modified in ${REPO_WORKDIR}:" >&2
    echo "${TRACKED_CHANGES}" >&2
    exit 1
fi
if [ "${SOURCE_COMMIT}" != "${EXPECTED_COMMIT}" ]; then
    echo "REFUSING to run: repo HEAD (${SOURCE_COMMIT}) does not match the explicitly" >&2
    echo "authorized EXPECTED_COMMIT (${EXPECTED_COMMIT})." >&2
    exit 1
fi
echo "Commit pin OK: HEAD matches EXPECTED_COMMIT (${EXPECTED_COMMIT})."

# --- ephemeral venv: real wandb, installed and removed within this job -----
VENV_DIR="${RESULT_DIR}/venv"
python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet wandb pyyaml
python -c "import wandb; print('wandb', wandb.__version__, 'installed in ephemeral venv')"

echo "=== launching scripts/wandb_offline_launch_contract_qualification.py ==="
python scripts/wandb_offline_launch_contract_qualification.py \
    --wandb-policy-path "${WANDB_POLICY_PATH}" \
    --wandb-dir "${WANDB_DIR}" \
    --out-dir "${RESULT_DIR}"
QUAL_STATUS=$?

deactivate || true
rm -rf "${VENV_DIR}"

echo "=== done (exit ${QUAL_STATUS}) ==="
exit "${QUAL_STATUS}"

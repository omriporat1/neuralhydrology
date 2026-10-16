#!/usr/bin/env bash
#SBATCH --job-name=flashnh-seqlen-c336
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=50:00:00
#SBATCH --array=0-14
#SBATCH --chdir=/sci/labs/efratmorin/omripo/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation/.scratch_local/seqlen_common336_seed3_v001/logs/%x-%A_%a.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation/.scratch_local/seqlen_common336_seed3_v001/logs/%x-%A_%a.err
#
# seqlen_common336_seed3_v001: 15 fresh trials (L in {72,120,168,240,336} x
# seeds A/B/C), one array task per trial, task index = campaign_manifest.json
# row index (seed-major: 0-4 seed A, 5-9 seed B, 10-14 seed C). Relaunches
# pass --array/--time per length on the sbatch command line (overriding the
# header); start order follows Slurm priority and capacity, never seed order.
# Never contacts W&B or a sweep controller; every
# config declares flashnh_observation_cutoff, so no observation after the
# validation period is read and the test period is refused.
#
# PREPARED FOR REVIEW ONLY -- not submitted by the preparation milestone.
# Submission requires the user's approval of the batch and resource estimate,
# and a reviewed commit (EXPECTED_COMMIT) whose clean tracked tree is pinned by
# src.baseline.sweep_v1_runtime_contract inside the runner.
#
# Required environment:
#   EXPECTED_COMMIT  full 40-hex commit this launch is pinned to
# Fixed (not overridable): checkout, interpreter, campaign manifest.
set -euo pipefail

: "${EXPECTED_COMMIT:?set the full 40-hex git commit this launch is pinned to}"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo "EXPECTED_COMMIT is not a 40-hex commit" >&2; exit 1; }
: "${SLURM_ARRAY_TASK_ID:?this launcher must run as a Slurm array task}"

FLASHNH_BASE=/sci/labs/efratmorin/omripo/Flash-NH
REPO_WORKDIR="${FLASHNH_BASE}/repos/flash-nh/US_data/data_download/Disk_volume_estimation"
CANONICAL_PYTHON="${FLASHNH_BASE}/envs/flashnh-moriah/bin/python"
CAMPAIGN_ROOT="${REPO_WORKDIR}/.scratch_local/seqlen_common336_seed3_v001"
# v002 manifest (schema 2: derived seed B, observation cutoff). The v001
# training_output/ (seed B 1729, no cutoff) is superseded and never launched.
CAMPAIGN_MANIFEST="${CAMPAIGN_ROOT}/training_output_v002/campaign_manifest.json"

# No W&B; caches stay under the project root.
export WANDB_MODE=disabled WANDB_DISABLED=true
export XDG_CACHE_HOME="${CAMPAIGN_ROOT}/cache" MPLCONFIGDIR="${CAMPAIGN_ROOT}/cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}"
# User-approved narrow exception (2026-10-16): TMPDIR is a private per-task
# directory in the compute node's /tmp, for ephemeral multiprocessing / torch
# DataLoader IPC only (AF_UNIX sockets under pymp-*/). An AF_UNIX path is at
# most 107 bytes and the worker socket is $TMPDIR/pymp-XXXXXXXX/listener-XXXXXXXX
# (+33), while every project-local path is >= 98 chars: the 6d1efc6 project-local
# TMPDIR (154 chars) made DataLoader workers fail and epoch 1 hang. Datasets,
# logs, checkpoints and results never go here. SLURM_JOB_ID is unique per array
# task; only this task's directory is removed on exit.
IPC_TMPDIR="/tmp/fnh-$(id -un)-${SLURM_JOB_ID:?}"
(( ${#IPC_TMPDIR} <= 74 )) || { echo "IPC TMPDIR too long: ${IPC_TMPDIR}" >&2; exit 97; }
mkdir -m 700 "${IPC_TMPDIR}"
cleanup_ipc_tmpdir() { [[ "${IPC_TMPDIR}" == /tmp/fnh-* && -d "${IPC_TMPDIR}" ]] && rm -rf -- "${IPC_TMPDIR}"; return 0; }
trap cleanup_ipc_tmpdir EXIT
trap 'exit 143' TERM
export TMPDIR="${IPC_TMPDIR}"

echo "=== seqlen_common336 trial ${SLURM_ARRAY_TASK_ID} ==="; hostname; date -u
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-} ARRAY=${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID} EXPECTED_COMMIT=${EXPECTED_COMMIT}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
cd "${REPO_WORKDIR}"
test -x "${CANONICAL_PYTHON}" && test -f "${CAMPAIGN_MANIFEST}"

# IPC preflight: fail within minutes instead of hanging. Same interpreter and
# worker count as training; each worker creates a real Listener under this
# TMPDIR and reports its socket path length; the worker-collated tensors then
# cross processes through the same shared-memory fd channel training uses.
echo "IPC TMPDIR=${TMPDIR} (len ${#TMPDIR})"
timeout 900 "${CANONICAL_PYTHON}" -I -c '
import torch
from multiprocessing.connection import Listener
from torch.utils.data import DataLoader, Dataset
class WorkerSocket(Dataset):
    def __len__(self): return 8
    def __getitem__(self, i):
        with Listener() as listener:
            return len(listener.address.encode())
lengths = torch.cat(list(DataLoader(WorkerSocket(), batch_size=1, num_workers=4, timeout=60))).tolist()
assert len(lengths) == 8 and max(lengths) <= 107, lengths
print(f"IPC_PREFLIGHT_OK worker_socket_path_bytes_max={max(lengths)} limit=107", flush=True)
' || { echo "IPC preflight failed under TMPDIR=${TMPDIR}" >&2; exit 97; }

# No exec: the EXIT trap must run to remove this task's IPC directory.
"${CANONICAL_PYTHON}" scripts/run_seqlen_common336_trial.py \
    --campaign-manifest "${CAMPAIGN_MANIFEST}" \
    --index "${SLURM_ARRAY_TASK_ID}" \
    --expected-commit "${EXPECTED_COMMIT}" \
    --expected-runtime-python "${CANONICAL_PYTHON}"

#!/usr/bin/env bash
#SBATCH --job-name=flashnh-fullpop-lead06-seq24-seed-train-v001
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/fullpop-lead06-seq24-seed-train-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/fullpop-lead06-seq24-seed-train-%j.err
#
# Stage 1 full-population initial scientific SEED training (Part D of the
# 2026-07-25 "train and evaluate one real full-population seed model"
# directive; see docs/decision_log.md's "initial full-population seed
# training profile" entry). Trains the frozen
# initial_full_population_seed_v001 profile (cudalstm, hidden_size 128,
# dropout 0.25, batch_size 256, Adam lr 1e-3, NSE loss, 40 epochs,
# checkpoint+validate every epoch, validate_n_random_basins = full 2,307-
# basin development population) against the DEVELOPMENT bundle only
# (2,307 basins, train 2020-10-14..2023-12-31). Never the spatial_holdout
# bundle -- see the explicit guard below, backed by
# src.baseline.nh_config_generation.raise_if_holdout_bundle.
#
# Does not do checkpoint selection, does not evaluate the temporal-test or
# spatial-holdout periods, and does not run more than one seed -- this
# script's only job is to produce per-epoch checkpoints on development data.
# Checkpoint selection (development-validation-only) and held-out evaluation
# are separate steps (Part E / Part F), run only after this job's evidence is
# reviewed.
#
# Partition/GRES: defaults above (--partition=catfish --gres=gpu:l4:1) are
# the only Moriah GPU class this repo has confirmed working
# (run_stage1_smoke0/1_moriah.sbatch, the paired GPU-smoke launcher). Override
# at submission time with `sbatch --partition=... --gres=... this_script` if
# L4 proves inadequate (OOM / infeasible epoch runtime) -- the directive
# permits exactly ONE narrow, documented resource correction; do not tune
# model hyperparameters in response to a resource problem.
#
# Restart-safety: full-population training (2,307 basins x ~45,720 hourly
# timestamps, 40 epochs) may not fit inside one 24h Slurm allocation. This
# script persists the *actual* NeuralHydrology-generated run directory to a
# fixed marker file (RUN_DIR_MARKER_FILE, NOT job-ID-scoped) within seconds of
# starting training -- well before the first epoch completes -- so that if
# this job is killed at the wall-time limit, resubmitting the exact same
# script detects the existing run + its latest on-disk checkpoint and calls
# `scripts/run_stage1_nh.py continue <run_dir>` (neuralhydrology.nh_run.
# continue_run) instead of starting over from epoch 0. save_weights_every=1
# in the seed profile means at most one epoch of progress is ever at risk.
# #SBATCH --signal=B:TERM@180 delivers SIGTERM ~3 minutes before the wall-time
# kill so this is logged, not silently truncated (NH itself is not asked to
# intercept the signal; the next resubmission's `continue` picks up wherever
# the last on-disk checkpoint left off).

set -uo pipefail

FLASHNH_BASE="/sci/labs/efratmorin/omripo/Flash-NH"
ENV_PREFIX="${FLASHNH_BASE}/envs/flashnh-moriah"
REPO_CLONE_DIR="${FLASHNH_BASE}/repos/flash-nh"
REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"

TRANSFERRED_PACKAGE="${FLASHNH_BASE}/data/stage1_scientific_package_v002"
GENERATED_OUT_DIR="${FLASHNH_BASE}/runs/stage1_full_population_nh_config_lead06_seq24_seed_v001"
DEVELOPMENT_DIR="${GENERATED_OUT_DIR}/development"
CONFIG_YAML="${DEVELOPMENT_DIR}/config.yaml"

# Fixed (not $SLURM_JOB_ID-scoped) so every resubmission of this script finds
# the SAME run identity -- this is what makes `continue` possible across
# separate Slurm allocations. Part C creates DEVELOPMENT_DIR before this job
# is ever submitted; this job never (re)generates the config itself.
RUN_BASE="${DEVELOPMENT_DIR}/runs"
RUN_DIR_MARKER_FILE="${GENERATED_OUT_DIR}/SEED_TRAIN_ACTIVE_RUN_DIR.txt"

RESULT_DIR="${FLASHNH_BASE}/evidence/fullpop_lead06_seq24_seed_train_${SLURM_JOB_ID}"
mkdir -p "${RESULT_DIR}"
RESULT_LOG="${RESULT_DIR}/seed_train_human.log"
exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== part D: full-population seed training (development bundle only) ==="
hostname
date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-unknown}"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unknown}"
scontrol show job "${SLURM_JOB_ID}" || true

test -d "${TRANSFERRED_PACKAGE}" || { echo "MISSING transferred package: ${TRANSFERRED_PACKAGE}"; exit 1; }
test -f "${CONFIG_YAML}" || { echo "MISSING development config (run Part C generation first): ${CONFIG_YAML}"; exit 1; }

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

cd "${REPO_WORKDIR}"
git log --oneline -3
SOURCE_COMMIT=$(git rev-parse HEAD)

echo "=== guard: refuse to train against a spatial-holdout (test-only) bundle ==="
python -c "
import sys
sys.path.insert(0, '.')
from src.baseline.nh_config_generation import raise_if_holdout_bundle
raise_if_holdout_bundle('${DEVELOPMENT_DIR}')
print('OK: ${DEVELOPMENT_DIR} is not a holdout bundle')
" || { echo "REFUSED: development dir failed the holdout-bundle guard"; exit 1; }

echo "=== guard: config.yaml must render the initial full-population seed profile, not the compact-smoke profile ==="
python -c "
import sys, yaml
sys.path.insert(0, '.')
cfg = yaml.safe_load(open('${CONFIG_YAML}'))
assert cfg.get('epochs') == 40, f\"expected epochs=40 (seed profile), got {cfg.get('epochs')!r}\"
assert cfg.get('hidden_size') == 128, f\"expected hidden_size=128 (seed profile), got {cfg.get('hidden_size')!r}\"
assert cfg.get('batch_size') == 256, f\"expected batch_size=256 (seed profile), got {cfg.get('batch_size')!r}\"
print('OK: config.yaml matches the initial_full_population_seed_v001 profile')
" || { echo "REFUSED: config.yaml did not match the expected seed profile"; exit 1; }

mkdir -p "${RUN_BASE}"

# --- restart-safe run-identity resolution ------------------------------
RUN_DIR=""
if [ -f "${RUN_DIR_MARKER_FILE}" ]; then
    CANDIDATE_RUN_DIR=$(cat "${RUN_DIR_MARKER_FILE}")
    if [ -d "${CANDIDATE_RUN_DIR}" ]; then
        LATEST_CKPT=$(find "${CANDIDATE_RUN_DIR}" -maxdepth 1 -name 'model_epoch*.pt' | sort | tail -1)
        if [ -n "${LATEST_CKPT}" ]; then
            RUN_DIR="${CANDIDATE_RUN_DIR}"
            echo "=== resuming existing run: ${RUN_DIR} (latest checkpoint: ${LATEST_CKPT}) ==="
        else
            echo "=== marker points at ${CANDIDATE_RUN_DIR} but it has no checkpoint yet; treating as fresh start ==="
        fi
    else
        echo "=== marker file present but directory missing (${CANDIDATE_RUN_DIR}); treating as fresh start ==="
    fi
fi

if [ -n "${RUN_DIR}" ]; then
    echo "=== continue training: ${RUN_DIR} ==="
    python scripts/run_stage1_nh.py continue "${RUN_DIR}" &
    RUN_PID=$!
else
    echo "=== fresh start: training from epoch 0 ==="
    BEFORE_RUN_MARKER="${RESULT_DIR}/.before_run"
    touch "${BEFORE_RUN_MARKER}"

    python scripts/run_stage1_nh.py train "${CONFIG_YAML}" &
    RUN_PID=$!

    # Poll for the newly created run dir and persist it to the fixed marker
    # file as soon as it appears (NH creates the run_dir near-immediately,
    # well before epoch 1 finishes) so a wall-time kill can never lose the
    # run identity, even if training never reaches its first checkpoint this
    # allocation.
    NEW_RUN_DIR=""
    for _i in $(seq 1 120); do
        NEW_RUN_DIR=$(find "${RUN_BASE}" -mindepth 1 -maxdepth 1 -type d -newer "${BEFORE_RUN_MARKER}" 2>/dev/null | sort | tail -1)
        if [ -n "${NEW_RUN_DIR}" ]; then
            echo "${NEW_RUN_DIR}" > "${RUN_DIR_MARKER_FILE}"
            echo "=== run directory created and recorded: ${NEW_RUN_DIR} ==="
            break
        fi
        if ! kill -0 "${RUN_PID}" 2>/dev/null; then
            echo "=== training process exited before a run directory was observed ==="
            break
        fi
        sleep 5
    done
    rm -f "${BEFORE_RUN_MARKER}"
    RUN_DIR="${NEW_RUN_DIR}"
fi

wait "${RUN_PID}"
TRAIN_STATUS=$?

echo "=== training process exit code: ${TRAIN_STATUS} ==="
echo "run directory: ${RUN_DIR:-UNKNOWN}"

LATEST_CKPT_AFTER=""
LATEST_EPOCH_AFTER=""
if [ -n "${RUN_DIR}" ] && [ -d "${RUN_DIR}" ]; then
    LATEST_CKPT_AFTER=$(find "${RUN_DIR}" -maxdepth 1 -name 'model_epoch*.pt' | sort | tail -1)
    if [ -n "${LATEST_CKPT_AFTER}" ]; then
        LATEST_EPOCH_AFTER=$(basename "${LATEST_CKPT_AFTER}" | sed -E 's/model_epoch0*([0-9]+)\.pt/\1/')
    fi
fi

# Non-zero exit with a checkpoint already on disk (e.g. wall-time SIGTERM
# after epoch 1 saved) is a resumable partial run, not an unqualified
# failure -- report both facts distinctly rather than collapsing to FAIL.
if [ "${TRAIN_STATUS}" -eq 0 ]; then
    RUN_STATUS="COMPLETED"
elif [ -n "${LATEST_CKPT_AFTER}" ]; then
    RUN_STATUS="INTERRUPTED_RESUMABLE"
else
    RUN_STATUS="FAILED_NO_CHECKPOINT"
fi

python -c "
import json
out = {
    'status': '${RUN_STATUS}',
    'train_exit_code': ${TRAIN_STATUS},
    'config_yaml': '${CONFIG_YAML}',
    'run_dir': '${RUN_DIR}',
    'latest_checkpoint': '${LATEST_CKPT_AFTER}',
    'latest_epoch': '${LATEST_EPOCH_AFTER}' or None,
    'run_dir_marker_file': '${RUN_DIR_MARKER_FILE}',
    'slurm_job_id': '${SLURM_JOB_ID}',
    'slurm_job_partition': '${SLURM_JOB_PARTITION:-unknown}',
    'source_commit': '${SOURCE_COMMIT}',
    'hostname': __import__('socket').gethostname(),
}
json.dump(out, open('${RESULT_DIR}/seed_train_result.json', 'w'), indent=2)
print(json.dumps(out, indent=2))
"

echo "=== done (run_status ${RUN_STATUS}, exit ${TRAIN_STATUS}) ==="
if [ "${RUN_STATUS}" = "FAILED_NO_CHECKPOINT" ]; then
    exit 1
fi
exit 0

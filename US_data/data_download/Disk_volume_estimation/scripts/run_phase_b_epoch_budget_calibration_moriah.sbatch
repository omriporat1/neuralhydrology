#!/usr/bin/env bash
#SBATCH --job-name=flashnh-epochcal
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/epochcal-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/epochcal-%j.err
# Future single-candidate launcher. Do not submit until C1 is separately authorized.
set -euo pipefail

RUN_ID="${1:-}"
case "${RUN_ID}" in C1_anchor|C2_low_lr|C3_high_lr|C4_late_h64|C5_convergence_stress) ;; *) echo "usage: sbatch $0 <C1_anchor|C2_low_lr|C3_high_lr|C4_late_h64|C5_convergence_stress>" >&2; exit 2;; esac
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"
[ -n "${EXPECTED_COMMIT}" ] || { echo "EXPECTED_COMMIT is required." >&2; exit 2; }
readonly FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
readonly REPO_CLONE_DIR="${FLASHNH_REPO_CLONE_DIR:-${FLASHNH_BASE}/repos/flash-nh}"
readonly REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"
readonly MORIAH_PYTHON="/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah/bin/python"
readonly PACKAGE_ROOT="${PILOT_PACKAGE_ROOT:-${FLASHNH_BASE}/data/stage1_scientific_package_v002}"
readonly SCREENING_ARTIFACT="${PHASEB_SCREENING_ARTIFACT:-${FLASHNH_BASE}/data/screening_subsets/stage1_provisional_operational_screening_subset_v001/screening_subset_basin_ids.txt}"
readonly CONFIG_ROOT="${FLASHNH_BASE}/runs/phase_b_epoch_budget_calibration_v001/${RUN_ID}"
readonly EVIDENCE_ROOT="${FLASHNH_BASE}/evidence/phase_b_epoch_budget_calibration_v001/${RUN_ID}"

test -d "${REPO_WORKDIR}" && test -d "${PACKAGE_ROOT}" && test -f "${SCREENING_ARTIFACT}" && test -x "${MORIAH_PYTHON}" || exit 1
readonly REPOSITORY=$(git -C "${REPO_WORKDIR}" rev-parse --show-toplevel)
test -z "$(git -C "${REPOSITORY}" status --porcelain | grep -v '^??' || true)" || exit 1
test "$(git -C "${REPOSITORY}" rev-parse HEAD)" = "${EXPECTED_COMMIT}" || exit 1
cd "${REPO_WORKDIR}"
# The committed tracking policy is disabled; this launcher never opts into W&B.
PHASEB_SCREENING_ARTIFACT="${SCREENING_ARTIFACT}" "${MORIAH_PYTHON}" -u - "${RUN_ID}" "${PACKAGE_ROOT}" "${CONFIG_ROOT}" "${EVIDENCE_ROOT}" <<'PY'
import importlib.util, os, sys
from pathlib import Path
root = Path.cwd()
spec = importlib.util.spec_from_file_location('epochcal', root / 'scripts' / 'prepare_phase_b_epoch_budget_calibration.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
from src.baseline.pilot_lead06_config import load_pilot_policy
from src.baseline.pilot_orchestration import run_pilot
policy = module.build_epoch_budget_calibration_policy(module._resolve_policy_relative_paths(
    load_pilot_policy(root / 'config' / 'stage1_lead06_pilot_v001.yaml'),
    screening_artifact_path=Path(os.environ['PHASEB_SCREENING_ARTIFACT']),
))
result = run_pilot(pilot_policy=policy, run_id=sys.argv[1], baseline_policy_path=root / 'config' / 'stage1_scientific_baseline_v001.yaml', package_root=Path(sys.argv[2]), splits_dir=root / 'config' / 'stage1_baseline_splits_v001', config_out_dir=Path(sys.argv[3]), evidence_out_dir=Path(sys.argv[4]), slurm_identity={'job_id': os.environ.get('SLURM_JOB_ID'), 'node': os.environ.get('SLURMD_NODENAME'), 'partition': os.environ.get('SLURM_JOB_PARTITION'), 'gres': os.environ.get('SLURM_JOB_GRES')})
print(result)
PY

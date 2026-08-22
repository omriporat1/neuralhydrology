#!/usr/bin/env bash
#SBATCH --job-name=flashnh-epochcal-preflight
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/epochcal-preflight-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/epochcal-preflight-%j.err
# CPU-only canonical-package preparation gate.  It never imports CUDA,
# invokes start_run(), or performs model/optimizer work.
set -euo pipefail

EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"
[ -n "${EXPECTED_COMMIT}" ] || { echo "EXPECTED_COMMIT is required." >&2; exit 2; }
readonly FLASHNH_BASE="${FLASHNH_BASE:-/sci/labs/efratmorin/omripo/Flash-NH}"
readonly REPO_CLONE_DIR="${FLASHNH_REPO_CLONE_DIR:-${FLASHNH_BASE}/repos/flash-nh}"
readonly REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"
readonly MORIAH_PYTHON="/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah/bin/python"
readonly PACKAGE_ROOT="${PILOT_PACKAGE_ROOT:-${FLASHNH_BASE}/data/stage1_scientific_package_v002}"
readonly SCREENING_ARTIFACT="${PHASEB_SCREENING_ARTIFACT:-${FLASHNH_BASE}/data/screening_subsets/stage1_provisional_operational_screening_subset_v001/screening_subset_basin_ids.txt}"
readonly EVIDENCE_ROOT="${EPOCHCAL_PREFLIGHT_EVIDENCE_ROOT:-${FLASHNH_BASE}/evidence/phase_b_epoch_budget_calibration_preflight_v001}"
readonly RESULT_DIR="${EVIDENCE_ROOT}/job${SLURM_JOB_ID:-unknown}"

mkdir -p "${RESULT_DIR}"
exec > >(tee -a "${RESULT_DIR}/launcher.log") 2>&1
echo "CPU PREFLIGHT ONLY -- NO CUDA, TRAINING, OR EVALUATION"
test -d "${REPO_WORKDIR}" || { echo "missing canonical repo workdir" >&2; exit 1; }
test -d "${PACKAGE_ROOT}" || { echo "missing canonical package root" >&2; exit 1; }
test -f "${SCREENING_ARTIFACT}" || { echo "missing promoted screening artifact" >&2; exit 1; }
readonly REPOSITORY=$(git -C "${REPO_WORKDIR}" rev-parse --show-toplevel)
readonly ACTUAL_COMMIT=$(git -C "${REPOSITORY}" rev-parse HEAD)
test -z "$(git -C "${REPOSITORY}" status --porcelain | grep -v '^??' || true)" || { echo "tracked working-tree changes" >&2; exit 1; }
test "${ACTUAL_COMMIT}" = "${EXPECTED_COMMIT}" || { echo "commit guard failed" >&2; exit 1; }
test -x "${MORIAH_PYTHON}" || { echo "missing required Moriah Python" >&2; exit 1; }
echo "source_commit=${ACTUAL_COMMIT} package_root=${PACKAGE_ROOT} python=${MORIAH_PYTHON}"
"${MORIAH_PYTHON}" --version
"${MORIAH_PYTHON}" - <<'PY'
import yaml, torch, neuralhydrology
assert not torch.cuda.is_initialized()
print("CPU_RUNTIME_IMPORT_PREFLIGHT: PASS (CUDA not initialized)")
PY
cd "${REPO_WORKDIR}"
"${MORIAH_PYTHON}" -u scripts/prepare_phase_b_epoch_budget_calibration.py \
  --package-root "${PACKAGE_ROOT}" --screening-artifact-path "${SCREENING_ARTIFACT}" \
  --output-dir "${RESULT_DIR}/canonical_config_audit" --canonical-package-preflight
RESULT_DIR="${RESULT_DIR}" EXPECTED_COMMIT="${EXPECTED_COMMIT}" PACKAGE_ROOT="${PACKAGE_ROOT}" "${MORIAH_PYTHON}" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['RESULT_DIR'])
audit = json.loads((root / 'canonical_config_audit' / 'config_audit.json').read_text())
required = audit['candidate_count'] == 5 and audit['target_epoch'] == 14 and audit['max_updates_per_epoch'] == 50000
required = required and audit['screening_epochs'] == list(range(1, 15)) and not audit['performance_early_stopping_enabled']
required = required and all(row['training_segment_epochs'] == 14 and row['checkpoint_save_every_epochs'] == 1 for row in audit['candidates'])
screening = audit['screening_identity']
required = required and screening['expected_count'] == 400 and screening['expected_sha256'] == 'd4395d93ebc567cf09e149c0121463d75cf4f7ecc02c07a7c4a7999763baa372'
summary = {'schema_name': 'phase_b_epoch_budget_calibration_canonical_preflight', 'status': 'PASS' if required else 'FAIL',
           'source_commit_expected': os.environ['EXPECTED_COMMIT'], 'package_root': os.environ['PACKAGE_ROOT'],
           'audit_path': str(root / 'canonical_config_audit' / 'config_audit.json'), 'candidate_count': audit['candidate_count'],
           'screening_identity': screening,
           'no_training': True, 'no_cuda': True, 'no_sealed_scope': audit['no_sealed_scope']}
(root / 'canonical_preflight_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
raise SystemExit(0 if required and summary['no_sealed_scope'] else 1)
PY

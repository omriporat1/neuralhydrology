#!/usr/bin/env bash
#SBATCH --job-name=flashnh-nh-lead06-seq24-cpu-preflight
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/sci/labs/efratmorin/omripo/Flash-NH/logs/nh-lead06-seq24-cpu-preflight-%j.out
#SBATCH --error=/sci/labs/efratmorin/omripo/Flash-NH/logs/nh-lead06-seq24-cpu-preflight-%j.err
#
# First real compact NH smoke run -- Job 1 of 2 (CPU-only construction preflight,
# no training). Runs on the `glacier` partition: confirmed CPU partition, no GRES,
# default partition (docs/stage1_neuralhydrology_preflight.md Sec 10.6 sinfo table).
# `catfish`/`gpu:l4:1` (the only partition/GRES ever used by run_stage1_smoke0/1_moriah.sbatch
# and setup_flashnh_moriah_env.sbatch) is a GPU-only convention and intentionally NOT reused
# here -- this job never touches CUDA/torch.cuda, only real FlashNHDataset construction.
#
# Regenerates one lead06/seq24 NH config bundle against the Moriah-transferred certified
# package (the h2o-generated bundle from Deliverable 2 has h2o-absolute paths baked into
# config.yaml -- data_dir/train_basin_file/etc. -- and is not usable as-is on Moriah), then
# runs the existing, unmodified scripts/check_stage1_nh_config_preflight.py WITHOUT
# --skip-dataset-construction, so Layer 2 (real FlashNHDataset train/validation/test
# construction, fresh scaler={} for train + reused for validation/test, finite-value checks,
# filter stats, single-target check) executes for real. No new check logic is added here;
# this script is orchestration only, per "do not refactor the config generator or preflight."
#
# Precondition: the certified compact package + this repo (checked out at the commit
# documented in the transfer marker) must already be transferred/cloned onto Moriah
# (Deliverable 3 / repo clone step). This job does not train -- see the paired GPU-smoke
# script (run_stage1_nh_lead06_seq24_gpu_smoke_moriah.sbatch) for that.

set -uo pipefail

FLASHNH_BASE="/sci/labs/efratmorin/omripo/Flash-NH"
ENV_PREFIX="${FLASHNH_BASE}/envs/flashnh-moriah"
# Repo clone location: not yet confirmed as executed on Moriah (no prior job ever needed
# repo source there -- Smoke 0/1 only used the pip-installed `nh-run` entry point). Chosen
# for consistency with the one CONFIRMED clone-of-this-repo layout (h2o,
# docs/stage1_h2o_operations_preflight.md: /data42/omrip/Flash-NH/repos/flash-nh/US_data/...).
# If you already cloned this repo elsewhere on Moriah, edit REPO_WORKDIR below instead of
# re-cloning.
REPO_CLONE_DIR="${FLASHNH_BASE}/repos/flash-nh"
REPO_WORKDIR="${REPO_CLONE_DIR}/US_data/data_download/Disk_volume_estimation"

TRANSFERRED_PACKAGE="${FLASHNH_BASE}/data/stage1_compact_scientific_package_v001"
GENERATED_OUT_DIR="${FLASHNH_BASE}/runs/stage1_nh_config_lead06_seq24_v001"
RESULT_DIR="${FLASHNH_BASE}/evidence/nh_lead06_seq24_cpu_preflight_${SLURM_JOB_ID}"

mkdir -p "${RESULT_DIR}"
RESULT_JSON="${RESULT_DIR}/cpu_preflight_result.json"
RESULT_LOG="${RESULT_DIR}/cpu_preflight_human.log"

exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "=== preflight ==="
hostname
date -u
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "partition: ${SLURM_JOB_PARTITION:-unknown}"

# Initialize module system for non-interactive Slurm shells (same fix applied in
# setup_flashnh_moriah_env.sbatch / run_stage1_smoke0_moriah.sbatch).
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
# No nvidia/cuda modules: this job is CPU-only and never calls torch.cuda.
# (nh_dataset.py / nh_structural_preflight.py never read cfg.device.)

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
which python
python --version

echo "=== environment info ==="
python -u -c "
import sys
print('Python:', sys.version.split()[0])
import torch
print('PyTorch:', torch.__version__)
print('cuda available (expected False on this CPU-only partition):', torch.cuda.is_available())
import neuralhydrology
print('neuralhydrology import OK (no __version__ attribute is expected on 1.13)')
"

echo "=== verify transferred inputs ==="
FAIL=0
test -d "${REPO_WORKDIR}" && echo "repo workdir OK: ${REPO_WORKDIR}" || { echo "MISSING repo workdir: ${REPO_WORKDIR}"; FAIL=1; }
test -d "${TRANSFERRED_PACKAGE}" && echo "transferred package OK: ${TRANSFERRED_PACKAGE}" || { echo "MISSING transferred package: ${TRANSFERRED_PACKAGE}"; FAIL=1; }
if [ "${FAIL}" -ne 0 ]; then
    echo '{"status": "FAIL", "reason": "missing transferred inputs"}' > "${RESULT_JSON}"
    exit 1
fi

cd "${REPO_WORKDIR}"

echo "=== git state ==="
git log --oneline -3
git status --short

echo "=== regenerate lead06/seq24 config against the Moriah-local package copy ==="
python scripts/generate_stage1_nh_config.py \
    --package-root "${TRANSFERRED_PACKAGE}" \
    --lead-hours 6 \
    --seq-length 24 \
    --out-dir "${GENERATED_OUT_DIR}" \
    --force
GEN_STATUS=$?
if [ "${GEN_STATUS}" -ne 0 ]; then
    echo '{"status": "FAIL", "reason": "config generation failed"}' > "${RESULT_JSON}"
    exit 1
fi

CONFIG_YAML="${GENERATED_OUT_DIR}/config.yaml"

# --- smoke-only static-attribute adjustment ------------------------------
# Real Moriah evidence (CPU-preflight job 45624540 + diagnostic job 45624596)
# found exactly 13 of the package's 473 static columns have zero (or NaN)
# standard deviation across this compact package's 32-basin subset, which NH
# cannot normalize. This is a property of the small compact smoke population,
# not a package defect -- the full 473-column package stays authoritative.
# This block recomputes std over the transferred package's real
# attributes.csv, requires the zero/NaN-std set to equal exactly the 13
# columns confirmed above, drops only those from this generated config's
# static_attributes list (order preserved), and writes adjustment evidence.
# Must not be read as a change to the package contract or to the generator's
# normal 473-column validation.
echo "=== smoke-only static-attribute adjustment (32-basin zero-variance exclusion) ==="
ADJUSTMENT_EVIDENCE="${RESULT_DIR}/smoke_static_adjustment.json"
python -u -c "
import csv
import hashlib
import json
import math
import os
import sys
import yaml

config_path = '${CONFIG_YAML}'
attributes_csv = '${TRANSFERRED_PACKAGE}/attributes/attributes.csv'
evidence_path = '${ADJUSTMENT_EVIDENCE}'

EXPECTED_EXCLUDED = {
    'CANALS_MAINSTEM_PCT', 'CDL_DURUM_WHEAT', 'CDL_ORANGES', 'CDL_RICE', 'HGBC',
    'PCT_6TH_ORDER_OR_MORE', 'glc_pc_u01', 'glc_pc_u18', 'pnv_pc_u02',
    'wet_pc_u02', 'wet_pc_u03', 'wet_pc_u07', 'wet_pc_u09',
}

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        h.update(fh.read())
    return h.hexdigest()

with open(config_path, 'r', encoding='utf-8') as fh:
    config = yaml.safe_load(fh)

original_sha256 = sha256_of(config_path)
static_attributes = config.get('static_attributes')
if not isinstance(static_attributes, list):
    print('FATAL: config.yaml has no static_attributes list', file=sys.stderr)
    sys.exit(1)
original_count = len(static_attributes)
if original_count != 473:
    print(f'FATAL: expected 473 static_attributes in generated config, got {original_count}', file=sys.stderr)
    sys.exit(1)

with open(attributes_csv, 'r', newline='', encoding='utf-8') as fh:
    reader = csv.reader(fh)
    header = next(reader)
    if header[0] != 'gauge_id':
        print(f'FATAL: expected gauge_id as first attributes.csv column, got {header[0]!r}', file=sys.stderr)
        sys.exit(1)
    col_index = {name: i for i, name in enumerate(header[1:])}
    rows = [row for row in reader]

missing = [c for c in static_attributes if c not in col_index]
if missing:
    print(f'FATAL: {len(missing)} config static_attributes missing from attributes.csv header', file=sys.stderr)
    sys.exit(1)

nonfinite_cells = 0
zero_or_nan_std_columns = []
for name in static_attributes:
    idx = col_index[name]
    values = []
    for row in rows:
        raw = row[idx + 1]
        try:
            v = float(raw)
        except ValueError:
            v = float('nan')
        if not math.isfinite(v):
            nonfinite_cells += 1
        values.append(v)
    finite_values = [v for v in values if math.isfinite(v)]
    if len(finite_values) < 2:
        zero_or_nan_std_columns.append(name)
        continue
    mean = sum(finite_values) / len(finite_values)
    variance = sum((v - mean) ** 2 for v in finite_values) / (len(finite_values) - 1)
    std = math.sqrt(variance)
    if std == 0.0 or math.isnan(std):
        zero_or_nan_std_columns.append(name)

actual_excluded = set(zero_or_nan_std_columns)
if nonfinite_cells != 0:
    print(f'FATAL: expected 0 nonfinite static cells, found {nonfinite_cells}', file=sys.stderr)
    sys.exit(1)
if actual_excluded != EXPECTED_EXCLUDED:
    print('FATAL: zero/NaN-std column set does not match the confirmed 13-column set', file=sys.stderr)
    print(f'  expected: {sorted(EXPECTED_EXCLUDED)}', file=sys.stderr)
    print(f'  actual:   {sorted(actual_excluded)}', file=sys.stderr)
    sys.exit(1)

retained = [c for c in static_attributes if c not in actual_excluded]
if len(retained) != 460:
    print(f'FATAL: expected 460 retained static_attributes, got {len(retained)}', file=sys.stderr)
    sys.exit(1)

config['static_attributes'] = retained

tmp_path = config_path + '.tmp_smoke_adjust'
with open(tmp_path, 'w', encoding='utf-8') as fh:
    yaml.safe_dump(config, fh, default_flow_style=False, sort_keys=False)
os.replace(tmp_path, config_path)

adjusted_sha256 = sha256_of(config_path)

evidence = {
    'role': 'compact smoke only -- not the full-population baseline',
    'original_static_count': original_count,
    'retained_static_count': len(retained),
    'excluded_count': len(actual_excluded),
    'excluded_columns': sorted(actual_excluded),
    'package_attributes_csv': attributes_csv,
    'config_yaml': config_path,
    'original_config_sha256': original_sha256,
    'adjusted_config_sha256': adjusted_sha256,
    'note': ('This 13-column zero-variance exclusion applies only to the 32-basin '
             'compact integration smoke. The full 473-column package remains '
             'authoritative and this exclusion must not automatically be inherited '
             'by the future full-population baseline.'),
}
with open(evidence_path, 'w', encoding='utf-8') as fh:
    json.dump(evidence, fh, indent=2)
print(json.dumps(evidence, indent=2))
"
ADJUST_STATUS=$?
if [ "${ADJUST_STATUS}" -ne 0 ]; then
    python -u -c "
import json
out = {
    'status': 'FAIL',
    'reason': 'smoke-only static-attribute adjustment failed',
    'config_yaml': '${CONFIG_YAML}',
    'static_adjustment_evidence_path': '${ADJUSTMENT_EVIDENCE}',
    'slurm_job_id': '${SLURM_JOB_ID}',
    'hostname': __import__('socket').gethostname(),
}
json.dump(out, open('${RESULT_JSON}', 'w'), indent=2)
print(json.dumps(out, indent=2))
"
    echo "=== done (exit ${ADJUST_STATUS}) ==="
    exit ${ADJUST_STATUS}
fi

echo "=== run structural preflight (Layer 1 + Layer 2 real FlashNHDataset construction) ==="
python scripts/check_stage1_nh_config_preflight.py \
    --generated-dir "${GENERATED_OUT_DIR}" \
    --package-root "${TRANSFERRED_PACKAGE}" \
    --expected-basin-count 32 \
    --expected-seq-length 24 \
    --expected-target-variable qobs_mm_per_h_lead06 \
    --expected-dynamic-inputs mrms_qpe_1h_mm rtma_2t_K rtma_2d_K rtma_2sh_kgkg rtma_10u_ms rtma_10v_ms mrms_qpe_1h_mm_gap rtma_gap \
    --expected-static-column-count 460 \
    --expected-predict-last-n 1 \
    --expected-train-start-date 14/10/2020 \
    --expected-train-end-date 31/12/2023 \
    --expected-validation-start-date 01/01/2024 \
    --expected-validation-end-date 31/12/2024 \
    --expected-test-start-date 01/01/2025 \
    --expected-test-end-date 31/12/2025 \
    --repo-root "${REPO_WORKDIR}" \
    > "${RESULT_DIR}/structural_preflight_stdout.json" 2> "${RESULT_DIR}/structural_preflight_stderr.log"
PREFLIGHT_STATUS=$?

cat "${RESULT_DIR}/structural_preflight_stdout.json"
echo "=== preflight exit code: ${PREFLIGHT_STATUS} ==="

# The preflight CLI's stdout is a JSON object followed by two human-readable
# summary print()s, so json.load() on the whole file fails with
# "Extra data". Parse only the first JSON object via raw_decode, and always
# write cpu_preflight_result.json -- including when parsing itself fails --
# so the CPU job never silently omits its result file.
python -u -c "
import json
import sys

stdout_path = '${RESULT_DIR}/structural_preflight_stdout.json'
try:
    with open(stdout_path, 'r', encoding='utf-8') as fh:
        text = fh.read()
    d, _ = json.JSONDecoder().raw_decode(text)
    status = d.get('status', 'FAIL')
    out = {
        'status': status,
        'error_count': d.get('error_count'),
        'warning_count': d.get('warning_count'),
        'ok_count': d.get('ok_count'),
        'config_yaml': '${CONFIG_YAML}',
        'static_count_used': 460,
        'static_adjustment_evidence_path': '${ADJUSTMENT_EVIDENCE}',
        'generated_out_dir': '${GENERATED_OUT_DIR}',
        'slurm_job_id': '${SLURM_JOB_ID}',
        'hostname': __import__('socket').gethostname(),
    }
    exit_ok = (status == 'PASS')
except Exception as exc:
    out = {
        'status': 'FAIL',
        'reason': f'could not parse structural preflight output: {exc}',
        'config_yaml': '${CONFIG_YAML}',
        'static_count_used': 460,
        'static_adjustment_evidence_path': '${ADJUSTMENT_EVIDENCE}',
        'generated_out_dir': '${GENERATED_OUT_DIR}',
        'slurm_job_id': '${SLURM_JOB_ID}',
        'hostname': __import__('socket').gethostname(),
    }
    exit_ok = False

json.dump(out, open('${RESULT_JSON}', 'w'), indent=2)
print(json.dumps(out, indent=2))
sys.exit(0 if exit_ok else 1)
"
FINAL_STATUS=$?

echo "=== done (exit ${FINAL_STATUS}) ==="
exit ${FINAL_STATUS}

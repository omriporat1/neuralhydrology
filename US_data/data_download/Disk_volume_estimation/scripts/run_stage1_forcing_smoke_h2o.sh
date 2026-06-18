#!/usr/bin/env bash
# Stage 1 Forcing — SMOKE TEST launcher for h2o
#
# PURPOSE
# -------
# Runs a 48-hour × 10-basin smoke test of the forcing extraction pipeline
# before committing to the full 45,720-hour TB-scale run.
#
# Smoke parameters:
#   - Period:   2020-10-14T00:00:00 → 2020-10-15T23:00:00 (48 hours)
#   - Basins:   first 10 in the v001 basin list (--max-basins 10)
#   - Workers:  4 (conservative for smoke test only)
#   - Products: both MRMS and RTMA
#
# Worker count policy:
#   - Smoke uses 4 workers intentionally — low load, easy to diagnose failures.
#   - Full-period run defaults to 16 workers (conservative first-run setting).
#   - Increasing to 32 is only permitted after smoke PASS + one-month evidence
#     confirms h2o load/network behavior is safe under 16 workers.
#   - Do NOT use 64+ workers without explicit PI approval.
#
# Prerequisites (must be complete before running this smoke test):
#   0. bash scripts/verify_stage1_forcing_inputs_h2o.sh  →  all PASS or WARN
#   1. python scripts/export_v001_basin_list.py           →  v001_basin_list.csv exists
#   2. [2K-A] python scripts/build_stage1_basin_weights.py --basin-list ... --out-tag v001_2752
#      NOTE: pilot 50-basin weight Parquets (pilot_mrms_weights.parquet) are NOT valid
#            for v001 — new 2,752-basin weights must be built first.
#   3. CAMELSH shapefile confirmed on h2o (location may be unknown — check:
#        find /data42 -name "CAMELSH_shapefile.shp" 2>/dev/null)
#      If absent, transfer with:
#        [local Windows] PowerShell scripts/prepare_stage1_forcing_inputs_h2o.ps1 -TransferShapefiles
#
# Usage on h2o:
#   # Option A: run in foreground (for first-time inspection):
#   bash scripts/run_stage1_forcing_smoke_h2o.sh
#
#   # Option B: run under screen (recommended):
#   screen -S flashnh-smoke bash scripts/run_stage1_forcing_smoke_h2o.sh
#
# After the run, check:
#   cat ${MANIFEST_DIR}/smoke_manifest.json
#   cat ${MANIFEST_DIR}/smoke_summary.md
#   head ${MANIFEST_DIR}/smoke_hourly_runtime_and_volume.csv
#
# Expected outcomes (all must pass before full run):
#   1. MRMS OK: 48/48 hours
#   2. RTMA OK: 48/48 hours (S3 coverage is good from 2020-10 onward)
#   3. valid_weight_fraction >= 0.99 for continental basins
#   4. No all-null weighted_mean rows
#   5. At least 8 RTMA variables present per hour
#   6. Resume test (re-run same command) → 0 new downloads, same row count
#
# Evidence bundle to pull locally after the smoke PASS:
#   scp "flashnh-h2o:${MANIFEST_DIR}/smoke_manifest.json" tmp/smoke_evidence/
#   scp "flashnh-h2o:${MANIFEST_DIR}/smoke_summary.md"    tmp/smoke_evidence/
#   scp "flashnh-h2o:${MANIFEST_DIR}/smoke_hourly_runtime_and_volume.csv" tmp/smoke_evidence/
#   scp "flashnh-h2o:${MANIFEST_DIR}/smoke_missing_files.csv" tmp/smoke_evidence/  # if exists
#   scp "flashnh-h2o:${MANIFEST_DIR}/smoke_live_run.log"  tmp/smoke_evidence/
#
# Do NOT transfer raw GRIB2 or Parquet files.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — adjust FORCING_ROOT if your layout differs
# ---------------------------------------------------------------------------

FLASHNH_ROOT="${FLASHNH_ROOT:-/data42/omrip/Flash-NH}"
FORCING_ROOT="${FLASHNH_ROOT}/tmp/stage1_forcing_fullperiod"
BASIN_LIST="${FORCING_ROOT}/v001_basin_list.csv"
MRMS_WEIGHTS="${FORCING_ROOT}/02_basin_geometries/weights/mrms/v001_2752_mrms_weights.parquet"
RTMA_WEIGHTS="${FORCING_ROOT}/02_basin_geometries/weights/rtma/v001_2752_rtma_weights.parquet"
MANIFEST_DIR="${FORCING_ROOT}/manifests"
ENV_PREFIX="${FLASHNH_ROOT}/envs/flashnh-stage1"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---------------------------------------------------------------------------
# Smoke test parameters
# ---------------------------------------------------------------------------

SMOKE_START="2020-10-14T00:00:00"
SMOKE_END="2020-10-15T23:00:00"
CHUNK_LABEL="smoke"
MAX_BASINS=10
DOWNLOAD_WORKERS=4
RTMA_MODE="selected_messages"

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

echo "============================================================"
echo "Flash-NH Stage 1 Forcing — SMOKE TEST"
echo "============================================================"
echo "Start:     ${SMOKE_START}"
echo "End:       ${SMOKE_END}  (48 hours)"
echo "Basins:    first ${MAX_BASINS} from ${BASIN_LIST}"
echo "Workers:   ${DOWNLOAD_WORKERS}"
echo "RTMA mode: ${RTMA_MODE}"
echo "Output:    ${FORCING_ROOT}"
echo "============================================================"

# Check conda is available
if ! command -v conda &>/dev/null; then
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source /opt/conda/etc/profile.d/conda.sh
    elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    else
        echo "ERROR: conda not found. Please activate manually:"
        echo "  source /opt/conda/etc/profile.d/conda.sh"
        echo "  conda activate ${ENV_PREFIX}"
        exit 1
    fi
fi

# Activate environment
conda activate "${ENV_PREFIX}" || {
    echo "ERROR: Could not activate ${ENV_PREFIX}"
    echo "Check: conda env list | grep flashnh"
    exit 1
}
# conda activate can silently leave the shell pointing at a different env
# (observed on h2o: PS1 shows flashnh-stage1 but which python → iacpy3_2025).
# Force the correct env by prepending its bin dir and rehashing.
export PATH="${ENV_PREFIX}/bin:${PATH}"
hash -r
_actual_python=$(command -v python)
if [ "${_actual_python}" != "${ENV_PREFIX}/bin/python" ]; then
    echo "ERROR: python resolves to ${_actual_python}"
    echo "       expected ${ENV_PREFIX}/bin/python"
    echo "PATH: ${PATH}"
    exit 1
fi
_py_ver=$(python --version 2>&1)
if [[ "${_py_ver}" != *"Python 3.11"* ]]; then
    echo "ERROR: Expected Python 3.11.x, got: ${_py_ver}"
    exit 1
fi
echo "Python: ${_actual_python} (${_py_ver})"

# Check required files
for fpath in "${BASIN_LIST}" "${MRMS_WEIGHTS}" "${RTMA_WEIGHTS}"; do
    if [ ! -f "${fpath}" ]; then
        echo "ERROR: Required file missing: ${fpath}"
        if [ "${fpath}" = "${BASIN_LIST}" ]; then
            echo "  Create with: python scripts/export_v001_basin_list.py  (future helper)"
            echo "  Or manually: extract STAIDs from stage1_target_package_v001/manifest.json"
        elif [[ "${fpath}" == *"weights"* ]]; then
            echo "  Build with: python scripts/build_stage1_basin_weights.py --basin-list ${BASIN_LIST} --out-tag v001_2752"
        fi
        exit 1
    fi
done

mkdir -p "${MANIFEST_DIR}"
mkdir -p "${FORCING_ROOT}/raw/mrms"
mkdir -p "${FORCING_ROOT}/raw/rtma"
mkdir -p "${FORCING_ROOT}/staging/mrms"
mkdir -p "${FORCING_ROOT}/staging/rtma"

# Check system load before proceeding
LOAD=$(uptime | awk -F'load average:' '{ print $2 }' | awk '{ print $1 }' | tr -d ',')
echo "Current system load (1-min): ${LOAD}"
echo "NOTE: h2o etiquette — keep total CPU ≤50-60%. Check htop if load is high."

# ---------------------------------------------------------------------------
# Run extraction
# ---------------------------------------------------------------------------

echo ""
echo "Starting smoke extraction ..."
echo ""

cd "${REPO_ROOT}"

python scripts/extract_stage1_forcing_chunk.py \
    --start     "${SMOKE_START}" \
    --end       "${SMOKE_END}" \
    --basin-manifest "${BASIN_LIST}" \
    --mrms-weights   "${MRMS_WEIGHTS}" \
    --rtma-weights   "${RTMA_WEIGHTS}" \
    --out-dir        "${FORCING_ROOT}" \
    --chunk-label    "${CHUNK_LABEL}" \
    --max-basins     "${MAX_BASINS}" \
    --download-workers "${DOWNLOAD_WORKERS}" \
    --rtma-mode      "${RTMA_MODE}" \
    --no-plots

EXIT_CODE=$?

echo ""
echo "============================================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "SMOKE TEST: PASS"
else
    echo "SMOKE TEST: FAIL (exit code ${EXIT_CODE})"
fi
echo "============================================================"
echo ""
echo "Key outputs:"
echo "  Manifest:    ${MANIFEST_DIR}/${CHUNK_LABEL}_manifest.json"
echo "  Summary:     ${MANIFEST_DIR}/${CHUNK_LABEL}_summary.md"
echo "  Metrics CSV: ${MANIFEST_DIR}/${CHUNK_LABEL}_hourly_runtime_and_volume.csv"
echo "  Live log:    ${MANIFEST_DIR}/${CHUNK_LABEL}_live_run.log"
echo ""
echo "To inspect the summary:"
echo "  cat ${MANIFEST_DIR}/${CHUNK_LABEL}_summary.md"
echo ""
echo "Before documenting results, pull evidence bundle to local:"
echo "  mkdir -p tmp/smoke_evidence"
echo "  scp flashnh-h2o:${MANIFEST_DIR}/${CHUNK_LABEL}_manifest.json tmp/smoke_evidence/"
echo "  scp flashnh-h2o:${MANIFEST_DIR}/${CHUNK_LABEL}_summary.md    tmp/smoke_evidence/"
echo "  scp flashnh-h2o:${MANIFEST_DIR}/${CHUNK_LABEL}_hourly_runtime_and_volume.csv tmp/smoke_evidence/"
echo "  scp flashnh-h2o:${MANIFEST_DIR}/${CHUNK_LABEL}_live_run.log  tmp/smoke_evidence/"
echo ""

exit ${EXIT_CODE}

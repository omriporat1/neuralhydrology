#!/usr/bin/env bash
# Flash-NH — WY2026 forcing extraction launcher for h2o (Jan–Sep 2026 months)
#
# PURPOSE
# -------
# Extends the accepted Stage-1 MRMS + RTMA basin-average forcing extraction
# (run_stage1_forcing_fullperiod_h2o.sh, 63 months 2020-10-14..2025-12-31) by
# the nine months that complete water year 2026: 2026-01 .. 2026-09.
#
# It is a thin wrapper that reproduces the full-period launcher's per-month
# extractor call EXACTLY (same extractor, same basin list, same v001_2752
# weights, same FORCING_ROOT, DOWNLOAD_WORKERS=6, RTMA selected_messages,
# --resume --no-plots). Only the month list differs. The full-period launcher
# and its pinned 63-month list are not modified, so the v001 product stays
# reproducible.
#
# WY2026 Oct–Dec 2025 months (2025-10, 2025-11, 2025-12) were already
# extracted and accepted in the full-period run; this wrapper REFUSES to touch
# them (the extractor has no overwrite guard). See docs/wy2026_data_plan.md.
#
# USAGE (three parallel screen streams, as in the full-period run)
# ----------------------------------------------------------------
#   Start a screen (screen -S wy2026-forcing-d), then INSIDE it run:
#     GROUP_ID=D bash scripts/run_wy2026_forcing_extraction_h2o.sh 2>&1 | tee -a /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/logs/wy2026_group_d.log
#   and likewise for E and F in screens wy2026-forcing-e / wy2026-forcing-f.
#   (Full commands: docs/wy2026_data_plan.md §9 L2.)
#
#   Deliberate difference from the full-period launcher: a failed month is
#   recorded and the loop continues with the next month (|| MONTH_EXIT=$?),
#   instead of set -e aborting the whole group.
#
#   Groups: D = 2026-01..2026-03, E = 2026-04..2026-06, F = 2026-07..2026-09.
#   Leave GROUP_ID empty to run all nine months sequentially (~25 h).
#
# Environment overrides (same meaning as the full-period launcher):
#   FLASHNH_ROOT, DOWNLOAD_WORKERS (default 6), RTMA_MODE (default
#   selected_messages), SKIP_MONTHS, GROUP_ID, DRY_RUN (1 = print plan only),
#   MAX_MONTHS, stop file ${FORCING_ROOT}/STOP_AFTER_MONTH.
#
# Resume: months whose chunk parquet exists with manifest all_pass=true are
# skipped; an interrupted month is re-run with --resume (raw GRIB cache and
# per-hour staging files are reused).

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths (identical to run_stage1_forcing_fullperiod_h2o.sh)
# ---------------------------------------------------------------------------

FLASHNH_ROOT="${FLASHNH_ROOT:-/data42/omrip/Flash-NH}"
FORCING_ROOT="${FLASHNH_ROOT}/tmp/stage1_forcing_fullperiod"
BASIN_LIST="${FORCING_ROOT}/v001_basin_list.csv"
MRMS_WEIGHTS="${FORCING_ROOT}/02_basin_geometries/weights/mrms/v001_2752_mrms_weights.parquet"
RTMA_WEIGHTS="${FORCING_ROOT}/02_basin_geometries/weights/rtma/v001_2752_rtma_weights.parquet"
MANIFEST_DIR="${FORCING_ROOT}/manifests"
ENV_PREFIX="${FLASHNH_ROOT}/envs/flashnh-stage1"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

case "${FORCING_ROOT}" in
    /data42/omrip/Flash-NH/*) ;;
    *) echo "ERROR: FORCING_ROOT must begin with /data42/omrip/Flash-NH/ (got: ${FORCING_ROOT})"
       exit 1 ;;
esac

FLASHNH_TMPDIR="${FLASHNH_ROOT}/tmp/tmpdir_flashnh"
export TMPDIR="${FLASHNH_TMPDIR}"

DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-6}"
RTMA_MODE="${RTMA_MODE:-selected_messages}"
SKIP_MONTHS="${SKIP_MONTHS:-}"
STOP_FILE="${FORCING_ROOT}/STOP_AFTER_MONTH"
GROUP_ID="${GROUP_ID:-}"
DRY_RUN="${DRY_RUN:-0}"
MAX_MONTHS="${MAX_MONTHS:-}"

# ---------------------------------------------------------------------------
# Month list: the nine WY2026 months not yet extracted
# ---------------------------------------------------------------------------

MONTH_LIST=(
    "2026-01 2026-01-01T00:00:00 2026-01-31T23:00:00"
    "2026-02 2026-02-01T00:00:00 2026-02-28T23:00:00"
    "2026-03 2026-03-01T00:00:00 2026-03-31T23:00:00"
    "2026-04 2026-04-01T00:00:00 2026-04-30T23:00:00"
    "2026-05 2026-05-01T00:00:00 2026-05-31T23:00:00"
    "2026-06 2026-06-01T00:00:00 2026-06-30T23:00:00"
    "2026-07 2026-07-01T00:00:00 2026-07-31T23:00:00"
    "2026-08 2026-08-01T00:00:00 2026-08-31T23:00:00"
    "2026-09 2026-09-01T00:00:00 2026-09-30T23:00:00"
)

# Guard: never re-extract an accepted v001 month (anything before 2026-01).
for ENTRY in "${MONTH_LIST[@]}"; do
    LBL=$(echo "${ENTRY}" | awk '{print $1}')
    if [[ "${LBL}" < "2026-01" || "${LBL}" > "2026-09" ]]; then
        echo "ERROR: month ${LBL} is outside 2026-01..2026-09; refusing (accepted v001 months must not be re-extracted)."
        exit 1
    fi
done

case "${GROUP_ID}" in
    D) GROUP_START="2026-01"; GROUP_END="2026-03" ;;
    E) GROUP_START="2026-04"; GROUP_END="2026-06" ;;
    F) GROUP_START="2026-07"; GROUP_END="2026-09" ;;
    "") GROUP_START=""; GROUP_END="" ;;
    *) echo "ERROR: GROUP_ID must be D, E, or F (got: '${GROUP_ID}')"; exit 1 ;;
esac

if [ -n "${GROUP_ID}" ]; then
    FILTERED_LIST=()
    for ENTRY in "${MONTH_LIST[@]}"; do
        LBL=$(echo "${ENTRY}" | awk '{print $1}')
        if [[ ! "${LBL}" < "${GROUP_START}" && ! "${LBL}" > "${GROUP_END}" ]]; then
            FILTERED_LIST+=("${ENTRY}")
        fi
    done
    MONTH_LIST=("${FILTERED_LIST[@]}")
fi

if [ -n "${MAX_MONTHS}" ]; then
    if ! [[ "${MAX_MONTHS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: MAX_MONTHS must be a positive integer (got: '${MAX_MONTHS}')"; exit 1
    fi
    MONTH_LIST=("${MONTH_LIST[@]:0:${MAX_MONTHS}}")
fi
TOTAL_MONTHS=${#MONTH_LIST[@]}

case "${GROUP_ID}" in
    D) GLOBAL_LOG="${MANIFEST_DIR}/wy2026_group_d_run_log.txt" ;;
    E) GLOBAL_LOG="${MANIFEST_DIR}/wy2026_group_e_run_log.txt" ;;
    F) GLOBAL_LOG="${MANIFEST_DIR}/wy2026_group_f_run_log.txt" ;;
    *) GLOBAL_LOG="${MANIFEST_DIR}/wy2026_run_log.txt" ;;
esac

echo "============================================================"
echo "Flash-NH forcing — WY2026 extension (2026-01..2026-09)"
echo "============================================================"
echo "Group:     ${GROUP_ID:-ALL}"
echo "Months:    ${TOTAL_MONTHS}"
echo "Basins:    2,752 (${BASIN_LIST})"
echo "Workers:   ${DOWNLOAD_WORKERS}"
echo "RTMA mode: ${RTMA_MODE}"
echo "Output:    ${FORCING_ROOT}"
echo "Run log:   ${GLOBAL_LOG}"
echo "Started:   $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "============================================================"

if [ "${DRY_RUN}" = "1" ]; then
    echo "DRY_RUN=1 — printing plan; no extraction will run."
    for ENTRY in "${MONTH_LIST[@]}"; do
        printf "  %s\n" "${ENTRY}"
    done
    echo "Per-month command:"
    echo "  python scripts/extract_stage1_forcing_chunk.py --start <START> --end <END> \\"
    echo "      --basin-manifest ${BASIN_LIST} --mrms-weights ${MRMS_WEIGHTS} \\"
    echo "      --rtma-weights ${RTMA_WEIGHTS} --out-dir ${FORCING_ROOT} \\"
    echo "      --chunk-label <YYYY-MM> --download-workers ${DOWNLOAD_WORKERS} \\"
    echo "      --rtma-mode ${RTMA_MODE} --resume --no-plots"
    exit 0
fi

# ---------------------------------------------------------------------------
# Environment (same bootstrap as the full-period launcher)
# ---------------------------------------------------------------------------

if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source /opt/conda/etc/profile.d/conda.sh
fi
conda activate "${ENV_PREFIX}" 2>/dev/null || true
export PATH="${ENV_PREFIX}/bin:${PATH}"
hash -r

if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
    echo "ERROR: ${ENV_PREFIX}/bin/python not found or not executable"; exit 1
fi
_actual_python=$(command -v python || true)
if [ "${_actual_python}" != "${ENV_PREFIX}/bin/python" ]; then
    echo "ERROR: python resolves to ${_actual_python}, expected ${ENV_PREFIX}/bin/python"; exit 1
fi
_py_ver=$(python --version 2>&1)
if [[ "${_py_ver}" != *"Python 3.11"* ]]; then
    echo "ERROR: Expected Python 3.11.x, got: ${_py_ver}"; exit 1
fi
echo "Python: ${_actual_python} (${_py_ver})"
echo "Git:    $(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo UNKNOWN)"

for fpath in "${BASIN_LIST}" "${MRMS_WEIGHTS}" "${RTMA_WEIGHTS}"; do
    [ -f "${fpath}" ] || { echo "ERROR: Required file missing: ${fpath}"; exit 1; }
done

mkdir -p "${FLASHNH_TMPDIR}" "${MANIFEST_DIR}" "${FORCING_ROOT}/logs" \
         "${FORCING_ROOT}/raw/mrms" "${FORCING_ROOT}/raw/rtma" \
         "${FORCING_ROOT}/staging/mrms" "${FORCING_ROOT}/staging/rtma"

LOAD=$(uptime | awk -F'load average:' '{ print $2 }' | awk '{ print $1 }' | tr -d ',')
echo "Current system load (1-min): ${LOAD}"

cd "${REPO_ROOT}"

N_SKIPPED=0; N_FAILED=0; N_PASSED=0; FAILED_MONTHS=()
echo "WY2026 extension run started $(date -u +'%Y-%m-%dT%H:%M:%SZ') group=${GROUP_ID:-ALL}" >> "${GLOBAL_LOG}"

MONTH_IDX=0
for MONTH_ENTRY in "${MONTH_LIST[@]}"; do
    MONTH_IDX=$(( MONTH_IDX + 1 ))
    LABEL=$(echo "${MONTH_ENTRY}" | awk '{print $1}')
    START=$(echo "${MONTH_ENTRY}" | awk '{print $2}')
    END=$(echo  "${MONTH_ENTRY}" | awk '{print $3}')
    CHUNK_PARQUET="${FORCING_ROOT}/chunks/${LABEL}/combined_${LABEL}.parquet"
    CHUNK_MANIFEST="${MANIFEST_DIR}/${LABEL}_manifest.json"

    echo ""
    echo "[${MONTH_IDX}/${TOTAL_MONTHS}] ${LABEL}  (${START} → ${END})"

    if [[ " ${SKIP_MONTHS} " == *" ${LABEL} "* ]]; then
        echo "  SKIP: in SKIP_MONTHS list"
        N_SKIPPED=$(( N_SKIPPED + 1 ))
        echo "${LABEL}: MANUALLY_SKIPPED" >> "${GLOBAL_LOG}"
        continue
    fi

    if [ -f "${CHUNK_PARQUET}" ] && [ -f "${CHUNK_MANIFEST}" ]; then
        ALL_PASS=$(python -c "
import json
try:
    d = json.load(open('${CHUNK_MANIFEST}'))
    print('true' if d.get('all_pass', False) else 'false')
except Exception:
    print('false')
" 2>/dev/null)
        if [ "${ALL_PASS}" = "true" ]; then
            echo "  SKIP: chunk parquet and manifest exist with all_pass=true"
            N_SKIPPED=$(( N_SKIPPED + 1 )); N_PASSED=$(( N_PASSED + 1 ))
            echo "${LABEL}: SKIPPED_already_pass" >> "${GLOBAL_LOG}"
            continue
        fi
        echo "  Manifest exists but all_pass!=true — re-running with --resume"
    fi

    MONTH_START_T=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    MONTH_EXIT=0
    python scripts/extract_stage1_forcing_chunk.py \
        --start            "${START}" \
        --end              "${END}" \
        --basin-manifest   "${BASIN_LIST}" \
        --mrms-weights     "${MRMS_WEIGHTS}" \
        --rtma-weights     "${RTMA_WEIGHTS}" \
        --out-dir          "${FORCING_ROOT}" \
        --chunk-label      "${LABEL}" \
        --download-workers "${DOWNLOAD_WORKERS}" \
        --rtma-mode        "${RTMA_MODE}" \
        --resume \
        --no-plots || MONTH_EXIT=$?
    MONTH_END_T=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    if [ ${MONTH_EXIT} -eq 0 ]; then
        N_PASSED=$(( N_PASSED + 1 ))
        echo "  [OK] ${LABEL}: PASS (${MONTH_START_T} → ${MONTH_END_T})"
        echo "${LABEL}: PASS  start=${MONTH_START_T} end=${MONTH_END_T}" >> "${GLOBAL_LOG}"
    else
        N_FAILED=$(( N_FAILED + 1 )); FAILED_MONTHS+=("${LABEL}")
        echo "  [FAIL] ${LABEL}: exit ${MONTH_EXIT} (${MONTH_START_T} → ${MONTH_END_T})"
        echo "${LABEL}: FAIL  exit=${MONTH_EXIT}  start=${MONTH_START_T} end=${MONTH_END_T}" >> "${GLOBAL_LOG}"
    fi

    df -h "${FORCING_ROOT}" 2>/dev/null | tail -1 || true

    if [ -f "${STOP_FILE}" ]; then
        echo "  STOP FILE DETECTED — exiting cleanly after ${LABEL}."
        rm -f "${STOP_FILE}"
        echo "${LABEL}: STOPPED_BY_STOP_FILE  $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "${GLOBAL_LOG}"
        exit 0
    fi
done

echo ""
echo "WY2026 extension finished: PASS=${N_PASSED} FAIL=${N_FAILED} (of ${TOTAL_MONTHS})"
[ ${N_FAILED} -gt 0 ] && printf "  failed: %s\n" "${FAILED_MONTHS[@]}"
echo "WY2026 extension run finished $(date -u +'%Y-%m-%dT%H:%M:%SZ')  PASS=${N_PASSED} FAIL=${N_FAILED}" >> "${GLOBAL_LOG}"
[ ${N_FAILED} -gt 0 ] && exit 1
exit 0

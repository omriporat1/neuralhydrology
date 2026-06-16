#!/usr/bin/env bash
# Stage 1 Forcing — FULL-PERIOD launcher for h2o
#
# PURPOSE
# -------
# Runs the complete 63-month MRMS + RTMA forcing extraction for all 2,752 v001 basins
# covering 2020-10-14T00Z – 2025-12-31T23Z on h2o.
#
# PREREQUISITES (must all be complete before running this script)
# --------------------------------------------------------------
# 0. [Pre-2K-A] Inputs verified on h2o:
#       bash scripts/verify_stage1_forcing_inputs_h2o.sh   ← all checks PASS/WARN only
#       python scripts/export_v001_basin_list.py            ← generates v001_basin_list.csv
#       [local] PowerShell scripts/prepare_stage1_forcing_inputs_h2o.ps1  ← transfers grid defs
# 1. [2K-A] v001 weight tables built (2,752 basins; pilot 50-basin weights NOT valid):
#       ${MRMS_WEIGHTS}  and  ${RTMA_WEIGHTS}
# 2. [2K-B] Smoke test PASS — see run_stage1_forcing_smoke_h2o.sh
# 3. Scaling estimate confirmed acceptable (< 5 TB total)
# 4. PI notified (h2o etiquette: notify before long jobs)
# 5. System load checked: uptime  →  load avg ≤ 40
#
# USAGE
# -----
# Run under screen (required for multi-day jobs):
#   screen -S flashnh-forcing
#   bash scripts/run_stage1_forcing_fullperiod_h2o.sh
#
# To watch progress from another terminal (new SSH session, do NOT attach screen):
#   bash scripts/report_stage1_forcing_progress_h2o.sh
#   cat ${MANIFEST_DIR}/<YYYY-MM>_live_progress.json
#
# IMPORTANT: Detaching screen (Ctrl-A D) does NOT pause the run — the extractor
# continues running in the background. To pause safely, use the STOP_AFTER_MONTH
# stop-file mechanism described below.
#
# To skip specific months:
#   SKIP_MONTHS="2020-10 2021-03" bash scripts/run_stage1_forcing_fullperiod_h2o.sh
#
# PAUSING AND RESUMING SAFELY
# ----------------------------
# To pause after the current month completes (without killing mid-extraction):
#   touch ${FORCING_ROOT}/STOP_AFTER_MONTH
#   # The launcher will finish the currently running month, then exit cleanly.
#   # The stop file is deleted automatically on clean stop.
#
# To resume after a clean stop:
#   bash scripts/run_stage1_forcing_fullperiod_h2o.sh
#   # (no --resume flag needed: monthly skip logic handles already-passed months)
#
# To pause immediately (aborts mid-month; month must be re-run with --resume):
#   Ctrl-C inside the screen session, or:
#   kill <PID of extract_stage1_forcing_chunk.py>
#   # Resume: re-run this launcher — the interrupted month will re-run with --resume.
#
# RESUME BEHAVIOUR
# ----------------
# This script skips any month that already has:
#   ${FORCING_ROOT}/chunks/<YYYY-MM>/combined_<YYYY-MM>.parquet
# and a matching manifest JSON with "all_pass": true.
#
# To re-run a failed month manually:
#   python scripts/extract_stage1_forcing_chunk.py \
#       --start 2021-03-01T00:00:00 --end 2021-03-31T23:00:00 \
#       --basin-manifest "${BASIN_LIST}" \
#       --mrms-weights   "${MRMS_WEIGHTS}" \
#       --rtma-weights   "${RTMA_WEIGHTS}" \
#       --out-dir        "${FORCING_ROOT}" \
#       --chunk-label    2021-03 \
#       --download-workers 16 \
#       --resume
#
# WORKER COUNT POLICY
# -------------------
# Default: 16 workers (conservative for first full run on h2o).
# This keeps CPU usage well within the 50-60% etiquette cap while keeping
# the download queue saturated on h2o's 10 Gbps link.
#
# Increasing to 32 workers is permitted ONLY after:
#   1. Smoke test PASS (2K-B) confirms correct behavior at 4 workers.
#   2. First one-month evidence bundle transferred locally and inspected.
#   3. System load (uptime) confirms headroom (load avg < 40 under 16 workers).
#   To override: DOWNLOAD_WORKERS=32 bash run_stage1_forcing_fullperiod_h2o.sh
#
# Do NOT use 64 or more workers on h2o without explicit PI approval.
#
# BASIN WEIGHTS NOTE
# ------------------
# The 50-basin pilot weight Parquets (pilot_mrms_weights.parquet,
# pilot_rtma_weights.parquet) are NOT valid for v001 and must NOT be reused.
# Milestone 2K-A must build new weight tables for all 2,752 v001 basins:
#   v001_2752_mrms_weights.parquet  /  v001_2752_rtma_weights.parquet
# The grid definitions and weight computation code are reused; the output
# Parquets are completely new and specific to the v001 basin set.
#
# STORAGE NOTES
# -------------
# Total estimated storage: ~4-5 TB (within 20 TB informal limit on h2o)
#   Raw MRMS:         ~0.5 TB
#   Raw RTMA (sel):   ~3.2 TB
#   Staging Parquets: ~0.3-0.7 TB (deletable after chunk assembly)
#   Monthly chunks:   ~200-400 GB
#   Per-basin NCs:    ~130 GB (assembled in Milestone 2K-D, separate script)
#
# After each quarter's chunks are verified, raw GRIB2 can be deleted to free ~0.9 TB
# per quarter. Check with PI before bulk deletion.
#
# EVIDENCE BUNDLES
# ----------------
# Pull quarterly (not monthly) to keep local evidence manageable:
#   Q1-2021 complete example:
#     mkdir -p tmp/forcing_evidence/2021-Q1
#     for M in 2021-01 2021-02 2021-03; do
#       scp "flashnh-h2o:${MANIFEST_DIR}/${M}_manifest.json"       tmp/forcing_evidence/2021-Q1/
#       scp "flashnh-h2o:${MANIFEST_DIR}/${M}_summary.md"          tmp/forcing_evidence/2021-Q1/
#       scp "flashnh-h2o:${MANIFEST_DIR}/${M}_missing_files.csv"   tmp/forcing_evidence/2021-Q1/ 2>/dev/null || true
#     done
# Do NOT transfer raw GRIB2, staging Parquets, or monthly chunk Parquets.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
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
# Compute settings
# ---------------------------------------------------------------------------

DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-16}"
RTMA_MODE="${RTMA_MODE:-selected_messages}"

# Optional: space-separated list of chunk labels to skip (e.g. "2020-10 2021-06")
SKIP_MONTHS="${SKIP_MONTHS:-}"

# Stop-file: if this path exists at the BETWEEN-MONTH check point, the launcher
# exits cleanly after finishing the current month (does not abort mid-extraction).
# Create it with: touch ${FORCING_ROOT}/STOP_AFTER_MONTH
# The file is removed automatically on clean stop.
STOP_FILE="${FORCING_ROOT}/STOP_AFTER_MONTH"

# ---------------------------------------------------------------------------
# Month list: 63 monthly chunks
# 2020-10 starts 2020-10-14 (partial month — first day of v001 target period)
# All other months start on the 1st.
# 2025-12 ends 2025-12-31 (last hour 2025-12-31T23:00:00Z)
# ---------------------------------------------------------------------------

# Format: "YYYY-MM START_DATE END_DATE"
# Start/end dates define the exact hour range passed to extract_stage1_forcing_chunk.py
MONTH_LIST=(
    "2020-10 2020-10-14T00:00:00 2020-10-31T23:00:00"
    "2020-11 2020-11-01T00:00:00 2020-11-30T23:00:00"
    "2020-12 2020-12-01T00:00:00 2020-12-31T23:00:00"
    "2021-01 2021-01-01T00:00:00 2021-01-31T23:00:00"
    "2021-02 2021-02-01T00:00:00 2021-02-28T23:00:00"
    "2021-03 2021-03-01T00:00:00 2021-03-31T23:00:00"
    "2021-04 2021-04-01T00:00:00 2021-04-30T23:00:00"
    "2021-05 2021-05-01T00:00:00 2021-05-31T23:00:00"
    "2021-06 2021-06-01T00:00:00 2021-06-30T23:00:00"
    "2021-07 2021-07-01T00:00:00 2021-07-31T23:00:00"
    "2021-08 2021-08-01T00:00:00 2021-08-31T23:00:00"
    "2021-09 2021-09-01T00:00:00 2021-09-30T23:00:00"
    "2021-10 2021-10-01T00:00:00 2021-10-31T23:00:00"
    "2021-11 2021-11-01T00:00:00 2021-11-30T23:00:00"
    "2021-12 2021-12-01T00:00:00 2021-12-31T23:00:00"
    "2022-01 2022-01-01T00:00:00 2022-01-31T23:00:00"
    "2022-02 2022-02-01T00:00:00 2022-02-28T23:00:00"
    "2022-03 2022-03-01T00:00:00 2022-03-31T23:00:00"
    "2022-04 2022-04-01T00:00:00 2022-04-30T23:00:00"
    "2022-05 2022-05-01T00:00:00 2022-05-31T23:00:00"
    "2022-06 2022-06-01T00:00:00 2022-06-30T23:00:00"
    "2022-07 2022-07-01T00:00:00 2022-07-31T23:00:00"
    "2022-08 2022-08-01T00:00:00 2022-08-31T23:00:00"
    "2022-09 2022-09-01T00:00:00 2022-09-30T23:00:00"
    "2022-10 2022-10-01T00:00:00 2022-10-31T23:00:00"
    "2022-11 2022-11-01T00:00:00 2022-11-30T23:00:00"
    "2022-12 2022-12-01T00:00:00 2022-12-31T23:00:00"
    "2023-01 2023-01-01T00:00:00 2023-01-31T23:00:00"
    "2023-02 2023-02-01T00:00:00 2023-02-28T23:00:00"
    "2023-03 2023-03-01T00:00:00 2023-03-31T23:00:00"
    "2023-04 2023-04-01T00:00:00 2023-04-30T23:00:00"
    "2023-05 2023-05-01T00:00:00 2023-05-31T23:00:00"
    "2023-06 2023-06-01T00:00:00 2023-06-30T23:00:00"
    "2023-07 2023-07-01T00:00:00 2023-07-31T23:00:00"
    "2023-08 2023-08-01T00:00:00 2023-08-31T23:00:00"
    "2023-09 2023-09-01T00:00:00 2023-09-30T23:00:00"
    "2023-10 2023-10-01T00:00:00 2023-10-31T23:00:00"
    "2023-11 2023-11-01T00:00:00 2023-11-30T23:00:00"
    "2023-12 2023-12-01T00:00:00 2023-12-31T23:00:00"
    "2024-01 2024-01-01T00:00:00 2024-01-31T23:00:00"
    "2024-02 2024-02-01T00:00:00 2024-02-29T23:00:00"
    "2024-03 2024-03-01T00:00:00 2024-03-31T23:00:00"
    "2024-04 2024-04-01T00:00:00 2024-04-30T23:00:00"
    "2024-05 2024-05-01T00:00:00 2024-05-31T23:00:00"
    "2024-06 2024-06-01T00:00:00 2024-06-30T23:00:00"
    "2024-07 2024-07-01T00:00:00 2024-07-31T23:00:00"
    "2024-08 2024-08-01T00:00:00 2024-08-31T23:00:00"
    "2024-09 2024-09-01T00:00:00 2024-09-30T23:00:00"
    "2024-10 2024-10-01T00:00:00 2024-10-31T23:00:00"
    "2024-11 2024-11-01T00:00:00 2024-11-30T23:00:00"
    "2024-12 2024-12-01T00:00:00 2024-12-31T23:00:00"
    "2025-01 2025-01-01T00:00:00 2025-01-31T23:00:00"
    "2025-02 2025-02-01T00:00:00 2025-02-28T23:00:00"
    "2025-03 2025-03-01T00:00:00 2025-03-31T23:00:00"
    "2025-04 2025-04-01T00:00:00 2025-04-30T23:00:00"
    "2025-05 2025-05-01T00:00:00 2025-05-31T23:00:00"
    "2025-06 2025-06-01T00:00:00 2025-06-30T23:00:00"
    "2025-07 2025-07-01T00:00:00 2025-07-31T23:00:00"
    "2025-08 2025-08-01T00:00:00 2025-08-31T23:00:00"
    "2025-09 2025-09-01T00:00:00 2025-09-30T23:00:00"
    "2025-10 2025-10-01T00:00:00 2025-10-31T23:00:00"
    "2025-11 2025-11-01T00:00:00 2025-11-30T23:00:00"
    "2025-12 2025-12-01T00:00:00 2025-12-31T23:00:00"
)

TOTAL_MONTHS=${#MONTH_LIST[@]}

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

echo "============================================================"
echo "Flash-NH Stage 1 Forcing — FULL PERIOD"
echo "============================================================"
echo "Months:    ${TOTAL_MONTHS} (2020-10 through 2025-12)"
echo "Basins:    2,752 (from ${BASIN_LIST})"
echo "Workers:   ${DOWNLOAD_WORKERS}"
echo "RTMA mode: ${RTMA_MODE}"
echo "Output:    ${FORCING_ROOT}"
echo "Started:   $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "============================================================"

# Activate conda
if ! command -v conda &>/dev/null; then
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source /opt/conda/etc/profile.d/conda.sh
    elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    else
        echo "ERROR: conda not found"
        exit 1
    fi
fi

conda activate "${ENV_PREFIX}" || {
    echo "ERROR: Could not activate ${ENV_PREFIX}"
    exit 1
}
echo "Python: $(which python) ($(python --version 2>&1))"

# Preflight: required files
for fpath in "${BASIN_LIST}" "${MRMS_WEIGHTS}" "${RTMA_WEIGHTS}"; do
    if [ ! -f "${fpath}" ]; then
        echo "ERROR: Required file missing: ${fpath}"
        echo "Complete Milestone 2K-A (weight build) and 2K-B (smoke test) before full run."
        exit 1
    fi
done

# Output dirs
mkdir -p "${MANIFEST_DIR}"
mkdir -p "${FORCING_ROOT}/raw/mrms"
mkdir -p "${FORCING_ROOT}/raw/rtma"
mkdir -p "${FORCING_ROOT}/staging/mrms"
mkdir -p "${FORCING_ROOT}/staging/rtma"

# System load check
LOAD=$(uptime | awk -F'load average:' '{ print $2 }' | awk '{ print $1 }' | tr -d ',')
echo "Current system load (1-min): ${LOAD}"

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Progress state
# ---------------------------------------------------------------------------

N_SKIPPED=0
N_FAILED=0
N_PASSED=0
FAILED_MONTHS=()
GLOBAL_LOG="${MANIFEST_DIR}/fullperiod_run_log.txt"

echo "Full-period run started $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "${GLOBAL_LOG}"

# ---------------------------------------------------------------------------
# Month loop
# ---------------------------------------------------------------------------

MONTH_IDX=0
for MONTH_ENTRY in "${MONTH_LIST[@]}"; do
    MONTH_IDX=$(( MONTH_IDX + 1 ))
    LABEL=$(echo "${MONTH_ENTRY}" | awk '{print $1}')
    START=$(echo "${MONTH_ENTRY}" | awk '{print $2}')
    END=$(echo  "${MONTH_ENTRY}" | awk '{print $3}')

    CHUNK_PARQUET="${FORCING_ROOT}/chunks/${LABEL}/combined_${LABEL}.parquet"
    CHUNK_MANIFEST="${MANIFEST_DIR}/${LABEL}_manifest.json"

    echo ""
    echo "------------------------------------------------------------"
    echo "[${MONTH_IDX}/${TOTAL_MONTHS}] ${LABEL}  (${START} → ${END})"
    echo "------------------------------------------------------------"

    # Skip if in SKIP_MONTHS list
    if [[ " ${SKIP_MONTHS} " == *" ${LABEL} "* ]]; then
        echo "  SKIP: in SKIP_MONTHS list"
        N_SKIPPED=$(( N_SKIPPED + 1 ))
        echo "${LABEL}: MANUALLY_SKIPPED" >> "${GLOBAL_LOG}"
        continue
    fi

    # Resume: skip months with complete valid output
    if [ -f "${CHUNK_PARQUET}" ] && [ -f "${CHUNK_MANIFEST}" ]; then
        # Check manifest all_pass field using python (avoids jq dependency)
        ALL_PASS=$(python -c "
import json, sys
try:
    d = json.load(open('${CHUNK_MANIFEST}'))
    print('true' if d.get('all_pass', False) else 'false')
except Exception as e:
    print('false')
" 2>/dev/null)
        if [ "${ALL_PASS}" = "true" ]; then
            echo "  SKIP: chunk Parquet and manifest exist with all_pass=true"
            N_SKIPPED=$(( N_SKIPPED + 1 ))
            N_PASSED=$(( N_PASSED + 1 ))
            echo "${LABEL}: SKIPPED_already_pass" >> "${GLOBAL_LOG}"
            continue
        else
            echo "  Manifest exists but all_pass!=true — re-running with --resume"
        fi
    fi

    # Run extraction
    echo "  Launching: ${LABEL}  workers=${DOWNLOAD_WORKERS}  mode=${RTMA_MODE}"
    MONTH_START_T=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

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
        --no-plots
    MONTH_EXIT=$?

    MONTH_END_T=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    if [ ${MONTH_EXIT} -eq 0 ]; then
        N_PASSED=$(( N_PASSED + 1 ))
        echo "  [OK] ${LABEL}: PASS  (${MONTH_START_T} → ${MONTH_END_T})"
        echo "${LABEL}: PASS  start=${MONTH_START_T} end=${MONTH_END_T}" >> "${GLOBAL_LOG}"
    else
        N_FAILED=$(( N_FAILED + 1 ))
        FAILED_MONTHS+=("${LABEL}")
        echo "  [FAIL] ${LABEL}: exit ${MONTH_EXIT}  (${MONTH_START_T} → ${MONTH_END_T})"
        echo "${LABEL}: FAIL  exit=${MONTH_EXIT}  start=${MONTH_START_T} end=${MONTH_END_T}" >> "${GLOBAL_LOG}"
        # Do NOT abort — continue so other months aren't blocked by a single S3 outage
    fi

    # Periodic disk usage check (every 12 months)
    if [ $(( MONTH_IDX % 12 )) -eq 0 ]; then
        echo ""
        echo "  Disk check (${FORCING_ROOT}):"
        du -sh "${FORCING_ROOT}" 2>/dev/null || true
        df -h "${FORCING_ROOT}" 2>/dev/null | tail -1 || true
        echo ""
    fi

    # Stop-file check: checked BETWEEN months (never mid-extraction).
    # Create the file from another terminal to pause safely after the current month.
    if [ -f "${STOP_FILE}" ]; then
        echo ""
        echo "  STOP FILE DETECTED: ${STOP_FILE}"
        echo "  Completed month ${LABEL} — exiting cleanly before next month."
        rm -f "${STOP_FILE}"
        echo "${LABEL}: STOPPED_BY_STOP_FILE  $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "${GLOBAL_LOG}"
        echo ""
        echo "To resume from month after ${LABEL}:"
        echo "  bash scripts/run_stage1_forcing_fullperiod_h2o.sh"
        echo "  (already-passed months are auto-skipped)"
        exit 0
    fi

done

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

TOTAL_DONE=$(( N_PASSED + N_FAILED ))

echo ""
echo "============================================================"
echo "FULL-PERIOD RUN COMPLETE"
echo "============================================================"
echo "Total months:   ${TOTAL_MONTHS}"
echo "PASS:           ${N_PASSED}"
echo "FAIL:           ${N_FAILED}"
echo "Pre-skipped:    $(( N_SKIPPED - N_PASSED ))"
echo "Finished:       $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
if [ ${N_FAILED} -gt 0 ]; then
    echo ""
    echo "Failed months:"
    for M in "${FAILED_MONTHS[@]}"; do
        echo "  - ${M}"
    done
    echo ""
    echo "Re-run failed months with --resume to recover from partial failures."
fi
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Pull evidence bundles (manifests + summaries) for each quarter"
echo "  2. Check for missing_files.csv entries"
echo "  3. Run Milestone 2K-D: python scripts/build_stage1_forcing_basin_ncs.py"
echo ""
echo "Full run log: ${GLOBAL_LOG}"
echo ""

echo "Full-period run finished $(date -u +'%Y-%m-%dT%H:%M:%SZ')  PASS=${N_PASSED} FAIL=${N_FAILED}" >> "${GLOBAL_LOG}"

if [ ${N_FAILED} -gt 0 ]; then
    exit 1
fi
exit 0

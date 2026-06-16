#!/usr/bin/env bash
# Stage 1 Forcing — PROGRESS REPORTER for h2o (read-only)
#
# PURPOSE
# -------
# Provides a quick human-readable summary of the forcing extraction progress
# during or after the full-period run. Run at any time from h2o — read-only.
#
# Reports:
#   - Full-period run log (fullperiod_run_log.txt)
#   - Monthly manifest counts: all_pass=true / false / missing
#   - Most recent live progress JSON (current or last active month)
#   - Missing files summary (if any CSV entries exist)
#   - Disk usage of the forcing root and /data42
#   - Whether extractor Python processes are currently running
#
# USAGE
# -----
#   bash scripts/report_stage1_forcing_progress_h2o.sh
#
#   # Override forcing root:
#   FORCING_ROOT=/data42/omrip/Flash-NH/tmp/my_run \
#       bash scripts/report_stage1_forcing_progress_h2o.sh
#
# To watch progress continuously (refresh every 60s):
#   watch -n 60 "bash scripts/report_stage1_forcing_progress_h2o.sh"

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FLASHNH_ROOT="${FLASHNH_ROOT:-/data42/omrip/Flash-NH}"
FORCING_ROOT="${FORCING_ROOT:-${FLASHNH_ROOT}/tmp/stage1_forcing_fullperiod}"
MANIFEST_DIR="${FORCING_ROOT}/manifests"
GLOBAL_LOG="${MANIFEST_DIR}/fullperiod_run_log.txt"
TOTAL_MONTHS=63

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

echo "============================================================"
echo "Flash-NH Stage 1 Forcing — Progress Report"
echo "============================================================"
echo "Forcing root: ${FORCING_ROOT}"
echo "Report time:  $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Section 1: Full-period run log
# ---------------------------------------------------------------------------

echo "--- [1] Full-period run log ---"
if [ -f "${GLOBAL_LOG}" ]; then
    # Show last 10 lines of the log for recent activity
    echo "  (showing last 10 lines of fullperiod_run_log.txt)"
    tail -10 "${GLOBAL_LOG}" | sed 's/^/  /'
    TOTAL_LOG_LINES=$(wc -l < "${GLOBAL_LOG}")
    echo "  Total log entries: ${TOTAL_LOG_LINES}"
else
    echo "  NOT FOUND: ${GLOBAL_LOG}"
    echo "  The full-period launcher has not been started yet."
fi
echo ""

# ---------------------------------------------------------------------------
# Section 2: Monthly manifest pass/fail/missing counts
# ---------------------------------------------------------------------------

echo "--- [2] Monthly manifest status (all 63 months) ---"

N_PASS=0; N_FAIL=0; N_MISSING=0; N_WARN=0

if [ -d "${MANIFEST_DIR}" ]; then
    for year in 2020 2021 2022 2023 2024 2025; do
        for month in 01 02 03 04 05 06 07 08 09 10 11 12; do
            LABEL="${year}-${month}"

            # v001 target period starts 2020-10 — skip earlier months
            if [[ "${LABEL}" < "2020-10" ]]; then continue; fi
            # Period ends 2025-12
            if [[ "${LABEL}" > "2025-12" ]]; then break 2; fi

            MF="${MANIFEST_DIR}/${LABEL}_manifest.json"
            CHUNK_PQ="${FORCING_ROOT}/chunks/${LABEL}/combined_${LABEL}.parquet"

            if [ ! -f "${MF}" ]; then
                N_MISSING=$(( N_MISSING + 1 ))
            else
                ALL_PASS=$(python3 -c "
import json
try:
    d = json.load(open('${MF}'))
    print('true' if d.get('all_pass', False) else 'false')
except Exception:
    print('error')
" 2>/dev/null)

                if [ "${ALL_PASS}" = "true" ]; then
                    N_PASS=$(( N_PASS + 1 ))
                elif [ "${ALL_PASS}" = "false" ]; then
                    N_FAIL=$(( N_FAIL + 1 ))
                else
                    N_WARN=$(( N_WARN + 1 ))
                fi
            fi
        done
    done

    DONE=$(( N_PASS + N_FAIL + N_WARN ))
    PCT=$(python3 -c "print(f'{${N_PASS}/${TOTAL_MONTHS}*100:.1f}')" 2>/dev/null || echo "?")

    echo "  Total expected:    ${TOTAL_MONTHS}"
    echo "  PASS (all_pass):   ${N_PASS}  (${PCT}% complete)"
    echo "  FAIL (all_pass=F): ${N_FAIL}"
    echo "  WARN (parse err):  ${N_WARN}"
    echo "  MISSING (no JSON): ${N_MISSING}"
    echo "  Manifest dir:      ${MANIFEST_DIR}"

    # List failed months
    if [ ${N_FAIL} -gt 0 ]; then
        echo ""
        echo "  Failed months:"
        for year in 2020 2021 2022 2023 2024 2025; do
            for month in 01 02 03 04 05 06 07 08 09 10 11 12; do
                LABEL="${year}-${month}"
                if [[ "${LABEL}" < "2020-10" ]] || [[ "${LABEL}" > "2025-12" ]]; then continue; fi
                MF="${MANIFEST_DIR}/${LABEL}_manifest.json"
                if [ -f "${MF}" ]; then
                    AP=$(python3 -c "
import json
try:
    d = json.load(open('${MF}'))
    print('true' if d.get('all_pass', False) else 'false')
except Exception:
    print('error')
" 2>/dev/null)
                    if [ "${AP}" = "false" ]; then
                        echo "    - ${LABEL}"
                    fi
                fi
            done
        done
    fi
else
    echo "  Manifest directory not found: ${MANIFEST_DIR}"
    echo "  No extraction run has produced output yet."
fi
echo ""

# ---------------------------------------------------------------------------
# Section 3: Most recent live progress JSON
# ---------------------------------------------------------------------------

echo "--- [3] Live progress (most recent active chunk) ---"

if [ -d "${MANIFEST_DIR}" ]; then
    LATEST_PROGRESS=$(ls -t "${MANIFEST_DIR}"/*_live_progress.json 2>/dev/null | head -1 || true)
    if [ -n "${LATEST_PROGRESS}" ]; then
        LABEL_LIVE=$(basename "${LATEST_PROGRESS}" | sed 's/_live_progress.json//')
        echo "  Chunk: ${LABEL_LIVE}"
        echo "  File:  ${LATEST_PROGRESS}"
        python3 -c "
import json, sys
try:
    d = json.load(open('${LATEST_PROGRESS}'))
    pct  = d.get('percent_complete', 0)
    done = d.get('completed_hour_products', 0)
    mrms = d.get('completed_mrms_hours', 0)
    rtma = d.get('completed_rtma_hours', 0)
    fail = d.get('failed_hour_products', 0)
    ela  = d.get('elapsed_seconds', 0)
    eta  = d.get('estimated_remaining_seconds')
    dl   = d.get('bytes_downloaded_total', 0)
    ts   = d.get('current_time_utc', '?')
    msg  = d.get('latest_status_message', '')
    err  = d.get('latest_error_message', '')
    eta_s = f'{eta:.0f}s' if eta else 'unknown'
    print(f'  Progress:  {pct:.1f}%  ({done} hr-products done)')
    print(f'  MRMS OK:   {mrms} hours')
    print(f'  RTMA OK:   {rtma} hours')
    print(f'  Failed:    {fail}')
    print(f'  Elapsed:   {ela:.0f}s  ETA: {eta_s}')
    print(f'  Downloaded:{dl/1e9:.2f} GB')
    print(f'  Timestamp: {ts}')
    print(f'  Latest:    {msg}')
    if err:
        print(f'  Error:     {err}')
except Exception as e:
    print(f'  Could not parse: {e}')
" 2>/dev/null
    else
        echo "  No live_progress.json files found in ${MANIFEST_DIR}"
    fi
else
    echo "  Manifest directory not found."
fi
echo ""

# ---------------------------------------------------------------------------
# Section 4: Missing files summary
# ---------------------------------------------------------------------------

echo "--- [4] Missing files summary ---"

if [ -d "${MANIFEST_DIR}" ]; then
    MISSING_CSVS=$(ls "${MANIFEST_DIR}"/*_missing_files.csv 2>/dev/null || true)
    if [ -z "${MISSING_CSVS}" ]; then
        echo "  No missing_files.csv files found (good — no S3 gaps logged)"
    else
        TOTAL_MISSING=0
        for CSV in ${MISSING_CSVS}; do
            N=$(tail -n +2 "${CSV}" 2>/dev/null | wc -l | tr -d ' ')
            TOTAL_MISSING=$(( TOTAL_MISSING + N ))
            LABEL_M=$(basename "${CSV}" | sed 's/_missing_files.csv//')
            echo "  ${LABEL_M}: ${N} missing hour-products"
        done
        echo "  Total across all chunks: ${TOTAL_MISSING}"
        echo ""
        echo "  Missing files are expected for S3 gaps in MRMS/RTMA archives."
        echo "  Review the CSVs if counts are high (>5 per month may indicate S3 outage)."
    fi
else
    echo "  Manifest directory not found."
fi
echo ""

# ---------------------------------------------------------------------------
# Section 5: Disk usage
# ---------------------------------------------------------------------------

echo "--- [5] Disk usage ---"
if [ -d "${FORCING_ROOT}" ]; then
    echo "  Forcing root (may take a moment):"
    du -sh "${FORCING_ROOT}" 2>/dev/null | sed 's/^/  /' || echo "  (du failed)"

    # Sub-directory breakdown
    for subdir in raw/mrms raw/rtma staging/mrms staging/rtma chunks manifests; do
        SUBPATH="${FORCING_ROOT}/${subdir}"
        if [ -d "${SUBPATH}" ]; then
            SZ=$(du -sh "${SUBPATH}" 2>/dev/null | cut -f1)
            echo "  ${subdir}: ${SZ}"
        fi
    done

    echo ""
    echo "  /data42 partition:"
    df -h /data42 2>/dev/null | tail -1 | sed 's/^/  /' || echo "  (df failed)"
else
    echo "  Forcing root not yet created: ${FORCING_ROOT}"
fi
echo ""

# ---------------------------------------------------------------------------
# Section 6: Running processes
# ---------------------------------------------------------------------------

echo "--- [6] Currently running extractor processes ---"
RUNNING=$(pgrep -f "extract_stage1_forcing_chunk.py" 2>/dev/null || true)
LAUNCHER=$(pgrep -f "run_stage1_forcing_fullperiod" 2>/dev/null || true)

if [ -n "${RUNNING}" ]; then
    echo "  extract_stage1_forcing_chunk.py is RUNNING (PIDs: ${RUNNING})"
    # Show memory/CPU for each PID
    for PID in ${RUNNING}; do
        if ps -p "${PID}" -o pid,pcpu,pmem,etime,comm &>/dev/null; then
            ps -p "${PID}" -o pid,pcpu,pmem,etime,comm | tail -1 | sed 's/^/    /'
        fi
    done
else
    echo "  extract_stage1_forcing_chunk.py: NOT running"
fi

if [ -n "${LAUNCHER}" ]; then
    echo "  run_stage1_forcing_fullperiod: RUNNING (PIDs: ${LAUNCHER})"
else
    echo "  run_stage1_forcing_fullperiod: NOT running"
fi
echo ""
echo "  System load:"
uptime | sed 's/^/  /'
echo ""

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

echo "============================================================"
echo "END OF PROGRESS REPORT"
echo "============================================================"
echo ""
echo "Pull evidence bundle (quarterly):"
echo "  For each completed quarter, on local Windows:"
echo "  mkdir tmp\\forcing_evidence\\YYYY-QN"
echo "  for M in YYYY-MM ...; do"
echo "    scp \"flashnh-h2o:${MANIFEST_DIR}/\${M}_manifest.json\" tmp\\forcing_evidence\\YYYY-QN\\"
echo "    scp \"flashnh-h2o:${MANIFEST_DIR}/\${M}_summary.md\"     tmp\\forcing_evidence\\YYYY-QN\\"
echo "  done"

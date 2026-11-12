#!/usr/bin/env bash
# Flash-NH — read-only status report for the WY2026 downloads on h2o.
#
# Prints counts, percentages, timestamps and disk use ONLY. It never prints a
# discharge or forcing value (WY2026 is the protected test year).
#
# Usage (on h2o, any directory; no conda env needed):
#   bash scripts/report_wy2026_status_h2o.sh
# Optional overrides: USGS_ROOT, FORCING_ROOT.

set -u

USGS_ROOT="${USGS_ROOT:-/data42/omrip/Flash-NH/tmp/stage1_usgs_iv_wy2026_v001}"
FORCING_ROOT="${FORCING_ROOT:-/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod}"
MONTHS="2026-01 2026-02 2026-03 2026-04 2026-05 2026-06 2026-07 2026-08 2026-09"

echo "=== WY2026 status  $(date -u +%Y-%m-%dT%H:%M:%SZ)  host=$(hostname)"
echo "load: $(cut -d' ' -f1-3 /proc/loadavg)   cores: $(nproc)"
df -h /data42 | tail -1 | awk '{print "/data42: used " $3 " of " $2 ", free " $4 " (" $5 ")"}'
echo "screens: $(screen -ls 2>/dev/null | grep -oE '[0-9]+\.wy2026-[a-z-]+' | tr '\n' ' ')"

echo
echo "--- USGS fetch: ${USGS_ROOT}"
if [ ! -d "${USGS_ROOT}" ]; then
    echo "not started (root missing)"
else
    TOTAL_NC=0
    for SD in "${USGS_ROOT}"/shard_*; do
        [ -d "${SD}" ] || continue
        SID=$(basename "${SD}")
        NSTA="?"
        [ -f "${USGS_ROOT}/manifests/${SID}.csv" ] && NSTA=$(( $(wc -l < "${USGS_ROOT}/manifests/${SID}.csv") - 1 ))
        NNC=$(find "${SD}/canonical" -maxdepth 1 -name '*_hourly.nc' 2>/dev/null | wc -l)
        TOTAL_NC=$((TOTAL_NC + NNC))
        LOG="${SD}/logs/recovery.log"
        NFAIL=0; NEMPTY=0; LAST="-"
        if [ -f "${LOG}" ]; then
            NFAIL=$(grep -c 'fallback failed' "${LOG}")
            NEMPTY=$(grep -c 'no data retrieved' "${LOG}")
            LAST=$(date -u -r "${LOG}" +%H:%M:%SZ)
        fi
        echo "${SID}: ${NNC}/${NSTA} NCs written | monthly-fallback failures ${NFAIL} | WY requests with no data ${NEMPTY} | log updated ${LAST}"
    done
    echo "total NCs: ${TOTAL_NC}"
    if [ -f "${USGS_ROOT}/launcher_summary.json" ] && grep -qE '"dry_run": *true' "${USGS_ROOT}/launcher_summary.json"; then
        echo "launcher_summary.json is from the DRY RUN (real run still going, or not launched)"
    elif [ -f "${USGS_ROOT}/launcher_summary.json" ]; then
        echo "launcher_summary.json present (launcher finished); shard return codes:"
        grep -oE '"(shard_id|return_code)": *("[^"]*"|-?[0-9]+)' "${USGS_ROOT}/launcher_summary.json" | paste - - | sed 's/^/  /'
    else
        echo "launcher_summary.json absent (still running, or not launched)"
    fi
fi

echo
echo "--- Forcing extraction: ${FORCING_ROOT}/manifests"
for M in ${MONTHS}; do
    SUM="${FORCING_ROOT}/manifests/${M}_manifest.json"
    LP="${FORCING_ROOT}/manifests/${M}_live_progress.json"
    if [ -f "${SUM}" ]; then
        AP=$(grep -oE '"all_pass": *(true|false)' "${SUM}" | head -1 | sed 's/.*: *//')
        echo "${M}: finished  all_pass=${AP:-?}"
    elif [ -f "${LP}" ]; then
        PCT=$(grep -oE '"percent_complete": *[0-9.]+' "${LP}" | sed 's/.*: *//')
        REM=$(grep -oE '"estimated_remaining_seconds": *[0-9.]+' "${LP}" | sed 's/.*: *//')
        FAILED=$(grep -oE '"failed_hour_products": *[0-9]+' "${LP}" | sed 's/.*: *//')
        UPD=$(date -u -r "${LP}" +%H:%M:%SZ)
        if [ -n "${REM}" ]; then
            REM_TXT="~$(awk -v s="${REM}" 'BEGIN{printf "%.1f", s/3600}') h left"
        else
            REM_TXT="time left unknown"
        fi
        echo "${M}: running  ${PCT:-0}%  ${REM_TXT}  failed hour-products ${FAILED:-0}  (updated ${UPD})"
    else
        echo "${M}: not started"
    fi
done
for G in d e f; do
    RL="${FORCING_ROOT}/manifests/wy2026_group_${G}_run_log.txt"
    [ -f "${RL}" ] && echo "group ${G^^} last log line: $(tail -1 "${RL}")"
done
[ -f "${FORCING_ROOT}/STOP_AFTER_MONTH" ] && echo "NOTE: STOP_AFTER_MONTH file present (groups stop after current month)"
exit 0

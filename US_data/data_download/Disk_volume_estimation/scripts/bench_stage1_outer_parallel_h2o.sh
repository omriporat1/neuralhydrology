#!/usr/bin/env bash
# bench_stage1_outer_parallel_h2o.sh
#
# PURPOSE
# -------
# General outer-parallelism RTMA benchmark for Flash-NH Stage 1.
# Launches N concurrent chunk extraction jobs, captures one
# /usr/bin/time -v log per chunk, and records parent wall clock.
#
# To benchmark a different concurrency level or worker count, edit the
# CONFIGURATION block below and commit before running on h2o.
#
# All outputs under /data42 — NEVER write to system /tmp.
# No code changes. No full-period run. No process workers.
#
# USAGE
# -----
#   screen -S flashnh_outer_parallel
#   bash scripts/bench_stage1_outer_parallel_h2o.sh
#   # Ctrl-A D to detach;  screen -r flashnh_outer_parallel to reattach
#
# After all chunks complete, run the summary script (command printed at end).
#
# CURRENT CONFIGURATION: x3 chunks, dw6 (18 total S3 connections)
# Expected runtime: ~12-18 minutes.

# Do NOT use set -e: background process exit codes are captured explicitly
# via 'wait $PID; EXIT=$?'. set -e aborts before EXIT=$? if child exits nonzero.
set -uo pipefail

# ===========================================================================
# CONFIGURATION — edit these for each benchmark run, then commit
# ===========================================================================
BENCH_LABEL="outer_parallel_rtma_48h_dw6_x3"
DOWNLOAD_WORKERS=6
PRODUCTS="rtma"

# Parallel arrays: labels, start times, end times (one entry per chunk)
CHUNK_LABELS=( "outer-x3-a"          "outer-x3-b"          "outer-x3-c"          )
CHUNK_STARTS=( "2020-10-14T21:00:00"  "2020-10-16T21:00:00"  "2020-10-18T21:00:00"  )
CHUNK_ENDS=(   "2020-10-16T20:00:00"  "2020-10-18T20:00:00"  "2020-10-20T20:00:00"  )
# ===========================================================================

# ---------------------------------------------------------------------------
# Fixed paths (override FLASHNH_ROOT env var if your layout differs)
# ---------------------------------------------------------------------------

FLASHNH_ROOT="${FLASHNH_ROOT:-/data42/omrip/Flash-NH}"
FORCING_ROOT="${FLASHNH_ROOT}/tmp/stage1_forcing_fullperiod"
BENCH_BASE="${FLASHNH_ROOT}/tmp/stage1_bench/${BENCH_LABEL}"
LOG_DIR="${BENCH_BASE}/logs"
ENV_PREFIX="${FLASHNH_ROOT}/envs/flashnh-stage1"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

BASIN_LIST="${FORCING_ROOT}/v001_basin_list.csv"
MRMS_WT="${FORCING_ROOT}/02_basin_geometries/weights/mrms/v001_2752_mrms_weights.parquet"
RTMA_WT="${FORCING_ROOT}/02_basin_geometries/weights/rtma/v001_2752_rtma_weights.parquet"

N_CHUNKS=${#CHUNK_LABELS[@]}

# ---------------------------------------------------------------------------
# Safety guard: all outputs must stay under /data42
# ---------------------------------------------------------------------------

case "${BENCH_BASE}" in
    /data42/*) ;;
    *) echo "ERROR: BENCH_BASE must be under /data42, got: ${BENCH_BASE}"; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Create output directories
# ---------------------------------------------------------------------------

mkdir -p "${LOG_DIR}"
for LABEL in "${CHUNK_LABELS[@]}"; do
    mkdir -p "${BENCH_BASE}/${LABEL}"
done

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

echo "============================================================"
echo "Flash-NH Stage 1 — Outer-Parallelism Benchmark"
echo "============================================================"
echo "Label   : ${BENCH_LABEL}"
printf "Chunks  : %d\n" "${N_CHUNKS}"
for (( i=0; i<N_CHUNKS; i++ )); do
    printf "  [%d] %-20s %s -> %s\n" \
        "$((i+1))" "${CHUNK_LABELS[$i]}" "${CHUNK_STARTS[$i]}" "${CHUNK_ENDS[$i]}"
done
printf "Workers : %d per chunk  (%d total S3 connections)\n" \
    "${DOWNLOAD_WORKERS}" "$((DOWNLOAD_WORKERS * N_CHUNKS))"
echo "Products: ${PRODUCTS}"
echo "Output  : ${BENCH_BASE}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Python env setup (same pattern as all other h2o launchers)
# ---------------------------------------------------------------------------

if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source /opt/conda/etc/profile.d/conda.sh
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "${ENV_PREFIX}" 2>/dev/null || true
export PATH="${ENV_PREFIX}/bin:${PATH}"
hash -r

[ -x "${ENV_PREFIX}/bin/python" ] \
    || { echo "ERROR: ${ENV_PREFIX}/bin/python not found or not executable."; exit 1; }
_py_ver=$("${ENV_PREFIX}/bin/python" --version 2>&1)
[[ "${_py_ver}" == *"Python 3.11"* ]] \
    || { echo "ERROR: expected Python 3.11, got: ${_py_ver}"; exit 1; }
echo "Python  : ${ENV_PREFIX}/bin/python  (${_py_ver})"

# ---------------------------------------------------------------------------
# Preflight: required input files
# ---------------------------------------------------------------------------

for fpath in "${BASIN_LIST}" "${MRMS_WT}" "${RTMA_WT}"; do
    [ -f "${fpath}" ] || { echo "ERROR: required file missing: ${fpath}"; exit 1; }
done

LOAD=$(uptime | awk -F'load average:' '{ print $2 }' | awk '{ print $1 }' | tr -d ',')
echo "System load (1-min): ${LOAD}"
echo "NOTE: h2o etiquette — keep total CPU <= 50-60%. Check htop if load is high."

# ---------------------------------------------------------------------------
# Git state + system snapshot PRE
# ---------------------------------------------------------------------------

cd "${REPO_ROOT}"
git log --oneline -3 >  "${LOG_DIR}/git_state.txt"
git status --short   >> "${LOG_DIR}/git_state.txt"

{
    printf "=== PRE: %s ===\n" "$(date -u)"
    uptime
    free -h
    df -h /data42
} > "${LOG_DIR}/system.txt"

# ---------------------------------------------------------------------------
# Launch all chunks concurrently
# ---------------------------------------------------------------------------

T_START=$(date +%s)
printf "parent_start_utc=%s\nparent_start_epoch=%d\n" "$(date -u)" "${T_START}" \
    > "${LOG_DIR}/parent_timing.txt"

PIDS=()
echo ""
echo "Launching ${N_CHUNKS} chunks ..."
for (( i=0; i<N_CHUNKS; i++ )); do
    LABEL="${CHUNK_LABELS[$i]}"
    /usr/bin/time -v \
        "${ENV_PREFIX}/bin/python" scripts/extract_stage1_forcing_chunk.py \
            --start            "${CHUNK_STARTS[$i]}" \
            --end              "${CHUNK_ENDS[$i]}" \
            --basin-manifest   "${BASIN_LIST}" \
            --mrms-weights     "${MRMS_WT}" \
            --rtma-weights     "${RTMA_WT}" \
            --out-dir          "${BENCH_BASE}/${LABEL}" \
            --chunk-label      "${LABEL}" \
            --products         "${PRODUCTS}" \
            --download-workers "${DOWNLOAD_WORKERS}" \
            --no-plots \
        &> "${LOG_DIR}/bench_${LABEL}.log" &
    PID=$!
    PIDS+=($PID)
    printf "pid_%s=%d\n" "${LABEL}" "${PID}" >> "${LOG_DIR}/parent_timing.txt"
    printf "  Launched %-20s PID=%d\n" "${LABEL}" "${PID}"
done
echo "Waiting for all ${N_CHUNKS} chunks to finish ..."

# ---------------------------------------------------------------------------
# Wait for all chunks; capture exit codes individually
# ---------------------------------------------------------------------------

EXIT_CODES=()
for (( i=0; i<N_CHUNKS; i++ )); do
    wait "${PIDS[$i]}"; EXIT_CODE=$?
    EXIT_CODES+=($EXIT_CODE)
    printf "exit_%s=%d\n" "${CHUNK_LABELS[$i]}" "${EXIT_CODE}" \
        >> "${LOG_DIR}/parent_timing.txt"
done

T_END=$(date +%s)
PARENT_WALL=$((T_END - T_START))
printf "parent_end_utc=%s\nparent_end_epoch=%d\nparent_wall_seconds=%d\n" \
    "$(date -u)" "${T_END}" "${PARENT_WALL}" \
    >> "${LOG_DIR}/parent_timing.txt"

# ---------------------------------------------------------------------------
# System snapshot POST
# ---------------------------------------------------------------------------

{
    printf "\n=== POST: %s ===\n" "$(date -u)"
    uptime
    free -h
    df -h /data42
} >> "${LOG_DIR}/system.txt"

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "BENCHMARK COMPLETE"
ANY_FAIL=0
for (( i=0; i<N_CHUNKS; i++ )); do
    printf "  %-22s exit : %d\n" "${CHUNK_LABELS[$i]}" "${EXIT_CODES[$i]}"
    [ "${EXIT_CODES[$i]}" -ne 0 ] && ANY_FAIL=1
done
printf "  Parent wall clock          : %d s\n" "${PARENT_WALL}"
echo "============================================================"

if [ "${ANY_FAIL}" -ne 0 ]; then
    echo ""
    echo "WARNING: one or more chunks exited non-zero. Check logs in ${LOG_DIR}/"
fi

# ---------------------------------------------------------------------------
# Next-step commands (printed for easy copy-paste)
# ---------------------------------------------------------------------------

echo ""
echo "Run summary script:"
echo ""
echo "  ${ENV_PREFIX}/bin/python \\"
echo "      scripts/summarize_stage1_outer_parallel.py \\"
echo "      ${BENCH_BASE} \\"
echo "      2>&1 | tee ${LOG_DIR}/sanity_output.txt"
echo ""
echo "Transfer evidence locally (PowerShell on Windows):"
echo ""
echo "  \$local = \"tmp\\stage1_2kd_evidence\\${BENCH_LABEL}\""
echo "  New-Item -ItemType Directory -Force -Path \$local | Out-Null"
echo "  scp omrip@h2o:${LOG_DIR}/parent_timing.txt  \"\$local\\\""
echo "  scp omrip@h2o:${LOG_DIR}/system.txt          \"\$local\\\""
echo "  scp omrip@h2o:${LOG_DIR}/git_state.txt       \"\$local\\\""
echo "  scp omrip@h2o:${LOG_DIR}/sanity_output.txt   \"\$local\\\""
echo "  scp omrip@h2o:${LOG_DIR}/summary.csv         \"\$local\\\""
for LABEL in "${CHUNK_LABELS[@]}"; do
    echo "  scp omrip@h2o:${LOG_DIR}/bench_${LABEL}.log   \"\$local\\\""
    echo "  scp \"omrip@h2o:${BENCH_BASE}/${LABEL}/manifests/${LABEL}_manifest.json\"                 \"\$local\\\""
    echo "  scp \"omrip@h2o:${BENCH_BASE}/${LABEL}/manifests/${LABEL}_hourly_runtime_and_volume.csv\" \"\$local\\\""
    echo "  scp \"omrip@h2o:${BENCH_BASE}/${LABEL}/manifests/${LABEL}_validation_checks.csv\"         \"\$local\\\""
done

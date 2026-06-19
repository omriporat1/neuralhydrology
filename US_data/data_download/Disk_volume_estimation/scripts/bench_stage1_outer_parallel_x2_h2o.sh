#!/usr/bin/env bash
# bench_stage1_outer_parallel_x2_h2o.sh
#
# PURPOSE
# -------
# Outer-parallelism benchmark for Flash-NH Stage 1 forcing extraction.
# Runs two independent RTMA-only 48-hour chunks concurrently, each with
# --download-workers 8, to measure effective full-period throughput under
# controlled outer parallelism.
#
# Benchmark question:
#   Does 2x concurrent chunk throughput project the 45,720-hour full period
#   to <= 4 days wall clock on h2o?
#
# Chunks:
#   outer-x2-a : 2020-10-14T21:00:00 -> 2020-10-16T20:00:00  (48 h)
#   outer-x2-b : 2020-10-16T21:00:00 -> 2020-10-18T20:00:00  (48 h)
#
# Expected effective projection:
#   45720 * parent_wall_seconds / (2 * 48) / 86400
#   At dw8-equivalent S3 load (~570-650 s): ~3.1-3.6 days
#
# Decision rule (applied by summarize_stage1_outer_parallel_x2.py):
#   proj_effective_days <= 4.0   GREEN  -> run 3-concurrent-chunk benchmark
#   proj_effective_days 4.0-6.0  YELLOW -> discuss before scaling
#   proj_effective_days >= 6.0   RED    -> do not increase concurrency
#
# All outputs under /data42 — NEVER write to system /tmp.
# No code changes. No full-period run. No process workers.
#
# USAGE
# -----
#   # Run under screen to survive SSH disconnection (recommended):
#   screen -S flashnh_outer_x2
#   bash scripts/bench_stage1_outer_parallel_x2_h2o.sh
#   # Ctrl-A D to detach; screen -r flashnh_outer_x2 to reattach
#
# After both chunks complete, run:
#   /data42/omrip/Flash-NH/envs/flashnh-stage1/bin/python \
#       scripts/summarize_stage1_outer_parallel_x2.py \
#       2>&1 | tee /data42/omrip/Flash-NH/tmp/stage1_bench/outer_parallel_rtma_48h_dw8_x2/logs/sanity_output.txt
#
# Expected runtime: 10-14 minutes.

# Do NOT use set -e here. Background process exit codes are captured
# explicitly with 'wait $PID; EXIT=$?'. set -e would abort the script
# before EXIT=$? could run if the child exited non-zero.
set -uo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FLASHNH_ROOT="${FLASHNH_ROOT:-/data42/omrip/Flash-NH}"
FORCING_ROOT="${FLASHNH_ROOT}/tmp/stage1_forcing_fullperiod"
BENCH_BASE="${FLASHNH_ROOT}/tmp/stage1_bench/outer_parallel_rtma_48h_dw8_x2"
LOG_DIR="${BENCH_BASE}/logs"
ENV_PREFIX="${FLASHNH_ROOT}/envs/flashnh-stage1"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

BASIN_LIST="${FORCING_ROOT}/v001_basin_list.csv"
MRMS_WT="${FORCING_ROOT}/02_basin_geometries/weights/mrms/v001_2752_mrms_weights.parquet"
RTMA_WT="${FORCING_ROOT}/02_basin_geometries/weights/rtma/v001_2752_rtma_weights.parquet"

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------

LABEL_A="outer-x2-a"
START_A="2020-10-14T21:00:00"
END_A="2020-10-16T20:00:00"

LABEL_B="outer-x2-b"
START_B="2020-10-16T21:00:00"
END_B="2020-10-18T20:00:00"

DOWNLOAD_WORKERS=8

# ---------------------------------------------------------------------------
# Safety guard: outputs must stay under /data42
# ---------------------------------------------------------------------------

case "${BENCH_BASE}" in
    /data42/*) ;;
    *) echo "ERROR: BENCH_BASE must be under /data42, got: ${BENCH_BASE}"; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Create output directories
# ---------------------------------------------------------------------------

mkdir -p "${LOG_DIR}" "${BENCH_BASE}/chunk_a" "${BENCH_BASE}/chunk_b"

# ---------------------------------------------------------------------------
# Python env setup
# ---------------------------------------------------------------------------

echo "============================================================"
echo "Flash-NH Stage 1 — Outer-Parallelism x2 Benchmark"
echo "============================================================"
echo "Chunk A : ${START_A} -> ${END_A}  label=${LABEL_A}"
echo "Chunk B : ${START_B} -> ${END_B}  label=${LABEL_B}"
echo "Workers : ${DOWNLOAD_WORKERS} per chunk  (${DOWNLOAD_WORKERS} x 2 = $((DOWNLOAD_WORKERS * 2)) total S3 connections)"
echo "Outputs : ${BENCH_BASE}"
echo "============================================================"

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
# Launch both chunks concurrently
# ---------------------------------------------------------------------------

T_START=$(date +%s)
printf "parent_start_utc=%s\nparent_start_epoch=%d\n" "$(date -u)" "${T_START}" \
    > "${LOG_DIR}/parent_timing.txt"

echo ""
echo "Launching chunk A (${LABEL_A}) ..."
/usr/bin/time -v \
    "${ENV_PREFIX}/bin/python" scripts/extract_stage1_forcing_chunk.py \
        --start            "${START_A}" \
        --end              "${END_A}" \
        --basin-manifest   "${BASIN_LIST}" \
        --mrms-weights     "${MRMS_WT}" \
        --rtma-weights     "${RTMA_WT}" \
        --out-dir          "${BENCH_BASE}/chunk_a" \
        --chunk-label      "${LABEL_A}" \
        --products         rtma \
        --download-workers "${DOWNLOAD_WORKERS}" \
        --no-plots \
    &> "${LOG_DIR}/bench_chunk_a.log" &
PID_A=$!

echo "Launching chunk B (${LABEL_B}) ..."
/usr/bin/time -v \
    "${ENV_PREFIX}/bin/python" scripts/extract_stage1_forcing_chunk.py \
        --start            "${START_B}" \
        --end              "${END_B}" \
        --basin-manifest   "${BASIN_LIST}" \
        --mrms-weights     "${MRMS_WT}" \
        --rtma-weights     "${RTMA_WT}" \
        --out-dir          "${BENCH_BASE}/chunk_b" \
        --chunk-label      "${LABEL_B}" \
        --products         rtma \
        --download-workers "${DOWNLOAD_WORKERS}" \
        --no-plots \
    &> "${LOG_DIR}/bench_chunk_b.log" &
PID_B=$!

printf "pid_a=%d\npid_b=%d\n" "${PID_A}" "${PID_B}" >> "${LOG_DIR}/parent_timing.txt"
echo ""
echo "Both chunks running — A=PID:${PID_A}  B=PID:${PID_B}"
echo "Tailing bench_chunk_a.log and bench_chunk_b.log shows per-chunk progress."
echo "Expected runtime: ~10-14 minutes. Waiting ..."

# ---------------------------------------------------------------------------
# Wait for both; capture exit codes explicitly (requires set +e)
# ---------------------------------------------------------------------------

wait "${PID_A}"; EXIT_A=$?
wait "${PID_B}"; EXIT_B=$?

T_END=$(date +%s)
PARENT_WALL=$((T_END - T_START))

printf "parent_end_utc=%s\nparent_end_epoch=%d\nparent_wall_seconds=%d\nexit_a=%d\nexit_b=%d\n" \
    "$(date -u)" "${T_END}" "${PARENT_WALL}" "${EXIT_A}" "${EXIT_B}" \
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
# Results + next steps
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "BENCHMARK COMPLETE"
printf "  Chunk A (%s) exit : %d\n" "${LABEL_A}" "${EXIT_A}"
printf "  Chunk B (%s) exit : %d\n" "${LABEL_B}" "${EXIT_B}"
printf "  Parent wall clock        : %d s\n" "${PARENT_WALL}"
echo "============================================================"

if [ "${EXIT_A}" -ne 0 ] || [ "${EXIT_B}" -ne 0 ]; then
    echo ""
    echo "WARNING: one or both chunks exited non-zero. Check:"
    echo "  tail -50 ${LOG_DIR}/bench_chunk_a.log"
    echo "  tail -50 ${LOG_DIR}/bench_chunk_b.log"
fi

echo ""
echo "Next: run the summary script:"
echo ""
echo "  ${ENV_PREFIX}/bin/python \\"
echo "      scripts/summarize_stage1_outer_parallel_x2.py \\"
echo "      2>&1 | tee ${LOG_DIR}/sanity_output.txt"
echo ""
echo "Then transfer evidence locally (PowerShell on Windows):"
echo ""
echo "  \$local = \"tmp\\stage1_2kd_evidence\\outer_parallel_rtma_48h_dw8_x2\""
echo "  New-Item -ItemType Directory -Force -Path \$local | Out-Null"
echo "  scp omrip@h2o:${LOG_DIR}/bench_chunk_a.log                                              \"\$local\\\""
echo "  scp omrip@h2o:${LOG_DIR}/bench_chunk_b.log                                              \"\$local\\\""
echo "  scp omrip@h2o:${LOG_DIR}/parent_timing.txt                                              \"\$local\\\""
echo "  scp omrip@h2o:${LOG_DIR}/system.txt                                                     \"\$local\\\""
echo "  scp omrip@h2o:${LOG_DIR}/git_state.txt                                                  \"\$local\\\""
echo "  scp omrip@h2o:${LOG_DIR}/sanity_output.txt                                              \"\$local\\\""
echo "  scp omrip@h2o:${LOG_DIR}/summary_outer_x2.csv                                          \"\$local\\\""
echo "  scp \"omrip@h2o:${BENCH_BASE}/chunk_a/manifests/${LABEL_A}_manifest.json\"               \"\$local\\\""
echo "  scp \"omrip@h2o:${BENCH_BASE}/chunk_a/manifests/${LABEL_A}_hourly_runtime_and_volume.csv\" \"\$local\\\""
echo "  scp \"omrip@h2o:${BENCH_BASE}/chunk_a/manifests/${LABEL_A}_validation_checks.csv\"       \"\$local\\\""
echo "  scp \"omrip@h2o:${BENCH_BASE}/chunk_b/manifests/${LABEL_B}_manifest.json\"               \"\$local\\\""
echo "  scp \"omrip@h2o:${BENCH_BASE}/chunk_b/manifests/${LABEL_B}_hourly_runtime_and_volume.csv\" \"\$local\\\""
echo "  scp \"omrip@h2o:${BENCH_BASE}/chunk_b/manifests/${LABEL_B}_validation_checks.csv\"       \"\$local\\\""

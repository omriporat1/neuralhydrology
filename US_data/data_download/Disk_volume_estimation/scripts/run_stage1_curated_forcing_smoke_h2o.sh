#!/usr/bin/env bash
# run_stage1_curated_forcing_smoke_h2o.sh
#
# Smoke launcher for the Stage 1 curated forcing product v001 builder + auditor.
# Run ONLY on h2o where the full-period forcing extraction Parquets reside.
#
# Builds 5 basins × 2020-11 into a timestamped output directory under h2o tmp/,
# then runs the auditor against the output.
#
# USAGE:
#   bash scripts/run_stage1_curated_forcing_smoke_h2o.sh
#   bash scripts/run_stage1_curated_forcing_smoke_h2o.sh --staids 01440000 02231000 03021350 08155541 09484000
#   bash scripts/run_stage1_curated_forcing_smoke_h2o.sh --dry-run
#
# OUTPUT:
#   /data42/omrip/Flash-NH/tmp/stage1_curated_forcing_smoke_YYYYMMDDTHHMMSSZ/
#     time_series/          per-basin Parquets
#     manifest.json
#     checksums.sha256
#     dataset_config.json
#     run_provenance.json
#     build_summary.md
#     smoke.log             full tee'd log
#
# ENVIRONMENT:
#   Uses the explicit Python binary from the flashnh-stage1 conda env.
#   Does NOT rely on 'conda activate' in non-interactive shells (known h2o issue).
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────

H2O_ROOT="/data42/omrip/Flash-NH"
FORCING_ROOT="${H2O_ROOT}/tmp/stage1_forcing_fullperiod"
ENV_PATH="${H2O_ROOT}/envs/flashnh-stage1"
PYTHON_BIN="${ENV_PATH}/bin/python"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MONTH="2020-11"
MAX_BASINS=5
DRY_RUN=0
EXTRA_STAIDS=()

# ── Parse arguments ─────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=1; shift ;;
        --month)        MONTH="$2"; shift 2 ;;
        --max-basins)   MAX_BASINS="$2"; shift 2 ;;
        --staids)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                EXTRA_STAIDS+=("$1"); shift
            done
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--dry-run] [--month YYYY-MM] [--max-basins N] [--staids STAID ...]" >&2
            exit 1
            ;;
    esac
done

# ── Output directory ────────────────────────────────────────────────────────

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${H2O_ROOT}/tmp/stage1_curated_forcing_smoke_${TIMESTAMP}"

# ── Logging ─────────────────────────────────────────────────────────────────

log() { echo "[$(date -u +%H:%M:%S)]  $*"; }

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY-RUN mode — no files will be written"
    log "Repo root:    ${REPO_ROOT}"
    log "Forcing root: ${FORCING_ROOT}"
    log "Output would: ${OUT_DIR}"
    log "Month:        ${MONTH}"
    log "Max basins:   ${MAX_BASINS}"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_stage1_curated_forcing_basin_parquets.py" \
        --forcing-root "${FORCING_ROOT}" \
        --out-dir "${OUT_DIR}" \
        --month "${MONTH}" \
        --max-basins "${MAX_BASINS}" \
        --dry-run
    exit 0
fi

# ── Pre-flight checks ────────────────────────────────────────────────────────

log "=== Stage 1 curated forcing smoke build ==="
log "Repo root:    ${REPO_ROOT}"
log "Forcing root: ${FORCING_ROOT}"
log "Output dir:   ${OUT_DIR}"
log "Month:        ${MONTH}"
log "Max basins:   ${MAX_BASINS}"

# Validate h2o safety guard
if [[ "${OUT_DIR}" != /data42/omrip/Flash-NH/* ]]; then
    echo "ERROR: OUT_DIR does not begin with /data42/omrip/Flash-NH/" >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python binary not found or not executable: ${PYTHON_BIN}" >&2
    echo "  Verify the flashnh-stage1 env exists at: ${ENV_PATH}" >&2
    exit 1
fi

CHUNK_PATH="${FORCING_ROOT}/chunks/${MONTH}/combined_${MONTH}.parquet"
if [[ ! -f "${CHUNK_PATH}" ]]; then
    echo "ERROR: Source Parquet not found: ${CHUNK_PATH}" >&2
    exit 1
fi

log "Python:       $("${PYTHON_BIN}" --version 2>&1)"
log "Source:       ${CHUNK_PATH}"

# ── Create output directory and tee log ─────────────────────────────────────

mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/smoke.log"

exec > >(tee -a "${LOG_FILE}") 2>&1
log "Log file:     ${LOG_FILE}"

# ── Build ────────────────────────────────────────────────────────────────────

BUILD_ARGS=(
    --forcing-root "${FORCING_ROOT}"
    --out-dir      "${OUT_DIR}"
    --month        "${MONTH}"
    --max-basins   "${MAX_BASINS}"
)
if [[ ${#EXTRA_STAIDS[@]} -gt 0 ]]; then
    BUILD_ARGS+=(--staids "${EXTRA_STAIDS[@]}")
fi

log "--- Running builder ---"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_stage1_curated_forcing_basin_parquets.py" \
    "${BUILD_ARGS[@]}"

BUILD_EXIT=$?
if [[ "${BUILD_EXIT}" -ne 0 ]]; then
    log "ERROR: Builder exited with code ${BUILD_EXIT}"
    exit "${BUILD_EXIT}"
fi

log "Builder complete."

# ── Audit ────────────────────────────────────────────────────────────────────

log "--- Running auditor ---"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/audit_stage1_curated_forcing_basin_parquets.py" \
    --product-dir "${OUT_DIR}" \
    --month       "${MONTH}"

AUDIT_EXIT=$?
if [[ "${AUDIT_EXIT}" -ne 0 ]]; then
    log "ERROR: Auditor exited with code ${AUDIT_EXIT} — smoke FAIL"
    exit "${AUDIT_EXIT}"
fi

# ── Summary ──────────────────────────────────────────────────────────────────

log "=== SMOKE RESULT: PASS ==="
log "Output:   ${OUT_DIR}"
log "Log:      ${LOG_FILE}"
log "To review: cat ${OUT_DIR}/build_summary.md"
#!/usr/bin/env bash
# run_wy2026_curated_forcing_h2o.sh
#
# Launcher for the WY2026 curated forcing product (docs/wy2026_data_plan.md §3 C1, §9 L5).
# Run ONLY on h2o, where the monthly chunk Parquets reside.
#
# Builds a separate product for water year 2026 from the same monthly chunks as v001:
#   - period 2025-10-01T00:00:00Z .. 2026-09-30T23:00:00Z (12 months, 8,760 h per basin)
#   - v001 basin list (2,752 basins) OR fewer with --max-basins / --staids
#   - expected per-basin gaps: 0 MRMS, 1 RTMA (2026-09-23T19:00Z, absent upstream)
#   - output: PRODUCT_DIR below, never the v001 product directory
#
# Steps: builder (period args) -> curated auditor (--full-period, WY2026 pins)
#        -> overlap identity check of 2025-10-01..2025-12-31 against v001.
#
# USAGE:
#   bash scripts/run_wy2026_curated_forcing_h2o.sh --dry-run
#   bash scripts/run_wy2026_curated_forcing_h2o.sh --max-basins 5     # bounded test
#   bash scripts/run_wy2026_curated_forcing_h2o.sh                    # full build
#
# The builder is never passed --overwrite; a rerun resumes from what exists.
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────

H2O_ROOT="/data42/omrip/Flash-NH"
FORCING_ROOT="${H2O_ROOT}/tmp/stage1_forcing_fullperiod"
PRODUCT_NAME="stage1_basin_hourly_forcings_wy2026_v001"
PRODUCT_DIR="${FORCING_ROOT}/${PRODUCT_NAME}"
V001_DIR="${FORCING_ROOT}/stage1_basin_hourly_forcings_v001"
BASIN_LIST="${FORCING_ROOT}/v001_basin_list.csv"
ENV_PATH="${H2O_ROOT}/envs/flashnh-stage1"
PYTHON_BIN="${ENV_PATH}/bin/python"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PERIOD_START="2025-10-01T00:00:00Z"
PERIOD_END="2026-09-30T23:00:00Z"
EXPECTED_HOURS=8760
EXPECTED_MRMS_GAPS=0
EXPECTED_RTMA_GAPS=1
KNOWN_RTMA_GAPS="2026-09-23T19:00:00Z"
OVERLAP_START="2025-10-01T00:00:00Z"
OVERLAP_END="2025-12-31T23:00:00Z"

DRY_RUN=0
MAX_BASINS=""
EXTRA_STAIDS=()

# ── Parse arguments ─────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=1; shift ;;
        --max-basins)   MAX_BASINS="$2"; shift 2 ;;
        --staids)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                EXTRA_STAIDS+=("$1"); shift
            done
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--dry-run] [--max-basins N] [--staids STAID ...]" >&2
            exit 1
            ;;
    esac
done

if [[ -n "${MAX_BASINS}" && ${#EXTRA_STAIDS[@]} -gt 0 ]]; then
    echo "ERROR: use --max-basins or --staids, not both" >&2
    exit 1
fi

log() { echo "[$(date -u +%H:%M:%S)]  $*"; }

# ── Pre-flight checks ────────────────────────────────────────────────────────

log "=== WY2026 curated forcing build ==="
log "Repo root:    ${REPO_ROOT}"
log "Forcing root: ${FORCING_ROOT}"
log "Product dir:  ${PRODUCT_DIR}"
log "Period:       ${PERIOD_START} .. ${PERIOD_END} (${EXPECTED_HOURS} h)"

if [[ "${PRODUCT_DIR}" != /data42/omrip/Flash-NH/* ]]; then
    echo "ERROR: PRODUCT_DIR does not begin with /data42/omrip/Flash-NH/" >&2
    exit 1
fi
if [[ "$(realpath -m "${PRODUCT_DIR}")" == "$(realpath -m "${V001_DIR}")" ]]; then
    echo "ERROR: PRODUCT_DIR is the v001 product directory; refusing" >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python binary not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
fi
log "Python version: $("${PYTHON_BIN}" --version 2>&1)"

if [[ ! -f "${BASIN_LIST}" ]]; then
    echo "ERROR: Basin list CSV not found: ${BASIN_LIST}" >&2
    exit 1
fi
# All 12 monthly chunks must exist; the builder also refuses a partial period.
WY_MONTHS=(2025-10 2025-11 2025-12 2026-01 2026-02 2026-03
           2026-04 2026-05 2026-06 2026-07 2026-08 2026-09)
MISSING_CHUNKS=0
for M in "${WY_MONTHS[@]}"; do
    C="${FORCING_ROOT}/chunks/${M}/combined_${M}.parquet"
    if [[ ! -f "${C}" ]]; then
        echo "ERROR: Monthly chunk not found: ${C}" >&2
        MISSING_CHUNKS=$((MISSING_CHUNKS + 1))
    fi
done
if [[ "${MISSING_CHUNKS}" -gt 0 ]]; then
    echo "ERROR: ${MISSING_CHUNKS} of ${#WY_MONTHS[@]} monthly chunks missing; refusing" >&2
    exit 1
fi

BUILDER_ARGS=(
    --forcing-root       "${FORCING_ROOT}"
    --out-dir            "${PRODUCT_DIR}"
    --basin-list         "${BASIN_LIST}"
    --period-start       "${PERIOD_START}"
    --period-end         "${PERIOD_END}"
    --expected-hours     "${EXPECTED_HOURS}"
    --expected-mrms-gaps "${EXPECTED_MRMS_GAPS}"
    --expected-rtma-gaps "${EXPECTED_RTMA_GAPS}"
    --product-name       "${PRODUCT_NAME}"
)
[[ -n "${MAX_BASINS}" ]]         && BUILDER_ARGS+=(--max-basins "${MAX_BASINS}")
[[ ${#EXTRA_STAIDS[@]} -gt 0 ]] && BUILDER_ARGS+=(--staids "${EXTRA_STAIDS[@]}")

# ── Dry-run ──────────────────────────────────────────────────────────────────

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY-RUN mode — no files will be written"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_stage1_curated_forcing_fullperiod.py" \
        "${BUILDER_ARGS[@]}" --dry-run
    log "Dry-run complete."
    exit 0
fi

# ── Create output directory and tee log ─────────────────────────────────────

mkdir -p "${PRODUCT_DIR}"
LOG_FILE="${PRODUCT_DIR}/build.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
log "Log file: ${LOG_FILE}"

# ── Build ────────────────────────────────────────────────────────────────────

log "--- Running builder (WY2026 period) ---"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_stage1_curated_forcing_fullperiod.py" \
    "${BUILDER_ARGS[@]}"
log "Builder complete."

# ── Audit ────────────────────────────────────────────────────────────────────

AUDITOR_ARGS=(
    --product-dir                "${PRODUCT_DIR}"
    --full-period
    --period-start               "${PERIOD_START}"
    --period-end                 "${PERIOD_END}"
    --expected-rows              "${EXPECTED_HOURS}"
    --expected-mrms-gaps         "${EXPECTED_MRMS_GAPS}"
    --expected-rtma-gaps         "${EXPECTED_RTMA_GAPS}"
    --known-rtma-gap-timestamps  "${KNOWN_RTMA_GAPS}"
    --product-name               "${PRODUCT_NAME}"
)
if [[ -n "${MAX_BASINS}" ]]; then
    AUDITOR_ARGS+=(--expected-basins "${MAX_BASINS}")
elif [[ ${#EXTRA_STAIDS[@]} -gt 0 ]]; then
    AUDITOR_ARGS+=(--expected-basins "${#EXTRA_STAIDS[@]}")
fi

log "--- Running curated auditor (WY2026 pins) ---"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/audit_stage1_curated_forcing_basin_parquets.py" \
    "${AUDITOR_ARGS[@]}"
log "Auditor PASS."

# ── Overlap identity vs v001 (2025-10..12 from the same chunks) ─────────────

log "--- Overlap identity check vs v001 (${OVERLAP_START} .. ${OVERLAP_END}) ---"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_curated_forcing_overlap_identity.py" \
    --new-dir      "${PRODUCT_DIR}" \
    --ref-dir      "${V001_DIR}" \
    --window-start "${OVERLAP_START}" \
    --window-end   "${OVERLAP_END}"

log "=== WY2026 BUILD + AUDIT + OVERLAP IDENTITY: PASS ==="
log "Product: ${PRODUCT_DIR}"
log "Log:     ${LOG_FILE}"

#!/usr/bin/env bash
# run_stage1_curated_forcing_fullperiod_h2o.sh
#
# Full-period launcher for the Stage 1 curated forcing product v001 builder + auditor.
# Run ONLY on h2o where the full-period extraction chunk Parquets reside.
#
# Builds the complete Stage 1 curated product:
#   - 2,752 basins (v001 basin list) OR fewer when --max-basins is given
#   - 63 months: 2020-10 through 2025-12 (45,720 hours per basin)
#   - Output under: PRODUCT_DIR (see Configuration below)
#
# USAGE:
#   # Dry-run: validate inputs and print plan without writing files
#   bash scripts/run_stage1_curated_forcing_fullperiod_h2o.sh --dry-run
#
#   # Bounded test (5 basins, all 63 months) — safe before full launch
#   bash scripts/run_stage1_curated_forcing_fullperiod_h2o.sh --max-basins 5
#
#   # Full 2,752-basin build (requires explicit authorization)
#   bash scripts/run_stage1_curated_forcing_fullperiod_h2o.sh
#
# OUTPUTS:
#   {PRODUCT_DIR}/
#     time_series/          2,752 per-basin Parquets (one file per STAID)
#     manifest.json
#     checksums.sha256
#     dataset_config.json
#     run_provenance.json
#     build_summary.md
#     audit_summary.md      (written by auditor)
#     build.log             tee'd log of builder + auditor
#
# EVIDENCE BUNDLE (after successful audit):
#   tar czf /data42/omrip/Flash-NH/tmp/stage1_curated_forcing_v001_evidence_$(date -u +%Y%m%dT%H%M%SZ).tar.gz \
#       -C ${PRODUCT_DIR} manifest.json checksums.sha256 dataset_config.json \
#       run_provenance.json build_summary.md audit_summary.md
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────

H2O_ROOT="/data42/omrip/Flash-NH"
FORCING_ROOT="${H2O_ROOT}/tmp/stage1_forcing_fullperiod"
PRODUCT_DIR="${FORCING_ROOT}/stage1_basin_hourly_forcings_v001"
BASIN_LIST="${FORCING_ROOT}/v001_basin_list.csv"
ENV_PATH="${H2O_ROOT}/envs/flashnh-stage1"
PYTHON_BIN="${ENV_PATH}/bin/python"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

# ── Logging helper ──────────────────────────────────────────────────────────

log() { echo "[$(date -u +%H:%M:%S)]  $*"; }

# ── Pre-flight checks ────────────────────────────────────────────────────────

log "=== Stage 1 curated forcing — full-period build ==="
log "Repo root:    ${REPO_ROOT}"
log "Forcing root: ${FORCING_ROOT}"
log "Product dir:  ${PRODUCT_DIR}"
log "Basin list:   ${BASIN_LIST}"
log "Python:       ${PYTHON_BIN}"

# Path safety: output must stay under h2o
if [[ "${PRODUCT_DIR}" != /data42/omrip/Flash-NH/* ]]; then
    echo "ERROR: PRODUCT_DIR does not begin with /data42/omrip/Flash-NH/" >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python binary not found or not executable: ${PYTHON_BIN}" >&2
    echo "  Verify the flashnh-stage1 env: ${ENV_PATH}" >&2
    exit 1
fi

log "Python version: $("${PYTHON_BIN}" --version 2>&1)"

if [[ "${DRY_RUN}" -eq 0 && ! -f "${BASIN_LIST}" ]]; then
    echo "ERROR: Basin list CSV not found: ${BASIN_LIST}" >&2
    exit 1
fi

FIRST_CHUNK="${FORCING_ROOT}/chunks/2020-10/combined_2020-10.parquet"
if [[ ! -f "${FIRST_CHUNK}" ]]; then
    echo "ERROR: First monthly chunk not found: ${FIRST_CHUNK}" >&2
    exit 1
fi

# ── Dry-run ──────────────────────────────────────────────────────────────────

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY-RUN mode — no files will be written"
    BUILDER_ARGS=(
        --forcing-root "${FORCING_ROOT}"
        --out-dir      "${PRODUCT_DIR}"
        --dry-run
    )
    [[ -n "${MAX_BASINS}" ]] && BUILDER_ARGS+=(--max-basins "${MAX_BASINS}")
    [[ -f "${BASIN_LIST}" ]] && BUILDER_ARGS+=(--basin-list "${BASIN_LIST}")
    [[ ${#EXTRA_STAIDS[@]} -gt 0 ]] && BUILDER_ARGS+=(--staids "${EXTRA_STAIDS[@]}")
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_stage1_curated_forcing_fullperiod.py" \
        "${BUILDER_ARGS[@]}"
    log "Dry-run complete. Review the plan above, then launch without --dry-run."
    exit 0
fi

# ── Create output directory and tee log ─────────────────────────────────────

mkdir -p "${PRODUCT_DIR}"
LOG_FILE="${PRODUCT_DIR}/build.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
log "Log file: ${LOG_FILE}"

# ── Build ────────────────────────────────────────────────────────────────────

BUILDER_ARGS=(
    --forcing-root "${FORCING_ROOT}"
    --out-dir      "${PRODUCT_DIR}"
    --basin-list   "${BASIN_LIST}"
)
[[ -n "${MAX_BASINS}" ]]             && BUILDER_ARGS+=(--max-basins "${MAX_BASINS}")
[[ ${#EXTRA_STAIDS[@]} -gt 0 ]]     && BUILDER_ARGS+=(--staids "${EXTRA_STAIDS[@]}")

log "--- Running full-period builder ---"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_stage1_curated_forcing_fullperiod.py" \
    "${BUILDER_ARGS[@]}"

BUILD_EXIT=$?
if [[ "${BUILD_EXIT}" -ne 0 ]]; then
    log "ERROR: Builder exited with code ${BUILD_EXIT}"
    exit "${BUILD_EXIT}"
fi
log "Builder complete."

# ── Audit ────────────────────────────────────────────────────────────────────

AUDITOR_ARGS=(
    --product-dir "${PRODUCT_DIR}"
    --full-period
)
[[ -n "${MAX_BASINS}" ]] && AUDITOR_ARGS+=(--expected-basins "${MAX_BASINS}")

log "--- Running full-period auditor ---"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/audit_stage1_curated_forcing_basin_parquets.py" \
    "${AUDITOR_ARGS[@]}"

AUDIT_EXIT=$?
if [[ "${AUDIT_EXIT}" -ne 0 ]]; then
    log "ERROR: Auditor exited with code ${AUDIT_EXIT} — BUILD FAIL"
    exit "${AUDIT_EXIT}"
fi

# ── Summary ──────────────────────────────────────────────────────────────────

log "=== BUILD + AUDIT RESULT: PASS ==="
log "Product:  ${PRODUCT_DIR}"
log "Log:      ${LOG_FILE}"
log ""
log "Evidence bundle command (run after reviewing build_summary.md and audit_summary.md):"
log "  BUNDLE=\"${H2O_ROOT}/tmp/stage1_curated_forcing_v001_evidence_\$(date -u +%Y%m%dT%H%M%SZ).tar.gz\""
log "  tar czf \"\${BUNDLE}\" -C \"${PRODUCT_DIR}\" \\"
log "      manifest.json checksums.sha256 dataset_config.json \\"
log "      run_provenance.json build_summary.md audit_summary.md"
log "  echo \"Bundle: \${BUNDLE}\""
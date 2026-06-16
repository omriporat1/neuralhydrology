#!/usr/bin/env bash
# Stage 1 Forcing — INPUT VERIFICATION for h2o (read-only)
#
# PURPOSE
# -------
# Verifies that all required inputs for the Stage 1 forcing acquisition
# pipeline are present and valid on h2o BEFORE any download or extraction
# runs. This script is read-only: it never modifies files, downloads data,
# or runs model code.
#
# Run this first after SSH-ing to h2o, and resolve all MISSING items before
# proceeding to any extraction milestone.
#
# CHECKS PERFORMED
# ----------------
#  1. v001 target package manifest exists
#  2. v001_basin_list.csv exists and has expected row count (2,752)
#  3. CAMELSH shapefile — discovery attempt (location may be unknown)
#  4. MRMS grid definition JSON exists
#  5. RTMA grid definition JSON exists
#  6. MRMS weight Parquet (v001_2752) — optional; OK if not yet built
#  7. RTMA weight Parquet (v001_2752) — optional; OK if not yet built
#  8. Forcing output root exists or is creatable
#  9. flashnh-stage1 conda env exists
# 10. No credentials embedded in this script (self-check documentation)
#
# USAGE
# -----
#   bash scripts/verify_stage1_forcing_inputs_h2o.sh
#
#   # Override forcing root:
#   FORCING_ROOT=/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod \
#       bash scripts/verify_stage1_forcing_inputs_h2o.sh
#
# CAMELSH SHAPEFILE DISCOVERY
# ---------------------------
# The CAMELSH shapefile location on h2o is not known at script-write time.
# This script searches for it automatically with 'find'. If not found,
# it prints the transfer command needed.
#
# If found at a non-standard path, pass it explicitly to build_stage1_basin_weights.py:
#   python scripts/build_stage1_basin_weights.py \
#       --camelsh-polygons /found/path/CAMELSH_shapefile.shp \
#       --basin-list ${FORCING_ROOT}/v001_basin_list.csv \
#       --out-tag v001_2752

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FLASHNH_ROOT="${FLASHNH_ROOT:-/data42/omrip/Flash-NH}"
FORCING_ROOT="${FORCING_ROOT:-${FLASHNH_ROOT}/tmp/stage1_forcing_fullperiod}"
V001_PKG="${FLASHNH_ROOT}/tmp/stage1_target_package_v001"
ENV_PREFIX="${FLASHNH_ROOT}/envs/flashnh-stage1"

BASIN_LIST="${FORCING_ROOT}/v001_basin_list.csv"
MRMS_WEIGHTS="${FORCING_ROOT}/02_basin_geometries/weights/mrms/v001_2752_mrms_weights.parquet"
RTMA_WEIGHTS="${FORCING_ROOT}/02_basin_geometries/weights/rtma/v001_2752_rtma_weights.parquet"
GRID_DEF_DIR="${FORCING_ROOT}/grid_definitions"
MRMS_GRID_JSON="${GRID_DEF_DIR}/mrms_grid_definition.json"
RTMA_GRID_JSON="${GRID_DEF_DIR}/rtma_grid_definition.json"

# Standard auto-discovery path for CAMELSH shapefile
CAMELSH_STANDARD="${FORCING_ROOT}/02_basin_geometries/camelsh/shapefiles/CAMELSH_shapefile.shp"

EXPECTED_BASIN_COUNT=2752

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

_pass() { echo "  [PASS] $1"; PASS_COUNT=$(( PASS_COUNT + 1 )); }
_warn() { echo "  [WARN] $1"; WARN_COUNT=$(( WARN_COUNT + 1 )); }
_fail() { echo "  [FAIL] $1"; FAIL_COUNT=$(( FAIL_COUNT + 1 )); }
_info() { echo "         $1"; }

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

echo "============================================================"
echo "Flash-NH Stage 1 Forcing — Input Verification"
echo "============================================================"
echo "Forcing root: ${FORCING_ROOT}"
echo "v001 package: ${V001_PKG}"
echo "Checked:      $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Check 1: v001 target package manifest
# ---------------------------------------------------------------------------

echo "[1] v001 target package manifest"
if [ -f "${V001_PKG}/manifest.json" ]; then
    N_BASINS=$(python3 -c "
import json
try:
    d = json.load(open('${V001_PKG}/manifest.json'))
    print(len(d.get('basins', [])))
except Exception as e:
    print(0)
" 2>/dev/null)
    if [ "${N_BASINS}" = "${EXPECTED_BASIN_COUNT}" ]; then
        _pass "manifest.json exists, basins=${N_BASINS}"
    elif [ "${N_BASINS}" = "0" ]; then
        _fail "manifest.json exists but could not read 'basins' list"
        _info "File: ${V001_PKG}/manifest.json"
        _info "Check: python3 -c \"import json; print(list(json.load(open('${V001_PKG}/manifest.json')).keys()))\""
    else
        _warn "manifest.json exists but basins=${N_BASINS} (expected ${EXPECTED_BASIN_COUNT})"
        _info "If this is intentional, adjust EXPECTED_BASIN_COUNT in this script."
    fi
else
    _fail "v001 manifest not found: ${V001_PKG}/manifest.json"
    _info "The v001 target package must exist on h2o before forcing acquisition."
    _info "Expected location: ${V001_PKG}"
fi
echo ""

# ---------------------------------------------------------------------------
# Check 2: v001_basin_list.csv
# ---------------------------------------------------------------------------

echo "[2] v001_basin_list.csv"
if [ -f "${BASIN_LIST}" ]; then
    # Count data rows (excluding header)
    N_ROWS=$(tail -n +2 "${BASIN_LIST}" | wc -l | tr -d ' ')
    if [ "${N_ROWS}" = "${EXPECTED_BASIN_COUNT}" ]; then
        _pass "v001_basin_list.csv exists, rows=${N_ROWS}"
    else
        _warn "v001_basin_list.csv exists but has ${N_ROWS} data rows (expected ${EXPECTED_BASIN_COUNT})"
        _info "Regenerate with: python scripts/export_v001_basin_list.py --force"
    fi
    # Check header
    HEADER=$(head -1 "${BASIN_LIST}")
    if [ "${HEADER}" != "STAID" ]; then
        _fail "First column header is '${HEADER}', expected 'STAID'"
    fi
else
    _fail "v001_basin_list.csv not found: ${BASIN_LIST}"
    _info "Generate with:"
    _info "  python scripts/export_v001_basin_list.py"
    _info "  (reads ${V001_PKG}/manifest.json)"
fi
echo ""

# ---------------------------------------------------------------------------
# Check 3: CAMELSH shapefile (discovery)
# ---------------------------------------------------------------------------

echo "[3] CAMELSH shapefile (GAGES-II polygons for weight build)"
_info "Required for: python scripts/build_stage1_basin_weights.py (Milestone 2K-A)"
_info "Note: pilot weights (50-basin) are NOT valid for v001 — new 2,752-basin"
_info "      weight tables must be built from the CAMELSH shapefile."
_info "Searching /data42 for CAMELSH_shapefile.shp ..."

CAMELSH_FOUND=""
if command -v find &>/dev/null; then
    CAMELSH_FOUND=$(find /data42 -name "CAMELSH_shapefile.shp" 2>/dev/null | head -5 || true)
fi

if [ -n "${CAMELSH_FOUND}" ]; then
    while IFS= read -r found_path; do
        _pass "Found: ${found_path}"
    done <<< "${CAMELSH_FOUND}"
    FIRST_FOUND=$(echo "${CAMELSH_FOUND}" | head -1)
    if [ "${FIRST_FOUND}" != "${CAMELSH_STANDARD}" ]; then
        _info "Standard auto-discovery path: ${CAMELSH_STANDARD}"
        _info "If not at the standard path, pass explicitly:"
        _info "  python scripts/build_stage1_basin_weights.py \\"
        _info "      --camelsh-polygons ${FIRST_FOUND} \\"
        _info "      --basin-list ${BASIN_LIST} --out-tag v001_2752"
    fi
elif [ -f "${CAMELSH_STANDARD}" ]; then
    _pass "Found at standard path: ${CAMELSH_STANDARD}"
else
    _fail "CAMELSH_shapefile.shp not found anywhere under /data42"
    _info "Options:"
    _info "  A) Transfer from local Windows machine (run on local PowerShell):"
    _info "       scripts/prepare_stage1_forcing_inputs_h2o.ps1"
    _info "  B) Download from DOI 10.5281/zenodo.15066778 on h2o:"
    _info "       wget -O /tmp/shapefiles.7z 'https://zenodo.org/records/15066778/files/shapefiles.7z'"
    _info "       7z x /tmp/shapefiles.7z -o${FORCING_ROOT}/02_basin_geometries/camelsh/"
    _info "       # Result: ${CAMELSH_STANDARD}"
fi
echo ""

# ---------------------------------------------------------------------------
# Check 4: MRMS grid definition JSON
# ---------------------------------------------------------------------------

echo "[4] MRMS grid definition JSON"
if [ -f "${MRMS_GRID_JSON}" ]; then
    _pass "${MRMS_GRID_JSON}"
else
    _fail "MRMS grid definition not found: ${MRMS_GRID_JSON}"
    _info "Transfer from local machine (run on local PowerShell):"
    _info "  scripts/prepare_stage1_forcing_inputs_h2o.ps1"
    _info "Or manually:"
    _info "  scp <local>:tmp/stage1_pilot_dryrun/09_manifests/stage1_pilot/grid_definitions/mrms_grid_definition.json \\"
    _info "      flashnh-h2o:${GRID_DEF_DIR}/"
fi
echo ""

# ---------------------------------------------------------------------------
# Check 5: RTMA grid definition JSON
# ---------------------------------------------------------------------------

echo "[5] RTMA grid definition JSON"
if [ -f "${RTMA_GRID_JSON}" ]; then
    _pass "${RTMA_GRID_JSON}"
else
    _fail "RTMA grid definition not found: ${RTMA_GRID_JSON}"
    _info "Transfer from local machine (run on local PowerShell):"
    _info "  scripts/prepare_stage1_forcing_inputs_h2o.ps1"
    _info "Or manually:"
    _info "  scp <local>:tmp/stage1_pilot_dryrun/09_manifests/stage1_pilot/grid_definitions/rtma_grid_definition.json \\"
    _info "      flashnh-h2o:${GRID_DEF_DIR}/"
fi
echo ""

# ---------------------------------------------------------------------------
# Check 6–7: Weight Parquets (optional — expected absent before 2K-A)
# ---------------------------------------------------------------------------

echo "[6] MRMS weight Parquet (v001_2752) — optional before Milestone 2K-A"
if [ -f "${MRMS_WEIGHTS}" ]; then
    SIZE=$(du -sh "${MRMS_WEIGHTS}" 2>/dev/null | cut -f1)
    _pass "v001_2752_mrms_weights.parquet exists (${SIZE})"
else
    _warn "v001_2752_mrms_weights.parquet not yet built — expected; build in Milestone 2K-A"
    _info "Build command:"
    _info "  python scripts/build_stage1_basin_weights.py \\"
    _info "      --basin-list ${BASIN_LIST} --out-tag v001_2752 \\"
    _info "      --data-root ${FORCING_ROOT}"
fi
echo ""

echo "[7] RTMA weight Parquet (v001_2752) — optional before Milestone 2K-A"
if [ -f "${RTMA_WEIGHTS}" ]; then
    SIZE=$(du -sh "${RTMA_WEIGHTS}" 2>/dev/null | cut -f1)
    _pass "v001_2752_rtma_weights.parquet exists (${SIZE})"
else
    _warn "v001_2752_rtma_weights.parquet not yet built — expected; build in Milestone 2K-A"
fi
echo ""

# ---------------------------------------------------------------------------
# Check 8: Output root creatable
# ---------------------------------------------------------------------------

echo "[8] Forcing output root"
if [ -d "${FORCING_ROOT}" ]; then
    SIZE=$(du -sh "${FORCING_ROOT}" 2>/dev/null | cut -f1 || echo "unknown")
    _pass "Exists: ${FORCING_ROOT} (${SIZE})"
    # Disk check
    AVAIL=$(df -h "${FORCING_ROOT}" 2>/dev/null | tail -1 | awk '{print $4}' || echo "unknown")
    _info "Disk available on this partition: ${AVAIL}"
elif mkdir -p "${FORCING_ROOT}" 2>/dev/null; then
    _pass "Created: ${FORCING_ROOT}"
    rmdir "${FORCING_ROOT}" 2>/dev/null || true
else
    _fail "Cannot create output root: ${FORCING_ROOT}"
    _info "Check permissions on /data42/omrip"
fi
echo ""

# ---------------------------------------------------------------------------
# Check 9: flashnh-stage1 conda environment
# ---------------------------------------------------------------------------

echo "[9] flashnh-stage1 conda environment"
if [ -d "${ENV_PREFIX}" ]; then
    PY_VER=$(${ENV_PREFIX}/bin/python --version 2>&1 || echo "unknown")
    _pass "Environment exists: ${ENV_PREFIX} (${PY_VER})"
else
    _fail "Environment not found: ${ENV_PREFIX}"
    _info "See docs/stage1_environment.md for install instructions."
fi
echo ""

# ---------------------------------------------------------------------------
# Check 10: Self-check — no credentials in this script
# ---------------------------------------------------------------------------

echo "[10] Credential safety (self-check)"
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
if grep -Ei "(password|passwd|secret|token|api.?key)\s*=\s*['\"][^'\"]{4,}" "${SCRIPT_PATH}" &>/dev/null; then
    _fail "Potential credential found in this script — review immediately"
else
    _pass "No hardcoded credentials found in script"
fi
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo "============================================================"
echo "VERIFICATION SUMMARY"
echo "============================================================"
echo "  PASS: ${PASS_COUNT}"
echo "  WARN: ${WARN_COUNT}  (weight Parquets not yet built is expected at this stage)"
echo "  FAIL: ${FAIL_COUNT}"
echo "============================================================"
echo ""

if [ ${FAIL_COUNT} -gt 0 ]; then
    echo "Action required: resolve FAIL items above before running any h2o jobs."
    echo ""
    echo "Typical resolution order:"
    echo "  1. Confirm v001 target package exists (Check 1)"
    echo "  2. Generate v001_basin_list.csv (Check 2):"
    echo "       python scripts/export_v001_basin_list.py"
    echo "  3. Transfer grid definitions from local (Check 4-5):"
    echo "       [on local Windows] PowerShell scripts/prepare_stage1_forcing_inputs_h2o.ps1"
    echo "  4. Locate or transfer CAMELSH shapefile (Check 3):"
    echo "       [on local Windows] PowerShell scripts/prepare_stage1_forcing_inputs_h2o.ps1 -TransferShapefiles"
    echo "  5. Build weight tables — Milestone 2K-A (Checks 6-7):"
    echo "       python scripts/build_stage1_basin_weights.py --basin-list ... --out-tag v001_2752"
    exit 1
fi

if [ ${WARN_COUNT} -gt 0 ]; then
    echo "All required inputs present. WARN items are expected at this stage."
    echo "Proceed to Milestone 2K-A (weight build) to resolve weight Parquet warnings."
    exit 0
fi

echo "All checks PASS. Ready to proceed to Milestone 2K-A."
exit 0

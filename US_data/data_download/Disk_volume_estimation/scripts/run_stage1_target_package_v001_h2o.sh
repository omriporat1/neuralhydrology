#!/usr/bin/env bash
# Flash-NH Stage 1 -- Full target package build v001 (h2o)
#
# Excludes special-review basins 02299472 and 04073468.
# All paths are hard-coded for h2o.es.huji.ac.il layout.
#
# Run inside a screen session on h2o:
#   screen -S flashnh_target_v001
#   bash scripts/run_stage1_target_package_v001_h2o.sh
set -euo pipefail

# -- Configuration (hard-coded for v001) ------------------------------------
CANONICAL_DIR="/data42/omrip/Flash-NH/tmp/stage1_full_2843/canonical_merged"
STATUS_CSV="/data42/omrip/Flash-NH/tmp/stage1_full_2843/audit/target_status.csv"
POLICY="config/stage1_target_policy.yaml"
OUT_DIR="/data42/omrip/Flash-NH/tmp/stage1_target_package_v001"
LOG_DIR="/data42/omrip/Flash-NH/tmp/stage1_target_package_v001_logs"
EXCLUDE_STAIDS="02299472,04073468"
EXPECTED_BASINS=2752
EXPECTED_NC_COUNT=2843
CONDA_PROFILE="/opt/conda/etc/profile.d/conda.sh"
ENV_PREFIX="/data42/omrip/Flash-NH/envs/flashnh-stage1"

FORCE=0

# -- Help -------------------------------------------------------------------
usage() {
    cat <<'EOF'
Usage: bash scripts/run_stage1_target_package_v001_h2o.sh [--force] [--help]

Runs the Flash-NH Stage 1 full target package build + audit (v001) on h2o.
Designed for h2o.es.huji.ac.il -- paths are hard-coded for the h2o layout.
Run this inside a named screen session so it survives disconnections.

Excluded special-review basins: 02299472 (2,605 neg), 04073468 (2,054 neg).
Expected output: 2,752 basin NCs in stage1_target_package_v001/.

Options:
  --force    Allow overwriting an existing output directory.
  --help     Show this message.

Generated outputs (do NOT stage or commit):
  /data42/omrip/Flash-NH/tmp/stage1_target_package_v001/
  /data42/omrip/Flash-NH/tmp/stage1_target_package_v001_logs/
EOF
}

# -- Argument parsing -------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --force)    FORCE=1 ;;
        --help|-h)  usage; exit 0 ;;
        *)          echo "Unknown argument: $arg" >&2; usage >&2; exit 1 ;;
    esac
done

# -- Hostname warning -------------------------------------------------------
CURRENT_HOST=$(hostname -f 2>/dev/null || hostname)
if [[ "$CURRENT_HOST" != *h2o* ]]; then
    echo "WARNING: hostname='$CURRENT_HOST' (expected h2o.es.huji.ac.il)"
    echo "         Paths are h2o-specific and may not exist on this machine."
fi

# -- Conda activation -------------------------------------------------------
# Temporarily relax -u; conda init scripts may reference unbound variables.
set +u
if [ -f "$CONDA_PROFILE" ]; then
    # shellcheck source=/dev/null
    source "$CONDA_PROFILE"
fi
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not available. Source a conda profile or pre-activate the env." >&2
    exit 1
fi
conda activate "$ENV_PREFIX"
set -u

# -- Repo root -------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# -- Preflight -------------------------------------------------------------
echo "======================================================================"
echo "Flash-NH Stage 1 v001 Launcher -- Preflight"
echo "======================================================================"
echo "  Repo root:  $REPO_ROOT"
echo ""

echo "[preflight] Required files ..."
for f in \
    "$POLICY" \
    "$STATUS_CSV" \
    "scripts/build_stage1_target_package.py" \
    "scripts/audit_stage1_target_package.py"
do
    if [ ! -f "$f" ]; then
        echo "  ERROR: Missing required file: $f" >&2; exit 1
    fi
    echo "  OK: $f"
done

echo ""
echo "[preflight] canonical_merged ..."
if [ ! -d "$CANONICAL_DIR" ]; then
    echo "  ERROR: Directory not found: $CANONICAL_DIR" >&2; exit 1
fi
NC_FLAT=$(find "$CANONICAL_DIR" -maxdepth 1 -name "*_hourly.nc" | wc -l | tr -d ' ')
UNIQUE_N=$(find "$CANONICAL_DIR" -maxdepth 1 -name "*_hourly.nc" \
    | sed 's|.*/||; s|_hourly\.nc$||' | sort -u | wc -l | tr -d ' ')
echo "  flat NC count:  $NC_FLAT  (expected $EXPECTED_NC_COUNT)"
echo "  unique STAIDs:  $UNIQUE_N  (expected $EXPECTED_NC_COUNT)"
if [ "$NC_FLAT" -ne "$EXPECTED_NC_COUNT" ]; then
    echo "  ERROR: flat NC count $NC_FLAT != $EXPECTED_NC_COUNT" >&2; exit 1
fi
if [ "$UNIQUE_N" -ne "$EXPECTED_NC_COUNT" ]; then
    echo "  ERROR: unique STAID count $UNIQUE_N != $EXPECTED_NC_COUNT" >&2; exit 1
fi
echo "  canonical_merged: OK"

echo ""
echo "[preflight] Output directory ..."
if [ -e "$OUT_DIR" ]; then
    if [ "$FORCE" -eq 0 ]; then
        echo "  ERROR: $OUT_DIR already exists. Pass --force to overwrite." >&2; exit 1
    fi
    echo "  Exists -- --force set, will overwrite."
else
    echo "  Absent: OK"
fi

echo ""
echo "[preflight] Python: $(python --version 2>&1)"

mkdir -p "$LOG_DIR"
BUILD_LOG="$LOG_DIR/build.log"
AUDIT_LOG="$LOG_DIR/audit.log"
echo "[preflight] Log dir: $LOG_DIR"
echo "[preflight] Preflight PASS"

# -- Build -----------------------------------------------------------------
echo ""
echo "======================================================================"
echo "STEP 1: Build"
echo "  exclude-staids:   $EXCLUDE_STAIDS"
echo "  expected basins:  $EXPECTED_BASINS"
echo "  output:           $OUT_DIR"
echo "  log:              $BUILD_LOG"
echo "======================================================================"

set +e
python scripts/build_stage1_target_package.py \
    --canonical-dir  "$CANONICAL_DIR" \
    --policy         "$POLICY" \
    --status-csv     "$STATUS_CSV" \
    --out-dir        "$OUT_DIR" \
    --exclude-staids "$EXCLUDE_STAIDS" \
    --force 2>&1 | tee "$BUILD_LOG"
BUILD_EXIT=${PIPESTATUS[0]}
set -e

echo "BUILD EXIT CODE: $BUILD_EXIT" | tee -a "$BUILD_LOG"
if [ "$BUILD_EXIT" -ne 0 ]; then
    echo "ERROR: Build failed (exit $BUILD_EXIT). See: $BUILD_LOG" >&2
    exit "$BUILD_EXIT"
fi

# -- Audit -----------------------------------------------------------------
echo ""
echo "======================================================================"
echo "STEP 2: Audit"
echo "  package:          $OUT_DIR"
echo "  expected basins:  $EXPECTED_BASINS"
echo "  log:              $AUDIT_LOG"
echo "======================================================================"

set +e
python scripts/audit_stage1_target_package.py \
    --package-dir     "$OUT_DIR" \
    --policy          "$POLICY" \
    --status-csv      "$STATUS_CSV" \
    --expected-basins "$EXPECTED_BASINS" 2>&1 | tee "$AUDIT_LOG"
AUDIT_EXIT=${PIPESTATUS[0]}
set -e

echo "AUDIT EXIT CODE: $AUDIT_EXIT" | tee -a "$AUDIT_LOG"

# -- Final report ----------------------------------------------------------
NC_OUT=$(find "$OUT_DIR/time_series" -name "*.nc" 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "======================================================================"
echo "Final Report"
echo "======================================================================"
echo "  Output dir:      $OUT_DIR"
echo "  Log dir:         $LOG_DIR"
echo "  NC count:        $NC_OUT"
echo ""
echo "--- Build log (last 20 lines) ---"
tail -20 "$BUILD_LOG"
echo ""
echo "--- Audit log (last 20 lines) ---"
tail -20 "$AUDIT_LOG"
echo ""
echo "NOTE: Generated outputs -- do NOT stage or commit:"
echo "      $OUT_DIR"
echo "      $LOG_DIR"

if [ "$AUDIT_EXIT" -ne 0 ]; then
    echo "" >&2
    echo "ERROR: Audit failed (exit $AUDIT_EXIT). See: $AUDIT_LOG" >&2
    exit "$AUDIT_EXIT"
fi

echo ""
echo "======================================================================"
echo "Build + Audit: ALL PASS"
echo "======================================================================"

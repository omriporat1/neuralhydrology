#!/usr/bin/env bash
# Stage 1 Forcing — ONE-MONTH launcher for h2o (Milestone 2K-C)
#
# PURPOSE
# -------
# Runs a single monthly chunk of the Stage 1 forcing extraction pipeline.
# Default target: October 2020 — the first production month.
#
# Run this cautious one-month launch and review evidence before starting
# the full 63-month run_stage1_forcing_fullperiod_h2o.sh loop.
#
# USAGE
# -----
# Dry run (print configuration, verify inputs, no downloads):
#   DRY_RUN=1 bash scripts/run_stage1_forcing_onemonth_h2o.sh
#
# Real run under screen (recommended):
#   screen -S flashnh-202010 bash scripts/run_stage1_forcing_onemonth_h2o.sh
#
# Resume an interrupted run:
#   RESUME=1 bash scripts/run_stage1_forcing_onemonth_h2o.sh
#
# Different month:
#   MONTH_LABEL=2020-11 START=2020-11-01T00:00:00 END=2020-11-30T23:00:00 \
#     bash scripts/run_stage1_forcing_onemonth_h2o.sh
#
# MRMS only (skip RTMA):
#   PRODUCTS=mrms bash scripts/run_stage1_forcing_onemonth_h2o.sh
#
# ENVIRONMENT VARIABLES
# ---------------------
# FLASHNH_ROOT       h2o Flash-NH base dir       (default: /data42/omrip/Flash-NH)
# MONTH_LABEL        Chunk label e.g. 2020-10    (default: 2020-10)
# START              ISO 8601 start datetime     (default: 2020-10-14T00:00:00)
# END                ISO 8601 end datetime       (default: 2020-10-31T23:00:00)
# DRY_RUN            1 = print config, no run   (default: 0)
# RESUME             1 = skip completed hours   (default: 0)
# DOWNLOAD_WORKERS   Parallel S3 workers         (default: 16)
# RTMA_MODE          selected_messages|full_file (default: selected_messages)
# PRODUCTS           Comma-separated: mrms,rtma  (default: mrms,rtma)
#
# EXPECTED OUTPUTS (October 2020, all 2,752 v001 basins)
# -------------------------------------------------------
# Total scheduled hours:  432  (2020-10-14T00Z → 2020-10-31T23Z)
# MRMS not_in_s3:          21  (hours 00Z–20Z on 2020-10-14; permanent S3 gap)
# MRMS extracted hours:   411  → rows: 411 × 2,752 = 1,130,472
# RTMA extracted hours:   432  → rows: 432 × 2,752 × ~10 vars ≈ 11,888,640
# Chunk Parquet:  chunks/2020-10/combined_2020-10.parquet
# Manifest:       manifests/2020-10_manifest.json  (all_pass=true if OK)
#
# EVIDENCE EXPORT (after run completes)
# -------------------------------------
# Pull compact evidence bundle locally — see printed instructions at end of run.
# Do NOT transfer raw GRIB2 or chunk Parquets.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (env-var overridable)
# ---------------------------------------------------------------------------

FLASHNH_ROOT="${FLASHNH_ROOT:-/data42/omrip/Flash-NH}"
MONTH_LABEL="${MONTH_LABEL:-2020-10}"
START="${START:-2020-10-14T00:00:00}"
END="${END:-2020-10-31T23:00:00}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-16}"
RTMA_MODE="${RTMA_MODE:-selected_messages}"
PRODUCTS="${PRODUCTS:-mrms,rtma}"

# Derived paths (match fullperiod launcher)
FORCING_ROOT="${FLASHNH_ROOT}/tmp/stage1_forcing_fullperiod"
BASIN_LIST="${FORCING_ROOT}/v001_basin_list.csv"
MRMS_WEIGHTS="${FORCING_ROOT}/02_basin_geometries/weights/mrms/v001_2752_mrms_weights.parquet"
RTMA_WEIGHTS="${FORCING_ROOT}/02_basin_geometries/weights/rtma/v001_2752_rtma_weights.parquet"
MANIFEST_DIR="${FORCING_ROOT}/manifests"
PROV_DIR="${MANIFEST_DIR}/run_provenance"
ENV_PREFIX="${FLASHNH_ROOT}/envs/flashnh-stage1"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

CHUNK_MANIFEST="${MANIFEST_DIR}/${MONTH_LABEL}_manifest.json"
CHUNK_PARQUET="${FORCING_ROOT}/chunks/${MONTH_LABEL}/combined_${MONTH_LABEL}.parquet"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

echo "============================================================"
if [ "${DRY_RUN}" = "1" ]; then
    echo "Flash-NH Stage 1 Forcing — ONE-MONTH DRY RUN"
else
    echo "Flash-NH Stage 1 Forcing — ONE-MONTH LAUNCH"
fi
echo "============================================================"
echo "Month:     ${MONTH_LABEL}  (${START} → ${END})"
echo "Workers:   ${DOWNLOAD_WORKERS}"
echo "RTMA mode: ${RTMA_MODE}"
echo "Products:  ${PRODUCTS}"
echo "Resume:    ${RESUME}"
echo "Output:    ${FORCING_ROOT}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Conda env activation (same pattern as smoke and fullperiod launchers)
# ---------------------------------------------------------------------------

# Source conda shell integration unconditionally — 'conda activate' requires the
# shell function registered by conda.sh, not just the conda binary being in PATH.
# Non-interactive shells (bash script.sh) do not source ~/.bashrc.
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source /opt/conda/etc/profile.d/conda.sh
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

# Attempt conda activate — non-fatal. PATH-prepend is authoritative.
conda activate "${ENV_PREFIX}" 2>/dev/null || true
export PATH="${ENV_PREFIX}/bin:${PATH}"
hash -r

if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
    echo "ERROR: ${ENV_PREFIX}/bin/python not found or not executable."
    echo "       Ensure the flashnh-stage1 conda env is installed at ${ENV_PREFIX}."
    exit 1
fi
_actual_python=$(command -v python)
if [ "${_actual_python}" != "${ENV_PREFIX}/bin/python" ]; then
    echo "ERROR: python resolves to ${_actual_python}"
    echo "       expected ${ENV_PREFIX}/bin/python"
    exit 1
fi
_py_ver=$(python --version 2>&1)
if [[ "${_py_ver}" != *"Python 3.11"* ]]; then
    echo "ERROR: Expected Python 3.11.x, got: ${_py_ver}"
    exit 1
fi
echo "Python: ${_actual_python} (${_py_ver})"

# ---------------------------------------------------------------------------
# Input file checks
# ---------------------------------------------------------------------------

for _fpath in "${BASIN_LIST}" "${MRMS_WEIGHTS}" "${RTMA_WEIGHTS}"; do
    if [ ! -f "${_fpath}" ]; then
        echo "ERROR: Required file missing: ${_fpath}"
        echo "  Complete Milestone 2K-A (weight build) and 2K-B (smoke test) before launch."
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Derived metadata (computed with Python for reliability)
# ---------------------------------------------------------------------------

HOUR_COUNT=$(python -c "
from datetime import datetime
s = datetime.fromisoformat('${START}')
e = datetime.fromisoformat('${END}')
print(int((e - s).total_seconds() / 3600) + 1)
" 2>/dev/null || echo "432")

BASIN_COUNT=$(python -c "
with open('${BASIN_LIST}') as f:
    print(sum(1 for _ in f) - 1)
" 2>/dev/null || echo "2752")

GIT_COMMIT=$(git -C "${REPO_ROOT}" log --oneline -1 2>/dev/null || echo "unknown")
GIT_STATUS=$(git -C "${REPO_ROOT}" status --short 2>/dev/null | head -5 || echo "")

# ---------------------------------------------------------------------------
# DRY RUN — print configuration and exit without downloading
# ---------------------------------------------------------------------------

if [ "${DRY_RUN}" = "1" ]; then
    echo ""
    echo "--- DRY RUN CONFIGURATION ---"
    echo "Git commit:    ${GIT_COMMIT}"
    if [ -n "${GIT_STATUS}" ]; then
        echo "Git status:"
        echo "${GIT_STATUS}" | sed 's/^/  /'
    else
        echo "Git status:    (clean)"
    fi
    echo ""
    echo "Inputs:"
    echo "  Basin list:    ${BASIN_LIST}  (${BASIN_COUNT} basins)"
    echo "  MRMS weights:  ${MRMS_WEIGHTS}"
    echo "  RTMA weights:  ${RTMA_WEIGHTS}"
    echo ""
    echo "Period:"
    echo "  Start:         ${START}"
    echo "  End:           ${END}"
    echo "  Hours:         ${HOUR_COUNT} scheduled"
    if [ "${MONTH_LABEL}" = "2020-10" ]; then
        echo "  MRMS gap:      21 h not_in_s3 (00Z–20Z on 2020-10-14; permanent S3 gap)"
    fi
    echo ""
    echo "Expected row counts:"
    python - <<PYEOF
n_h = ${HOUR_COUNT}
n_b = ${BASIN_COUNT}
mrms_gap = 21 if "${MONTH_LABEL}" == "2020-10" else 0
mrms_h = n_h - mrms_gap
print(f"  MRMS rows:  {mrms_h:,} h × {n_b:,} basins = {mrms_h * n_b:,}")
print(f"  RTMA rows:  {n_h:,} h × {n_b:,} basins × ~10 vars ≈ {n_h * n_b * 10:,}")
print(f"  Combined:   ≈ {mrms_h * n_b + n_h * n_b * 10:,} rows")
PYEOF
    echo ""
    echo "Planned outputs:"
    echo "  Chunk Parquet: ${CHUNK_PARQUET}"
    echo "  Manifest:      ${CHUNK_MANIFEST}"
    echo "  Log:           ${MANIFEST_DIR}/${MONTH_LABEL}_live_run.log"
    echo "  Provenance:    ${PROV_DIR}/${MONTH_LABEL}_run_provenance.json"
    echo ""
    echo "Extractor command:"
    echo ""
    _RESUME_FLAG=""
    [ "${RESUME}" = "1" ] && _RESUME_FLAG=" --resume"
    echo "  python scripts/extract_stage1_forcing_chunk.py \\"
    echo "      --start            ${START} \\"
    echo "      --end              ${END} \\"
    echo "      --basin-manifest   ${BASIN_LIST} \\"
    echo "      --mrms-weights     ${MRMS_WEIGHTS} \\"
    echo "      --rtma-weights     ${RTMA_WEIGHTS} \\"
    echo "      --out-dir          ${FORCING_ROOT} \\"
    echo "      --chunk-label      ${MONTH_LABEL} \\"
    echo "      --download-workers ${DOWNLOAD_WORKERS} \\"
    echo "      --rtma-mode        ${RTMA_MODE} \\"
    echo "      --products         ${PRODUCTS}${_RESUME_FLAG} \\"
    echo "      --no-plots"
    echo ""
    echo "Disk:"
    df -h "${FORCING_ROOT}" 2>/dev/null | tail -1 || df -h "/" 2>/dev/null | tail -1
    echo ""
    echo "System load:"
    uptime 2>/dev/null
    echo ""
    echo "DRY RUN COMPLETE — no files written. Unset DRY_RUN=1 to launch."
    exit 0
fi

# ---------------------------------------------------------------------------
# Real run: create dirs and write pre-launch provenance
# ---------------------------------------------------------------------------

mkdir -p "${MANIFEST_DIR}" "${PROV_DIR}"
mkdir -p "${FORCING_ROOT}/raw/mrms" "${FORCING_ROOT}/raw/rtma"
mkdir -p "${FORCING_ROOT}/staging/mrms" "${FORCING_ROOT}/staging/rtma"

LOAD=$(uptime | awk -F'load average:' '{ print $2 }' | awk '{ print $1 }' | tr -d ',')
echo "System load (1-min): ${LOAD}"
echo "NOTE: h2o etiquette — keep total CPU ≤50-60%."

# Write run provenance JSON
python - <<PYEOF
import json
from datetime import datetime, timezone
from pathlib import Path

prov = {
    "month_label":       "${MONTH_LABEL}",
    "start":             "${START}",
    "end":               "${END}",
    "hour_count":        ${HOUR_COUNT},
    "basin_list":        "${BASIN_LIST}",
    "basin_count":       ${BASIN_COUNT},
    "mrms_weights":      "${MRMS_WEIGHTS}",
    "rtma_weights":      "${RTMA_WEIGHTS}",
    "download_workers":  ${DOWNLOAD_WORKERS},
    "rtma_mode":         "${RTMA_MODE}",
    "products":          "${PRODUCTS}".split(","),
    "resume":            "${RESUME}" == "1",
    "launched_utc":      datetime.now(timezone.utc).isoformat(),
    "git_commit":        "${GIT_COMMIT}",
    "python_executable": "${_actual_python}",
    "python_version":    "${_py_ver}",
}
out = Path("${PROV_DIR}/${MONTH_LABEL}_run_provenance.json")
out.write_text(json.dumps(prov, indent=2))
print(f"Provenance:  {out}")
PYEOF

# Write run command text file
cat > "${PROV_DIR}/${MONTH_LABEL}_run_command.txt" <<CMDEOF
# Flash-NH Stage 1 Forcing — one-month run command
# Month: ${MONTH_LABEL}
# Git:   ${GIT_COMMIT}
# UTC:   $(date -u +'%Y-%m-%dT%H:%M:%SZ')

cd ${REPO_ROOT} && \\
python scripts/extract_stage1_forcing_chunk.py \\
    --start            ${START} \\
    --end              ${END} \\
    --basin-manifest   ${BASIN_LIST} \\
    --mrms-weights     ${MRMS_WEIGHTS} \\
    --rtma-weights     ${RTMA_WEIGHTS} \\
    --out-dir          ${FORCING_ROOT} \\
    --chunk-label      ${MONTH_LABEL} \\
    --download-workers ${DOWNLOAD_WORKERS} \\
    --rtma-mode        ${RTMA_MODE} \\
    --products         ${PRODUCTS} \\
    --no-plots
CMDEOF
echo "Run command: ${PROV_DIR}/${MONTH_LABEL}_run_command.txt"

# Write environment snapshot
{
    echo "python: ${_actual_python}"
    echo "version: ${_py_ver}"
    echo "conda_env: ${ENV_PREFIX}"
    echo "---"
    python -c "
import importlib.metadata as _m
pkgs = sorted(_m.packages_distributions().items())
for name, _ in pkgs:
    try:
        v = _m.version(name)
        print(f'{name}=={v}')
    except Exception:
        pass
" 2>/dev/null || python -c "
import pkg_resources
for d in sorted(pkg_resources.working_set, key=lambda x: x.project_name.lower()):
    print(f'{d.project_name}=={d.version}')
" 2>/dev/null || true
} > "${PROV_DIR}/${MONTH_LABEL}_environment.txt"
echo "Environment: ${PROV_DIR}/${MONTH_LABEL}_environment.txt"

# ---------------------------------------------------------------------------
# Run extraction
# ---------------------------------------------------------------------------

_RESUME_ARGS=()
[ "${RESUME}" = "1" ] && _RESUME_ARGS=(--resume)

echo ""
echo "Launching extraction: ${MONTH_LABEL}  (${START} → ${END})"
echo "  ${HOUR_COUNT} hours × ${BASIN_COUNT} basins  workers=${DOWNLOAD_WORKERS}  mode=${RTMA_MODE}"
echo ""

cd "${REPO_ROOT}"

EXIT_CODE=0
python scripts/extract_stage1_forcing_chunk.py \
    --start            "${START}" \
    --end              "${END}" \
    --basin-manifest   "${BASIN_LIST}" \
    --mrms-weights     "${MRMS_WEIGHTS}" \
    --rtma-weights     "${RTMA_WEIGHTS}" \
    --out-dir          "${FORCING_ROOT}" \
    --chunk-label      "${MONTH_LABEL}" \
    --download-workers "${DOWNLOAD_WORKERS}" \
    --rtma-mode        "${RTMA_MODE}" \
    --products         "${PRODUCTS}" \
    "${_RESUME_ARGS[@]}" \
    --no-plots || EXIT_CODE=$?

# ---------------------------------------------------------------------------
# Post-run: manifest summary, scaling estimates, validation checks
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "EXTRACTION: PASS (exit 0)"
else
    echo "EXTRACTION: FAIL (exit ${EXIT_CODE})"
fi
echo "============================================================"
echo ""

python - <<PYEOF
import csv, json, statistics
from datetime import datetime, timezone
from pathlib import Path

LABEL   = "${MONTH_LABEL}"
MDIR    = Path("${MANIFEST_DIR}")
N_HOURS = ${HOUR_COUNT}
N_BASINS = ${BASIN_COUNT}

manifest_path = MDIR / f"{LABEL}_manifest.json"
runtime_path  = MDIR / f"{LABEL}_hourly_runtime_and_volume.csv"
missing_path  = MDIR / f"{LABEL}_missing_files.csv"

m = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
all_pass = m.get("all_pass", None)

# Count outcomes from runtime CSV
n_mrms_ok = n_rtma_ok = 0
mrms_dl_times = []; rtma_dl_times = []
if runtime_path.exists():
    for row in csv.DictReader(runtime_path.open()):
        prod   = row.get("product", "")
        status = row.get("status", "")
        dt_s   = row.get("download_time_s") or ""
        is_mrms = "mrms" in prod.lower()
        if status == "success":
            if is_mrms:
                n_mrms_ok += 1
                if dt_s.strip():
                    mrms_dl_times.append(float(dt_s))
            else:
                n_rtma_ok += 1
                if dt_s.strip():
                    rtma_dl_times.append(float(dt_s))

n_missing_s3 = 0
if missing_path.exists():
    n_missing_s3 = sum(1 for r in csv.DictReader(missing_path.open())
                       if r.get("reason") == "not_in_s3"
                       and "mrms" in r.get("product", "").lower())

# Scaling estimates
estimates = {
    "chunk_label":           LABEL,
    "hour_count_scheduled":  N_HOURS,
    "basin_count":           N_BASINS,
    "n_mrms_ok":             n_mrms_ok,
    "n_rtma_ok":             n_rtma_ok,
    "n_missing_s3":          n_missing_s3,
    "all_pass":              all_pass,
    "full_period_hours":     45720,
    "generated_utc":         datetime.now(timezone.utc).isoformat(),
}
if rtma_dl_times:
    med = statistics.median(rtma_dl_times)
    estimates["rtma_median_dl_s"]             = round(med, 2)
    estimates["rtma_full_45720h_estimate_h"]  = round(45720 * med / 3600, 1)
    estimates["rtma_full_45720h_estimate_d"]  = round(45720 * med / 86400, 1)
if mrms_dl_times:
    estimates["mrms_median_dl_s"]             = round(statistics.median(mrms_dl_times), 2)

est_path = MDIR / f"{LABEL}_scaling_estimates.json"
est_path.write_text(json.dumps(estimates, indent=2))

# Validation checks
expected_gap = 21 if LABEL == "2020-10" else 0
checks = [
    {"check": "mrms_hours_ok",     "result": "PASS" if n_mrms_ok > 0       else "FAIL",
     "value": n_mrms_ok,   "threshold": ">0",                  "note": "MRMS extracted hours"},
    {"check": "rtma_hours_ok",     "result": "PASS" if n_rtma_ok > 0       else "FAIL",
     "value": n_rtma_ok,   "threshold": ">0",                  "note": "RTMA extracted hours"},
    {"check": "manifest_all_pass", "result": "PASS" if all_pass is True     else "FAIL",
     "value": all_pass,    "threshold": "true",                "note": "extractor all_pass flag"},
]
if expected_gap > 0:
    gap_ok = n_missing_s3 == expected_gap
    checks.append({
        "check": "mrms_202010_gap", "result": "PASS" if gap_ok else "WARN",
        "value": n_missing_s3, "threshold": f"=={expected_gap}",
        "note": "known MRMS S3 gap hours 00Z-20Z on 2020-10-14"
    })

chk_path = MDIR / f"{LABEL}_validation_checks.csv"
with chk_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["check","result","value","threshold","note"])
    w.writeheader()
    w.writerows(checks)

# Print summary
print(f"Manifest all_pass:  {all_pass}")
print(f"MRMS ok hours:      {n_mrms_ok}/{N_HOURS}")
print(f"RTMA ok hours:      {n_rtma_ok}/{N_HOURS}")
print(f"Missing (not_in_s3): {n_missing_s3}  ({'expected' if n_missing_s3 == expected_gap else 'unexpected'})")
if "rtma_full_45720h_estimate_h" in estimates:
    print(f"Scale estimate:     RTMA full 45,720h ≈ {estimates['rtma_full_45720h_estimate_h']}h"
          f" / {estimates['rtma_full_45720h_estimate_d']}d at {len(rtma_dl_times)} worker-median {estimates['rtma_median_dl_s']}s/file")
print(f"")
n_fail = sum(1 for c in checks if c["result"] == "FAIL")
n_pass = sum(1 for c in checks if c["result"] == "PASS")
print(f"Validation: {n_pass}/{len(checks)} PASS  {n_fail} FAIL")
for c in checks:
    print(f"  {c['result']:5s}  {c['check']}={c['value']}  (threshold:{c['threshold']})  {c['note']}")
print(f"")
print(f"Scaling estimates:  {est_path}")
print(f"Validation checks:  {chk_path}")
PYEOF

# ---------------------------------------------------------------------------
# Evidence export instructions
# ---------------------------------------------------------------------------

_TARBALL_LABEL="${MONTH_LABEL//-/}"
_EXPORT_DIR="${FORCING_ROOT}/evidence_exports"
_TARBALL="${_EXPORT_DIR}/stage1_forcing_${_TARBALL_LABEL}_v001_audit_export.tar.gz"

echo ""
echo "============================================================"
echo "EVIDENCE EXPORT"
echo "============================================================"
echo "Run these commands on h2o to create a compact audit bundle:"
echo ""
echo "  mkdir -p ${_EXPORT_DIR}"
echo "  cd ${MANIFEST_DIR} && \\"
echo "  tar czf ${_TARBALL} \\"
echo "      ${MONTH_LABEL}_manifest.json \\"
echo "      ${MONTH_LABEL}_summary.json \\"
echo "      ${MONTH_LABEL}_summary.md \\"
echo "      ${MONTH_LABEL}_hourly_runtime_and_volume.csv \\"
echo "      ${MONTH_LABEL}_scaling_estimates.json \\"
echo "      ${MONTH_LABEL}_validation_checks.csv \\"
echo "      ${MONTH_LABEL}_live_run.log \\"
echo "      \$(ls ${MONTH_LABEL}_missing_files.csv 2>/dev/null) \\"
echo "      \$(ls ${MONTH_LABEL}_variable_completeness.csv 2>/dev/null) \\"
echo "      \$(ls ${MONTH_LABEL}_basin_completeness.csv 2>/dev/null) \\"
echo "      run_provenance/${MONTH_LABEL}_run_provenance.json \\"
echo "      run_provenance/${MONTH_LABEL}_run_command.txt"
echo ""
echo "  # Transfer to local:"
echo "  mkdir -p tmp/stage1_forcing_${_TARBALL_LABEL}_evidence"
echo "  scp flashnh-h2o:${_TARBALL} tmp/stage1_forcing_${_TARBALL_LABEL}_evidence/"
echo ""
echo "  # Or transfer individual files:"
echo "  scp flashnh-h2o:${MANIFEST_DIR}/${MONTH_LABEL}_manifest.json        tmp/..."
echo "  scp flashnh-h2o:${MANIFEST_DIR}/${MONTH_LABEL}_summary.md           tmp/..."
echo "  scp flashnh-h2o:${MANIFEST_DIR}/${MONTH_LABEL}_scaling_estimates.json tmp/..."
echo ""
echo "Monitoring (while running or to inspect log):"
echo "  screen -r flashnh-${_TARBALL_LABEL}"
echo "  tail -f ${MANIFEST_DIR}/${MONTH_LABEL}_live_run.log"
echo "  uptime && df -h ${FORCING_ROOT} | tail -1"
echo ""
echo "Log: ${MANIFEST_DIR}/${MONTH_LABEL}_live_run.log"

exit "${EXIT_CODE}"

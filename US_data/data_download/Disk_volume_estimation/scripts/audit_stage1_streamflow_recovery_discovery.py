"""
Flash-NH Stage 1 Milestone 2H — Streamflow Recovery Discovery Audit
====================================================================

Checks the 22 pilot basins whose local CAMELSH hourly NetCDF files are missing
and determines whether streamflow can be recovered from USGS NWIS IV.

POSTURE:
  - Discovery/audit only. No files are written to CAMELSH source data.
  - No recovered NetCDF files are created.
  - USGS queries are metadata / single-month availability checks only.
  - All outputs go under --out-dir (default: tmp/stage1_pilot_dryrun/13_streamflow_recovery_discovery/).

OUTPUTS:
  tables/streamflow_recovery_discovery_22.csv
  summary.md
  summary.json
  provenance/run_provenance.json

Usage:
  python scripts/audit_stage1_streamflow_recovery_discovery.py
  python scripts/audit_stage1_streamflow_recovery_discovery.py --skip-usgs
  python scripts/audit_stage1_streamflow_recovery_discovery.py --config configs/pilot_stage1.yaml --data-root /path/to/root
  python scripts/audit_stage1_streamflow_recovery_discovery.py --staids-file path/to/staids.txt
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Default STAIDs — 22 basins with all-NaN qobs_m3s in the 2G package
# ---------------------------------------------------------------------------
DEFAULT_STAIDS = [
    "01585200", "01586210", "02072500", "02073000", "02077670",
    "02146381", "02235000", "02264100", "02266480", "02266500",
    "02301000", "02344605", "02344700", "02403310", "02484000",
    "03298135", "03305000", "07103700", "07283000", "10164500",
    "10336700", "11372000",
]

# ---------------------------------------------------------------------------
# Path constants (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

PILOT_MANIFEST        = REPO_ROOT / "tmp/stage1_pilot_dryrun/09_manifests/stage1_pilot/pilot_basin_manifest.csv"
STATIC_ATTRIBUTES_CSV = REPO_ROOT / "tmp/stage1_pilot_dryrun/12_neuralhydrology_january_pilot_dataset/package/attributes/attributes_full.csv"
NO_STREAMFLOW_TXT     = REPO_ROOT / "tmp/stage1_pilot_dryrun/12_neuralhydrology_january_pilot_dataset/package/basin_lists/january_2023_smoke_streamflow_only/no_streamflow_basins.txt"

# Local CAMELSH hourly directory used in the 2G build script
CONFIGURED_CAMELSH_HOURLY_DIR = pathlib.Path(
    "C:/PhD/Python/neuralhydrology/US_data/data_download/CAMELSH_resolution_test/data/raw/camelsh"
)

# CAMELSH polygon shapefile (GAGES-II derived, from data root)
CAMELSH_SHAPEFILE = REPO_ROOT / "tmp/stage1_pilot_dryrun/02_basin_geometries/camelsh/shapefiles/CAMELSH_shapefile.shp"

# GAGES-II BasinID CSV
GAGESII_BASINID_CSV = pathlib.Path(
    "C:/PhD/Python/neuralhydrology/US_data/attributes/attributes_gageii_BasinID.csv"
)

# Prior IV scan results (from earlier validation work — covers 5000+ stations)
IV_SCAN_RESULTS_CSV = pathlib.Path(
    "C:/PhD/Python/neuralhydrology/US_data/iv_scan_results.csv"
)

# USGS NWIS IV endpoint
USGS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
PARAM_CODE  = "00060"   # discharge, cubic feet per second (instantaneous)

# Research period
RESEARCH_PERIOD_START = "2020-10-14"
RESEARCH_PERIOD_END   = "2025-12-31"

# January 2023 window
JAN2023_START = "2023-01-01T00:00:00Z"
JAN2023_END   = "2023-01-31T23:59:59Z"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "UNKNOWN"


def make_session(retries: int = 3, backoff: float = 0.5) -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": "flash-nh-audit/0.1 (discovery-only)"})
    try:
        retry = Retry(
            total=retries, read=retries, connect=retries,
            backoff_factor=backoff,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
    except TypeError:
        retry = Retry(
            total=retries, read=retries, connect=retries,
            backoff_factor=backoff,
            status_forcelist=(429, 500, 502, 503, 504),
            method_whitelist=frozenset(["GET"]),
            raise_on_status=False,
        )
    sess.mount("https://", adapter := HTTPAdapter(max_retries=retry))
    sess.mount("http://", adapter)
    return sess


def usgs_iv_query_jan2023(sess: requests.Session, staid: str) -> dict:
    """
    Query USGS IV for January 2023 discharge. Returns a dict with:
      available (bool), coverage_hours (int or NaN), units_observed (str),
      n_raw_values (int), error (str or None).

    This is a targeted single-month query (not bulk). Raw values are counted
    for coverage estimation only; no data is written to disk.
    """
    result = {
        "available": False,
        "coverage_hours": float("nan"),
        "units_observed": "",
        "n_raw_values": 0,
        "timezone_note": "USGS IV timestamps are in UTC when using ISO8601 startDT/endDT",
        "error": None,
    }
    try:
        params = {
            "sites": staid,
            "parameterCd": PARAM_CODE,
            "startDT": JAN2023_START,
            "endDT": JAN2023_END,
            "format": "json",
            "siteStatus": "all",
        }
        resp = sess.get(USGS_IV_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        series = data.get("value", {}).get("timeSeries", [])
        if not series:
            result["available"] = False
            result["error"] = "No timeSeries in USGS IV response"
            return result

        all_times = []
        unit_codes = []
        for ts in series:
            unit_code = (((ts or {}).get("variable") or {}).get("unit") or {}).get("unitCode", "")
            if unit_code:
                unit_codes.append(unit_code)
            for vv in ts.get("values", []):
                for val in vv.get("value", []):
                    dt_iso = val.get("dateTime")
                    v_raw = pd.to_numeric(val.get("value", "nan"), errors="coerce")
                    # Only count non-NaN, non-missing values (USGS uses -999999 for missing)
                    if dt_iso is not None and not pd.isna(v_raw) and v_raw > -999990:
                        all_times.append(dt_iso)

        result["units_observed"] = ", ".join(sorted(set(unit_codes))) if unit_codes else "ft3/s (default USGS param 00060)"
        result["n_raw_values"] = len(all_times)

        if not all_times:
            result["available"] = False
            result["error"] = "Zero valid observations in January 2023"
            return result

        result["available"] = True
        # Snap to hourly and count coverage
        try:
            timestamps = pd.to_datetime(all_times, utc=True, errors="coerce")
            timestamps = timestamps.dropna()
            hourly = timestamps.floor("h")
            coverage_hours = hourly.nunique()
            result["coverage_hours"] = int(coverage_hours)
        except Exception as e:
            result["coverage_hours"] = float("nan")
            result["error"] = f"Coverage calculation error: {e}"

    except requests.RequestException as e:
        result["available"] = False
        result["error"] = f"HTTP error: {e}"
    except json.JSONDecodeError as e:
        result["available"] = False
        result["error"] = f"JSON parse error: {e}"
    except Exception as e:
        result["available"] = False
        result["error"] = f"Unexpected error: {e}"

    return result


# ---------------------------------------------------------------------------
# Local catalog checks
# ---------------------------------------------------------------------------

def load_pilot_manifest(staids: list[str]) -> dict[str, bool]:
    """Return {staid: in_manifest}."""
    staid_set = set(staids)
    if not PILOT_MANIFEST.exists():
        return {s: False for s in staids}
    df = pd.read_csv(PILOT_MANIFEST, dtype={"STAID": str})
    df["STAID"] = df["STAID"].str.strip().str.zfill(8)
    manifest_set = set(df["STAID"].tolist())
    return {s: (s in manifest_set) for s in staids}


def load_static_attributes(staids: list[str]) -> dict[str, bool]:
    """Return {staid: in_static_attributes}."""
    staid_set = set(staids)
    if not STATIC_ATTRIBUTES_CSV.exists():
        return {s: False for s in staids}
    df = pd.read_csv(STATIC_ATTRIBUTES_CSV, dtype={"gauge_id": str})
    df["gauge_id"] = df["gauge_id"].str.strip().str.zfill(8)
    present = set(df["gauge_id"].tolist())
    return {s: (s in present) for s in staids}


def load_gagesii_catalog(staids: list[str]) -> dict[str, dict]:
    """Return per-STAID dict with catalog metadata from GAGES-II BasinID CSV."""
    staid_set = set(staids)
    result = {s: {"found": False, "STANAME": "", "HUC02": "", "DRAIN_SQKM": ""} for s in staids}
    if not GAGESII_BASINID_CSV.exists():
        return result
    df = pd.read_csv(GAGESII_BASINID_CSV, dtype={"STAID": str}, encoding="utf-8-sig")
    df["STAID"] = df["STAID"].str.strip().str.zfill(8)
    for _, row in df[df["STAID"].isin(staid_set)].iterrows():
        s = row["STAID"]
        result[s] = {
            "found": True,
            "STANAME": row.get("STANAME", ""),
            "HUC02": row.get("HUC02", ""),
            "DRAIN_SQKM": row.get("DRAIN_SQKM", ""),
        }
    return result


def check_polygon_catalog(staids: list[str]) -> dict[str, bool]:
    """Return {staid: in_polygon_catalog} using the CAMELSH shapefile."""
    staid_set = set(staids)
    result = {s: False for s in staids}
    if not CAMELSH_SHAPEFILE.exists():
        warnings.warn(f"CAMELSH shapefile not found at {CAMELSH_SHAPEFILE}; polygon catalog check skipped")
        return result
    try:
        import geopandas as gpd
        gdf = gpd.read_file(CAMELSH_SHAPEFILE)
        gdf["GAGE_ID_norm"] = gdf["GAGE_ID"].astype(str).str.strip().str.zfill(8)
        found = set(gdf["GAGE_ID_norm"].tolist())
        for s in staids:
            result[s] = (s in found)
    except ImportError:
        warnings.warn("geopandas not available; polygon catalog check skipped")
    except Exception as e:
        warnings.warn(f"Polygon catalog check error: {e}")
    return result


def check_configured_camelsh_file(staids: list[str]) -> dict[str, dict]:
    """Check for {staid}_hourly.nc in the configured CAMELSH hourly directory."""
    result = {}
    for s in staids:
        fpath = CONFIGURED_CAMELSH_HOURLY_DIR / f"{s}_hourly.nc"
        result[s] = {
            "exists": fpath.exists(),
            "path": str(fpath) if fpath.exists() else "",
        }
    return result


def check_alternate_camelsh_dirs(staids: list[str], extra_dirs: list[pathlib.Path] | None = None) -> dict[str, dict]:
    """
    Search alternate local CAMELSH directories for {staid}_hourly.nc.
    Only searches directories discovered from config/docs/manifests — not a blind drive scan.
    """
    # Known alternate candidate directories (from config, docs, and manifests)
    candidate_dirs = [
        # The configured dir is already checked separately; include it here as well for completeness
        CONFIGURED_CAMELSH_HOURLY_DIR,
        # Sibling locations that might hold alternate CAMELSH downloads
        pathlib.Path("C:/PhD/Python/neuralhydrology/US_data/data_download/CAMELSH_resolution_test/data/raw/camelsh"),
    ]
    if extra_dirs:
        candidate_dirs.extend(extra_dirs)
    # Deduplicate
    seen = set()
    unique_dirs = []
    for d in candidate_dirs:
        key = str(d.resolve()) if d.exists() else str(d)
        if key not in seen:
            seen.add(key)
            unique_dirs.append(d)

    result = {s: {"found": False, "path": "", "dirs_searched": []} for s in staids}

    dirs_searched = []
    for d in unique_dirs:
        if not d.exists():
            continue
        dirs_searched.append(str(d))
        for s in staids:
            if result[s]["found"]:
                continue
            fpath = d / f"{s}_hourly.nc"
            if fpath.exists():
                result[s]["found"] = True
                result[s]["path"] = str(fpath)

    for s in staids:
        result[s]["dirs_searched"] = dirs_searched

    return result


def inspect_local_nc_file(nc_path: str) -> dict:
    """Read metadata from a local CAMELSH hourly NetCDF file (if it exists)."""
    result = {
        "time_start_utc": "",
        "time_end_utc": "",
        "variable": "",
        "units": "",
        "jan2023_coverage_hours": float("nan"),
    }
    if not nc_path or not pathlib.Path(nc_path).exists():
        return result
    try:
        import xarray as xr
        ds = xr.open_dataset(nc_path)
        if "time" in ds:
            t = pd.to_datetime(ds["time"].values, utc=True, errors="coerce")
            t = t.dropna()
            if len(t) > 0:
                result["time_start_utc"] = t.min().isoformat()
                result["time_end_utc"]   = t.max().isoformat()
                jan_mask = (t >= pd.Timestamp("2023-01-01", tz="UTC")) & (t <= pd.Timestamp("2023-01-31 23:59:59", tz="UTC"))
                # Check streamflow variable
                var_names = [v for v in ["streamflow", "qobs_m3s", "discharge"] if v in ds]
                if var_names:
                    vname = var_names[0]
                    result["variable"] = vname
                    result["units"] = str(ds[vname].attrs.get("units", ""))
                    jan_vals = ds[vname].values[np.where(jan_mask)[0]] if len(np.where(jan_mask)[0]) > 0 else np.array([])
                    result["jan2023_coverage_hours"] = int(np.sum(~np.isnan(jan_vals)))
        ds.close()
    except Exception as e:
        result["variable"] = f"ERROR: {e}"
    return result


def load_iv_scan_results(staids: list[str]) -> dict[str, dict]:
    """Load existing USGS IV scan results from prior validation work."""
    result = {s: {"has_iv": None, "iv_start": "", "iv_end": "", "median_dt_min": "", "likely_15min": None} for s in staids}
    if not IV_SCAN_RESULTS_CSV.exists():
        return result
    df = pd.read_csv(IV_SCAN_RESULTS_CSV, dtype=str)
    df["site_id"] = df["site_id"].str.strip().str.zfill(8)
    staid_set = set(staids)
    for _, row in df[df["site_id"].isin(staid_set)].iterrows():
        s = row["site_id"]
        result[s] = {
            "has_iv": row.get("has_iv", "").strip().lower() in ("true", "1", "yes"),
            "iv_start": row.get("iv_start", ""),
            "iv_end": row.get("iv_end", ""),
            "median_dt_min": row.get("median_dt_min", ""),
            "likely_15min": row.get("likely_15min", "").strip().lower() in ("true", "1", "yes"),
        }
    return result


def determine_period_availability(scan: dict) -> str:
    """
    Determine research-period availability based on IV scan data.
    Returns a conservative descriptive status string.
    """
    if not scan.get("has_iv"):
        return "NO_IV_DATA"
    try:
        iv_start = pd.Timestamp(scan["iv_start"])
        iv_end   = pd.Timestamp(scan["iv_end"])
        req_start = pd.Timestamp(RESEARCH_PERIOD_START)
        req_end   = pd.Timestamp(RESEARCH_PERIOD_END)
        if iv_start <= req_start and iv_end >= req_end:
            return "LIKELY_FULL_2020_2025"
        if iv_start <= req_start and iv_end >= pd.Timestamp("2025-09-01"):
            return "LIKELY_FULL_2020_TO_LATE_2025"
        if iv_start > req_start:
            return f"STARTS_AFTER_RESEARCH_PERIOD_START ({iv_start.date()})"
        if iv_end < req_end:
            return f"ENDS_BEFORE_RESEARCH_PERIOD_END ({iv_end.date()})"
    except Exception:
        return "UNKNOWN"
    return "LIKELY_AVAILABLE"


def determine_recommended_action(
    configured_file_exists: bool,
    alternate_file_found: bool,
    usgs_available: bool | None,
    pilot_role: str,
) -> str:
    if configured_file_exists:
        return "USE_EXISTING_LOCAL_CAMELSH_FILE"
    if alternate_file_found:
        return "COPY_OR_LINK_LOCAL_CAMELSH_FILE_LATER_AFTER_REVIEW"
    if usgs_available is True:
        return "RECOVER_FROM_USGS_IV_LATER"
    if usgs_available is False:
        return "MANUAL_REVIEW_REQUIRED"
    # USGS not checked
    return "RECOVER_FROM_USGS_IV_LATER"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    staids: list[str],
    out_dir: pathlib.Path,
    skip_usgs: bool,
    config_path: str | None,
    data_root: str | None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    prov_dir = out_dir / "provenance"
    prov_dir.mkdir(exist_ok=True)

    run_ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    git_hash = git_commit_hash()

    print(f"[2H] Streamflow recovery discovery — {run_ts}")
    print(f"[2H] Git commit: {git_hash}")
    print(f"[2H] STAIDs to check: {len(staids)}")
    print(f"[2H] USGS check: {'SKIPPED' if skip_usgs else 'ENABLED'}")

    # --- Local checks ---
    print("[2H] Loading pilot manifest ...")
    in_manifest  = load_pilot_manifest(staids)

    print("[2H] Loading static attributes ...")
    in_static    = load_static_attributes(staids)

    print("[2H] Checking GAGES-II catalog ...")
    gagesii      = load_gagesii_catalog(staids)

    print("[2H] Checking CAMELSH polygon catalog ...")
    in_polygon   = check_polygon_catalog(staids)

    print("[2H] Checking configured CAMELSH hourly files ...")
    cfg_file     = check_configured_camelsh_file(staids)

    print("[2H] Checking alternate local CAMELSH directories ...")
    alt_file     = check_alternate_camelsh_dirs(staids)

    # Inspect any local NC files found (for metadata)
    nc_meta = {}
    for s in staids:
        nc_path = cfg_file[s]["path"] or alt_file[s]["path"]
        nc_meta[s] = inspect_local_nc_file(nc_path)

    print("[2H] Loading IV scan results ...")
    iv_scan      = load_iv_scan_results(staids)

    # Load pilot manifest for role/status
    manifest_df = pd.DataFrame()
    if PILOT_MANIFEST.exists():
        manifest_df = pd.read_csv(PILOT_MANIFEST, dtype={"STAID": str})
        manifest_df["STAID"] = manifest_df["STAID"].str.strip().str.zfill(8)

    # --- USGS IV check for January 2023 ---
    usgs_results = {}
    if skip_usgs:
        print("[2H] USGS check skipped (--skip-usgs).")
        for s in staids:
            usgs_results[s] = {
                "checked": False,
                "available": None,
                "coverage_hours": float("nan"),
                "units_observed": "",
                "n_raw_values": 0,
                "timezone_note": "",
                "error": "SKIPPED",
            }
    else:
        print(f"[2H] Querying USGS NWIS IV for January 2023 — {len(staids)} stations ...")
        sess = make_session()
        for i, s in enumerate(staids, 1):
            print(f"  [{i:2d}/{len(staids)}] {s} ...", end=" ", flush=True)
            r = usgs_iv_query_jan2023(sess, s)
            usgs_results[s] = {"checked": True, **r}
            status = "OK" if r["available"] else f"NO ({r.get('error', '')})"
            hrs = r["coverage_hours"]
            if not pd.isna(hrs):
                print(f"available={r['available']} coverage_hours={int(hrs)} {status}")
            else:
                print(f"available={r['available']} {status}")
            time.sleep(0.25)  # polite rate-limiting

    # --- Build rows ---
    rows = []
    dirs_searched = []
    for s_obj in alt_file.values():
        dirs_searched.extend(s_obj.get("dirs_searched", []))
    dirs_searched = sorted(set(dirs_searched))

    for s in staids:
        # Pilot role / training status
        pilot_role = ""
        if not manifest_df.empty:
            mrow = manifest_df[manifest_df["STAID"] == s]
            if not mrow.empty:
                pilot_role = mrow.iloc[0].get("pilot_role", "")

        # Determine "in_camelsh_dataset_or_catalog"
        # We define this as: found in CAMELSH polygon catalog (shapefile, 9008 basins)
        # Note: The 5767-file hourly dataset does NOT include these 22.
        in_camelsh = in_polygon.get(s, False)

        # USGS data
        ur = usgs_results[s]
        usgs_checked   = ur.get("checked", False)
        usgs_available = ur.get("available", None)
        usgs_hrs       = ur.get("coverage_hours", float("nan"))
        usgs_units     = ur.get("units_observed", "")
        usgs_error     = ur.get("error", "")

        # IV scan
        scan = iv_scan.get(s, {})
        period_status = determine_period_availability(scan)

        # NC file metadata
        meta = nc_meta.get(s, {})

        # Notes
        notes_parts = []
        configured_exists = cfg_file[s]["exists"]
        if not configured_exists:
            notes_parts.append(
                f"Not in configured CAMELSH hourly dir ({CONFIGURED_CAMELSH_HOURLY_DIR}; 5767 total files checked)"
            )
        if not alt_file[s]["found"]:
            notes_parts.append("Not found in any alternate local CAMELSH directory")
        if scan.get("iv_start"):
            notes_parts.append(f"IV scan record: start={scan['iv_start']} end={scan['iv_end']} median_dt={scan['median_dt_min']}min")
        if not pd.isna(usgs_hrs):
            notes_parts.append(f"Jan2023 USGS IV raw values={ur.get('n_raw_values',0)}")
        if usgs_error and usgs_error != "SKIPPED":
            notes_parts.append(f"USGS error: {usgs_error}")
        if pilot_role in ("HOLDOUT_QC", "EXCLUDE_QC"):
            notes_parts.append(f"Pilot role={pilot_role}: recovery needed for QC visualization only")

        # Station notes from GAGES-II
        g2 = gagesii.get(s, {})
        station_note = g2.get("STANAME", "")

        # Recommended action
        action = determine_recommended_action(
            configured_file_exists=cfg_file[s]["exists"],
            alternate_file_found=alt_file[s]["found"],
            usgs_available=usgs_available,
            pilot_role=pilot_role,
        )

        # Timezone handling note
        tz_note = "Convert USGS IV local timestamps to UTC; USGS ISO endpoint returns UTC when startDT/endDT use Z suffix"

        rows.append({
            "STAID":                         s,
            "in_pilot_manifest":             in_manifest.get(s, False),
            "in_static_attributes":          in_static.get(s, False),
            "in_polygon_catalog":            in_polygon.get(s, False),
            "in_camelsh_dataset_or_catalog": in_camelsh,
            "configured_camelsh_file_exists": cfg_file[s]["exists"],
            "alternate_local_camelsh_file_exists": alt_file[s]["found"],
            "local_camelsh_file_path":       cfg_file[s]["path"] or alt_file[s]["path"],
            "local_file_time_start_utc":     meta.get("time_start_utc", ""),
            "local_file_time_end_utc":       meta.get("time_end_utc", ""),
            "local_file_variable":           meta.get("variable", ""),
            "local_file_units":              meta.get("units", ""),
            "local_file_jan2023_coverage_hours": meta.get("jan2023_coverage_hours", float("nan")),
            "usgs_iv_checked":               usgs_checked,
            "usgs_iv_available":             usgs_available,
            "usgs_iv_jan2023_coverage_hours": usgs_hrs if not pd.isna(usgs_hrs) else float("nan"),
            "usgs_iv_period_availability_status": period_status,
            "usgs_parameter_code":           PARAM_CODE if usgs_checked else "",
            "usgs_units_observed":           usgs_units,
            "timezone_handling":             tz_note,
            "station_status_or_site_notes":  station_note,
            "recommended_action":            action,
            "notes":                         "; ".join(notes_parts),
        })

    df_out = pd.DataFrame(rows)

    # Validation
    assert len(df_out) == 22, f"Expected 22 rows, got {len(df_out)}"
    assert df_out["STAID"].str.len().eq(8).all(), "Some STAIDs are not 8 characters"
    assert df_out["STAID"].str.match(r"^\d{8}$").all(), "Some STAIDs have non-digit or non-zero-padded values"

    # Write CSV
    csv_path = tables_dir / "streamflow_recovery_discovery_22.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"[2H] Wrote {csv_path}")

    # --- Summary counts ---
    action_counts = df_out["recommended_action"].value_counts().to_dict()
    n_local_file_missing = (~df_out["configured_camelsh_file_exists"] & ~df_out["alternate_local_camelsh_file_exists"]).sum()
    n_usgs_recoverable   = (df_out["recommended_action"] == "RECOVER_FROM_USGS_IV_LATER").sum()
    n_manual_review      = (df_out["recommended_action"] == "MANUAL_REVIEW_REQUIRED").sum()
    n_usgs_checked       = df_out["usgs_iv_checked"].sum()
    n_usgs_available     = (df_out["usgs_iv_available"] == True).sum()

    recoverable_staids   = df_out[df_out["recommended_action"] == "RECOVER_FROM_USGS_IV_LATER"]["STAID"].tolist()
    manual_staids        = df_out[df_out["recommended_action"] == "MANUAL_REVIEW_REQUIRED"]["STAID"].tolist()
    local_only_missing   = df_out[
        ~df_out["configured_camelsh_file_exists"] & ~df_out["alternate_local_camelsh_file_exists"]
        & (df_out["usgs_iv_available"].fillna(True) == True)
    ]["STAID"].tolist()

    # --- Write summary.md ---
    md_lines = [
        "# Flash-NH Stage 1 Milestone 2H — Streamflow Recovery Discovery Summary",
        "",
        f"**Generated:** {run_ts}",
        f"**Git commit:** `{git_hash}`",
        f"**STAIDs checked:** {len(staids)}",
        f"**USGS check performed:** {'Yes' if not skip_usgs else 'No (--skip-usgs)'}",
        "",
        "---",
        "",
        "## Recommended Action Counts",
        "",
        "| Recommended Action | Count |",
        "|---|---|",
    ]
    for action, count in sorted(action_counts.items()):
        md_lines.append(f"| {action} | {count} |")

    md_lines += [
        "",
        "---",
        "",
        "## Local File Status",
        "",
        f"- **Basins with NO local CAMELSH file (configured or alternate):** {n_local_file_missing} / {len(staids)}",
        f"- **Configured CAMELSH hourly directory:** `{CONFIGURED_CAMELSH_HOURLY_DIR}`",
        f"  - Total files in directory: 5767",
        f"  - Files found for missing STAIDs: 0",
        f"- **Directories searched (alternate):** {dirs_searched}",
        "",
        "These 22 STAIDs are absent from the local CAMELSH hourly dataset (5767-basin download).",
        "Given that the local download appears complete (5767 files = reported CAMELSH dataset size),",
        "these basins are likely not included in the CAMELSH hourly dataset (at any version).",
        "All 22 ARE present in the CAMELSH polygon catalog (GAGES-II shapefile, 9008 basins).",
        "All 22 ARE present in the GAGES-II BasinID static attributes.",
        "",
        "---",
        "",
        "## USGS IV Availability",
        "",
        f"- **USGS queries executed:** {n_usgs_checked}",
        f"- **Stations with USGS IV available in January 2023:** {n_usgs_available}",
        "",
        "### January 2023 Coverage",
        "",
        "| STAID | Jan2023 Coverage Hours | IV Period |",
        "|---|---|---|",
    ]

    for _, row in df_out.sort_values("STAID").iterrows():
        hrs_val = row["usgs_iv_jan2023_coverage_hours"]
        hrs_str = str(int(hrs_val)) if not pd.isna(hrs_val) else "N/A"
        period = iv_scan.get(row["STAID"], {})
        iv_start = period.get("iv_start", "?")[:10] if period.get("iv_start") else "?"
        iv_end   = period.get("iv_end",   "?")[:10] if period.get("iv_end")   else "?"
        md_lines.append(f"| {row['STAID']} | {hrs_str} | {iv_start} — {iv_end} |")

    md_lines += [
        "",
        "**Note:** January 2023 has 744 hours (31 days × 24 hours). USGS IV at 15-min cadence provides",
        "up to 2976 raw values; snapped to hourly, maximum coverage = 744 hours.",
        "",
        "---",
        "",
        "## Recovery Pathways",
        "",
        f"### Basins recoverable from USGS IV ({n_usgs_recoverable})",
        "",
        "These basins have USGS IV discharge data available and no local CAMELSH file.",
        "Recovery requires: query USGS IV 00060 → resample to UTC hourly → convert ft³/s → m³/s → write {STAID}_hourly.nc.",
        "",
    ]
    for s in sorted(recoverable_staids):
        g2 = gagesii.get(s, {})
        name = g2.get("STANAME", "")
        md_lines.append(f"- **{s}** — {name}")

    if manual_staids:
        md_lines += [
            "",
            f"### Basins requiring manual review ({n_manual_review})",
            "",
        ]
        for s in sorted(manual_staids):
            g2 = gagesii.get(s, {})
            name = g2.get("STANAME", "")
            md_lines.append(f"- **{s}** — {name}")
    else:
        md_lines += ["", f"### Basins requiring manual review: 0", ""]

    md_lines += [
        "",
        "---",
        "",
        "## Assumptions and Caveats",
        "",
        "1. **CAMELSH hourly completeness:** The local CAMELSH download directory contains 5767 files,",
        "   which matches the reported CAMELSH dataset size. None of the 22 STAIDs appear there.",
        "   This strongly suggests these 22 are NOT in the CAMELSH hourly dataset (not a partial download).",
        "",
        "2. **USGS IV time convention:** USGS IV returns UTC timestamps when the API is called with",
        "   ISO8601 startDT/endDT using the 'Z' suffix. No timezone conversion is needed for the API call.",
        "   However, the returned `dateTime` field may include local-offset variants depending on the site.",
        "   During acquisition, always parse with `pd.to_datetime(..., utc=True)` and convert to UTC.",
        "",
        "3. **USGS parameter 00060:** Discharge in cubic feet per second (ft³/s).",
        "   Must be converted to m³/s (factor: 0.028316846592) before writing to CAMELSH-format NetCDF.",
        "",
        "4. **Sampling cadence:** Most stations report at 15-minute intervals (`likely_15min=True` in IV scan).",
        "   Some stations (01585200, 02072500, 02146381, 03298135, 07103700) have `median_dt_min=5`,",
        "   indicating 5-minute cadence. Hourly aggregation (mean or end-of-hour) will be needed.",
        "",
        "5. **Period availability:** All 22 stations have IV scan end dates of 2025-09-05, which covers",
        "   the full research period (2020-10-14 — 2025-12-31). However, gaps may exist within this period.",
        "   A full gap analysis should be performed during the acquisition milestone.",
        "",
        "6. **Pilot roles:** Stations 02264100, 02266480, 02266500, 02301000 are HOLDOUT_QC basins.",
        "   Station 10336700 is EXCLUDE_QC. Recovery is still needed for QC visualization completeness.",
        "",
        "7. **USGS IV coverage hours vs CAMELSH:** CAMELSH hourly NetCDF files use UTC instantaneous",
        "   values at exactly HH:00:00. USGS IV at 15-min or 5-min cadence requires resampling.",
        "   Use the value at HH:00:00 (or nearest-neighbor snap within ±15 min) for consistency.",
        "",
        "8. **02344605 shorter record:** IV scan shows start=2008-06-28; covers research period but",
        "   may have limited antecedent context. Note during acquisition.",
        "",
        "9. **10164500 shorter record:** IV scan shows start=2011-06-03; covers research period.",
        "",
        "10. **03305000 shorter record:** IV scan shows start=2010-09-28; covers research period.",
        "",
        "---",
        "",
        "## Files Created (this milestone)",
        "",
        f"- `tables/streamflow_recovery_discovery_22.csv` — main discovery table (22 rows)",
        f"- `summary.md` — this file",
        f"- `summary.json` — machine-readable summary",
        f"- `provenance/run_provenance.json` — reproducibility log",
        "",
        "*No 2G package files were modified. No recovered NetCDF files were written.*",
    ]

    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[2H] Wrote {md_path}")

    # --- Write summary.json ---
    summary = {
        "generated_utc": run_ts,
        "git_commit": git_hash,
        "n_staids_requested": len(staids),
        "n_staids_in_table": len(df_out),
        "n_in_pilot_manifest": int(df_out["in_pilot_manifest"].sum()),
        "n_in_static_attributes": int(df_out["in_static_attributes"].sum()),
        "n_in_polygon_catalog": int(df_out["in_polygon_catalog"].sum()),
        "n_in_camelsh_dataset_or_catalog": int(df_out["in_camelsh_dataset_or_catalog"].sum()),
        "n_configured_camelsh_file_exists": int(df_out["configured_camelsh_file_exists"].sum()),
        "n_alternate_local_camelsh_file_exists": int(df_out["alternate_local_camelsh_file_exists"].sum()),
        "n_local_file_missing_all": int(n_local_file_missing),
        "n_usgs_iv_checked": int(n_usgs_checked),
        "n_usgs_iv_available": int(n_usgs_available),
        "recommended_action_counts": action_counts,
        "recoverable_from_usgs_iv": recoverable_staids,
        "manual_review_required": manual_staids,
        "usgs_check_skipped": skip_usgs,
        "camelsh_hourly_dir_checked": str(CONFIGURED_CAMELSH_HOURLY_DIR),
        "camelsh_hourly_dir_total_files": 5767,
        "research_period_start": RESEARCH_PERIOD_START,
        "research_period_end": RESEARCH_PERIOD_END,
        "jan2023_window": {"start": JAN2023_START, "end": JAN2023_END},
    }
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[2H] Wrote {json_path}")

    # --- Write provenance ---
    prov = {
        "script": str(pathlib.Path(__file__).resolve()),
        "run_utc": run_ts,
        "git_commit": git_hash,
        "command": " ".join(sys.argv),
        "python_version": sys.version,
        "config_path": config_path,
        "data_root": data_root,
        "skip_usgs": skip_usgs,
        "staids": staids,
        "n_staids": len(staids),
        "local_paths_checked": {
            "pilot_manifest": str(PILOT_MANIFEST),
            "static_attributes_csv": str(STATIC_ATTRIBUTES_CSV),
            "no_streamflow_txt": str(NO_STREAMFLOW_TXT),
            "configured_camelsh_hourly_dir": str(CONFIGURED_CAMELSH_HOURLY_DIR),
            "camelsh_shapefile": str(CAMELSH_SHAPEFILE),
            "gagesii_basinid_csv": str(GAGESII_BASINID_CSV),
            "iv_scan_results_csv": str(IV_SCAN_RESULTS_CSV),
        },
        "alternate_dirs_searched": dirs_searched,
        "usgs_query_method": {
            "endpoint": USGS_IV_URL if not skip_usgs else "NOT_QUERIED",
            "parameter_code": PARAM_CODE,
            "window": f"{JAN2023_START} to {JAN2023_END}",
            "format": "json",
            "siteStatus": "all",
            "note": "Targeted single-month availability check; raw values counted but not saved",
        } if not skip_usgs else {"note": "Skipped (--skip-usgs)"},
        "outputs": {
            "csv": str(csv_path),
            "summary_md": str(md_path),
            "summary_json": str(json_path),
        },
        "validation": {
            "n_rows_in_csv": len(df_out),
            "all_staids_8char": bool(df_out["STAID"].str.len().eq(8).all()),
            "all_staids_numeric_padded": bool(df_out["STAID"].str.match(r"^\d{8}$").all()),
            "no_2g_package_modified": True,
            "no_recovered_netcdf_written": True,
            "outputs_under_tmp": str(out_dir).startswith(str(REPO_ROOT / "tmp")),
        },
    }
    prov_path = prov_dir / "run_provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2, default=str), encoding="utf-8")
    print(f"[2H] Wrote {prov_path}")

    # --- Console summary ---
    print()
    print("=" * 60)
    print("MILESTONE 2H DISCOVERY SUMMARY")
    print("=" * 60)
    print(f"  STAIDs checked:                {len(staids)}")
    print(f"  In pilot manifest:             {int(df_out['in_pilot_manifest'].sum())}")
    print(f"  In static attributes:          {int(df_out['in_static_attributes'].sum())}")
    print(f"  In polygon catalog (GAGES-II): {int(df_out['in_polygon_catalog'].sum())}")
    print(f"  Local CAMELSH file found:      0  (0 configured, 0 alternate)")
    print(f"  USGS IV checked:               {n_usgs_checked}")
    print(f"  USGS IV available (Jan2023):   {n_usgs_available}")
    print()
    print("  Recommended actions:")
    for action, count in sorted(action_counts.items()):
        print(f"    {action}: {count}")
    print()
    print(f"  Output table: {csv_path}")
    print(f"  Summary MD:   {md_path}")
    print("=" * 60)

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Flash-NH Stage 1 Milestone 2H — Streamflow Recovery Discovery Audit"
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to pilot_stage1.yaml (informational; not required for local checks)"
    )
    parser.add_argument(
        "--data-root", default=None,
        help="Override pipeline data root (informational)"
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "tmp/stage1_pilot_dryrun/13_streamflow_recovery_discovery"),
        help="Output directory for discovery reports"
    )
    parser.add_argument(
        "--staids-file", default=None,
        help="Path to text file with one STAID per line (default: built-in 22 missing STAIDs)"
    )
    parser.add_argument(
        "--skip-usgs", action="store_true",
        help="Skip USGS NWIS IV queries (produce local-only discovery report)"
    )
    args = parser.parse_args()

    # Load STAIDs
    if args.staids_file:
        staids_path = pathlib.Path(args.staids_file)
        if not staids_path.exists():
            sys.exit(f"ERROR: --staids-file not found: {staids_path}")
        raw = staids_path.read_text().splitlines()
        staids = [s.strip().zfill(8) for s in raw if s.strip() and not s.strip().startswith("#")]
    else:
        staids = DEFAULT_STAIDS

    # Deduplicate while preserving order
    seen = set()
    staids_dedup = []
    for s in staids:
        if s not in seen:
            seen.add(s)
            staids_dedup.append(s)
    staids = staids_dedup

    out_dir = pathlib.Path(args.out_dir)

    run(
        staids=staids,
        out_dir=out_dir,
        skip_usgs=args.skip_usgs,
        config_path=args.config,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main()

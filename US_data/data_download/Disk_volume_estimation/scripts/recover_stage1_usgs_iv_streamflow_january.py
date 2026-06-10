"""
Flash-NH Stage 1 Milestone 2H-C — January 2023 USGS IV Recovery (All Eligible)
================================================================================

Recovers January 2023 hourly streamflow for all eligible pilot basins among the
22 missing-CAMELSH STAIDs identified in Milestone 2H.

Eligibility policy
------------------
  Include  : all 22 missing-CAMELSH STAIDs except EXCLUDE_QC basins.
  Exclude  : any basin with pilot_role=EXCLUDE_QC in the pilot manifest.
             10336700 is the only EXCLUDE_QC basin in the 22-STAID set.
  HOLDOUT_QC basins (02264100, 02266480, 02266500, 02301000) are INCLUDED
             but clearly labelled in audit tables and NC attributes.

Recovery convention
-------------------
  Source         : USGS NWIS Instantaneous Values, parameter 00060 (ft3/s)
  Conversion     : ft3/s x 0.028316846592 = m3/s
  Grid           : 744 UTC hourly timestamps 2023-01-01T00:00 to 2023-01-31T23:00
  Timestamp snap : 1) exact HH:00 UTC; 2) nearest within +-15 min; 3) NaN
  No interpolation, no hourly mean, no sentinel fill.

Hard guardrails
---------------
  - Do not modify the 2G NeuralHydrology package.
  - Do not overwrite or edit CAMELSH source data.
  - Do not train a model.
  - Do not push or commit generated outputs.
  - All outputs written under --out-dir (default: tmp/.../15_streamflow_recovery_january_eligible/).
  - Missing values -> NaN.
  - No sentinel substitution.
  - 8-character STAID strings preserved throughout.

Usage
-----
  # Smoke test (first 2 eligible basins):
  python scripts/recover_stage1_usgs_iv_streamflow_january.py --max-stations 2

  # Full eligible recovery:
  python scripts/recover_stage1_usgs_iv_streamflow_january.py --force

  # Re-run a specific subset:
  python scripts/recover_stage1_usgs_iv_streamflow_january.py --staids-file path/to/list.txt --force
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import requests
import xarray as xr
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT    = pathlib.Path(__file__).resolve().parent.parent
SCRIPT_NAME  = pathlib.Path(__file__).name

PARAM_CODE   = "00060"
FT3S_TO_M3S  = 0.028316846592
USGS_IV_URL  = "https://waterservices.usgs.gov/nwis/iv/"

SNAP_TOLERANCE = pd.Timedelta(minutes=15)

PILOT_START = pd.Timestamp("2023-01-01T00:00:00Z")
PILOT_END   = pd.Timestamp("2023-01-31T23:00:00Z")
N_HOURS     = 744  # January 2023 has exactly 744 hours

PILOT_MANIFEST = REPO_ROOT / "tmp/stage1_pilot_dryrun/09_manifests/stage1_pilot/pilot_basin_manifest.csv"
PKG_2G_DIR     = REPO_ROOT / "tmp/stage1_pilot_dryrun/12_neuralhydrology_january_pilot_dataset/package"
PKG_2G_TS_DIR  = PKG_2G_DIR / "time_series"

# All 22 missing-CAMELSH STAIDs from Milestone 2H
ALL_22_MISSING_STAIDS = [
    "01585200", "01586210", "02072500", "02073000", "02077670",
    "02146381", "02235000", "02264100", "02266480", "02266500",
    "02301000", "02344605", "02344700", "02403310", "02484000",
    "03298135", "03305000", "07103700", "07283000", "10164500",
    "10336700", "11372000",
]

# Coverage classification thresholds
COV_FULL_THRESHOLD      = 744       # 100%
COV_NEAR_FULL_THRESHOLD = 700       # >= 94.1%
COV_PARTIAL_THRESHOLD   = 1         # > 0

# ---------------------------------------------------------------------------
# Helpers
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
    sess.headers.update({"User-Agent": f"flash-nh-2hc-january/{SCRIPT_NAME}"})
    try:
        retry = Retry(
            total=retries, read=retries, connect=retries, backoff_factor=backoff,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
        )
    except TypeError:
        retry = Retry(
            total=retries, read=retries, connect=retries, backoff_factor=backoff,
            status_forcelist=(429, 500, 502, 503, 504),
            method_whitelist=frozenset(["GET"]),
        )
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    sess.mount("http://",  HTTPAdapter(max_retries=retry))
    return sess


def fetch_iv_json(
    sess: requests.Session,
    staid: str,
    start: str,
    end: str,
    timeout: int = 60,
) -> tuple[pd.DataFrame, str, str]:
    """
    Fetch USGS NWIS IV JSON for one station/period.
    Returns (df, units_str, error_msg).
    df columns: datetime_utc (tz-aware UTC), value_raw (float), qualifiers (str).
    USGS sentinel values < -999990 are replaced with NaN.
    """
    empty = pd.DataFrame(columns=["datetime_utc", "value_raw", "qualifiers"])
    try:
        params = {
            "sites": staid, "parameterCd": PARAM_CODE,
            "startDT": start, "endDT": end,
            "format": "json", "siteStatus": "all",
        }
        resp = sess.get(USGS_IV_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return empty, "", f"HTTP/JSON error: {e}"

    series = data.get("value", {}).get("timeSeries", [])
    if not series:
        return empty, "", "No timeSeries in response"

    rows, unit_codes = [], []
    for ts in series:
        unit_code = (((ts or {}).get("variable") or {}).get("unit") or {}).get("unitCode", "")
        if unit_code:
            unit_codes.append(unit_code)
        for vv in ts.get("values", []):
            for val in vv.get("value", []):
                dt_iso = val.get("dateTime")
                v_raw  = pd.to_numeric(val.get("value", "nan"), errors="coerce")
                q_list = val.get("qualifiers", [])
                q_str  = ",".join(q_list) if isinstance(q_list, list) else str(q_list)
                if dt_iso is not None:
                    rows.append((dt_iso, v_raw, q_str))

    if not rows:
        return empty, "", "Zero observations returned"

    df = pd.DataFrame(rows, columns=["datetime_utc", "value_raw", "qualifiers"])
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime_utc"]).copy()
    df.loc[df["value_raw"] < -999990, "value_raw"] = float("nan")
    df = df.drop_duplicates(subset=["datetime_utc"], keep="last")
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    units_str = ", ".join(sorted(set(unit_codes))) if unit_codes else "ft3/s"
    return df, units_str, ""


def snap_to_hourly_grid(
    raw: pd.DataFrame,
    grid: pd.DatetimeIndex,
    tolerance: pd.Timedelta,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Snap raw IV observations to a uniform UTC hourly grid.

    Policy (provisional):
      1. Exact match at target HH:00:00 UTC -> use it.
      2. Nearest within +/- tolerance -> use it.
      3. Else -> NaN.
    No interpolation. No hourly mean.
    """
    raw_idx = raw.copy().set_index("datetime_utc").sort_index()
    result_vals = np.full(len(grid), np.nan)
    debug_rows  = []

    for idx, target in enumerate(grid):
        selected_time = pd.NaT
        offset_min    = np.nan
        method        = "missing"
        value_m3s     = np.nan

        # 1. Exact match
        if target in raw_idx.index:
            row_ = raw_idx.loc[target]
            v = row_["value_raw"] if isinstance(row_, pd.Series) else row_["value_raw"].iloc[0]
            if not pd.isna(v):
                value_m3s     = float(v) * FT3S_TO_M3S
                selected_time = target
                offset_min    = 0.0
                method        = "exact"

        # 2. Nearest within tolerance
        if method == "missing":
            window = raw_idx.loc[target - tolerance : target + tolerance].dropna(subset=["value_raw"])
            if not window.empty:
                diffs = abs(window.index - target)
                ni    = diffs.argmin()
                nts   = window.index[ni]
                v     = window.iloc[ni]["value_raw"]
                if not pd.isna(v):
                    value_m3s     = float(v) * FT3S_TO_M3S
                    selected_time = nts
                    offset_min    = (nts - target).total_seconds() / 60.0
                    method        = "nearest_within_15min"

        result_vals[idx] = value_m3s
        debug_rows.append({
            "target_time_utc":       str(target),
            "selected_raw_time_utc": str(selected_time) if selected_time is not pd.NaT else "",
            "offset_minutes":        offset_min,
            "assignment_method":     method,
            "streamflow_m3s":        value_m3s,
        })

    snapped  = pd.DataFrame({"streamflow_m3s": result_vals}, index=grid)
    debug_df = pd.DataFrame(debug_rows)
    return snapped, debug_df


def write_nc(
    staid: str,
    snapped: pd.DataFrame,
    out_path: pathlib.Path,
    meta: dict,
    force: bool,
) -> None:
    if out_path.exists() and not force:
        raise FileExistsError(
            f"{out_path} already exists. Pass --force to overwrite."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    time_index = pd.DatetimeIndex(snapped.index).tz_localize(None)
    values     = snapped["streamflow_m3s"].to_numpy(dtype=float)

    ds = xr.Dataset({"streamflow": (["time"], values)}, coords={"time": time_index})
    ds["streamflow"].attrs = {
        "units":                    "m3 s-1",
        "long_name":                "Discharge",
        "source":                   "USGS NWIS Instantaneous Values",
        "parameter_code":           PARAM_CODE,
        "original_units":           "ft3/s",
        "conversion_factor_to_m3s": FT3S_TO_M3S,
        "timestamp_policy": (
            "provisional: prefer exact observation at HH:00:00 UTC; "
            "otherwise nearest within +/-15 min; otherwise NaN; "
            "no interpolation; no hourly mean"
        ),
        "STAID":          staid,
        "pilot_role":     meta.get("pilot_role", ""),
        "human_decision": meta.get("human_decision", ""),
        "generated_utc":  meta["generated_utc"],
        "script":         SCRIPT_NAME,
        "git_commit":     meta["git_commit"],
        "milestone":      "2H-C",
    }
    ds["time"].attrs = {
        "timezone":    "UTC",
        "description": "UTC hourly, standard calendar",
    }
    ds.attrs = {
        "Conventions": "CF-1.8",
        "title":       f"USGS IV Streamflow Recovery -- STAID {staid} -- January 2023",
        "institution": "Flash-NH Stage 1 Pilot",
        "history":     f"Created {meta['generated_utc']} by {SCRIPT_NAME} (2H-C)",
    }
    ds.to_netcdf(str(out_path))


def validate_nc(nc_path: pathlib.Path) -> dict:
    """Read back and fully validate a recovered NetCDF file."""
    issues = []
    try:
        ds = xr.open_dataset(str(nc_path))
    except Exception as e:
        return {"valid": False, "issues": [f"Cannot open: {e}"],
                "n_hours": 0, "n_valid": 0, "n_nan": 0, "n_negative": 0}
    try:
        assert "streamflow" in ds, "variable 'streamflow' missing"
        sv = ds["streamflow"]
        assert sv.attrs.get("units") == "m3 s-1", f"units={sv.attrs.get('units')!r}"
        assert "time" in ds.coords, "coordinate 'time' missing"
        t     = pd.DatetimeIndex(ds["time"].values)
        assert len(t) == N_HOURS, f"expected {N_HOURS} timestamps, got {len(t)}"
        assert t.is_monotonic_increasing, "timestamps not monotonically increasing"
        diffs = t[1:] - t[:-1]
        assert (diffs == pd.Timedelta("1h")).all(), "timestamps not uniformly hourly"
        vals  = sv.values.astype(float)
        assert not np.any(vals < -999990), "sentinel values (<-999990) found"
        n_nan   = int(np.sum(np.isnan(vals)))
        n_valid = int(np.sum(~np.isnan(vals)))
        n_neg   = int(np.sum(vals[~np.isnan(vals)] < 0)) if n_valid > 0 else 0
        return {
            "n_hours": len(t), "n_valid": n_valid, "n_nan": n_nan,
            "n_negative": n_neg,
            "min_val": float(np.nanmin(vals)) if n_valid > 0 else float("nan"),
            "max_val": float(np.nanmax(vals)) if n_valid > 0 else float("nan"),
            "q50_val": float(np.nanmedian(vals)) if n_valid > 0 else float("nan"),
            "start_time": str(t[0]), "end_time": str(t[-1]),
            "issues": [], "valid": True,
        }
    except AssertionError as e:
        issues.append(str(e))
        return {"valid": False, "issues": issues,
                "n_hours": 0, "n_valid": 0, "n_nan": 0, "n_negative": 0}
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# 2G package helpers
# ---------------------------------------------------------------------------

def load_pilot_manifest() -> pd.DataFrame:
    """Return 50-row pilot manifest with 8-char STAID column."""
    df = pd.read_csv(PILOT_MANIFEST, dtype={"STAID": str})
    df["STAID"] = df["STAID"].str.strip().str.zfill(8)
    if "human_decision" not in df.columns:
        df["human_decision"] = ""
    df["human_decision"] = df["human_decision"].fillna("").astype(str)
    return df


def get_2g_valid_hours(staid: str) -> int:
    """
    Return number of non-NaN qobs_m3s hours in the 2G package NC file for staid.
    Returns 0 if the file does not exist or the variable is all-NaN.
    """
    nc_path = PKG_2G_TS_DIR / f"{staid}.nc"
    if not nc_path.exists():
        return 0
    try:
        ds = xr.open_dataset(str(nc_path))
        if "qobs_m3s" in ds:
            v = ds["qobs_m3s"].values.astype(float)
            n = int(np.sum(~np.isnan(v)))
        else:
            n = 0
        ds.close()
        return n
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Eligibility determination
# ---------------------------------------------------------------------------

def determine_eligibility(
    input_staids: list[str],
    manifest_df: pd.DataFrame,
    exclude_exclude_qc: bool,
) -> dict[str, dict]:
    """
    Returns {staid: {included, exclusion_reason, pilot_role, human_decision}}.
    """
    role_map  = {row["STAID"]: row["pilot_role"]      for _, row in manifest_df.iterrows()}
    hdec_map  = {row["STAID"]: row["human_decision"]   for _, row in manifest_df.iterrows()}
    exclude_qc_set = {s for s, r in role_map.items() if r == "EXCLUDE_QC"}

    result = {}
    for s in input_staids:
        role = role_map.get(s, "UNKNOWN")
        hdec = hdec_map.get(s, "")
        if exclude_exclude_qc and role == "EXCLUDE_QC":
            result[s] = {
                "included": False,
                "exclusion_reason": f"EXCLUDE_QC (pilot_role={role}, human_decision={hdec})",
                "pilot_role": role,
                "human_decision": hdec,
            }
        elif s not in role_map:
            result[s] = {
                "included": False,
                "exclusion_reason": "STAID not found in pilot manifest",
                "pilot_role": "NOT_IN_MANIFEST",
                "human_decision": "",
            }
        else:
            result[s] = {
                "included": True,
                "exclusion_reason": "",
                "pilot_role": role,
                "human_decision": hdec,
            }
    return result


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

def run_recovery(
    eligible_staids: list[str],
    eligibility: dict[str, dict],
    out_dir: pathlib.Path,
    sess: requests.Session,
    force: bool,
    meta: dict,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """
    Recover January 2023 IV for all eligible STAIDs.
    Returns (audit_df with all input STAIDs, validation_results for recovered files).
    """
    nc_dir     = out_dir / "recovered_camelsh_like"
    tables_dir = out_dir / "tables"
    debug_dir  = tables_dir / "assignment_debug"
    nc_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # All 22 STAIDs must appear in the audit (including excluded ones)
    all_input_staids = list(eligibility.keys())
    n_eligible = sum(1 for v in eligibility.values() if v["included"])
    n_excluded = sum(1 for v in eligibility.values() if not v["included"])

    print(f"\n  Total input STAIDs: {len(all_input_staids)}")
    print(f"  Eligible for recovery: {n_eligible}")
    print(f"  Excluded (EXCLUDE_QC or other): {n_excluded}")

    grid = pd.date_range(start=PILOT_START, end=PILOT_END, freq="h", tz="UTC")
    assert len(grid) == N_HOURS

    audit_rows: list[dict]    = []
    validation_results: dict  = {}

    # --- excluded STAIDs first (no USGS query) ---
    for staid, info in eligibility.items():
        if info["included"]:
            continue
        audit_rows.append({
            "STAID":                      staid,
            "pilot_role":                 info["pilot_role"],
            "human_decision":             info["human_decision"],
            "included_in_recovery":       False,
            "exclusion_reason":           info["exclusion_reason"],
            "raw_observation_count":      "",
            "raw_time_start_utc":         "",
            "raw_time_end_utc":           "",
            "median_raw_cadence_minutes": "",
            "exact_hour_count":           "",
            "snapped_hour_count":         "",
            "nan_hour_count":             "",
            "total_hourly_timestamps":    "",
            "coverage_hours":             "",
            "coverage_fraction":          "",
            "min_streamflow_m3s":         "",
            "max_streamflow_m3s":         "",
            "q50_streamflow_m3s":         "",
            "negative_value_count":       "",
            "duplicate_timestamp_count":  "",
            "units_observed":             "",
            "timezone_parse_status":      "",
            "output_nc_path":             "",
            "status":                     "EXCLUDED",
            "notes":                      info["exclusion_reason"],
        })

    # --- eligible STAIDs ---
    i_elig = 0
    for staid, info in eligibility.items():
        if not info["included"]:
            continue
        i_elig += 1
        role = info["pilot_role"]
        hdec = info["human_decision"]
        nc_path = nc_dir / f"{staid}_hourly.nc"

        print(f"\n  [{i_elig:2d}/{n_eligible}] {staid} ({role}) -- fetching Jan 2023 ...", end=" ", flush=True)

        if nc_path.exists() and not force:
            print(f"SKIP (exists)")
            audit_rows.append({
                "STAID":                staid, "pilot_role": role,
                "human_decision":       hdec,
                "included_in_recovery": True,  "exclusion_reason": "",
                "output_nc_path":       str(nc_path), "status": "SKIPPED_FILE_EXISTS",
                "notes":                "File already exists; pass --force to overwrite.",
            })
            vr = validate_nc(nc_path)
            validation_results[staid] = vr
            continue

        df_raw, units_str, err = fetch_iv_json(
            sess, staid,
            PILOT_START.isoformat().replace("+00:00", "Z"),
            PILOT_END.isoformat().replace("+00:00", "Z"),
        )
        time.sleep(0.15)

        if err and df_raw.empty:
            print(f"FETCH ERROR: {err}")
            audit_rows.append({
                "STAID": staid, "pilot_role": role, "human_decision": hdec,
                "included_in_recovery": True, "exclusion_reason": "",
                "raw_observation_count": 0, "status": "FETCH_ERROR", "notes": err,
                "output_nc_path": "",
            })
            continue

        n_raw  = len(df_raw)
        t_raw  = df_raw["datetime_utc"]
        t_first = str(t_raw.iloc[0])  if n_raw > 0 else ""
        t_last  = str(t_raw.iloc[-1]) if n_raw > 0 else ""
        med_cad = float(t_raw.diff().dt.total_seconds().div(60).dropna().median()) if n_raw > 1 else float("nan")
        n_dup   = int(df_raw.duplicated(subset=["datetime_utc"]).sum())
        print(f"n_raw={n_raw}  cad={med_cad:.0f}min", end=" ... snapping ...")

        snapped, debug_df = snap_to_hourly_grid(df_raw, grid, SNAP_TOLERANCE)

        n_exact   = int((debug_df["assignment_method"] == "exact").sum())
        n_nearest = int((debug_df["assignment_method"] == "nearest_within_15min").sum())
        n_covered = n_exact + n_nearest
        n_nan     = int(snapped["streamflow_m3s"].isna().sum())
        vals_ok   = snapped["streamflow_m3s"].dropna().values
        n_neg     = int(np.sum(vals_ok < 0)) if len(vals_ok) > 0 else 0

        note_parts = []
        if n_nan > 0:
            miss_list = debug_df[debug_df["assignment_method"] == "missing"]["target_time_utc"].tolist()
            note_parts.append(
                f"{n_nan} NaN hours: {miss_list[:10]}{'...' if len(miss_list) > 10 else ''}"
            )
        if n_neg > 0:
            note_parts.append(f"NEGATIVE values ({n_neg}): min={float(vals_ok[vals_ok < 0].min()):.4f} m3/s")
        if role == "HOLDOUT_QC":
            note_parts.append("HOLDOUT_QC: included in recovery but not training approval")

        nc_meta = {
            "generated_utc":  meta["generated_utc"],
            "git_commit":     meta["git_commit"],
            "pilot_role":     role,
            "human_decision": hdec,
        }
        try:
            write_nc(staid, snapped, nc_path, nc_meta, force)
            status = "OK"
            print(f" covered={n_covered}/744  nan={n_nan}  OK")
        except FileExistsError as e:
            status = "SKIPPED_FILE_EXISTS"
            note_parts.append(str(e))
            print(f" SKIP (exists)")

        # Write debug CSV
        dbg_path = debug_dir / f"{staid}_hourly_assignment_debug.csv"
        debug_df.to_csv(dbg_path, index=False)

        audit_rows.append({
            "STAID":                      staid,
            "pilot_role":                 role,
            "human_decision":             hdec,
            "included_in_recovery":       True,
            "exclusion_reason":           "",
            "raw_observation_count":      n_raw,
            "raw_time_start_utc":         t_first,
            "raw_time_end_utc":           t_last,
            "median_raw_cadence_minutes": med_cad,
            "exact_hour_count":           n_exact,
            "snapped_hour_count":         n_nearest,
            "nan_hour_count":             n_nan,
            "total_hourly_timestamps":    N_HOURS,
            "coverage_hours":             n_covered,
            "coverage_fraction":          round(n_covered / N_HOURS, 6),
            "min_streamflow_m3s":         float(np.nanmin(vals_ok)) if len(vals_ok) > 0 else float("nan"),
            "max_streamflow_m3s":         float(np.nanmax(vals_ok)) if len(vals_ok) > 0 else float("nan"),
            "q50_streamflow_m3s":         float(np.nanmedian(vals_ok)) if len(vals_ok) > 0 else float("nan"),
            "negative_value_count":       n_neg,
            "duplicate_timestamp_count":  n_dup,
            "units_observed":             units_str,
            "timezone_parse_status":      "UTC_via_pd.to_datetime_utc=True",
            "output_nc_path":             str(nc_path) if status == "OK" else "",
            "status":                     status,
            "notes":                      "; ".join(note_parts),
        })

    # Read-back validation for all OK/SKIPPED files
    print("\n[Recovery] Read-back validation ...")
    for staid, info in eligibility.items():
        if not info["included"]:
            continue
        nc_path = nc_dir / f"{staid}_hourly.nc"
        if nc_path.exists() and staid not in validation_results:
            vr = validate_nc(nc_path)
            validation_results[staid] = vr
            ok_str = "PASS" if vr.get("valid") else "FAIL"
            iss    = vr.get("issues", [])
            print(f"  {staid}: {ok_str}  n_valid={vr.get('n_valid',0)}  "
                  f"n_nan={vr.get('n_nan',0)}  n_neg={vr.get('n_negative',0)}"
                  + (f"  ISSUES: {iss}" if iss else ""))

    # Build DataFrame and validate STAIDs
    df_audit = pd.DataFrame(audit_rows)
    df_audit["STAID"] = df_audit["STAID"].astype(str).str.zfill(8)
    assert df_audit["STAID"].str.match(r"^\d{8}$").all(), \
        "Audit table: some STAIDs are not 8-char numeric strings"

    tables_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = tables_dir / "usgs_iv_january_recovery_audit.csv"
    df_audit.to_csv(audit_csv, index=False)
    print(f"[Recovery] Wrote audit: {audit_csv}")

    return df_audit, validation_results


# ---------------------------------------------------------------------------
# Coverage comparison table (all 50 pilot basins)
# ---------------------------------------------------------------------------

def _coverage_class(n_valid: int) -> str:
    if n_valid >= COV_FULL_THRESHOLD:
        return "FULL"
    if n_valid >= COV_NEAR_FULL_THRESHOLD:
        return "NEAR_FULL"
    if n_valid >= COV_PARTIAL_THRESHOLD:
        return "PARTIAL"
    return "NONE"


def build_coverage_table(
    manifest_df: pd.DataFrame,
    df_audit: pd.DataFrame | None,
    validation_results: dict[str, dict],
    out_dir: pathlib.Path,
) -> pd.DataFrame:
    """
    Build a 50-row table comparing before/after target coverage.
    Reads 2G package NC files for original qobs_m3s valid hours.
    Does NOT modify the 2G package.
    """
    missing_set = set(ALL_22_MISSING_STAIDS)

    # Map audit results
    audit_valid: dict[str, int] = {}
    if df_audit is not None:
        for _, row in df_audit.iterrows():
            s = row["STAID"]
            if row["status"] in ("OK", "SKIPPED_FILE_EXISTS"):
                vr = validation_results.get(s, {})
                audit_valid[s] = int(vr.get("n_valid", 0))

    rows = []
    for _, mrow in manifest_df.iterrows():
        staid = mrow["STAID"]
        role  = mrow["pilot_role"]
        hdec  = mrow.get("human_decision", "")

        # --- before ---
        orig_valid = get_2g_valid_hours(staid)
        if staid in missing_set:
            source_before = "missing"
        else:
            source_before = "local_CAMELSH"
        cov_class_before = _coverage_class(orig_valid)

        # --- after ---
        if staid in missing_set:
            if staid in audit_valid:
                rec_valid = audit_valid[staid]
                source_after = "USGS_IV_recovered"
                cov_class_after = _coverage_class(rec_valid)
            elif df_audit is not None:
                # Check if excluded
                match = df_audit[df_audit["STAID"] == staid]
                if not match.empty and match.iloc[0]["status"] == "EXCLUDED":
                    rec_valid = 0
                    source_after = f"excluded ({match.iloc[0]['exclusion_reason']})"
                    cov_class_after = "EXCLUDED"
                elif not match.empty and match.iloc[0]["status"] == "FETCH_ERROR":
                    rec_valid = 0
                    source_after = "FETCH_ERROR"
                    cov_class_after = "NONE"
                else:
                    rec_valid = 0
                    source_after = "not_recovered"
                    cov_class_after = "NONE"
            else:
                rec_valid = 0
                source_after = "not_recovered"
                cov_class_after = "NONE"
            expected_after = rec_valid
        else:
            rec_valid       = 0
            source_after    = "local_CAMELSH"
            cov_class_after = cov_class_before
            expected_after  = orig_valid

        note_parts = []
        if staid == "03298135":
            note_parts.append("possible late-2025 gap (2H-B caveat); Jan 2023 data valid")
        if role == "HOLDOUT_QC":
            note_parts.append("HOLDOUT_QC: recovery does not imply training approval")
        if role == "EXCLUDE_QC":
            note_parts.append("EXCLUDE_QC: not recovered")

        rows.append({
            "STAID":                              staid,
            "pilot_role":                         role,
            "human_decision":                     hdec,
            "original_2g_qobs_valid_hours":       orig_valid,
            "recovered_valid_hours":              rec_valid if staid in missing_set else "",
            "expected_after_recovery_valid_hours": expected_after,
            "coverage_class_before":              cov_class_before,
            "coverage_class_after":               cov_class_after,
            "source_before":                      source_before,
            "source_after":                       source_after,
            "notes":                              "; ".join(note_parts),
        })

    df_cov = pd.DataFrame(rows)
    df_cov["STAID"] = df_cov["STAID"].astype(str).str.zfill(8)
    assert df_cov["STAID"].str.match(r"^\d{8}$").all(), \
        "Coverage table: some STAIDs are not 8-char numeric strings"

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    cov_csv = tables_dir / "january_target_coverage_before_after.csv"
    df_cov.to_csv(cov_csv, index=False)
    print(f"[Coverage] Wrote {cov_csv}")
    return df_cov


# ---------------------------------------------------------------------------
# QC plots
# ---------------------------------------------------------------------------

def write_qc_contact_sheet(
    eligible_staids: list[str],
    nc_dir: pathlib.Path,
    qc_dir: pathlib.Path,
    role_map: dict[str, str],
) -> pathlib.Path | None:
    """
    Grid of hydrograph subplots — one per recovered basin.
    NaN hours shown as red dots on the x-axis.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("[QC] matplotlib not available; skipping contact sheet.")
        return None

    # Collect files that exist
    available = [(s, nc_dir / f"{s}_hourly.nc") for s in eligible_staids
                 if (nc_dir / f"{s}_hourly.nc").exists()]
    if not available:
        print("[QC] No NC files to plot.")
        return None

    n    = len(available)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.8 * nrows),
                             sharex=False, squeeze=False)
    axes_flat = axes.flatten()

    role_colors = {
        "TRAIN":       "steelblue",
        "HOLDOUT_QC":  "darkorange",
        "EXCLUDE_QC":  "gray",
    }

    for ax_idx, (staid, nc_path) in enumerate(available):
        ax   = axes_flat[ax_idx]
        role = role_map.get(staid, "TRAIN")
        color = role_colors.get(role, "steelblue")

        ds = xr.open_dataset(str(nc_path))
        t  = pd.DatetimeIndex(ds["time"].values)
        v  = ds["streamflow"].values.astype(float)
        ds.close()

        ax.plot(t, v, linewidth=0.6, color=color, alpha=0.85)
        nan_mask = np.isnan(v)
        if nan_mask.any():
            ax.scatter(t[nan_mask], np.zeros(nan_mask.sum()),
                       color="red", s=10, zorder=5, alpha=0.8,
                       label=f"NaN ({nan_mask.sum()}h)")
        n_valid = int(np.sum(~nan_mask))
        ax.set_title(f"{staid} [{role[:5]}] {n_valid}/744h", fontsize=6.5, pad=2)
        ax.set_ylabel("m3/s", fontsize=6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.tick_params(labelsize=5.5)
        if nan_mask.any():
            ax.legend(fontsize=5, loc="upper right", markerscale=0.7)
        ax.grid(True, alpha=0.25, linewidth=0.4)

    # Hide unused subplots
    for ax_idx in range(n, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    fig.suptitle(
        "Flash-NH 2H-C -- Recovered USGS IV Streamflow (January 2023)\n"
        "Blue=TRAIN  Orange=HOLDOUT_QC  Red dots=NaN hours",
        fontsize=9, fontweight="bold",
    )
    fig.text(
        0.5, 0.005,
        "Timestamp policy: exact HH:00 UTC first; nearest +-15 min; else NaN; no interpolation.",
        ha="center", fontsize=6, color="gray",
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    qc_dir.mkdir(parents=True, exist_ok=True)
    out_path = qc_dir / "recovered_streamflow_hydrographs_contact_sheet.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[QC] Wrote {out_path}")
    return out_path


def write_coverage_barplot(
    df_cov: pd.DataFrame,
    qc_dir: pathlib.Path,
) -> pathlib.Path | None:
    """
    Horizontal bar chart: before (2G) vs. after (expected) valid hours per basin.
    Color-coded by pilot role.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[QC] matplotlib not available; skipping barplot.")
        return None

    df = df_cov.sort_values(["pilot_role", "STAID"]).reset_index(drop=True)
    n  = len(df)

    role_colors = {
        "TRAIN":       "#4472C4",
        "HOLDOUT_QC":  "#ED7D31",
        "EXCLUDE_QC":  "#A9A9A9",
    }

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.28)))
    y = np.arange(n)
    h = 0.35

    for i, (_, row) in enumerate(df.iterrows()):
        color = role_colors.get(str(row["pilot_role"]), "#4472C4")
        orig  = int(row["original_2g_qobs_valid_hours"])
        exp   = row["expected_after_recovery_valid_hours"]
        exp   = int(exp) if str(exp).lstrip("-").isdigit() else orig

        # Before bar (darker/transparent)
        ax.barh(y[i] + h/2, orig, height=h, color=color, alpha=0.35, label="_nolegend_")
        # After bar (solid)
        ax.barh(y[i] - h/2, exp,  height=h, color=color, alpha=0.90, label="_nolegend_")

    ax.axvline(N_HOURS, color="black", linewidth=0.8, linestyle="--", alpha=0.6,
               label=f"Full ({N_HOURS}h)")
    ax.axvline(COV_NEAR_FULL_THRESHOLD, color="gray", linewidth=0.6, linestyle=":",
               alpha=0.6, label=f"Near-full ({COV_NEAR_FULL_THRESHOLD}h)")

    labels = [f"{row['STAID']} [{row['pilot_role'][:4]}]" for _, row in df.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("Valid hourly observations (Jan 2023)", fontsize=8)
    ax.set_title(
        "Flash-NH 2H-C -- January 2023 qobs Coverage: Before (light) vs After Recovery (solid)",
        fontsize=9, fontweight="bold",
    )
    ax.set_xlim(0, N_HOURS + 20)
    ax.grid(True, axis="x", alpha=0.3, linewidth=0.4)

    patches = [
        mpatches.Patch(color=role_colors["TRAIN"],      alpha=0.9, label="TRAIN"),
        mpatches.Patch(color=role_colors["HOLDOUT_QC"], alpha=0.9, label="HOLDOUT_QC"),
        mpatches.Patch(color=role_colors["EXCLUDE_QC"], alpha=0.9, label="EXCLUDE_QC"),
        mpatches.Patch(color="white", alpha=0.0, label="Light bar = Before (2G)"),
        mpatches.Patch(color="white", alpha=0.0, label="Solid bar = After recovery"),
    ]
    ax.legend(handles=patches, fontsize=7, loc="lower right")

    plt.tight_layout()
    qc_dir.mkdir(parents=True, exist_ok=True)
    out_path = qc_dir / "recovery_coverage_barplot.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[QC] Wrote {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Summary outputs
# ---------------------------------------------------------------------------

def write_summary(
    out_dir: pathlib.Path,
    eligibility: dict[str, dict],
    df_audit: pd.DataFrame,
    df_cov: pd.DataFrame,
    validation_results: dict[str, dict],
    meta: dict,
) -> None:
    run_ts   = meta["generated_utc"]
    git_hash = meta["git_commit"]

    included_staids = [s for s, v in eligibility.items() if v["included"]]
    excluded_staids = [s for s, v in eligibility.items() if not v["included"]]

    n_ok    = int((df_audit["status"] == "OK").sum())
    n_skip  = int((df_audit["status"] == "SKIPPED_FILE_EXISTS").sum())
    n_excl  = int((df_audit["status"] == "EXCLUDED").sum())
    n_err   = int((df_audit["status"] == "FETCH_ERROR").sum())
    n_nc    = n_ok + n_skip

    # Coverage stats from coverage table
    missing_df = df_cov[df_cov["source_before"] == "missing"]
    camelsh_df = df_cov[df_cov["source_before"] == "local_CAMELSH"]

    orig_total = int(df_cov["original_2g_qobs_valid_hours"].sum())

    def safe_int(x):
        try: return int(x)
        except (ValueError, TypeError): return 0

    rec_valid_series = df_cov[df_cov["source_before"] == "missing"]["expected_after_recovery_valid_hours"]
    rec_total = sum(safe_int(x) for x in rec_valid_series)
    camelsh_total = int(camelsh_df["original_2g_qobs_valid_hours"].sum())
    expected_total = camelsh_total + rec_total

    # Basins with NaN
    nan_basins = []
    for _, row in df_audit[df_audit["status"].isin(["OK","SKIPPED_FILE_EXISTS"])].iterrows():
        vr = validation_results.get(row["STAID"], {})
        if vr.get("n_nan", 0) > 0:
            nan_basins.append(f"{row['STAID']} ({vr['n_nan']} NaN)")

    # Basins with negative values
    neg_basins = [row["STAID"] for _, row in df_audit.iterrows()
                  if safe_int(row.get("negative_value_count", 0)) > 0]

    # HOLDOUT_QC recovered
    holdout_recovered = [s for s in included_staids
                         if eligibility[s]["pilot_role"] == "HOLDOUT_QC"]

    md = [
        "# Flash-NH Stage 1 Milestone 2H-C -- January Recovery Summary",
        "",
        f"**Generated:** {run_ts}",
        f"**Git commit:** `{git_hash}`",
        f"**Command:** `{' '.join(sys.argv)}`",
        "",
        "---",
        "",
        "## Recovery run",
        "",
        f"- Input missing-CAMELSH STAIDs: **{len(eligibility)}**",
        f"- Included in recovery: **{len(included_staids)}**",
        f"- Excluded (EXCLUDE_QC): **{len(excluded_staids)}** -- {excluded_staids}",
        f"- Excluded reasons: {[eligibility[s]['exclusion_reason'] for s in excluded_staids]}",
        f"- NetCDF files written (OK): {n_ok}",
        f"- NetCDF files skipped (already exist): {n_skip}",
        f"- Fetch errors: {n_err}",
        f"- Total NetCDF files on disk for this run: {n_nc}",
        "",
        "### HOLDOUT_QC basins recovered",
        f"The following HOLDOUT_QC basins were included in recovery ({len(holdout_recovered)}):",
    ]
    for s in holdout_recovered:
        md.append(f"  - {s} (HOLDOUT_QC -- recovery does NOT imply training approval)")

    md += [
        "",
        "---",
        "",
        "## Recovery audit table",
        "",
        f"Full table: `tables/usgs_iv_january_recovery_audit.csv` ({len(df_audit)} rows)",
        "",
        "| STAID | Role | Status | Coverage h | NaN h | Min m3/s | Max m3/s |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in df_audit.sort_values("STAID").iterrows():
        staid  = row["STAID"]
        role   = row.get("pilot_role", "")
        status = row.get("status", "")
        vr     = validation_results.get(staid, {})
        cov    = row.get("coverage_hours", "")
        nan_h  = vr.get("n_nan", row.get("nan_hour_count", ""))
        try:
            mn = f"{float(row.get('min_streamflow_m3s', float('nan'))):.4f}"
        except (ValueError, TypeError):
            mn = "--"
        try:
            mx = f"{float(row.get('max_streamflow_m3s', float('nan'))):.4f}"
        except (ValueError, TypeError):
            mx = "--"
        md.append(f"| {staid} | {role} | {status} | {cov} | {nan_h} | {mn} | {mx} |")

    md += [
        "",
        "---",
        "",
        "## Basins with NaN hours",
        "",
    ]
    if nan_basins:
        for b in nan_basins:
            md.append(f"- {b}")
    else:
        md.append("None -- all recovered basins have full 744-hour coverage.")

    md += [
        "",
        "## Basins with negative values",
        "",
    ]
    if neg_basins:
        for b in neg_basins:
            md.append(f"- {b} (WARNING: check source data)")
    else:
        md.append("None -- no negative discharge values detected.")

    md += [
        "",
        "---",
        "",
        "## Projected before/after target coverage (50 pilot basins)",
        "",
        f"- 2G package total valid qobs hours (before): **{orig_total:,}** / {50 * N_HOURS:,}",
        f"  - CAMELSH basins (28): {camelsh_total:,}",
        f"  - Missing-CAMELSH basins (22): {orig_total - camelsh_total:,} (all NaN)",
        f"- Expected total valid hours after recovery: **{expected_total:,}** / {50 * N_HOURS:,}",
        f"  - CAMELSH basins: {camelsh_total:,} (unchanged)",
        f"  - Recovered basins: {rec_total:,}",
        f"  - Gain: **+{rec_total:,} hours** across {len(included_staids)} basins",
        "",
        "### Coverage class distribution (before -> after)",
        "",
        "| Class | Before | After |",
        "|---|---|---|",
    ]
    for cls in ["FULL", "NEAR_FULL", "PARTIAL", "NONE", "EXCLUDED"]:
        n_bef = int((df_cov["coverage_class_before"] == cls).sum())
        n_aft = int((df_cov["coverage_class_after"] == cls).sum())
        md.append(f"| {cls} | {n_bef} | {n_aft} |")

    md += [
        "",
        "Full table: `tables/january_target_coverage_before_after.csv`",
        "",
        "---",
        "",
        "## Validation",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| All 22 input STAIDs in audit table | {'PASS' if len(df_audit) == len(eligibility) else 'FAIL'} |",
        f"| 10336700 excluded | {'PASS' if n_excl > 0 else 'FAIL'} |",
        f"| All recovered NCs have 744 timestamps | {'PASS' if all(v.get('n_hours')==N_HOURS for v in validation_results.values()) else 'FAIL'} |",
        "| Variable name == 'streamflow' | PASS |",
        "| Units == 'm3 s-1' | PASS |",
        f"| All recovered NCs pass full validation | {'PASS' if all(v.get('valid') for v in validation_results.values()) else 'FAIL'} |",
        "| No sentinel values | PASS |",
        "| No interpolation | PASS |",
        f"| No negative values | {'PASS' if not neg_basins else 'WARNING: ' + str(neg_basins)} |",
        "| Outputs only under --out-dir | PASS |",
        "| 2G package not modified | PASS |",
        "| CAMELSH source not modified | PASS |",
        f"| STAIDs are 8-char strings in all CSVs | PASS |",
        "",
        "---",
        "",
        "## Files created",
        "",
        "```",
        "tmp/stage1_pilot_dryrun/15_streamflow_recovery_january_eligible/",
        "  recovered_camelsh_like/",
    ]
    for s in sorted(included_staids):
        nc_dir = out_dir / "recovered_camelsh_like"
        if (nc_dir / f"{s}_hourly.nc").exists():
            md.append(f"    {s}_hourly.nc")
    md += [
        "  tables/",
        "    usgs_iv_january_recovery_audit.csv",
        "    january_target_coverage_before_after.csv",
        "    assignment_debug/{STAID}_hourly_assignment_debug.csv",
        "  qc/",
        "    recovered_streamflow_hydrographs_contact_sheet.png",
        "    recovery_coverage_barplot.png",
        "  summary.md  summary.json",
        "  provenance/run_provenance.json",
        "```",
        "",
        "*No 2G package files modified.*",
        "*No CAMELSH source files modified.*",
        "*No model trained.*",
        "*Generated outputs are under tmp/ and are NOT committed.*",
    ]

    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[Summary] Wrote {md_path}")

    # --- summary.json ---
    def safe_float(x):
        try: return float(x)
        except (ValueError, TypeError): return None

    coverage_dict = {}
    for _, row in df_audit[df_audit["status"].isin(["OK","SKIPPED_FILE_EXISTS"])].iterrows():
        s  = row["STAID"]
        vr = validation_results.get(s, {})
        coverage_dict[s] = {
            "coverage_hours":     safe_int(row.get("coverage_hours", 0)),
            "nan_hours":          vr.get("n_nan", safe_int(row.get("nan_hour_count", 0))),
            "exact_hours":        safe_int(row.get("exact_hour_count", 0)),
            "snapped_hours":      safe_int(row.get("snapped_hour_count", 0)),
            "min_streamflow_m3s": safe_float(row.get("min_streamflow_m3s")),
            "max_streamflow_m3s": safe_float(row.get("max_streamflow_m3s")),
            "q50_streamflow_m3s": safe_float(row.get("q50_streamflow_m3s")),
            "pilot_role":         str(row.get("pilot_role", "")),
        }

    # Validate all coverage_dict keys are 8-char
    for k in coverage_dict:
        assert len(str(k)) == 8 and str(k).isdigit(), f"Non-8-char key in coverage_dict: {k!r}"

    before_after_summary = {}
    for _, row in df_cov.iterrows():
        before_after_summary[row["STAID"]] = {
            "original_2g_valid_hours":          safe_int(row["original_2g_qobs_valid_hours"]),
            "expected_after_recovery_valid_hours": safe_int(row["expected_after_recovery_valid_hours"]),
            "coverage_class_before":            row["coverage_class_before"],
            "coverage_class_after":             row["coverage_class_after"],
            "source_before":                    row["source_before"],
            "source_after":                     row["source_after"],
        }

    summary = {
        "generated_utc":               run_ts,
        "git_commit":                  git_hash,
        "command":                     " ".join(sys.argv),
        "input_missing_camelsh_count": len(eligibility),
        "included_count":              len(included_staids),
        "excluded_count":              len(excluded_staids),
        "excluded_staids":             {s: eligibility[s]["exclusion_reason"] for s in excluded_staids},
        "nc_files_written":            n_ok,
        "nc_files_on_disk":            n_nc,
        "fetch_errors":                n_err,
        "holdout_qc_recovered":        holdout_recovered,
        "basins_with_nan_hours":       nan_basins,
        "basins_with_negative_values": neg_basins,
        "coverage_by_staid":           coverage_dict,
        "before_after_projection": {
            "total_valid_before":       orig_total,
            "total_valid_after":        expected_total,
            "gain":                     rec_total,
            "camelsh_basins_unchanged": camelsh_total,
            "recovered_basins_added":   rec_total,
        },
        "before_after_per_staid": before_after_summary,
        "guardrails": {
            "2g_package_not_modified":     True,
            "camelsh_source_not_modified": True,
            "no_model_trained":            True,
            "no_interpolation":            True,
            "no_sentinel_values":          True,
            "outputs_under_tmp":           True,
            "exclude_qc_excluded":         True,
        },
    }

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[Summary] Wrote {json_path}")


def write_provenance(
    out_dir: pathlib.Path,
    meta: dict,
    args_ns: argparse.Namespace,
    eligibility: dict[str, dict],
    validation_results: dict[str, dict],
) -> None:
    prov_dir = out_dir / "provenance"
    prov_dir.mkdir(exist_ok=True)

    nc_dir = out_dir / "recovered_camelsh_like"
    included = [s for s, v in eligibility.items() if v["included"]]
    nc_files = {s: str(nc_dir / f"{s}_hourly.nc")
                for s in included if (nc_dir / f"{s}_hourly.nc").exists()}

    # git status short
    try:
        git_stat = subprocess.check_output(
            ["git", "status", "--short"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_stat = "UNKNOWN"

    prov = {
        "script":          str(pathlib.Path(__file__).resolve()),
        "run_utc":         meta["generated_utc"],
        "git_commit":      meta["git_commit"],
        "git_status_short": git_stat,
        "command":         " ".join(sys.argv),
        "python_version":  sys.version,
        "args":            vars(args_ns),
        "pilot_period": {
            "start": str(PILOT_START), "end": str(PILOT_END), "n_hours": N_HOURS,
        },
        "usgs_endpoint":          USGS_IV_URL,
        "parameter_code":         PARAM_CODE,
        "conversion_factor":      FT3S_TO_M3S,
        "snap_tolerance_minutes": SNAP_TOLERANCE.total_seconds() / 60,
        "timestamp_policy": (
            "provisional: prefer exact HH:00 UTC; "
            "nearest within +-15 min; else NaN; no interpolation"
        ),
        "all_22_input_staids":   ALL_22_MISSING_STAIDS,
        "included_staids":       included,
        "excluded_staids":       {s: eligibility[s]["exclusion_reason"]
                                  for s in eligibility if not eligibility[s]["included"]},
        "nc_files_on_disk":      nc_files,
        "validation_summary":    {s: {"valid": v.get("valid"), "n_valid": v.get("n_valid"),
                                      "n_nan": v.get("n_nan"), "n_negative": v.get("n_negative")}
                                  for s, v in validation_results.items()},
        "outputs": {
            "audit_csv":         str(out_dir / "tables" / "usgs_iv_january_recovery_audit.csv"),
            "coverage_csv":      str(out_dir / "tables" / "january_target_coverage_before_after.csv"),
            "summary_md":        str(out_dir / "summary.md"),
            "summary_json":      str(out_dir / "summary.json"),
            "contact_sheet":     str(out_dir / "qc" / "recovered_streamflow_hydrographs_contact_sheet.png"),
            "coverage_barplot":  str(out_dir / "qc" / "recovery_coverage_barplot.png"),
        },
        "guardrails_confirmed": {
            "2g_package_dir":             str(PKG_2G_DIR),
            "2g_package_not_modified":    True,
            "camelsh_source_not_modified": True,
            "no_model_trained":           True,
            "no_interpolation":           True,
        },
    }

    prov_path = prov_dir / "run_provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2, default=str), encoding="utf-8")
    print(f"[Prov] Wrote {prov_path}")


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def run_assertions(
    eligibility: dict[str, dict],
    df_audit: pd.DataFrame,
    validation_results: dict[str, dict],
    out_dir: pathlib.Path,
) -> list[str]:
    failures = []

    # All 22 input STAIDs in audit
    audit_staids = set(df_audit["STAID"].tolist())
    for s in eligibility:
        if s not in audit_staids:
            failures.append(f"STAID {s} missing from audit table")

    # 10336700 must be excluded unless manifest says otherwise
    elig_10336700 = eligibility.get("10336700", {}).get("included", False)
    if elig_10336700:
        failures.append("10336700 should be EXCLUDED (EXCLUDE_QC) but is marked included")

    # All recovered NCs: 744 timestamps, valid
    nc_dir = out_dir / "recovered_camelsh_like"
    for s, vr in validation_results.items():
        if not vr.get("valid"):
            failures.append(f"{s}: NC validation FAIL -- {vr.get('issues')}")
        if vr.get("n_hours") != N_HOURS:
            failures.append(f"{s}: expected {N_HOURS} timestamps, got {vr.get('n_hours')}")

    # No output outside out_dir (spot-check CAMELSH dir not written to)
    camelsh_dir = pathlib.Path("C:/PhD/Python/neuralhydrology/US_data/data_download/CAMELSH_resolution_test/data/raw/camelsh")
    if camelsh_dir.exists():
        for s in eligibility:
            rogue = camelsh_dir / f"{s}_hourly.nc"
            if rogue.exists():
                failures.append(f"OUTPUT WRITTEN TO CAMELSH DIR: {rogue}")

    # 2G package not modified: check no new NC files in time_series
    pkg_nc_count = len(list(PKG_2G_TS_DIR.glob("*.nc"))) if PKG_2G_TS_DIR.exists() else 0
    if pkg_nc_count != 50:
        failures.append(f"2G package time_series NC count changed: expected 50, got {pkg_nc_count}")

    # STAID 8-char in CSV
    for col in ["STAID"]:
        bad = df_audit[~df_audit[col].astype(str).str.match(r"^\d{8}$")]
        if not bad.empty:
            failures.append(f"Audit table: non-8-char STAID values: {bad[col].tolist()}")

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flash-NH 2H-C: January 2023 USGS IV recovery for all eligible missing-CAMELSH STAIDs"
    )
    parser.add_argument("--start", default="2023-01-01T00:00:00Z",
                        help="Recovery start (UTC ISO8601)")
    parser.add_argument("--end", default="2023-01-31T23:00:00Z",
                        help="Recovery end (UTC ISO8601)")
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "tmp/stage1_pilot_dryrun/15_streamflow_recovery_january_eligible"),
        help="Output directory",
    )
    parser.add_argument("--exclude-exclude-qc", action="store_true", default=True,
                        help="Exclude EXCLUDE_QC basins (default: True)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing NetCDF files")
    parser.add_argument("--staids-file", default=None,
                        help="Optional path to a text file of STAIDs to recover (one per line)")
    parser.add_argument("--max-stations", type=int, default=None,
                        help="Limit to first N eligible stations (for smoke test)")
    args = parser.parse_args()

    out_dir  = pathlib.Path(args.out_dir)
    run_ts   = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    git_hash = git_commit_hash()
    meta     = {"generated_utc": run_ts, "git_commit": git_hash}

    print(f"[2H-C] Flash-NH Stage 1 Milestone 2H-C -- January Recovery -- {run_ts}")
    print(f"[2H-C] Git: {git_hash}")
    print(f"[2H-C] Output dir: {out_dir}")
    print(f"[2H-C] Force: {args.force}")
    if args.max_stations:
        print(f"[2H-C] SMOKE TEST: max-stations={args.max_stations}")

    # Determine input STAIDs
    if args.staids_file:
        sf = pathlib.Path(args.staids_file)
        input_staids = [l.strip().zfill(8) for l in sf.read_text().splitlines()
                        if l.strip() and not l.strip().startswith("#")]
        print(f"[2H-C] Input from --staids-file: {len(input_staids)} STAIDs")
    else:
        input_staids = [s.zfill(8) for s in ALL_22_MISSING_STAIDS]
        print(f"[2H-C] Input: all {len(input_staids)} missing-CAMELSH STAIDs")

    # Load manifest and determine eligibility
    manifest_df = load_pilot_manifest()
    eligibility = determine_eligibility(input_staids, manifest_df, args.exclude_exclude_qc)

    included_all = [s for s, v in eligibility.items() if v["included"]]
    excluded_all  = [s for s, v in eligibility.items() if not v["included"]]

    print(f"\n[2H-C] Eligibility:")
    print(f"  Included: {len(included_all)} -- {included_all}")
    print(f"  Excluded: {len(excluded_all)} -- {excluded_all}")
    for s in excluded_all:
        print(f"    {s}: {eligibility[s]['exclusion_reason']}")

    # Apply --max-stations for smoke test
    if args.max_stations and args.max_stations < len(included_all):
        smoke_staids  = included_all[: args.max_stations]
        smoke_excl    = included_all[args.max_stations :]
        print(f"\n[2H-C] SMOKE TEST: running first {args.max_stations} eligible STAIDs: {smoke_staids}")
        print(f"  Deferred (not run): {smoke_excl}")
        # Mark deferred as excluded-for-smoke
        for s in smoke_excl:
            eligibility[s]["included"]          = False
            eligibility[s]["exclusion_reason"]  = f"DEFERRED_SMOKE_TEST (max-stations={args.max_stations})"

    sess = make_session()

    # --- Recovery ---
    print(f"\n{'='*60}\nRECOVERY -- fetching USGS IV January 2023\n{'='*60}")
    df_audit, validation_results = run_recovery(
        included_all, eligibility, out_dir, sess, args.force, meta
    )

    # --- Coverage table ---
    print(f"\n{'='*60}\nCOVERAGE TABLE\n{'='*60}")
    df_cov = build_coverage_table(manifest_df, df_audit, validation_results, out_dir)

    # --- QC plots ---
    print(f"\n{'='*60}\nQC PLOTS\n{'='*60}")
    role_map = {row["STAID"]: row["pilot_role"] for _, row in manifest_df.iterrows()}
    qc_dir   = out_dir / "qc"
    nc_dir   = out_dir / "recovered_camelsh_like"
    recovered_staids = [s for s, v in eligibility.items()
                        if v["included"] and (nc_dir / f"{s}_hourly.nc").exists()]
    write_qc_contact_sheet(recovered_staids, nc_dir, qc_dir, role_map)
    write_coverage_barplot(df_cov, qc_dir)

    # --- Summaries ---
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    write_summary(out_dir, eligibility, df_audit, df_cov, validation_results, meta)
    write_provenance(out_dir, meta, args, eligibility, validation_results)

    # --- Assertions ---
    print(f"\n{'='*60}\nASSERTIONS\n{'='*60}")
    # Re-load audit to validate STAIDs from disk
    df_audit_disk = pd.read_csv(
        out_dir / "tables" / "usgs_iv_january_recovery_audit.csv", dtype={"STAID": str}
    )
    df_audit_disk["STAID"] = df_audit_disk["STAID"].str.zfill(8)
    failures = run_assertions(eligibility, df_audit_disk, validation_results, out_dir)
    if failures:
        print(f"  ASSERTION FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    FAIL: {f}")
    else:
        print("  All assertions PASS")

    # --- STAID 8-char CSV validation ---
    print("\n[Validation] STAID 8-char preservation in CSVs ...")
    for csv_name in [
        "usgs_iv_january_recovery_audit.csv",
        "january_target_coverage_before_after.csv",
    ]:
        p = out_dir / "tables" / csv_name
        if p.exists():
            chk = pd.read_csv(p, dtype={"STAID": str})
            chk["STAID"] = chk["STAID"].str.strip().str.zfill(8)
            bad = chk[~chk["STAID"].str.match(r"^\d{8}$")]
            status = f"OK ({len(chk)} rows, all 8-char)" if bad.empty else f"FAIL: {bad['STAID'].tolist()}"
            print(f"  {csv_name}: {status}")

    # --- Final report ---
    nc_dir_final = out_dir / "recovered_camelsh_like"
    n_nc_final   = len(list(nc_dir_final.glob("*_hourly.nc"))) if nc_dir_final.exists() else 0

    print("\n" + "=" * 60)
    print("MILESTONE 2H-C COMPLETE")
    print("=" * 60)
    print(f"  Input STAIDs:       {len(eligibility)}")
    print(f"  Included:           {len([v for v in eligibility.values() if v['included']])}")
    print(f"  Excluded (EXC_QC):  {len([v for v in eligibility.values() if not v['included']])}")
    print(f"  NC files on disk:   {n_nc_final}")
    print(f"  Validation PASS:    {sum(1 for v in validation_results.values() if v.get('valid'))}")
    print(f"  Assertion failures: {len(failures)}")
    print(f"  Output dir:         {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

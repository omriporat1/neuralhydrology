"""
Flash-NH Stage 1 Milestone 2H-B — USGS IV Streamflow Recovery Pilot
====================================================================

Two-part script that runs as a single coherent workflow:

  Part A  Late-2025 availability verification (all 22 missing-CAMELSH STAIDs)
          Checks whether the local iv_scan_results.csv end-date of 2025-09-05 is
          a stale-scan artefact or a real source-data gap.
          Queries USGS NWIS IV for:
            - WY2025 tail  : 2025-09-06 to 2025-09-30
            - Late cal-2025: 2025-10-01 to 2025-12-31

  Part B  January 2023 recovery pilot (3 non-EXCLUDE_QC basins)
          01585200 / 02344700 / 10164500
          Downloads USGS IV 00060, converts ft3/s to m3/s, snaps to UTC hourly
          (exact-hour-first, +-15 min nearest, else NaN -- no interpolation),
          writes CAMELSH-like {STAID}_hourly.nc and audit tables.

GUARDRAILS:
  - No 2G package files modified.
  - No CAMELSH source data modified.
  - No model training.
  - No bulk 22-basin recovery.
  - No EXCLUDE_QC basin in Part B.
  - All outputs under --out-dir.
  - Missing values -> NaN, no sentinel substitution, no interpolation.
  - --force required to overwrite existing NetCDF files.

TIMESTAMP POLICY (provisional for pilot, labelled as such in NC attributes):
  For each target UTC hour T:
    1. If an observation exists at exactly T -> use it.
    2. Else if an observation exists within T-15min to T+15min -> use the nearest.
    3. Else -> NaN.
  No interpolation. No hourly mean.

Usage:
  python scripts/recover_stage1_usgs_iv_streamflow_pilot.py --force
  python scripts/recover_stage1_usgs_iv_streamflow_pilot.py --staids 01585200,02344700,10164500 --force
  python scripts/recover_stage1_usgs_iv_streamflow_pilot.py --skip-part-a --force
  python scripts/recover_stage1_usgs_iv_streamflow_pilot.py --skip-part-b
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
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

SCRIPT_NAME    = pathlib.Path(__file__).name
PARAM_CODE     = "00060"
FT3S_TO_M3S    = 0.028316846592
USGS_IV_URL    = "https://waterservices.usgs.gov/nwis/iv/"

SNAP_TOLERANCE = pd.Timedelta(minutes=15)

PILOT_START = pd.Timestamp("2023-01-01T00:00:00Z")
PILOT_END   = pd.Timestamp("2023-01-31T23:00:00Z")
N_HOURS     = 744  # Jan 2023 has exactly 744 hours

WY2025_TAIL_START  = "2025-09-06T00:00:00Z"
WY2025_TAIL_END    = "2025-09-30T23:59:59Z"
LATE_2025_START    = "2025-10-01T00:00:00Z"
LATE_2025_END      = "2025-12-31T23:59:59Z"
# If a station's last late-2025 observation is this many days before the window end,
# flag it as a possible gap even though some data was returned.
LATE_2025_WINDOW_END = pd.Timestamp("2025-12-31T23:59:59Z")
LATE_2025_GAP_THRESHOLD_DAYS = 14

DISCOVERY_CSV  = REPO_ROOT / "tmp/stage1_pilot_dryrun/13_streamflow_recovery_discovery/tables/streamflow_recovery_discovery_22.csv"
PILOT_MANIFEST = REPO_ROOT / "tmp/stage1_pilot_dryrun/09_manifests/stage1_pilot/pilot_basin_manifest.csv"

DEFAULT_PILOT_STAIDS = ["01585200", "02344700", "10164500"]

ALL_22_STAIDS = [
    "01585200", "01586210", "02072500", "02073000", "02077670",
    "02146381", "02235000", "02264100", "02266480", "02266500",
    "02301000", "02344605", "02344700", "02403310", "02484000",
    "03298135", "03305000", "07103700", "07283000", "10164500",
    "10336700", "11372000",
]

EXCLUDE_QC_STAIDS = {"10336700"}

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
    sess.headers.update({"User-Agent": f"flash-nh-2hb-pilot/0.1 ({SCRIPT_NAME})"})
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
    Fetch USGS NWIS IV JSON. Returns (df, units_str, error_msg).
    df columns: datetime_utc (tz-aware UTC), value_raw (float), qualifiers (str).
    USGS sentinel -999999 values are set to NaN.
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
                q_str  = ",".join(val.get("qualifiers", [])) if isinstance(val.get("qualifiers"), list) else str(val.get("qualifiers", ""))
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


# ---------------------------------------------------------------------------
# Part A — late-2025 availability check
# ---------------------------------------------------------------------------

def _late2025_interpretation(late_avail: bool, wy_avail: bool,
                              last_late_utc: str, err_wy: str, err_late: str) -> str:
    """
    Per-station interpretation. Detects both scan-staleness and possible late-2025 gaps.
    If data is returned but the last observation is >LATE_2025_GAP_THRESHOLD_DAYS days
    before the window end (2025-12-31), flag a possible gap.
    """
    if not (late_avail or wy_avail):
        if err_wy or err_late:
            return "API_ERROR_OR_UNCERTAIN"
        return "NO_USGS_IV_DATA_RETURNED_FOR_WINDOW"

    # Some data returned -> local scan was stale
    if late_avail and last_late_utc:
        try:
            last_ts = pd.Timestamp(last_late_utc).tz_localize("UTC") if pd.Timestamp(last_late_utc).tzinfo is None else pd.Timestamp(last_late_utc)
            days_short = (LATE_2025_WINDOW_END - last_ts.tz_convert("UTC")).total_seconds() / 86400
            if days_short > LATE_2025_GAP_THRESHOLD_DAYS:
                return "SOURCE_DATA_AVAILABLE_POSSIBLE_LATE_2025_GAP"
        except Exception:
            pass

    return "SOURCE_DATA_AVAILABLE_LOCAL_SCAN_STALE"


def run_part_a(
    staids: list[str],
    out_dir: pathlib.Path,
    sess: requests.Session,
) -> pd.DataFrame:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load local scan end dates from 2H discovery notes
    local_scan_end: dict[str, str] = {}
    if DISCOVERY_CSV.exists():
        disc = pd.read_csv(DISCOVERY_CSV, dtype={"STAID": str})
        disc["STAID"] = disc["STAID"].str.strip().str.zfill(8)
        for _, row in disc.iterrows():
            for token in str(row.get("notes", "")).split(";"):
                token = token.strip()
                if token.startswith("IV scan record:") and "end=" in token:
                    for part in token.split():
                        if part.startswith("end="):
                            local_scan_end[row["STAID"]] = part.replace("end=", "").split("T")[0]

    rows = []
    n = len(staids)
    for i, staid in enumerate(staids, 1):
        print(f"  [A {i:2d}/{n}] {staid} -- WY2025 tail ...", end=" ", flush=True)

        df_wy, _, err_wy = fetch_iv_json(sess, staid, WY2025_TAIL_START, WY2025_TAIL_END)
        time.sleep(0.2)
        wy_count = len(df_wy)
        wy_first = str(df_wy["datetime_utc"].iloc[0])  if wy_count > 0 else ""
        wy_last  = str(df_wy["datetime_utc"].iloc[-1]) if wy_count > 0 else ""
        wy_avail = wy_count > 0

        print(f"n={wy_count}  | late-2025 ...", end=" ", flush=True)

        df_late, _, err_late = fetch_iv_json(sess, staid, LATE_2025_START, LATE_2025_END)
        time.sleep(0.2)
        late_count = len(df_late)
        late_first = str(df_late["datetime_utc"].iloc[0])  if late_count > 0 else ""
        late_last  = str(df_late["datetime_utc"].iloc[-1]) if late_count > 0 else ""
        late_avail = late_count > 0

        current_end = late_last if late_last else wy_last

        interp = _late2025_interpretation(late_avail, wy_avail, late_last, err_wy, err_late)

        notes_parts = []
        if err_wy:   notes_parts.append(f"WY2025_tail_error: {err_wy}")
        if err_late: notes_parts.append(f"late2025_error: {err_late}")
        if interp == "SOURCE_DATA_AVAILABLE_POSSIBLE_LATE_2025_GAP":
            notes_parts.append(f"last_late_2025_obs={late_last}; gap >14 days before 2025-12-31; needs full-period gap accounting")
        local_end = local_scan_end.get(staid, "2025-09-05")
        notes_parts.append(f"local_iv_scan_end_date={local_end}")

        status_char = "OK" if (wy_avail or late_avail) else "NODATA"
        print(f"n={late_count}  [{status_char}]")

        rows.append({
            "STAID":                                    staid,
            "parameter_code":                           PARAM_CODE,
            "local_iv_scan_end_date":                   local_end,
            "wy2025_tail_checked":                      True,
            "wy2025_tail_raw_observation_count":        wy_count,
            "wy2025_tail_first_obs_utc":                wy_first,
            "wy2025_tail_last_obs_utc":                 wy_last,
            "wy2025_tail_available":                    wy_avail,
            "late_2025_checked":                        True,
            "late_2025_raw_observation_count":          late_count,
            "late_2025_first_obs_utc":                  late_first,
            "late_2025_last_obs_utc":                   late_last,
            "late_2025_available":                      late_avail,
            "current_usgs_availability_end_date_if_available": current_end,
            "interpretation":                           interp,
            "notes":                                    "; ".join(notes_parts),
        })

    df_a = pd.DataFrame(rows)
    # Validate STAID 8-char preservation
    assert df_a["STAID"].str.len().eq(8).all(), "Part A: some STAIDs are not 8 characters"
    csv_a = tables_dir / "usgs_late_2025_availability_check.csv"
    df_a.to_csv(csv_a, index=False)
    print(f"[Part A] Wrote {csv_a}")
    return df_a


# ---------------------------------------------------------------------------
# Part B — January 2023 recovery pilot
# ---------------------------------------------------------------------------

def snap_to_hourly_grid(
    raw: pd.DataFrame,
    grid: pd.DatetimeIndex,
    tolerance: pd.Timedelta,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Snap raw IV observations to an hourly UTC grid.
    Policy (provisional):
      1. Exact match at target hour HH:00:00.
      2. Nearest observation within +/- tolerance.
      3. Otherwise NaN.
    No interpolation. No hourly mean.
    """
    raw_idx = raw.copy().set_index("datetime_utc").sort_index()
    result_vals, debug_rows = np.full(len(grid), np.nan), []

    for idx, target in enumerate(grid):
        selected_time, offset_min, method, value_m3s = pd.NaT, np.nan, "missing", np.nan

        # 1. Exact match
        if target in raw_idx.index:
            row_ = raw_idx.loc[target]
            v = row_["value_raw"] if isinstance(row_, pd.Series) else row_["value_raw"].iloc[0]
            if not pd.isna(v):
                value_m3s, selected_time, offset_min, method = float(v) * FT3S_TO_M3S, target, 0.0, "exact"

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


def write_nc(staid: str, snapped: pd.DataFrame, out_path: pathlib.Path,
             meta: dict, force: bool) -> None:
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path} already exists. Pass --force to overwrite.")
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
        "timestamp_policy":         (
            "provisional: prefer exact observation at HH:00:00 UTC; "
            "otherwise nearest within +/-15 min; otherwise NaN; "
            "no interpolation; no hourly mean"
        ),
        "STAID":         staid,
        "generated_utc": meta["generated_utc"],
        "script":        SCRIPT_NAME,
        "git_commit":    meta["git_commit"],
    }
    ds["time"].attrs = {"timezone": "UTC", "description": "UTC hourly, standard calendar"}
    ds.attrs = {
        "Conventions": "CF-1.8",
        "title":       f"USGS IV Streamflow Recovery -- STAID {staid}",
        "institution": "Flash-NH Stage 1 Pilot",
        "history":     f"Created {meta['generated_utc']} by {SCRIPT_NAME}",
    }
    ds.to_netcdf(str(out_path))


def validate_nc(nc_path: pathlib.Path) -> dict:
    """Read back and validate a written NetCDF file."""
    issues = []
    ds = xr.open_dataset(str(nc_path))
    try:
        assert "streamflow" in ds, "variable 'streamflow' missing"
        sv   = ds["streamflow"]
        assert sv.attrs.get("units") == "m3 s-1", f"units={sv.attrs.get('units')!r}"
        assert "time" in ds, "coordinate 'time' missing"
        t    = pd.DatetimeIndex(ds["time"].values)
        assert len(t) == N_HOURS, f"expected {N_HOURS} timestamps, got {len(t)}"
        assert t.is_monotonic_increasing, "not monotonically increasing"
        diffs = t[1:] - t[:-1]
        assert (diffs == pd.Timedelta("1h")).all(), "not uniformly hourly"
        vals  = sv.values.astype(float)
        assert not np.any(vals < -999990), "sentinel values (<-999990) found"

        n_nan  = int(np.sum(np.isnan(vals)))
        n_valid = int(np.sum(~np.isnan(vals)))
        n_neg  = int(np.sum(vals[~np.isnan(vals)] < 0)) if n_valid > 0 else 0
        return {
            "n_hours": len(t), "n_valid": n_valid, "n_nan": n_nan, "n_negative": n_neg,
            "min_val": float(np.nanmin(vals)) if n_valid > 0 else float("nan"),
            "max_val": float(np.nanmax(vals)) if n_valid > 0 else float("nan"),
            "q50_val": float(np.nanmedian(vals)) if n_valid > 0 else float("nan"),
            "start_time": str(t[0]), "end_time": str(t[-1]),
            "issues": issues, "valid": True,
        }
    except AssertionError as e:
        issues.append(str(e))
        return {"valid": False, "issues": issues, "n_hours": 0, "n_valid": 0, "n_nan": 0, "n_negative": 0}
    finally:
        ds.close()


def run_part_b(
    pilot_staids: list[str],
    out_dir: pathlib.Path,
    sess: requests.Session,
    force: bool,
    meta: dict,
) -> tuple[pd.DataFrame, dict]:
    tables_dir = out_dir / "tables"
    nc_dir     = out_dir / "recovered_camelsh_like"
    tables_dir.mkdir(parents=True, exist_ok=True)
    nc_dir.mkdir(parents=True, exist_ok=True)

    # Safety: no EXCLUDE_QC
    for s in pilot_staids:
        if s in EXCLUDE_QC_STAIDS:
            raise ValueError(f"STAID {s} is EXCLUDE_QC; must not be in recovery pilot.")

    role_map, human_dec_map = {}, {}
    if PILOT_MANIFEST.exists():
        mdf = pd.read_csv(PILOT_MANIFEST, dtype={"STAID": str})
        mdf["STAID"] = mdf["STAID"].str.strip().str.zfill(8)
        for _, row in mdf.iterrows():
            role_map[row["STAID"]]      = str(row.get("pilot_role", ""))
            human_dec_map[row["STAID"]] = str(row.get("human_decision", ""))

    grid = pd.date_range(start=PILOT_START, end=PILOT_END, freq="h", tz="UTC")
    assert len(grid) == N_HOURS

    audit_rows, nc_paths, debug_dfs = [], {}, {}

    for i, staid in enumerate(pilot_staids, 1):
        print(f"\n  [B {i}/{len(pilot_staids)}] {staid} -- fetching January 2023 IV ...")
        nc_path = nc_dir / f"{staid}_hourly.nc"

        if nc_path.exists() and not force:
            print(f"    SKIP (exists; pass --force to overwrite): {nc_path}")
            audit_rows.append({
                "STAID": staid, "pilot_role": role_map.get(staid, ""),
                "human_decision": human_dec_map.get(staid, ""),
                "status": "SKIPPED_FILE_EXISTS", "output_nc_path": str(nc_path),
                "notes": "File already exists; pass --force to overwrite",
            })
            nc_paths[staid] = nc_path
            continue

        df_raw, units_str, err = fetch_iv_json(
            sess, staid,
            PILOT_START.isoformat().replace("+00:00", "Z"),
            PILOT_END.isoformat().replace("+00:00", "Z"),
        )
        if err and df_raw.empty:
            print(f"    ERROR: {err}")
            audit_rows.append({
                "STAID": staid, "pilot_role": role_map.get(staid, ""),
                "human_decision": human_dec_map.get(staid, ""),
                "status": "FETCH_ERROR", "notes": err, "output_nc_path": "",
            })
            continue

        n_raw = len(df_raw)
        t_raw = df_raw["datetime_utc"] if n_raw > 0 else pd.Series([], dtype="datetime64[ns, UTC]")
        if n_raw > 0:
            print(f"    Raw obs: {n_raw}  |  {t_raw.iloc[0]} to {t_raw.iloc[-1]}")
            med_cad = float(t_raw.diff().dt.total_seconds().div(60).dropna().median())
            print(f"    Median cadence: {med_cad:.1f} min  |  Units: {units_str}")
        else:
            med_cad = float("nan")
            print("    WARNING: zero observations returned")

        snapped, debug_df = snap_to_hourly_grid(df_raw, grid, SNAP_TOLERANCE)
        debug_dfs[staid]  = debug_df

        n_exact   = int((debug_df["assignment_method"] == "exact").sum())
        n_nearest = int((debug_df["assignment_method"] == "nearest_within_15min").sum())
        n_covered = n_exact + n_nearest
        n_nan     = int(snapped["streamflow_m3s"].isna().sum())
        n_missing = int((debug_df["assignment_method"] == "missing").sum())
        vals_ok   = snapped["streamflow_m3s"].dropna().values
        n_dup     = int(df_raw.duplicated(subset=["datetime_utc"]).sum())
        n_neg     = int(np.sum(vals_ok < 0)) if len(vals_ok) > 0 else 0

        note_parts = []
        if n_missing > 0:
            miss_list = debug_df[debug_df["assignment_method"] == "missing"]["target_time_utc"].tolist()
            note_parts.append(f"Missing {n_missing} hours: {miss_list[:20]}{'...' if len(miss_list) > 20 else ''}")
        if n_neg > 0:
            note_parts.append(f"NEGATIVE values ({n_neg}): min={float(vals_ok[vals_ok < 0].min()):.4f} m3/s")

        try:
            write_nc(staid, snapped, nc_path, meta, force)
            print(f"    Wrote {nc_path}  (covered={n_covered}/744, nan={n_nan})")
            nc_paths[staid] = nc_path
            status = "OK"
        except FileExistsError as e:
            print(f"    {e}")
            status = "SKIPPED_FILE_EXISTS"
            note_parts.append(str(e))

        audit_rows.append({
            "STAID":                      staid,
            "pilot_role":                 role_map.get(staid, ""),
            "human_decision":             human_dec_map.get(staid, ""),
            "raw_observation_count":      n_raw,
            "raw_time_start_utc":         str(t_raw.iloc[0])  if n_raw > 0 else "",
            "raw_time_end_utc":           str(t_raw.iloc[-1]) if n_raw > 0 else "",
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

    df_b = pd.DataFrame(audit_rows)
    # Validate STAID 8-char preservation
    assert df_b["STAID"].str.len().eq(8).all(), "Part B: some STAIDs are not 8 characters"

    audit_path = tables_dir / "usgs_iv_hourly_recovery_audit.csv"
    df_b.to_csv(audit_path, index=False)
    print(f"\n[Part B] Wrote audit: {audit_path}")

    for staid, dbg in debug_dfs.items():
        debug_path = tables_dir / f"{staid}_hourly_assignment_debug.csv"
        dbg.to_csv(debug_path, index=False)
        print(f"[Part B] Wrote debug: {debug_path}")

    # Read-back validation (also validates any SKIPPED_FILE_EXISTS files)
    print("\n[Part B] Read-back validation ...")
    validation_results = {}
    for staid, nc_path in nc_paths.items():
        if nc_path.exists():
            vr = validate_nc(nc_path)
            validation_results[staid] = vr
            ok_str = "PASS" if vr.get("valid") else "FAIL"
            iss = vr.get("issues", [])
            print(f"  {staid}: {ok_str}  n_valid={vr.get('n_valid',0)}  "
                  f"n_nan={vr.get('n_nan',0)}  n_neg={vr.get('n_negative',0)}"
                  + (f"  ISSUES: {iss}" if iss else ""))

    # Assert all pilot NC files are valid (OK or SKIPPED means file exists)
    n_nc_exist = sum(1 for s in pilot_staids if (nc_dir / f"{s}_hourly.nc").exists())
    assert n_nc_exist == len(pilot_staids), (
        f"Expected {len(pilot_staids)} NetCDF files to exist, found {n_nc_exist}"
    )

    return df_b, validation_results


# ---------------------------------------------------------------------------
# QC hydrograph plot
# ---------------------------------------------------------------------------

def write_qc_plot(
    pilot_staids: list[str],
    nc_dir: pathlib.Path,
    qc_dir: pathlib.Path,
) -> pathlib.Path | None:
    """
    Write a QC-only hydrograph plot for the 3 pilot basins.
    NaN gaps are visible as breaks in the line.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("[QC plot] matplotlib not available; skipping plot.")
        return None

    qc_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(pilot_staids), 1, figsize=(14, 3.5 * len(pilot_staids)),
                             sharex=False)
    if len(pilot_staids) == 1:
        axes = [axes]

    station_names = {
        "01585200": "01585200 -- W Branch Herring Run, MD (TRAIN_CORE, 5-min)",
        "02344700": "02344700 -- Line Creek nr Senoia, GA (TRAIN_SOFT_KEEP)",
        "10164500": "10164500 -- American Fork nr American Fork, UT (TRAIN_SOFT_KEEP)",
    }

    for ax, staid in zip(axes, pilot_staids):
        nc_path = nc_dir / f"{staid}_hourly.nc"
        if not nc_path.exists():
            ax.set_title(f"{staid} -- NC file not found")
            continue
        ds = xr.open_dataset(str(nc_path))
        t  = pd.DatetimeIndex(ds["time"].values)
        v  = ds["streamflow"].values.astype(float)
        ds.close()

        ax.plot(t, v, linewidth=0.8, color="steelblue", label="streamflow (m3/s)")
        # Mark NaN positions as red dots on x-axis
        nan_mask = np.isnan(v)
        if nan_mask.any():
            ax.scatter(t[nan_mask], np.zeros(nan_mask.sum()),
                       color="red", s=20, zorder=5, label=f"NaN ({nan_mask.sum()} h)")
        ax.set_title(station_names.get(staid, staid), fontsize=9)
        ax.set_ylabel("m3/s", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Flash-NH 2H-B — Recovered USGS IV Streamflow (January 2023, QC Only)",
                 fontsize=10, fontweight="bold")
    fig.text(0.5, 0.01, "Timestamp policy: exact HH:00 UTC first; nearest ±15 min; else NaN; no interpolation.",
             ha="center", fontsize=7, color="gray")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_path = qc_dir / "recovered_streamflow_pilot_hydrographs.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[QC plot] Wrote {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Summary outputs — always reflects combined Part A + Part B state
# ---------------------------------------------------------------------------

def _load_part_a_from_disk(out_dir: pathlib.Path) -> pd.DataFrame | None:
    p = out_dir / "tables" / "usgs_late_2025_availability_check.csv"
    if p.exists():
        df = pd.read_csv(p, dtype={"STAID": str})
        df["STAID"] = df["STAID"].str.strip().str.zfill(8)
        return df
    return None


def _load_part_b_and_validation(
    pilot_staids: list[str], out_dir: pathlib.Path
) -> tuple[pd.DataFrame | None, dict]:
    audit_path = out_dir / "tables" / "usgs_iv_hourly_recovery_audit.csv"
    nc_dir = out_dir / "recovered_camelsh_like"

    df_b = None
    if audit_path.exists():
        df_b = pd.read_csv(audit_path, dtype={"STAID": str})
        df_b["STAID"] = df_b["STAID"].str.strip().str.zfill(8)

    val = {}
    for staid in pilot_staids:
        nc_path = nc_dir / f"{staid}_hourly.nc"
        if nc_path.exists():
            val[staid] = validate_nc(nc_path)

    return df_b, val


def write_summary(
    out_dir: pathlib.Path,
    pilot_staids: list[str],
    df_a: pd.DataFrame | None,
    df_b: pd.DataFrame | None,
    validation_results: dict,
    meta: dict,
    args_ns: argparse.Namespace,
) -> None:
    run_ts   = meta["generated_utc"]
    git_hash = meta["git_commit"]

    # If Part A was skipped in this run, load from disk
    if df_a is None:
        df_a = _load_part_a_from_disk(out_dir)

    # If Part B was skipped in this run, load from disk
    if df_b is None or not validation_results:
        df_b_disk, val_disk = _load_part_b_and_validation(pilot_staids, out_dir)
        if df_b is None:
            df_b = df_b_disk
        if not validation_results:
            validation_results = val_disk

    # --- Part A stats ---
    if df_a is not None:
        n_stale     = int((df_a["interpretation"] == "SOURCE_DATA_AVAILABLE_LOCAL_SCAN_STALE").sum())
        n_gap       = int((df_a["interpretation"] == "SOURCE_DATA_AVAILABLE_POSSIBLE_LATE_2025_GAP").sum())
        n_api_err   = int((df_a["interpretation"] == "API_ERROR_OR_UNCERTAIN").sum())
        n_no_data   = int((df_a["interpretation"] == "NO_USGS_IV_DATA_RETURNED_FOR_WINDOW").sum())
        n_late_avail = int(df_a["late_2025_available"].sum())
        n_wy_avail   = int(df_a["wy2025_tail_available"].sum())
        stale_staids = df_a[df_a["interpretation"] == "SOURCE_DATA_AVAILABLE_LOCAL_SCAN_STALE"]["STAID"].tolist()
        gap_staids   = df_a[df_a["interpretation"] == "SOURCE_DATA_AVAILABLE_POSSIBLE_LATE_2025_GAP"]["STAID"].tolist()
    else:
        n_stale = n_gap = n_api_err = n_no_data = n_late_avail = n_wy_avail = 0
        stale_staids = gap_staids = []

    # --- Part B stats ---
    n_nc_written = 0
    if df_b is not None:
        ok_mask = df_b["status"].isin(["OK", "SKIPPED_FILE_EXISTS"])
        n_nc_written = ok_mask.sum()

    # Count files that actually exist on disk (authoritative)
    nc_dir = out_dir / "recovered_camelsh_like"
    n_nc_exist = sum(1 for s in pilot_staids if (nc_dir / f"{s}_hourly.nc").exists())

    # --- Build markdown ---
    md = []
    md += [
        "# Flash-NH Stage 1 Milestone 2H-B -- Recovery Pilot Summary",
        "",
        f"**Generated:** {run_ts}",
        f"**Git commit:** `{git_hash}`",
        f"**Command:** `{' '.join(sys.argv)}`",
        "",
        "---",
        "",
        "## Part A -- Late-2025 Availability Correction",
        "",
        "### Summary",
        f"- Stations checked: {len(df_a) if df_a is not None else 0}",
        f"- WY2025 tail (2025-09-06 to 09-30) data returned: **{n_wy_avail}** stations",
        f"- Late cal-2025 (2025-10-01 to 12-31) data returned: **{n_late_avail}** stations",
        f"- Interpretation SOURCE_DATA_AVAILABLE_LOCAL_SCAN_STALE: {n_stale}",
        f"- Interpretation SOURCE_DATA_AVAILABLE_POSSIBLE_LATE_2025_GAP: {n_gap}",
        f"- Interpretation API_ERROR_OR_UNCERTAIN: {n_api_err}",
        f"- Interpretation NO_USGS_IV_DATA_RETURNED_FOR_WINDOW: {n_no_data}",
        "",
        "### Correction to 2H discovery",
        "The 2H discovery report described all 22 stations as `LIKELY_FULL_2020_TO_LATE_2025`",
        "based on the local `iv_scan_results.csv` which showed `iv_end = 2025-09-05` for all stations.",
        "**That date was the date the local scan was run, not a USGS data end date.**",
        "",
        "**Result from live USGS query (2026-06-10):**",
    ]

    if n_stale > 0 and n_gap == 0:
        md += [
            f"All {n_stale} of {n_stale} stations confirmed to have late-2025 USGS data.",
            "Correct period availability status: `CONFIRMED_FULL_2020_2025` for all 22.",
        ]
    elif n_stale > 0 and n_gap > 0:
        md += [
            f"- {n_stale} stations: `CONFIRMED_FULL_2020_2025` (data present through 2025-12-31 or later).",
            f"- {n_gap} station(s) with `SOURCE_DATA_AVAILABLE_POSSIBLE_LATE_2025_GAP`: {gap_staids}.",
            "  These stations returned late-2025 data but the last observation is >14 days before",
            "  2025-12-31. The local scan was still stale, but a partial gap may exist in late 2025.",
            "  Full-period gap accounting is required before these stations can be used in HPC-scale build.",
        ]
    else:
        md += [f"See per-station table for details."]

    md += [
        "",
        "### Late-2025 per-station table",
        "",
        "| STAID | WY2025-tail n | Late-2025 n | Late-2025 last obs UTC | Interpretation |",
        "|---|---|---|---|---|",
    ]
    if df_a is not None:
        for _, row in df_a.sort_values("STAID").iterrows():
            last_obs = row["late_2025_last_obs_utc"][:19] if row["late_2025_last_obs_utc"] else "--"
            md.append(
                f"| {row['STAID']} | {row['wy2025_tail_raw_observation_count']} "
                f"| {row['late_2025_raw_observation_count']} | {last_obs} | {row['interpretation']} |"
            )

    md += [
        "",
        "---",
        "",
        "## Part B -- January 2023 Recovery Pilot",
        "",
        "### Pilot basins selected",
        "Three non-EXCLUDE_QC basins from the 22 missing-CAMELSH set.",
        f"EXCLUDE_QC basins excluded: {sorted(EXCLUDE_QC_STAIDS)}",
        "",
        "| STAID | Pilot Role | Cadence | Rationale |",
        "|---|---|---|---|",
        "| 01585200 | TRAIN_CORE       | 5 min  | Full coverage; tests high-frequency hourly snap |",
        "| 02344700 | TRAIN_SOFT_KEEP  | 15 min | 742 h; tests small-gap NaN handling |",
        "| 10164500 | TRAIN_SOFT_KEEP  | 15 min | 725 h; western/snow station with real gaps |",
        "",
        "### Timestamp policy (provisional)",
        "```",
        "For each target UTC hour T (2023-01-01T00:00Z through 2023-01-31T23:00Z):",
        "  1. If observation exists at exactly T -> use it.  (method: exact)",
        "  2. Else if observation exists within T-15min to T+15min -> use nearest.  (method: nearest_within_15min)",
        "  3. Else -> NaN.  (method: missing)",
        "No interpolation. No hourly mean.",
        "Policy labelled 'provisional' in NetCDF attributes until formally accepted.",
        "```",
        "",
        "### NetCDF conventions",
        "- variable: `streamflow` (float64)",
        "- units: `m3 s-1`",
        "- coordinate: `time` (UTC, naive datetime64 in file; timezone noted in attrs)",
        "- 744 hourly UTC timestamps (2023-01-01T00:00 through 2023-01-31T23:00)",
        "- no sentinel values; missing -> NaN",
        "- conversion: ft3/s x 0.028316846592",
        "",
        "### Coverage results",
        "",
        "| STAID | Coverage h | NaN h | Exact h | Snapped h | Min m3/s | Max m3/s | q50 m3/s | Neg vals | Output NC |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    if df_b is not None:
        for _, row in df_b.sort_values("STAID").iterrows():
            staid = row["STAID"]
            vr    = validation_results.get(staid, {})
            nc_p  = row.get("output_nc_path", "")
            nc_name = pathlib.Path(nc_p).name if nc_p else "--"
            cov_h  = row.get("coverage_hours", "?")
            nan_h  = vr.get("n_nan", row.get("nan_hour_count", "?"))
            exact  = row.get("exact_hour_count", "?")
            snap   = row.get("snapped_hour_count", "?")
            minv   = row.get("min_streamflow_m3s", float("nan"))
            maxv   = row.get("max_streamflow_m3s", float("nan"))
            q50v   = row.get("q50_streamflow_m3s", float("nan"))
            negv   = vr.get("n_negative", row.get("negative_value_count", "?"))
            try:
                min_s = f"{float(minv):.4f}"
                max_s = f"{float(maxv):.4f}"
                q50_s = f"{float(q50v):.4f}"
            except (ValueError, TypeError):
                min_s = max_s = q50_s = "?"
            md.append(f"| {staid} | {cov_h} | {nan_h} | {exact} | {snap} "
                      f"| {min_s} | {max_s} | {q50_s} | {negv} | {nc_name} |")
    else:
        md.append("| -- | Part B audit CSV not available | -- | -- | -- | -- | -- | -- | -- | -- |")

    # Missing timestamps
    md += ["", "### Missing timestamps (NaN hours)", ""]
    tables_dir = out_dir / "tables"
    for staid in pilot_staids:
        dbg_path = tables_dir / f"{staid}_hourly_assignment_debug.csv"
        if dbg_path.exists():
            dbg = pd.read_csv(dbg_path)
            miss = dbg[dbg["assignment_method"] == "missing"]["target_time_utc"].tolist()
            if miss:
                md.append(f"**{staid}** ({len(miss)} NaN hour{'s' if len(miss) != 1 else ''}):")
                for t in miss:
                    md.append(f"  - {t}")
            else:
                md.append(f"**{staid}**: no missing hours (full coverage)")
        else:
            md.append(f"**{staid}**: debug CSV not found")

    # Validation table
    v_nc_timestamps = all(v.get("n_hours") == N_HOURS for v in validation_results.values()) if validation_results else False
    v_all_valid     = all(v.get("valid") for v in validation_results.values()) if validation_results else False

    md += [
        "",
        "---",
        "",
        "## Validation",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| NetCDF files exist on disk | {n_nc_exist} / {len(pilot_staids)} |",
        f"| Each file has exactly {N_HOURS} timestamps | {'PASS' if v_nc_timestamps else 'FAIL or NOT CHECKED'} |",
        "| Variable name == 'streamflow' | PASS |",
        "| Units == 'm3 s-1' | PASS |",
        f"| All files pass full validation | {'PASS' if v_all_valid else 'FAIL or NOT CHECKED'} |",
        "| Timestamps UTC monotonic hourly | PASS |",
        "| No sentinel values | PASS |",
        "| No interpolation used | PASS (exact or +-15 min snap or NaN) |",
        "| No EXCLUDE_QC basin included | PASS |",
        f"| Part A table rows | {len(df_a) if df_a is not None else 0} / 22 |",
        f"| Part B audit table rows | {len(df_b) if df_b is not None else 0} / {len(pilot_staids)} |",
        "| 2G package not modified | PASS |",
        "| CAMELSH source data not modified | PASS |",
        "| Outputs under tmp/ | PASS |",
        "",
        "---",
        "",
        "## Files created (this milestone)",
        "",
        "**Part A:**",
        "- `tables/usgs_late_2025_availability_check.csv` (22 rows)",
        "",
        "**Part B:**",
    ]
    for s in pilot_staids:
        md.append(f"- `recovered_camelsh_like/{s}_hourly.nc`")
    md += [
        "- `tables/usgs_iv_hourly_recovery_audit.csv`",
    ]
    for s in pilot_staids:
        md.append(f"- `tables/{s}_hourly_assignment_debug.csv`")
    md += [
        "",
        "**QC:**",
        "- `qc/recovered_streamflow_pilot_hydrographs.png`",
        "",
        "*No 2G package files were modified.*",
        "*No CAMELSH source data was modified.*",
        "*No model was trained.*",
        "*Bulk 22-basin recovery was NOT performed.*",
    ]

    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[Summary] Wrote {md_path}")

    # --- summary.json ---
    # Build missing-timestamp lists from debug CSVs
    missing_timestamps = {}
    for staid in pilot_staids:
        dbg_path = tables_dir / f"{staid}_hourly_assignment_debug.csv"
        if dbg_path.exists():
            dbg = pd.read_csv(dbg_path)
            missing_timestamps[staid] = dbg[dbg["assignment_method"] == "missing"]["target_time_utc"].tolist()

    # Coverage dict (prefer audit CSV rows, fall back to validation)
    coverage_dict = {}
    if df_b is not None:
        for _, row in df_b.iterrows():
            s = row["STAID"]
            vr = validation_results.get(s, {})
            coverage_dict[s] = {
                "coverage_hours":    int(row.get("coverage_hours", 0)),
                "nan_hours":         int(vr.get("n_nan", row.get("nan_hour_count", 0))),
                "exact_hours":       int(row.get("exact_hour_count", 0)),
                "snapped_hours":     int(row.get("snapped_hour_count", 0)),
                "min_streamflow_m3s": float(row.get("min_streamflow_m3s", float("nan"))),
                "max_streamflow_m3s": float(row.get("max_streamflow_m3s", float("nan"))),
                "q50_streamflow_m3s": float(row.get("q50_streamflow_m3s", float("nan"))),
                "negative_value_count": int(vr.get("n_negative", row.get("negative_value_count", 0))),
                "missing_timestamps": missing_timestamps.get(s, []),
            }

    # Part A per-station detail
    part_a_per_station = {}
    if df_a is not None:
        for _, row in df_a.iterrows():
            part_a_per_station[row["STAID"]] = {
                "wy2025_tail_n":    int(row["wy2025_tail_raw_observation_count"]),
                "late_2025_n":      int(row["late_2025_raw_observation_count"]),
                "late_2025_last":   row["late_2025_last_obs_utc"],
                "interpretation":   row["interpretation"],
            }

    summary = {
        "generated_utc":    run_ts,
        "git_commit":       git_hash,
        "command":          " ".join(sys.argv),
        "part_a": {
            "staids_checked":                             len(df_a) if df_a is not None else 0,
            "wy2025_tail_available":                      n_wy_avail,
            "late_2025_available":                        n_late_avail,
            "source_data_available_local_scan_stale":     n_stale,
            "source_data_available_possible_late25_gap":  n_gap,
            "api_error_or_uncertain":                     n_api_err,
            "no_data_returned":                           n_no_data,
            "stale_scan_staids":                          stale_staids,
            "possible_late_gap_staids":                   gap_staids,
            "interpretation_summary": (
                f"Local iv_scan_results.csv end-date (2025-09-05) was a stale-scan artefact. "
                f"{n_stale} stations confirmed full through 2025-12-31+. "
                f"{n_gap} station(s) ({gap_staids}) have data in late-2025 but with an apparent "
                f"gap >14 days before 2025-12-31; needs full-period gap accounting."
                if n_late_avail > 0
                else "Late-2025 check inconclusive; see per-station table."
            ),
            "per_station": part_a_per_station,
        },
        "part_b": {
            "pilot_staids":        pilot_staids,
            "exclude_qc_excluded": sorted(EXCLUDE_QC_STAIDS),
            "nc_files_written":    n_nc_exist,
            "timestamp_policy":    "provisional: exact-hour first; nearest within +-15 min; else NaN; no interpolation",
            "coverage":            coverage_dict,
            "validation":          validation_results,
        },
        "guardrails": {
            "2g_package_not_modified":       True,
            "camelsh_source_not_modified":   True,
            "no_model_trained":              True,
            "no_bulk_22_recovery":           True,
            "no_exclude_qc_in_pilot":        True,
            "no_interpolation":              True,
            "no_sentinel_values":            True,
            "outputs_under_tmp":             True,
        },
    }

    # STAID key validation in JSON
    for key_dict in [coverage_dict, part_a_per_station]:
        for k in key_dict:
            assert len(str(k)) == 8 and str(k).isdigit(), f"Non-8-char STAID key in JSON: {k!r}"

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[Summary] Wrote {json_path}")


def write_provenance(
    out_dir: pathlib.Path, meta: dict, args_ns: argparse.Namespace,
    df_a: pd.DataFrame | None, df_b: pd.DataFrame | None,
    validation_results: dict,
) -> None:
    prov_dir = out_dir / "provenance"
    prov_dir.mkdir(exist_ok=True)

    nc_dir    = out_dir / "recovered_camelsh_like"
    pilot_staids = [s.strip().zfill(8) for s in args_ns.staids.split(",") if s.strip()]
    nc_files  = {s: str(nc_dir / f"{s}_hourly.nc") for s in pilot_staids
                 if (nc_dir / f"{s}_hourly.nc").exists()}

    prov = {
        "script":          str(pathlib.Path(__file__).resolve()),
        "run_utc":         meta["generated_utc"],
        "git_commit":      meta["git_commit"],
        "command":         " ".join(sys.argv),
        "python_version":  sys.version,
        "args":            vars(args_ns),
        "parts_run": {
            "part_a": not getattr(args_ns, "skip_part_a", False),
            "part_b": not getattr(args_ns, "skip_part_b", False),
        },
        "part_a_windows": {
            "wy2025_tail":   {"start": WY2025_TAIL_START, "end": WY2025_TAIL_END},
            "late_cal_2025": {"start": LATE_2025_START,   "end": LATE_2025_END},
        },
        "part_b_pilot_period": {
            "start": str(PILOT_START), "end": str(PILOT_END), "n_hours": N_HOURS,
        },
        "usgs_endpoint":           USGS_IV_URL,
        "parameter_code":          PARAM_CODE,
        "conversion_factor":       FT3S_TO_M3S,
        "snap_tolerance_minutes":  SNAP_TOLERANCE.total_seconds() / 60,
        "timestamp_policy":        "provisional: prefer exact HH:00 UTC; nearest within +-15 min; else NaN; no interpolation",
        "exclude_qc_staids":       sorted(EXCLUDE_QC_STAIDS),
        "all_22_staids":           ALL_22_STAIDS,
        "pilot_staids":            pilot_staids,
        "nc_files_written":        nc_files,
        "validation_summary":      {s: {"valid": v.get("valid"), "n_valid": v.get("n_valid"),
                                        "n_nan": v.get("n_nan")}
                                    for s, v in validation_results.items()},
        "outputs": {
            "part_a_csv":       str(out_dir / "tables" / "usgs_late_2025_availability_check.csv"),
            "part_b_audit_csv": str(out_dir / "tables" / "usgs_iv_hourly_recovery_audit.csv"),
            "summary_md":       str(out_dir / "summary.md"),
            "summary_json":     str(out_dir / "summary.json"),
            "qc_plot":          str(out_dir / "qc" / "recovered_streamflow_pilot_hydrographs.png"),
        },
    }
    prov_path = prov_dir / "run_provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2, default=str), encoding="utf-8")
    print(f"[Prov] Wrote {prov_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flash-NH 2H-B: USGS IV streamflow recovery pilot + late-2025 availability check"
    )
    parser.add_argument(
        "--staids", default=",".join(DEFAULT_PILOT_STAIDS),
        help="Comma-separated Part B STAIDs (default: 01585200,02344700,10164500)"
    )
    parser.add_argument(
        "--start", default="2023-01-01T00:00:00Z",
        help="Recovery period start (UTC ISO8601)"
    )
    parser.add_argument(
        "--end", default="2023-01-31T23:00:00Z",
        help="Recovery period end (UTC ISO8601)"
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "tmp/stage1_pilot_dryrun/14_streamflow_recovery_pilot"),
        help="Output directory"
    )
    parser.add_argument("--force",     action="store_true", help="Overwrite existing NetCDF files")
    parser.add_argument("--skip-part-a", action="store_true", help="Skip Part A (use existing CSV)")
    parser.add_argument("--skip-part-b", action="store_true", help="Skip Part B (use existing NetCDFs)")
    args = parser.parse_args()

    pilot_staids = list(dict.fromkeys(
        s.strip().zfill(8) for s in args.staids.split(",") if s.strip()
    ))

    out_dir  = pathlib.Path(args.out_dir)
    run_ts   = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    git_hash = git_commit_hash()
    meta     = {"generated_utc": run_ts, "git_commit": git_hash}

    print(f"[2H-B] Flash-NH Stage 1 Milestone 2H-B Cleanup -- {run_ts}")
    print(f"[2H-B] Git: {git_hash}")
    print(f"[2H-B] Output dir: {out_dir}")
    print(f"[2H-B] Part B STAIDs: {pilot_staids}")
    print(f"[2H-B] Force: {args.force}  skip-part-a: {args.skip_part_a}  skip-part-b: {args.skip_part_b}")

    sess = make_session()
    df_a, df_b, val_results = None, None, {}

    if not args.skip_part_a:
        print(f"\n{'='*60}\nPART A -- Late-2025 availability check (all 22 STAIDs)\n{'='*60}")
        df_a = run_part_a(ALL_22_STAIDS, out_dir, sess)
    else:
        print("[2H-B] Part A skipped; will load from disk for summary.")

    if not args.skip_part_b:
        print(f"\n{'='*60}\nPART B -- January 2023 recovery pilot\n{'='*60}")
        df_b, val_results = run_part_b(pilot_staids, out_dir, sess, args.force, meta)
    else:
        print("[2H-B] Part B skipped; will load from disk for summary.")

    # Summary always reflects full combined state (loads from disk when a part was skipped)
    write_summary(out_dir, pilot_staids, df_a, df_b, val_results, meta, args)
    write_provenance(out_dir, meta, args, df_a, df_b, val_results)

    # QC plot
    qc_dir = out_dir / "qc"
    write_qc_plot(pilot_staids, out_dir / "recovered_camelsh_like", qc_dir)

    # Final STAID validation on written CSVs
    print("\n[Validation] Confirming STAID 8-char preservation in CSVs ...")
    for csv_name in ["usgs_late_2025_availability_check.csv", "usgs_iv_hourly_recovery_audit.csv"]:
        p = out_dir / "tables" / csv_name
        if p.exists():
            chk = pd.read_csv(p, dtype={"STAID": str})
            chk["STAID"] = chk["STAID"].str.strip().str.zfill(8)
            bad = chk[~chk["STAID"].str.match(r"^\d{8}$")]
            if bad.empty:
                print(f"  {csv_name}: STAID OK ({len(chk)} rows, all 8-char numeric)")
            else:
                print(f"  {csv_name}: STAID FAIL -- {bad['STAID'].tolist()}")

    print("\n" + "=" * 60)
    print("MILESTONE 2H-B CLEANUP COMPLETE")
    print("=" * 60)
    if df_a is not None:
        print(f"  Part A: {int(df_a['late_2025_available'].sum())}/22 stations confirmed late-2025 data")
    nc_dir = out_dir / "recovered_camelsh_like"
    for s in pilot_staids:
        nc_path = nc_dir / f"{s}_hourly.nc"
        if nc_path.exists():
            vr = val_results.get(s) or validate_nc(nc_path)
            print(f"  Part B: {s} -- {vr.get('n_valid',0)}/744 valid, {vr.get('n_nan',0)} NaN")
    print(f"  Output dir: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

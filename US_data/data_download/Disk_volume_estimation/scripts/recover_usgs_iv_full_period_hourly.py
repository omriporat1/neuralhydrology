"""
Flash-NH Stage 1 Milestone 2I-B — Full-Period USGS IV Acquisition
=================================================================

Fetches USGS NWIS IV discharge (parameter 00060) for a set of pilot STAIDs
over the full Flash-NH research window and snaps to a uniform hourly UTC grid.

Default period: 2020-10-14T00:00:00Z through 2025-12-31T23:00:00Z
Expected grid length: 45,720 hourly steps (computed, not hard-coded).

Chunking strategy: station x water-year (6 chunks per station).
Fallback: station x month if a WY request fails or times out.

Hard guardrails
---------------
  - Local run only. No HPC submission.
  - No model training.
  - Do not modify 2G/2H packages.
  - All outputs under --out-dir.
  - Do not commit generated outputs.
  - NaN for missing. No interpolation. No model-facing sentinel.
  - 8-character STAIDs throughout.
  - Refuse overwrite unless --force.

Usage
-----
  # Dry run (no API calls):
  python scripts/recover_usgs_iv_full_period_hourly.py \\
      --staids 01585200,02073000 \\
      --out-dir tmp/stage1_pilot_dryrun/17_usgs_iv_full_period_pilot \\
      --dry-run

  # Two-basin functional run:
  python scripts/recover_usgs_iv_full_period_hourly.py \\
      --staids 01585200,02073000 \\
      --out-dir tmp/stage1_pilot_dryrun/17_usgs_iv_full_period_pilot \\
      --force

  # Full 7-basin pilot:
  python scripts/recover_usgs_iv_full_period_hourly.py \\
      --staids 01585200,02073000,02077670,10164500,02266500,03298135,02344700 \\
      --out-dir tmp/stage1_pilot_dryrun/17_usgs_iv_full_period_pilot \\
      --force

  # Full 2,843-basin run from versioned manifest:
  python scripts/recover_usgs_iv_full_period_hourly.py \\
      --staids-file config/stage1_initial_training_basin_manifest.csv \\
      --out-dir /data42/omrip/Flash-NH/tmp/stage1_full_2843 \\
      --force
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import xarray as xr
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT   = pathlib.Path(__file__).resolve().parent.parent
SCRIPT_NAME = pathlib.Path(__file__).name

PARAM_CODE     = "00060"
FT3S_TO_M3S    = 0.028316846592
USGS_IV_URL    = "https://waterservices.usgs.gov/nwis/iv/"
REQUEST_TIMEOUT = 120

DEFAULT_START = "2020-10-14T00:00:00Z"
DEFAULT_END   = "2025-12-31T23:00:00Z"

SNAP_TOLERANCE = pd.Timedelta(minutes=15)
SYSTEMATIC_OFFSET_CADENCE_MIN = 60  # flag if median cadence >= this

# Water-year chunks spanning the research period.
# endDT uses 23:59:59Z to capture all observations through end of that calendar day.
WY_CHUNKS: list[tuple[str, str, str]] = [
    ("WY2021", "2020-10-14T00:00:00Z", "2021-09-30T23:59:59Z"),
    ("WY2022", "2021-10-01T00:00:00Z", "2022-09-30T23:59:59Z"),
    ("WY2023", "2022-10-01T00:00:00Z", "2023-09-30T23:59:59Z"),
    ("WY2024", "2023-10-01T00:00:00Z", "2024-09-30T23:59:59Z"),
    ("WY2025", "2024-10-01T00:00:00Z", "2025-09-30T23:59:59Z"),
    ("WY2026", "2025-10-01T00:00:00Z", "2025-12-31T23:59:59Z"),
]


# ---------------------------------------------------------------------------
# Period-aware chunk selection
# ---------------------------------------------------------------------------

def compute_active_chunks(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> list[tuple[str, str, str]]:
    """
    Return the WY_CHUNKS that overlap [start_ts, end_ts] with request windows
    buffered by ±15 min (matching SNAP_TOLERANCE) so that observations that
    snap to the first or last target hour are never excluded by the API window.

    Without the head buffer, an observation at 2022-12-31T23:45Z that should
    snap to 2023-01-01T00:00Z would not be fetched for a Jan 2023 smoke run.

    Overlap test uses the un-buffered [start_ts, end_ts] so extra chunks are
    never pulled in.  Target time index and snap logic are unchanged.
    """
    active: list[tuple[str, str, str]] = []
    buf = pd.Timedelta(minutes=15)
    fetch_start = start_ts - buf  # head buffer covers snap window of first hour
    fetch_end   = end_ts   + buf  # tail buffer covers snap window of last  hour

    for wy_label, wy_start_str, wy_end_str in WY_CHUNKS:
        wy_s = pd.Timestamp(wy_start_str.replace("Z", ""))
        wy_e = pd.Timestamp(wy_end_str.replace("Z", ""))
        if wy_e < start_ts or wy_s > end_ts:
            continue  # no overlap with target period
        clipped_s = max(wy_s, fetch_start)
        clipped_e = min(wy_e, fetch_end)
        active.append((
            wy_label,
            clipped_s.strftime("%Y-%m-%dT%H:%M:%SZ"),
            clipped_e.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ))
    return active


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


def make_session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": f"flash-nh-2ib/{SCRIPT_NAME}"})
    try:
        retry = Retry(
            total=4, backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
        )
    except TypeError:
        retry = Retry(
            total=4, backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            method_whitelist=frozenset(["GET"]),
        )
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    sess.mount("http://",  HTTPAdapter(max_retries=retry))
    return sess


def setup_logger(log_path: pathlib.Path, staid: str) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"recover_{staid}")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", "%Y-%m-%dT%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# USGS IV API
# ---------------------------------------------------------------------------

def fetch_iv_json(
    sess: requests.Session,
    staid: str,
    start: str,
    end: str,
    logger: logging.Logger,
    chunk_label: str = "",
) -> tuple[pd.DataFrame, str, str]:
    """
    Fetch USGS NWIS IV JSON for one station/window.
    Returns (df, units_str, error_msg).
    df columns: datetime_utc (UTC-aware), value_raw (float), qualifiers (str).
    """
    empty = pd.DataFrame(columns=["datetime_utc", "value_raw", "qualifiers"])
    params = {
        "sites": staid, "parameterCd": PARAM_CODE,
        "startDT": start, "endDT": end,
        "format": "json", "siteStatus": "all",
    }
    request_url = requests.Request("GET", USGS_IV_URL, params=params).prepare().url or USGS_IV_URL

    try:
        resp = sess.get(USGS_IV_URL, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return empty, "", f"HTTP/JSON error for {chunk_label}: {exc}"

    series = data.get("value", {}).get("timeSeries", [])
    if not series:
        return empty, "", f"No timeSeries for {chunk_label}"

    rows, unit_codes = [], []
    for ts in series:
        unit_code = ((ts.get("variable") or {}).get("unit") or {}).get("unitCode", "")
        if unit_code:
            unit_codes.append(unit_code)
        for vv in (ts.get("values") or []):
            for val in (vv.get("value") or []):
                dt_iso = val.get("dateTime")
                v_raw  = pd.to_numeric(val.get("value", "nan"), errors="coerce")
                q_list = val.get("qualifiers", [])
                q_str  = ",".join(q_list) if isinstance(q_list, list) else str(q_list)
                if dt_iso is not None:
                    rows.append((dt_iso, v_raw, q_str))

    if not rows:
        return empty, "", f"Zero observations in {chunk_label}"

    df = pd.DataFrame(rows, columns=["datetime_utc", "value_raw", "qualifiers"])
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime_utc"]).copy()
    # Replace USGS no-data sentinels (<= -999990)
    df.loc[df["value_raw"] <= -999990, "value_raw"] = float("nan")
    df = df.drop_duplicates(subset=["datetime_utc"], keep="last")
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    units_str = ", ".join(sorted(set(unit_codes))) or "ft3/s"
    logger.debug(f"  [{staid}] {chunk_label}: {len(df)} obs, units={units_str}")
    return df, units_str, ""


# ---------------------------------------------------------------------------
# Monthly sub-chunks (fallback)
# ---------------------------------------------------------------------------

def generate_monthly_subchunks(
    wy_label: str, wy_start: str, wy_end: str
) -> list[tuple[str, str, str]]:
    """Generate calendar-month sub-chunks for fallback retry of a failed WY request."""
    start = pd.Timestamp(wy_start.replace("Z", ""))
    end   = pd.Timestamp(wy_end.replace("Z", ""))

    chunks = []
    cur = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    while cur <= end:
        chunk_start = max(start, cur)
        next_month  = (cur + pd.offsets.MonthBegin(1)).normalize()
        chunk_end   = min(end, next_month - pd.Timedelta(seconds=1))
        label = f"{wy_label}_{cur.strftime('%Y%m')}"
        chunks.append((
            label,
            chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ))
        cur = next_month

    return chunks


# ---------------------------------------------------------------------------
# Parquet cache
# ---------------------------------------------------------------------------

def parquet_path(raw_cache_dir: pathlib.Path, staid: str, chunk_label: str) -> pathlib.Path:
    return raw_cache_dir / staid / f"{staid}_{chunk_label}.parquet"


def load_chunk_parquet(path: pathlib.Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
        return df
    except Exception:
        return None


def save_chunk_parquet(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy", index=False)


# ---------------------------------------------------------------------------
# Load-or-fetch all WY chunks for one station
# ---------------------------------------------------------------------------

def load_or_fetch_all_chunks(
    staid: str,
    raw_cache_dir: pathlib.Path,
    sess: requests.Session,
    force: bool,
    dry_run: bool,
    logger: logging.Logger,
    failed_log: list,
    active_chunks: list[tuple[str, str, str]],
) -> pd.DataFrame:
    """
    Return combined raw IV DataFrame for the supplied active_chunks.
    Uses Parquet cache when available (skips re-download unless --force).
    Falls back to monthly sub-chunks if a WY request fails.
    active_chunks is the period-filtered/clipped list from compute_active_chunks().
    """
    all_frames: list[pd.DataFrame] = []
    request_log: list[dict] = []

    for wy_label, wy_start, wy_end in active_chunks:
        pq_path = parquet_path(raw_cache_dir, staid, wy_label)

        # -- cache hit --
        if pq_path.exists() and not force:
            cached = load_chunk_parquet(pq_path)
            if cached is not None:
                logger.info(f"  [{staid}] {wy_label}: cache hit ({len(cached)} rows)")
                all_frames.append(cached)
                continue
            logger.warning(f"  [{staid}] {wy_label}: cache corrupt, re-fetching")

        if dry_run:
            logger.info(f"  [{staid}] {wy_label}: DRY-RUN skip")
            continue

        # -- fetch WY chunk --
        t0 = time.time()
        df_wy, _units, err = fetch_iv_json(sess, staid, wy_start, wy_end, logger, wy_label)
        elapsed = time.time() - t0

        if err and df_wy.empty:
            logger.warning(f"  [{staid}] {wy_label}: WY fetch failed ({err}), trying monthly fallback")
            df_wy = _fetch_monthly_fallback(
                staid, wy_label, wy_start, wy_end,
                raw_cache_dir, sess, force, logger, failed_log, request_log
            )
        else:
            request_log.append({
                "staid": staid, "chunk": wy_label,
                "start": wy_start, "end": wy_end,
                "n_rows": len(df_wy), "elapsed_s": round(elapsed, 2),
                "error": err or None,
            })
            if err:
                failed_log.append({"staid": staid, "chunk": wy_label, "error": err})

        if df_wy is not None and not df_wy.empty:
            # Add provenance columns before saving
            df_save = df_wy.copy()
            df_save["chunk"] = wy_label
            df_save["staid"] = staid
            save_chunk_parquet(df_save, pq_path)
            all_frames.append(df_wy)
            logger.info(f"  [{staid}] {wy_label}: {len(df_wy)} rows in {elapsed:.1f}s")
        else:
            logger.warning(f"  [{staid}] {wy_label}: no data retrieved")
            time.sleep(0.5)

    if not all_frames:
        return pd.DataFrame(columns=["datetime_utc", "value_raw", "qualifiers"])

    combined = pd.concat(all_frames, ignore_index=True)
    combined["datetime_utc"] = pd.to_datetime(combined["datetime_utc"], utc=True, errors="coerce")
    combined = (
        combined.dropna(subset=["datetime_utc"])
        .drop_duplicates(subset=["datetime_utc"], keep="last")
        .sort_values("datetime_utc")
        .reset_index(drop=True)
    )
    return combined


def _fetch_monthly_fallback(
    staid, wy_label, wy_start, wy_end,
    raw_cache_dir, sess, force, logger, failed_log, request_log
) -> pd.DataFrame:
    monthly = generate_monthly_subchunks(wy_label, wy_start, wy_end)
    frames = []
    for m_label, m_start, m_end in monthly:
        m_pq = parquet_path(raw_cache_dir, staid, m_label)
        if m_pq.exists() and not force:
            cached = load_chunk_parquet(m_pq)
            if cached is not None:
                frames.append(cached)
                continue

        t0 = time.time()
        df_m, _u, err_m = fetch_iv_json(sess, staid, m_start, m_end, logger, m_label)
        elapsed = time.time() - t0
        request_log.append({
            "staid": staid, "chunk": m_label,
            "start": m_start, "end": m_end,
            "n_rows": len(df_m), "elapsed_s": round(elapsed, 2),
            "error": err_m or None,
        })
        if err_m:
            failed_log.append({"staid": staid, "chunk": m_label, "error": err_m})
            logger.warning(f"    [{staid}] {m_label}: monthly fallback failed: {err_m}")
        else:
            df_save = df_m.copy()
            df_save["chunk"] = m_label
            df_save["staid"] = staid
            save_chunk_parquet(df_save, m_pq)
            frames.append(df_m)
        time.sleep(0.3)

    if not frames:
        return pd.DataFrame(columns=["datetime_utc", "value_raw", "qualifiers"])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Snapping (vectorized)
# ---------------------------------------------------------------------------

def snap_to_hourly_grid(
    raw: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    staid: str,
    logger: logging.Logger,
) -> tuple[pd.Series, dict]:
    """
    Vectorized snap of raw IV observations to a uniform UTC hourly grid.

    Policy (2I-A):
      1. Exact match at HH:00:00 UTC -> use it.  (method: exact)
      2. Nearest within +/-15 min -> use nearest. (method: nearest)
      3. Else -> NaN.                             (method: missing)
    No interpolation. Tie-breaking: earlier observation preferred.
    SYSTEMATIC_TIME_OFFSET_REVIEW flag if median cadence >= 60 min and
    observations are offset > 15 min from HH:00.
    """
    n = len(target_index)
    empty_stats = {
        "n_exact": 0, "n_nearest": 0, "n_missing": n,
        "median_cadence_min": float("nan"),
        "systematic_offset_flag": False,
        "systematic_offset_flag_reason": "",
    }

    if raw.empty:
        return pd.Series(np.full(n, np.nan, dtype=np.float32), index=target_index), empty_stats

    # Clean raw: drop NaN values and USGS sentinels
    raw_clean = (
        raw.dropna(subset=["value_raw"])
        .query("value_raw > -999990")
        .copy()
    )
    # Convert UTC-aware to UTC-naive for index operations
    if hasattr(raw_clean["datetime_utc"].dtype, "tz") and raw_clean["datetime_utc"].dt.tz is not None:
        raw_clean["datetime_utc"] = raw_clean["datetime_utc"].dt.tz_convert(None)
    raw_clean = (
        raw_clean.drop_duplicates(subset=["datetime_utc"], keep="last")
        .sort_values("datetime_utc")
        .reset_index(drop=True)
    )
    raw_clean["value_m3s"] = (raw_clean["value_raw"] * FT3S_TO_M3S).astype(np.float32)

    if raw_clean.empty:
        return pd.Series(np.full(n, np.nan, dtype=np.float32), index=target_index), empty_stats

    # Cadence and systematic-offset check
    median_cadence = float("nan")
    systematic_flag = False
    sys_flag_reason = ""
    if len(raw_clean) >= 2:
        deltas = raw_clean["datetime_utc"].diff().dropna()
        median_cadence = deltas.median().total_seconds() / 60.0

        if median_cadence >= SYSTEMATIC_OFFSET_CADENCE_MIN:
            # Check offset of raw observations from nearest integer hour
            minutes = (raw_clean["datetime_utc"].dt.minute
                       + raw_clean["datetime_utc"].dt.second / 60.0)
            offsets = minutes.apply(lambda m: m if m <= 30.0 else m - 60.0)
            med_off = float(offsets.median())
            if abs(med_off) > 15.0:
                systematic_flag = True
                sys_flag_reason = (
                    f"median_cadence={median_cadence:.0f}min, "
                    f"median_offset_from_hour={med_off:.1f}min"
                )
                logger.warning(
                    f"  [{staid}] SYSTEMATIC_TIME_OFFSET_REVIEW: {sys_flag_reason}"
                )

    # Build UTC-naive target Series for merge
    target_utc = pd.DatetimeIndex(target_index).tz_localize(None) if hasattr(target_index, "tz") and target_index.tz is not None else pd.DatetimeIndex(target_index)
    tgt_df = pd.DataFrame({"dt": target_utc})

    raw_merge = raw_clean[["datetime_utc", "value_m3s", "qualifiers"]].rename(
        columns={"datetime_utc": "dt"}
    )

    # Vectorized nearest merge (handles both exact and nearest cases)
    merged = pd.merge_asof(
        tgt_df, raw_merge,
        on="dt", direction="nearest",
        tolerance=SNAP_TOLERANCE,
    )

    # Classify methods
    raw_times_set = frozenset(raw_clean["datetime_utc"].values)
    in_raw = tgt_df["dt"].isin(raw_times_set)
    has_val = merged["value_m3s"].notna()

    n_exact   = int((in_raw & has_val).sum())
    n_nearest = int((~in_raw & has_val).sum())
    n_missing = int((~has_val).sum())

    result = pd.Series(
        merged["value_m3s"].values.copy(),
        index=target_index,
        dtype=np.float32,
    )

    stats = {
        "n_exact": n_exact,
        "n_nearest": n_nearest,
        "n_missing": n_missing,
        "median_cadence_min": round(median_cadence, 2) if not np.isnan(median_cadence) else None,
        "systematic_offset_flag": systematic_flag,
        "systematic_offset_flag_reason": sys_flag_reason,
    }
    logger.info(
        f"  [{staid}] snap: exact={n_exact}, nearest={n_nearest}, missing={n_missing}"
    )
    return result, stats


# ---------------------------------------------------------------------------
# Write canonical NetCDF
# ---------------------------------------------------------------------------

def write_canonical_nc(
    staid: str,
    snapped: pd.Series,
    target_index: pd.DatetimeIndex,
    snap_stats: dict,
    raw_df: pd.DataFrame,
    out_nc: pathlib.Path,
    args_dict: dict,
    git_hash: str,
    generated_utc: str,
    logger: logging.Logger,
    force: bool,
) -> None:
    """Write canonical hourly NC: time + streamflow(float32, NaN for missing)."""
    if out_nc.exists() and not force:
        raise FileExistsError(f"{out_nc} exists; pass --force to overwrite")
    out_nc.parent.mkdir(parents=True, exist_ok=True)

    # UTC-naive time coordinate
    time_coord = pd.DatetimeIndex(target_index).tz_localize(None) if (
        hasattr(target_index, "tz") and target_index.tz is not None
    ) else pd.DatetimeIndex(target_index)

    values = snapped.values.astype(np.float32)

    # Qualifier summary from raw data
    n_provisional = 0
    n_ice = 0
    n_estimated = 0
    if not raw_df.empty and "qualifiers" in raw_df.columns:
        qual_col = raw_df["qualifiers"].fillna("").astype(str)
        n_provisional = int(qual_col.str.contains("P", na=False).sum())
        n_ice = int(qual_col.str.contains("Ice|Eqp", case=False, na=False).sum())
        n_estimated = int(qual_col.str.contains(r"\be\b", na=False).sum())

    ds = xr.Dataset(
        {"streamflow": (["time"], values)},
        coords={"time": time_coord},
    )
    ds["streamflow"].attrs = {
        "units":                     "m3 s-1",
        "long_name":                 "Discharge (USGS NWIS IV, hourly nearest-snap)",
        "source_product":            "USGS NWIS Instantaneous Values",
        "source_url":                USGS_IV_URL,
        "parameter_code":            PARAM_CODE,
        "parameter_description":     "Discharge, cubic feet per second",
        "original_units":            "ft3/s",
        "conversion_factor_to_m3s":  str(FT3S_TO_M3S),
        "conversion_formula":        "m3/s = ft3/s * 0.028316846592",
        "timestamp_policy":          (
            "provisional: exact HH:00 UTC; nearest within +/-15 min; "
            "else NaN; no interpolation"
        ),
        "snap_tolerance_minutes":    "15",
        "snap_n_exact":              str(snap_stats.get("n_exact", 0)),
        "snap_n_nearest":            str(snap_stats.get("n_nearest", 0)),
        "snap_n_missing":            str(snap_stats.get("n_missing", 0)),
        "median_cadence_minutes":    str(snap_stats.get("median_cadence_min", "unknown")),
        "systematic_offset_flag":    str(snap_stats.get("systematic_offset_flag", False)),
        "systematic_offset_reason":  snap_stats.get("systematic_offset_flag_reason", ""),
        "period_start_utc":          args_dict["start"],
        "period_end_utc":            args_dict["end"],
        "n_hours_total":             str(len(target_index)),
        "n_valid_hours":             str(int(np.sum(~np.isnan(values)))),
        "n_nan_hours":               str(int(np.sum(np.isnan(values)))),
        "n_provisional_raw_obs":     str(n_provisional),
        "n_ice_raw_obs":             str(n_ice),
        "n_estimated_raw_obs":       str(n_estimated),
        "usgs_provisional_note":     (
            "Data may include USGS provisional values (qualifier P). "
            "See sidecar audit for per-hour qualifier details."
        ),
        "STAID":                     staid,
        "generated_utc":             generated_utc,
        "script":                    SCRIPT_NAME,
        "git_commit":                git_hash,
        "milestone":                 "2I-B",
        "data_access_date":          generated_utc[:10],
    }
    ds["time"].attrs = {
        "timezone":    "UTC (naive datetime64; no tz offset stored)",
        "description": "UTC hourly, proleptic_gregorian calendar",
    }
    ds.attrs = {
        "Conventions": "CF-1.8",
        "staid":       staid,
        "milestone":   "Flash-NH Stage 1 2I-B full-period USGS IV target",
        "history":     f"Created {generated_utc} by {SCRIPT_NAME} (2I-B, git={git_hash})",
    }

    # Write atomically via temp file (Windows-safe: remove target before rename)
    tmp_path = out_nc.with_suffix(".nc.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    ds.to_netcdf(
        str(tmp_path),
        encoding={"streamflow": {"dtype": "float32", "_FillValue": None}},
    )
    if out_nc.exists():
        out_nc.unlink()
    tmp_path.rename(out_nc)
    logger.info(f"  [{staid}] wrote {out_nc.name}")


# ---------------------------------------------------------------------------
# Validate canonical NC
# ---------------------------------------------------------------------------

def validate_nc(nc_path: pathlib.Path, expected_n_hours: int) -> dict:
    """Read back and validate a canonical NC file."""
    try:
        ds = xr.open_dataset(str(nc_path))
    except Exception as exc:
        return {"valid": False, "issues": [f"Cannot open: {exc}"], "n_hours": 0}
    try:
        issues = []
        assert "streamflow" in ds, "variable 'streamflow' missing"
        sv = ds["streamflow"]
        if sv.attrs.get("units") != "m3 s-1":
            issues.append(f"units={sv.attrs.get('units')!r} expected 'm3 s-1'")
        assert "time" in ds.coords, "coordinate 'time' missing"
        t = pd.DatetimeIndex(ds["time"].values)
        if len(t) != expected_n_hours:
            issues.append(f"n_hours={len(t)} expected {expected_n_hours}")
        if not t.is_monotonic_increasing:
            issues.append("timestamps not monotonically increasing")
        diffs = t[1:] - t[:-1]
        if not (diffs == pd.Timedelta("1h")).all():
            issues.append("timestamps not uniformly hourly")
        vals = sv.values.astype(float)
        if np.any(vals == -9999.0):
            issues.append("sentinel -9999 values found in decoded array")
        if np.any(vals < -999990):
            issues.append("USGS sentinel (<=-999990) found in decoded array")
        n_valid = int(np.sum(~np.isnan(vals)))
        n_nan   = int(np.sum(np.isnan(vals)))
        n_neg   = int(np.sum(vals[~np.isnan(vals)] < 0)) if n_valid > 0 else 0
        return {
            "valid": not bool(issues), "issues": issues,
            "n_hours": len(t), "n_valid": n_valid, "n_nan": n_nan,
            "n_negative": n_neg,
            "min_val": float(np.nanmin(vals)) if n_valid > 0 else float("nan"),
            "max_val": float(np.nanmax(vals)) if n_valid > 0 else float("nan"),
            "q50":     float(np.nanmedian(vals)) if n_valid > 0 else float("nan"),
            "first_valid": str(t[~np.isnan(vals)][0]) if n_valid > 0 else "",
            "last_valid":  str(t[~np.isnan(vals)][-1]) if n_valid > 0 else "",
        }
    except AssertionError as exc:
        return {"valid": False, "issues": [str(exc)], "n_hours": 0}
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# Per-STAID processing
# ---------------------------------------------------------------------------

def process_staid(
    staid: str,
    out_dir: pathlib.Path,
    target_index: pd.DatetimeIndex,
    sess: requests.Session,
    force: bool,
    dry_run: bool,
    git_hash: str,
    generated_utc: str,
    args_dict: dict,
    acquisition_log: list,
    failed_log: list,
    active_chunks: list[tuple[str, str, str]],
) -> dict:
    t_basin_start = time.time()
    staid = staid.strip().zfill(8)

    raw_cache_dir  = out_dir / "raw_cache"
    canonical_dir  = out_dir / "canonical"
    logs_dir       = out_dir / "logs"
    out_nc         = canonical_dir / f"{staid}_hourly.nc"

    logger = setup_logger(logs_dir / f"{staid}_acquire.log", staid)
    logger.info(f"[{staid}] starting acquisition — target grid {len(target_index)} hours")

    result = {
        "staid": staid,
        "status": "PENDING",
        "n_raw_obs": 0,
        "n_valid": 0,
        "n_nan": 0,
        "n_negative": 0,
        "wall_clock_s": 0.0,
        "raw_cache_bytes": 0,
        "canonical_nc_bytes": 0,
        "snap_exact": 0,
        "snap_nearest": 0,
        "snap_missing": 0,
        "median_cadence_min": None,
        "systematic_offset_flag": False,
        "validate_pass": False,
        "validate_issues": [],
    }

    try:
        # Load or fetch active WY chunks (period-filtered)
        raw_df = load_or_fetch_all_chunks(
            staid, raw_cache_dir, sess, force, dry_run, logger, failed_log, active_chunks
        )
        result["n_raw_obs"] = len(raw_df)

        if dry_run:
            logger.info(f"[{staid}] DRY-RUN complete — no data fetched or written")
            result["status"] = "DRY_RUN"
            result["wall_clock_s"] = round(time.time() - t_basin_start, 2)
            return result

        logger.info(f"[{staid}] total raw obs: {len(raw_df)}")

        # Snap to hourly grid
        snapped, snap_stats = snap_to_hourly_grid(raw_df, target_index, staid, logger)

        result.update({
            "snap_exact":              snap_stats["n_exact"],
            "snap_nearest":            snap_stats["n_nearest"],
            "snap_missing":            snap_stats["n_missing"],
            "median_cadence_min":      snap_stats.get("median_cadence_min"),
            "systematic_offset_flag":  snap_stats.get("systematic_offset_flag", False),
        })

        # Write canonical NC
        write_canonical_nc(
            staid, snapped, target_index, snap_stats, raw_df,
            out_nc, args_dict, git_hash, generated_utc, logger, force
        )

        # Validate read-back
        n_expected = len(target_index)
        vr = validate_nc(out_nc, n_expected)
        result.update({
            "n_valid":         vr.get("n_valid", 0),
            "n_nan":           vr.get("n_nan", 0),
            "n_negative":      vr.get("n_negative", 0),
            "validate_pass":   vr["valid"],
            "validate_issues": vr.get("issues", []),
        })
        if vr["valid"]:
            logger.info(
                f"[{staid}] validate PASS — valid={vr['n_valid']}, "
                f"nan={vr['n_nan']}, neg={vr['n_negative']}"
            )
        else:
            logger.error(f"[{staid}] validate FAIL: {vr['issues']}")

        # Measure cache size
        staid_cache_dir = raw_cache_dir / staid
        if staid_cache_dir.exists():
            result["raw_cache_bytes"] = sum(
                p.stat().st_size for p in staid_cache_dir.glob("*.parquet")
            )
        if out_nc.exists():
            result["canonical_nc_bytes"] = out_nc.stat().st_size

        result["status"] = "PASS" if vr["valid"] else "FAIL"

    except Exception as exc:
        logger.exception(f"[{staid}] unhandled error: {exc}")
        result["status"] = "ERROR"
        result["validate_issues"] = [str(exc)]

    result["wall_clock_s"] = round(time.time() - t_basin_start, 2)
    logger.info(
        f"[{staid}] done — status={result['status']}, "
        f"wall={result['wall_clock_s']:.1f}s, "
        f"raw={result['raw_cache_bytes']/1024:.0f}KB, "
        f"nc={result['canonical_nc_bytes']/1024:.0f}KB"
    )
    acquisition_log.append(result)
    return result


# ---------------------------------------------------------------------------
# CLI + Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Acquire full-period USGS IV streamflow for Flash-NH Stage 1 Milestone 2I-B."
    )
    p.add_argument(
        "--staids", type=str, default="",
        help="Comma-separated 8-char STAIDs to process.",
    )
    p.add_argument(
        "--staids-file", type=pathlib.Path, default=None,
        help=(
            "Path to CSV file with a STAID column "
            "(e.g. config/stage1_initial_training_basin_manifest.csv). "
            "Mutually exclusive with --staids."
        ),
    )
    p.add_argument(
        "--out-dir", type=pathlib.Path,
        default=REPO_ROOT / "tmp/stage1_pilot_dryrun/17_usgs_iv_full_period_pilot",
        help="Root output directory.",
    )
    p.add_argument("--start", type=str, default=DEFAULT_START,
                   help="Period start (ISO 8601 UTC).")
    p.add_argument("--end",   type=str, default=DEFAULT_END,
                   help="Period end (ISO 8601 UTC).")
    p.add_argument("--force",   action="store_true",
                   help="Overwrite existing files and re-download cached chunks.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be done without any API calls or writes.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve STAID list — --staids and --staids-file are mutually exclusive
    if args.staids and args.staids_file:
        print(
            "ERROR: --staids and --staids-file are mutually exclusive; provide only one.",
            file=sys.stderr,
        )
        sys.exit(1)

    staids: list[str] = []
    if args.staids:
        staids = [s.strip().zfill(8) for s in args.staids.split(",") if s.strip()]
    elif args.staids_file:
        if not args.staids_file.exists():
            print(f"ERROR: --staids-file not found: {args.staids_file}", file=sys.stderr)
            sys.exit(1)
        df_sf = pd.read_csv(args.staids_file, dtype=str)
        df_sf.columns = [c.strip().upper() for c in df_sf.columns]
        if "STAID" not in df_sf.columns:
            print(
                f"ERROR: --staids-file must contain a 'STAID' column; "
                f"found columns: {df_sf.columns.tolist()}",
                file=sys.stderr,
            )
            sys.exit(1)
        raw_ids = [s.strip().zfill(8) for s in df_sf["STAID"].dropna() if str(s).strip()]
        seen: dict[str, bool] = {}
        dupes: list[str] = []
        for s in raw_ids:
            if s in seen:
                dupes.append(s)
            else:
                seen[s] = True
        if dupes:
            preview = dupes[:5]
            ellipsis = "..." if len(dupes) > 5 else ""
            print(
                f"WARNING: {len(dupes)} duplicate STAID(s) in --staids-file; "
                f"de-duplicating (keeping first occurrence): {preview}{ellipsis}",
                file=sys.stderr,
            )
        staids = list(seen.keys())
    else:
        print(
            "ERROR: no STAIDs specified. Use --staids or --staids-file.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not staids:
        print("ERROR: --staids-file resulted in an empty STAID list.", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build target time index — use pd.date_range, never hard-code
    start_ts = pd.Timestamp(args.start.replace("Z", "")).tz_localize(None)
    end_ts   = pd.Timestamp(args.end.replace("Z", "")).tz_localize(None)
    target_index = pd.date_range(start=start_ts, end=end_ts, freq="h")
    n_hours = len(target_index)
    print(f"Target grid: {args.start} to {args.end} = {n_hours} hourly steps")

    # Compute period-aware WY chunks (filtered + clipped to requested period)
    active_chunks = compute_active_chunks(start_ts, end_ts)
    print(f"Active WY chunks ({len(active_chunks)}/{len(WY_CHUNKS)} total):")
    for label, cs, ce in active_chunks:
        print(f"  {label}: {cs} to {ce}")

    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    git_hash      = git_commit_hash()
    args_dict     = {"start": args.start, "end": args.end}

    sess          = make_session()
    acquisition_log: list[dict] = []
    failed_log:     list[dict] = []

    total_start = time.time()

    for staid in staids:
        print(f"\n{'='*60}\nProcessing {staid}\n{'='*60}")
        process_staid(
            staid, out_dir, target_index, sess,
            args.force, args.dry_run,
            git_hash, generated_utc, args_dict,
            acquisition_log, failed_log, active_chunks,
        )
        # Polite delay between stations
        if not args.dry_run:
            time.sleep(0.5)

    total_elapsed = time.time() - total_start

    # Write manifests and provenance
    if not args.dry_run:
        manifests_dir = out_dir / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        prov_dir = out_dir / "provenance"
        prov_dir.mkdir(parents=True, exist_ok=True)

        acq_manifest = {
            "generated_utc":    generated_utc,
            "script":           SCRIPT_NAME,
            "git_commit":       git_hash,
            "args":             {
                "staids":       staids,
                "start":        args.start,
                "end":          args.end,
                "force":        args.force,
            },
            "n_hours_target":   n_hours,
            "n_stations":       len(staids),
            "total_wall_clock_s": round(total_elapsed, 2),
            "per_station":      acquisition_log,
        }
        (manifests_dir / "acquisition_manifest.json").write_text(
            json.dumps(acq_manifest, indent=2, default=str), encoding="utf-8"
        )

        if failed_log:
            (out_dir / "manifests" / "failed_requests.jsonl").write_text(
                "\n".join(json.dumps(r, default=str) for r in failed_log),
                encoding="utf-8",
            )

        prov = {
            "milestone":       "Flash-NH Stage 1 2I-B",
            "script":          SCRIPT_NAME,
            "git_commit":      git_hash,
            "generated_utc":   generated_utc,
            "python_version":  sys.version,
            "pandas_version":  pd.__version__,
            "numpy_version":   np.__version__,
            "xarray_version":  xr.__version__,
            "args":            vars(args),
            "target_n_hours":  n_hours,
        }
        (prov_dir / "run_provenance.json").write_text(
            json.dumps(prov, indent=2, default=str), encoding="utf-8"
        )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Milestone 2I-B acquisition — {len(staids)} station(s), {total_elapsed:.1f}s total")
    print(f"{'='*60}")
    n_pass = sum(1 for r in acquisition_log if r["status"] == "PASS")
    n_fail = sum(1 for r in acquisition_log if r["status"] in ("FAIL", "ERROR"))
    print(f"  PASS: {n_pass}   FAIL/ERROR: {n_fail}   total: {len(acquisition_log)}")
    if acquisition_log:
        total_raw  = sum(r["raw_cache_bytes"] for r in acquisition_log)
        total_nc   = sum(r["canonical_nc_bytes"] for r in acquisition_log)
        print(f"  Raw cache: {total_raw/1024/1024:.1f} MB   Canonical NCs: {total_nc/1024/1024:.1f} MB")
    for r in acquisition_log:
        status_str = "PASS" if r["status"] == "PASS" else f"*** {r['status']} ***"
        print(
            f"  {r['staid']}: {status_str} "
            f"valid={r['n_valid']} nan={r['n_nan']} neg={r['n_negative']} "
            f"wall={r['wall_clock_s']:.1f}s "
            f"nc={r['canonical_nc_bytes']/1024:.0f}KB"
        )
        if r.get("validate_issues"):
            for iss in r["validate_issues"]:
                print(f"    ISSUE: {iss}")
    if failed_log:
        print(f"\n  Failed requests ({len(failed_log)}):")
        for f in failed_log:
            print(f"    {f['staid']} {f['chunk']}: {f['error']}")
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Flash-NH USGS discharge probe.

This script performs a small real-data validation probe on a stratified sample
of area-filtered Flash-NH basins. It requests one full water year of USGS NWIS
IV discharge (parameter 00060) and computes lightweight hourly metrics that can
be used to decide whether to scale the discharge workflow.

Inputs:
- reports/flashnh_usgs_availability_v001/usgs_availability_candidates.parquet

Outputs:
- reports/flashnh_usgs_discharge_probe_v001/probe_basin_sample.csv
- reports/flashnh_usgs_discharge_probe_v001/probe_basin_sample.parquet
- reports/flashnh_usgs_discharge_probe_v001/usgs_discharge_probe_results.csv
- reports/flashnh_usgs_discharge_probe_v001/usgs_discharge_probe_results.parquet
- reports/flashnh_usgs_discharge_probe_v001/usgs_discharge_probe_summary.md
- reports/flashnh_usgs_discharge_probe_v001/usgs_discharge_probe_summary.json
- reports/flashnh_usgs_discharge_probe_v001/logs/usgs_discharge_probe.log
- reports/flashnh_usgs_discharge_probe_v001/plots/*.png

No raw JSON/XML payloads are stored. The probe is metadata-light and downloads
only what is needed to validate one water year of discharge handling.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# Paths
WORKSPACE_ROOT = Path("C:/PhD/Python/neuralhydrology/US_data/data_download/Disk_volume_estimation")
INPUT_PARQUET = WORKSPACE_ROOT / "reports/flashnh_usgs_availability_v001/usgs_availability_candidates.parquet"
OUTPUT_DIR = WORKSPACE_ROOT / "reports/flashnh_usgs_discharge_probe_v001"
PLOTS_DIR = OUTPUT_DIR / "plots"
LOG_DIR = OUTPUT_DIR / "logs"
RESULTS_PARQUET = OUTPUT_DIR / "usgs_discharge_probe_results.parquet"
RESULTS_CSV = OUTPUT_DIR / "usgs_discharge_probe_results.csv"
SAMPLE_PARQUET = OUTPUT_DIR / "probe_basin_sample.parquet"
SAMPLE_CSV = OUTPUT_DIR / "probe_basin_sample.csv"

# Probe window: one full water year for RBI screening.
PROBE_START = pd.Timestamp("2023-10-01", tz="UTC")
PROBE_END = pd.Timestamp("2024-09-30 23:00:00", tz="UTC")
EXPECTED_HOURLY_INDEX = pd.date_range(start=PROBE_START, end=PROBE_END, freq="1h")
EXPECTED_HOURLY_COUNT = len(EXPECTED_HOURLY_INDEX)

# Request behavior
USGS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
USER_AGENT = "Flash-NH USGS discharge probe (metadata-lite real-data validation)"
REQUEST_DELAY_SECONDS = 0.4
MAX_RETRIES = 5
BACKOFF_INITIAL_SECONDS = 1.5
TIMEOUT_SECONDS = 90

# Sampling design
AREA_BIN_LABELS = ["1-10 km²", "10-100 km²", "100-1000 km²"]
BFI_BIN_LABELS = ["<=20", "20-30", "30-40", "40-50", ">50"]
DEFAULT_TARGET_SAMPLE_SIZE = 75
DEFAULT_PER_STRATUM = 5

STATUS_ORDER = ["RBI_READY", "PARTIAL_USABLE", "INSUFFICIENT", "NO_DATA", "ERROR"]
STATUS_COLORS = {
    "RBI_READY": "#2ca25f",
    "PARTIAL_USABLE": "#fe9929",
    "INSUFFICIENT": "#de2d26",
    "NO_DATA": "#9e9e9e",
    "ERROR": "#756bb1",
}

TIMESTEP_ORDER = ["hourly", "sub-hourly", "daily", "irregular", "sparse"]
TIMESTEP_COLORS = {
    "hourly": "#2b8cbe",
    "sub-hourly": "#31a354",
    "daily": "#fd8d3c",
    "irregular": "#756bb1",
    "sparse": "#bdbdbd",
}


@dataclass
class ProbeResult:
    STAID: str
    DRAIN_SQKM: float
    BFI_AVE: float
    area_bin: str
    BFI_bin: str
    LAT_GAGE: Optional[float]
    LNG_GAGE: Optional[float]
    STATE: Optional[str]
    HUC02: Optional[str]
    request_success: bool
    request_http_status: Optional[int]
    request_error: Optional[str]
    usgs_site_valid: bool
    has_parameter_00060: bool
    returned_observation_count: int
    first_timestamp_utc: Optional[str]
    last_timestamp_utc: Optional[str]
    native_timestep_minutes: Optional[float]
    native_timestep_share: Optional[float]
    inferred_timestep_mode: str
    hourly_values_count: int
    expected_hourly_count: int
    hourly_completeness_pct: float
    units_original: Optional[str]
    units_output: str
    unit_conversion_applied: bool
    completeness_class: str
    preliminary_status: str
    rbi: Optional[float]
    max_hourly_dqdt_m3s_per_hr: Optional[float]
    normalized_max_hourly_dqdt: Optional[float]
    q95_event_count: Optional[int]
    q99_event_count: Optional[int]
    notes: str
    probe_stratum: str
    probe_rank_in_stratum: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small USGS discharge probe for Flash-NH")
    parser.add_argument("--target-sample-size", type=int, default=DEFAULT_TARGET_SAMPLE_SIZE, help="Desired probe sample size")
    parser.add_argument("--per-stratum", type=int, default=DEFAULT_PER_STRATUM, help="Max basins per area/BFI stratum")
    parser.add_argument("--refresh-sample", action="store_true", help="Rebuild the probe sample even if it already exists")
    parser.add_argument("--refresh-results", action="store_true", help="Ignore existing probe results and re-download everything")
    parser.add_argument("--max-sites", type=int, default=None, help="Optional limit for testing")
    return parser.parse_args()


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("flashnh_usgs_discharge_probe")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(LOG_DIR / "usgs_discharge_probe.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def load_probe_candidates() -> pd.DataFrame:
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Missing input parquet: {INPUT_PARQUET}")

    candidates = pd.read_parquet(INPUT_PARQUET).copy()
    candidates["STAID"] = candidates["STAID"].astype(str).str.zfill(8)

    if "area_bin" not in candidates.columns:
        candidates["area_bin"] = pd.cut(
            candidates["DRAIN_SQKM"],
            bins=[1, 10, 100, 1000],
            labels=AREA_BIN_LABELS,
            include_lowest=True,
        )
    if "BFI_bin" not in candidates.columns:
        candidates["BFI_bin"] = pd.cut(
            candidates["BFI_AVE"],
            bins=[-math.inf, 20, 30, 40, 50, math.inf],
            labels=BFI_BIN_LABELS,
            include_lowest=True,
        )

    required_mask = pd.Series(True, index=candidates.index)
    if "preliminary_status" in candidates.columns:
        required_mask &= candidates["preliminary_status"].eq("PARTIAL")
    if "has_parameter_00060" in candidates.columns:
        required_mask &= candidates["has_parameter_00060"].fillna(False)
    if "usgs_site_valid" in candidates.columns:
        required_mask &= candidates["usgs_site_valid"].fillna(False)

    probe_universe = candidates.loc[required_mask].copy()
    probe_universe = probe_universe.dropna(subset=["DRAIN_SQKM", "BFI_AVE", "LAT_GAGE", "LNG_GAGE"])
    probe_universe = probe_universe[probe_universe["area_bin"].isin(AREA_BIN_LABELS)]
    probe_universe = probe_universe[probe_universe["BFI_bin"].isin(BFI_BIN_LABELS)]
    return probe_universe


def spaced_sample(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(frame) <= n:
        result = frame.copy()
        result["probe_rank_in_stratum"] = np.arange(1, len(result) + 1)
        return result

    ordered = frame.sort_values(["LNG_GAGE", "LAT_GAGE", "STAID"]).reset_index(drop=True)
    positions = np.unique(np.round(np.linspace(0, len(ordered) - 1, n)).astype(int))
    chosen = ordered.iloc[positions].copy()
    if len(chosen) < n:
        remaining = ordered.drop(index=positions)
        top_up = remaining.head(n - len(chosen))
        chosen = pd.concat([chosen, top_up], ignore_index=True)
    chosen["probe_rank_in_stratum"] = np.arange(1, len(chosen) + 1)
    return chosen


def select_probe_sample(candidates: pd.DataFrame, per_stratum: int, target_sample_size: int) -> pd.DataFrame:
    if candidates.empty:
        raise RuntimeError("No candidates available for the discharge probe")

    sampled_frames = []
    for area_bin in AREA_BIN_LABELS:
        area_subset = candidates[candidates["area_bin"] == area_bin]
        for bfi_bin in BFI_BIN_LABELS:
            stratum = area_subset[area_subset["BFI_bin"] == bfi_bin].copy()
            if stratum.empty:
                continue
            picked = spaced_sample(stratum, per_stratum)
            picked["probe_stratum"] = f"{area_bin} | {bfi_bin}"
            sampled_frames.append(picked)

    sample = pd.concat(sampled_frames, ignore_index=True)
    sample = sample.drop_duplicates(subset=["STAID"]).copy()
    sample = sample.sort_values(["area_bin", "BFI_bin", "LNG_GAGE", "LAT_GAGE", "STAID"]).reset_index(drop=True)

    if len(sample) > target_sample_size:
        # Preserve stratum coverage while trimming to the target size.
        sample = sample.head(target_sample_size).copy()

    sample["probe_order"] = np.arange(1, len(sample) + 1)
    return sample


def load_existing_results() -> pd.DataFrame:
    if RESULTS_PARQUET.exists():
        return pd.read_parquet(RESULTS_PARQUET)
    if RESULTS_CSV.exists():
        return pd.read_csv(RESULTS_CSV, dtype={"STAID": str})
    return pd.DataFrame()


def save_sample(sample: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(SAMPLE_PARQUET, index=False)
    sample.to_csv(SAMPLE_CSV, index=False)


def save_results(results: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_parquet(RESULTS_PARQUET, index=False)
    results.to_csv(RESULTS_CSV, index=False)


def request_iv_json(session: requests.Session, site_no: str, logger: logging.Logger) -> tuple[dict, int]:
    params = {
        "sites": site_no,
        "parameterCd": "00060",
        "startDT": PROBE_START.date().isoformat(),
        "endDT": PROBE_END.date().isoformat(),
        "format": "json",
        "siteStatus": "all",
    }
    backoff = BACKOFF_INITIAL_SECONDS
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(USGS_IV_URL, params=params, timeout=TIMEOUT_SECONDS)
            status = response.status_code
            response.raise_for_status()
            payload = response.json()
            time.sleep(REQUEST_DELAY_SECONDS)
            return payload, status
        except Exception as exc:  # noqa: BLE001 - explicit retry loop
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            logger.warning("USGS IV request failed for %s (attempt %s/%s): %s", site_no, attempt, MAX_RETRIES, exc)
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError(f"USGS IV request failed for {site_no}") from last_error


def flatten_iv_payload(payload: dict) -> tuple[pd.DataFrame, dict[str, object]]:
    value_block = payload.get("value", {})
    time_series = value_block.get("timeSeries", []) if isinstance(value_block, dict) else []

    rows = []
    series_meta = []
    for series in time_series:
        variable = series.get("variable", {})
        variable_codes = variable.get("variableCode", []) or []
        code_values = {str(item.get("value")) for item in variable_codes if isinstance(item, dict)}
        if "00060" not in code_values and "00060" not in str(variable.get("variableDescription", "")):
            continue

        unit_code = variable.get("unit", {}).get("unitCode")
        source_info = series.get("sourceInfo", {})
        site_no = source_info.get("siteCode", [{}])[0].get("value") if source_info.get("siteCode") else None
        site_name = source_info.get("siteName")
        series_meta.append({
            "site_no": site_no,
            "site_name": site_name,
            "unit_code": unit_code,
            "series_name": variable.get("variableName") or variable.get("variableDescription"),
        })

        for values_block in series.get("values", []):
            for item in values_block.get("value", []):
                rows.append({
                    "site_no": site_no,
                    "dateTime": item.get("dateTime"),
                    "value": item.get("value"),
                    "qualifiers": ";".join(item.get("qualifiers", [])) if item.get("qualifiers") else None,
                    "unit_code": unit_code,
                })

    frame = pd.DataFrame(rows)
    meta = {
        "series_count": len(time_series),
        "selected_series_count": len(series_meta),
        "site_name": series_meta[0]["site_name"] if series_meta else None,
        "unit_code": series_meta[0]["unit_code"] if series_meta else None,
        "series_name": series_meta[0]["series_name"] if series_meta else None,
    }
    return frame, meta


def convert_units(series: pd.Series, unit_code: Optional[str]) -> tuple[pd.Series, str, bool]:
    unit_text = (unit_code or "").strip().lower()
    if unit_text in {"ft3/s", "cfs", "ft^3/s", "cubic feet per second"}:
        return series * 0.028316846592, "m3/s", True
    if unit_text in {"m3/s", "m^3/s", "cms", "cubic meters per second"}:
        return series, "m3/s", False
    return series, unit_code or "unknown", False


def infer_native_timestep(timestamp_index: pd.DatetimeIndex) -> tuple[Optional[float], float, str]:
    if len(timestamp_index) < 2:
        return None, 0.0, "sparse"

    sorted_index = pd.DatetimeIndex(timestamp_index).sort_values().unique()
    if len(sorted_index) < 2:
        return None, 0.0, "sparse"

    deltas_minutes = pd.Series(sorted_index[1:] - sorted_index[:-1]).dt.total_seconds() / 60.0
    if deltas_minutes.empty:
        return None, 0.0, "sparse"

    rounded = deltas_minutes.round().astype(int)
    mode_value = float(rounded.mode().iloc[0])
    mode_share = float((rounded == mode_value).mean())

    if len(rounded) < 5:
        return mode_value, mode_share, "sparse"
    if 50 <= mode_value <= 70 and mode_share >= 0.4:
        return mode_value, mode_share, "hourly"
    if mode_value < 50 and mode_share >= 0.35:
        return mode_value, mode_share, "sub-hourly"
    if 1300 <= mode_value <= 1500 and mode_share >= 0.35:
        return mode_value, mode_share, "daily"
    if mode_share < 0.35 and rounded.nunique() > 4:
        return mode_value, mode_share, "irregular"
    return mode_value, mode_share, "irregular"


def build_hourly_series(raw_series_converted: pd.Series) -> pd.Series:
    hourly = raw_series_converted.resample("1h").mean()
    return hourly.reindex(EXPECTED_HOURLY_INDEX)


def compute_probe_metrics(hourly_series: pd.Series) -> tuple[Optional[float], Optional[float], Optional[float], Optional[int], Optional[int]]:
    valid = hourly_series.dropna()
    if len(valid) < 2:
        return None, None, None, None, None

    total = valid.sum()
    if not np.isfinite(total) or total <= 0:
        return None, None, None, None, None

    consecutive_mask = valid.index.to_series().diff().eq(pd.Timedelta(hours=1))
    diffs = valid.diff().abs().where(consecutive_mask).dropna()
    if diffs.empty:
        return None, None, None, None, None
    rbi = float(diffs.sum() / total)
    max_hourly_dqdt = float(diffs.max()) if not diffs.empty else None
    normalized = float(max_hourly_dqdt / valid.mean()) if max_hourly_dqdt is not None and valid.mean() > 0 else None

    q95 = valid.quantile(0.95)
    q99 = valid.quantile(0.99)
    q95_count = int((valid >= q95).sum())
    q99_count = int((valid >= q99).sum())
    return rbi, max_hourly_dqdt, normalized, q95_count, q99_count


calculate_probe_metrics = compute_probe_metrics


def process_site(site_row: pd.Series, session: requests.Session, logger: logging.Logger) -> ProbeResult:
    site_no = str(site_row["STAID"]).zfill(8)
    base_kwargs = dict(
        STAID=site_no,
        DRAIN_SQKM=float(site_row["DRAIN_SQKM"]),
        BFI_AVE=float(site_row["BFI_AVE"]),
        area_bin=str(site_row["area_bin"]),
        BFI_bin=str(site_row["BFI_bin"]),
        LAT_GAGE=float(site_row["LAT_GAGE"]) if pd.notna(site_row.get("LAT_GAGE")) else None,
        LNG_GAGE=float(site_row["LNG_GAGE"]) if pd.notna(site_row.get("LNG_GAGE")) else None,
        STATE=site_row.get("STATE"),
        HUC02=site_row.get("HUC02"),
    )

    try:
        payload, http_status = request_iv_json(session, site_no, logger)
    except Exception as exc:  # noqa: BLE001 - network error handling
        return ProbeResult(
            **base_kwargs,
            request_success=False,
            request_http_status=None,
            request_error=str(exc),
            usgs_site_valid=True,
            has_parameter_00060=True,
            returned_observation_count=0,
            first_timestamp_utc=None,
            last_timestamp_utc=None,
            native_timestep_minutes=None,
            native_timestep_share=None,
            inferred_timestep_mode="sparse",
            hourly_values_count=0,
            expected_hourly_count=EXPECTED_HOURLY_COUNT,
            hourly_completeness_pct=0.0,
            units_original=None,
            units_output="m3/s",
            unit_conversion_applied=False,
            completeness_class="INSUFFICIENT",
            preliminary_status="ERROR",
            rbi=None,
            max_hourly_dqdt_m3s_per_hr=None,
            normalized_max_hourly_dqdt=None,
            q95_event_count=None,
            q99_event_count=None,
            notes=str(exc),
            probe_stratum=f"{site_row['area_bin']} | {site_row['BFI_bin']}",
            probe_rank_in_stratum=int(site_row.get("probe_rank_in_stratum", 0)),
        )

    raw_frame, meta = flatten_iv_payload(payload)
    if raw_frame.empty:
        return ProbeResult(
            **base_kwargs,
            request_success=True,
            request_http_status=http_status,
            request_error=None,
            usgs_site_valid=True,
            has_parameter_00060=True,
            returned_observation_count=0,
            first_timestamp_utc=None,
            last_timestamp_utc=None,
            native_timestep_minutes=None,
            native_timestep_share=None,
            inferred_timestep_mode="sparse",
            hourly_values_count=0,
            expected_hourly_count=EXPECTED_HOURLY_COUNT,
            hourly_completeness_pct=0.0,
            units_original=meta.get("unit_code"),
            units_output="m3/s",
            unit_conversion_applied=False,
            completeness_class="INSUFFICIENT",
            preliminary_status="NO_DATA",
            rbi=None,
            max_hourly_dqdt_m3s_per_hr=None,
            normalized_max_hourly_dqdt=None,
            q95_event_count=None,
            q99_event_count=None,
            notes="No 00060 observations returned for the probe window",
            probe_stratum=f"{site_row['area_bin']} | {site_row['BFI_bin']}",
            probe_rank_in_stratum=int(site_row.get("probe_rank_in_stratum", 0)),
        )

    raw_frame = raw_frame.copy()
    raw_frame["dateTime"] = pd.to_datetime(raw_frame["dateTime"], utc=True, errors="coerce")
    raw_frame["value"] = pd.to_numeric(raw_frame["value"], errors="coerce")
    raw_frame = raw_frame.dropna(subset=["dateTime", "value"])

    if raw_frame.empty:
        return ProbeResult(
            **base_kwargs,
            request_success=True,
            request_http_status=http_status,
            request_error=None,
            usgs_site_valid=True,
            has_parameter_00060=True,
            returned_observation_count=0,
            first_timestamp_utc=None,
            last_timestamp_utc=None,
            native_timestep_minutes=None,
            native_timestep_share=None,
            inferred_timestep_mode="sparse",
            hourly_values_count=0,
            expected_hourly_count=EXPECTED_HOURLY_COUNT,
            hourly_completeness_pct=0.0,
            units_original=meta.get("unit_code"),
            units_output="m3/s",
            unit_conversion_applied=False,
            completeness_class="INSUFFICIENT",
            preliminary_status="NO_DATA",
            rbi=None,
            max_hourly_dqdt_m3s_per_hr=None,
            normalized_max_hourly_dqdt=None,
            q95_event_count=None,
            q99_event_count=None,
            notes="Returned observations could not be parsed",
            probe_stratum=f"{site_row['area_bin']} | {site_row['BFI_bin']}",
            probe_rank_in_stratum=int(site_row.get("probe_rank_in_stratum", 0)),
        )

    raw_frame = raw_frame.sort_values("dateTime")
    raw_series = raw_frame.set_index("dateTime")["value"]
    raw_series = raw_series.groupby(level=0).mean().sort_index()

    converted_values, units_output, conversion_applied = convert_units(raw_series, meta.get("unit_code"))
    raw_series_converted = converted_values.groupby(level=0).mean().sort_index()
    native_timestep_minutes, native_timestep_share, timestep_mode = infer_native_timestep(raw_series_converted.index)

    hourly = build_hourly_series(raw_series_converted)
    hourly_values_count = int(hourly.notna().sum())
    completeness_pct = 100.0 * hourly_values_count / EXPECTED_HOURLY_COUNT

    if completeness_pct >= 90.0:
        completeness_class = "RBI_READY"
    elif completeness_pct >= 70.0:
        completeness_class = "PARTIAL_USABLE"
    else:
        completeness_class = "INSUFFICIENT"

    if completeness_class == "RBI_READY":
        preliminary_status = "RBI_READY"
    elif completeness_class == "PARTIAL_USABLE":
        preliminary_status = "PARTIAL_USABLE"
    else:
        preliminary_status = "INSUFFICIENT"

    rbi = max_hourly_dqdt = normalized_max_hourly_dqdt = q95_count = q99_count = None
    notes = []
    if completeness_class in {"RBI_READY", "PARTIAL_USABLE"}:
        rbi, max_hourly_dqdt, normalized_max_hourly_dqdt, q95_count, q99_count = compute_probe_metrics(hourly)
    else:
        notes.append("Hourly completeness below 70%; RBI metrics withheld")

    if timestep_mode == "hourly":
        notes.append("Direct hourly data appear available")
    elif timestep_mode == "sub-hourly":
        notes.append("Native sub-hourly data required hourly resampling")
    elif timestep_mode == "daily":
        notes.append("Daily or coarser cadence detected")
    else:
        notes.append("Native cadence is irregular or sparse")

    return ProbeResult(
        **base_kwargs,
        request_success=True,
        request_http_status=http_status,
        request_error=None,
        usgs_site_valid=True,
        has_parameter_00060=True,
        returned_observation_count=int(len(raw_series_converted)),
        first_timestamp_utc=raw_series_converted.index.min().isoformat(),
        last_timestamp_utc=raw_series_converted.index.max().isoformat(),
        native_timestep_minutes=native_timestep_minutes,
        native_timestep_share=native_timestep_share,
        inferred_timestep_mode=timestep_mode,
        hourly_values_count=hourly_values_count,
        expected_hourly_count=EXPECTED_HOURLY_COUNT,
        hourly_completeness_pct=round(completeness_pct, 3),
        units_original=meta.get("unit_code"),
        units_output=units_output,
        unit_conversion_applied=conversion_applied,
        completeness_class=completeness_class,
        preliminary_status=preliminary_status,
        rbi=rbi,
        max_hourly_dqdt_m3s_per_hr=max_hourly_dqdt,
        normalized_max_hourly_dqdt=normalized_max_hourly_dqdt,
        q95_event_count=q95_count,
        q99_event_count=q99_count,
        notes="; ".join(notes),
        probe_stratum=f"{site_row['area_bin']} | {site_row['BFI_bin']}",
        probe_rank_in_stratum=int(site_row.get("probe_rank_in_stratum", 0)),
    )


def merge_results(existing: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    combined = pd.DataFrame(new_rows) if existing.empty else pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    if combined.empty:
        return combined
    combined["STAID"] = combined["STAID"].astype(str).str.zfill(8)
    combined = combined.drop_duplicates(subset=["STAID"], keep="last")
    combined = combined.sort_values(["preliminary_status", "area_bin", "BFI_bin", "STAID"]).reset_index(drop=True)
    return combined


def build_summary(results: pd.DataFrame, sample: pd.DataFrame) -> dict[str, object]:
    status_counts = results["preliminary_status"].value_counts().reindex(STATUS_ORDER, fill_value=0).to_dict()
    timestep_counts = results["inferred_timestep_mode"].value_counts().reindex(TIMESTEP_ORDER, fill_value=0).to_dict()
    completeness_bins = pd.cut(results["hourly_completeness_pct"], bins=[-0.1, 70, 90, 100.1], labels=["<70", "70-90", ">=90"]).value_counts().sort_index().to_dict()

    summary = {
        "probe_window": {
            "start": PROBE_START.isoformat(),
            "end": PROBE_END.isoformat(),
            "expected_hourly_count": EXPECTED_HOURLY_COUNT,
        },
        "sample_size": int(len(sample)),
        "status_counts": status_counts,
        "timestep_counts": timestep_counts,
        "completeness_bins": {str(k): int(v) for k, v in completeness_bins.items()},
        "rbi_ready_count": int((results["preliminary_status"] == "RBI_READY").sum()),
        "partial_usable_count": int((results["preliminary_status"] == "PARTIAL_USABLE").sum()),
        "insufficient_count": int((results["preliminary_status"] == "INSUFFICIENT").sum()),
        "no_data_count": int((results["preliminary_status"] == "NO_DATA").sum()),
        "error_count": int((results["preliminary_status"] == "ERROR").sum()),
        "unit_conversion_count": int(results["unit_conversion_applied"].fillna(False).sum()),
        "hourly_completeness_median": float(results["hourly_completeness_pct"].median()),
        "hourly_completeness_mean": float(results["hourly_completeness_pct"].mean()),
        "rbi_count": int(results["rbi"].notna().sum()),
        "rbi_median": float(results["rbi"].median(skipna=True)) if results["rbi"].notna().any() else None,
        "rbi_q75": float(results["rbi"].quantile(0.75)) if results["rbi"].notna().any() else None,
        "rbi_q90": float(results["rbi"].quantile(0.90)) if results["rbi"].notna().any() else None,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    return summary


def write_summary(results: pd.DataFrame, sample: pd.DataFrame) -> None:
    summary = build_summary(results, sample)
    with open(OUTPUT_DIR / "usgs_discharge_probe_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    lines = [
        "# Flash-NH USGS Discharge Probe Summary",
        "",
        "## Purpose",
        "",
        "This probe validates real USGS NWIS IV discharge retrieval for a stratified sample of area-filtered basins before scaling to the full candidate set.",
        "",
        "## Probe Window",
        "",
        f"- Water year: {PROBE_START.date().isoformat()} to {PROBE_END.date().isoformat()}",
        f"- Expected hourly timestamps: {EXPECTED_HOURLY_COUNT}",
        "",
        "## Sample",
        "",
        f"- Probe sample size: {len(sample)}",
        f"- Sample universe: PARTIAL basins with valid USGS metadata and parameter 00060",
        "",
        "## Status Counts",
        "",
        "| Status | Count |",
        "| --- | ---: |",
    ]
    for status in STATUS_ORDER:
        lines.append(f"| {status} | {summary['status_counts'].get(status, 0)} |")

    lines.extend([
        "",
        "## Hourly Completeness",
        "",
        f"- Median completeness: {summary['hourly_completeness_median']:.1f}%",
        f"- Mean completeness: {summary['hourly_completeness_mean']:.1f}%",
        "",
        "## RBI Readiness",
        "",
        f"- RBI_READY (>=90% completeness): {summary['rbi_ready_count']}",
        f"- PARTIAL_USABLE (70-90% completeness): {summary['partial_usable_count']}",
        f"- INSUFFICIENT (<70% completeness): {summary['insufficient_count']}",
        "",
        "## Native Timestep Summary",
        "",
        "| Inferred timestep | Count |",
        "| --- | ---: |",
    ])
    for timestep in TIMESTEP_ORDER:
        lines.append(f"| {timestep} | {summary['timestep_counts'].get(timestep, 0)} |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- This probe is intended to decide whether the metadata-only audit was too conservative.",
        "- Hourly-preferred handling is used first; native or sub-hourly series are resampled to hourly when needed.",
        "- RBI will only be interpreted for basins with adequate completeness.",
        "- If the probe shows acceptable coverage for a meaningful fraction of the sample, the same workflow can be scaled to the full area-filtered basin set.",
    ])

    with open(OUTPUT_DIR / "usgs_discharge_probe_summary.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def generate_plots(results: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = results.copy()

    # 1. Status counts
    status_counts = plot_df["preliminary_status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(status_counts.index, status_counts.values, color=[STATUS_COLORS[s] for s in status_counts.index], edgecolor="black")
    for bar, value in zip(bars, status_counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(status_counts.values) * 0.01, f"{int(value)}", ha="center", va="bottom")
    plt.ylabel("Probe basin count")
    plt.title("USGS Discharge Probe Status Counts")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "probe_status_counts.png", dpi=160)
    plt.close()

    # 2. Completeness distribution
    plt.figure(figsize=(10, 6))
    plt.hist(plot_df["hourly_completeness_pct"], bins=20, edgecolor="black", color="#2b8cbe", alpha=0.8)
    plt.axvline(70, color="#fe9929", linestyle="--", linewidth=2, label="70% threshold")
    plt.axvline(90, color="#de2d26", linestyle="--", linewidth=2, label="90% threshold")
    plt.xlabel("Hourly completeness (%)")
    plt.ylabel("Probe basin count")
    plt.title("Hourly Completeness Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "hourly_completeness_distribution.png", dpi=160)
    plt.close()

    # 3. Native timestep distribution
    timestep_counts = plot_df["inferred_timestep_mode"].value_counts().reindex(TIMESTEP_ORDER, fill_value=0)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(timestep_counts.index, timestep_counts.values, color=[TIMESTEP_COLORS[t] for t in timestep_counts.index], edgecolor="black")
    for bar, value in zip(bars, timestep_counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(timestep_counts.values) * 0.01, f"{int(value)}", ha="center", va="bottom")
    plt.ylabel("Probe basin count")
    plt.title("Inferred Native Timestep Distribution")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "inferred_timestep_distribution.png", dpi=160)
    plt.close()

    metric_df = plot_df[plot_df["rbi"].notna()].copy()
    if metric_df.empty:
        return

    # 4. RBI distribution
    plt.figure(figsize=(10, 6))
    plt.hist(metric_df["rbi"], bins=20, edgecolor="black", color="#31a354", alpha=0.8)
    plt.xlabel("RBI")
    plt.ylabel("Probe basin count")
    plt.title("RBI Distribution for Probe Basins with Adequate Completeness")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rbi_distribution_probe.png", dpi=160)
    plt.close()

    # 5. RBI by area bin
    box_df = metric_df.copy()
    box_df["area_bin"] = pd.Categorical(box_df["area_bin"], categories=AREA_BIN_LABELS, ordered=True)
    plt.figure(figsize=(10, 6))
    box_df.boxplot(column="rbi", by="area_bin")
    plt.suptitle("")
    plt.title("RBI by Drainage Area Bin")
    plt.xlabel("Area bin")
    plt.ylabel("RBI")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rbi_by_area_bin_probe.png", dpi=160)
    plt.close()

    # 6. RBI vs BFI
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(metric_df["BFI_AVE"], metric_df["rbi"], c=metric_df["hourly_completeness_pct"], cmap="viridis", s=35, alpha=0.85, edgecolors="none")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Hourly completeness (%)")
    plt.xlabel("BFI_AVE")
    plt.ylabel("RBI")
    plt.title("RBI vs BFI for Probe Basins")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rbi_vs_bfi_probe.png", dpi=160)
    plt.close()

    # 7. RBI vs drainage area
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(metric_df["DRAIN_SQKM"], metric_df["rbi"], c=metric_df["hourly_completeness_pct"], cmap="viridis", s=35, alpha=0.85, edgecolors="none")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Hourly completeness (%)")
    plt.xscale("log")
    plt.xlabel("Drainage area (km², log scale)")
    plt.ylabel("RBI")
    plt.title("RBI vs Drainage Area for Probe Basins")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rbi_vs_drainage_area_probe.png", dpi=160)
    plt.close()

    if plot_df["LNG_GAGE"].notna().any() and plot_df["LAT_GAGE"].notna().any():
        lon_min, lon_max = plot_df["LNG_GAGE"].min(), plot_df["LNG_GAGE"].max()
        lat_min, lat_max = plot_df["LAT_GAGE"].min(), plot_df["LAT_GAGE"].max()
        lon_pad = max(0.5, (lon_max - lon_min) * 0.03)
        lat_pad = max(0.5, (lat_max - lat_min) * 0.03)
        x_limits = (lon_min - lon_pad, lon_max + lon_pad)
        y_limits = (lat_min - lat_pad, lat_max + lat_pad)

        # 8. Map by status
        plt.figure(figsize=(14, 8))
        for status in STATUS_ORDER:
            subset = plot_df[plot_df["preliminary_status"] == status]
            plt.scatter(subset["LNG_GAGE"], subset["LAT_GAGE"], c=STATUS_COLORS[status], s=28, alpha=0.85, label=f"{status} ({len(subset)})")
        plt.xlim(x_limits)
        plt.ylim(y_limits)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Probe Basin Status Map")
        plt.grid(True, alpha=0.25)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "map_probe_status.png", dpi=160)
        plt.close()

        # 9. Map RBI
        plt.figure(figsize=(14, 8))
        unavailable = plot_df[plot_df["rbi"].isna()]
        ready = plot_df[plot_df["rbi"].notna()]
        plt.scatter(unavailable["LNG_GAGE"], unavailable["LAT_GAGE"], c="#d9d9d9", s=22, alpha=0.55, label=f"No RBI ({len(unavailable)})")
        if not ready.empty:
            scatter = plt.scatter(ready["LNG_GAGE"], ready["LAT_GAGE"], c=ready["rbi"], cmap="magma", s=30, alpha=0.9, edgecolors="none")
            cbar = plt.colorbar(scatter)
            cbar.set_label("RBI")
        plt.xlim(x_limits)
        plt.ylim(y_limits)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Probe Basin RBI Map")
        plt.grid(True, alpha=0.25)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "map_probe_rbi.png", dpi=160)
        plt.close()


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    logger.info("Starting Flash-NH USGS discharge probe")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    candidates = load_probe_candidates()
    logger.info("Loaded %s probe candidates", len(candidates))

    sample_path_exists = SAMPLE_PARQUET.exists() and SAMPLE_CSV.exists()
    if sample_path_exists and not args.refresh_sample:
        sample = pd.read_parquet(SAMPLE_PARQUET)
        sample["STAID"] = sample["STAID"].astype(str).str.zfill(8)
        logger.info("Loaded existing probe sample with %s basins", len(sample))
    else:
        sample = select_probe_sample(candidates, per_stratum=args.per_stratum, target_sample_size=args.target_sample_size)
        logger.info("Selected new probe sample with %s basins", len(sample))
        save_sample(sample)

    if args.max_sites is not None:
        sample = sample.head(args.max_sites).copy()
        logger.info("Restricted probe to %s sites for testing", len(sample))

    existing = load_existing_results()
    if args.refresh_results or existing.empty:
        existing = pd.DataFrame()
        done_sites: set[str] = set()
    else:
        existing["STAID"] = existing["STAID"].astype(str).str.zfill(8)
        done_sites = set(existing["STAID"].tolist())

    logger.info("Resuming with %s already-probed sites", len(done_sites))
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    result_rows = [] if existing.empty else existing.to_dict(orient="records")
    processed = len(done_sites)

    for _, site_row in sample.iterrows():
        staid = str(site_row["STAID"]).zfill(8)
        if staid in done_sites:
            continue

        logger.info(
            "Probing %s | area=%s | BFI=%s | completeness so far: %s/%s",
            staid,
            site_row["area_bin"],
            site_row["BFI_bin"],
            processed,
            len(sample),
        )
        result = process_site(site_row, session, logger)
        result_dict = result.__dict__.copy()
        result_dict["probe_order"] = int(site_row.get("probe_order", len(result_rows) + 1))
        result_rows.append(result_dict)
        done_sites.add(staid)
        processed += 1

        if processed % 25 == 0 or processed == len(sample):
            partial = pd.DataFrame(result_rows)
            if not partial.empty:
                partial["STAID"] = partial["STAID"].astype(str).str.zfill(8)
                save_results(partial)
                logger.info("Checkpointed %s probe rows", len(partial))

    results = pd.DataFrame(result_rows)
    if results.empty:
        raise RuntimeError("No discharge probe results were produced")

    results["STAID"] = results["STAID"].astype(str).str.zfill(8)
    results = results.drop_duplicates(subset=["STAID"], keep="last").reset_index(drop=True)
    results = results.sort_values(["probe_order", "STAID"]).reset_index(drop=True)

    save_results(results)
    write_summary(results, sample)
    generate_plots(results)

    logger.info("Probe complete: %s rows", len(results))
    logger.info("Outputs written to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
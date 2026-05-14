#!/usr/bin/env python3
"""Event-centered hydrograph review for WY2024 USGS RBI screening results.

This v002 workflow keeps the run bounded, reuses cached hourly series when
available, downloads only selected review basins when needed, and writes a
compact review bundle focused on event-centered hydrographs.

The review sample is stratified across RBI bands and intentionally forced to
include basins that were visually useful in the manual review:
- 01521500
- 07382000
- 02310700
- 09513860

Outputs are written under reports/flashnh_usgs_event_hydrograph_review_v002/.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from scripts.usgs_discharge_probe import build_hourly_series, convert_units, flatten_iv_payload
from scripts.usgs_rbi_screening_scale import fetch_iv_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "reports" / "flashnh_usgs_rbi_screening_wy2024_v001"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "flashnh_usgs_event_hydrograph_review_v002"
DEFAULT_CACHE_DIR = ROOT / "reports" / "flashnh_usgs_event_hydrograph_review_v001" / "hourly_series"
DEFAULT_MAX_REVIEW_BASINS = 96
WY_START = pd.Timestamp("2023-10-01T00:00:00Z")
WY_END = pd.Timestamp("2024-09-30T23:00:00Z")
EXPECTED_HOURLY_INDEX = pd.date_range(start=WY_START, end=WY_END, freq="1h")
MANUAL_INCLUDE_STAIDS = ["01521500", "07382000", "02310700", "09513860"]
EVENT_PRE_HOURS = 72
EVENT_POST_HOURS = 120
EVENT_MIN_SEPARATION_HOURS = 72

REVIEW_REASON_PRIORITY = {
    "manual_include": 0,
    "top_rbi": 1,
    "top_normalized_jump": 2,
    "low_rbi_high_response": 3,
    "rbi_0p04_0p07": 4,
    "rbi_0p07_0p10": 5,
    "rbi_0p10_0p20": 6,
    "rbi_0p20_0p50": 7,
    "rbi_ge_0p50": 8,
}

SEVERE_QC_FLAGS = {
    "NEGATIVE_FLOW_PRESENT",
    "HIGH_NEGATIVE_FLOW_FRACTION",
    "ZERO_FLOW_DOMINATED",
    "VERY_LOW_SPECIFIC_FLOW",
    "SUSPICIOUS_SPIKE",
    "POSSIBLE_REGULATION_OR_ARTIFACT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Event-centered hydrograph review for USGS RBI screening results")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory containing the completed screening results")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for review outputs")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Directory containing cached hourly CSVs from a prior review run")
    parser.add_argument("--max-review-basins", type=int, default=DEFAULT_MAX_REVIEW_BASINS, help="Cap on the final number of reviewed basins")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for small random samples")
    parser.add_argument("--sleep-seconds", type=float, default=0.15, help="Delay between USGS requests")
    parser.add_argument("--timeout-seconds", type=int, default=45, help="USGS request timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=4, help="Maximum USGS request retries per basin")
    parser.add_argument("--no-download", action="store_true", help="Do not fetch missing hourly series; use cache only")
    parser.add_argument("--force-refresh-selected", action="store_true", help="Ignore cached hourly CSVs and re-download selected basins")
    parser.add_argument("--debug-raw", action="store_true", help="Write raw USGS JSON payloads for the selected basins")
    return parser.parse_args()


def setup_logging(output_dir: Path) -> logging.Logger:
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("usgs_event_hydrograph_review_v002")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(logs_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def find_results_file(results_dir: Path) -> Path:
    csv_path = results_dir / "usgs_rbi_screening_results.csv"
    if csv_path.exists():
        return csv_path
    parquet_path = results_dir / "usgs_rbi_screening_results.parquet"
    if parquet_path.exists():
        return parquet_path
    raise FileNotFoundError(f"Could not find screening results in {results_dir}")


def load_screening_results(results_path: Path) -> pd.DataFrame:
    if results_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(results_path)
    else:
        df = pd.read_csv(results_path)
    df["STAID"] = df["STAID"].astype(str).str.zfill(8)
    numeric_cols = [
        "DRAIN_SQKM",
        "BFI_AVE",
        "LAT_GAGE",
        "LNG_GAGE",
        "rbi",
        "max_hourly_dqdt_m3s_per_hr",
        "normalized_max_hourly_dqdt",
        "q95_event_count",
        "q99_event_count",
        "hourly_completeness_pct",
        "hourly_values_count",
        "expected_hourly_count",
        "returned_observation_count",
        "native_timestep_minutes",
        "native_timestep_share",
        "request_http_status",
        "response_size_bytes",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def band_rbi(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    if value < 0.04:
        return "rbi_lt_0p04"
    if value < 0.07:
        return "rbi_0p04_0p07"
    if value < 0.10:
        return "rbi_0p07_0p10"
    if value < 0.20:
        return "rbi_0p10_0p20"
    if value < 0.50:
        return "rbi_0p20_0p50"
    return "rbi_ge_0p50"


def safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return None
    return float(numerator / denominator)


def combine_reasons(values: Iterable[str]) -> str:
    ordered = []
    for value in values:
        if value not in ordered:
            ordered.append(value)
    ordered.sort(key=lambda item: REVIEW_REASON_PRIORITY.get(item, 999))
    return ";".join(ordered)


def normalize_hourly_dataframe(hourly_df: pd.DataFrame) -> pd.Series:
    frame = hourly_df.copy()
    if "dateTime" not in frame.columns:
        raise ValueError("Hourly CSV missing dateTime column")
    discharge_col = None
    for candidate in ["discharge", "discharge_m3s", "value", "values"]:
        if candidate in frame.columns:
            discharge_col = candidate
            break
    if discharge_col is None:
        remaining = [c for c in frame.columns if c != "dateTime"]
        if not remaining:
            raise ValueError("Hourly CSV does not contain a discharge column")
        discharge_col = remaining[0]

    frame["dateTime"] = pd.to_datetime(frame["dateTime"], utc=True, errors="coerce")
    frame[discharge_col] = pd.to_numeric(frame[discharge_col], errors="coerce")
    frame = frame.dropna(subset=["dateTime"])
    frame = frame.set_index("dateTime")[discharge_col].sort_index()
    frame.name = "discharge"
    return build_hourly_series(frame)


def read_cached_hourly(cache_dir: Path, staid: str) -> Optional[pd.Series]:
    cache_file = cache_dir / f"{staid}_hourly.csv"
    if not cache_file.exists():
        return None
    return normalize_hourly_dataframe(pd.read_csv(cache_file))


def write_hourly_csv(hourly: pd.Series, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    hourly_df = hourly.reset_index()
    hourly_df.columns = ["dateTime", "discharge"]
    hourly_df.to_csv(output_file, index=False)


def load_or_fetch_hourly(
    session: requests.Session,
    staid: str,
    output_hourly_dir: Path,
    cache_dir: Path,
    logger: logging.Logger,
    sleep_seconds: float,
    max_retries: int,
    timeout_seconds: int,
    no_download: bool,
    force_refresh_selected: bool,
    debug_raw: bool,
    raw_dir: Path,
) -> tuple[Optional[pd.Series], str]:
    output_file = output_hourly_dir / f"{staid}_hourly.csv"

    if not force_refresh_selected and output_file.exists():
        return normalize_hourly_dataframe(pd.read_csv(output_file)), "output_cache"

    if not force_refresh_selected:
        cached = read_cached_hourly(cache_dir, staid)
        if cached is not None:
            write_hourly_csv(cached, output_file)
            return cached, "shared_cache"

    if no_download:
        return None, "missing_cache_no_download"

    payload, http_status, response_size, request_url = fetch_iv_json(
        session=session,
        site_no=staid,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        logger=logger,
    )
    if debug_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"{staid}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    values, meta = flatten_iv_payload(payload)
    if values.empty:
        logger.warning("No IV values returned for %s (%s, %s bytes)", staid, http_status, response_size)
        return None, "empty_response"

    units = meta.get("unit_code")
    if "value" not in values.columns:
        raise ValueError(f"Could not find value column in USGS payload for {staid}")
    values["value"] = pd.to_numeric(values["value"], errors="coerce")

    converted, _, _ = convert_units(values["value"], units)
    raw_series = pd.Series(converted.values, index=pd.DatetimeIndex(pd.to_datetime(values["dateTime"], utc=True)), name="discharge")
    hourly = build_hourly_series(raw_series.sort_index())
    write_hourly_csv(hourly, output_file)
    return hourly, f"downloaded:{http_status}:{Path(request_url).name}"


def find_local_peaks(series: pd.Series) -> pd.DataFrame:
    valid = series.dropna()
    if len(valid) < 3:
        return pd.DataFrame(columns=["peak_time", "peak_q"])

    prev_vals = valid.shift(1)
    next_vals = valid.shift(-1)
    is_peak = ((valid >= prev_vals) & (valid > next_vals)) | ((valid > prev_vals) & (valid >= next_vals))
    peaks = pd.DataFrame({"peak_time": valid.index[is_peak], "peak_q": valid[is_peak].values})
    return peaks.sort_values("peak_q", ascending=False).reset_index(drop=True)


def select_peaks(peaks: pd.DataFrame, series: pd.Series, q95: float, q99: float, max_events: int = 5) -> list[dict]:
    chosen: list[dict] = []
    if peaks.empty:
        fallback = series.dropna().sort_values(ascending=False).head(max_events)
        for peak_time, peak_q in fallback.items():
            chosen.append({
                "peak_time": pd.Timestamp(peak_time),
                "peak_q": float(peak_q),
                "threshold": "fallback_high_value",
            })
        return chosen

    for row in peaks.itertuples(index=False):
        peak_time = pd.Timestamp(row.peak_time)
        if any(abs((peak_time - candidate["peak_time"]).total_seconds()) < EVENT_MIN_SEPARATION_HOURS * 3600 for candidate in chosen):
            continue
        threshold = "q99" if row.peak_q >= q99 else "q95" if row.peak_q >= q95 else "peak"
        chosen.append({"peak_time": peak_time, "peak_q": float(row.peak_q), "threshold": threshold})
        if len(chosen) >= max_events:
            break

    if not chosen:
        fallback = series.dropna().sort_values(ascending=False).head(max_events)
        for peak_time, peak_q in fallback.items():
            chosen.append({
                "peak_time": pd.Timestamp(peak_time),
                "peak_q": float(peak_q),
                "threshold": "fallback_high_value",
            })
    return chosen


def event_window_metrics(series: pd.Series, peak_time: pd.Timestamp) -> dict:
    start = max(series.index.min(), peak_time - pd.Timedelta(hours=EVENT_PRE_HOURS))
    end = min(series.index.max(), peak_time + pd.Timedelta(hours=EVENT_POST_HOURS))
    window = series.loc[start:end]
    valid_window = window.dropna()

    if valid_window.empty:
        return {
            "window_start": start,
            "window_end": end,
            "rise_time": None,
            "rise_value": None,
            "rise_jump": None,
            "fall_time": None,
            "fall_value": None,
            "fall_jump": None,
            "window_peak": None,
            "window_q_min": None,
            "window_q_max": None,
            "window_q_mean": None,
            "window_duration_hours": None,
        }

    pre = valid_window.loc[:peak_time]
    post = valid_window.loc[peak_time:]
    pre_diffs = pre.diff().dropna()
    post_diffs = post.diff().dropna()

    rise_time = pre_diffs.idxmax() if not pre_diffs.empty else None
    rise_jump = float(pre_diffs.max()) if not pre_diffs.empty and pd.notna(pre_diffs.max()) else None
    rise_value = float(valid_window.loc[rise_time]) if rise_time is not None and rise_time in valid_window.index else None

    fall_time = post_diffs.idxmin() if not post_diffs.empty else None
    fall_jump = float(post_diffs.min()) if not post_diffs.empty and pd.notna(post_diffs.min()) else None
    fall_value = float(valid_window.loc[fall_time]) if fall_time is not None and fall_time in valid_window.index else None

    return {
        "window_start": start,
        "window_end": end,
        "rise_time": rise_time,
        "rise_value": rise_value,
        "rise_jump": rise_jump,
        "fall_time": fall_time,
        "fall_value": fall_value,
        "fall_jump": fall_jump,
        "window_peak": float(valid_window.max()),
        "window_q_min": float(valid_window.min()),
        "window_q_max": float(valid_window.max()),
        "window_q_mean": float(valid_window.mean()),
        "window_duration_hours": int((end - start) / pd.Timedelta(hours=1)) + 1,
    }


def compute_metrics(series: pd.Series, drain_sqkm: Optional[float]) -> dict:
    aligned = series.reindex(EXPECTED_HOURLY_INDEX)
    valid = aligned.dropna()
    expected_count = len(EXPECTED_HOURLY_INDEX)
    valid_count = int(valid.size)
    completeness_pct = float(valid_count / expected_count * 100.0) if expected_count else float("nan")

    q50 = float(valid.quantile(0.50)) if not valid.empty else float("nan")
    q95 = float(valid.quantile(0.95)) if not valid.empty else float("nan")
    q99 = float(valid.quantile(0.99)) if not valid.empty else float("nan")
    qmin = float(valid.min()) if not valid.empty else float("nan")
    qmax = float(valid.max()) if not valid.empty else float("nan")
    qmean = float(valid.mean()) if not valid.empty else float("nan")
    zero_count = int((valid == 0).sum()) if not valid.empty else 0
    negative_count = int((valid < 0).sum()) if not valid.empty else 0
    nonpositive_count = int((valid <= 0).sum()) if not valid.empty else 0
    zero_flow_fraction = float(zero_count / valid_count) if valid_count else float("nan")
    negative_flow_fraction = float(negative_count / valid_count) if valid_count else float("nan")
    nonpositive_flow_fraction = float(nonpositive_count / valid_count) if valid_count else float("nan")

    diffs = valid.diff().dropna()
    max_rise = float(diffs.max()) if not diffs.empty and pd.notna(diffs.max()) else float("nan")
    max_fall = float(diffs.min()) if not diffs.empty and pd.notna(diffs.min()) else float("nan")
    max_abs_jump = float(diffs.abs().max()) if not diffs.empty and pd.notna(diffs.abs().max()) else float("nan")
    max_rise_time = diffs.idxmax() if not diffs.empty else None
    max_fall_time = diffs.idxmin() if not diffs.empty else None
    max_abs_jump_time = diffs.abs().idxmax() if not diffs.empty else None

    q95_count = int((valid >= q95).sum()) if not valid.empty and pd.notna(q95) else 0
    q99_count = int((valid >= q99).sum()) if not valid.empty and pd.notna(q99) else 0
    peak_candidates = find_local_peaks(aligned)
    peaks_q95 = peak_candidates[peak_candidates["peak_q"] >= q95] if not peak_candidates.empty and pd.notna(q95) else peak_candidates.iloc[0:0]
    peaks_q99 = peak_candidates[peak_candidates["peak_q"] >= q99] if not peak_candidates.empty and pd.notna(q99) else peak_candidates.iloc[0:0]

    selected_peaks = select_peaks(peak_candidates, aligned, q95=q95, q99=q99, max_events=5)
    event_count_q95 = int(len(peaks_q95))
    event_count_q99 = int(len(peaks_q99))

    event_details = []
    for event in selected_peaks:
        window_metrics = event_window_metrics(aligned, event["peak_time"])
        event_details.append({
            **event,
            **window_metrics,
            "peak_over_median": safe_ratio(event["peak_q"], q50),
            "rise_over_median": safe_ratio(window_metrics["rise_jump"], q50),
            "fall_over_median": safe_ratio(abs(window_metrics["fall_jump"]) if window_metrics["fall_jump"] is not None else None, q50),
        })

    top_event = event_details[0] if event_details else {}
    longest_flat_run = 0
    if valid_count >= 2:
        run_ids = (valid != valid.shift()).cumsum()
        flat_lengths = valid.groupby(run_ids).size()
        if not flat_lengths.empty:
            longest_flat_run = int(flat_lengths.max())

    metrics = {
        "hourly_completeness_pct": completeness_pct,
        "valid_hour_count": valid_count,
        "expected_hour_count": expected_count,
        "missing_hour_count": int(expected_count - valid_count),
        "zero_flow_count": zero_count,
        "zero_flow_fraction": zero_flow_fraction,
        "negative_flow_count": negative_count,
        "negative_flow_fraction": negative_flow_fraction,
        "nonpositive_flow_count": nonpositive_count,
        "nonpositive_flow_fraction": nonpositive_flow_fraction,
        "q50": q50,
        "q95": q95,
        "q99": q99,
        "qmin": qmin,
        "qmax": qmax,
        "qmean": qmean,
        "q95_over_q50": safe_ratio(q95, q50),
        "q99_over_q50": safe_ratio(q99, q50),
        "qmax_over_q50": safe_ratio(qmax, q50),
        "max_hourly_rise": max_rise,
        "max_hourly_fall": max_fall,
        "max_abs_hourly_jump": max_abs_jump,
        "max_hourly_rise_time": max_rise_time,
        "max_hourly_fall_time": max_fall_time,
        "max_abs_hourly_jump_time": max_abs_jump_time,
        "max_hourly_rise_over_q50": safe_ratio(max_rise, q50),
        "max_hourly_fall_over_q50": safe_ratio(abs(max_fall) if np.isfinite(max_fall) else None, q50),
        "max_abs_hourly_jump_over_q50": safe_ratio(max_abs_jump, q50),
        "event_count_q95": event_count_q95,
        "event_count_q99": event_count_q99,
        "selected_peak_count": len(selected_peaks),
        "longest_flat_run_hours": longest_flat_run,
        "event_details": event_details,
        "top_event_peak_time": top_event.get("peak_time"),
        "top_event_peak_q": top_event.get("peak_q"),
        "top_event_peak_over_median": top_event.get("peak_over_median"),
        "top_event_rise_time": top_event.get("rise_time"),
        "top_event_rise_jump": top_event.get("rise_jump"),
        "top_event_rise_over_median": top_event.get("rise_over_median"),
        "top_event_fall_time": top_event.get("fall_time"),
        "top_event_fall_jump": top_event.get("fall_jump"),
        "top_event_fall_over_median": top_event.get("fall_over_median"),
        "top_event_window_start": top_event.get("window_start"),
        "top_event_window_end": top_event.get("window_end"),
        "top_event_window_q_min": top_event.get("window_q_min"),
        "top_event_window_q_max": top_event.get("window_q_max"),
        "top_event_window_q_mean": top_event.get("window_q_mean"),
        "top_event_window_duration_hours": top_event.get("window_duration_hours"),
    }

    if drain_sqkm is not None and np.isfinite(drain_sqkm) and drain_sqkm > 0:
        metrics.update(
            {
                "q50_per_km2": q50 / drain_sqkm,
                "q95_per_km2": q95 / drain_sqkm,
                "q99_per_km2": q99 / drain_sqkm,
                "qmax_per_km2": qmax / drain_sqkm,
                "max_abs_hourly_jump_per_km2": max_abs_jump / drain_sqkm,
                "max_hourly_rise_per_km2": max_rise / drain_sqkm if np.isfinite(max_rise) else float("nan"),
                "top_event_peak_per_km2": top_event.get("peak_q") / drain_sqkm if top_event.get("peak_q") is not None else None,
            }
        )
    else:
        metrics.update(
            {
                "q50_per_km2": None,
                "q95_per_km2": None,
                "q99_per_km2": None,
                "qmax_per_km2": None,
                "max_abs_hourly_jump_per_km2": None,
                "max_hourly_rise_per_km2": None,
                "top_event_peak_per_km2": None,
            }
        )

    return metrics


def pick_from_subset(frame: pd.DataFrame, count: int, seed: int, sort_cols: list[str] | None = None, ascending: list[bool] | bool | None = None) -> pd.DataFrame:
    if frame.empty or count <= 0:
        return frame.iloc[0:0].copy()
    subset = frame.copy()
    if sort_cols:
        subset = subset.sort_values(sort_cols, ascending=ascending)
    else:
        subset = subset.sample(frac=1.0, random_state=seed)
    return subset.head(count).copy()


def build_review_sample(df: pd.DataFrame, max_review_basins: int, seed: int) -> pd.DataFrame:
    ready = df[df["screening_status"] == "RBI_READY"].copy()
    ready["rbi_band"] = ready["rbi"].apply(band_rbi)
    ready["selection_reason"] = ""
    ready["selection_priority"] = 999

    frames: list[pd.DataFrame] = []

    def add_candidates(frame: pd.DataFrame, reason: str, priority: int, count: int, sort_cols: list[str] | None = None, ascending: list[bool] | bool | None = None) -> None:
        if frame.empty:
            return
        subset = pick_from_subset(frame, count, seed=seed, sort_cols=sort_cols, ascending=ascending)
        if subset.empty:
            return
        subset = subset.copy()
        subset["selection_reason"] = reason
        subset["selection_priority"] = priority
        frames.append(subset)

    manual = ready[ready["STAID"].isin(MANUAL_INCLUDE_STAIDS)]
    if not manual.empty:
        manual = manual.copy()
        manual["selection_reason"] = "manual_include"
        manual["selection_priority"] = REVIEW_REASON_PRIORITY["manual_include"]
        frames.append(manual)

    add_candidates(
        ready,
        "top_rbi",
        REVIEW_REASON_PRIORITY["top_rbi"],
        25,
        sort_cols=["rbi", "normalized_max_hourly_dqdt", "q99_event_count", "q95_event_count"],
        ascending=[False, False, False, False],
    )
    add_candidates(
        ready,
        "top_normalized_jump",
        REVIEW_REASON_PRIORITY["top_normalized_jump"],
        25,
        sort_cols=["normalized_max_hourly_dqdt", "q99_event_count", "q95_event_count", "rbi"],
        ascending=[False, False, False, False],
    )

    low_flash = ready[ready["rbi"] < 0.05].copy()
    add_candidates(
        low_flash,
        "low_rbi_high_response",
        REVIEW_REASON_PRIORITY["low_rbi_high_response"],
        15,
        sort_cols=["normalized_max_hourly_dqdt", "q99_event_count", "q95_event_count", "rbi"],
        ascending=[False, False, False, True],
    )

    add_candidates(
        ready[(ready["rbi"] >= 0.04) & (ready["rbi"] < 0.07)],
        "rbi_0p04_0p07",
        REVIEW_REASON_PRIORITY["rbi_0p04_0p07"],
        12,
        sort_cols=["area_bin", "BFI_bin", "rbi", "normalized_max_hourly_dqdt"],
        ascending=[True, True, False, False],
    )
    add_candidates(
        ready[(ready["rbi"] >= 0.07) & (ready["rbi"] < 0.10)],
        "rbi_0p07_0p10",
        REVIEW_REASON_PRIORITY["rbi_0p07_0p10"],
        12,
        sort_cols=["area_bin", "BFI_bin", "rbi", "normalized_max_hourly_dqdt"],
        ascending=[True, True, False, False],
    )
    add_candidates(
        ready[(ready["rbi"] >= 0.10) & (ready["rbi"] < 0.20)],
        "rbi_0p10_0p20",
        REVIEW_REASON_PRIORITY["rbi_0p10_0p20"],
        15,
        sort_cols=["area_bin", "BFI_bin", "rbi", "normalized_max_hourly_dqdt"],
        ascending=[True, True, False, False],
    )
    add_candidates(
        ready[(ready["rbi"] >= 0.20) & (ready["rbi"] < 0.50)],
        "rbi_0p20_0p50",
        REVIEW_REASON_PRIORITY["rbi_0p20_0p50"],
        15,
        sort_cols=["area_bin", "BFI_bin", "rbi", "normalized_max_hourly_dqdt"],
        ascending=[True, True, False, False],
    )
    add_candidates(
        ready[ready["rbi"] >= 0.50],
        "rbi_ge_0p50",
        REVIEW_REASON_PRIORITY["rbi_ge_0p50"],
        15,
        sort_cols=["rbi", "normalized_max_hourly_dqdt", "q99_event_count", "q95_event_count"],
        ascending=[False, False, False, False],
    )

    sample = pd.concat(frames, ignore_index=True) if frames else ready.iloc[0:0].copy()
    if sample.empty:
        return sample

    group_cols = ["STAID"]
    agg_map = {col: "first" for col in sample.columns if col not in {"selection_reason", "selection_priority"}}
    agg_map["selection_reason"] = lambda s: combine_reasons(s.astype(str).tolist())
    agg_map["selection_priority"] = "min"
    sample = sample.groupby(group_cols, as_index=False).agg(agg_map)
    sample = sample.sort_values(["selection_priority", "rbi"], ascending=[True, False]).reset_index(drop=True)
    if max_review_basins > 0:
        sample = sample.head(max_review_basins).copy()
    return sample


def annotate_axes(ax: plt.Axes, text: str, loc: str = "upper left") -> None:
    x = 0.02 if "left" in loc else 0.98
    ha = "left" if "left" in loc else "right"
    ax.text(
        x,
        0.98,
        text,
        transform=ax.transAxes,
        ha=ha,
        va="top",
        fontsize=8,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
    )


def build_metrics_box(row: pd.Series) -> str:
    lines = [
        f"RBI={row.get('rbi', float('nan')):.3f}",
        f"completeness={row.get('hourly_completeness_pct', float('nan')):.1f}%",
        f"Q50={row.get('q50', float('nan')):.3g}",
        f"Q95={row.get('q95', float('nan')):.3g}",
        f"Q99={row.get('q99', float('nan')):.3g}",
        f"jump={row.get('max_abs_hourly_jump', float('nan')):.3g}",
        f"jump/Q50={row.get('max_abs_hourly_jump_over_q50', float('nan')):.3g}",
        f"peak count={int(row.get('selected_peak_count', 0))}",
    ]
    flags = row.get("qc_flags", [])
    if isinstance(flags, str):
        try:
            flags = json.loads(flags)
        except Exception:
            flags = [flags]
    if flags:
        lines.append(f"flags={','.join(flags[:4])}")
    return "\n".join(lines)


def make_event_plot(
    series: pd.Series,
    row: pd.Series,
    event: dict,
    outpath: Path,
    event_rank: int,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    window_start = pd.Timestamp(event["window_start"])
    window_end = pd.Timestamp(event["window_end"])
    window = series.loc[window_start:window_end]
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=170)
    ax.plot(window.index, window.values, color="#1f4e79", linewidth=1.6, label="Hourly discharge")

    peak_time = pd.Timestamp(event["peak_time"])
    ax.axvline(peak_time, color="#c62828", linewidth=2.0, label="Peak Q")
    if event.get("rise_time") is not None:
        ax.axvline(pd.Timestamp(event["rise_time"]), color="#1565c0", linestyle="--", linewidth=1.6, label="Max pre-peak rise")
    if event.get("fall_time") is not None:
        ax.axvline(pd.Timestamp(event["fall_time"]), color="#6a1b9a", linestyle=":", linewidth=1.8, label="Max post-peak fall")

    ax.set_title(
        f"{row['STAID']} event {event_rank} | RBI={row.get('rbi', float('nan')):.3f} | peak={event['peak_q']:.3g} {event.get('threshold', 'peak')}"
    )
    ax.set_ylabel("Discharge (m3/s)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    annotate_axes(ax, build_metrics_box(row), loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def make_full_wy_plot(series: pd.Series, row: pd.Series, events: list[dict], outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=170)
    ax.plot(series.index, series.values, color="#283593", linewidth=1.0, label="Hourly discharge")

    for idx, event in enumerate(events[:3], start=1):
        peak_time = pd.Timestamp(event["peak_time"])
        ax.axvline(peak_time, color="#c62828", linewidth=1.8, alpha=0.9 if idx == 1 else 0.6)
        ax.scatter([peak_time], [event["peak_q"]], color="#c62828", s=16, zorder=5)
        if event.get("rise_time") is not None:
            ax.axvline(pd.Timestamp(event["rise_time"]), color="#1565c0", linestyle="--", linewidth=1.0, alpha=0.8)
        if event.get("fall_time") is not None:
            ax.axvline(pd.Timestamp(event["fall_time"]), color="#6a1b9a", linestyle=":", linewidth=1.0, alpha=0.8)

    ax.set_title(f"{row['STAID']} full WY2024 | RBI={row.get('rbi', float('nan')):.3f} | class={row.get('candidate_class', 'unknown')}")
    ax.set_ylabel("Discharge (m3/s)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    annotate_axes(ax, build_metrics_box(row), loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def classify_qc_and_candidate_classes(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.copy()
    if df.empty:
        return df

    def safe_quantile(series: pd.Series, q: float) -> float:
        valid = pd.to_numeric(series, errors="coerce").dropna()
        if valid.empty:
            return float("nan")
        return float(valid.quantile(q))

    thresholds = {
        "normalized_jump_q99": safe_quantile(df["normalized_max_hourly_dqdt"], 0.99),
        "normalized_jump_q95": safe_quantile(df["normalized_max_hourly_dqdt"], 0.95),
        "q95_count_q99": safe_quantile(df["event_count_q95"], 0.99),
        "q99_count_q99": safe_quantile(df["event_count_q99"], 0.99),
        "q99_per_km2_q10": safe_quantile(df["q99_per_km2"], 0.10),
        "jump_over_q50_q95": safe_quantile(df["max_abs_hourly_jump_over_q50"], 0.95),
        "jump_over_q50_q99": safe_quantile(df["max_abs_hourly_jump_over_q50"], 0.99),
        "zero_flow_fraction_q95": safe_quantile(df["zero_flow_fraction"], 0.95),
    }

    qc_flags: list[list[str]] = []
    candidate_classes: list[str] = []

    for _, row in df.iterrows():
        flags: list[str] = []
        if pd.notna(row.get("hourly_completeness_pct")) and row["hourly_completeness_pct"] < 90:
            flags.append("LOW_COMPLETENESS_LT_90")
        if pd.notna(row.get("hourly_completeness_pct")) and row["hourly_completeness_pct"] < 70:
            flags.append("LOW_COMPLETENESS_LT_70")
        if int(row.get("negative_flow_count", 0)) > 0:
            flags.append("NEGATIVE_FLOW_PRESENT")
        if pd.notna(row.get("negative_flow_fraction")) and row["negative_flow_fraction"] > 0.001:
            flags.append("HIGH_NEGATIVE_FLOW_FRACTION")
        if pd.notna(row.get("zero_flow_fraction")) and row["zero_flow_fraction"] >= 0.25:
            flags.append("ZERO_FLOW_DOMINATED")
        if pd.notna(row.get("normalized_max_hourly_dqdt")) and pd.notna(thresholds["normalized_jump_q99"]) and row["normalized_max_hourly_dqdt"] >= thresholds["normalized_jump_q99"]:
            flags.append("HIGH_NORMALIZED_JUMP")
        if pd.notna(row.get("q95_event_count")) and pd.notna(thresholds["q95_count_q99"]) and row["q95_event_count"] >= thresholds["q95_count_q99"]:
            flags.append("HIGH_Q95_EVENT_COUNT")
        if pd.notna(row.get("q99_event_count")) and pd.notna(thresholds["q99_count_q99"]) and row["q99_event_count"] >= thresholds["q99_count_q99"]:
            flags.append("HIGH_Q99_EVENT_COUNT")
        if pd.notna(row.get("q99_per_km2")) and pd.notna(thresholds["q99_per_km2_q10"]) and row["q99_per_km2"] <= thresholds["q99_per_km2_q10"]:
            flags.append("VERY_LOW_SPECIFIC_FLOW")
        if pd.notna(row.get("max_abs_hourly_jump_over_q50")) and pd.notna(thresholds["jump_over_q50_q95"]) and row["max_abs_hourly_jump_over_q50"] >= thresholds["jump_over_q50_q95"]:
            flags.append("SUSPICIOUS_SPIKE")
        if int(row.get("longest_flat_run_hours", 0)) >= 24:
            flags.append("POSSIBLE_REGULATION_OR_ARTIFACT")
        if pd.notna(row.get("max_hourly_rise_over_q50")) and pd.notna(thresholds["jump_over_q50_q95"]) and row["max_hourly_rise_over_q50"] >= thresholds["jump_over_q50_q95"]:
            if "SUSPICIOUS_SPIKE" not in flags:
                flags.append("SUSPICIOUS_SPIKE")

        if not flags:
            flags.append("OK")

        qc_flags.append(flags)

        rbi = row.get("rbi")
        completeness = row.get("hourly_completeness_pct")
        strong_response = (
            (pd.notna(row.get("normalized_max_hourly_dqdt")) and pd.notna(thresholds["normalized_jump_q95"]) and row["normalized_max_hourly_dqdt"] >= thresholds["normalized_jump_q95"])
            or (pd.notna(row.get("max_abs_hourly_jump_over_q50")) and pd.notna(thresholds["jump_over_q50_q95"]) and row["max_abs_hourly_jump_over_q50"] >= thresholds["jump_over_q50_q95"])
            or (int(row.get("event_count_q99", 0)) >= 2)
            or (int(row.get("event_count_q95", 0)) >= 3)
        )
        severe = any(flag in SEVERE_QC_FLAGS for flag in flags)
        if severe:
            candidate_class = "EXCLUDE_QC"
        elif pd.isna(completeness) or completeness < 90:
            candidate_class = "REVIEW_MANUAL"
        elif pd.notna(rbi) and rbi < 0.05 and strong_response:
            candidate_class = "POSSIBLE_FLASHY_LOW_RBI"
        elif pd.notna(rbi) and 0.05 <= rbi < 0.10 and strong_response:
            candidate_class = "MODERATE_FLASHY"
        elif pd.notna(rbi) and rbi >= 0.10 and strong_response:
            candidate_class = "HIGH_CONFIDENCE_FLASHY"
        elif pd.notna(rbi) and completeness >= 90 and not strong_response:
            candidate_class = "LOW_PRIORITY_NONFLASHY"
        else:
            candidate_class = "REVIEW_MANUAL"
        candidate_classes.append(candidate_class)

    df["qc_flags"] = [json.dumps(flags) for flags in qc_flags]
    df["qc_primary_flag"] = [flags[0] for flags in qc_flags]
    df["candidate_class"] = candidate_classes
    df["qc_status"] = ["OK" if flags == ["OK"] else "REVIEW" for flags in qc_flags]
    df["selection_reason"] = df["selection_reason"].fillna("")
    df["selection_priority"] = pd.to_numeric(df["selection_priority"], errors="coerce")
    df.attrs["qc_thresholds"] = thresholds
    return df


def write_tables_and_summaries(output_dir: Path, metrics_df: pd.DataFrame, sample_df: pd.DataFrame, logger: logging.Logger) -> dict:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = output_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = tables_dir / "event_hydrograph_review_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    metrics_parquet = tables_dir / "event_hydrograph_review_metrics.parquet"
    try:
        metrics_df.to_parquet(metrics_parquet, index=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write metrics parquet: %s", exc)
        metrics_parquet = None

    sample_csv = tables_dir / "event_hydrograph_review_sample.csv"
    sample_df.to_csv(sample_csv, index=False)

    class_summary = (
        metrics_df.groupby("candidate_class", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    class_summary.to_csv(tables_dir / "candidate_class_summary.csv", index=False)

    qc_flag_rows = []
    flag_counter: dict[str, int] = {}
    for raw_flags in metrics_df["qc_flags"]:
        try:
            flags = json.loads(raw_flags) if isinstance(raw_flags, str) else list(raw_flags)
        except Exception:
            flags = [str(raw_flags)]
        for flag in flags:
            flag_counter[flag] = flag_counter.get(flag, 0) + 1
    for flag, count in sorted(flag_counter.items(), key=lambda item: (-item[1], item[0])):
        qc_flag_rows.append({"qc_flag": flag, "count": count})
    pd.DataFrame(qc_flag_rows).to_csv(tables_dir / "qc_flag_summary.csv", index=False)

    sample_by_band = metrics_df.groupby("rbi_band", dropna=False).size().reset_index(name="count")
    sample_by_band.to_csv(tables_dir / "sample_by_rbi_band.csv", index=False)

    top_reviewed = metrics_df.sort_values(["rbi", "normalized_max_hourly_dqdt"], ascending=[False, False]).head(20)

    thresholds = metrics_df.attrs.get("qc_thresholds", {})
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selected_basins": int(len(sample_df)),
        "successful_basins": int(len(metrics_df)),
        "failed_basins": int(len(sample_df) - len(metrics_df)),
        "selected_by_rbi_band": sample_by_band.to_dict(orient="records"),
        "candidate_class_counts": class_summary.to_dict(orient="records"),
        "qc_flag_counts": qc_flag_rows,
        "thresholds": thresholds,
        "top_reviewed_basins": top_reviewed[["STAID", "rbi", "candidate_class", "qc_primary_flag", "selection_reason"]].to_dict(orient="records"),
    }

    (summary_dir / "event_hydrograph_review_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    md_lines = [
        "# Event Hydrograph Review Summary",
        "",
        f"Selected basins: {summary['selected_basins']}",
        f"Successful basins: {summary['successful_basins']}",
        f"Failed basins: {summary['failed_basins']}",
        "",
        "## Candidate classes",
        "",
    ]
    for row in class_summary.itertuples(index=False):
        md_lines.append(f"- {row.candidate_class}: {row.count}")
    md_lines.extend([
        "",
        "## QC flags",
        "",
    ])
    for row in qc_flag_rows:
        md_lines.append(f"- {row['qc_flag']}: {row['count']}")
    md_lines.extend([
        "",
        "## Threshold guidance",
        "",
        "This review is still exploratory. RBI alone is not a sufficient final decision rule.",
        "Use event-response shape, completeness, zero-flow behavior, and the QC flags together.",
        "",
        "## Top reviewed basins",
        "",
    ])
    for row in top_reviewed.itertuples(index=False):
        md_lines.append(
            f"- {row.STAID}: RBI={row.rbi:.3f}, class={row.candidate_class}, qc={row.qc_primary_flag}, reason={row.selection_reason}"
        )

    (summary_dir / "event_hydrograph_review_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    return summary


def build_diagnostic_plots(output_dir: Path, metrics_df: pd.DataFrame) -> list[Path]:
    plots_dir = output_dir / "plots" / "diagnostics"
    plots_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=160)
    for candidate_class, group in metrics_df.groupby("candidate_class"):
        ax.scatter(group["rbi"], group["normalized_max_hourly_dqdt"], s=18, alpha=0.8, label=candidate_class)
    ax.set_xlabel("RBI")
    ax.set_ylabel("Normalized max hourly jump")
    ax.set_title("Reviewed sample: RBI versus normalized hourly jump")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, frameon=True)
    fig.tight_layout()
    out = plots_dir / "rbi_vs_normalized_jump.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    created.append(out)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=160)
    ax.hist(metrics_df["hourly_completeness_pct"].dropna(), bins=20, color="#2e7d32", alpha=0.85)
    ax.set_xlabel("Hourly completeness (%)")
    ax.set_ylabel("Count")
    ax.set_title("Reviewed sample completeness")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out = plots_dir / "completeness_hist.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    created.append(out)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=160)
    ax.scatter(metrics_df["q99_per_km2"], metrics_df["max_abs_hourly_jump_over_q50"], s=18, alpha=0.8, c=metrics_df["rbi"], cmap="viridis")
    ax.set_xlabel("Q99 per km²")
    ax.set_ylabel("Max hourly jump / median Q")
    ax.set_title("Reviewed sample: specific flow versus jump magnitude")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out = plots_dir / "specific_flow_vs_jump.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    created.append(out)

    return created


def build_review_bundle(output_dir: Path, metrics_df: pd.DataFrame) -> Optional[Path]:
    bundle_dir = output_dir / "review_bundle"
    plots_root = output_dir / "plots"
    if not plots_root.exists():
        return None

    selected = []
    for rbi_band, group in metrics_df.groupby("rbi_band"):
        if group.empty:
            continue
        selected.extend(group.sort_values("rbi", ascending=False).head(5)["STAID"].astype(str).tolist())

    selected = list(dict.fromkeys(selected))
    if not selected:
        return None

    review_plots_dir = bundle_dir / "selected_review_plots"
    review_plots_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    for staid in selected:
        basin_plot_dir = next(iter((plots_root).glob(f"**/{staid}")), None)
        if basin_plot_dir is None:
            continue
        for name in [f"full_wy_{staid}.png"]:
            source = basin_plot_dir / name
            if source.exists():
                target = review_plots_dir / source.name
                shutil.copy2(source, target)
                files.append(str(target.relative_to(output_dir)).replace("\\", "/"))
        event_plots = sorted(basin_plot_dir.glob("event_*.png"))
        if event_plots:
            source = event_plots[0]
            target = review_plots_dir / source.name
            shutil.copy2(source, target)
            files.append(str(target.relative_to(output_dir)).replace("\\", "/"))

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "selected_basins": selected,
    }
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (bundle_dir / "summary.md").write_text(
        "\n".join([
            "# Review Bundle",
            "",
            f"Selected basins: {len(selected)}",
            f"Files copied: {len(files)}",
        ]),
        encoding="utf-8",
    )
    return bundle_dir


def ensure_float(value: object) -> Optional[float]:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    results_path = find_results_file(args.results_dir)
    logger.info("Loading screening results from %s", results_path)
    df = load_screening_results(results_path)
    df_ready = df[df["screening_status"] == "RBI_READY"].copy()
    if df_ready.empty:
        logger.error("No RBI_READY basins found; aborting")
        return

    sample_df = build_review_sample(df_ready, max_review_basins=args.max_review_basins, seed=args.seed)
    if sample_df.empty:
        logger.error("No review sample could be selected")
        return

    logger.info("Selected %s basins for event review", len(sample_df))

    output_hourly_dir = output_dir / "hourly_series"
    raw_dir = output_dir / "raw_json" if args.debug_raw else output_dir / "raw_json_disabled"

    session = requests.Session()
    session.headers.update({"User-Agent": "Flash-NH event hydrograph review script v002"})

    review_rows: list[dict] = []
    failed_sites: list[dict] = []

    for _, row in sample_df.iterrows():
        staid = str(row["STAID"]).zfill(8)
        logger.info("Processing basin %s", staid)
        try:
            hourly, source = load_or_fetch_hourly(
                session=session,
                staid=staid,
                output_hourly_dir=output_hourly_dir,
                cache_dir=args.cache_dir,
                logger=logger,
                sleep_seconds=args.sleep_seconds,
                max_retries=args.max_retries,
                timeout_seconds=args.timeout_seconds,
                no_download=args.no_download,
                force_refresh_selected=args.force_refresh_selected,
                debug_raw=args.debug_raw,
                raw_dir=raw_dir,
            )
            if hourly is None or hourly.dropna().empty:
                failed_sites.append({"STAID": staid, "reason": source})
                logger.warning("Skipping %s because no hourly data were available (%s)", staid, source)
                continue

            metrics = compute_metrics(hourly, ensure_float(row.get("DRAIN_SQKM")))
            metrics.update(
                {
                    "STAID": staid,
                    "DRAIN_SQKM": ensure_float(row.get("DRAIN_SQKM")),
                    "BFI_AVE": ensure_float(row.get("BFI_AVE")),
                    "LAT_GAGE": ensure_float(row.get("LAT_GAGE")),
                    "LNG_GAGE": ensure_float(row.get("LNG_GAGE")),
                    "STATE": row.get("STATE"),
                    "HUC02": row.get("HUC02"),
                    "rbi": ensure_float(row.get("rbi")),
                    "rbi_band": row.get("rbi_band"),
                    "area_bin": row.get("area_bin"),
                    "BFI_bin": row.get("BFI_bin"),
                    "selection_reason": row.get("selection_reason"),
                    "selection_priority": ensure_float(row.get("selection_priority")),
                    "screening_status": row.get("screening_status"),
                    "source": source,
                    "normalized_max_hourly_dqdt": ensure_float(row.get("normalized_max_hourly_dqdt")),
                    "max_hourly_dqdt_m3s_per_hr": ensure_float(row.get("max_hourly_dqdt_m3s_per_hr")),
                    "q95_event_count_screening": ensure_float(row.get("q95_event_count")),
                    "q99_event_count_screening": ensure_float(row.get("q99_event_count")),
                    "notes": row.get("notes"),
                }
            )
            metrics["rbi_band"] = band_rbi(metrics["rbi"]) if pd.notna(metrics["rbi"]) else row.get("rbi_band")
            review_rows.append(metrics)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed processing %s: %s", staid, exc)
            failed_sites.append({"STAID": staid, "reason": str(exc)})

    metrics_df = pd.DataFrame(review_rows)
    if metrics_df.empty:
        logger.error("No review basins completed successfully")
        (output_dir / "event_hydrograph_review_summary.json").write_text(
            json.dumps(
                {
                    "selected_basins": int(len(sample_df)),
                    "successful_basins": 0,
                    "failed_basins": failed_sites,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    metrics_df = classify_qc_and_candidate_classes(metrics_df)

    metrics_df["top_event_details"] = metrics_df["event_details"].apply(lambda value: json.dumps(value, default=str))
    metrics_df["top_event_peak_time"] = pd.to_datetime(metrics_df["top_event_peak_time"], utc=True, errors="coerce")
    metrics_df["top_event_rise_time"] = pd.to_datetime(metrics_df["top_event_rise_time"], utc=True, errors="coerce")
    metrics_df["top_event_fall_time"] = pd.to_datetime(metrics_df["top_event_fall_time"], utc=True, errors="coerce")
    metrics_df["top_event_window_start"] = pd.to_datetime(metrics_df["top_event_window_start"], utc=True, errors="coerce")
    metrics_df["top_event_window_end"] = pd.to_datetime(metrics_df["top_event_window_end"], utc=True, errors="coerce")

    # The plotting functions expect qc_flags to be usable as JSON strings.
    metrics_df["qc_flags"] = metrics_df["qc_flags"].astype(str)

    plots_root = output_dir / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)
    total_event_plots = 0
    total_full_plots = 0

    for _, row in metrics_df.iterrows():
        staid = str(row["STAID"])
        hourly = None
        hourly_file = output_hourly_dir / f"{staid}_hourly.csv"
        if hourly_file.exists():
            hourly = normalize_hourly_dataframe(pd.read_csv(hourly_file))
        elif not args.no_download:
            cached = read_cached_hourly(args.cache_dir, staid)
            if cached is not None:
                hourly = cached
                write_hourly_csv(cached, hourly_file)
        if hourly is None or hourly.dropna().empty:
            continue

        basin_plot_dir = plots_root / str(row["rbi_band"]) / staid
        basin_plot_dir.mkdir(parents=True, exist_ok=True)
        try:
            event_details = json.loads(row["top_event_details"]) if isinstance(row["top_event_details"], str) else row["event_details"]
        except Exception:
            event_details = []
        if not event_details and row.get("event_details"):
            event_details = row["event_details"]
        event_details = event_details or []
        for idx, event in enumerate(event_details[:5], start=1):
            if event.get("peak_time") is None:
                continue
            event_plot = basin_plot_dir / f"event_{idx}_{pd.Timestamp(event['peak_time']).strftime('%Y%m%dT%H%M%SZ')}.png"
            make_event_plot(hourly, row, event, event_plot, idx)
            total_event_plots += 1
        full_plot = basin_plot_dir / f"full_wy_{staid}.png"
        make_full_wy_plot(hourly, row, event_details, full_plot)
        total_full_plots += 1

    summary = write_tables_and_summaries(output_dir, metrics_df, sample_df, logger)
    summary["failed_sites"] = failed_sites
    summary["total_event_plots"] = int(total_event_plots)
    summary["total_full_year_plots"] = int(total_full_plots)

    # Basin-specific notes for the four manually highlighted sites.
    highlight_notes = []
    for highlight in MANUAL_INCLUDE_STAIDS:
        match = metrics_df[metrics_df["STAID"] == highlight]
        if match.empty:
            continue
        row = match.iloc[0]
        highlight_notes.append(
            {
                "STAID": highlight,
                "RBI": row.get("rbi"),
                "candidate_class": row.get("candidate_class"),
                "qc_primary_flag": row.get("qc_primary_flag"),
                "selection_reason": row.get("selection_reason"),
            }
        )
    summary["highlighted_basins"] = highlight_notes
    (output_dir / "summaries" / "event_hydrograph_review_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    md_path = output_dir / "summaries" / "event_hydrograph_review_summary.md"
    md_text = md_path.read_text(encoding="utf-8")
    md_text += "\n\n## Highlighted basins\n\n"
    if highlight_notes:
        for row in highlight_notes:
            md_text += f"- {row['STAID']}: RBI={row['RBI']:.3f} class={row['candidate_class']} qc={row['qc_primary_flag']} reason={row['selection_reason']}\n"
    else:
        md_text += "- No highlighted basins were present in the completed review sample.\n"
    md_path.write_text(md_text, encoding="utf-8")

    build_diagnostic_plots(output_dir, metrics_df)
    bundle_dir = build_review_bundle(output_dir, metrics_df)
    if bundle_dir is not None:
        summary["review_bundle"] = str(bundle_dir)
        (output_dir / "summaries" / "event_hydrograph_review_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"Event hydrograph review outputs written to {output_dir}")
    print(f"Selected: {len(sample_df)}; Successful: {len(metrics_df)}; Event plots: {total_event_plots}; Full plots: {total_full_plots}")


if __name__ == "__main__":
    main()

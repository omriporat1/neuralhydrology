#!/usr/bin/env python3
"""Scale USGS IV discharge retrieval and RBI screening for Flash-NH basins.

This script runs RBI screening over all basins classified as ELIGIBLE_SCREENING_WY,
using one full water year (2023-10-01 to 2024-09-30) and a resumable,
batch-checkpointed workflow.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.usgs_discharge_probe import (  # noqa: E402
    BFI_BIN_LABELS,
    AREA_BIN_LABELS,
    USER_AGENT,
    build_hourly_series,
    calculate_probe_metrics,
    convert_units,
    flatten_iv_payload,
    infer_native_timestep,
)

INPUT_PARQUET = ROOT / "reports/flashnh_usgs_coverage_eligibility_v001/usgs_coverage_eligibility.parquet"
INPUT_CSV = ROOT / "reports/flashnh_usgs_coverage_eligibility_v001/usgs_coverage_eligibility.csv"
DEFAULT_OUTPUT_DIR = ROOT / "reports/flashnh_usgs_rbi_screening_wy2024_v001"
USGS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"

SCREENING_START = pd.Timestamp("2023-10-01 00:00:00", tz="UTC")
SCREENING_END = pd.Timestamp("2024-09-30 23:00:00", tz="UTC")
EXPECTED_HOURLY_INDEX = pd.date_range(start=SCREENING_START, end=SCREENING_END, freq="1h")
EXPECTED_HOURLY_COUNT = len(EXPECTED_HOURLY_INDEX)

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
class BasinResult:
    STAID: str
    DRAIN_SQKM: Optional[float]
    BFI_AVE: Optional[float]
    area_bin: Optional[str]
    BFI_bin: Optional[str]
    LAT_GAGE: Optional[float]
    LNG_GAGE: Optional[float]
    STATE: Optional[str]
    HUC02: Optional[str]
    request_success: bool
    request_http_status: Optional[int]
    response_size_bytes: Optional[int]
    request_url: Optional[str]
    request_error: Optional[str]
    first_timestamp_utc: Optional[str]
    last_timestamp_utc: Optional[str]
    returned_observation_count: int
    native_timestep_minutes: Optional[float]
    native_timestep_share: Optional[float]
    inferred_timestep_mode: str
    hourly_values_count: int
    expected_hourly_count: int
    hourly_completeness_pct: float
    screening_status: str
    units_original: Optional[str]
    units_output: str
    unit_conversion_applied: bool
    rbi: Optional[float]
    max_hourly_dqdt_m3s_per_hr: Optional[float]
    normalized_max_hourly_dqdt: Optional[float]
    q95_event_count: Optional[int]
    q99_event_count: Optional[int]
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scale USGS RBI screening for ELIGIBLE_SCREENING_WY basins")
    parser.add_argument("--max-basins", type=int, default=None, help="Optional cap on number of basins to process")
    parser.add_argument("--batch-size", type=int, default=25, help="Checkpoint every N newly processed basins")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output rows")
    parser.add_argument("--sleep-seconds", type=float, default=0.35, help="Polite delay between successful USGS requests")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for results and artifacts")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum retries per site request")
    parser.add_argument("--timeout-seconds", type=int, default=90, help="HTTP timeout in seconds")
    return parser.parse_args()


def setup_paths(output_dir: Path) -> dict[str, Path]:
    paths = {
        "output_dir": output_dir,
        "plots_dir": output_dir / "plots",
        "logs_dir": output_dir / "logs",
        "review_bundle_dir": output_dir / "review_bundle",
        "review_plots_dir": output_dir / "review_bundle" / "plots",
        "results_parquet": output_dir / "usgs_rbi_screening_results.parquet",
        "results_csv": output_dir / "usgs_rbi_screening_results.csv",
        "summary_json": output_dir / "usgs_rbi_screening_summary.json",
        "summary_md": output_dir / "usgs_rbi_screening_summary.md",
        "log_file": output_dir / "logs" / "usgs_rbi_screening.log",
    }
    for key in ["output_dir", "plots_dir", "logs_dir", "review_bundle_dir", "review_plots_dir"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def setup_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("usgs_rbi_screening_scale")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def load_eligibility_table() -> pd.DataFrame:
    if INPUT_PARQUET.exists():
        frame = pd.read_parquet(INPUT_PARQUET)
    elif INPUT_CSV.exists():
        frame = pd.read_csv(INPUT_CSV, dtype={"STAID": str})
    else:
        raise FileNotFoundError(f"Missing input: {INPUT_PARQUET} and {INPUT_CSV}")

    frame["STAID"] = frame["STAID"].astype(str).str.zfill(8)
    if "eligibility_class" not in frame.columns:
        raise RuntimeError("Input table is missing eligibility_class column")

    eligible = frame[frame["eligibility_class"] == "ELIGIBLE_SCREENING_WY"].copy()
    eligible = eligible.sort_values(["area_bin", "BFI_bin", "STAID"], na_position="last").reset_index(drop=True)
    return eligible


def load_existing_results(paths: dict[str, Path]) -> pd.DataFrame:
    if paths["results_parquet"].exists():
        frame = pd.read_parquet(paths["results_parquet"])
    elif paths["results_csv"].exists():
        frame = pd.read_csv(paths["results_csv"], dtype={"STAID": str})
    else:
        return pd.DataFrame()

    if not frame.empty:
        frame["STAID"] = frame["STAID"].astype(str).str.zfill(8)
    return frame


def save_results(paths: dict[str, Path], results: pd.DataFrame) -> None:
    results.to_parquet(paths["results_parquet"], index=False)
    results.to_csv(paths["results_csv"], index=False)


def fetch_iv_json(
    session: requests.Session,
    site_no: str,
    sleep_seconds: float,
    max_retries: int,
    timeout_seconds: int,
    logger: logging.Logger,
) -> tuple[dict, int, int, str]:
    params = {
        "sites": site_no,
        "parameterCd": "00060",
        "startDT": SCREENING_START.date().isoformat(),
        "endDT": SCREENING_END.date().isoformat(),
        "format": "json",
        "siteStatus": "all",
    }

    backoff = 1.0
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(USGS_IV_URL, params=params, timeout=timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            status = response.status_code
            response_size = len(response.content)
            request_url = response.url
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            return payload, status, response_size, request_url
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_retries:
                break
            logger.warning("Request failed for %s attempt %s/%s: %s", site_no, attempt, max_retries, exc)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    raise RuntimeError(f"USGS IV request failed for {site_no}") from last_error


def compute_status(completeness_pct: float, has_data: bool) -> str:
    if not has_data:
        return "NO_DATA"
    if completeness_pct >= 90.0:
        return "RBI_READY"
    if completeness_pct >= 70.0:
        return "PARTIAL_USABLE"
    return "INSUFFICIENT"


def process_site(
    site_row: pd.Series,
    session: requests.Session,
    sleep_seconds: float,
    max_retries: int,
    timeout_seconds: int,
    logger: logging.Logger,
) -> BasinResult:
    staid = str(site_row["STAID"]).zfill(8)

    def _float_or_none(name: str) -> Optional[float]:
        value = site_row.get(name)
        return float(value) if pd.notna(value) else None

    base_kwargs = {
        "STAID": staid,
        "DRAIN_SQKM": _float_or_none("DRAIN_SQKM"),
        "BFI_AVE": _float_or_none("BFI_AVE"),
        "area_bin": str(site_row.get("area_bin")) if pd.notna(site_row.get("area_bin")) else None,
        "BFI_bin": str(site_row.get("BFI_bin")) if pd.notna(site_row.get("BFI_bin")) else None,
        "LAT_GAGE": _float_or_none("LAT_GAGE") if pd.notna(site_row.get("LAT_GAGE")) else _float_or_none("usgs_dec_lat_va"),
        "LNG_GAGE": _float_or_none("LNG_GAGE") if pd.notna(site_row.get("LNG_GAGE")) else _float_or_none("usgs_dec_long_va"),
        "STATE": str(site_row.get("STATE")) if pd.notna(site_row.get("STATE")) else None,
        "HUC02": str(site_row.get("HUC02")) if pd.notna(site_row.get("HUC02")) else None,
    }

    try:
        payload, status_code, response_size, request_url = fetch_iv_json(
            session=session,
            site_no=staid,
            sleep_seconds=sleep_seconds,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            logger=logger,
        )
    except Exception as exc:  # noqa: BLE001
        return BasinResult(
            **base_kwargs,
            request_success=False,
            request_http_status=None,
            response_size_bytes=None,
            request_url=None,
            request_error=str(exc),
            first_timestamp_utc=None,
            last_timestamp_utc=None,
            returned_observation_count=0,
            native_timestep_minutes=None,
            native_timestep_share=None,
            inferred_timestep_mode="sparse",
            hourly_values_count=0,
            expected_hourly_count=EXPECTED_HOURLY_COUNT,
            hourly_completeness_pct=0.0,
            screening_status="ERROR",
            units_original=None,
            units_output="m3/s",
            unit_conversion_applied=False,
            rbi=None,
            max_hourly_dqdt_m3s_per_hr=None,
            normalized_max_hourly_dqdt=None,
            q95_event_count=None,
            q99_event_count=None,
            notes=str(exc),
        )

    raw_frame, meta = flatten_iv_payload(payload)
    if raw_frame.empty:
        return BasinResult(
            **base_kwargs,
            request_success=True,
            request_http_status=status_code,
            response_size_bytes=response_size,
            request_url=request_url,
            request_error=None,
            first_timestamp_utc=None,
            last_timestamp_utc=None,
            returned_observation_count=0,
            native_timestep_minutes=None,
            native_timestep_share=None,
            inferred_timestep_mode="sparse",
            hourly_values_count=0,
            expected_hourly_count=EXPECTED_HOURLY_COUNT,
            hourly_completeness_pct=0.0,
            screening_status="NO_DATA",
            units_original=meta.get("unit_code") if isinstance(meta.get("unit_code"), str) else None,
            units_output="m3/s",
            unit_conversion_applied=False,
            rbi=None,
            max_hourly_dqdt_m3s_per_hr=None,
            normalized_max_hourly_dqdt=None,
            q95_event_count=None,
            q99_event_count=None,
            notes="No 00060 observations returned for screening water year",
        )

    try:
        parsed = raw_frame.copy()
        parsed["dateTime"] = pd.to_datetime(parsed["dateTime"], utc=True, errors="coerce")
        parsed["value"] = pd.to_numeric(parsed["value"], errors="coerce")
        parsed = parsed.dropna(subset=["dateTime", "value"]).sort_values("dateTime")
        if parsed.empty:
            raise ValueError("Returned observations could not be parsed into numeric time series")

        raw_count = int(len(parsed))
        raw_series = parsed.set_index("dateTime")["value"].groupby(level=0).mean().sort_index()
        converted_series, units_output, conversion_applied = convert_units(raw_series, meta.get("unit_code"))
        converted_series = converted_series.groupby(level=0).mean().sort_index()

        timestep_minutes, timestep_share, timestep_mode = infer_native_timestep(converted_series.index)
        hourly = build_hourly_series(converted_series)
        hourly_valid_count = int(hourly.notna().sum())
        completeness_pct = round(100.0 * hourly_valid_count / EXPECTED_HOURLY_COUNT, 3)
        screening_status = compute_status(completeness_pct, has_data=raw_count > 0)

        rbi, max_dqdt, normalized_dqdt, q95_count, q99_count = calculate_probe_metrics(hourly)

        notes = []
        if timestep_mode == "hourly":
            notes.append("Direct hourly cadence")
        elif timestep_mode == "sub-hourly":
            notes.append("Native sub-hourly cadence; resampled to hourly")
        elif timestep_mode == "daily":
            notes.append("Native daily cadence")
        elif timestep_mode == "irregular":
            notes.append("Native cadence irregular")
        else:
            notes.append("Sparse series")

        if screening_status == "INSUFFICIENT":
            notes.append("Hourly completeness below 70%")

        return BasinResult(
            **base_kwargs,
            request_success=True,
            request_http_status=status_code,
            response_size_bytes=response_size,
            request_url=request_url,
            request_error=None,
            first_timestamp_utc=converted_series.index.min().isoformat() if len(converted_series) > 0 else None,
            last_timestamp_utc=converted_series.index.max().isoformat() if len(converted_series) > 0 else None,
            returned_observation_count=raw_count,
            native_timestep_minutes=timestep_minutes,
            native_timestep_share=timestep_share,
            inferred_timestep_mode=timestep_mode,
            hourly_values_count=hourly_valid_count,
            expected_hourly_count=EXPECTED_HOURLY_COUNT,
            hourly_completeness_pct=completeness_pct,
            screening_status=screening_status,
            units_original=meta.get("unit_code") if isinstance(meta.get("unit_code"), str) else None,
            units_output=units_output,
            unit_conversion_applied=bool(conversion_applied),
            rbi=rbi,
            max_hourly_dqdt_m3s_per_hr=max_dqdt,
            normalized_max_hourly_dqdt=normalized_dqdt,
            q95_event_count=q95_count,
            q99_event_count=q99_count,
            notes="; ".join(notes),
        )
    except Exception as exc:  # noqa: BLE001
        return BasinResult(
            **base_kwargs,
            request_success=True,
            request_http_status=status_code,
            response_size_bytes=response_size,
            request_url=request_url,
            request_error=str(exc),
            first_timestamp_utc=None,
            last_timestamp_utc=None,
            returned_observation_count=0,
            native_timestep_minutes=None,
            native_timestep_share=None,
            inferred_timestep_mode="sparse",
            hourly_values_count=0,
            expected_hourly_count=EXPECTED_HOURLY_COUNT,
            hourly_completeness_pct=0.0,
            screening_status="ERROR",
            units_original=meta.get("unit_code") if isinstance(meta.get("unit_code"), str) else None,
            units_output="m3/s",
            unit_conversion_applied=False,
            rbi=None,
            max_hourly_dqdt_m3s_per_hr=None,
            normalized_max_hourly_dqdt=None,
            q95_event_count=None,
            q99_event_count=None,
            notes=f"Parsing/processing failed: {exc}",
        )


def merge_results(existing: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)
    if existing.empty:
        combined = new_df
    elif new_df.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, new_df], ignore_index=True)

    if combined.empty:
        return combined

    combined["STAID"] = combined["STAID"].astype(str).str.zfill(8)
    combined = combined.drop_duplicates(subset=["STAID"], keep="last")
    combined = combined.sort_values(["screening_status", "area_bin", "BFI_bin", "STAID"], na_position="last").reset_index(drop=True)
    return combined


def build_summary(results: pd.DataFrame, attempted_basins: int, total_eligible_basins: int) -> dict[str, object]:
    status_counts = results["screening_status"].value_counts().reindex(STATUS_ORDER, fill_value=0).to_dict()
    timestep_counts = results["inferred_timestep_mode"].value_counts().reindex(TIMESTEP_ORDER, fill_value=0).to_dict()

    ready_df = results[results["screening_status"] == "RBI_READY"]
    rbi_ready_median = float(ready_df["rbi"].median()) if ready_df["rbi"].notna().any() else None

    summary = {
        "screening_window": {
            "start": SCREENING_START.isoformat(),
            "end": SCREENING_END.isoformat(),
            "expected_hourly_count": EXPECTED_HOURLY_COUNT,
        },
        "candidate_universe_total_eligible_screening_wy": int(total_eligible_basins),
        "attempted_basins": int(attempted_basins),
        "results_rows": int(len(results)),
        "status_counts": {k: int(v) for k, v in status_counts.items()},
        "timestep_counts": {k: int(v) for k, v in timestep_counts.items()},
        "median_hourly_completeness_pct": float(results["hourly_completeness_pct"].median()) if not results.empty else 0.0,
        "median_rbi_among_rbi_ready": rbi_ready_median,
        "rbi_non_null_count": int(results["rbi"].notna().sum()) if "rbi" in results.columns else 0,
        "unit_conversion_count": int(results["unit_conversion_applied"].fillna(False).sum()) if "unit_conversion_applied" in results.columns else 0,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    return summary


def write_summary_files(paths: dict[str, Path], summary: dict[str, object]) -> None:
    paths["summary_json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# USGS RBI Screening Summary (WY2024)",
        "",
        "## Scope",
        "",
        f"- Candidate universe (ELIGIBLE_SCREENING_WY): {summary['candidate_universe_total_eligible_screening_wy']}",
        f"- Attempted basins in this run: {summary['attempted_basins']}",
        f"- Result rows currently stored: {summary['results_rows']}",
        "",
        "## Status Counts",
        "",
        "| Status | Count |",
        "| --- | ---: |",
    ]

    for status in STATUS_ORDER:
        lines.append(f"| {status} | {summary['status_counts'].get(status, 0)} |")

    lines.extend(
        [
            "",
            "## Key Metrics",
            "",
            f"- Median hourly completeness: {summary['median_hourly_completeness_pct']:.3f}%",
            f"- Median RBI among RBI_READY: {summary['median_rbi_among_rbi_ready']}",
            "",
            "## Timestep Modes",
            "",
            "| Inferred native timestep | Count |",
            "| --- | ---: |",
        ]
    )

    for timestep in TIMESTEP_ORDER:
        lines.append(f"| {timestep} | {summary['timestep_counts'].get(timestep, 0)} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- RBI is computed on the final hourly series only.",
            "- Internal missing hourly gaps are not bridged when computing dQ/dt or RBI.",
            "- NO_DATA means the request succeeded but no 00060 observations were returned for the window.",
            "- ERROR means request, parsing, or processing failed.",
        ]
    )

    paths["summary_md"].write_text("\n".join(lines), encoding="utf-8")


def generate_plots(paths: dict[str, Path], results: pd.DataFrame, eligible: pd.DataFrame) -> None:
    plots_dir = paths["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    if results.empty:
        return

    status_counts = results["screening_status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(status_counts.index, status_counts.values, color=[STATUS_COLORS[s] for s in status_counts.index], edgecolor="black")
    for bar, value in zip(bars, status_counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(status_counts.values, default=1) * 0.01, str(int(value)), ha="center", va="bottom")
    plt.ylabel("Basin count")
    plt.title("RBI Screening Status Counts")
    plt.tight_layout()
    plt.savefig(plots_dir / "rbi_status_counts.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(results["hourly_completeness_pct"].fillna(0), bins=30, color="#2b8cbe", edgecolor="black", alpha=0.85)
    plt.axvline(70, color="#fe9929", linestyle="--", linewidth=2, label="70%")
    plt.axvline(90, color="#de2d26", linestyle="--", linewidth=2, label="90%")
    plt.xlabel("Hourly completeness (%)")
    plt.ylabel("Basin count")
    plt.title("Hourly Completeness Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "hourly_completeness_distribution.png", dpi=160)
    plt.close()

    timestep_counts = results["inferred_timestep_mode"].value_counts().reindex(TIMESTEP_ORDER, fill_value=0)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(timestep_counts.index, timestep_counts.values, color=[TIMESTEP_COLORS[t] for t in timestep_counts.index], edgecolor="black")
    for bar, value in zip(bars, timestep_counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(timestep_counts.values, default=1) * 0.01, str(int(value)), ha="center", va="bottom")
    plt.ylabel("Basin count")
    plt.title("Inferred Native Timestep Distribution")
    plt.tight_layout()
    plt.savefig(plots_dir / "inferred_timestep_distribution.png", dpi=160)
    plt.close()

    metric_df = results[results["rbi"].notna()].copy()
    if not metric_df.empty:
        plt.figure(figsize=(10, 6))
        plt.hist(metric_df["rbi"], bins=30, color="#31a354", edgecolor="black", alpha=0.85)
        plt.xlabel("RBI")
        plt.ylabel("Basin count")
        plt.title("RBI Distribution")
        plt.tight_layout()
        plt.savefig(plots_dir / "rbi_distribution.png", dpi=160)
        plt.close()

        metric_df["area_bin"] = pd.Categorical(metric_df["area_bin"], categories=AREA_BIN_LABELS, ordered=True)
        plt.figure(figsize=(10, 6))
        metric_df.boxplot(column="rbi", by="area_bin")
        plt.suptitle("")
        plt.title("RBI by Area Bin")
        plt.xlabel("Area bin")
        plt.ylabel("RBI")
        plt.tight_layout()
        plt.savefig(plots_dir / "rbi_by_area_bin.png", dpi=160)
        plt.close()

        metric_df["BFI_bin"] = pd.Categorical(metric_df["BFI_bin"], categories=BFI_BIN_LABELS, ordered=True)
        plt.figure(figsize=(10, 6))
        metric_df.boxplot(column="rbi", by="BFI_bin")
        plt.suptitle("")
        plt.title("RBI by BFI Bin")
        plt.xlabel("BFI bin")
        plt.ylabel("RBI")
        plt.tight_layout()
        plt.savefig(plots_dir / "rbi_by_bfi_bin.png", dpi=160)
        plt.close()

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(metric_df["BFI_AVE"], metric_df["rbi"], c=metric_df["hourly_completeness_pct"], cmap="viridis", s=20, alpha=0.8, edgecolors="none")
        cbar = plt.colorbar(scatter)
        cbar.set_label("Hourly completeness (%)")
        plt.xlabel("BFI_AVE")
        plt.ylabel("RBI")
        plt.title("RBI vs BFI")
        plt.tight_layout()
        plt.savefig(plots_dir / "rbi_vs_bfi.png", dpi=160)
        plt.close()

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(metric_df["DRAIN_SQKM"], metric_df["rbi"], c=metric_df["hourly_completeness_pct"], cmap="viridis", s=20, alpha=0.8, edgecolors="none")
        cbar = plt.colorbar(scatter)
        cbar.set_label("Hourly completeness (%)")
        plt.xscale("log")
        plt.xlabel("Drainage area (km², log scale)")
        plt.ylabel("RBI")
        plt.title("RBI vs Drainage Area")
        plt.tight_layout()
        plt.savefig(plots_dir / "rbi_vs_drainage_area.png", dpi=160)
        plt.close()

    if results["LNG_GAGE"].notna().any() and results["LAT_GAGE"].notna().any():
        lon_min, lon_max = results["LNG_GAGE"].min(), results["LNG_GAGE"].max()
        lat_min, lat_max = results["LAT_GAGE"].min(), results["LAT_GAGE"].max()
        lon_pad = max(0.5, (lon_max - lon_min) * 0.03)
        lat_pad = max(0.5, (lat_max - lat_min) * 0.03)

        plt.figure(figsize=(14, 8))
        for status in STATUS_ORDER:
            subset = results[results["screening_status"] == status]
            plt.scatter(subset["LNG_GAGE"], subset["LAT_GAGE"], c=STATUS_COLORS[status], s=14, alpha=0.75, label=f"{status} ({len(subset)})")
        plt.xlim(lon_min - lon_pad, lon_max + lon_pad)
        plt.ylim(lat_min - lat_pad, lat_max + lat_pad)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Map: RBI Screening Status")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(plots_dir / "map_rbi_ready_status.png", dpi=160)
        plt.close()

        plt.figure(figsize=(14, 8))
        missing = results[results["rbi"].isna()]
        present = results[results["rbi"].notna()]
        plt.scatter(missing["LNG_GAGE"], missing["LAT_GAGE"], c="#d9d9d9", s=10, alpha=0.4, label=f"No RBI ({len(missing)})")
        if not present.empty:
            scatter = plt.scatter(present["LNG_GAGE"], present["LAT_GAGE"], c=present["rbi"], cmap="magma", s=14, alpha=0.85, edgecolors="none")
            cbar = plt.colorbar(scatter)
            cbar.set_label("RBI")
        plt.xlim(lon_min - lon_pad, lon_max + lon_pad)
        plt.ylim(lat_min - lat_pad, lat_max + lat_pad)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Map: RBI Continuous")
        plt.grid(True, alpha=0.25)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(plots_dir / "map_rbi_continuous.png", dpi=160)
        plt.close()

        plt.figure(figsize=(14, 8))
        scatter = plt.scatter(results["LNG_GAGE"], results["LAT_GAGE"], c=results["hourly_completeness_pct"], cmap="viridis", s=14, alpha=0.85, edgecolors="none")
        cbar = plt.colorbar(scatter)
        cbar.set_label("Hourly completeness (%)")
        plt.xlim(lon_min - lon_pad, lon_max + lon_pad)
        plt.ylim(lat_min - lat_pad, lat_max + lat_pad)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Map: Hourly Completeness")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(plots_dir / "map_hourly_completeness.png", dpi=160)
        plt.close()

        eligible_geo = eligible[["STAID", "LNG_GAGE", "LAT_GAGE"]].copy()
        eligible_geo = eligible_geo.dropna(subset=["LNG_GAGE", "LAT_GAGE"]) if {"LNG_GAGE", "LAT_GAGE"}.issubset(eligible_geo.columns) else pd.DataFrame(columns=["STAID", "LNG_GAGE", "LAT_GAGE"])
        ready_staids = set(results.loc[results["screening_status"] == "RBI_READY", "STAID"].astype(str).tolist())
        if not eligible_geo.empty:
            eligible_geo["is_rbi_ready"] = eligible_geo["STAID"].astype(str).isin(ready_staids)
            dropped = eligible_geo[~eligible_geo["is_rbi_ready"]]
            kept = eligible_geo[eligible_geo["is_rbi_ready"]]
            plt.figure(figsize=(14, 8))
            plt.scatter(dropped["LNG_GAGE"], dropped["LAT_GAGE"], c="#cccccc", s=10, alpha=0.4, label=f"Not RBI_READY ({len(dropped)})")
            plt.scatter(kept["LNG_GAGE"], kept["LAT_GAGE"], c="#2ca25f", s=14, alpha=0.85, label=f"RBI_READY ({len(kept)})")
            plt.xlim(lon_min - lon_pad, lon_max + lon_pad)
            plt.ylim(lat_min - lat_pad, lat_max + lat_pad)
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title("Map: Dropout from ELIGIBLE_SCREENING_WY to RBI_READY")
            plt.legend(loc="best")
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(plots_dir / "map_dropout_eligibility_to_rbi_ready.png", dpi=160)
            plt.close()


def write_review_bundle(paths: dict[str, Path], summary: dict[str, object], argv: list[str]) -> None:
    review_bundle_dir = paths["review_bundle_dir"]
    review_plots_dir = paths["review_plots_dir"]

    selected_plot_names = [
        "rbi_status_counts.png",
        "hourly_completeness_distribution.png",
        "inferred_timestep_distribution.png",
        "rbi_distribution.png",
        "map_rbi_ready_status.png",
        "map_hourly_completeness.png",
    ]

    copied = []
    for name in selected_plot_names:
        src = paths["plots_dir"] / name
        if src.exists():
            dst = review_plots_dir / name
            dst.write_bytes(src.read_bytes())
            copied.append(dst)

    review_summary = {
        "status_counts": summary.get("status_counts", {}),
        "attempted_basins": summary.get("attempted_basins", 0),
        "candidate_universe_total_eligible_screening_wy": summary.get("candidate_universe_total_eligible_screening_wy", 0),
        "median_hourly_completeness_pct": summary.get("median_hourly_completeness_pct", None),
        "median_rbi_among_rbi_ready": summary.get("median_rbi_among_rbi_ready", None),
        "selected_plot_count": len(copied),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    (review_bundle_dir / "summary.json").write_text(json.dumps(review_summary, indent=2), encoding="utf-8")
    (review_bundle_dir / "summary.md").write_text(
        "\n".join(
            [
                "# RBI Screening Review Bundle",
                "",
                f"- Attempted basins: {review_summary['attempted_basins']}",
                f"- Candidate universe (ELIGIBLE_SCREENING_WY): {review_summary['candidate_universe_total_eligible_screening_wy']}",
                f"- Status counts: {review_summary['status_counts']}",
                f"- Median hourly completeness: {review_summary['median_hourly_completeness_pct']}",
                f"- Median RBI among RBI_READY: {review_summary['median_rbi_among_rbi_ready']}",
                f"- Selected plot count: {review_summary['selected_plot_count']}",
            ]
        ),
        encoding="utf-8",
    )

    run_command = "python " + " ".join(argv)
    (review_bundle_dir / "run_command.txt").write_text(run_command + "\n", encoding="utf-8")

    commit_text = "unknown"
    try:
        commit_text = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        pass
    (review_bundle_dir / "git_commit.txt").write_text(commit_text + "\n", encoding="utf-8")

    manifest_files = [
        review_bundle_dir / "summary.md",
        review_bundle_dir / "summary.json",
        review_bundle_dir / "run_command.txt",
        review_bundle_dir / "git_commit.txt",
        *copied,
    ]

    manifest = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "files": [
            {
                "path": str(path.relative_to(paths["output_dir"])).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
            }
            for path in manifest_files
            if path.exists()
        ],
    }
    (review_bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    paths = setup_paths(output_dir)
    logger = setup_logging(paths["log_file"])

    logger.info("Starting USGS RBI screening scale workflow")
    logger.info("Screening window: %s to %s", SCREENING_START.isoformat(), SCREENING_END.isoformat())

    eligible = load_eligibility_table()
    total_eligible = len(eligible)
    logger.info("Loaded %s ELIGIBLE_SCREENING_WY basins", total_eligible)

    if args.max_basins is not None:
        eligible = eligible.head(args.max_basins).copy()
        logger.info("Restricted run to first %s basins due to --max-basins", len(eligible))

    existing = load_existing_results(paths)
    if args.resume and not existing.empty:
        done_ids = set(existing["STAID"].astype(str).str.zfill(8).tolist())
        logger.info("Resume enabled: %s basins already completed", len(done_ids))
    else:
        existing = pd.DataFrame()
        done_ids = set()
        logger.info("Starting fresh run (resume disabled or no existing results)")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    rows = [] if existing.empty else existing.to_dict(orient="records")
    to_process = eligible[~eligible["STAID"].astype(str).isin(done_ids)].copy()
    logger.info("Basins remaining in this run: %s", len(to_process))

    newly_processed = 0
    attempted = 0
    start_time = time.time()

    for idx, (_, site_row) in enumerate(to_process.iterrows(), start=1):
        staid = str(site_row["STAID"]).zfill(8)
        attempted += 1
        logger.info("Processing %s (%s/%s remaining-run)", staid, idx, len(to_process))

        result = process_site(
            site_row=site_row,
            session=session,
            sleep_seconds=args.sleep_seconds,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout_seconds,
            logger=logger,
        )

        rows.append(result.__dict__.copy())
        newly_processed += 1

        if newly_processed % args.batch_size == 0 or idx == len(to_process):
            merged = merge_results(existing, rows)
            save_results(paths, merged)
            status_counts = merged["screening_status"].value_counts().to_dict() if not merged.empty else {}
            logger.info("Checkpoint saved after %s new basins; cumulative rows=%s; status=%s", newly_processed, len(merged), status_counts)

    final_results = merge_results(existing, rows)
    if final_results.empty:
        logger.warning("No results rows exist after run; writing empty summary")
    else:
        save_results(paths, final_results)

    summary = build_summary(final_results, attempted_basins=attempted, total_eligible_basins=total_eligible)
    write_summary_files(paths, summary)
    generate_plots(paths, final_results, eligible)
    write_review_bundle(paths, summary, sys.argv)

    elapsed = time.time() - start_time
    logger.info("Run complete in %.1f seconds", elapsed)
    logger.info("Output directory: %s", paths["output_dir"])


if __name__ == "__main__":
    main()

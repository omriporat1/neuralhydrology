#!/usr/bin/env python3
"""
Flash-NH USGS NWIS IV availability audit.

This script audits USGS discharge metadata for all area-filtered Flash-NH basins
without downloading full discharge time series.

Inputs:
- reports/flashnh_basin_screening_v001/area_filtered_basins.parquet

Outputs:
- reports/flashnh_usgs_availability_v001/usgs_availability_summary.md
- reports/flashnh_usgs_availability_v001/usgs_availability_summary.json
- reports/flashnh_usgs_availability_v001/usgs_availability_candidates.parquet
- reports/flashnh_usgs_availability_v001/usgs_availability_candidates.csv
- reports/flashnh_usgs_availability_v001/logs/usgs_availability_audit.log
- reports/flashnh_usgs_availability_v001/plots/*.png

The audit uses the USGS site service with seriesCatalogOutput=true to check
whether parameter 00060 exists, what date range is available, and whether the
series appears hourly-compatible without downloading the full discharge record.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import requests

# Paths
WORKSPACE_ROOT = Path("C:/PhD/Python/neuralhydrology/US_data/data_download/Disk_volume_estimation")
INPUT_PARQUET = WORKSPACE_ROOT / "reports/flashnh_basin_screening_v001/area_filtered_basins.parquet"
OUTPUT_DIR = WORKSPACE_ROOT / "reports/flashnh_usgs_availability_v001"
PLOTS_DIR = OUTPUT_DIR / "plots"
LOG_DIR = OUTPUT_DIR / "logs"
CACHE_DIR = OUTPUT_DIR / "cache"
BATCH_CACHE_DIR = CACHE_DIR / "site_catalog_batches"

# USGS metadata endpoint
USGS_SITE_SERVICE_URL = "https://waterservices.usgs.gov/nwis/site/"

# Time windows
RESEARCH_START = pd.Timestamp("2020-10-14")
RESEARCH_END = pd.Timestamp("2025-12-31")
SCREENING_WATER_YEAR_START = pd.Timestamp("2023-10-01")
SCREENING_WATER_YEAR_END = pd.Timestamp("2024-09-30")

# Request behavior
BATCH_SIZE = 50
REQUEST_DELAY_SECONDS = 0.35
MAX_RETRIES = 5
BACKOFF_INITIAL_SECONDS = 1.5
TIMEOUT_SECONDS = 60
USER_AGENT = "Flash-NH USGS availability audit (metadata-only; contact: local research workflow)"

# Screening bins
AREA_BINS = [1, 10, 100, 1000]
AREA_BIN_LABELS = ["1-10 km²", "10-100 km²", "100-1000 km²"]
BFI_BINS = [-math.inf, 20, 30, 40, 50, math.inf]
BFI_BIN_LABELS = ["<=20", "20-30", "30-40", "40-50", ">50"]

STATUS_ORDER = ["AVAILABLE", "PARTIAL", "NO_DATA", "ERROR"]
STATUS_COLORS = {
    "AVAILABLE": "#2ca25f",
    "PARTIAL": "#fe9929",
    "NO_DATA": "#de2d26",
    "ERROR": "#756bb1",
}

PATH_ORDER = ["hourly direct", "native/sub-hourly + resample", "unavailable", "unknown/error"]
PATH_COLORS = {
    "hourly direct": "#2b8cbe",
    "native/sub-hourly + resample": "#31a354",
    "unavailable": "#bdbdbd",
    "unknown/error": "#756bb1",
}

EXPECTED_FIELDS = [
    "STAID",
    "DRAIN_SQKM",
    "BFI_AVE",
    "area_bin",
    "BFI_bin",
    "LAT_GAGE",
    "LNG_GAGE",
    "STATE",
    "HUC02",
    "usgs_site_valid",
    "has_parameter_00060",
    "available_begin_date",
    "available_end_date",
    "research_overlap_days",
    "research_overlap_pct",
    "screening_overlap_days",
    "screening_overlap_pct",
    "likely_data_resolution",
    "likely_retrieval_path",
    "preliminary_status",
    "notes",
    "site_catalog_count",
    "site_catalog_response_rows",
    "usgs_site_tp_cd",
    "usgs_dec_lat_va",
    "usgs_dec_long_va",
    "usgs_huc_cd",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit USGS NWIS IV availability for area-filtered Flash-NH basins")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Number of sites per USGS metadata request")
    parser.add_argument("--max-sites", type=int, default=None, help="Optional limit for testing")
    parser.add_argument("--refresh-cache", action="store_true", help="Ignore cached USGS batch responses")
    return parser.parse_args()


@dataclass
class AvailabilityResult:
    staid: str
    usgs_site_valid: bool
    has_parameter_00060: bool
    available_begin_date: Optional[str]
    available_end_date: Optional[str]
    research_overlap_days: int
    research_overlap_pct: float
    screening_overlap_days: int
    screening_overlap_pct: float
    likely_data_resolution: str
    likely_retrieval_path: str
    preliminary_status: str
    notes: str
    site_catalog_count: int
    site_catalog_response_rows: int
    usgs_site_tp_cd: Optional[str]
    usgs_dec_lat_va: Optional[float]
    usgs_dec_long_va: Optional[float]
    usgs_huc_cd: Optional[str]



def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "usgs_availability_audit.log"

    logger = logging.getLogger("flashnh_usgs_availability")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Logging to %s", log_path)
    return logger



def load_area_filtered_basins() -> pd.DataFrame:
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Missing input parquet: {INPUT_PARQUET}")

    basins = pd.read_parquet(INPUT_PARQUET)
    basins = basins.copy()
    basins["STAID"] = basins["STAID"].astype(str).str.zfill(8)
    basins["area_bin"] = pd.cut(
        basins["DRAIN_SQKM"],
        bins=AREA_BINS,
        labels=AREA_BIN_LABELS,
        include_lowest=True,
    )
    basins["BFI_bin"] = pd.cut(
        basins["BFI_AVE"],
        bins=BFI_BINS,
        labels=BFI_BIN_LABELS,
        include_lowest=True,
        right=True,
    )
    return basins



def inclusive_overlap_days(start_a: pd.Timestamp, end_a: pd.Timestamp, start_b: pd.Timestamp, end_b: pd.Timestamp) -> int:
    latest_start = max(start_a, start_b)
    earliest_end = min(end_a, end_b)
    if latest_start > earliest_end:
        return 0
    return int((earliest_end - latest_start).days + 1)



def to_iso_date(value: object) -> Optional[str]:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date().isoformat()



def request_with_backoff(session: requests.Session, url: str, params: dict, logger: logging.Logger, cache_path: Path, refresh_cache: bool) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not refresh_cache:
        cached = cache_path.read_text(encoding="utf-8")
        return parse_usgs_rdb(cached)

    backoff = BACKOFF_INITIAL_SECONDS
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, params=params, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            text = response.text
            cache_path.write_text(text, encoding="utf-8")
            return parse_usgs_rdb(text)
        except Exception as exc:  # noqa: BLE001 - explicit retry handling
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            logger.warning("Batch request failed (attempt %s/%s): %s", attempt, MAX_RETRIES, exc)
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError(f"USGS batch request failed after {MAX_RETRIES} attempts") from last_error



def parse_usgs_rdb(text: str) -> pd.DataFrame:
    lines = []
    for line in text.splitlines():
        if not line.startswith("#") and line.strip():
            lines.append(line)
    if not lines:
        return pd.DataFrame()
    return pd.read_csv(StringIO("\n".join(lines)), sep="\t", dtype=str)



def infer_resolution_and_path(site_rows: pd.DataFrame) -> tuple[str, str, str, Optional[str], Optional[str], str]:
    """Return resolution label, retrieval path, status, begin, end, notes."""
    if site_rows.empty:
        return "unknown", "unavailable", "NO_DATA", None, None, "No USGS site catalog rows returned"

    rows_00060 = site_rows[site_rows["parm_cd"] == "00060"].copy()
    if rows_00060.empty:
        return "unknown", "unavailable", "NO_DATA", None, None, "USGS site exists but parameter 00060 was not found"

    rows_00060["begin_date_dt"] = pd.to_datetime(rows_00060["begin_date"], errors="coerce")
    rows_00060["end_date_dt"] = pd.to_datetime(rows_00060["end_date"], errors="coerce")

    # Prefer UV rows for IV discharge availability; fall back to any 00060 rows.
    uv_rows = rows_00060[rows_00060["data_type_cd"] == "uv"].copy()
    candidate_rows = uv_rows if not uv_rows.empty else rows_00060

    candidate_rows = candidate_rows.copy()
    candidate_rows["count_nu_num"] = pd.to_numeric(candidate_rows["count_nu"], errors="coerce")
    candidate_rows = candidate_rows.dropna(subset=["begin_date_dt", "end_date_dt", "count_nu_num"])

    if candidate_rows.empty:
        begin = to_iso_date(rows_00060["begin_date"].min())
        end = to_iso_date(rows_00060["end_date"].max())
        return (
            "unknown",
            "unknown/error",
            "PARTIAL",
            begin,
            end,
            "00060 exists but cadence could not be inferred from metadata",
        )

    def cadence_score(row: pd.Series) -> float:
        span_days = max((row["end_date_dt"] - row["begin_date_dt"]).days + 1, 1)
        return float(row["count_nu_num"] / span_days)

    best_row = candidate_rows.loc[candidate_rows.apply(cadence_score, axis=1).idxmax()]
    samples_per_day = cadence_score(best_row)

    begin = to_iso_date(candidate_rows["begin_date_dt"].min())
    end = to_iso_date(candidate_rows["end_date_dt"].max())

    if samples_per_day >= 20:
        return (
            "hourly or finer",
            "hourly direct",
            "AVAILABLE",
            begin,
            end,
            f"Estimated cadence {samples_per_day:.1f} samples/day from metadata",
        )

    if samples_per_day >= 2:
        return (
            "sub-daily native",
            "native/sub-hourly + resample",
            "PARTIAL",
            begin,
            end,
            f"Estimated cadence {samples_per_day:.1f} samples/day; resampling likely required",
        )

    return (
        "daily or sparse",
        "unavailable",
        "PARTIAL",
        begin,
        end,
        f"Estimated cadence {samples_per_day:.2f} samples/day; too sparse for hourly target",
    )



def audit_site(site_row: pd.Series, site_catalog_rows: pd.DataFrame, logger: logging.Logger) -> AvailabilityResult:
    staid = site_row["STAID"]
    site_no = staid
    site_rows = site_catalog_rows[site_catalog_rows["site_no"] == site_no].copy()
    site_valid = not site_rows.empty

    rows_00060 = site_rows[site_rows["parm_cd"] == "00060"].copy() if site_valid else pd.DataFrame()
    has_00060 = not rows_00060.empty

    begin_date = None
    end_date = None
    research_overlap_days = 0
    screening_overlap_days = 0
    research_overlap_pct = 0.0
    screening_overlap_pct = 0.0
    resolution = "unknown"
    retrieval_path = "unavailable"
    status = "NO_DATA"
    notes = ""
    site_tp_cd = None
    usgs_lat = None
    usgs_lon = None
    usgs_huc = None

    if site_valid:
        first_site_row = site_rows.iloc[0]
        site_tp_cd = first_site_row.get("site_tp_cd") if pd.notna(first_site_row.get("site_tp_cd")) else None
        usgs_lat = pd.to_numeric(first_site_row.get("dec_lat_va"), errors="coerce")
        usgs_lon = pd.to_numeric(first_site_row.get("dec_long_va"), errors="coerce")
        usgs_huc = first_site_row.get("huc_cd") if pd.notna(first_site_row.get("huc_cd")) else None

    if site_valid and has_00060:
        resolution, retrieval_path, provisional_status, begin_date, end_date, notes = infer_resolution_and_path(site_rows)
        rows_for_dates = rows_00060.copy()
        rows_for_dates["begin_date_dt"] = pd.to_datetime(rows_for_dates["begin_date"], errors="coerce")
        rows_for_dates["end_date_dt"] = pd.to_datetime(rows_for_dates["end_date"], errors="coerce")
        rows_for_dates = rows_for_dates.dropna(subset=["begin_date_dt", "end_date_dt"])

        if not rows_for_dates.empty:
            begin_dt = rows_for_dates["begin_date_dt"].min()
            end_dt = rows_for_dates["end_date_dt"].max()
            research_overlap_days = inclusive_overlap_days(begin_dt, end_dt, RESEARCH_START, RESEARCH_END)
            screening_overlap_days = inclusive_overlap_days(begin_dt, end_dt, SCREENING_WATER_YEAR_START, SCREENING_WATER_YEAR_END)
            research_total_days = int((RESEARCH_END - RESEARCH_START).days + 1)
            screening_total_days = int((SCREENING_WATER_YEAR_END - SCREENING_WATER_YEAR_START).days + 1)
            research_overlap_pct = 100.0 * research_overlap_days / research_total_days
            screening_overlap_pct = 100.0 * screening_overlap_days / screening_total_days
        else:
            provisional_status = "PARTIAL"
            notes = "00060 exists but begin/end dates were missing or unusable"

        if provisional_status == "AVAILABLE":
            if screening_overlap_days < int((SCREENING_WATER_YEAR_END - SCREENING_WATER_YEAR_START).days + 1):
                status = "PARTIAL"
                notes = notes + "; screening water year not fully covered" if notes else "screening water year not fully covered"
            else:
                status = "AVAILABLE"
        else:
            status = provisional_status

        if status == "AVAILABLE" and research_overlap_days <= 0:
            status = "PARTIAL"
            notes = notes + "; no overlap with research period" if notes else "no overlap with research period"
    elif site_valid and not has_00060:
        resolution = "unknown"
        retrieval_path = "unavailable"
        status = "NO_DATA"
        notes = "USGS site catalog exists but parameter 00060 is absent"
    elif not site_valid:
        resolution = "unknown"
        retrieval_path = "unavailable"
        status = "NO_DATA"
        notes = "No USGS site catalog rows returned"

    return AvailabilityResult(
        staid=staid,
        usgs_site_valid=site_valid,
        has_parameter_00060=has_00060,
        available_begin_date=begin_date,
        available_end_date=end_date,
        research_overlap_days=research_overlap_days,
        research_overlap_pct=round(research_overlap_pct, 3),
        screening_overlap_days=screening_overlap_days,
        screening_overlap_pct=round(screening_overlap_pct, 3),
        likely_data_resolution=resolution,
        likely_retrieval_path=retrieval_path,
        preliminary_status=status,
        notes=notes,
        site_catalog_count=int(site_rows.shape[0]),
        site_catalog_response_rows=int(site_rows.shape[0]),
        usgs_site_tp_cd=site_tp_cd,
        usgs_dec_lat_va=float(usgs_lat) if pd.notna(usgs_lat) else None,
        usgs_dec_long_va=float(usgs_lon) if pd.notna(usgs_lon) else None,
        usgs_huc_cd=usgs_huc,
    )



def load_existing_results() -> pd.DataFrame:
    parquet_path = OUTPUT_DIR / "usgs_availability_candidates.parquet"
    csv_path = OUTPUT_DIR / "usgs_availability_candidates.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, dtype={"STAID": str})
    return pd.DataFrame(columns=EXPECTED_FIELDS)



def write_results(results: pd.DataFrame, logger: Optional[logging.Logger] = None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results = results.copy()
    if "STAID" in results.columns:
        results["STAID"] = results["STAID"].astype(str).str.zfill(8)
    results.to_parquet(OUTPUT_DIR / "usgs_availability_candidates.parquet", index=False)
    results.to_csv(OUTPUT_DIR / "usgs_availability_candidates.csv", index=False)
    if logger is not None:
        logger.info("Checkpointed %s audited sites", len(results))



def batch_site_ids(site_ids: List[str], batch_size: int) -> List[List[str]]:
    return [site_ids[i : i + batch_size] for i in range(0, len(site_ids), batch_size)]



def fetch_batch(site_ids: List[str], session: requests.Session, logger: logging.Logger, batch_index: int, refresh_cache: bool) -> pd.DataFrame:
    cache_path = BATCH_CACHE_DIR / f"site_catalog_batch_{batch_index:05d}.rdb"
    params = {
        "sites": ",".join(site_ids),
        "seriesCatalogOutput": "true",
        "format": "rdb",
    }
    logger.info("Fetching batch %s with %s sites", batch_index, len(site_ids))
    frame = request_with_backoff(session, USGS_SITE_SERVICE_URL, params, logger, cache_path, refresh_cache)
    time.sleep(REQUEST_DELAY_SECONDS)
    return frame



def build_status_maps(results: pd.DataFrame) -> dict[str, str]:
    return {
        "AVAILABLE": "#2ca25f",
        "PARTIAL": "#fe9929",
        "NO_DATA": "#de2d26",
        "ERROR": "#756bb1",
    }



def generate_plots(results: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = results.copy()
    plot_df["available_flag"] = plot_df["preliminary_status"].eq("AVAILABLE")

    # 1. status counts
    status_counts = plot_df["preliminary_status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(status_counts.index, status_counts.values, color=[STATUS_COLORS[s] for s in status_counts.index], edgecolor="black")
    for bar, count in zip(bars, status_counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(status_counts.values) * 0.01, f"{int(count)}", ha="center", va="bottom")
    plt.ylabel("Basin count")
    plt.title("USGS Availability Status Counts")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "availability_status_counts.png", dpi=160)
    plt.close()

    # 2. availability by area bin
    area_table = pd.crosstab(plot_df["area_bin"], plot_df["preliminary_status"]).reindex(AREA_BIN_LABELS)
    area_table = area_table.fillna(0)
    plt.figure(figsize=(11, 6))
    bottom = pd.Series([0] * len(area_table), index=area_table.index)
    for status in STATUS_ORDER:
        values = area_table.get(status, pd.Series([0] * len(area_table), index=area_table.index))
        plt.bar(area_table.index, values.values, bottom=bottom.values, label=status, color=STATUS_COLORS[status], edgecolor="black")
        bottom = bottom + values
    plt.ylabel("Basin count")
    plt.title("USGS Availability by Drainage Area Bin")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "availability_by_area_bin.png", dpi=160)
    plt.close()

    # 3. availability by BFI bin
    bfi_table = pd.crosstab(plot_df["BFI_bin"], plot_df["preliminary_status"]).reindex(BFI_BIN_LABELS)
    bfi_table = bfi_table.fillna(0)
    plt.figure(figsize=(11, 6))
    bottom = pd.Series([0] * len(bfi_table), index=bfi_table.index)
    for status in STATUS_ORDER:
        values = bfi_table.get(status, pd.Series([0] * len(bfi_table), index=bfi_table.index))
        plt.bar(bfi_table.index, values.values, bottom=bottom.values, label=status, color=STATUS_COLORS[status], edgecolor="black")
        bottom = bottom + values
    plt.ylabel("Basin count")
    plt.title("USGS Availability by BFI Bin")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "availability_by_bfi_bin.png", dpi=160)
    plt.close()

    if not plot_df["LNG_GAGE"].notna().any() or not plot_df["LAT_GAGE"].notna().any():
        return

    lon_min, lon_max = plot_df["LNG_GAGE"].min(), plot_df["LNG_GAGE"].max()
    lat_min, lat_max = plot_df["LAT_GAGE"].min(), plot_df["LAT_GAGE"].max()
    lon_pad = max(0.5, (lon_max - lon_min) * 0.03)
    lat_pad = max(0.5, (lat_max - lat_min) * 0.03)
    x_limits = (lon_min - lon_pad, lon_max + lon_pad)
    y_limits = (lat_min - lat_pad, lat_max + lat_pad)

    # 4. map available vs unavailable
    available = plot_df[plot_df["preliminary_status"] == "AVAILABLE"]
    unavailable = plot_df[plot_df["preliminary_status"] != "AVAILABLE"]
    plt.figure(figsize=(14, 8))
    plt.scatter(unavailable["LNG_GAGE"], unavailable["LAT_GAGE"], c="#d9d9d9", s=16, alpha=0.5, label=f"Unavailable / partial ({len(unavailable)})")
    plt.scatter(available["LNG_GAGE"], available["LAT_GAGE"], c="#2ca25f", s=22, alpha=0.9, label=f"AVAILABLE ({len(available)})")
    plt.xlim(x_limits)
    plt.ylim(y_limits)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("USGS Availability: Available vs Unavailable")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "map_available_vs_unavailable.png", dpi=160)
    plt.close()

    # 5. map by status
    plt.figure(figsize=(14, 8))
    for status in STATUS_ORDER:
        subset = plot_df[plot_df["preliminary_status"] == status]
        plt.scatter(subset["LNG_GAGE"], subset["LAT_GAGE"], c=STATUS_COLORS[status], s=20, alpha=0.85, label=f"{status} ({len(subset)})")
    plt.xlim(x_limits)
    plt.ylim(y_limits)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("USGS Availability by Preliminary Status")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "map_availability_status.png", dpi=160)
    plt.close()

    # 6. map dropout stages
    stages = [
        (plot_df, "All area-filtered basins"),
        (plot_df[plot_df["usgs_site_valid"]], "Valid USGS site metadata"),
        (plot_df[plot_df["has_parameter_00060"]], "Parameter 00060 present"),
        (plot_df[plot_df["preliminary_status"] == "AVAILABLE"], "AVAILABLE basins"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (subset, title) in zip(axes, stages):
        ax.scatter(plot_df["LNG_GAGE"], plot_df["LAT_GAGE"], c="#e0e0e0", s=12, alpha=0.35, linewidths=0)
        ax.scatter(subset["LNG_GAGE"], subset["LAT_GAGE"], c="#2b8cbe", s=18, alpha=0.8, linewidths=0)
        ax.set_title(f"{title} | n={len(subset)}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Flash-NH Dropout Stages: Static Screening to USGS Availability", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(PLOTS_DIR / "map_dropout_stages_static_to_usgs.png", dpi=160)
    plt.close(fig)



def write_outputs(results: pd.DataFrame, logger: logging.Logger) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ordered = results.copy()
    for field in EXPECTED_FIELDS:
        if field not in ordered.columns:
            ordered[field] = None
    ordered = ordered[EXPECTED_FIELDS]
    ordered.to_parquet(OUTPUT_DIR / "usgs_availability_candidates.parquet", index=False)
    ordered.to_csv(OUTPUT_DIR / "usgs_availability_candidates.csv", index=False)

    summary = {
        "total_area_filtered_basins": int(len(ordered)),
        "research_period": {
            "start": RESEARCH_START.date().isoformat(),
            "end": RESEARCH_END.date().isoformat(),
        },
        "screening_water_year": {
            "start": SCREENING_WATER_YEAR_START.date().isoformat(),
            "end": SCREENING_WATER_YEAR_END.date().isoformat(),
        },
        "status_counts": ordered["preliminary_status"].value_counts().reindex(STATUS_ORDER, fill_value=0).to_dict(),
        "retrieval_path_counts": ordered["likely_retrieval_path"].value_counts().reindex(PATH_ORDER, fill_value=0).to_dict(),
        "usgs_site_valid_count": int(ordered["usgs_site_valid"].sum()),
        "has_00060_count": int(ordered["has_parameter_00060"].sum()),
        "available_count": int((ordered["preliminary_status"] == "AVAILABLE").sum()),
        "partial_count": int((ordered["preliminary_status"] == "PARTIAL").sum()),
        "no_data_count": int((ordered["preliminary_status"] == "NO_DATA").sum()),
        "error_count": int((ordered["preliminary_status"] == "ERROR").sum()),
        "area_bin_counts": ordered["area_bin"].value_counts().reindex(AREA_BIN_LABELS, fill_value=0).to_dict(),
        "bfi_bin_counts": ordered["BFI_bin"].value_counts().reindex(BFI_BIN_LABELS, fill_value=0).to_dict(),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    with open(OUTPUT_DIR / "usgs_availability_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    summary_lines = [
        "# Flash-NH USGS Availability Audit Summary",
        "",
        "## Purpose",
        "",
        "Audit USGS NWIS IV discharge metadata for all area-filtered basins without downloading full discharge time series.",
        "",
        "## Candidate Universe",
        "",
        f"- Total area-filtered basins: {len(ordered)}",
        f"- Research period: {RESEARCH_START.date().isoformat()} to {RESEARCH_END.date().isoformat()}",
        f"- Screening water year: {SCREENING_WATER_YEAR_START.date().isoformat()} to {SCREENING_WATER_YEAR_END.date().isoformat()}",
        "",
        "## Status Counts",
        "",
        "| Status | Count |",
        "| --- | ---: |",
    ]
    for status in STATUS_ORDER:
        summary_lines.append(f"| {status} | {summary['status_counts'].get(status, 0)} |")
    summary_lines.extend([
        "",
        "## Retrieval Path Counts",
        "",
        "| Path | Count |",
        "| --- | ---: |",
    ])
    for path in PATH_ORDER:
        summary_lines.append(f"| {path} | {summary['retrieval_path_counts'].get(path, 0)} |")
    summary_lines.extend([
        "",
        "## Key Notes",
        "",
        "- The audit uses metadata only; no discharge time series were downloaded.",
        "- Availability may be geographically biased, so later stages should map all basins, area-filtered basins, USGS-available basins, RBI-computed basins, and final pilot basins.",
        "- RBI screening should be based on at least one full water year; the default screening window is 2023-10-01 to 2024-09-30.",
        "- One month is too seasonal for RBI-based screening.",
    ])
    with open(OUTPUT_DIR / "usgs_availability_summary.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines))

    logger.info("Wrote outputs to %s", OUTPUT_DIR)



def main() -> None:
    args = parse_args()
    logger = setup_logging()
    logger.info("Starting USGS availability audit")
    logger.info("Input parquet: %s", INPUT_PARQUET)
    logger.info("Output directory: %s", OUTPUT_DIR)

    basins = load_area_filtered_basins()
    if args.max_sites is not None:
        basins = basins.head(args.max_sites).copy()
        logger.info("Limited audit to %s sites for testing", len(basins))

    existing = load_existing_results()
    if not existing.empty:
        existing["STAID"] = existing["STAID"].astype(str).str.zfill(8)
        done_sites = set(existing["STAID"].astype(str))
    else:
        done_sites = set()

    logger.info("Loaded %s basins", len(basins))
    logger.info("Resuming with %s already-audited sites", len(done_sites))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    results_rows = []
    if not existing.empty:
        results_rows.extend(existing.to_dict(orient="records"))

    site_batches = batch_site_ids(basins["STAID"].tolist(), args.batch_size)
    logger.info("Processing %s batches at batch size %s", len(site_batches), args.batch_size)

    for batch_index, site_batch in enumerate(site_batches, start=1):
        batch_existing = [site for site in site_batch if site in done_sites]
        if len(batch_existing) == len(site_batch):
            logger.info("Skipping batch %s because all sites are already cached", batch_index)
            continue

        batch_frame = fetch_batch(site_batch, session, logger, batch_index, args.refresh_cache)
        for _, basin_row in basins[basins["STAID"].isin(site_batch)].iterrows():
            result = audit_site(basin_row, batch_frame, logger)
            result_dict = {
                "STAID": result.staid,
                "DRAIN_SQKM": float(basin_row["DRAIN_SQKM"]),
                "BFI_AVE": float(basin_row["BFI_AVE"]),
                "area_bin": basin_row["area_bin"],
                "BFI_bin": basin_row["BFI_bin"],
                "LAT_GAGE": float(basin_row["LAT_GAGE"]) if pd.notna(basin_row["LAT_GAGE"]) else None,
                "LNG_GAGE": float(basin_row["LNG_GAGE"]) if pd.notna(basin_row["LNG_GAGE"]) else None,
                "STATE": basin_row.get("STATE"),
                "HUC02": basin_row.get("HUC02"),
                "usgs_site_valid": result.usgs_site_valid,
                "has_parameter_00060": result.has_parameter_00060,
                "available_begin_date": result.available_begin_date,
                "available_end_date": result.available_end_date,
                "research_overlap_days": result.research_overlap_days,
                "research_overlap_pct": result.research_overlap_pct,
                "screening_overlap_days": result.screening_overlap_days,
                "screening_overlap_pct": result.screening_overlap_pct,
                "likely_data_resolution": result.likely_data_resolution,
                "likely_retrieval_path": result.likely_retrieval_path,
                "preliminary_status": result.preliminary_status,
                "notes": result.notes,
                "site_catalog_count": result.site_catalog_count,
                "site_catalog_response_rows": result.site_catalog_response_rows,
                "usgs_site_tp_cd": result.usgs_site_tp_cd,
                "usgs_dec_lat_va": result.usgs_dec_lat_va,
                "usgs_dec_long_va": result.usgs_dec_long_va,
                "usgs_huc_cd": result.usgs_huc_cd,
            }
            results_rows.append(result_dict)
            done_sites.add(result.staid)

        if batch_index % 2 == 0 or len(done_sites) == len(basins):
            partial = pd.DataFrame(results_rows)
            if not partial.empty:
                partial["STAID"] = partial["STAID"].astype(str).str.zfill(8)
                write_results(partial, logger)
            logger.info("Progress: %s/%s sites audited", len(done_sites), len(basins))

    final_results = pd.DataFrame(results_rows)
    if final_results.empty:
        raise RuntimeError("No USGS availability results were produced")

    final_results["STAID"] = final_results["STAID"].astype(str).str.zfill(8)
    final_results = final_results.drop_duplicates(subset=["STAID"], keep="last")
    final_results = final_results.merge(
        basins[["STAID", "DRAIN_SQKM", "BFI_AVE", "area_bin", "BFI_bin", "LAT_GAGE", "LNG_GAGE", "STATE", "HUC02"]],
        on="STAID",
        how="right",
        suffixes=("", "_basin"),
    )
    for column in ["DRAIN_SQKM", "BFI_AVE", "area_bin", "BFI_bin", "LAT_GAGE", "LNG_GAGE", "STATE", "HUC02"]:
        if f"{column}_basin" in final_results.columns:
            final_results[column] = final_results[column].fillna(final_results[f"{column}_basin"])
            final_results.drop(columns=[f"{column}_basin"], inplace=True)
    final_results = final_results.sort_values(["preliminary_status", "STAID"])

    # Keep the user-facing output stable and predictable.
    final_results = final_results.reindex(columns=EXPECTED_FIELDS + [c for c in final_results.columns if c not in EXPECTED_FIELDS])

    write_results(final_results, logger)
    generate_plots(final_results)
    write_outputs(final_results, logger)

    logger.info("USGS availability audit complete")
    logger.info("Results: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
